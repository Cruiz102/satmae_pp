# --------------------------------------------------------
# Single-GPU Swin fine-tuning for NASA Geographical Objects (Kaggle)
# - Multi-label classification (BCEWithLogitsLoss)
# - Uses Hugging Face Transformers (SwinForImageClassification)
# - CSV columns: FileName, Label Vector (e.g., "[0,1,0,1,...]")
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
import ast
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import cv2
from sklearn.metrics import f1_score

from transformers import (
    AutoImageProcessor,
    SwinForImageClassification,
    get_cosine_schedule_with_warmup,
)

# ---------------------------
# Dataset
# ---------------------------
class KaggleGeographicalDataset(Dataset):
    """Kaggle NASA Geographical Objects dataset:
       - Reads `FileName` and `Label Vector`
       - Returns (pixel_values, label_tensor)
    """
    def __init__(self, csv_file, images_dir, processor_name, image_size=224, augment=False):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.augment = augment
        self.image_size = image_size

        # HF image processor handles resize/normalize
        self.processor = AutoImageProcessor.from_pretrained(
            processor_name,
            do_resize=True,
            size={"height": image_size, "width": image_size},
            do_center_crop=False,
        )

        # Count channels (RGB)
        self.in_c = 3

        # Filter out missing images during initialization
        print(f"Checking for missing images in {images_dir}...")
        valid_indices = []
        missing_count = 0
        for idx, row in self.data.iterrows():
            img_name = row['FileName']
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                missing_count += 1
                if missing_count <= 5:
                    print(f"Missing image: {img_path}")
        if missing_count > 5:
            print(f"... and {missing_count - 5} more missing images")
        print(f"Found {len(valid_indices)} valid images out of {len(self.data)} total ({missing_count} missing)")

        self.data = self.data.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def _maybe_augment(self, img_rgb):
        if not self.augment:
            return img_rgb
        # Minimal, geometry-safe augs (horizontal flip). Keep it simple for multi-label.
        if np.random.rand() < 0.5:
            img_rgb = np.ascontiguousarray(img_rgb[:, ::-1, :])
        return img_rgb

    def __getitem__(self, idx):
        # image
        img_name = self.data.iloc[idx]['FileName']
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)  # BGR
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # <-- FIXED

        image = self._maybe_augment(image)

        # labels
        label_str = self.data.iloc[idx]['Label Vector']
        label_vector = ast.literal_eval(label_str)  # string -> list
        label = torch.tensor(label_vector, dtype=torch.float32)

        # HF processor -> pixel_values [3,H,W], normalized
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return pixel_values, label


# ---------------------------
# Train / Eval
# ---------------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_pred = []
    all_true = []

    for pixel_values, labels in dataloader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(pixel_values=pixel_values).logits
        loss = criterion(outputs, labels)
        total_loss += loss.item() * pixel_values.size(0)

        preds = (outputs.sigmoid() > 0.5).int().cpu().numpy()
        all_pred.append(preds)
        all_true.append(labels.int().cpu().numpy())

    all_pred = np.vstack(all_pred)
    all_true = np.vstack(all_true)

    f1_micro = f1_score(all_true, all_pred, average='micro', zero_division=0)
    f1_macro = f1_score(all_true, all_pred, average='macro', zero_division=0)
    avg_loss = total_loss / max(len(dataloader.dataset), 1)

    return {"loss": avg_loss, "f1_micro": f1_micro, "f1_macro": f1_macro}


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    seen = 0
    running = 0.0

    for pixel_values, labels in dataloader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            logits = model(pixel_values=pixel_values).logits
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running += loss.item() * pixel_values.size(0)
        seen += pixel_values.size(0)

    return running / max(seen, 1)


# ---------------------------
# Argparse / Main
# ---------------------------
def get_args():
    p = argparse.ArgumentParser("Swin fine-tuning for Kaggle NASA Geographical Objects")
    # Data
    p.add_argument("--data_path", type=str, default="/home/cesar/kickoff_material/kickoff_pack")
    p.add_argument("--train_csv", type=str, default="data/train.csv")
    p.add_argument("--val_csv", type=str, default="data/val.csv")
    p.add_argument("--images_dir", type=str, default="images")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--kaggle_path", default=None, type=str,
                    help="Path to Kaggle dataset if different from data_path")

    # Model
    p.add_argument("--model_id", type=str, default="microsoft/swin-base-patch4-window7-224")
    p.add_argument("--nb_classes", type=int, default=10)

    # Train
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    # I/O
    p.add_argument("--output_dir", type=str, default="./finetune_logs_kaggle")
    p.add_argument("--log_dir", type=str, default="./finetune_logs_kaggle")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()

def find_images_path(args):
    """Find the correct path for images"""
    possible_paths = [
        os.path.join(args.data_path, args.images_dir),
    ]

    # If kaggle_path is provided, try that too
    if getattr(args, "kaggle_path", None):
        possible_paths.insert(0, os.path.join(args.kaggle_path, "images"))

    # Also try common Kaggle cache locations
    home_dir = os.path.expanduser("~")
    kaggle_cache = os.path.join(home_dir, ".cache", "kagglehub")
    if os.path.exists(kaggle_cache):
        for root, dirs, files in os.walk(kaggle_cache):
            if "images" in dirs:
                possible_paths.append(os.path.join(root, "images"))

    for path in possible_paths:
        if os.path.exists(path):
            image_files = [
                f for f in os.listdir(path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if len(image_files) > 0:
                print(f"Found {len(image_files)} images in: {path}")
                return path

    print(f"Warning: No images found in any of these paths: {possible_paths}")
    return os.path.join(args.data_path, args.images_dir)  # fallback

def main():
    args = get_args()
    print("Args:", args)

    # Repro & device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve paths
    train_csv = os.path.join(args.data_path, args.train_csv)
    val_csv   = os.path.join(args.data_path, args.val_csv)
    images_path = find_images_path(args)  # <— uses your helper
    print(f"[Resolved] images_path = {images_path}")

    # Datasets
    print("Loading datasets...")
    train_ds = KaggleGeographicalDataset(
        csv_file=train_csv,
        images_dir=images_path,
        processor_name=args.model_id,
        image_size=args.image_size,
        augment=True,
    )
    val_ds = KaggleGeographicalDataset(
        csv_file=val_csv,
        images_dir=images_path,
        processor_name=args.model_id,
        image_size=args.image_size,
        augment=False,
    )

    # Infer/validate class count from CSV label vector
    first_vec_len = len(ast.literal_eval(pd.read_csv(train_csv).iloc[0]["Label Vector"]))
    if args.nb_classes != first_vec_len:
        print(f"[Info] Overriding nb_classes: {args.nb_classes} -> {first_vec_len}")
        args.nb_classes = first_vec_len

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)} | Classes: {args.nb_classes}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("[ERROR] No images found after resolution. "
              "Check --data_path/--images_dir/--kaggle_path and filenames in CSV.")
        return

    # Dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model (multi-label head)
    model = SwinForImageClassification.from_pretrained(
        args.model_id,
        num_labels=args.nb_classes,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    # Optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    total_steps = max(1, len(train_loader)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # I/O
    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = SummaryWriter(args.log_dir) if (args.log_dir and not args.eval_only) else None

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Eval only
    if args.eval_only:
        stats = evaluate(model, val_loader, device)
        print(f"[Eval] loss={stats['loss']:.4f} | f1_micro={stats['f1_micro']:.4f} | f1_macro={stats['f1_macro']:.4f}")
        return

    # Train
    best_f1 = 0.0
    t0 = time.time()
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_stats = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1:03d}/{args.epochs:03d} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_stats['loss']:.4f} | "
              f"f1_micro={val_stats['f1_micro']:.4f} | "
              f"f1_macro={val_stats['f1_macro']:.4f}")

        if log_writer:
            log_writer.add_scalar("train/loss", train_loss, epoch)
            log_writer.add_scalar("val/loss", val_stats["loss"], epoch)
            log_writer.add_scalar("val/f1_micro", val_stats["f1_micro"], epoch)
            log_writer.add_scalar("val/f1_macro", val_stats["f1_macro"], epoch)
            log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Save periodic
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Track best
        if val_stats["f1_micro"] > best_f1:
            best_f1 = val_stats["f1_micro"]
            best_path = os.path.join(args.output_dir, f"best_f1_{best_f1:.4f}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, best_path)
            print(f"New best F1_micro={best_f1:.4f} -> {best_path}")

    total = str(datetime.timedelta(seconds=int(time.time() - t0)))
    print(f"\nDone. Best F1_micro={best_f1:.4f}. Total time: {total}")
    if log_writer:
        log_writer.close()


if __name__ == "__main__":
    main()
