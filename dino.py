# --------------------------------------------------------
# Single-GPU DINOv2 fine-tuning for NASA Geographical Objects (Kaggle)
# - Multi-label classification (BCEWithLogitsLoss)
# - CSV columns: FileName, Label Vector (e.g., "[0,1,0,1,...]")
# - Auto-discovers images folder; tolerant to case/extension mismatches
# --------------------------------------------------------
import argparse
import ast
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,  # works with DINOv2 (Dinov2ForImageClassification)
    get_cosine_schedule_with_warmup,
)

# ---------------------------
# Dataset (robust resolver)
# ---------------------------
class KaggleGeographicalDataset(Dataset):
    """
    Reads `FileName` and `Label Vector`.
    Builds a case-insensitive map of files and tolerates extension differences.
    Returns (pixel_values, label_tensor).
    """
    def __init__(self, csv_file, images_dir, processor_name, image_size=224, augment=False):
        self.df = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.augment = augment
        self.image_size = image_size

        self.processor = AutoImageProcessor.from_pretrained(
            processor_name,
            do_resize=True,
            size={"height": image_size, "width": image_size},
            do_center_crop=False,
        )

        # Build lookup tables
        self._path_map = {}   # key: lowercased filename (with ext) -> full path
        self._stem_map = {}   # key: lowercased stem -> [full paths]
        if os.path.isdir(images_dir):
            for root, _, files in os.walk(images_dir):
                for f in files:
                    full = os.path.join(root, f)
                    key = f.lower()
                    self._path_map[key] = full
                    stem = os.path.splitext(f)[0].lower()
                    self._stem_map.setdefault(stem, []).append(full)
        else:
            print(f"[WARN] images_dir does not exist: {images_dir}")

        def resolve_path(requested_name: str):
            key = requested_name.lower()
            if key in self._path_map:
                return self._path_map[key]

            stem = os.path.splitext(requested_name)[0].lower()
            if stem in self._stem_map:
                return self._stem_map[stem][0]

            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                k2 = (requested_name + ext).lower()
                if k2 in self._path_map:
                    return self._path_map[k2]
                k3 = (stem + ext).lower()
                if k3 in self._path_map:
                    return self._path_map[k3]
            return None

        print(f"Indexing images under: {images_dir}")
        valid_rows = []
        missing = 0
        to_show = 5

        for _, row in self.df.iterrows():
            img_name = row["FileName"]
            resolved = resolve_path(img_name)
            if resolved and os.path.exists(resolved):
                r = dict(row)
                r["_resolved_path"] = resolved
                valid_rows.append(r)
            else:
                if missing < to_show:
                    print(f"Missing image (after resolving): {os.path.join(images_dir, img_name)}")
                missing += 1

        if missing > to_show:
            print(f"... and {missing - to_show} more missing images")

        self.df = pd.DataFrame(valid_rows)
        print(f"Found {len(self.df)} valid images out of {len(valid_rows) + missing} total")

    def __len__(self):
        return len(self.df)

    def _maybe_augment(self, img_rgb):
        if not self.augment:
            return img_rgb
        # light augments that are safe for multi-label
        if np.random.rand() < 0.5:
            img_rgb = np.ascontiguousarray(img_rgb[:, ::-1, :])
        return img_rgb

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["_resolved_path"]
        image = cv2.imread(img_path)  # BGR
        if image is None:
            raise FileNotFoundError(f"Image not found at runtime: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self._maybe_augment(image)

        label_vec = ast.literal_eval(self.df.iloc[idx]["Label Vector"])
        label = torch.tensor(label_vec, dtype=torch.float32)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, label

@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    model.eval()

    total_loss = 0.0
    all_pred, all_true = [], []

    for pixel_values, labels in dataloader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(pixel_values=pixel_values).logits
        loss = criterion(logits, labels)
        total_loss += loss.item() * pixel_values.size(0)

        preds = (logits.sigmoid() > 0.5).int().cpu().numpy()
        all_pred.append(preds)
        all_true.append(labels.int().cpu().numpy())

    all_pred = np.vstack(all_pred) if all_pred else np.zeros((0, 0))
    all_true = np.vstack(all_true) if all_true else np.zeros((0, 0))

    f1_micro = f1_score(all_true, all_pred, average="micro", zero_division=0) if len(all_true) else 0.0
    f1_macro = f1_score(all_true, all_pred, average="macro", zero_division=0) if len(all_true) else 0.0
    avg_loss = total_loss / max(len(dataloader.dataset), 1)

    return {"loss": avg_loss, "f1_micro": f1_micro, "f1_macro": f1_macro}

def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler, criterion):
    model.train()
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

# --- NEW: compute per-class pos_weight from TRAIN CSV ---
def compute_pos_weight_from_csv(train_csv_path: str, device: torch.device, eps: float = 1e-6):
    import ast, numpy as np, pandas as pd, torch
    df = pd.read_csv(train_csv_path)
    # Stack label vectors into an (N, C) matrix
    Y = np.stack([np.array(ast.literal_eval(s), dtype=np.float32) for s in df["Label Vector"]])  # (N, C)
    pos = Y.sum(axis=0)                     # positives per class
    neg = Y.shape[0] - pos                  # negatives per class
    pw = neg / (pos + eps)                  # ratio -> larger weight for rare classes
    # (Optional) clamp huge values if you have classes with 0 positives
    pw = np.clip(pw, 1.0, 100.0)
    return torch.tensor(pw, dtype=torch.float32, device=device)

# ---------------------------
# Args / Main
# ---------------------------
def get_args():
    p = argparse.ArgumentParser("DINOv2 fine-tuning for Kaggle NASA Geographical Objects")
    # Data
    p.add_argument("--data_path", type=str, default="./")
    p.add_argument("--train_csv", type=str, default="data/train.csv")
    p.add_argument("--val_csv", type=str, default="data/val.csv")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--augment", action="store_true")

    # Model
    p.add_argument("--model_id", type=str, default="facebook/dinov2-base")  # or dinov2-large
    p.add_argument("--nb_classes", type=int, default=None)  # inferred if None

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
    p.add_argument("--output_dir", type=str, default="./finetune_logs_dinov2")
    p.add_argument("--log_dir", type=str, default="./finetune_logs_dinov2")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    print("Args:", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_csv = os.path.join(args.data_path, args.train_csv)
    val_csv = os.path.join(args.data_path, args.val_csv)
    images_path = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "images")
    print(f"[Resolved] images_path = {images_path}")

    # Datasets
    print("Loading datasets...")
    train_ds = KaggleGeographicalDataset(
        train_csv, images_path, processor_name=args.model_id,
        image_size=args.image_size, augment=True
    )
    val_ds = KaggleGeographicalDataset(
        val_csv, images_path, processor_name=args.model_id,
        image_size=args.image_size, augment=False
    )

    # Infer classes from label vector length if not provided
    if args.nb_classes is None:
        first_vec_len = len(ast.literal_eval(pd.read_csv(train_csv).iloc[0]["Label Vector"]))
        args.nb_classes = first_vec_len
    else:
        # sanity check
        vec_len = len(ast.literal_eval(pd.read_csv(train_csv).iloc[0]["Label Vector"]))
        if args.nb_classes != vec_len:
            print(f"[Info] Overriding nb_classes: {args.nb_classes} -> {vec_len}")
            args.nb_classes = vec_len

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)} | Classes: {args.nb_classes}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("[ERROR] No images found after resolution. "
              "Check --data_path/--images_dir/--kaggle_path and filenames in CSV.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = AutoModelForImageClassification.from_pretrained(
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

    # NEW: per-class pos_weight & criterion (define BEFORE resume/eval)
    pos_weight = compute_pos_weight_from_csv(train_csv, device)
    print("pos_weight:", pos_weight.detach().cpu().numpy())  # optional
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Scaler (new API avoids deprecation warning)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # I/O
    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = SummaryWriter(args.log_dir) if (args.log_dir and not args.eval_only) else None

    # Resume (optionally override pos_weight from checkpoint)
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
        if "pos_weight" in ckpt:
            pos_weight = torch.tensor(ckpt["pos_weight"], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Eval only
    if args.eval_only:
        stats = evaluate(model, val_loader, device, criterion)
        print(f"[Eval] loss={stats['loss']:.4f} | f1_micro={stats['f1_micro']:.4f} | f1_macro={stats['f1_macro']:.4f}")
        return

    # Train
    best_f1 = 0.0
    t0 = time.time()
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler, criterion)
        val_stats  = evaluate(model, val_loader, device, criterion)

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

        # Save periodic checkpoints (SAVE pos_weight too)
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "pos_weight": pos_weight.detach().cpu().tolist(),  # NEW
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
                "pos_weight": pos_weight.detach().cpu().tolist(),  # NEW
            }, best_path)
            print(f"New best F1_micro={best_f1:.4f} -> {best_path}")

    total = str(datetime.timedelta(seconds=int(time.time() - t0)))
    print(f"\nDone. Best F1_micro={best_f1:.4f}. Total time: {total}")
    if log_writer:
        log_writer.close()


if __name__ == "__main__":
    main()
