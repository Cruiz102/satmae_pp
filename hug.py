#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Hiera (🤗 Transformers) on Kaggle NASA Geographical Objects (multi-label, 10 classes)

- CSV columns: FileName, Label Vector (a Python-style list string of 0/1s)
- Multi-label: BCEWithLogitsLoss + sigmoid thresholding (0.5)
- Cosine LR schedule with warmup
- Optional freezing of the backbone (train only classifier head)
- TensorBoard logging + torch checkpoints + HF save_pretrained
- Robust image path resolution:
    * Recursively indexes candidate image roots
    * Case-insensitive filename matching
    * Stem-based fallback (handles .JPG vs .jpg)
"""

import argparse
import ast
import datetime
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Optional: mixup/label smoothing via timm (if installed)
try:
    from timm.data.mixup import Mixup
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    TIMM_OK = True
except Exception:
    TIMM_OK = False


# =========================
# Helpers for image discovery
# =========================
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def _norm(s: str) -> str:
    return s.strip().lower()


def find_candidate_image_dirs(args):
    """Return a de-duplicated list of plausible image roots."""
    candidates = [
        os.path.join(args.data_path, args.images_dir),
        os.path.join(args.data_path, "..", "images"),
        os.path.join(args.data_path, "..", "__MACOSX", "kickoff_pack", "images"),
        "/tmp/nasa-geographical-objects-multilabel-dataset/images",
    ]
    # Kaggle cache
    kh = os.path.expanduser("~/.cache/kagglehub")
    if os.path.isdir(kh):
        for root, dirs, _ in os.walk(kh):
            if "images" in dirs:
                candidates.append(os.path.join(root, "images"))

    # Dedup and keep only existing dirs
    out, seen = [], set()
    for d in candidates:
        d = os.path.abspath(os.path.expanduser(d))
        if os.path.isdir(d) and d not in seen:
            out.append(d)
            seen.add(d)
    return out


def build_image_index(roots):
    """
    Walk roots and index image files two ways:
      - by_name: full filename (with ext) lowercased -> path
      - by_stem: filename without extension lowercased -> path
    """
    by_name, by_stem = {}, {}
    count = 0
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in SUPPORTED_EXTS:
                    full = os.path.join(dirpath, fn)
                    name_key = _norm(fn)
                    stem_key = _norm(os.path.splitext(fn)[0])
                    by_name.setdefault(name_key, full)
                    by_stem.setdefault(stem_key, full)
                    count += 1
    print(f"Indexed {count} images from {len(roots)} candidate root(s).")
    return {"by_name": by_name, "by_stem": by_stem}


def resolve_image_path(img_name: str, index: dict, images_dir_fallback: str | None = None):
    """
    Resolve a CSV filename to an actual path using the index.
    Tries:
      1) exact filename match (case-insensitive)
      2) stem match (any supported extension)
      3) fallback: join(images_dir_fallback, img_name) as last resort
    """
    name_key = _norm(img_name)
    if name_key in index["by_name"]:
        return index["by_name"][name_key]

    stem_key = _norm(os.path.splitext(img_name)[0])
    if stem_key in index["by_stem"]:
        return index["by_stem"][stem_key]

    # Try alt extensions with same stem
    for ext in SUPPORTED_EXTS:
        alt = stem_key + ext
        if alt in index["by_name"]:
            return index["by_name"][alt]

    if images_dir_fallback is not None:
        p = os.path.join(images_dir_fallback, img_name)
        if os.path.exists(p):
            return p

    return None


# =========================
# Dataset
# =========================
class KaggleGeographicalDataset(Dataset):
    """
    CSV columns:
      - FileName: image filename (relative or just name)
      - Label Vector: Python-style list string of 0/1s (multi-hot)
    We resolve filenames via an index (case-insensitive, stem-based).
    """

    def __init__(self, csv_file, images_dir, transform=None, target_transform=None, image_index=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.target_transform = target_transform
        self.in_c = 3
        self.image_index = image_index or {"by_name": {}, "by_stem": {}}

        print(f"Checking for missing images (with case-insensitive matching)...")
        valid_rows, missing_examples = [], []
        for idx, row in self.data.iterrows():
            img_name = str(row["FileName"])
            path = resolve_image_path(img_name, self.image_index, images_dir_fallback=self.images_dir)
            if path is not None and os.path.exists(path):
                valid_rows.append((idx, path))
            else:
                if len(missing_examples) < 5:
                    missing_examples.append(os.path.join(self.images_dir, img_name))

        total = len(self.data)
        missing = total - len(valid_rows)
        for m in missing_examples:
            print(f"Missing image example: {m}")
        if missing > len(missing_examples):
            print(f"...and {missing - len(missing_examples)} more missing images")
        print(f"Found {len(valid_rows)} valid images out of {total} total ({missing} missing)")

        # Keep only valid rows & store resolved filepaths
        self.data = self.data.iloc[[i for i, _ in valid_rows]].reset_index(drop=True)
        self.filepaths = [p for _, p in valid_rows]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image unreadable: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        label_str = self.data.iloc[idx]["Label Vector"]
        label_list = ast.literal_eval(label_str)
        label = torch.tensor(label_list, dtype=torch.float32)

        if self.transform is not None:
            pixel_values = self.transform(image_rgb)
        else:
            # Fallback simple preprocessing (224, [0,1] float, CHW)
            img = cv2.resize(image_rgb, (224, 224)).astype(np.float32) / 255.0
            pixel_values = torch.from_numpy(img).permute(2, 0, 1)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return pixel_values, label


# =========================
# HF Processor transform
# =========================
class HFProcessorTransform:
    """
    Wraps a Hugging Face AutoImageProcessor (Bit/ViT-style for Hiera).
    Accepts RGB HWC np.ndarray; returns CHW float tensor ready for model.
    """
    def __init__(self, image_processor):
        self.processor = image_processor

    def __call__(self, image_rgb_hwc: np.ndarray) -> torch.Tensor:
        out = self.processor(images=image_rgb_hwc, return_tensors="pt")
        return out["pixel_values"][0]  # CHW tensor


# =========================
# Cosine LR with Warmup
# =========================
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters))) / 2
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    clip_grad=None, mixup_fn=None, lr_schedule=None):
    model.train()
    total_loss, all_predictions, all_targets = 0.0, [], []

    for batch_idx, (samples, targets) in enumerate(data_loader):
        if lr_schedule is not None:
            it = epoch * len(data_loader) + batch_idx
            if it < len(lr_schedule):
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_schedule[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        optimizer.zero_grad(set_to_none=True)

        out = model(pixel_values=samples)  # Hiera expects pixel_values -> logits
        logits = out.logits

        loss = criterion(logits, targets)
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        total_loss += loss.item()

        # Metrics (skip during mixup)
        if mixup_fn is None:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_predictions.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        if batch_idx % 50 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(data_loader)} | Loss {loss.item():.4f} | LR {cur_lr:.6e}")

    avg_loss = total_loss / len(data_loader)

    if all_predictions and mixup_fn is None:
        y_pred = np.vstack(all_predictions)
        y_true = np.vstack(all_targets)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    else:
        f1_micro = f1_macro = 0.0

    return {"loss": avg_loss, "f1_micro": f1_micro, "f1_macro": f1_macro}


@torch.no_grad()
def evaluate(model, data_loader, device, threshold=0.5):
    model.eval()
    total_loss, all_predictions, all_targets = 0.0, [], []
    criterion = nn.BCEWithLogitsLoss()

    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        out = model(pixel_values=samples)
        logits = out.logits

        loss = criterion(logits, targets)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > threshold).float()
        all_predictions.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    y_pred = np.vstack(all_predictions)
    y_true = np.vstack(all_targets)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    avg_loss = total_loss / len(data_loader)

    return {"loss": avg_loss, "f1_micro": f1_micro, "f1_macro": f1_macro}


def save_checkpoint(model, optimizer, epoch, args, filename=None):
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, filename)
    torch.save(payload, path)
    print(f"Checkpoint saved: {path}")


# =========================
# Arg parsing
# =========================
def get_args_parser():
    p = argparse.ArgumentParser("Hiera fine-tuning for Kaggle Geographical Dataset", add_help=True)

    # Training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip_grad", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=5)

    # Multi-label task
    p.add_argument("--nb_classes", type=int, default=10, help="number of labels/classes")

    # Mixup / smoothing (optional, needs timm)
    p.add_argument("--smoothing", type=float, default=0.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--cutmix", type=float, default=0.0)

    # Data
    p.add_argument("--data_path", type=str, default="/home/cesar/kickoff_material/kickoff_pack")
    p.add_argument("--train_csv", type=str, default="data/train.csv")
    p.add_argument("--val_csv", type=str, default="data/val.csv")
    p.add_argument("--images_dir", type=str, default="images")

    # Device / logging
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_mem", action="store_true", default=True)
    p.add_argument("--output_dir", type=str, default="./finetune_hiera_logs")
    p.add_argument("--log_dir", type=str, default="./finetune_hiera_logs")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_only", action="store_true", help="evaluation only")

    # HF (Hiera)
    p.add_argument("--hf_ckpt", type=str, default="facebook/hiera-large-224-in1k-hf",
                   help="Hiera checkpoint from Hugging Face Hub")

    # Freezing
    p.add_argument("--freeze_backbone", action="store_true", help="train only the classifier head")

    return p


# =========================
# Main
# =========================
def main(args):
    from transformers import AutoConfig, AutoImageProcessor, HieraForImageClassification

    print("Hiera fine-tuning - Kaggle Geographical Objects (multi-label)")
    print(f"Args:\n{json.dumps(vars(args), indent=2)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Build processor and datasets
    print("Loading HF image processor...")
    image_processor = AutoImageProcessor.from_pretrained(args.hf_ckpt)  # Handles resize/normalize/crop for Hiera
    transform = HFProcessorTransform(image_processor)

    # Discover image roots and build an index
    print("Discovering image roots...")
    candidate_roots = find_candidate_image_dirs(args)
    for r in candidate_roots:
        print(f"  - {r}")
    image_index = build_image_index(candidate_roots)

    print("Loading datasets...")
    train_csv = os.path.join(args.data_path, args.train_csv)
    val_csv = os.path.join(args.data_path, args.val_csv)
    images_path = os.path.join(args.data_path, args.images_dir)  # used as a fallback

    ds_train = KaggleGeographicalDataset(train_csv, images_path, transform=transform, image_index=image_index)
    ds_val = KaggleGeographicalDataset(val_csv, images_path, transform=transform, image_index=image_index)
    print(f"Train dataset size: {len(ds_train)}")
    print(f"Val dataset size:   {len(ds_val)}")

    # Bail out early if zero, with guidance
    if len(ds_train) == 0 or len(ds_val) == 0:
        print("\n[ERROR] No images were resolved from your CSV filenames.")
        print("Hints:")
        print("  • Ensure the dataset is fully extracted (not just the CSV).")
        print("  • Check that image names in CSV match actual files (case/extension).")
        print("  • If images live elsewhere, add a symlink, e.g.:")
        print(f"      ln -s /path/to/actual/images {images_path}")
        print("  • Or move/rename the folder so one of the listed roots above contains the images.")
        return

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )

    # Setup mixup if requested (optional; requires timm)
    mixup_fn = None
    if (args.mixup > 0.0 or args.cutmix > 0.0 or args.smoothing > 0.0) and not TIMM_OK:
        print("[WARN] timm not installed; disabling mixup/label smoothing.")
    elif TIMM_OK and (args.mixup > 0.0 or args.cutmix > 0.0):
        print("Mixup activated.")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    # Build model
    print("Loading Hiera model...")
    config = AutoConfig.from_pretrained(args.hf_ckpt, num_labels=args.nb_classes)
    model = HieraForImageClassification.from_pretrained(
        args.hf_ckpt, config=config, ignore_mismatched_sizes=True
    ).to(device)

    # Optionally freeze backbone; keep only the classifier head trainable
    if args.freeze_backbone:
        print("Freezing backbone; training only classifier head.")
        frozen, trainable = 0, 0
        for name, p in model.named_parameters():
            keep = ('classifier' in name)  # Hiera’s head params contain "classifier"
            p.requires_grad = keep
            if keep:
                trainable += p.numel()
            else:
                frozen += p.numel()
        print(f"Frozen params: {frozen/1e6:.2f}M | Trainable params: {trainable/1e6:.2f}M")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable/1e6:.2f}M")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Criterion
    if mixup_fn is not None and TIMM_OK:
        criterion = SoftTargetCrossEntropy()
    elif TIMM_OK and args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.BCEWithLogitsLoss()  # robust for multi-label

    print(f"Criterion: {criterion.__class__.__name__}")

    # LR schedule
    lr_schedule = cosine_scheduler(
        base_value=args.lr, final_value=1e-6,
        epochs=args.epochs, niter_per_ep=len(dl_train),
        warmup_epochs=args.warmup_epochs, start_warmup_value=1e-6
    )

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        chk = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(chk["model"])
        optimizer.load_state_dict(chk["optimizer"])
        start_epoch = chk.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Eval-only
    if args.eval_only:
        stats = evaluate(model, dl_val, device)
        print(f"[Eval] Loss: {stats['loss']:.4f} | F1 micro: {stats['f1_micro']:.4f} | F1 macro: {stats['f1_macro']:.4f}")
        return

    # Logging
    log_writer = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    # Train
    print(f"Start training for {args.epochs} epochs")
    best_f1 = 0.0
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs-1}")
        train_stats = train_one_epoch(
            model, criterion, dl_train, optimizer, device, epoch,
            clip_grad=args.clip_grad, mixup_fn=mixup_fn, lr_schedule=lr_schedule,
        )
        val_stats = evaluate(model, dl_val, device)

        print(f"Train  | loss {train_stats['loss']:.4f} | f1_micro {train_stats['f1_micro']:.4f} | f1_macro {train_stats['f1_macro']:.4f}")
        print(f"Valid  | loss {val_stats['loss']:.4f} | f1_micro {val_stats['f1_micro']:.4f} | f1_macro {val_stats['f1_macro']:.4f}")

        best_f1 = max(best_f1, val_stats["f1_micro"])
        print(f"Best F1 micro so far: {best_f1:.4f}")

        # Log
        if log_writer is not None:
            log_writer.add_scalar("train/loss", train_stats["loss"], epoch)
            log_writer.add_scalar("train/f1_micro", train_stats["f1_micro"], epoch)
            log_writer.add_scalar("train/f1_macro", train_stats["f1_macro"], epoch)
            log_writer.add_scalar("val/loss", val_stats["loss"], epoch)
            log_writer.add_scalar("val/f1_micro", val_stats["f1_micro"], epoch)
            log_writer.add_scalar("val/f1_macro", val_stats["f1_macro"], epoch)
            log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Save torch checkpoint
        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            save_checkpoint(model, optimizer, epoch, args)

        # Append to log.txt
        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                row = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"val_{k}": v for k, v in val_stats.items()},
                    "epoch": epoch,
                    "best_f1_micro": best_f1,
                }
                f.write(json.dumps(row) + "\n")

    dt = str(datetime.timedelta(seconds=int(time.time() - t0)))
    print(f"\nTraining completed in {dt} | Best F1 micro: {best_f1:.4f}")

    # Save HF artifacts for easy reloading
    if args.output_dir:
        model.save_pretrained(args.output_dir)
        image_processor.save_pretrained(args.output_dir)
        print(f"Saved HF model + processor to: {args.output_dir}")

    if log_writer is not None:
        log_writer.close()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
