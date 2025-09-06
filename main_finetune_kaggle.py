# --------------------------------------------------------
# Simplified Single-GPU version of SatMAE++ fine-tuning for Kaggle Dataset
# Modified to work with NASA Geographical Objects dataset
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
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import cv2
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import timm
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# Import SatMAE components if available
try:
    import util.lr_decay as lrd
    from util.pos_embed import interpolate_pos_embed
    import models_vit
    import models_vit_group_channels
    SATMAE_AVAILABLE = True
except ImportError:
    print("SatMAE modules not found. Using basic CNN model only.")
    SATMAE_AVAILABLE = False

assert timm.__version__ >= "0.3.2"


class KaggleGeographicalDataset(Dataset):
    """Custom dataset for NASA Geographical Objects from Kaggle"""
    
    def __init__(self, csv_file, images_dir, transform=None, target_transform=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Number of channels (RGB = 3)
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
                if missing_count <= 5:  # Only print first 5 missing files
                    print(f"Missing image: {img_path}")
                    
        if missing_count > 5:
            print(f"... and {missing_count - 5} more missing images")
            
        print(f"Found {len(valid_indices)} valid images out of {len(self.data)} total ({missing_count} missing)")
        
        # Keep only rows with existing images
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image path and load image
        img_name = self.data.iloc[idx]['FileName']
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load image using cv2 (BGR format)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get label vector and convert to tensor
        label_str = self.data.iloc[idx]['Label Vector']
        label_vector = ast.literal_eval(label_str)  # Convert string to list
        label = torch.tensor(label_vector, dtype=torch.float32)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default preprocessing: resize, normalize, and convert to tensor
            image = cv2.resize(image, (224, 224))  # Resize to 96x96 (SatMAE default)
            image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label




def get_args_parser():
    parser = argparse.ArgumentParser('SatMAE++ fine-tuning for Kaggle Geographical Dataset', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for single GPU training')
    parser.add_argument('--epochs', default=30, type=int)

    # Model parameters
    parser.add_argument('--model_type', default='group_c', choices=['group_c', 'vanilla'],
                        help='Model type to use')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train (for SatMAE models)')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--patch_size', default=8, type=int, help='patch embedding patch size')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT', help='Drop path rate')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0, help='Label smoothing')
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup alpha (0 to disable)')
    parser.add_argument('--cutmix', type=float, default=0.0, help='cutmix alpha (0 to disable)')

    # Finetuning params
    parser.add_argument('--finetune', default=None, help='finetune from checkpoint')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze all layers except the head/classifier')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)

    # Dataset parameters - Updated for Kaggle dataset
    parser.add_argument('--data_path', default='/home/cesar/kickoff_material/kickoff_pack', type=str,
                        help='Path to the kickoff_pack folder containing data/ and images/')
    parser.add_argument('--train_csv', default='data/train.csv', type=str,
                        help='Train CSV file path (relative to data_path)')
    parser.add_argument('--val_csv', default='data/val.csv', type=str,
                        help='Validation CSV file path (relative to data_path)')
    parser.add_argument('--images_dir', default='images', type=str,
                        help='Images directory (relative to data_path)')
    parser.add_argument('--kaggle_path', default=None, type=str,
                        help='Path to kaggle downloaded dataset (if different from data_path)')
    
    # For compatibility with SatMAE
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[], help="Bands to group for GroupC vit")

    parser.add_argument('--nb_classes', default=10, type=int, help='number of classes (geographical features)')
    parser.add_argument('--output_dir', default='./finetune_logs_kaggle', help='path where to save')
    parser.add_argument('--log_dir', default='./finetune_logs_kaggle', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=5, help='How frequently (in epochs) to save ckpt')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """Cosine learning rate scheduler with warmup"""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters))) / 2

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def train_one_epoch_simple(model, criterion, data_loader, optimizer, device, epoch, 
                          clip_grad=None, mixup_fn=None, lr_schedule=None):
    """Simplified training loop for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    for batch_idx, (samples, targets) in enumerate(data_loader):
        # Update learning rate
        if lr_schedule is not None:
            it = epoch * len(data_loader) + batch_idx
            if it < len(lr_schedule):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule[it]
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        optimizer.zero_grad()
        
        outputs = model(samples)
        loss = criterion(outputs, targets)
        
        loss.backward()
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_samples += samples.size(0)
        
        # Store predictions and targets for F1 calculation (only if not using mixup)
        if mixup_fn is None:
            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                all_predictions.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(data_loader)}, '
                  f'Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate F1 score if we have predictions
    if all_predictions and mixup_fn is None:
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        f1_micro = f1_score(all_targets, all_predictions, average='micro')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')
    else:
        f1_micro = f1_macro = 0.0
    
    return {'loss': avg_loss, 'f1_micro': f1_micro, 'f1_macro': f1_macro}


def evaluate_simple(model, data_loader, device):
    """Simplified evaluation"""
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(samples)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Store predictions
            preds = torch.sigmoid(outputs) > 0.5
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    f1_micro = f1_score(all_targets, all_predictions, average='micro')
    f1_macro = f1_score(all_targets, all_predictions, average='macro')
    avg_loss = total_loss / len(data_loader)
    
    return {'loss': avg_loss, 'f1_micro': f1_micro, 'f1_macro': f1_macro}


def save_checkpoint(model, optimizer, epoch, args, filename=None):
    """Save model checkpoint"""
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    
    filepath = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def find_images_path(args):
    """Find the correct path for images"""
    possible_paths = [
        os.path.join(args.data_path, args.images_dir),
    ]
    
    # If kaggle_path is provided, try that too
    if args.kaggle_path:
        possible_paths.insert(0, os.path.join(args.kaggle_path, 'images'))
    
    # Also try common kaggle download locations
    home_dir = os.path.expanduser("~")
    kaggle_cache = os.path.join(home_dir, '.cache', 'kagglehub')
    if os.path.exists(kaggle_cache):
        # Look for any directory containing images
        for root, dirs, files in os.walk(kaggle_cache):
            if 'images' in dirs:
                possible_paths.append(os.path.join(root, 'images'))
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it actually contains image files
            image_files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if len(image_files) > 0:
                print(f"Found {len(image_files)} images in: {path}")
                return path
    
    print(f"Warning: No images found in any of these paths: {possible_paths}")
    return os.path.join(args.data_path, args.images_dir)  # fallback


def main(args):
    print('Single GPU Training - Kaggle Geographical Dataset')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Build datasets
    train_csv_path = os.path.join(args.data_path, args.train_csv)
    val_csv_path = os.path.join(args.data_path, args.val_csv)
    images_path = os.path.join(args.data_path, args.images_dir)
    
    # Check if images directory exists, if not try alternative locations
    if not os.path.exists(images_path):
        print(f"Images directory not found at: {images_path}")
        
        # Try some alternative locations
        alternative_paths = [
            os.path.join(args.data_path, '..', '__MACOSX', 'kickoff_pack', 'images'),
            os.path.join(args.data_path, '..', 'images'),
            '/tmp/nasa-geographical-objects-multilabel-dataset/images',
            # Add kagglehub cache locations
            os.path.expanduser('~/.cache/kagglehub/datasets/olebro/nasa-geographical-objects-multilabel-dataset/versions/1/images'),
            os.path.expanduser('~/.cache/kagglehub/datasets/olebro/nasa-geographical-objects-multilabel-dataset/versions/2/images'),
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"Found images at alternative location: {alt_path}")
                images_path = alt_path
                break
        else:
            print("Warning: No images directory found. Please check the data setup.")
            print("Available directories:")
            for item in os.listdir(args.data_path):
                item_path = os.path.join(args.data_path, item)
                if os.path.isdir(item_path):
                    print(f"  - {item}/")
    
    print("Loading datasets...")
    print(f"Train CSV: {train_csv_path}")
    print(f"Val CSV: {val_csv_path}")
    print(f"Images path: {images_path}")
    
    dataset_train = KaggleGeographicalDataset(train_csv_path, images_path)
    dataset_val = KaggleGeographicalDataset(val_csv_path, images_path)

    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Val dataset size: {len(dataset_val)}")

    # Setup logging
    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Data loaders
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Setup mixup (disabled by default for this dataset)
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    if SATMAE_AVAILABLE and args.model_type == 'group_c':
        print("Using SatMAE Group Channel model")
        if len(args.grouped_bands) == 0:
            args.grouped_bands = [[0, 1, 2]]  # RGB channels
        
        print(f"Grouping bands {args.grouped_bands}")
        model = models_vit_group_channels.__dict__[args.model](
            patch_size=args.patch_size, img_size=args.input_size, in_chans=dataset_train.in_c,
            channel_groups=args.grouped_bands, num_classes=args.nb_classes,
            drop_path_rate=args.drop_path, global_pool=args.global_pool
        )
    
    elif SATMAE_AVAILABLE and args.model_type == 'vanilla':
        print("Using SatMAE Vanilla model")
        model = models_vit.__dict__[args.model](
            patch_size=args.patch_size, img_size=args.input_size, in_chans=dataset_train.in_c,
            num_classes=args.nb_classes, drop_path_rate=args.drop_path, global_pool=args.global_pool,
        )
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Load pre-trained weights if available
    if args.finetune and not args.eval:
        if os.path.exists(args.finetune):
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
            print("Load pre-trained checkpoint from: %s" % args.finetune)
            
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
                
            state_dict = model.state_dict()

            # Remove incompatible keys for SatMAE models
            if SATMAE_AVAILABLE and args.model_type in ['group_c', 'vanilla']:
                for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
                    if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
                
                # interpolate position embedding
                interpolate_pos_embed(model, checkpoint_model)

            # Load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(f"Loading checkpoint: {msg}")

            # manually initialize fc layer for SatMAE models
            if SATMAE_AVAILABLE and args.model_type in ['group_c', 'vanilla']:
                trunc_normal_(model.head.weight, std=2e-5)
        else:
            print(f"Checkpoint file {args.finetune} not found. Training from scratch.")

    model.to(device)
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        print("Freezing backbone layers, only training the head/classifier...")
        frozen_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            # Keep head/classifier layers trainable
            if any(head_name in name for head_name in ['head', 'classifier', 'fc']):
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"Keeping trainable: {name}")
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"Frozen parameters: {frozen_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # Build optimizer
    if SATMAE_AVAILABLE and not args.freeze_backbone:
        # Use layer-wise learning rate decay for SatMAE models (only when not freezing)
        param_groups = lrd.param_groups_lrd(model, args.weight_decay,
                                            no_weight_decay_list=model.no_weight_decay(),
                                            layer_decay=args.layer_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    else:
        # Simple optimizer for frozen backbone or non-SatMAE models
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Setup criterion
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    print("criterion = %s" % str(criterion))

    # Setup learning rate schedule
    lr_schedule = cosine_scheduler(
        args.lr, 1e-6, args.epochs, len(data_loader_train),
        warmup_epochs=args.warmup_epochs, start_warmup_value=1e-6
    )    # Resume from checkpoint if specified
    start_epoch = args.start_epoch
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from checkpoint: {args.resume}, epoch {start_epoch}")

    # Evaluation only
    if args.eval:
        test_stats = evaluate_simple(model, data_loader_val, device)
        print(f"Evaluation on {len(dataset_val)} test images:")
        print(f"Loss: {test_stats['loss']:.4f}")
        print(f"F1 Micro: {test_stats['f1_micro']:.4f}")
        print(f"F1 Macro: {test_stats['f1_macro']:.4f}")
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_f1 = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Training
        train_stats = train_one_epoch_simple(
            model, criterion, data_loader_train, optimizer, device, epoch,
            clip_grad=args.clip_grad, mixup_fn=mixup_fn, lr_schedule=lr_schedule
        )

        # Validation
        test_stats = evaluate_simple(model, data_loader_val, device)

        print(f"Training - Loss: {train_stats['loss']:.4f}, F1 Micro: {train_stats['f1_micro']:.4f}, F1 Macro: {train_stats['f1_macro']:.4f}")
        print(f"Validation - Loss: {test_stats['loss']:.4f}, F1 Micro: {test_stats['f1_micro']:.4f}, F1 Macro: {test_stats['f1_macro']:.4f}")
        
        best_f1 = max(best_f1, test_stats["f1_micro"])
        print(f'Best F1 Micro so far: {best_f1:.4f}')

        # Logging
        if log_writer is not None:
            log_writer.add_scalar('train/loss', train_stats['loss'], epoch)
            log_writer.add_scalar('train/f1_micro', train_stats['f1_micro'], epoch)
            log_writer.add_scalar('train/f1_macro', train_stats['f1_macro'], epoch)
            log_writer.add_scalar('val/loss', test_stats['loss'], epoch)
            log_writer.add_scalar('val/f1_micro', test_stats['f1_micro'], epoch)
            log_writer.add_scalar('val/f1_macro', test_stats['f1_macro'], epoch)
            log_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            save_checkpoint(model, optimizer, epoch, args)

        # Save logs
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
            'best_f1_micro': best_f1
        }

        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nTraining completed in {total_time_str}')
    print(f'Final best F1 Micro: {best_f1:.4f}')

    if log_writer is not None:
        log_writer.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
