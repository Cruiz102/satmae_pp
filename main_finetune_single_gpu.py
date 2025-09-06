# --------------------------------------------------------
# Simplified Single-GPU version of SatMAE++ fine-tuning
# Based on the original distributed training script
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
from util.datasets_finetune import build_fmow_dataset
from util.pos_embed import interpolate_pos_embed

import models_vit
import models_vit_group_channels

assert timm.__version__ >= "0.3.2"


def get_args_parser():
    parser = argparse.ArgumentParser('SatMAE++ fine-tuning for image classification (Single GPU)', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for single GPU training')
    parser.add_argument('--epochs', default=30, type=int)

    # Model parameters
    parser.add_argument('--model_type', default='group_c', choices=['group_c', 'vanilla'],
                        help='Use channel model')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=96, type=int, help='images input size')
    parser.add_argument('--patch_size', default=8, type=int, help='patch embedding patch size')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1., help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='./output_dir/checkpoint-50.pth', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--base_path', default='dataset/eurosat/2750/', type=str, help='dataset folder path')
    parser.add_argument('--train_path', default='dataset/fmow_sentinel/train.csv', type=str,
                        help='Train .csv path')
    parser.add_argument('--test_path', default='dataset/fmow_sentinel/val.csv', type=str,
                        help='Test .csv path')
    parser.add_argument('--dataset_type', default='sentinel', choices=['rgb', 'sentinel', 'euro_sat', 'resisc', 'ucmerced'],
                        help='Whether to use fmow rgb, sentinel, or other dataset.')
    parser.add_argument('--masked_bands', default=None, nargs='+', type=int,
                        help='Sequence of band indices to mask (with mean val) in sentinel dataset')
    parser.add_argument('--dropped_bands', type=int, nargs='+', default=None,
                        help="Which bands (0 indexed) to drop from sentinel data.")
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[], help="Bands to group for GroupC vit")

    parser.add_argument('--nb_classes', default=62, type=int, help='number of the classification types')
    parser.add_argument('--output_dir', default='./finetune_logs_single_gpu', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./finetune_logs_single_gpu', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=5, help='How frequently (in epochs) to save ckpt')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
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
    correct = 0
    
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
        
        # Calculate accuracy for non-mixup cases
        if mixup_fn is None:
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(data_loader)}, '
                  f'Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total_samples if mixup_fn is None else 0.0
    
    return {'loss': avg_loss, 'acc1': accuracy}


def evaluate_simple(model, data_loader, device):
    """Simplified evaluation"""
    model.eval()
    
    total_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    total_samples = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(samples)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Top-1 accuracy
            _, pred = outputs.topk(1, 1, True, True)
            correct_1 += pred.eq(targets.view(-1, 1).expand_as(pred)).sum().item()
            
            # Top-5 accuracy
            _, pred_5 = outputs.topk(5, 1, True, True)
            correct_5 += pred_5.eq(targets.view(-1, 1).expand_as(pred_5)).sum().item()
            
            total_samples += targets.size(0)
    
    acc1 = 100. * correct_1 / total_samples
    acc5 = 100. * correct_5 / total_samples
    avg_loss = total_loss / len(data_loader)
    
    return {'loss': avg_loss, 'acc1': acc1, 'acc5': acc5}


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


def main(args):
    print('Single GPU Training')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Build datasets
    dataset_train = build_fmow_dataset(is_train=True, args=args)
    dataset_val = build_fmow_dataset(is_train=False, args=args)

    # Simple samplers for single GPU
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Val dataset size: {len(dataset_val)}")

    # Setup logging
    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Setup mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Define the model
    if args.model_type == 'group_c':
        # Workaround because action append will add to default list
        if len(args.grouped_bands) == 0:
            args.grouped_bands = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]

        print(f"Grouping bands {args.grouped_bands}")

        model = models_vit_group_channels.__dict__[args.model](
            patch_size=args.patch_size, img_size=args.input_size, in_chans=dataset_train.in_c,
            channel_groups=args.grouped_bands, num_classes=args.nb_classes,
            drop_path_rate=args.drop_path, global_pool=args.global_pool
        )
    else:
        model = models_vit.__dict__[args.model](
            patch_size=args.patch_size, img_size=args.input_size, in_chans=dataset_train.in_c,
            num_classes=args.nb_classes, drop_path_rate=args.drop_path, global_pool=args.global_pool,
        )

    # Load pre-trained weights
    if args.finetune and not args.eval:
        if os.path.exists(args.finetune):
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
            print("Load pre-trained checkpoint from: %s" % args.finetune)
            
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()

            # Remove incompatible keys
            for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            
            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)
        else:
            print(f"Checkpoint file {args.finetune} not found. Training from scratch.")

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # Build optimizer with layer-wise lr decay
    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
                                        no_weight_decay_list=model.no_weight_decay(),
                                        layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    # Setup criterion
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # Setup learning rate schedule
    lr_schedule = cosine_scheduler(
        args.lr, 1e-6, args.epochs, len(data_loader_train),
        warmup_epochs=args.warmup_epochs, start_warmup_value=1e-6
    )

    # Resume from checkpoint if specified
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
        print(f"Evaluation on {len(dataset_val)} test images- acc1: {test_stats['acc1']:.2f}%, "
              f"acc5: {test_stats['acc5']:.2f}%")
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Training
        train_stats = train_one_epoch_simple(
            model, criterion, data_loader_train, optimizer, device, epoch,
            clip_grad=args.clip_grad, mixup_fn=mixup_fn, lr_schedule=lr_schedule
        )

        # Validation
        test_stats = evaluate_simple(model, data_loader_val, device)

        print(f"Training - Loss: {train_stats['loss']:.4f}")
        if mixup_fn is None:
            print(f"Training - Acc@1: {train_stats['acc1']:.2f}%")
        print(f"Validation - Loss: {test_stats['loss']:.4f}, Acc@1: {test_stats['acc1']:.2f}%, Acc@5: {test_stats['acc5']:.2f}%")
        
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy so far: {max_accuracy:.2f}%')

        # Logging
        if log_writer is not None:
            log_writer.add_scalar('train/loss', train_stats['loss'], epoch)
            log_writer.add_scalar('val/loss', test_stats['loss'], epoch)
            log_writer.add_scalar('val/acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('val/acc5', test_stats['acc5'], epoch)
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
            'max_accuracy': max_accuracy
        }

        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nTraining completed in {total_time_str}')
    print(f'Final max accuracy: {max_accuracy:.2f}%')

    if log_writer is not None:
        log_writer.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
