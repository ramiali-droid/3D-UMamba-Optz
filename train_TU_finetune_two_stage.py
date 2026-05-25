"""
Two-stage fine-tuning script (single file, self-contained instructions).

Workflow (automatic):
  1) Stage A - Head-only training (adapt classifier):
       - Freeze backbone, train only head layers for a few epochs
       - Use higher LR for head
  2) Stage B - Progressive unfreeze (fine-tune backbone):
       - Unfreeze additional layers (or whole model) and continue training
       - Use lower LR for pretrained params, slightly higher for head

Usage example (quick):
  python3 train_TU_finetune_two_stage.py --model mamba_msg --stage1_epochs 8 --stage2_epochs 40 --gpu 0

Notes / decisions inside the file as comments.
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import datetime
from zoneinfo import ZoneInfo

import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

# Dataloader (simple fixed split): uses the SIMPLE loader created earlier
from data_utils.TUBBlockDataLoader_v3_SIMPLE import TUBDataset
from ErrorMatrix import ConfusionMatrix

# Minimal utilities copied/adapted from your repo

def load_state_dict_ignore_channels(model, ckpt_path, strict=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    model_dict = model.state_dict()
    new_state = {}

    for k, v in state_dict.items():
        if k not in model_dict:
            continue

        if v.shape == model_dict[k].shape:
            new_state[k] = v
        else:
            # Handle channel mismatch (slice along channel dim)
            # Works for common convolution/linear weights
            try:
                min_c = min(v.shape[1], model_dict[k].shape[1])
                sliced = v[:, :min_c, ...]
                target = model_dict[k].clone()
                target[:, :min_c, ...] = sliced
                new_state[k] = target
                print(f"[PARTIAL LOAD] {k}: {v.shape} → {model_dict[k].shape}")
            except Exception:
                # fallback: skip
                print(f"[SKIP LOAD] {k}: incompatible shapes {v.shape} vs {model_dict[k].shape}")
                continue

    model.load_state_dict(new_state, strict=strict)


def set_requires_grad_by_name(model, name_list, requires_grad):
    for n, p in model.named_parameters():
        if any(k in n for k in name_list):
            p.requires_grad = requires_grad


def collect_param_groups(model, head_names, lr_head, lr_backbone, weight_decay):
    head_params = []
    backbone_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in head_names):
            head_params.append(p)
        else:
            backbone_params.append(p)
    param_groups = []
    if head_params:
        param_groups.append({'params': head_params, 'lr': lr_head})
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr_backbone})
    return param_groups


def eval_model(classifier, dataloader, criterion, num_classes, device, writer=None, epoch=None, weight=None, class_names=None):
    classifier.eval()
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=class_names)
    total_loss = 0.0
    total_seen = 0
    total_correct = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for points, target, fps_idx, series_idx in tqdm(dataloader, desc='Eval', leave=False):
            points = torch.Tensor(points).float().to(device)
            target = target.long().to(device)
            fps_idx = fps_idx.long().to(device)
            series_idx = series_idx.long().to(device)
            points = points.transpose(2, 1)

            out = classifier(points, fps_idx, series_idx)
            seg_pred = out.contiguous().cpu().data.numpy()
            out_flat = out.contiguous().view(-1, num_classes)
            target_flat = target.view(-1)

            # call criterion with / without weight depending on signature
            try:
                loss = criterion(out_flat, target_flat, weight)
            except TypeError:
                loss = criterion(out_flat, target_flat)
            total_loss += loss.item()

            pred = np.argmax(seg_pred, axis=2)
            batch_label = target.cpu().data.numpy()
            correct = np.sum((pred == batch_label))
            total_correct += correct
            total_seen += batch_label.size

            pred_flat = pred.reshape(-1,1)
            label_flat = batch_label.reshape(-1,1)
            confusion.update(pred_flat, label_flat)

    ave_F1, miou, acc = confusion.summary()
    mean_loss = total_loss / max(1, num_batches)
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/Eval', mean_loss, epoch)
        writer.add_scalar('mIoU/Eval', miou, epoch)
    return mean_loss, miou, ave_F1, acc


def train_epoch(classifier, dataloader, optimizer, scheduler, criterion, device, num_classes, writer=None, epoch=None, weight=None):
    classifier.train()
    total_loss = 0.0
    total_seen = 0
    total_correct = 0
    num_batches = len(dataloader)

    for points, target, fps_idx, series_idx in tqdm(dataloader, desc='Train', leave=False):
        optimizer.zero_grad()
        points = torch.Tensor(points).float().to(device)
        target = target.long().to(device)
        fps_idx = fps_idx.long().to(device)
        series_idx = series_idx.long().to(device)
        points = points.transpose(2, 1)

        out = classifier(points, fps_idx, series_idx)
        # model returns (B, N, C) where C == num_classes
        out_flat = out.contiguous().view(-1, num_classes)
        target_flat = target.view(-1)

        try:
            loss = criterion(out_flat, target_flat, weight)
        except TypeError:
            loss = criterion(out_flat, target_flat)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        pred = out_flat.cpu().data.max(1)[1].numpy()
        total_correct += np.sum(pred == target_flat.cpu().numpy())
        total_seen += target_flat.numel()

    mean_loss = total_loss / max(1, num_batches)
    acc = total_correct / float(total_seen) if total_seen > 0 else 0.0
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/Train', mean_loss, epoch)
        writer.add_scalar('Acc/Train', acc, epoch)
    return mean_loss, acc


def parse_args():
    berlin_tz = ZoneInfo("Europe/Berlin")
    berlin_time = datetime.datetime.now(berlin_tz)
    berlin_iso = berlin_time.isoformat().split('T')[0]

    parser = argparse.ArgumentParser('Two-stage fine-tune')
    parser.add_argument('--model', type=str, default='mamba_msg')
    parser.add_argument('--pretrained_ckpt', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--stage1_epochs', type=int, default=8)
    parser.add_argument('--stage2_epochs', type=int, default=40)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--lr_backbone', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--npoint', type=int, default=8192)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--data_root', type=str, default='/home/ramiali/3dumamba/data/TUB/combined/crt')
    parser.add_argument('--log_dir', type=str, default=f'finetune_{berlin_iso}')
    parser.add_argument('--bn_update', action='store_true', help='Allow BN running stats to update during stage2')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup logging dir
    exp_dir = Path('./log') / args.log_dir
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_dir / 'checkpoints'
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # model import
    import importlib
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Add the repository's models/ directory to sys.path so importlib can find model modules
    sys.path.append(os.path.join(BASE_DIR, 'models'))
    try:
        MODEL = importlib.import_module(args.model)
    except ModuleNotFoundError:
        # If the model is provided as e.g. 'mamba_msg_s3dis' but not found directly,
        # try importing as a submodule of the models package (if it's a package),
        # or raise the original error for the user to inspect.
        try:
            MODEL = importlib.import_module(f"models.{args.model}")
        except Exception:
            raise
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    model = MODEL.get_model(args.num_classes, fps_n_list=[512,128,32]).to(device)
    model = DataParallel(model)

    # load pretrained if provided
    if args.pretrained_ckpt:
        try:
            load_state_dict_ignore_channels(model, args.pretrained_ckpt)
            print('[INFO] Loaded pretrained checkpoint')
        except Exception as e:
            print('[WARN] Could not load pretrained:', e)

    # default head layers to train in stage1. tune if needed.
    head_names = ['mamba3', 'sa4', 'fp4', 'fp3', 'fp2', 'fp1', 'conv1', 'conv2']

    # Data
    train_ds = TUBDataset(split='train', data_root=args.data_root, npoints=args.npoint, augment=True)
    test_ds = TUBDataset(split='test', data_root=args.data_root, npoints=args.npoint, augment=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # criterion
    criterion = MODEL.get_loss_weighted().to(device) if True else MODEL.get_loss().to(device)

    # compute label weight tensor from dataset if available
    try:
        weight = torch.tensor(train_ds.labelweights).float().to(device)
        print('[INFO] Using label weights from dataset')
    except Exception:
        weight = None
        print('[INFO] No label weights available; using unweighted loss')

    # stage 1: freeze all except head
    for n, p in model.named_parameters():
        p.requires_grad = False
    set_requires_grad_by_name(model, head_names, True)

    # param groups for stage 1 (only head params will be in optimizer)
    pg = collect_param_groups(model, head_names, args.lr_head, 0.0, args.weight_decay)
    optimizer = torch.optim.Adam(pg, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.stage1_epochs * len(train_loader), eta_min=args.lr_head*0.1)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(exp_dir / 'tb'))

    best_mIoU = 0.0
    epoch_global = 0

    print('[STAGE 1] Training head only')
    # Class names for confusion matrix
    class_names = ['ceiling', 'floor', 'wall', 'column', 'window', 'door', 'furniture', 'clutter']
    
    for e in range(args.stage1_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, args.num_classes, writer, epoch_global, weight=weight)
        val_loss, val_miou, val_f1, val_acc = eval_model(model, test_loader, criterion, args.num_classes, device, writer, epoch_global, weight=weight, class_names=class_names)
        print(f"Stage1 Epoch {e+1}/{args.stage1_epochs}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} mIoU={val_miou:.4f}")
        if val_miou > best_mIoU:
            best_mIoU = val_miou
            torch.save({'epoch': epoch_global, 'class_avg_iou': best_mIoU, 'model_state_dict': model.state_dict()}, str(ckpt_path / 'best_stage1.pth'))
        epoch_global += 1

    # stage2: unfreeze backbone (or progressively unfreeze if desired)
    print('[STAGE 2] Unfreezing backbone and fine-tuning')
    # Option: progressively unfreeze specific layers; here we unfreeze all
    for n, p in model.named_parameters():
        p.requires_grad = True

    # BN handling: optionally freeze running stats (keep eval) or allow updates
    if not args.bn_update:
        # keep running stats from pretrained: set BNs to eval (affine still trainable)
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.eval()
    else:
        # allow BN to update (recommended if dataset distribution differs)
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.train()

    # new param groups: head with lr_head, backbone with lr_backbone
    pg = collect_param_groups(model, head_names, args.lr_head, args.lr_backbone, args.weight_decay)
    optimizer = torch.optim.Adam(pg, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.stage2_epochs * len(train_loader), eta_min=args.lr_backbone*0.1)

    for e in range(args.stage2_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, args.num_classes, writer, epoch_global, weight=weight)
        val_loss, val_miou, val_f1, val_acc = eval_model(model, test_loader, criterion, args.num_classes, device, writer, epoch_global, weight=weight, class_names=class_names)
        print(f"Stage2 Epoch {e+1}/{args.stage2_epochs}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} mIoU={val_miou:.4f}")
        if val_miou > best_mIoU:
            best_mIoU = val_miou
            torch.save({'epoch': epoch_global, 'class_avg_iou': best_mIoU, 'model_state_dict': model.state_dict()}, str(ckpt_path / 'best_stage2.pth'))
        epoch_global += 1

    writer.close()
    print('[DONE] Best mIoU:', best_mIoU)


if __name__ == '__main__':
    main()
