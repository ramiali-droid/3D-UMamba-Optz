#!/usr/bin/env python3
"""
infer_S3DIS_final.py
Your original working code + RGB + scalar fields using plyfile
"""
import os
import argparse
import importlib
import numpy as np
import torch
from pathlib import Path
from torch.nn import DataParallel
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import glob
import sys

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))

from data_utils.S3DISBlockDataLoader_v3 import voxelization, fps_series_func
from ErrorMatrix import ConfusionMatrix

# S3DIS colors (normalized for plyfile/Open3D)
LABEL_COLORS = np.array([
    [128,128,128], # ceiling
    [0,255,0],     # floor
    [255,0,0],     # wall
    [255,128,0],   # column
    [255,255,0],   # window
    [0,0,255],     # door
    [128,0,128],   # furniture
    [255,0,255]    # clutter
], dtype=np.float32) / 255.0

CLASSES = ['ceiling', 'floor', 'wall', 'column', 'window', 'door', 'furniture', 'clutter']
NUM_CLASSES = len(CLASSES)

def strip_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[len('module.'):]] = v
        else:
            new_state[k] = v
    return new_state

def load_checkpoint(checkpoint_path, model, device):
    ck = torch.load(checkpoint_path, map_location=device)
    state = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
    example_key = next(iter(state.keys()))
    if example_key.startswith('module.') and not next(iter(model.state_dict().keys())).startswith('module.'):
        state = strip_module_prefix(state)
    model.load_state_dict(state)
    return ck

def aggregate_votes(coords, preds, gts=None, voxel_size=0.01):
    if voxel_size <= 0:
        return coords, preds, gts if gts is not None else None
    idxs = np.floor(coords / voxel_size).astype(np.int64)
    keymap = {}
    for i, k in enumerate(map(tuple, idxs)):
        keymap.setdefault(k, []).append(i)
    agg_coords, agg_preds, agg_gts = [], [], []
    for k, idlist in keymap.items():
        pts = coords[idlist]
        centroid = np.mean(pts, axis=0)
        agg_coords.append(centroid)
        labels = preds[idlist]
        vals, counts = np.unique(labels, return_counts=True)
        agg_preds.append(int(vals[np.argmax(counts)]))
        if gts is not None:
            gvals, gcounts = np.unique(gts[idlist], return_counts=True)
            agg_gts.append(int(gvals[np.argmax(gcounts)]))
        else:
            agg_gts.append(-1)
    return np.array(agg_coords), np.array(agg_preds), np.array(agg_gts)

class SingleFileDataset(torch.utils.data.Dataset):
    def __init__(self, npy_file, fps_n_list=[512,128,32], label_number=8, npoints=8192, voxel_size=0.01):
        self.fps_n_list = fps_n_list
        self.npoints = npoints
        self.sample_points = []
        self.sample_labels = []
        self.fps_index_array_list = []
        self.series_idx_arrays_list = []

        room_data = np.load(npy_file) # shape: (num_samples, N, features)
        for i in range(room_data.shape[0]):
            points, labels = room_data[i][:, :-1], room_data[i][:, -1]
            labels = labels - 1 if labels.min() == 1 else labels # adjust if labels start from 1
            coor_min = np.amin(points[:, :3], axis=0)
            points[:, 2] = points[:, 2] - coor_min[2]
            # Shuffle points
            array = np.arange(points.shape[0])
            np.random.shuffle(array)
            points = points[array]
            labels = labels[array]
            # Apply voxelization
            coords = points.copy()
            coords, voxel_indices, voxel_total, voxel_valid = voxelization(coords[:, :3], voxel_size)
            fps_index_array, series_idx_arrays = fps_series_func(points, voxel_indices, fps_n_list)
            self.sample_points.append(points)
            self.sample_labels.append(labels)
            self.fps_index_array_list.append(fps_index_array)
            self.series_idx_arrays_list.append(series_idx_arrays)
        print("Unique labels:", np.unique(labels))

    def __len__(self):
        return len(self.sample_points)

    def __getitem__(self, idx):
        return (self.sample_points[idx],
                self.sample_labels[idx],
                self.fps_index_array_list[idx],
                self.series_idx_arrays_list[idx])
"""
def build_ply_with_scalars(coords, rgb, gt_labels, pred_labels, out_ply_path):
    vertices = np.zeros(coords.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'),
        ('GT_label', 'i4'),
        ('Pred_label', 'i4'),
        ('Correct', 'f4')
    ])

    vertices['x'] = coords[:, 0]
    vertices['y'] = coords[:, 1]
    vertices['z'] = coords[:, 2]
    vertices['red'] = rgb[:, 0]
    vertices['green'] = rgb[:, 1]
    vertices['blue'] = rgb[:, 2]
    vertices['GT_label'] = gt_labels
    vertices['Pred_label'] = pred_labels
    vertices['Correct'] = (gt_labels == pred_labels).astype(np.float32)

    ply_element = PlyElement.describe(vertices, 'vertex')
    PlyData([ply_element], text=True).write(out_ply_path)
    print(f"[INFO] Saved PLY with RGB + scalars: {out_ply_path}")
"""

def main():
    parser = argparse.ArgumentParser(description="Inference + save PLY per file (3D-UMamba style)")
    parser.add_argument('--model', type=str, default='mamba_msg_s3dis', help='model module name')
    parser.add_argument('--checkpoint', required=True, help='path to best_model.pth')
    parser.add_argument('--data_root', required=True, help='folder containing .npy block files')
    parser.add_argument('--output_dir', default='./predictions_stitched', help='where to save stitched .ply')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--fps_n_list', nargs='+', type=int, default=[512,128,32])
    parser.add_argument('--npoint', type=int, default=8192)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    parser.add_argument('--voxel_size', type=float, default=0.01)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu')
    print("[INFO] Device:", device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Import model module
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES, args.fps_n_list).to(device)
    if torch.cuda.device_count() > 1:
        classifier = DataParallel(classifier)
    classifier.eval()

    # Load checkpoint
    print("[INFO] Loading checkpoint:", args.checkpoint)
    load_checkpoint(args.checkpoint, classifier, device)
    print("[INFO] Model loaded and set to eval.")

    # Get all test .npy files
    npy_files = glob.glob(os.path.join(args.data_root, "*2_room_fil_sub.npy"))
    print(f"[INFO] Found {len(npy_files)} test files.")

    for npy_path in npy_files:
        print(f"[INFO] Processing file: {npy_path}")
        test_dataset = SingleFileDataset(npy_file=npy_path,
                                         fps_n_list=args.fps_n_list,
                                         label_number=NUM_CLASSES,
                                         npoints=args.npoint,
                                         voxel_size=args.voxel_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.num_workers)

        all_xyz = []
        all_rgb = []
        all_pred = []
        all_gt = []

        total_correct = 0
        total_seen = 0
        total_seen_class = [0] * NUM_CLASSES
        total_correct_class = [0] * NUM_CLASSES
        total_iou_deno_class = [0] * NUM_CLASSES

        confusion = ConfusionMatrix(num_classes=NUM_CLASSES, labels=CLASSES)

        with torch.no_grad():
            for points, target, fps_index_array, series_idx_arrays in tqdm(test_loader, desc='Inference'):
                points = points.float().to(device)
                target = target.long().to(device)
                fps_index_array = fps_index_array.long().to(device)
                series_idx_arrays = series_idx_arrays.long().to(device)

                pts_trans = points.transpose(2, 1)
                seg_pred = classifier(pts_trans, fps_index_array, series_idx_arrays)

                pred_np = seg_pred.contiguous().cpu().numpy()
                
                # Convert to per-point labels
                if pred_np.ndim == 3:
                    if pred_np.shape[2] == NUM_CLASSES:
                        preds = np.argmax(pred_np, axis=2)
                    elif pred_np.shape[1] == NUM_CLASSES:
                        preds = np.argmax(pred_np, axis=1)
                    else:
                        preds = np.argmax(pred_np, axis=-1)
                else:
                    raise RuntimeError("Unexpected model output shape: " + str(pred_np.shape))

                points_np = points.cpu().numpy()
                B = points_np.shape[0]
                for b in range(B):
                    block_pts = points_np[b]
                    global_xyz = block_pts[:, 6:9].astype(np.float64)  # ax ay az
                    rgb = block_pts[:, 3:6] / 255.0 if block_pts[:, 3:6].max() > 1 else block_pts[:, 3:6]
                    gt_labels = target.cpu().numpy()[b]
                    pred_labels = preds[b]

                    all_xyz.append(global_xyz)
                    all_rgb.append(rgb)
                    all_gt.append(gt_labels)
                    all_pred.append(pred_labels)

                    correct = np.sum(pred_labels == gt_labels)
                    total_correct += correct
                    total_seen += len(gt_labels)

                    for l in range(NUM_CLASSES):
                        total_seen_class[l] += np.sum(gt_labels == l)
                        total_correct_class[l] += np.sum((pred_labels == l) & (gt_labels == l))
                        total_iou_deno_class[l] += np.sum((pred_labels == l) | (gt_labels == l))

                    confusion.update(pred_labels.reshape(-1, 1), gt_labels.reshape(-1, 1))

        coords = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)
        preds = np.concatenate(all_pred, axis=0).astype(int)
        gts = np.concatenate(all_gt, axis=0).astype(int)

        # Aggregate if voxel size > 0
        # if args.voxel_size > 0:
        #     coords, preds, gts = aggregate_votes(coords, preds, gts, voxel_size=args.voxel_size)

        # Accuracy metrics
        valid_mask = (gts >= 0)
        if np.any(valid_mask):
            overall_acc = float(np.mean(preds[valid_mask] == gts[valid_mask]))
            print(f"[RESULT] Overall point accuracy for {os.path.basename(npy_path)}: {overall_acc}")

        if confusion is not None:
            try:
                ave_F1_score, miou, acc = confusion.summary()
                print(f"[INFO] Confusion summary for {os.path.basename(npy_path)}: ave_F1_score={ave_F1_score}, miou={miou}, acc={acc}")
            except Exception as e:
                print(f"[WARN] Could not compute confusion summary: {e}")
    
        # Save .ply with RGB + scalars
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        out_ply = os.path.join(args.output_dir, f"{base_name}.ply")
        build_ply_with_scalars(coords, rgb, gts, preds, out_ply)

def build_ply_with_scalars(coords, rgb, gt_labels, pred_labels, out_ply_path):
    vertices = np.zeros(coords.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'),
        ('GT_label', 'i4'),
        ('Pred_label', 'i4'),
        ('Correct', 'f4')
    ])

    vertices['x'] = coords[:, 0]
    vertices['y'] = coords[:, 1]
    vertices['z'] = coords[:, 2]
    vertices['red'] = rgb[:, 0]
    vertices['green'] = rgb[:, 1]
    vertices['blue'] = rgb[:, 2]
    vertices['GT_label'] = gt_labels
    vertices['Pred_label'] = pred_labels
    vertices['Correct'] = (gt_labels == pred_labels).astype(np.float32)

    ply_element = PlyElement.describe(vertices, 'vertex')
    PlyData([ply_element], text=True).write(out_ply_path)
    print(f"[INFO] Saved PLY with RGB + scalars: {out_ply_path}")
    

if __name__ == "__main__":
    main()