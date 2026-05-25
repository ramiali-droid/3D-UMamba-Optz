#!/usr/bin/env python3
"""
infer_DALES_final.py

Load trained best_model.pth (3D-UMamba style), run inference on DALES block .npy files individually,
aggregate points optionally, save each block as a separate .ply file, and print accuracy.

Usage example:
python infer_stitch_save_ply_per_file.py \
  --model mamba_msg \
  --checkpoint ./log/dales_seg/checkpoints/best_model.pth \
  --data_root /path/to/DALESObjects/test \
  --output_dir ./predictions_stitched \
  --voxel_size 0.02 \
  --device cuda
"""

import os
import argparse
import importlib
import numpy as np
import torch
from pathlib import Path
from torch.nn import DataParallel
from tqdm import tqdm
import open3d as o3d
import sys
import glob

# Make sure repo models and data_utils are importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))

from data_utils.DALESBlockDataLoader_v3 import voxelization, fps_series_func
from ErrorMatrix import ConfusionMatrix

# --- configuration ---
LABEL_COLORS = np.array([
    [128,128,128], # ground
    [0,255,0],     # vegetation
    [255,0,0],     # cars
    [255,128,0],   # trucks
    [255,255,0],   # powerlines
    [0,0,255],     # fences
    [128,0,128],   # poles
    [255,0,255],   # buildings
], dtype=np.uint8)

CLASSES = ['ground','vegetation','cars','trucks','powerlines','fences','poles','buildings']
NUM_CLASSES = len(CLASSES)

# ------------------- Utility functions -------------------

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

def aggregate_votes(coords, preds, gts=None, voxel_size=0.0):
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

def build_pcd_and_save(xyz, labels, out_ply_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    labs = np.array(labels).astype(int)
    colors = LABEL_COLORS[labs % LABEL_COLORS.shape[0]] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_ply_path, pcd)
    return pcd

# ------------------- Dataset for single .npy file -------------------

class SingleFileDataset(torch.utils.data.Dataset):
    def __init__(self, npy_file, fps_n_list=[512,128,32], label_number=8, npoints=8192, voxel_size=0.4):
        self.fps_n_list = fps_n_list
        self.npoints = npoints
        self.sample_points = []
        self.sample_labels = []
        self.fps_index_array_list = []
        self.series_idx_arrays_list = []

        room_data = np.load(npy_file)  # shape: (num_samples, N, features)

        for i in range(room_data.shape[0]):
            points, labels = room_data[i][:, :-1], room_data[i][:, -1]
            labels = labels - 1

            # Shuffle points
            array = np.arange(points.shape[0])
            np.random.shuffle(array)
            points = points[array]
            labels = labels[array]

            # Apply voxelization
            points, voxel_indices, voxel_total, voxel_valid = voxelization(points, voxel_size)
            fps_index_array, series_idx_arrays = fps_series_func(points, voxel_indices, fps_n_list)

            self.sample_points.append(points)
            self.sample_labels.append(labels)
            self.fps_index_array_list.append(fps_index_array)
            self.series_idx_arrays_list.append(series_idx_arrays)

    def __len__(self):
        return len(self.sample_points)

    def __getitem__(self, idx):
        return (self.sample_points[idx],
                self.sample_labels[idx],
                self.fps_index_array_list[idx],
                self.series_idx_arrays_list[idx])

# ------------------- Main function -------------------

def main():
    parser = argparse.ArgumentParser(description="Inference + save PLY per file (3D-UMamba style)")
    parser.add_argument('--model', type=str, default='mamba_msg', help='model module name')
    parser.add_argument('--checkpoint', required=True, help='path to best_model.pth')
    parser.add_argument('--data_root', required=True, help='folder containing .npy block files')
    parser.add_argument('--output_dir', default='./predictions_stitched', help='where to save stitched .ply')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--fps_n_list', nargs='+', type=int, default=[512,128,32])
    parser.add_argument('--npoint', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    parser.add_argument('--voxel_size', type=float, default=0.02)
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
    classifier.apply(lambda m: None)

    # Load checkpoint
    print("[INFO] Loading checkpoint:", args.checkpoint)
    load_checkpoint(args.checkpoint, classifier, device)
    classifier.eval()
    print("[INFO] Model loaded and set to eval.")

    # Get all test .npy files
    npy_files = glob.glob(os.path.join(args.data_root, "*test*.npy"))
    print(f"[INFO] Found {len(npy_files)} test files.")

    for npy_path in npy_files:
        print(f"[INFO] Processing file: {npy_path}")
        test_dataset = SingleFileDataset(npy_file=npy_path,
                                         fps_n_list=args.fps_n_list,
                                         label_number=NUM_CLASSES,
                                         npoints=args.npoint)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.num_workers)

        all_xyz, all_pred, all_gt = [], [], []

        try:
            cm = ConfusionMatrix(num_classes=NUM_CLASSES, labels=CLASSES)
        except Exception:
            cm = None

        with torch.no_grad():
            for points, target, fps_index_array, series_idx_arrays in tqdm(test_loader, desc='Inference'):
                points = points.float().to(device)
                fps_index_array = fps_index_array.long().to(device)
                series_idx_arrays = series_idx_arrays.long().to(device)
                target = target.long().to(device)

                pts_trans = points.transpose(2,1)
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
                    if block_pts.shape[1] < 7:
                        raise RuntimeError("Block points have fewer than 7 features.")
                    global_xyz = block_pts[:, 4:7].astype(np.float64)
                    gt_labels = target.cpu().numpy()[b]
                    pred_labels = preds[b]

                    all_xyz.append(global_xyz)
                    all_gt.append(gt_labels)
                    all_pred.append(pred_labels)

                    if cm is not None:
                        cm.update(pred_labels.reshape(-1,1), gt_labels.reshape(-1,1))

        coords = np.concatenate(all_xyz, axis=0)
        preds = np.concatenate(all_pred, axis=0).astype(int)
        gts = np.concatenate(all_gt, axis=0).astype(int)

        # Aggregate if voxel size > 0
        if args.voxel_size > 0:
            coords, preds, gts = aggregate_votes(coords, preds, gts, voxel_size=args.voxel_size)

        # Save .ply per file
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        out_ply = os.path.join(args.output_dir, f"{base_name}.ply")
        pcd = build_pcd_and_save(coords, preds, out_ply)
        print(f"[INFO] Saved PLY: {out_ply}")

        # Accuracy metrics
        valid_mask = (gts >= 0)
        if np.any(valid_mask):
            overall_acc = float(np.mean(preds[valid_mask] == gts[valid_mask]))
            print(f"[RESULT] Overall point accuracy for {base_name}: {overall_acc}")

        if cm is not None:
            try:
                ave_F1_score, miou, acc = cm.summary()
                print(f"[INFO] Confusion summary for {base_name}: ave_F1_score={ave_F1_score}, miou={miou}, acc={acc}")
            except Exception as e:
                print(f"[WARN] Could not compute confusion summary for {base_name}: {e}")

        if args.visualize:
            o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
