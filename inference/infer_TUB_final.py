#!/usr/bin/env python3
"""
Enhanced inference + evaluation 
Your original code preserved + evaluation added
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))

from data_utils.S3DISBlockDataLoader_v3 import voxelization, fps_series_func
from ErrorMatrix import ConfusionMatrix

LABEL_COLORS = np.array([
    [128,128,128],[0,255,0],[255,0,0],[255,128,0],
    [255,255,0],[0,0,255],[128,0,128],[255,0,255]
], dtype=np.float32)/255.0

CLASSES = ['ceiling','floor','wall','column','window','door','furniture','clutter']
NUM_CLASSES = len(CLASSES)

# ---------------- NEW: IoU FUNCTION ----------------
def compute_iou(gt, pred):
    ious = []
    for c in range(NUM_CLASSES):
        tp = np.sum((gt==c)&(pred==c))
        fp = np.sum((gt!=c)&(pred==c))
        fn = np.sum((gt==c)&(pred!=c))
        denom = tp+fp+fn
        ious.append(tp/denom if denom>0 else 0)
    return np.array(ious)

def update_global_confusion(confusion, gt, pred):
    mask = (gt >= 0) & (gt < NUM_CLASSES) & (pred >= 0) & (pred < NUM_CLASSES)
    encoded = gt[mask] * NUM_CLASSES + pred[mask]
    confusion += np.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

def metrics_from_confusion(confusion):
    tp = np.diag(confusion).astype(np.float64)
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    oa = float(tp.sum() / confusion.sum()) if confusion.sum() else 0.0
    return {
        "mIoU": float(np.mean(iou)),
        "OA": oa,
        "IoU": iou,
        "gt_points": gt_count.astype(np.int64),
        "pred_points": pred_count.astype(np.int64),
        "correct_points": tp.astype(np.int64),
        "union_points": union.astype(np.int64),
        "total_points": int(confusion.sum()),
    }

# ---------------- ORIGINAL FUNCTIONS (UNCHANGED) ----------------
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

class SingleFileDataset(torch.utils.data.Dataset):
    def __init__(self, npy_file, fps_n_list=[512,128,32], label_number=8, npoints=8192, voxel_size=0.01):
        self.sample_points = []
        self.sample_labels = []
        self.fps_index_array_list = []
        self.series_idx_arrays_list = []

        room_data = np.load(npy_file)
        for i in range(room_data.shape[0]):
            points, labels = room_data[i][:, :-1], room_data[i][:, -1]
            labels = labels - 1 if labels.min() == 1 else labels

            coor_min = np.amin(points[:, :3], axis=0)
            points[:, 2] -= coor_min[2]

            idx = np.arange(points.shape[0])
            np.random.shuffle(idx)
            points, labels = points[idx], labels[idx]

            coords = points.copy()
            coords, voxel_indices, _, _ = voxelization(coords[:, :3], voxel_size)
            fps_index_array, series_idx_arrays = fps_series_func(points, voxel_indices, fps_n_list)

            self.sample_points.append(points)
            self.sample_labels.append(labels)
            self.fps_index_array_list.append(fps_index_array)
            self.series_idx_arrays_list.append(series_idx_arrays)

    def __len__(self): return len(self.sample_points)

    def __getitem__(self, idx):
        return (self.sample_points[idx],
                self.sample_labels[idx],
                self.fps_index_array_list[idx],
                self.series_idx_arrays_list[idx])

# ---------------- PLY ----------------
def build_ply_with_scalars(coords, rgb, gt_labels, pred_labels, out_ply_path):
    vertices = np.zeros(coords.shape[0], dtype=[
        ('x','f4'),('y','f4'),('z','f4'),
        ('red','f4'),('green','f4'),('blue','f4'),
        ('GT_label','i4'),('Pred_label','i4'),('Correct','f4')
    ])
    vertices['x'],vertices['y'],vertices['z'] = coords.T
    vertices['red'],vertices['green'],vertices['blue'] = rgb.T
    vertices['GT_label'] = gt_labels
    vertices['Pred_label'] = pred_labels
    vertices['Correct'] = (gt_labels==pred_labels).astype(np.float32)

    PlyData([PlyElement.describe(vertices,'vertex')], text=True).write(out_ply_path)

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mamba_msg_s3dis')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', default='./predictions_stitched')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--fps_n_list', nargs='+', type=int, default=[512,128,32])
    parser.add_argument('--npoint', type=int, default=8192)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--voxel_size', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES, args.fps_n_list).to(device)
    classifier = DataParallel(classifier)
    load_checkpoint(args.checkpoint, classifier, device)
    classifier.eval()

    npy_files = sorted(glob.glob(os.path.join(args.data_root, "*.npy")))

    room_metrics = []
    confusion = ConfusionMatrix(num_classes=NUM_CLASSES, labels=CLASSES)
    global_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for npy_path in npy_files:
        print(f"[INFO] Processing {npy_path}")

        test_dataset = SingleFileDataset(npy_path,
                                        args.fps_n_list,
                                        NUM_CLASSES,
                                        args.npoint,
                                        args.voxel_size)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size)

        all_xyz, all_rgb, all_gt, all_pred = [],[],[],[]

        with torch.no_grad():
            for points, target, fps_index_array, series_idx_arrays in test_loader:
                points = points.float().to(device)
                target = target.long().to(device)
                fps_index_array = fps_index_array.long().to(device)
                series_idx_arrays = series_idx_arrays.long().to(device)

                seg_pred = classifier(points.transpose(2,1),
                                      fps_index_array,
                                      series_idx_arrays)

                preds = seg_pred.argmax(dim=-1).cpu().numpy()

                pts_np = points.cpu().numpy()
                gt_np = target.cpu().numpy()

                for b in range(pts_np.shape[0]):
                    block_pts = pts_np[b]
                    xyz = block_pts[:,6:9]
                    rgb = block_pts[:,3:6]/255.0

                    all_xyz.append(xyz)
                    all_rgb.append(rgb)
                    all_gt.append(gt_np[b])
                    all_pred.append(preds[b])

                    confusion.update(preds[b].reshape(-1,1),
                                     gt_np[b].reshape(-1,1))

        coords = np.concatenate(all_xyz)
        rgb = np.concatenate(all_rgb)
        gts = np.concatenate(all_gt)
        preds = np.concatenate(all_pred)
        update_global_confusion(global_confusion, gts.astype(np.int64), preds.astype(np.int64))

        # -------- NEW: METRICS --------
        iou = compute_iou(gts, preds)
        miou = np.mean(iou)
        acc = np.mean(gts == preds)

        row = {'room': os.path.basename(npy_path),
               'mIoU': miou,
               'accuracy': acc}

        for i,c in enumerate(CLASSES):
            row[c] = iou[i]

        room_metrics.append(row)

        print(f"[RESULT] {row}")

        # Save PLY
        out_ply = os.path.join(args.output_dir,
                               os.path.basename(npy_path)+".ply")
        build_ply_with_scalars(coords, rgb, gts, preds, out_ply)

    # -------- NEW: SAVE CSV --------
    df = pd.DataFrame(room_metrics)
    df.to_csv(os.path.join(args.output_dir,"room_metrics.csv"),index=False)

    summary = df.mean(numeric_only=True)
    summary.to_csv(os.path.join(args.output_dir,"apartment_summary.csv"))

    # -------- GLOBAL POINT-LEVEL METRICS --------
    # apartment_summary.csv above is the historical room-wise mean.
    # These files are computed from one confusion matrix over all points.
    global_metrics = metrics_from_confusion(global_confusion)
    global_summary = {
        "metric_type": "global_point_level",
        "num_rooms": len(npy_files),
        "num_points": global_metrics["total_points"],
        "mIoU": global_metrics["mIoU"],
        "OA": global_metrics["OA"],
    }
    for i, cls in enumerate(CLASSES):
        global_summary[f"IoU_{cls}"] = float(global_metrics["IoU"][i])

    pd.DataFrame([global_summary]).to_csv(
        os.path.join(args.output_dir, "global_point_summary.csv"),
        index=False,
    )

    pd.Series(
        {
            "mIoU": global_metrics["mIoU"],
            "accuracy": global_metrics["OA"],
            **{cls: float(global_metrics["IoU"][i]) for i, cls in enumerate(CLASSES)},
        }
    ).to_csv(os.path.join(args.output_dir, "apartment_summary_global.csv"))

    pd.DataFrame(
        [
            {
                "class": cls,
                "IoU": float(global_metrics["IoU"][i]),
                "gt_points": int(global_metrics["gt_points"][i]),
                "pred_points": int(global_metrics["pred_points"][i]),
                "correct_points": int(global_metrics["correct_points"][i]),
                "union_points": int(global_metrics["union_points"][i]),
            }
            for i, cls in enumerate(CLASSES)
        ]
    ).to_csv(os.path.join(args.output_dir, "global_point_class_iou.csv"), index=False)

    pd.DataFrame(global_confusion, index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(args.output_dir, "global_point_confusion_matrix.csv"),
        index_label="GT/PRED",
    )

    print("\n========== GLOBAL POINT-LEVEL SUMMARY ==========")
    print(f"Global mIoU: {global_metrics['mIoU']:.6f}")
    print(f"Global OA: {global_metrics['OA']:.6f}")

    # -------- NEW: PLOTS --------
    plt.figure()
    summary[CLASSES].plot(kind='bar')
    plt.title("Class IoU")
    plt.savefig(os.path.join(args.output_dir,"iou_bar.png"))

    plt.figure()
    plt.plot(df['mIoU'].values)
    plt.title("mIoU per room")
    plt.savefig(os.path.join(args.output_dir,"miou_rooms.png"))

    # Confusion matrix heatmap
    cm = confusion.matrix
    plt.figure()
    sns.heatmap(cm.astype(int), annot=True, fmt='d',
                xticklabels=CLASSES,
                yticklabels=CLASSES)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(args.output_dir,"confusion_matrix.png"))

    print("\n========== CONFUSION MATRIX SUMMARY ==========")
    try:
        ave_F1_score, miou_conf, acc_conf = confusion.summary()
        print(f"Avg F1: {ave_F1_score}, mIoU: {miou_conf}, Acc: {acc_conf}")
    except Exception as e:
        print(f"[WARN] Could not compute confusion summary: {e}")

if __name__ == "__main__":
    main()
