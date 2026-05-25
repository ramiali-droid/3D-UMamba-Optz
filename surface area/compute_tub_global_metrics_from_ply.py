#!/usr/bin/env python3
"""Compute global point-level TU-CSE metrics from saved prediction PLY files."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from plyfile import PlyData


CLASSES = ["ceiling", "floor", "wall", "column", "window", "door", "furniture", "clutter"]
RESULT_FOLDER_NAMES = ["results", "results_after_FT", "results_from_scratch", "results_after_finetune"]
DEFAULT_ROOT = Path("/home/ramiali/3dumamba/data/TUB/combined/crt")
BLOCK_DIR = "filtered/subsampled_0.010/Block_s2_min_final_8192_norm_enhance_rad_0.5"


def parse_args():
    parser = argparse.ArgumentParser(description="Save global point-level metrics into TU-CSE result folders.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apartments", nargs="+", default=["D2", "D4"])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing global metric CSVs.")
    return parser.parse_args()


def update_confusion(confusion, pred, gt):
    num_classes = len(CLASSES)
    mask = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    encoded = gt[mask] * num_classes + pred[mask]
    confusion += np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion):
    tp = np.diag(confusion).astype(np.float64)
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    oa = float(tp.sum() / confusion.sum()) if confusion.sum() else 0.0
    return tp.astype(np.int64), gt_count.astype(np.int64), pred_count.astype(np.int64), union.astype(np.int64), iou, float(np.mean(iou)), oa


def safe_write_check(paths, overwrite):
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing global metric files. "
            "Use --overwrite if you intentionally want to regenerate them:\n" + "\n".join(existing)
        )


def write_summary(path, row):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_class_iou(path, gt_count, pred_count, correct_count, union_count, iou):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class", "IoU", "gt_points", "pred_points", "correct_points", "union_points"],
        )
        writer.writeheader()
        for idx, cls in enumerate(CLASSES):
            writer.writerow(
                {
                    "class": cls,
                    "IoU": float(iou[idx]),
                    "gt_points": int(gt_count[idx]),
                    "pred_points": int(pred_count[idx]),
                    "correct_points": int(correct_count[idx]),
                    "union_points": int(union_count[idx]),
                }
            )


def write_confusion(path, confusion):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["GT/PRED", *CLASSES])
        for cls, row in zip(CLASSES, confusion):
            writer.writerow([cls, *row.tolist()])


def result_dirs(root, apartments):
    dirs = []
    for apartment in apartments:
        block_root = root / f"{apartment}_rooms_crt" / BLOCK_DIR
        for name in RESULT_FOLDER_NAMES:
            result_dir = block_root / name
            if result_dir.exists():
                dirs.append((apartment, name, result_dir))
    return dirs


def compute_for_dir(apartment, result_name, result_dir, overwrite):
    candidate_ply_files = sorted(result_dir.glob("*.ply"))
    ply_files = []
    skipped_ply_files = []
    for ply_file in candidate_ply_files:
        vertex = PlyData.read(str(ply_file))["vertex"].data
        names = vertex.dtype.names
        if "GT_label" in names and "Pred_label" in names:
            ply_files.append(ply_file)
        else:
            skipped_ply_files.append(ply_file.name)

    if not ply_files:
        return None

    output_paths = [
        result_dir / "global_point_summary.csv",
        result_dir / "global_point_class_iou.csv",
        result_dir / "global_point_confusion_matrix.csv",
        result_dir / "global_point_metrics_metadata.json",
    ]
    safe_write_check(output_paths, overwrite)

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    room_point_counts = []
    for ply_file in ply_files:
        vertex = PlyData.read(str(ply_file))["vertex"].data
        gt = np.asarray(vertex["GT_label"], dtype=np.int64)
        pred = np.asarray(vertex["Pred_label"], dtype=np.int64)
        update_confusion(confusion, pred, gt)
        room_point_counts.append({"file": ply_file.name, "points": int(len(gt))})

    correct_count, gt_count, pred_count, union_count, iou, miou, oa = metrics_from_confusion(confusion)
    summary = {
        "apartment": apartment,
        "result_folder": result_name,
        "metric_type": "global_point_level",
        "num_ply_rooms": len(ply_files),
        "num_points": int(confusion.sum()),
        "mIoU": miou,
        "OA": oa,
    }
    for idx, cls in enumerate(CLASSES):
        summary[f"IoU_{cls}"] = float(iou[idx])

    write_summary(output_paths[0], summary)
    write_class_iou(output_paths[1], gt_count, pred_count, correct_count, union_count, iou)
    write_confusion(output_paths[2], confusion)
    with open(output_paths[3], "w") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source": "computed from saved PLY GT_label and Pred_label fields",
                "classes": CLASSES,
                "room_point_counts": room_point_counts,
                "skipped_ply_files_without_gt_pred": skipped_ply_files,
                "outputs": [str(path) for path in output_paths[:3]],
            },
            f,
            indent=2,
        )
    return summary


def main():
    args = parse_args()
    summaries = []
    for apartment, result_name, directory in result_dirs(args.root, args.apartments):
        summary = compute_for_dir(apartment, result_name, directory, args.overwrite)
        if summary:
            summaries.append(summary)

    for summary in summaries:
        print(
            f"{summary['apartment']} {summary['result_folder']}: "
            f"rooms={summary['num_ply_rooms']} points={summary['num_points']} "
            f"mIoU={summary['mIoU']:.6f} OA={summary['OA']:.6f}"
        )


if __name__ == "__main__":
    main()
