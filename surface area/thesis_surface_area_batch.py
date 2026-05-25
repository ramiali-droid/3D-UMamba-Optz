"""
Batch surface-area analysis for comparison tables.

This script reads the existing TUB room-level PLY outputs from:
  - results          : pretrained S3DIS model
  - results_after_FT : pretrained S3DIS model fine-tuned on TUB-CSE

It computes prediction-based and GT-based ceiling/floor/wall areas for each
room, aggregates them per apartment, merges existing mIoU/IoU CSV metrics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from room_surface_area_estimation import (
    CLASS_NAMES,
    SURFACE_LABELS,
    estimate_all_surfaces,
    load_ply_label_pair,
)


TEST_APARTMENTS = {"D2", "D4"}
BASELINE_NAME = "S3DIS_pretrained"
FT_NAME = "S3DIS_pretrained_FT_TUB_CSE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Batch surface-area analysis")
    parser.add_argument(
        "--tub-root",
        type=Path,
        default=Path("/home/ramiali/3dumamba/data/TUB"),
        help="Root TUB data folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ramiali/3dumamba/data/TUB/THESIS SURFACE AREA RESULTS"),
        help="Output folder for tables and report.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate only the LaTeX report from existing CSV outputs.",
    )

    # Surface-area parameters mirror room_size_test.py.
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--alpha-radius", type=float, default=None)
    parser.add_argument("--floor-ceiling-alpha-scale", type=float, default=6.0)
    parser.add_argument("--floor-plane-tolerance", type=float, default=0.08)
    parser.add_argument("--ceiling-plane-tolerance", type=float, default=0.10)
    parser.add_argument("--ceiling-min-height-above-floor", type=float, default=1.8)
    parser.add_argument("--floor-max-height-above-lowest", type=float, default=0.35)
    parser.add_argument("--floor-max-slope-deg", type=float, default=12.0)
    parser.add_argument("--ceiling-max-slope-deg", type=float, default=45.0)
    parser.add_argument("--max-floor-planes", type=int, default=2)
    parser.add_argument("--max-ceiling-planes", type=int, default=3)
    parser.add_argument("--min-structural-plane-points", type=int, default=250)
    parser.add_argument("--plane-ransac-iterations", type=int, default=900)
    parser.add_argument("--wall-distance-threshold", type=float, default=0.06)
    parser.add_argument("--min-wall-plane-points", type=int, default=300)
    parser.add_argument("--max-wall-planes", type=int, default=12)
    parser.add_argument("--ransac-iterations", type=int, default=600)
    return parser.parse_args()


def discover_result_dirs(tub_root: Path) -> Dict[str, Dict[str, Path]]:
    """Return apartment -> model -> result directory."""
    result_dirs: Dict[str, Dict[str, Path]] = {}
    base = tub_root / "combined" / "crt"
    for apartment_dir in sorted(base.glob("D*_rooms_crt")):
        apartment = apartment_dir.name.split("_", 1)[0]
        block_dirs = list(apartment_dir.glob("filtered/subsampled_0.010/Block_s2_min_final_8192_norm_enhance_rad_0.5"))
        if not block_dirs:
            continue
        block_dir = block_dirs[0]
        model_dirs = {}
        if (block_dir / "results").is_dir():
            model_dirs[BASELINE_NAME] = block_dir / "results"
        if (block_dir / "results_after_FT").is_dir():
            model_dirs[FT_NAME] = block_dir / "results_after_FT"
        if model_dirs:
            result_dirs[apartment] = model_dirs
    return result_dirs


def estimate_room_areas(ply_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    """
    Compute pred and GT surface areas for one room PLY.

    Prediction areas are the measurement output. GT areas are included only for
    validation/comparison tables.
    """
    xyz, gt_labels, pred_labels = load_ply_label_pair(ply_path)
    pred_estimates, _ = estimate_all_surfaces(xyz, pred_labels, args)
    gt_estimates, _ = estimate_all_surfaces(xyz, gt_labels, args)

    row: Dict[str, object] = {
        "room_ply": ply_path.name,
        "room_metric_name": ply_path.name.replace(".ply", ""),
    }
    for estimate in pred_estimates:
        row[f"pred_{estimate.name}_area_m2"] = estimate.area_m2
        row[f"pred_{estimate.name}_points"] = estimate.point_count
        row[f"pred_{estimate.name}_method"] = estimate.method
        row[f"pred_{estimate.name}_warning"] = estimate.warning
    for estimate in gt_estimates:
        row[f"gt_{estimate.name}_area_m2"] = estimate.area_m2
        row[f"gt_{estimate.name}_points"] = estimate.point_count
        row[f"gt_{estimate.name}_method"] = estimate.method
        row[f"gt_{estimate.name}_warning"] = estimate.warning
    return row


def read_apartment_summary(result_dir: Path) -> Dict[str, float]:
    """Read apartment_summary.csv as metric -> value."""
    path = result_dir / "apartment_summary.csv"
    if not path.exists():
        return {}
    metrics: Dict[str, float] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2 or row[0] == "":
                continue
            try:
                metrics[row[0]] = float(row[1])
            except ValueError:
                continue
    return metrics


def read_room_metrics(result_dir: Path) -> Dict[str, Dict[str, float]]:
    """Read room_metrics.csv as room -> metric -> value."""
    path = result_dir / "room_metrics.csv"
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            room = row.get("room", "")
            out[room] = {}
            for key, value in row.items():
                if key == "room" or value in (None, ""):
                    continue
                try:
                    out[room][key] = float(value)
                except ValueError:
                    pass
    return out


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> List[Dict[str, object]]:
    """Read generated CSV rows and convert numeric fields where possible."""
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted: Dict[str, object] = {}
            for key, value in row.items():
                if value == "":
                    converted[key] = ""
                    continue
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            rows.append(converted)
    return rows


def aggregate_apartment_areas(room_rows: Iterable[Dict[str, object]]) -> Dict[str, float]:
    """Sum room surface areas into one apartment-level row."""
    totals: Dict[str, float] = {}
    for prefix in ("pred", "gt"):
        for surface in SURFACE_LABELS:
            key = f"{prefix}_{surface}_area_m2"
            totals[key] = sum(float(row.get(key, 0.0) or 0.0) for row in room_rows)
    return totals


def build_tables(args: argparse.Namespace) -> Dict[str, List[Dict[str, object]]]:
    result_dirs = discover_result_dirs(args.tub_root)
    all_room_rows: List[Dict[str, object]] = []
    apartment_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []

    for apartment, model_dirs in sorted(result_dirs.items()):
        for model_name, result_dir in sorted(model_dirs.items()):
            ply_paths = sorted(result_dir.glob("*.ply"))
            room_metrics = read_room_metrics(result_dir)
            model_room_rows: List[Dict[str, object]] = []

            for ply_path in ply_paths:
                room_row = estimate_room_areas(ply_path, args)
                room_metric_name = room_row["room_metric_name"]
                metrics = room_metrics.get(str(room_metric_name), {})
                room_row.update(
                    {
                        "apartment": apartment,
                        "model": model_name,
                        "source_folder": str(result_dir),
                        "mIoU": metrics.get("mIoU", ""),
                        "accuracy": metrics.get("accuracy", ""),
                        "ceiling_iou": metrics.get("ceiling", ""),
                        "floor_iou": metrics.get("floor", ""),
                        "wall_iou": metrics.get("wall", ""),
                    }
                )
                for class_name in CLASS_NAMES.values():
                    if class_name in metrics:
                        room_row[f"{class_name}_iou"] = metrics[class_name]
                model_room_rows.append(room_row)
                all_room_rows.append(room_row)

            totals = aggregate_apartment_areas(model_room_rows)
            summary_metrics = read_apartment_summary(result_dir)
            apartment_row: Dict[str, object] = {
                "apartment": apartment,
                "model": model_name,
                **totals,
                "real_ceiling_area_m2": "",
                "real_floor_area_m2": "",
                "real_wall_area_m2": "",
                "mIoU": summary_metrics.get("mIoU", ""),
                "accuracy": summary_metrics.get("accuracy", ""),
                "ceiling_iou": summary_metrics.get("ceiling", ""),
                "floor_iou": summary_metrics.get("floor", ""),
                "wall_iou": summary_metrics.get("wall", ""),
                "source_folder": str(result_dir),
            }
            apartment_rows.append(apartment_row)

            metric_row = {
                "apartment": apartment,
                "model": model_name,
                **summary_metrics,
                "source_folder": str(result_dir),
            }
            metric_rows.append(metric_row)

    return {
        "room_rows": all_room_rows,
        "apartment_rows": apartment_rows,
        "metric_rows": metric_rows,
    }


def build_test_metric_comparison(metric_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    by_key = {(r["apartment"], r["model"]): r for r in metric_rows}
    metric_names = ["mIoU", "accuracy", *CLASS_NAMES.values()]
    for apartment in sorted(TEST_APARTMENTS):
        baseline = by_key.get((apartment, BASELINE_NAME), {})
        ft = by_key.get((apartment, FT_NAME), {})
        for metric in metric_names:
            b = baseline.get(metric, "")
            f = ft.get(metric, "")
            diff = ""
            if isinstance(b, (float, int)) and isinstance(f, (float, int)):
                diff = float(f) - float(b)
            rows.append(
                {
                    "apartment": apartment,
                    "metric": metric,
                    "s3dis_pretrained": b,
                    "after_finetune": f,
                    "delta_ft_minus_s3dis": diff,
                }
            )
    return rows


def build_test_surface_comparison(apartment_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    by_key = {(r["apartment"], r["model"]): r for r in apartment_rows}
    for apartment in sorted(TEST_APARTMENTS):
        baseline = by_key.get((apartment, BASELINE_NAME), {})
        ft = by_key.get((apartment, FT_NAME), {})
        for prefix in ("pred", "gt"):
            for surface in SURFACE_LABELS:
                key = f"{prefix}_{surface}_area_m2"
                b = baseline.get(key, "")
                f = ft.get(key, "")
                diff = ""
                pct = ""
                if isinstance(b, (float, int)) and isinstance(f, (float, int)):
                    diff = float(f) - float(b)
                    pct = (diff / float(b) * 100.0) if float(b) else ""
                rows.append(
                    {
                        "apartment": apartment,
                        "surface": surface,
                        "area_source": prefix,
                        "s3dis_pretrained_area_m2": b,
                        "after_finetune_area_m2": f,
                        "delta_ft_minus_s3dis_m2": diff,
                        "delta_percent": pct,
                    }
                )
    return rows


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (float, int, np.floating)):
        return f"{float(value):.{digits}f}"
    if value == "":
        return "--"
    return str(value)



def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.rread:
        apartment_rows = read_csv_rows(args.output_dir / "apartment_surface_areas_all_models.csv")
        metric_comparison = read_csv_rows(args.output_dir / "D2_D4_metric_comparison_s3dis_vs_ft.csv")
        surface_comparison = read_csv_rows(args.output_dir / "D2_D4_surface_area_comparison_s3dis_vs_ft.csv")
        print(f"done: {args.output_dir}")
        return

    tables = build_tables(args)
    room_rows = tables["room_rows"]
    apartment_rows = tables["apartment_rows"]
    metric_rows = tables["metric_rows"]
    metric_comparison = build_test_metric_comparison(metric_rows)
    surface_comparison = build_test_surface_comparison(apartment_rows)

    write_csv(args.output_dir / "room_surface_areas_all_models.csv", room_rows)
    write_csv(args.output_dir / "apartment_surface_areas_all_models.csv", apartment_rows)
    write_csv(
        args.output_dir / "s3dis_apartment_surface_areas_with_real_measurement_columns.csv",
        [r for r in apartment_rows if r["model"] == BASELINE_NAME],
    )
    write_csv(args.output_dir / "apartment_metrics_all_models.csv", metric_rows)
    write_csv(args.output_dir / "D2_D4_metric_comparison_s3dis_vs_ft.csv", metric_comparison)
    write_csv(args.output_dir / "D2_D4_surface_area_comparison_s3dis_vs_ft.csv", surface_comparison)

    print(f"Rooms processed: {len(room_rows)}")
    print(f"Apartment/model rows: {len(apartment_rows)}")


if __name__ == "__main__":
    main()
