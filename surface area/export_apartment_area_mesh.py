"""
Export apartment-level area meshes from room-level prediction PLY files.

The mesh is generated from Pred_label only. GT_label is not used here. For each
room, the script reuses the structural surface-area mesh logic from
room_size_test.py, then concatenates the room meshes into one apartment PLY.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from plyfile import PlyData, PlyElement

from room_surface_area_batch import load_ply, write_area_mesh_ply


MODEL_FOLDERS = {
    "s3dis": ("results", "S3DIS_pretrained"),
    "ft": ("results_after_FT", "S3DIS_pretrained_FT_TUB_CSE"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export apartment area mesh PLYs")
    parser.add_argument("--apartment", default="D2", help="Apartment id, for example D2.")
    parser.add_argument(
        "--apartments",
        nargs="+",
        default=None,
        help="Optional list of apartment ids. Overrides --apartment when provided.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["s3dis", "ft"],
        default=["s3dis", "ft"],
        help="Which result folders to export.",
    )
    parser.add_argument(
        "--tub-root",
        type=Path,
        default=Path("/home/ramiali/3dumamba/data/TUB"),
        help="Root TUB data folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ramiali/3dumamba/data/TUB/THESIS SURFACE AREA RESULTS/AREA_MESH_PLY"),
        help="Output folder for generated mesh PLYs.",
    )

    # Same defaults as room_size_test.py.
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


def result_dir_for(tub_root: Path, apartment: str, model_key: str) -> Path:
    folder_name, _ = MODEL_FOLDERS[model_key]
    return (
        tub_root
        / "combined"
        / "crt"
        / f"{apartment}_rooms_crt"
        / "filtered"
        / "subsampled_0.010"
        / "Block_s2_min_final_8192_norm_enhance_rad_0.5"
        / folder_name
    )


def combine_meshes(mesh_paths: List[Path], output_path: Path) -> None:
    """Concatenate room mesh PLY files into one apartment mesh PLY."""
    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("label", "i4"),
    ]
    face_dtype = [
        ("vertex_indices", "i4", (3,)),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("label", "i4"),
    ]

    vertices: List[Tuple[float, float, float, int, int, int, int]] = []
    faces: List[Tuple[Tuple[int, int, int], int, int, int, int]] = []

    for mesh_path in mesh_paths:
        ply = PlyData.read(str(mesh_path))
        vertex_data = ply["vertex"].data
        face_data = ply["face"].data if "face" in ply else []
        offset = len(vertices)

        for vertex in vertex_data:
            vertices.append(
                (
                    float(vertex["x"]),
                    float(vertex["y"]),
                    float(vertex["z"]),
                    int(vertex["red"]),
                    int(vertex["green"]),
                    int(vertex["blue"]),
                    int(vertex["label"]),
                )
            )

        for face in face_data:
            idx = [int(i) + offset for i in face["vertex_indices"]]
            faces.append(
                (
                    (idx[0], idx[1], idx[2]),
                    int(face["red"]),
                    int(face["green"]),
                    int(face["blue"]),
                    int(face["label"]),
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertex_array = np.array(vertices, dtype=vertex_dtype)
    face_array = np.array(faces, dtype=face_dtype)
    PlyData(
        [
            PlyElement.describe(vertex_array, "vertex"),
            PlyElement.describe(face_array, "face"),
        ],
        text=True,
    ).write(str(output_path))


def export_model_meshes(args: argparse.Namespace, apartment: str, model_key: str) -> Path:
    """Export per-room meshes and one combined apartment mesh for one model."""
    result_dir = result_dir_for(args.tub_root, apartment, model_key)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result folder not found: {result_dir}")

    _, model_name = MODEL_FOLDERS[model_key]
    model_output_dir = args.output_dir / apartment / model_name
    room_output_dir = model_output_dir / "room_meshes"
    room_output_dir.mkdir(parents=True, exist_ok=True)

    mesh_paths: List[Path] = []
    # Use only original inference outputs. The result folders may also contain
    # derived visualization/mesh PLYs that do not have Pred_label.
    for input_ply in sorted(result_dir.glob("*_fil_sub.npy.ply")):
        xyz, labels = load_ply(input_ply, label_source="pred")
        output_mesh = room_output_dir / input_ply.name.replace(".ply", "_area_mesh.ply")
        write_area_mesh_ply(output_mesh, xyz, labels, args)
        mesh_paths.append(output_mesh)

    combined_path = model_output_dir / f"{apartment}_{model_name}_combined_area_mesh.ply"
    combine_meshes(mesh_paths, combined_path)
    return combined_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apartments = args.apartments if args.apartments is not None else [args.apartment]

    for apartment in apartments:
        for model_key in args.models:
            combined_path = export_model_meshes(args, apartment, model_key)
            print(f"Wrote {combined_path}")


if __name__ == "__main__":
    main()
