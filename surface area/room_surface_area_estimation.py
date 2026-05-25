"""
Estimate floor, ceiling, and wall surface areas from a labelled point cloud.

This version estimates actual labelled surface areas:

  - floor area from points labelled as floor
  - ceiling area from points labelled as ceiling
  - visible wall area from points labelled as wall

Supported inputs:
  - PLY files produced by the repo inference scripts, with GT_label/Pred_label
  - TXT/CSV numeric point clouds
  - NPY files with shape (N, C) or prepared blocks with shape (B, N, C)

Example:
    cd /home/ramiali/3dumamba
    /home/ramiali/miniconda3/envs/3dumamba/bin/python data_prepare/room_size_test.py \
        --input data/TUB/combined/crt/D1_rooms_crt/filtered/subsampled_0.010/Block_s2_min_final_8192_norm_enhance_rad_0.5/results/D1_room1_fil_sub.npy.ply \
        --label-source pred \
        --output-csv results_surface_area.csv \
        --output-report SURFACE_AREA_ESTIMATION_REPORT.md

Repository label convention:
    0 = ceiling, 1 = floor, 2 = wall
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree


# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------
# This is the 8-class mapping used by your inference/training scripts. The area
# estimator only needs ceiling, floor, and wall, but keeping the full mapping
# here makes the code easier to audit.
CLASS_NAMES = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "column",
    4: "window",
    5: "door",
    6: "furniture",
    7: "clutter",
}

SURFACE_LABELS = {
    "ceiling": 0,
    "floor": 1,
    "wall": 2,
}

LABEL_COLORS = {
    0: (180, 180, 180),  # ceiling
    1: (0, 200, 0),      # floor
    2: (230, 40, 40),    # wall
    3: (230, 170, 40),   # column
    4: (40, 160, 230),   # window
    5: (160, 80, 40),    # door
    6: (160, 80, 200),   # furniture
    7: (80, 80, 80),     # clutter
}

PLOTLY_COLORS = {
    0: "rgb(180,180,180)",
    1: "rgb(0,200,0)",
    2: "rgb(230,40,40)",
}


@dataclass
class SurfaceEstimate:
    """One final surface-area estimate plus simple diagnostics."""

    name: str
    label: int
    area_m2: float
    point_count: int
    method: str
    warning: str = ""


@dataclass
class WallPlaneEstimate:
    """One detected vertical wall plane and its local area estimate."""

    plane_id: int
    area_m2: float
    point_count: int
    line_normal_x: float
    line_normal_y: float
    line_offset: float


@dataclass
class StructuralPlaneEstimate:
    """One floor/ceiling structural plane accepted from predicted points."""

    plane_id: int
    area_m2: float
    point_count: int
    normal_x: float
    normal_y: float
    normal_z: float
    offset: float
    median_z: float


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def load_point_cloud(path: Path, label_source: str, label_column: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load XYZ coordinates and semantic labels from PLY, TXT/CSV, or NPY.

    Returns:
        xyz:    (N, 3) float array, assumed to be in meters.
        labels: (N,) integer semantic labels.
    """
    suffix = path.suffix.lower()

    if suffix == ".ply":
        return load_ply(path, label_source)

    if suffix == ".npy":
        data = np.load(path)
        if data.ndim == 3:
            # Prepared datasets often store blocks as (num_blocks, points, features).
            data = data.reshape(-1, data.shape[-1])
        elif data.ndim != 2:
            raise ValueError(f"Unsupported NPY shape {data.shape}; expected (N, C) or (B, N, C).")
        return split_numeric_array(data, label_column)

    # np.loadtxt handles whitespace-separated TXT and simple CSV files.
    delimiter = "," if suffix == ".csv" else None
    data = np.loadtxt(path, delimiter=delimiter)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return split_numeric_array(data, label_column)


def load_ply(path: Path, label_source: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read XYZ and labels from PLY.

    Your inference scripts write properties called GT_label and Pred_label. The
    --label-source option selects which one is used for measurement.
    """
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()

    for required in ("x", "y", "z"):
        if required not in names:
            raise ValueError(f"PLY file is missing vertex property '{required}'.")

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)

    if label_source == "pred":
        label_name = "Pred_label"
    elif label_source == "gt":
        label_name = "GT_label"
    else:
        # Auto mode prefers model predictions when available.
        label_name = "Pred_label" if "Pred_label" in names else "GT_label"

    if label_name not in names:
        raise ValueError(f"PLY file does not contain '{label_name}'. Available properties: {', '.join(names)}")

    labels = np.asarray(vertex[label_name]).astype(np.int64)
    return xyz, normalize_labels(labels)


def load_ply_label_pair(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load XYZ, GT labels, and predicted labels from a PLY inference output.

    This is used for model-quality comparison. It requires both GT_label and
    Pred_label to be present in the file.
    """
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()

    missing = [name for name in ("x", "y", "z", "GT_label", "Pred_label") if name not in names]
    if missing:
        raise ValueError(f"Cannot compare GT vs prediction; PLY is missing: {', '.join(missing)}")

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    gt_labels = normalize_labels(np.asarray(vertex["GT_label"]).astype(np.int64))
    pred_labels = normalize_labels(np.asarray(vertex["Pred_label"]).astype(np.int64))
    return xyz, gt_labels, pred_labels


def split_numeric_array(data: np.ndarray, label_column: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a numeric point array into XYZ and labels.

    The first three columns are assumed to be x, y, z. Label column handling:
      - use --label-column if provided;
      - otherwise try column 9 for your raw TUB TXT format;
      - otherwise use the last column, which is common for prepared NPY files.
    """
    if data.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns (x y z label), got shape {data.shape}.")

    xyz = data[:, :3].astype(np.float64)
    col = label_column if label_column is not None else infer_label_column(data)
    labels = data[:, col].astype(np.int64)
    return xyz, normalize_labels(labels)


def infer_label_column(data: np.ndarray) -> int:
    """
    Infer the label column for the common formats in this repo.

    Raw TUB TXT examples look like:
        x y z r g b feature feature feature label normal_x normal_y normal_z
    so the label is column 9. Prepared arrays usually keep label in the last
    column. We choose the first candidate that contains integer-like class ids.
    """
    candidate_columns = []
    if data.shape[1] > 9:
        candidate_columns.append(9)
    candidate_columns.append(data.shape[1] - 1)

    for col in candidate_columns:
        values = data[:, col]
        rounded = np.round(values)
        is_integer_like = np.mean(np.abs(values - rounded) < 1e-6) > 0.99
        in_reasonable_range = np.all((rounded >= 0) & (rounded <= 20))
        if is_integer_like and in_reasonable_range:
            return col

    return data.shape[1] - 1


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Convert 1-based labels to 0-based labels if the file uses 1..8."""
    labels = labels.astype(np.int64)
    if labels.size and labels.min() >= 1 and labels.max() <= 8:
        return labels - 1
    return labels


# ---------------------------------------------------------------------------
# 2D cleaning and projected area calculation
# ---------------------------------------------------------------------------
def voxel_downsample_2d(points_2d: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Keep one projected point per 2D grid cell.

    This reduces repeated/dense points and makes triangulation faster without
    changing the room footprint much.
    """
    if len(points_2d) == 0 or voxel_size <= 0:
        return points_2d

    grid = np.floor(points_2d / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(grid, axis=0, return_index=True)
    return points_2d[np.sort(unique_idx)]


def remove_2d_outliers(points_2d: np.ndarray, k: int = 8, std_ratio: float = 2.5) -> np.ndarray:
    """
    Remove isolated projected points using nearest-neighbour distances.

    A few wrongly labelled points can stretch a hull a long way. This step keeps
    the dense surface footprint and removes sparse outliers.
    """
    if len(points_2d) <= k + 2:
        return points_2d

    tree = cKDTree(points_2d)
    distances, _ = tree.query(points_2d, k=k + 1)
    mean_distance = distances[:, 1:].mean(axis=1)
    limit = mean_distance.mean() + std_ratio * mean_distance.std()
    return points_2d[mean_distance <= limit]


def estimate_projected_area(
    points_2d: np.ndarray,
    alpha_radius: Optional[float],
    alpha_scale: float = 3.0,
) -> Tuple[float, str, str]:
    """
    Estimate area from 2D points using a concave-hull style alpha shape.

    Steps:
      1. Build a Delaunay triangulation.
      2. Compute the circumradius of each triangle.
      3. Keep triangles with circumradius <= alpha_radius.
      4. Sum kept triangle areas.

    Since Delaunay triangles do not overlap, summing accepted triangle areas
    gives a practical concave area estimate without needing Shapely/Open3D.
    """
    if len(points_2d) < 3:
        return 0.0, "not_enough_points", "fewer than 3 projected points"

    try:
        if alpha_radius is None:
            alpha_radius = automatic_alpha_radius(points_2d, alpha_scale)

        area = alpha_shape_area(points_2d, alpha_radius)
        if area > 0:
            return area, f"alpha_shape_radius_{alpha_radius:.3f}", ""

        # If alpha was too strict, convex hull is a useful fallback estimate.
        hull = ConvexHull(points_2d)
        return float(hull.volume), "convex_hull_fallback", "alpha shape produced zero area"

    except QhullError as exc:
        return 0.0, "qhull_failed", str(exc).splitlines()[0]


def automatic_alpha_radius(points_2d: np.ndarray, alpha_scale: float = 3.0) -> float:
    """
    Pick alpha radius from point spacing.

    Smaller radius preserves concave details but can create holes. Larger radius
    fills more occlusion gaps. The alpha_scale argument controls this trade-off.
    """
    if len(points_2d) < 4:
        return 1.0

    tree = cKDTree(points_2d)
    distances, _ = tree.query(points_2d, k=min(4, len(points_2d)))
    neighbour_distances = distances[:, 1:].reshape(-1)
    neighbour_distances = neighbour_distances[neighbour_distances > 0]
    if neighbour_distances.size == 0:
        return 1.0

    return float(np.percentile(neighbour_distances, 90) * alpha_scale)


def alpha_shape_area(points_2d: np.ndarray, alpha_radius: float) -> float:
    """Calculate alpha-shape area by summing accepted Delaunay triangles."""
    _, triangles = alpha_shape_triangles(points_2d, alpha_radius)

    total_area = 0.0
    for triangle in triangles:
        area = triangle_area_2d(triangle)
        total_area += area

    return float(total_area)


def alpha_shape_triangles(points_2d: np.ndarray, alpha_radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return accepted alpha-shape triangle indices and triangle coordinates.

    The indices refer to the input points_2d array and are useful for exporting
    the exact mesh whose triangle areas were summed.
    """
    triangulation = Delaunay(points_2d)
    accepted_indices = []

    for simplex in triangulation.simplices:
        triangle = points_2d[simplex]
        area = triangle_area_2d(triangle)
        if area <= 1e-12:
            continue

        radius = triangle_circumradius(triangle, area)
        if radius <= alpha_radius:
            accepted_indices.append(simplex)

    if not accepted_indices:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 3, 2), dtype=np.float64)

    accepted_indices = np.asarray(accepted_indices, dtype=np.int32)
    return accepted_indices, points_2d[accepted_indices]


def triangle_area_2d(triangle: np.ndarray) -> float:
    """Area of a 2D triangle from its three vertices."""
    a, b, c = triangle
    return abs(np.cross(b - a, c - a)) * 0.5


def triangle_circumradius(triangle: np.ndarray, area: float) -> float:
    """Circumradius R = abc / (4A), where A is triangle area."""
    a = np.linalg.norm(triangle[1] - triangle[0])
    b = np.linalg.norm(triangle[2] - triangle[1])
    c = np.linalg.norm(triangle[0] - triangle[2])
    return (a * b * c) / max(4.0 * area, 1e-12)


# ---------------------------------------------------------------------------
# Structural floor/ceiling plane extraction
# ---------------------------------------------------------------------------
def estimate_floor_reference_z(xyz: np.ndarray, labels: np.ndarray) -> float:
    """
    Estimate the room's low reference height from predicted floor if available.

    This is used only as a geometric prior. It never looks at GT labels.
    """
    floor_points = xyz[labels == SURFACE_LABELS["floor"]]
    if len(floor_points) >= 50:
        return float(np.percentile(floor_points[:, 2], 5))
    return float(np.percentile(xyz[:, 2], 2))


def estimate_structural_surface(
    xyz: np.ndarray,
    labels: np.ndarray,
    surface_name: str,
    args: argparse.Namespace,
) -> Tuple[SurfaceEstimate, List[StructuralPlaneEstimate]]:
    """
    Estimate floor/ceiling from structural planes in the predicted class.

    This rejects isolated wrongly predicted points by requiring them to belong to
    one or more large planes with plausible height and slope.
    """
    label = SURFACE_LABELS[surface_name]
    candidate_points = xyz[labels == label]
    floor_ref_z = estimate_floor_reference_z(xyz, labels)

    if len(candidate_points) < args.min_structural_plane_points:
        return (
            SurfaceEstimate(
                name=surface_name,
                label=label,
                area_m2=0.0,
                point_count=len(candidate_points),
                method="structural_plane_ransac",
                warning=f"not enough predicted {surface_name} points for plane extraction",
            ),
            [],
        )

    if surface_name == "floor":
        max_planes = args.max_floor_planes
        max_slope_deg = args.floor_max_slope_deg
        plane_tolerance = args.floor_plane_tolerance
        min_median_z = None
        max_median_z = floor_ref_z + args.floor_max_height_above_lowest
        alpha_scale = args.floor_ceiling_alpha_scale
    else:
        max_planes = args.max_ceiling_planes
        max_slope_deg = args.ceiling_max_slope_deg
        plane_tolerance = args.ceiling_plane_tolerance
        min_median_z = floor_ref_z + args.ceiling_min_height_above_floor
        max_median_z = None
        alpha_scale = args.floor_ceiling_alpha_scale

    remaining = candidate_points.copy()
    planes: List[StructuralPlaneEstimate] = []
    total_area = 0.0

    for plane_id in range(max_planes):
        if len(remaining) < args.min_structural_plane_points:
            break

        plane = ransac_plane_3d(
            remaining,
            distance_threshold=plane_tolerance,
            iterations=args.plane_ransac_iterations,
            max_slope_deg=max_slope_deg,
            min_median_z=min_median_z,
            max_median_z=max_median_z,
        )
        if plane is None:
            break

        normal, offset, inlier_mask = plane
        inliers = remaining[inlier_mask]
        if len(inliers) < args.min_structural_plane_points:
            break

        projected = project_points_to_plane_2d(inliers, normal)
        projected = voxel_downsample_2d(projected, args.voxel_size)
        projected = remove_2d_outliers(projected)
        area, _, _ = estimate_projected_area(projected, args.alpha_radius, alpha_scale=alpha_scale)
        total_area += area

        planes.append(
            StructuralPlaneEstimate(
                plane_id=plane_id,
                area_m2=area,
                point_count=len(inliers),
                normal_x=float(normal[0]),
                normal_y=float(normal[1]),
                normal_z=float(normal[2]),
                offset=float(offset),
                median_z=float(np.median(inliers[:, 2])),
            )
        )

        remaining = remaining[~inlier_mask]

    warning = ""
    if not planes:
        warning = "no plausible structural plane detected from predicted points"
    elif len(remaining) > 0.5 * len(candidate_points):
        warning = "more than half of predicted points were rejected by structural-plane filtering"

    method = f"structural_plane_ransac_{len(planes)}"
    return (
        SurfaceEstimate(
            name=surface_name,
            label=label,
            area_m2=float(total_area),
            point_count=len(candidate_points),
            method=method,
            warning=warning,
        ),
        planes,
    )


def ransac_plane_3d(
    points: np.ndarray,
    distance_threshold: float,
    iterations: int,
    max_slope_deg: float,
    min_median_z: Optional[float],
    max_median_z: Optional[float],
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """
    Fit a constrained 3D plane using RANSAC.

    The normal is constrained by max_slope_deg relative to vertical. For floors
    this should be strict; for sloped ceilings it can be larger.
    """
    if len(points) < 3:
        return None

    min_abs_nz = np.cos(np.deg2rad(max_slope_deg))
    rng = np.random.default_rng(42)
    best_normal = None
    best_offset = 0.0
    best_mask = None
    best_count = 0

    for _ in range(iterations):
        idx = rng.choice(len(points), size=3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue

        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        if abs(normal[2]) < min_abs_nz:
            continue

        offset = -float(normal @ p1)
        distances = np.abs(points @ normal + offset)
        mask = distances <= distance_threshold
        if not accepted_plane_height(points[mask], min_median_z, max_median_z):
            continue

        count = int(mask.sum())
        if count > best_count:
            best_normal = normal
            best_offset = offset
            best_mask = mask
            best_count = count

    if best_normal is None or best_mask is None:
        return None

    refined_normal, refined_offset = fit_plane_pca(points[best_mask])
    if refined_normal[2] < 0:
        refined_normal = -refined_normal
        refined_offset = -refined_offset
    if abs(refined_normal[2]) < min_abs_nz:
        return None

    distances = np.abs(points @ refined_normal + refined_offset)
    refined_mask = distances <= distance_threshold
    if not accepted_plane_height(points[refined_mask], min_median_z, max_median_z):
        return None
    return refined_normal, refined_offset, refined_mask


def accepted_plane_height(
    points: np.ndarray,
    min_median_z: Optional[float],
    max_median_z: Optional[float],
) -> bool:
    """Check whether a plane's inlier height is plausible for floor/ceiling."""
    if len(points) == 0:
        return False
    median_z = float(np.median(points[:, 2]))
    if min_median_z is not None and median_z < min_median_z:
        return False
    if max_median_z is not None and median_z > max_median_z:
        return False
    return True


def fit_plane_pca(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least-squares plane fit using PCA."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal /= max(np.linalg.norm(normal), 1e-12)
    offset = -float(normal @ centroid)
    return normal, offset


def plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build two orthonormal in-plane axes for a plane normal.

    Returns normalized normal, basis_u, basis_v.
    """
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(normal @ reference) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    basis_u = np.cross(normal, reference)
    basis_u /= max(np.linalg.norm(basis_u), 1e-12)
    basis_v = np.cross(normal, basis_u)
    basis_v /= max(np.linalg.norm(basis_v), 1e-12)
    return normal, basis_u, basis_v


def project_points_to_plane_2d(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Project 3D points into local 2D coordinates on a plane.

    For sloped ceilings this preserves true surface area better than XY
    projection, because the measurement is made in the ceiling plane itself.
    """
    _, basis_u, basis_v = plane_basis(normal)
    return np.column_stack([points @ basis_u, points @ basis_v])


def lift_plane_2d_to_3d(points_2d: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Map local 2D plane coordinates back to 3D points on the fitted plane."""
    normal, basis_u, basis_v = plane_basis(normal)
    return (
        points_2d[:, :1] * basis_u.reshape(1, 3)
        + points_2d[:, 1:2] * basis_v.reshape(1, 3)
        - offset * normal.reshape(1, 3)
    )


# ---------------------------------------------------------------------------
# Floor and ceiling area
# ---------------------------------------------------------------------------
def estimate_horizontal_surface(
    xyz: np.ndarray,
    labels: np.ndarray,
    surface_name: str,
    voxel_size: float,
    alpha_radius: Optional[float],
) -> SurfaceEstimate:
    """
    Estimate floor/ceiling area by projecting that class onto XY.

    This assumes the floor and ceiling are mostly horizontal. For a normal room,
    the 3D surface area is approximately equal to the top-down projected area.
    """
    label = SURFACE_LABELS[surface_name]
    surface_points = xyz[labels == label]
    projected = surface_points[:, :2]
    projected = voxel_downsample_2d(projected, voxel_size)
    projected = remove_2d_outliers(projected)

    area, method, warning = estimate_projected_area(projected, alpha_radius)
    return SurfaceEstimate(
        name=surface_name,
        label=label,
        area_m2=area,
        point_count=len(surface_points),
        method=method,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Wall area
# ---------------------------------------------------------------------------
def estimate_wall_surface(
    xyz: np.ndarray,
    labels: np.ndarray,
    voxel_size: float,
    alpha_radius: Optional[float],
    distance_threshold: float,
    min_plane_points: int,
    max_planes: int,
    ransac_iterations: int,
) -> Tuple[SurfaceEstimate, List[WallPlaneEstimate]]:
    """
    Estimate wall area by detecting vertical wall planes.

    Wall points are not projected globally, because different walls face
    different directions. Instead, each wall plane is detected in XY, then
    unfolded into a local 2D coordinate system: distance along wall vs height z.
    """
    wall_label = SURFACE_LABELS["wall"]
    wall_points = xyz[labels == wall_label]

    if len(wall_points) < min_plane_points:
        estimate = SurfaceEstimate(
            name="wall",
            label=wall_label,
            area_m2=0.0,
            point_count=len(wall_points),
            method="vertical_ransac_planes",
            warning=f"not enough wall points for plane extraction: {len(wall_points)}",
        )
        return estimate, []

    remaining = wall_points.copy()
    plane_estimates: List[WallPlaneEstimate] = []

    for plane_id in range(max_planes):
        if len(remaining) < min_plane_points:
            break

        line = ransac_line_2d(
            remaining[:, :2],
            distance_threshold=distance_threshold,
            iterations=ransac_iterations,
        )
        if line is None:
            break

        normal, offset, inlier_mask = line
        inliers = remaining[inlier_mask]
        if len(inliers) < min_plane_points:
            break

        # Convert 3D wall points to local 2D coordinates.
        # u = position along the wall, z = vertical height.
        direction = np.array([-normal[1], normal[0]], dtype=np.float64)
        direction /= max(np.linalg.norm(direction), 1e-12)
        wall_u = inliers[:, :2] @ direction
        wall_z = inliers[:, 2]
        wall_2d = np.column_stack([wall_u, wall_z])

        wall_2d = voxel_downsample_2d(wall_2d, voxel_size)
        wall_2d = remove_2d_outliers(wall_2d)
        area, _, _ = estimate_projected_area(wall_2d, alpha_radius)

        plane_estimates.append(
            WallPlaneEstimate(
                plane_id=plane_id,
                area_m2=area,
                point_count=len(inliers),
                line_normal_x=float(normal[0]),
                line_normal_y=float(normal[1]),
                line_offset=float(offset),
            )
        )

        # Remove this wall's inliers before looking for the next wall.
        remaining = remaining[~inlier_mask]

    total_area = float(sum(p.area_m2 for p in plane_estimates))
    warning = ""
    if not plane_estimates:
        warning = "no vertical wall planes detected"
    elif len(remaining) > 0.5 * len(wall_points):
        warning = "more than half of wall-labelled points were not assigned to wall planes"

    estimate = SurfaceEstimate(
        name="wall",
        label=wall_label,
        area_m2=total_area,
        point_count=len(wall_points),
        method=f"vertical_ransac_planes_{len(plane_estimates)}",
        warning=warning,
    )
    return estimate, plane_estimates


def ransac_line_2d(
    points_2d: np.ndarray,
    distance_threshold: float,
    iterations: int,
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """
    Fit a 2D line with RANSAC.

    Line equation:
        normal_x * x + normal_y * y + offset = 0
    """
    if len(points_2d) < 2:
        return None

    best_normal = None
    best_offset = 0.0
    best_mask = None
    best_count = 0
    rng = np.random.default_rng(42)

    for _ in range(iterations):
        i, j = rng.choice(len(points_2d), size=2, replace=False)
        p1 = points_2d[i]
        p2 = points_2d[j]
        direction = p2 - p1
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-9:
            continue

        direction /= direction_norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        offset = -float(normal @ p1)
        distances = np.abs(points_2d @ normal + offset)
        mask = distances <= distance_threshold
        count = int(mask.sum())

        if count > best_count:
            best_normal = normal
            best_offset = offset
            best_mask = mask
            best_count = count

    if best_normal is None or best_mask is None:
        return None

    # Refine the winning line using PCA over its inliers.
    inliers = points_2d[best_mask]
    refined_normal, refined_offset = fit_line_pca(inliers)
    distances = np.abs(points_2d @ refined_normal + refined_offset)
    refined_mask = distances <= distance_threshold
    return refined_normal, refined_offset, refined_mask


def fit_line_pca(points_2d: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least-squares 2D line fit using PCA."""
    centroid = points_2d.mean(axis=0)
    centered = points_2d - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    normal /= max(np.linalg.norm(normal), 1e-12)
    offset = -float(normal @ centroid)
    return normal, offset


# ---------------------------------------------------------------------------
# Shared estimation helper
# ---------------------------------------------------------------------------
def estimate_all_surfaces(
    xyz: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[List[SurfaceEstimate], List[WallPlaneEstimate]]:
    """Estimate ceiling, floor, and wall areas using the current CLI settings."""
    ceiling, _ = estimate_structural_surface(xyz, labels, "ceiling", args)
    floor, _ = estimate_structural_surface(xyz, labels, "floor", args)
    wall, wall_planes = estimate_wall_surface(
        xyz=xyz,
        labels=labels,
        voxel_size=args.voxel_size,
        alpha_radius=args.alpha_radius,
        distance_threshold=args.wall_distance_threshold,
        min_plane_points=args.min_wall_plane_points,
        max_planes=args.max_wall_planes,
        ransac_iterations=args.ransac_iterations,
    )
    return [ceiling, floor, wall], wall_planes


def build_measured_surface_mask(
    xyz: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """
    Mark points that belong to the structural surfaces used for reporting.

    The mask is based on predicted labels plus geometric plane constraints. GT is
    never used here.
    """
    mask = np.zeros(len(labels), dtype=bool)

    for surface_name in ("ceiling", "floor"):
        _, planes = estimate_structural_surface(xyz, labels, surface_name, args)
        label = SURFACE_LABELS[surface_name]
        label_mask = labels == label
        points = xyz[label_mask]
        local_keep = np.zeros(len(points), dtype=bool)
        tolerance = args.ceiling_plane_tolerance if surface_name == "ceiling" else args.floor_plane_tolerance

        for plane in planes:
            normal = np.array([plane.normal_x, plane.normal_y, plane.normal_z], dtype=np.float64)
            distances = np.abs(points @ normal + plane.offset)
            local_keep |= distances <= tolerance

        label_indices = np.where(label_mask)[0]
        mask[label_indices[local_keep]] = True

    wall_estimate, wall_planes = estimate_wall_surface(
        xyz=xyz,
        labels=labels,
        voxel_size=args.voxel_size,
        alpha_radius=args.alpha_radius,
        distance_threshold=args.wall_distance_threshold,
        min_plane_points=args.min_wall_plane_points,
        max_planes=args.max_wall_planes,
        ransac_iterations=args.ransac_iterations,
    )
    _ = wall_estimate
    wall_label = SURFACE_LABELS["wall"]
    wall_mask = labels == wall_label
    wall_points = xyz[wall_mask]
    wall_keep = np.zeros(len(wall_points), dtype=bool)
    for plane in wall_planes:
        normal = np.array([plane.line_normal_x, plane.line_normal_y], dtype=np.float64)
        distances = np.abs(wall_points[:, :2] @ normal + plane.line_offset)
        wall_keep |= distances <= args.wall_distance_threshold

    wall_indices = np.where(wall_mask)[0]
    mask[wall_indices[wall_keep]] = True
    return mask


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
def write_csv(
    output_csv: Path,
    input_path: Path,
    estimates: Iterable[SurfaceEstimate],
    wall_planes: Iterable[WallPlaneEstimate],
) -> None:
    """Write surface totals and per-wall-plane details to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "surface", "label", "area_m2", "point_count", "method", "warning"])
        for estimate in estimates:
            writer.writerow(
                [
                    str(input_path),
                    estimate.name,
                    estimate.label,
                    f"{estimate.area_m2:.6f}",
                    estimate.point_count,
                    estimate.method,
                    estimate.warning,
                ]
            )

        writer.writerow([])
        writer.writerow(["wall_plane_id", "area_m2", "point_count", "normal_x", "normal_y", "offset"])
        for plane in wall_planes:
            writer.writerow(
                [
                    plane.plane_id,
                    f"{plane.area_m2:.6f}",
                    plane.point_count,
                    f"{plane.line_normal_x:.6f}",
                    f"{plane.line_normal_y:.6f}",
                    f"{plane.line_offset:.6f}",
                ]
            )


def write_method_report(
    output_report: Path,
    input_path: Path,
    estimates: List[SurfaceEstimate],
    wall_planes: List[WallPlaneEstimate],
    args: argparse.Namespace,
) -> None:
    """Write Markdown documentation explaining the area calculation."""
    output_report.parent.mkdir(parents=True, exist_ok=True)
    estimate_by_name: Dict[str, SurfaceEstimate] = {e.name: e for e in estimates}

    text = f"""# Surface Area Estimation Report

## Input

- Point cloud: `{input_path}`
- Label source: `{args.label_source}`
- Coordinate source: `x, y, z`
- Semantic labels: `0=ceiling`, `1=floor`, `2=wall`
- Units: assumed meters, so reported areas are square meters

## Results

| Surface | Label | Area m2 | Points | Method | Warning |
|---|---:|---:|---:|---|---|
| Ceiling | 0 | {estimate_by_name['ceiling'].area_m2:.3f} | {estimate_by_name['ceiling'].point_count} | {estimate_by_name['ceiling'].method} | {estimate_by_name['ceiling'].warning or '-'} |
| Floor | 1 | {estimate_by_name['floor'].area_m2:.3f} | {estimate_by_name['floor'].point_count} | {estimate_by_name['floor'].method} | {estimate_by_name['floor'].warning or '-'} |
| Wall | 2 | {estimate_by_name['wall'].area_m2:.3f} | {estimate_by_name['wall'].point_count} | {estimate_by_name['wall'].method} | {estimate_by_name['wall'].warning or '-'} |

## How Floor Area Is Calculated

1. Points predicted with semantic label `1` are selected. GT labels are not used.
2. A constrained RANSAC plane search finds plausible floor plane(s).
3. Floor planes are required to be close to horizontal, with maximum slope `{args.floor_max_slope_deg}` degrees.
4. Floor planes higher than `{args.floor_max_height_above_lowest}` meters above the predicted floor reference are rejected.
5. Accepted floor points are projected into the local 2D plane coordinates.
6. Projected points are voxel-downsampled with voxel size `{args.voxel_size}` meters.
7. Sparse projected outliers are removed using nearest-neighbour distance filtering.
8. A 2D Delaunay triangulation is built over the remaining projected points.
9. Triangles with circumradius below the alpha radius are kept.
10. The floor area is the sum of the kept triangle areas.

This gives a concave footprint estimate, so it can represent L-shaped rooms
better than a simple bounding box or convex hull. Floor/ceiling use alpha scale
`{args.floor_ceiling_alpha_scale}` so furniture/clutter occlusion holes can be
bridged when enough boundary floor evidence exists.

## How Ceiling Area Is Calculated

Ceiling area uses predicted semantic label `0`. A constrained RANSAC plane search
keeps only plausible ceiling plane(s). A predicted ceiling plane is rejected if
its median height is lower than `{args.ceiling_min_height_above_floor}` meters
above the predicted floor reference. This removes low furniture patches that are
wrongly predicted as ceiling. Ceiling planes may have slope up to
`{args.ceiling_max_slope_deg}` degrees, so split or sloped ceilings can be
measured as multiple planes. Area is calculated in each accepted ceiling plane's
own 2D coordinates and then summed.

## How Wall Area Is Calculated

1. Points with semantic label `2` are selected.
2. Because walls are vertical, the wall points are projected onto `XY`.
3. RANSAC repeatedly fits straight wall lines in `XY`.
4. For each detected wall line, inlier points are treated as one vertical wall plane.
5. Each wall plane is converted into local 2D coordinates:
   - horizontal coordinate: distance along the wall line
   - vertical coordinate: original `z`
6. The same alpha-shape triangulation method is applied in this local wall plane.
7. The total wall area is the sum of all detected wall-plane areas.

This estimates **visible wall surface area** from points labelled as wall. Doors,
windows, occlusions, and furniture-covered regions are not included unless they
are also labelled as wall. Wall alpha remains stricter than floor/ceiling because
curtains, furniture near walls, and window/door boundaries can otherwise inflate
the wall estimate.

## Wall Plane Details

| Plane | Area m2 | Points | Line equation |
|---:|---:|---:|---|
"""

    for plane in wall_planes:
        text += (
            f"| {plane.plane_id} | {plane.area_m2:.3f} | {plane.point_count} | "
            f"`{plane.line_normal_x:.4f}x + {plane.line_normal_y:.4f}y + {plane.line_offset:.4f} = 0` |\n"
        )

    if not wall_planes:
        text += "| - | 0.000 | 0 | No wall planes detected |\n"

    text += f"""
## Parameters

- `voxel_size`: `{args.voxel_size}`
- `alpha_radius`: `{args.alpha_radius if args.alpha_radius is not None else 'automatic'}`
- `floor_ceiling_alpha_scale`: `{args.floor_ceiling_alpha_scale}`
- `floor_plane_tolerance`: `{args.floor_plane_tolerance}`
- `ceiling_plane_tolerance`: `{args.ceiling_plane_tolerance}`
- `ceiling_min_height_above_floor`: `{args.ceiling_min_height_above_floor}`
- `floor_max_height_above_lowest`: `{args.floor_max_height_above_lowest}`
- `floor_max_slope_deg`: `{args.floor_max_slope_deg}`
- `ceiling_max_slope_deg`: `{args.ceiling_max_slope_deg}`
- `wall_distance_threshold`: `{args.wall_distance_threshold}`
- `min_wall_plane_points`: `{args.min_wall_plane_points}`
- `max_wall_planes`: `{args.max_wall_planes}`
- `ransac_iterations`: `{args.ransac_iterations}`
- `plane_ransac_iterations`: `{args.plane_ransac_iterations}`

## Important Assumptions And Limitations

- Coordinates are assumed to be metric. If the point cloud is in centimetres or
  millimetres, convert it before running this script.
- Floor is assumed mostly horizontal. Ceiling can be horizontal or moderately sloped.
- Walls are assumed mostly vertical and planar.
- Wall area is visible labelled wall area, not gross architectural wall area.
  Gross wall area would be calculated from room perimeter times room height.
- Segmentation quality strongly affects the result. False wall/floor/ceiling
  labels can inflate or shrink estimated areas.
"""

    output_report.write_text(text)


def write_visualization_ply(
    output_ply: Path,
    xyz: np.ndarray,
    labels: np.ndarray,
    estimates: List[SurfaceEstimate],
    gt_labels: Optional[np.ndarray] = None,
    keep_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Write a colored 3D point cloud for visual inspection.

    Open the resulting PLY in CloudCompare, MeshLab, or ParaView. Points are
    filtered to only ceiling, floor, and wall, then colored by semantic class.
    The scalar `surface_area_m2` stores the total estimated area of that surface,
    repeated on every point belonging to that label.
    """
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    area_by_label = {estimate.label: estimate.area_m2 for estimate in estimates}
    measured_labels = set(SURFACE_LABELS.values())
    keep = np.array([int(label) in measured_labels for label in labels])
    if keep_mask is not None:
        keep &= keep_mask
    xyz = xyz[keep]
    labels = labels[keep]
    if gt_labels is not None:
        gt_labels = gt_labels[keep]

    has_gt = gt_labels is not None and len(gt_labels) == len(labels)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("label", "i4"),
        ("surface_area_m2", "f4"),
    ]
    if has_gt:
        dtype.extend([("GT_label", "i4"), ("Correct", "u1")])

    vertices = np.empty(len(xyz), dtype=dtype)
    vertices["x"] = xyz[:, 0].astype(np.float32)
    vertices["y"] = xyz[:, 1].astype(np.float32)
    vertices["z"] = xyz[:, 2].astype(np.float32)
    vertices["label"] = labels.astype(np.int32)
    vertices["surface_area_m2"] = np.array([area_by_label.get(int(label), 0.0) for label in labels], dtype=np.float32)

    colors = np.array([LABEL_COLORS.get(int(label), (255, 255, 255)) for label in labels], dtype=np.uint8)
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    if has_gt:
        vertices["GT_label"] = gt_labels.astype(np.int32)
        vertices["Correct"] = (labels == gt_labels).astype(np.uint8)

    PlyData([PlyElement.describe(vertices, "vertex")], text=True).write(str(output_ply))


def write_surface_area_html(
    output_html: Path,
    xyz: np.ndarray,
    labels: np.ndarray,
    estimates: List[SurfaceEstimate],
    max_points_per_surface: int,
    keep_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Write an interactive report-oriented 3D visualization.

    Unlike a PLY, the HTML legend can directly show the calculated area values.
    Only ceiling, floor, and wall points are plotted, so the figure focuses on
    the surfaces used in the measurement.
    """
    import plotly.graph_objects as go

    output_html.parent.mkdir(parents=True, exist_ok=True)
    area_by_label = {estimate.label: estimate.area_m2 for estimate in estimates}
    rng = np.random.default_rng(42)
    fig = go.Figure()

    for surface_name, label in SURFACE_LABELS.items():
        surface_mask = labels == label
        if keep_mask is not None:
            surface_mask &= keep_mask
        surface_points = xyz[surface_mask]
        if len(surface_points) == 0:
            continue

        if max_points_per_surface > 0 and len(surface_points) > max_points_per_surface:
            idx = rng.choice(len(surface_points), size=max_points_per_surface, replace=False)
            surface_points = surface_points[np.sort(idx)]

        area = area_by_label.get(label, 0.0)
        fig.add_trace(
            go.Scatter3d(
                x=surface_points[:, 0],
                y=surface_points[:, 1],
                z=surface_points[:, 2],
                mode="markers",
                name=f"{surface_name}: {area:.2f} m2",
                marker={
                    "size": 2.2,
                    "color": PLOTLY_COLORS[label],
                    "opacity": 0.85,
                },
                hovertemplate=(
                    f"{surface_name}<br>"
                    f"area: {area:.3f} m2<br>"
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Predicted Ceiling, Floor, and Wall Surface Areas",
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)


def build_alpha_mesh_2d(
    points_2d: np.ndarray,
    alpha_radius: Optional[float],
    alpha_scale: float,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build the exact 2D alpha-shape triangle mesh used for area summation.

    Returns 2D vertices, triangle indices, and summed triangle area.
    """
    points_2d = voxel_downsample_2d(points_2d, voxel_size)
    points_2d = remove_2d_outliers(points_2d)
    if len(points_2d) < 3:
        return points_2d, np.empty((0, 3), dtype=np.int32), 0.0

    radius = alpha_radius if alpha_radius is not None else automatic_alpha_radius(points_2d, alpha_scale)
    triangle_indices, triangles = alpha_shape_triangles(points_2d, radius)
    area = float(sum(triangle_area_2d(triangle) for triangle in triangles))
    return points_2d, triangle_indices, area


def write_area_mesh_ply(
    output_mesh_ply: Path,
    xyz: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """
    Export the reconstructed area surface as a triangle mesh PLY.

    This is the best visual explanation of the area calculation: each face is
    one accepted alpha-shape triangle, and total area is the sum of these face
    areas. The mesh is built only from predicted labels plus geometric filters.
    """
    output_mesh_ply.parent.mkdir(parents=True, exist_ok=True)
    vertices: List[Tuple[float, float, float, int, int, int, int]] = []
    faces: List[Tuple[Tuple[int, int, int], int, int, int, int]] = []

    def add_mesh(vertices_3d: np.ndarray, triangle_indices: np.ndarray, label: int) -> None:
        if len(vertices_3d) == 0 or len(triangle_indices) == 0:
            return
        color = LABEL_COLORS[label]
        start = len(vertices)
        for vertex in vertices_3d:
            vertices.append((float(vertex[0]), float(vertex[1]), float(vertex[2]), color[0], color[1], color[2], label))
        for tri in triangle_indices:
            faces.append(((start + int(tri[0]), start + int(tri[1]), start + int(tri[2])), color[0], color[1], color[2], label))

    # Floor and ceiling: structural 3D planes, triangulated in their own local
    # 2D plane coordinates, then lifted back to 3D.
    for surface_name in ("floor", "ceiling"):
        label = SURFACE_LABELS[surface_name]
        candidate_points = xyz[labels == label]
        floor_ref_z = estimate_floor_reference_z(xyz, labels)

        if surface_name == "floor":
            max_planes = args.max_floor_planes
            max_slope_deg = args.floor_max_slope_deg
            plane_tolerance = args.floor_plane_tolerance
            min_median_z = None
            max_median_z = floor_ref_z + args.floor_max_height_above_lowest
        else:
            max_planes = args.max_ceiling_planes
            max_slope_deg = args.ceiling_max_slope_deg
            plane_tolerance = args.ceiling_plane_tolerance
            min_median_z = floor_ref_z + args.ceiling_min_height_above_floor
            max_median_z = None

        remaining = candidate_points.copy()
        for _ in range(max_planes):
            if len(remaining) < args.min_structural_plane_points:
                break

            plane = ransac_plane_3d(
                remaining,
                distance_threshold=plane_tolerance,
                iterations=args.plane_ransac_iterations,
                max_slope_deg=max_slope_deg,
                min_median_z=min_median_z,
                max_median_z=max_median_z,
            )
            if plane is None:
                break

            normal, offset, inlier_mask = plane
            inliers = remaining[inlier_mask]
            projected = project_points_to_plane_2d(inliers, normal)
            mesh_2d, triangle_indices, _ = build_alpha_mesh_2d(
                projected,
                args.alpha_radius,
                args.floor_ceiling_alpha_scale,
                args.voxel_size,
            )
            mesh_3d = lift_plane_2d_to_3d(mesh_2d, normal, offset)
            add_mesh(mesh_3d, triangle_indices, label)
            remaining = remaining[~inlier_mask]

    # Walls: each vertical wall line in XY becomes a local 2D wall plane
    # coordinate system: u along the wall, v = z.
    _, wall_planes = estimate_wall_surface(
        xyz=xyz,
        labels=labels,
        voxel_size=args.voxel_size,
        alpha_radius=args.alpha_radius,
        distance_threshold=args.wall_distance_threshold,
        min_plane_points=args.min_wall_plane_points,
        max_planes=args.max_wall_planes,
        ransac_iterations=args.ransac_iterations,
    )
    wall_label = SURFACE_LABELS["wall"]
    wall_points = xyz[labels == wall_label]
    for plane in wall_planes:
        normal_2d = np.array([plane.line_normal_x, plane.line_normal_y], dtype=np.float64)
        normal_2d /= max(np.linalg.norm(normal_2d), 1e-12)
        direction_2d = np.array([-normal_2d[1], normal_2d[0]], dtype=np.float64)
        distances = np.abs(wall_points[:, :2] @ normal_2d + plane.line_offset)
        inliers = wall_points[distances <= args.wall_distance_threshold]
        if len(inliers) < args.min_wall_plane_points:
            continue

        wall_u = inliers[:, :2] @ direction_2d
        wall_z = inliers[:, 2]
        wall_2d = np.column_stack([wall_u, wall_z])
        mesh_2d, triangle_indices, _ = build_alpha_mesh_2d(
            wall_2d,
            args.alpha_radius,
            3.0,
            args.voxel_size,
        )
        xy = (
            mesh_2d[:, :1] * direction_2d.reshape(1, 2)
            - plane.line_offset * normal_2d.reshape(1, 2)
        )
        mesh_3d = np.column_stack([xy[:, 0], xy[:, 1], mesh_2d[:, 1]])
        add_mesh(mesh_3d, triangle_indices, wall_label)

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
    vertex_array = np.array(vertices, dtype=vertex_dtype)
    face_array = np.array(faces, dtype=face_dtype)
    PlyData(
        [
            PlyElement.describe(vertex_array, "vertex"),
            PlyElement.describe(face_array, "face"),
        ],
        text=True,
    ).write(str(output_mesh_ply))


def write_comparison_csv(
    output_csv: Path,
    input_path: Path,
    gt_estimates: List[SurfaceEstimate],
    pred_estimates: List[SurfaceEstimate],
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    num_classes: int = 8,
) -> None:
    """
    Compare predicted segmentation against GT labels.

    This reports both ordinary semantic-segmentation metrics and area differences
    for the three measured architectural surfaces.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    gt_area = {estimate.label: estimate.area_m2 for estimate in gt_estimates}
    pred_area = {estimate.label: estimate.area_m2 for estimate in pred_estimates}

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (gt_labels >= 0) & (gt_labels < num_classes) & (pred_labels >= 0) & (pred_labels < num_classes)
    for gt, pred in zip(gt_labels[valid], pred_labels[valid]):
        confusion[int(gt), int(pred)] += 1

    total = confusion.sum()
    overall_accuracy = float(np.trace(confusion) / total) if total else 0.0

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", str(input_path)])
        writer.writerow(["overall_accuracy", f"{overall_accuracy:.6f}"])
        writer.writerow([])

        writer.writerow(["surface", "label", "gt_area_m2", "pred_area_m2", "difference_m2", "difference_percent"])
        for name, label in SURFACE_LABELS.items():
            gt_value = gt_area.get(label, 0.0)
            pred_value = pred_area.get(label, 0.0)
            diff = pred_value - gt_value
            diff_percent = (diff / gt_value * 100.0) if gt_value > 0 else 0.0
            writer.writerow([name, label, f"{gt_value:.6f}", f"{pred_value:.6f}", f"{diff:.6f}", f"{diff_percent:.3f}"])

        writer.writerow([])
        writer.writerow(["class", "label", "iou", "precision", "recall", "gt_points", "pred_points"])
        for label in range(num_classes):
            tp = confusion[label, label]
            fp = confusion[:, label].sum() - tp
            fn = confusion[label, :].sum() - tp
            iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            writer.writerow(
                [
                    CLASS_NAMES.get(label, f"class_{label}"),
                    label,
                    f"{iou:.6f}",
                    f"{precision:.6f}",
                    f"{recall:.6f}",
                    int(confusion[label, :].sum()),
                    int(confusion[:, label].sum()),
                ]
            )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Collect command-line options for one room/point-cloud measurement."""
    parser = argparse.ArgumentParser(
        description="Estimate floor, ceiling, and wall areas from a labelled point cloud."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .ply, .txt, .csv, or .npy file.")
    parser.add_argument(
        "--label-source",
        choices=["auto", "pred", "gt"],
        default="auto",
        help="For PLY files, choose Pred_label, GT_label, or auto-detect.",
    )
    parser.add_argument(
        "--label-column",
        type=int,
        default=None,
        help="For TXT/CSV/NPY files, explicit label column. Default: infer automatically.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.03,
        help="2D projection downsampling size in meters.",
    )
    parser.add_argument(
        "--alpha-radius",
        type=float,
        default=None,
        help="Concave-hull radius in meters. Default: estimate from point spacing.",
    )
    parser.add_argument(
        "--floor-ceiling-alpha-scale",
        type=float,
        default=6.0,
        help="Gap-fill scale for floor/ceiling footprints. Larger values fill occlusions from furniture/clutter.",
    )
    parser.add_argument(
        "--floor-plane-tolerance",
        type=float,
        default=0.08,
        help="Maximum distance in meters from predicted floor points to accepted floor planes.",
    )
    parser.add_argument(
        "--ceiling-plane-tolerance",
        type=float,
        default=0.10,
        help="Maximum distance in meters from predicted ceiling points to accepted ceiling planes.",
    )
    parser.add_argument(
        "--ceiling-min-height-above-floor",
        type=float,
        default=1.8,
        help="Reject predicted ceiling planes below this height above the predicted floor reference.",
    )
    parser.add_argument(
        "--floor-max-height-above-lowest",
        type=float,
        default=0.35,
        help="Reject predicted floor planes higher than this above the predicted floor reference.",
    )
    parser.add_argument(
        "--floor-max-slope-deg",
        type=float,
        default=12.0,
        help="Maximum floor plane slope from horizontal.",
    )
    parser.add_argument(
        "--ceiling-max-slope-deg",
        type=float,
        default=45.0,
        help="Maximum ceiling plane slope from horizontal; allows sloped ceilings.",
    )
    parser.add_argument(
        "--max-floor-planes",
        type=int,
        default=2,
        help="Maximum number of structural floor planes to measure.",
    )
    parser.add_argument(
        "--max-ceiling-planes",
        type=int,
        default=3,
        help="Maximum number of structural ceiling planes to measure; supports split/sloped ceilings.",
    )
    parser.add_argument(
        "--min-structural-plane-points",
        type=int,
        default=250,
        help="Minimum predicted points required to accept a floor/ceiling structural plane.",
    )
    parser.add_argument(
        "--plane-ransac-iterations",
        type=int,
        default=900,
        help="RANSAC iterations for floor/ceiling structural plane extraction.",
    )
    parser.add_argument(
        "--wall-distance-threshold",
        type=float,
        default=0.06,
        help="Maximum XY distance in meters for a wall point to belong to a RANSAC wall line.",
    )
    parser.add_argument(
        "--min-wall-plane-points",
        type=int,
        default=300,
        help="Minimum number of points required to accept one wall plane.",
    )
    parser.add_argument(
        "--max-wall-planes",
        type=int,
        default=12,
        help="Maximum number of vertical wall planes to detect.",
    )
    parser.add_argument(
        "--ransac-iterations",
        type=int,
        default=600,
        help="Number of random line hypotheses tested per wall plane.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("surface_area_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("surface_area_method_report.md"),
        help="Markdown documentation/report output path.",
    )
    parser.add_argument(
        "--output-visualization-ply",
        type=Path,
        default=None,
        help="Optional colored PLY output containing only ceiling, floor, and wall points.",
    )
    parser.add_argument(
        "--output-visualization-html",
        type=Path,
        default=None,
        help="Optional interactive HTML visualization with area values in the legend.",
    )
    parser.add_argument(
        "--output-area-mesh-ply",
        type=Path,
        default=None,
        help="Optional triangle mesh PLY showing the alpha-shape surfaces whose face areas were summed.",
    )
    parser.add_argument(
        "--include-gt-in-visualization",
        action="store_true",
        help="Add GT_label and Correct scalar fields to visualization PLY. Off by default to keep report visuals prediction-only.",
    )
    parser.add_argument(
        "--html-max-points-per-surface",
        type=int,
        default=12000,
        help="Maximum points shown per surface in the HTML plot. Use 0 to keep all points.",
    )
    parser.add_argument(
        "--compare-gt-pred",
        action="store_true",
        help="For PLY inputs with GT_label and Pred_label, compare model predictions against GT.",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("surface_area_gt_pred_comparison.csv"),
        help="CSV output path for --compare-gt-pred.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full surface-area workflow for one input point cloud."""
    args = parse_args()

    xyz, labels = load_point_cloud(args.input, args.label_source, args.label_column)
    if len(xyz) == 0:
        raise ValueError("Input point cloud is empty.")

    estimates, wall_planes = estimate_all_surfaces(xyz, labels, args)
    write_csv(args.output_csv, args.input, estimates, wall_planes)
    write_method_report(args.output_report, args.input, estimates, wall_planes, args)

    gt_labels_for_visualization = None
    if args.compare_gt_pred:
        if args.input.suffix.lower() != ".ply":
            raise ValueError("--compare-gt-pred is only supported for PLY files with GT_label and Pred_label.")

        pair_xyz, gt_labels, pred_labels = load_ply_label_pair(args.input)
        if len(pair_xyz) != len(xyz) or not np.allclose(pair_xyz, xyz):
            raise ValueError("GT/pred comparison loaded a different point layout than the selected input labels.")

        gt_estimates, _ = estimate_all_surfaces(pair_xyz, gt_labels, args)
        pred_estimates, _ = estimate_all_surfaces(pair_xyz, pred_labels, args)
        write_comparison_csv(args.comparison_csv, args.input, gt_estimates, pred_estimates, gt_labels, pred_labels)
        gt_labels_for_visualization = gt_labels

    visualization_mask = None
    if args.output_visualization_ply is not None or args.output_visualization_html is not None:
        visualization_mask = build_measured_surface_mask(xyz, labels, args)

    if args.output_visualization_ply is not None:
        visualization_gt = gt_labels_for_visualization if args.include_gt_in_visualization else None
        write_visualization_ply(
            args.output_visualization_ply,
            xyz,
            labels,
            estimates,
            visualization_gt,
            keep_mask=visualization_mask,
        )
    if args.output_visualization_html is not None:
        write_surface_area_html(
            args.output_visualization_html,
            xyz,
            labels,
            estimates,
            args.html_max_points_per_surface,
            keep_mask=visualization_mask,
        )
    if args.output_area_mesh_ply is not None:
        write_area_mesh_ply(args.output_area_mesh_ply, xyz, labels, args)

    print("Surface area estimation complete.")
    print(f"CSV:    {args.output_csv}")
    print(f"Report: {args.output_report}")
    if args.output_visualization_ply is not None:
        print(f"Visual PLY:  {args.output_visualization_ply}")
    if args.output_visualization_html is not None:
        print(f"Visual HTML: {args.output_visualization_html}")
    if args.output_area_mesh_ply is not None:
        print(f"Area mesh:   {args.output_area_mesh_ply}")
    if args.compare_gt_pred:
        print(f"Compare:{args.comparison_csv}")
    for estimate in estimates:
        warning = f" ({estimate.warning})" if estimate.warning else ""
        print(f"{estimate.name:8s}: {estimate.area_m2:8.3f} m2 from {estimate.point_count} points{warning}")


if __name__ == "__main__":
    main()
