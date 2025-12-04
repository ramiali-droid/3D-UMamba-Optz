import os
import sys
import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import colorsys
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))


# ---------------------------
# Python fallback for C++ subsampling
# ---------------------------
"""
class DummyCppSubsampling:
    @staticmethod
    def compute(points, features=None, classes=None, sampleDl=0.1, verbose=0):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        down_pcd = pcd.voxel_down_sample(voxel_size=sampleDl)
        down_points = np.asarray(down_pcd.points)

        if features is not None or classes is not None:
            print("[Warning] cpp_subsampling features/classes ignored in Python fallback.")
        return down_points

cpp_subsampling = DummyCppSubsampling()
"""
# ---------------------------
# FIXED Python fallback for C++ subsampling (Preserves features & labels)
# ---------------------------
# ---------------------------
# FINAL: Open3D voxel_down_sample_and_trace (CPU + GPU Compatible)
# ---------------------------
class FixedCppSubsampling:
    @staticmethod
    def compute(points, features=None, classes=None, sampleDl=0.1, verbose=0):
        import time
        start = time.time()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Compute bounds
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)

        # Check if Open3D is using CUDA (GPU)
        try:
            # Try GPU call
            down_pcd, voxel_index, point_to_voxel = pcd.voxel_down_sample_and_trace(
                voxel_size=sampleDl,
                min_bound=min_bound,
                max_bound=max_bound
            )
            if verbose:
                print(f"[VOXEL+TRACE GPU] {len(points):,} → {len(down_pcd.points):,} points")
        except:
            # Fall back to CPU (no bounds needed)
            down_pcd = pcd.voxel_down_sample(voxel_size=sampleDl)
            sub_points = np.asarray(down_pcd.points)
            n_sub = len(sub_points)
            if n_sub == 0:
                return np.zeros((0,3)), np.zeros((0,1)), np.zeros((0,), dtype=np.int32), [], []

            # CPU: Use KDTree for mapping
            from scipy.spatial import KDTree
            tree = KDTree(points)
            sub_feats = np.zeros((n_sub, features.shape[1])) if features is not None else None
            sub_classes = np.zeros(n_sub, dtype=np.int32) if classes is not None else None

            for i, pt in enumerate(sub_points):
                _, idx = tree.query(pt)
                if sub_feats is not None:
                    sub_feats[i] = features[idx]
                if sub_classes is not None:
                    sub_classes[i] = classes[idx]

            unique_map = np.unique(sub_classes) if classes is not None else np.array([])
            sub_idx = np.arange(n_sub)
            if verbose:
                print(f"[VOXEL+KDTree CPU] {len(points):,} → {n_sub:,} points in {time.time()-start:.2f}s")
            return sub_points, sub_feats, sub_classes, sub_idx, unique_map

        # GPU path succeeded
        sub_points = np.asarray(down_pcd.points)
        n_sub = len(sub_points)

        if n_sub == 0:
            return np.zeros((0,3)), np.zeros((0,1)), np.zeros((0,), dtype=np.int32), [], []

        sub_feats = np.zeros((n_sub, features.shape[1])) if features is not None else None
        sub_classes = np.zeros(n_sub, dtype=np.int32) if classes is not None else None
        filled = np.zeros(n_sub, dtype=bool)

        # Use point_to_voxel (list of lists)
        for i, voxel_list in enumerate(point_to_voxel):
            if len(voxel_list) == 0:
                continue
            orig_idx = voxel_list[0]  # Take first point in voxel
            if sub_feats is not None and not filled[i]:
                sub_feats[i] = features[orig_idx]
            if sub_classes is not None and not filled[i]:
                sub_classes[i] = classes[orig_idx]
            filled[i] = True

        # Fallback for empty voxels
        if not np.all(filled):
            from scipy.spatial import KDTree
            tree = KDTree(points)
            for i in range(n_sub):
                if not filled[i]:
                    _, idx = tree.query(sub_points[i])
                    if sub_feats is not None:
                        sub_feats[i] = features[idx]
                    if sub_classes is not None:
                        sub_classes[i] = classes[idx]

        unique_map = np.unique(sub_classes[sub_classes > 0]) if classes is not None else np.array([])
        sub_idx = np.arange(n_sub)

        if verbose:
            print(f"[VOXEL+TRACE GPU] {len(points):,} → {n_sub:,} points in {time.time()-start:.2f}s")

        return sub_points, sub_feats, sub_classes, sub_idx, unique_map

# === OVERRIDE ===
cpp_subsampling = FixedCppSubsampling()
print("[SUCCESS] Using Open3D voxel_down_sample_and_trace (CPU/GPU auto-detect)")
# ---------------------------
# Python fallback for NearestNeighbors
# ---------------------------
class DummyNearestNeighbors:
    @staticmethod
    def compute(point_cloud, query_cloud, k=1):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(point_cloud)
        distances, indices = nbrs.kneighbors(query_cloud)
        return indices, distances

nearest_neighbors = DummyNearestNeighbors()


# ---------------------------
# DALES Dataset Config
# ---------------------------
class ConfigDALES:
    k_n = 16
    num_layers = 5
    num_points = 32768
    num_classes = 8  # Set this to actual DALES classes
    sub_grid_size = 0.1  # matches voxel size
    use_rgb = False
    use_intensity = True

    batch_size = 4
    val_batch_size = 4
    train_steps = 1000
    val_steps = 25

    sub_sampling_ratio = [4, 4, 4, 4, 2]
    d_out = [16, 64, 128, 256, 512]

    radius = [0.1, 0.3, 2]
    radius_npoints = [4096, 4096, 8192]

    noise_init = 3.5
    max_epoch = 100
    learning_rate = 1e-2
    lr_decays = {i: 0.95 for i in range(0, 500)}

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001


# ---------------------------
# DataProcessing class
# ---------------------------
class DataProcessing:
    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size, verbose=verbose)

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        neighbor_idx, _ = nearest_neighbors.compute(support_pts, query_pts, k)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def get_class_weights(dataset_name):
        if dataset_name == "DALESOjects":
            # Replace with actual DALES class counts
            num_per_class = np.array([1000, 2000, 1500, 1800, 500, 1200, 800, 300, 600, 700, 400, 900, 100])
        else:
            num_per_class = np.array([1])
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_aug = np.concatenate([xyz, xyz[dup]], 0)
        color_aug = np.concatenate([color, color[dup]], 0)
        idx_aug = np.concatenate([idx, idx[dup]], 0)
        label_aug = np.concatenate([labels, labels[dup]], 0)
        return xyz_aug, color_aug, idx_aug, label_aug


# ---------------------------
# Plotting utilities
# ---------------------------
class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] > 3:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.0)
        o3d.visualization.draw_geometries([pc])

