# data_utils/TUBBlockDataLoader_v3.py
import os
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import random


cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def fps_series_func(points, voxel_indices, samplepoints_list):
    pad_width = points.shape[0]
    points = torch.Tensor(points).float().cuda().unsqueeze(0)
    voxel_indices = torch.Tensor(voxel_indices).float().cuda().unsqueeze(0)
    fps_index_list = []
    series_idx_lists = []
    x1y1z1 = [1, 1, 1]
    x0y1z1 = [-1, 1, 1]
    x1y0z1 = [1, -1, 1]
    x0y0z1 = [-1, -1, 1]
    x1y1z0 = [1, 1, -1]
    x0y1z0 = [-1, 1, -1]
    x1y0z0 = [1, -1, -1]
    x0y0z0 = [-1, -1, -1]
    series_list = []
    series_list.append(x0y0z1)
    series_list.append(x1y1z0)
    for i in range(len(samplepoints_list)):
        S = samplepoints_list[i]
        xyz = points[:, :, :3]
        fps_index = farthest_point_sample(xyz, S)
        points = index_points(points, fps_index)
        new_voxel_indices = index_points(voxel_indices, fps_index).squeeze(0).cpu().data.numpy()
        voxel_indices = index_points(voxel_indices, fps_index)
        fps_index = fps_index.cpu().data.numpy()
        padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(padded_fps_index)
        series_idx_list = []
        for j in range(len(series_list)):
            series = series_list[j]
            new_voxel_indices_ForSeries = new_voxel_indices * series
            sorting_indices = np.expand_dims(np.lexsort((new_voxel_indices_ForSeries[:, 0], new_voxel_indices_ForSeries[:, 1], new_voxel_indices_ForSeries[:, 2])), axis=0)
            padded_sorting_indices = np.expand_dims(np.pad(sorting_indices, ((0, 0), (0, pad_width - sorting_indices.shape[1])), mode='constant'), axis=0)
            series_idx_list.append(padded_sorting_indices)
        series_idx_array = np.concatenate(series_idx_list, axis=1)
        series_idx_lists.append(series_idx_array)
    series_idx_arrays = np.concatenate(series_idx_lists, axis=0)
    fps_index_array = np.vstack(fps_index_list)
    return fps_index_array, series_idx_arrays

def voxelization(points, voxel_size):
    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int32)
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    bounding_box = coord_max - coord_min
    voxel_total = np.ceil(bounding_box[0] * bounding_box[1] * bounding_box[2] / voxel_size**3).astype(np.int32)
    voxel_valid = np.unique(voxel_indices, axis=0)
    return points, voxel_indices, voxel_total, voxel_valid



class TUBDataset(Dataset):
    def __init__(self, split='train', data_root='../data/TUB/', fps_n_list=[512, 128, 32], label_number=8, npoints=8192, fold=0):
        """,
                 augment_scale_anisotropic=True,
                 augment_symmetries=[True, False, False],
                 augment_rotation='vertical',
                 augment_scale_min=0.8,
                 augment_scale_max=1.2,
                 augment_noise=0.001):
        super().__init__()
        self.fps_n_list = fps_n_list
        self.npoints = npoints
        self.fold = fold

        # Augmentation parameters
        self.augment_scale_anisotropic = augment_scale_anisotropic
        self.augment_symmetries = augment_symmetries
        self.augment_rotation = augment_rotation
        self.augment_scale_min = augment_scale_min
        self.augment_scale_max = augment_scale_max
        self.augment_noise = augment_noise
        """
        super().__init__()
        self.fps_n_list = fps_n_list
        self.npoints = npoints
        self.fold = fold
        self.split = split
        # Get all .npy files recursively
        all_files = []
        train_split = ['d1', 'd3', 'd5']
        test_split = ['d2', 'd4']
        for root, dirs, files in os.walk(os.path.join(data_root)):
            for file in files:
                if file.endswith('.npy'):
                    all_files.append(os.path.join(root, file))


        self.sample_points, self.sample_labels = [], []
        self.fps_index_array_list, self.series_idx_arrays_list = [], []
        labelweights = np.zeros(label_number)
        voxel_size = 0.01  # indoor

        selected_files = []
        if self.split == 'train':
            for npy_file in all_files:
                for prefix in train_split:
                    if os.path.basename(npy_file).lower().startswith(prefix):
                        selected_files.append(npy_file)
        else:
            for npy_file in all_files:
                for prefix in test_split:
                    if os.path.basename(npy_file).lower().startswith(prefix):
                        selected_files.append(npy_file)
                
        for room_path in tqdm(selected_files, total=len(selected_files)):
            room_data = np.load(room_path)  # shape (num_blocks, npoints, features+label)
            for i in tqdm(range(room_data.shape[0])):
                block = room_data[i]
                points = block[:, :-1]  # features
                labels = block[:, -1]   # label

                labels = labels - 1 if labels.min() == 1 else labels  # adjust if labels start from 1

                tmp, _ = np.histogram(labels, range(label_number + 1))
                labelweights += tmp

                coor_min = np.amin(points[:, :3], axis=0)
                points[:, 2] = points[:, 2] - coor_min[2]

                array = np.arange(points.shape[0])
                np.random.shuffle(array)
                points = points[array]
                labels = labels[array]

                coords = points.copy()

                coords, voxel_indices, voxel_total, voxel_valid = voxelization(coords[:, :3], voxel_size)

                fps_index_array, series_idx_arrays = fps_series_func(points, voxel_indices, self.fps_n_list)

                self.sample_points.append(points)
                self.sample_labels.append(labels)
                self.fps_index_array_list.append(fps_index_array)
                self.series_idx_arrays_list.append(series_idx_arrays)

        print("Totally {} samples in {} set (fold {}).".format(len(self.sample_points), split, fold))

        self.labelweights = np.ones(label_number)
        if split == 'train':
            eps = 1e-6
            labelweights = labelweights.astype(np.float32)
            labelweights = np.maximum(labelweights, eps)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 2.0)

    def __getitem__(self, idx):
        points = self.sample_points[idx]
        labels = self.sample_labels[idx]
        fps_index_array = self.fps_index_array_list[idx]
        series_idx_arrays = self.series_idx_arrays_list[idx]
        # if self.split == 'train':
        #     points = self._augment_points(points)
        return points, labels, fps_index_array, series_idx_arrays
    """
    
    def _augment_points(self, points):
        ""
        points: (N, C) with C >= 3 (x,y,z first)
        ""
        N, C = points.shape

        # 1. Random scaling (anisotropic or isotropic)
        if self.augment_scale_anisotropic:
            scale = np.random.uniform(self.augment_scale_min, self.augment_scale_max, size=3)
        else:
            scale = np.random.uniform(self.augment_scale_min, self.augment_scale_max, size=1)
            scale = np.repeat(scale, 3)
        points[:, :3] *= scale

        # 2. Random symmetries (flip along axes)
        for i, do_flip in enumerate(self.augment_symmetries):
            if do_flip and random.random() > 0.5:
                points[:, i] *= -1

        # 3. Random rotation (vertical only = around Z-axis)
        if self.augment_rotation == 'vertical':
            theta = np.random.uniform(-np.pi, np.pi)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            rot = np.array([
                [cos_t, -sin_t, 0],
                [sin_t,  cos_t, 0],
                [   0,     0,   1]
            ])
            points[:, :3] = points[:, :3] @ rot.T

        # 4. Gaussian noise
        if self.augment_noise > 0:
            noise = np.random.normal(0, self.augment_noise, size=(N, 3))
            points[:, :3] += noise

        return points
    """

    def __len__(self):
        return len(self.sample_points)