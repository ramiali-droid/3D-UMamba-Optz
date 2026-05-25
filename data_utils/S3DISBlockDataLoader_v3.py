# data_utils/S3DISDataset.py
import os
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

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

# Building groups for S3DIS cross-validation
# BUILDING_GROUPS = {
#    0: [1, 3, 6],  # Test on Building 1
#    1: [2, 4],     # Test on Building 2
#    2: [5]         # Test on Building 3
# }

BUILDING_GROUPS = {
    0: [1],     # Test on Building 1
    1: [2],     # Test on Building 2
    2: [3],     # Test on Building 3
    3: [4],
    4: [5],
    5: [6]
}


class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='../data/s3dis_samples/', fps_n_list=[512, 128, 32], label_number=13, npoints=8192, fold=0):
        super().__init__()
        self.fps_n_list = fps_n_list
        self.npoints = npoints
        self.fold = fold

        # Get all .npy files recursively
        all_files = []
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.npy'):
                    all_files.append(os.path.join(root, file))

        # Parse Area from filename
        file_areas = []
        for f in all_files:
            fname = os.path.basename(f)
            if 'Area_' in fname:
                try:
                    area_str = fname.split('Area_')[1].split('_')[0]
                    area_num = int(area_str)
                    file_areas.append((f, area_num))
                except:
                    file_areas.append((f, -1))
            else:
                file_areas.append((f, -1))

        # Split based on fold
        test_areas = BUILDING_GROUPS[fold]
        if split == 'train':
            selected = [f for f, a in file_areas if a not in test_areas and a != -1]
        elif split == 'test':
            selected = [f for f, a in file_areas if a in test_areas]
        else:
            raise ValueError("split must be 'train' or 'test'")

        rooms_split = selected

        self.sample_points, self.sample_labels = [], []
        self.fps_index_array_list, self.series_idx_arrays_list = [], []
        labelweights = np.zeros(label_number)
        voxel_size = 0.01  # indoor

        for room_path in tqdm(rooms_split, total=len(rooms_split)):
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
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, idx):
        points = self.sample_points[idx]
        labels = self.sample_labels[idx]
        fps_index_array = self.fps_index_array_list[idx]
        series_idx_arrays = self.series_idx_arrays_list[idx]

        return points, labels, fps_index_array, series_idx_arrays

    def __len__(self):
        return len(self.sample_points)