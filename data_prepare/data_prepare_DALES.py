from os.path import join, exists, dirname, abspath
import os
import numpy as np
import sys

from helper_ply import read_ply
from helper_tool import DataProcessing as DP
import time
start = time.time()

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)


def normalize_scale(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def read_lines(p):
    with open(p, 'r') as f:
        return [line.strip() for line in f.readlines()]


# -------------------- Dataset paths --------------------
grid_size = 0.1  # Voxel size for DALES
dataset_path = '../data/DALESObjects/DALESObjects'
trainset_path = join(dataset_path, 'train')
testset_path = join(dataset_path, 'test')

train_files = read_lines(join(dataset_path, 'train.txt'))
test_files = read_lines(join(dataset_path, 'test.txt'))
print("Total files:", len(train_files) + len(test_files))

# -------------------- Subsample folder --------------------
sub_pc_folder = join(dataset_path, f'input_{grid_size:.3f}')
os.makedirs(sub_pc_folder, exist_ok=True)


# -------------------- Process train files --------------------
for fname in train_files:
    pc_path = join(trainset_path, fname)
    print("Processing:", pc_path)
    file_name = fname.split('/')[-1][:-4]

    pc = read_ply(pc_path)
    xyz = np.vstack((pc['x'], pc['y'], pc['z'])).T.astype(np.float32)
    labels = pc['sem_class'].astype(np.uint8)
    intensity = pc['intensity'].astype(np.float32).reshape(-1,1)

    # Subsample
    sub_xyz, sub_intensity, sub_labels, _, _ = DP.grid_sub_sampling(
        points=xyz, features=intensity, labels=labels, grid_size=grid_size)
    sub_intensity /= 255.0  # normalize intensity
    
    sub_labels = sub_labels.reshape(-1, 1).astype(np.uint8)

    output = np.concatenate((sub_xyz, sub_intensity, sub_labels), axis=-1)
    sub_ply_file = join(sub_pc_folder, file_name + '_train.txt')
    np.savetxt(sub_ply_file, output)
    print(f"Saved subsampled train: {sub_ply_file}, shape: {output.shape}")
    print(f"Time: {time.time() - start:.2f} sec | Points: {output.shape[0]}")


# -------------------- Process test files --------------------
for fname in test_files:
    pc_path = join(testset_path, fname)
    print("Processing:", pc_path)
    file_name = fname.split('/')[-1][:-4]

    pc = read_ply(pc_path)
    xyz = np.vstack((pc['x'], pc['y'], pc['z'])).T.astype(np.float32)
    labels = pc['sem_class'].astype(np.uint8)
    intensity = pc['intensity'].astype(np.float32).reshape(-1,1)

    # Subsample
    sub_xyz, sub_intensity, sub_labels, _, _ = DP.grid_sub_sampling(
        points=xyz, features=intensity, labels=labels, grid_size=grid_size)
    sub_intensity /= 255.0
    
    sub_labels = sub_labels.reshape(-1, 1).astype(np.uint8)

    output = np.concatenate((sub_xyz, sub_intensity, sub_labels), axis=-1)
    sub_ply_file = join(sub_pc_folder, file_name + '_test.txt')
    np.savetxt(sub_ply_file, output)
    print(f"Saved subsampled test: {sub_ply_file}, shape: {output.shape}")
    print(f"Time: {time.time() - start:.2f} sec | Points: {output.shape[0]}")

