import math
import numpy as np
import glob
import os
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import datetime
import scipy.sparse as sp

dataset_path = '/home/ramiali/~umamba/data/DALESObjects/DALESObjects/input_0.100/'
file_paths = glob.glob(dataset_path + '*.txt')  # collecting all files

def norm_and_surface_variance(pc, neighbor_radius=1.0):
    """
    Vectorized replacement matching the original function's outputs exactly:
    - input `pc` shape (N, 11)
    - returns array shape (N, 15): [points (first 10 cols), nx,ny,nz, sv, label]
      where `points = pc[:, :-1]` and `label = pc[:, -1:]`
    """
    # Keep identical names/behavior as original
    points = pc[:, :-1]            # first 10 columns (as original code did)
    labels = pc[:, -1:].reshape(-1, 1)   # shape (N,1)
    point_cloud = pc[:, 4:7]       # ax, ay, az used for neighborhood PCA

    N = point_cloud.shape[0]
    if N == 0:
        return np.zeros((0, 15))

    # Build radius neighbor connectivity (sparse)
    nbrs = NearestNeighbors(radius=neighbor_radius, algorithm='ball_tree').fit(point_cloud)
    A = nbrs.radius_neighbors_graph(point_cloud, mode='connectivity').astype(float)  # shape (N,N) sparse

    # neighbor counts
    counts = np.array(A.sum(axis=1)).reshape(-1)   # (N,)
    # avoid divide by zero
    counts_safe = counts.copy()
    counts_safe[counts_safe == 0] = 1.0

    # sum of neighbor positions
    sum_p = A.dot(point_cloud)    # shape (N,3)
    mean = sum_p / counts_safe[:, None]   # shape (N,3)

    # compute second moments: px^2, py^2, pz^2, px*py, px*pz, py*pz for all points
    px = point_cloud[:, 0]
    py = point_cloud[:, 1]
    pz = point_cloud[:, 2]
    cols = np.stack([px * px, py * py, pz * pz, px * py, px * pz, py * pz], axis=1)  # (N,6)

    S2 = A.dot(cols)  # (N,6)
    E_xx = S2 / counts_safe[:, None]  # (N,6)

    mx = mean[:, 0]; my = mean[:, 1]; mz = mean[:, 2]
    cov_xx = E_xx[:, 0] - mx * mx
    cov_yy = E_xx[:, 1] - my * my
    cov_zz = E_xx[:, 2] - mz * mz
    cov_xy = E_xx[:, 3] - mx * my
    cov_xz = E_xx[:, 4] - mx * mz
    cov_yz = E_xx[:, 5] - my * mz

    Cov = np.zeros((N, 3, 3), dtype=np.float64)
    Cov[:, 0, 0] = cov_xx
    Cov[:, 1, 1] = cov_yy
    Cov[:, 2, 2] = cov_zz
    Cov[:, 0, 1] = cov_xy
    Cov[:, 1, 0] = cov_xy
    Cov[:, 0, 2] = cov_xz
    Cov[:, 2, 0] = cov_xz
    Cov[:, 1, 2] = cov_yz
    Cov[:, 2, 1] = cov_yz

    # Masks
    small_mask = counts <= 2      # neighborhoods with <=2 points: original fallback
    valid_mask = ~small_mask
    local_features = np.zeros((N, 3), dtype=np.float64)
    norm_features = np.zeros((N, 3), dtype=np.float64)
    surface_variance = np.zeros((N, 1), dtype=np.float64)

    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size > 0:
        Cov_valid = Cov[valid_idx]
        # symmetrize for numerical stability
        Cov_valid = 0.5 * (Cov_valid + np.transpose(Cov_valid, (0, 2, 1)))
        # eigen-decomposition (ascending eigenvalues)
        e_vals, e_vecs = np.linalg.eigh(Cov_valid)   # shapes (M,3), (M,3,3)
        # clip tiny negatives
        e_vals = np.clip(e_vals, a_min=0.0, a_max=None)
        sum_e = np.sum(e_vals, axis=1)
        sum_e_safe = sum_e.copy()
        sum_e_safe[sum_e_safe == 0] = 1.0

        # explained variance ratios in descending order to match sklearn.PCA().explained_variance_ratio_
        desc_evals = e_vals[:, ::-1]   # largest -> smallest
        desc_sum = np.sum(desc_evals, axis=1)
        desc_sum_safe = desc_sum.copy()
        desc_sum_safe[desc_sum_safe == 0] = 1.0
        local_features_valid = desc_evals / desc_sum_safe[:, None]   # (M,3)

        # smallest explained variance ratio: original code used local_features[:,2] / sum == smallest ratio
        smallest_ratio = e_vals[:, 0] / sum_e_safe    # since e_vals ascending, index 0 is smallest

        # PCA.components_[2] corresponds to principal component vector of the smallest eigenvalue.
        # eigh returns eigenvectors such that e_vecs[:,:,k] corresponds to e_vals[:,k]
        norm_valid = e_vecs[:, :, 0]   # eigenvector for smallest eigenvalue (M,3)

        # assign
        local_features[valid_idx, :] = local_features_valid
        norm_features[valid_idx, :] = norm_valid
        surface_variance[valid_idx, 0] = smallest_ratio

    # small neighborhoods (<=2) follow original defaults
    if np.any(small_mask):
        local_features[small_mask, :] = np.array([1.0, 0.0, 0.0])
        norm_features[small_mask, :] = np.array([0.0, 0.0, 0.0])
        surface_variance[small_mask, 0] = 0.0

    # IMPORTANT: original concatenation order:
    # output = [points (pc[:,:-1], shape 10), norm_features (3), surface_variance (1), labels (1)] -> total 15
    out = np.concatenate([points[:, :7], norm_features, surface_variance, labels], axis=-1)
    return out


def coordinate_normalize(input_data):
    input_xyz = input_data[:, :, :3]
    max_xyz = np.max(input_xyz, axis=1)
    min_xyz = np.min(input_xyz, axis=1)
    mean_xyz = (max_xyz + min_xyz) / 2
    mean_xyz = np.expand_dims(mean_xyz, axis=1)
    div = np.max(input_xyz - mean_xyz, axis=1)
    maxv = np.max(div, axis=-1)
    div = np.expand_dims(div, axis=1)
    maxv = np.expand_dims(maxv, axis=1)
    maxv = np.expand_dims(maxv, axis=1)
    new_xyz = (input_xyz - mean_xyz) / maxv
    input_data[:, :, :3] = new_xyz
    return input_data


block_size_list = [20]

for file_path in tqdm(file_paths):
    file_name = os.path.basename(file_path)[:-4]

    points = np.loadtxt(file_path)

    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    x = coord_max[0] - coord_min[0]
    y = coord_max[1] - coord_min[1]

    for size_ in range(len(block_size_list)):
        block_size = block_size_list[size_]
        min_npoints = 4096

        data_folder = dataset_path + 'Block_s' + str(block_size) + '_min_final_' + str(min_npoints) + '_norm_enhance/'
        data_root = data_folder
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        center_x = []
        if x < block_size:
            center_nx = 1
            center_x.append(x / 2)
        else:
            center_nx = math.ceil(x / block_size)
            center_x.append(block_size / 2)
            center_x.append(x - block_size / 2)
            center_x_length = x - block_size

            for i in range(1, center_nx - 1):
                center_x.append(block_size / 2 + center_x_length / (center_nx - 1) * i)
        center_x = sorted(center_x) + coord_min[0]

        center_y = []
        if y < block_size:
            center_ny = 1
            center_y.append(y / 2)
        else:
            center_ny = math.ceil(y / block_size)
            center_y.append(block_size / 2)
            center_y.append(y - block_size / 2)
            center_y_length = y - block_size

            for i in range(1, center_ny - 1):
                center_y.append(block_size / 2 + center_y_length / (center_ny - 1) * i)
        center_y = sorted(center_y) + coord_min[1]

        coordinates = [[xi, yi] for xi in center_x for yi in center_y]
        data_npy = []

        for i in range(len(coordinates)):
            block_x_min = coordinates[i][0] - block_size / 2
            block_x_max = coordinates[i][0] + block_size / 2
            block_y_min = coordinates[i][1] - block_size / 2
            block_y_max = coordinates[i][1] + block_size / 2
            point_idxs = np.where((points[:, 0] >= block_x_min) & (points[:, 0] <= block_x_max) &
                                  (points[:, 1] >= block_y_min) & (points[:, 1] <= block_y_max))[0]

            if point_idxs.size == 0:
                print('0 Existing!!!')
                continue

            if point_idxs.size >= min_npoints:
                selected_point_idxs = np.random.choice(point_idxs.size, min_npoints, replace=False)
            else:
                selected_point_idxs = np.random.choice(point_idxs.size, min_npoints, replace=True)

            selected_point_idxs = point_idxs[selected_point_idxs]
            selected_points = points[selected_point_idxs, :]

            current_points = np.zeros((selected_point_idxs.size, 11))  # num_point * 11 as before
            current_points[:, :4] = selected_points[:, :4]   # x,y,z,intensity
            current_points[:, 4:7] = selected_points[:, :3]  # extra coords (ax,ay,az)
            current_points[:, 7] = 0                         # placeholder nx
            current_points[:, 8] = 0                         # placeholder ny
            current_points[:, 9] = 0                         # placeholder nz
            current_points[:, 10] = selected_points[:, 4]   # label


            current_points = norm_and_surface_variance(current_points)

            current_points = np.expand_dims(current_points, axis=0)
            current_points = coordinate_normalize(current_points)  # rx ry rz intensity ax ay az nx ny nz sv label
            data_npy.append(current_points)

        file_save_path = data_root + file_name + '.npy'
        if len(data_npy) == 0:
            print('No blocks to save for this file')
            continue
        data_all = np.concatenate(data_npy, axis=0)
        np.save(file_save_path, data_all)


