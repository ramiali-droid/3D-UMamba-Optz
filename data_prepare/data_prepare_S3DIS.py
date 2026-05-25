# data_prepare_S3DIS.py
import os
import numpy as np
import glob
from helper_tool import DataProcessing as DP
import time

start = time.time()

# ── CONFIG ───────────────────────────────────────────────────────────────
GRID_SIZE = 0.01                    # indoor: 0.04–0.06m recommended
INPUT_FOLDER = '/home/ramiali/Downloads/Stanford3dDataset_v1.2_Aligned_Version/merged_rooms'
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, f'subsampled_{GRID_SIZE:.3f}')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ───────────────────────────────────────────────────────────────────────────

print(f"Subsampling S3DIS merged rooms to {GRID_SIZE}m voxels...")
print(f"Input folder:  {INPUT_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")

# Find all merged txt files recursively
all_files = glob.glob(os.path.join(INPUT_FOLDER, 'Area_*', '*_merged.txt'))
print(f"Found {len(all_files)} merged room files")

# 8-class remapping (your choice)
remap_dict = {
    0: 0,   # ceiling
    1: 1,   # floor
    2: 2,   # wall
    3: 2,   # beam → wall
    4: 3,   # column → separate (your key decision!)
    5: 4,   # window
    6: 5,   # door
    7: 6,   # table → furniture
    8: 6,   # chair → furniture
    9: 6,   # sofa → furniture
    10: 6,  # bookcase → furniture
    11: 6,  # board → furniture
    12: 7   # clutter
}

for file_path in all_files:
    parts = file_path.split(os.sep)
    area = parts[-2]                    # e.g. Area_5
    room = os.path.basename(file_path)[:-11]  # remove _merged.txt
    print(f"\nProcessing {area}/{room}")

    try:
        data = np.loadtxt(file_path, dtype=np.float32)
        xyz = data[:, 0:3]
        rgb = data[:, 3:6]
        labels = data[:, 6].astype(np.uint8)

        # Apply your 8-class remapping
        labels = np.array([remap_dict.get(int(l), 7) for l in labels], dtype=np.uint8)

        # Subsample
        sub_xyz, sub_rgb, sub_labels, _, _ = DP.grid_sub_sampling(
            points=xyz,
            features=rgb,
            labels=labels,
            grid_size=GRID_SIZE
        )

        # Normalize RGB to [0,1]
        sub_rgb /= 255.0

        # Reshape labels
        sub_labels = sub_labels.reshape(-1, 1).astype(np.uint8)

        # Final output: XYZ + RGB_norm + label (8 classes already applied)
        output = np.concatenate((sub_xyz, sub_rgb, sub_labels), axis=1)

        # Save
        out_file = os.path.join(OUTPUT_FOLDER, f"{area}_{room}_sub.txt")
        np.savetxt(out_file, output, fmt='%.6f %.6f %.6f %.6f %.6f %.6f %d')

        print(f"  Saved: {out_file}  |  points: {len(output):,}")

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")

print(f"\nSubsampling complete! Total time: {(time.time() - start)/60:.1f} minutes")
print(f"Subsampled files with 8-class labels ready in: {OUTPUT_FOLDER}")