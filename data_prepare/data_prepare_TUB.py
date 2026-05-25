# data_prepare_TUB.py
import os
import numpy as np
import glob
from helper_tool import DataProcessing as DP
import time

start = time.time()

# ── CONFIG ───────────────────────────────────────────────────────────────
GRID_SIZE = 0.01                    # indoor: 0.04–0.06m recommended
INPUT_FOLDER = '/home/ramiali/3dumamba/data/TUB/combined/crt/D5_rooms_crt/filtered/'
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, f'subsampled_{GRID_SIZE:.3f}')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ───────────────────────────────────────────────────────────────────────────

print(f"Subsampling TUB pointcloud to {GRID_SIZE}m voxels...")
print(f"Input folder:  {INPUT_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")

# Find all txt files recursively
all_files = glob.glob(os.path.join(INPUT_FOLDER, '*.txt'))
print(f"Found {len(all_files)} files")

# 8-class remapping
remap_dict = {
    0: 0,   # ceiling
    1: 1,   # floor
    2: 2,   # wall
    3: 3,   # column
    4: 4,   # window
    5: 5,   # door
    6: 6,   # furniture
    7: 7    # clutter
}

for file_path in all_files:
    
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
        #sub_rgb /= 255.0

        # Reshape labels
        sub_labels = sub_labels.reshape(-1, 1).astype(np.uint8)

        # Final output: XYZ + RGB_norm + label (8 classes already applied)
        output = np.concatenate((sub_xyz, sub_rgb, sub_labels), axis=1)

        # Output filename: original_name_sub.txt
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_sub.txt")
        
        # Save
        np.savetxt(out_file, output, fmt='%.6f %.6f %.6f %.6f %.6f %.6f %d')

        print(f"  Saved: {out_file}  |  points: {len(output):,}")

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")

print(f"\nSubsampling complete! Total time: {(time.time() - start)/60:.1f} minutes")
print(f"Subsampled files with 8-class labels ready in: {OUTPUT_FOLDER}")