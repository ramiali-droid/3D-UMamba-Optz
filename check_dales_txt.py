import numpy as np
import glob
import os

data_dir = "./data/DALESObjects/DALESObjects/input_0.100"   # change if needed
txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

print(f"Found {len(txt_files)} .txt files")

for i, f in enumerate(txt_files):
    data = np.loadtxt(f)
    
    if data.shape[0] != 8192:
        print(f"ERROR: {f} has {data.shape[0]} points (expected 8192)")
    if data.shape[1] != 11:
        print(f"ERROR: {f} has {data.shape[1]} columns (expected 11)")
        print("Columns meaning: X Y Z R G B Intensity Nx Ny Nz Label")
    
    # Optional: check label range (DALES: 0–8, sometimes 255=unlabeled)
    labels = data[:, -1].astype(int)
    unique_labels = np.unique(labels)
    if not all(l in range(0, 9) or l == 255 for l in unique_labels):
        print(f"Warning: {f} has strange labels: {unique_labels}")
    
    if i % 100 == 0:
        print(f"Checked {i+1}/{len(txt_files)} files...")

print("All files are CORRECT!" if i == len(txt_files)-1 else "Some files have issues!")
