import os

dataset_path = 'DALESObjects'
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

# Get .ply files
train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.ply')])
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.ply')])

# Write train.txt
with open(os.path.join(dataset_path, 'train.txt'), 'w') as f:
    for fname in train_files:
        f.write(fname + '\n')

# Write test.txt
with open(os.path.join(dataset_path, 'test.txt'), 'w') as f:
    for fname in test_files:
        f.write(fname + '\n')

print("Created train.txt and test.txt")
print(f"Train files: {len(train_files)}")
print(f"Test files: {len(test_files)}")
