# 3D-UMamba
3D-UMamba: 3D U-Net with state space model for semantic segmentation of multi-source LiDAR point clouds

This is a Pytorch implementation of 3D-UMamba.

## Abtract

Segmentation of point clouds is foundational to numerous remote sensing applications. Recently, the development of Transformers has further improved segmentation techniques thanks to their great long-range context modeling capability. However, Transformers have quadratic complexity in inference time and memory, which both limits the input size and poses a strict hardware requirement. This paper presents a novel 3D-UMamba network with linear complexity, which is the earliest to introduce the Selective State Space Model (i.e., Mamba) to multi-source LiDAR point cloud processing. 3D-UMamba integrates Mamba into the classic U-Net architecture, presenting outstanding global context modeling with high efficiency and achieving an effective combination of local and global information. In addition, we propose a simple yet efficient 3D-token serialization approach (Voxel-based Token Serialization, i.e., VTS) for Mamba, where the Bi-Scanning strategy enables the model to collect features from all input points in different directions effectively. The performance of 3D-UMamba on three challenging LiDAR point cloud datasets (airborne MultiSpectral LiDAR (MS-LiDAR), aerial DALES, and vehicle-mounted Toronto-3D) demonstrated its superiority in multi-source LiDAR point cloud semantic segmentation, as well as the strong adaptability of Mamba to different types of LiDAR data, exceeding current state-of-the-art models. Ablation studies demonstrated the higher efficiency and lower memory costs of 3D-UMamba than its Transformer-based counterparts.


## Install
The latest codes are tested on CUDA11.3 and above, PyTorch 1.10.1 and Python 3.9.

NOTE: Encountered issues with CUDA 11.3 and 11.7 while installing some dependacies and packages. Safe with CUDA 12.3

We do not need to compile mamba or cpp_wrapper from the previous model
For mamba, it is installed through 3dumamba_environment.yml [mamba-ssm==1.2.0.post1]



## Data Preparation
Download DALES 
First you need to fill the contact form wait for the download link. Download the original annotated files (DALESObjects.tar.gz), it contains DALESObjects in .ply files
(https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php) and save in "data/".

## Run
1. Data preprocessing in the folder of data_prepare

```
python data_prepare_DALES.py

python DALES_sample_generation_vectorized.py

```
#data_prepare_DALES.py -> Here, .ply files are converted to .txt files

#DALES_sample_generation_vectorized.py --. is much faster than the previous version [it used to take 7hrs, with this in 1hr]
.txt files are subsampled and converted to .npy files. Which is used in the train._DALES.py

2. Model Training


```
python train_DALES.py --model mamba_msg --log_dir 3dumamba --learning_rate 0.01 --batch_size 4 --optimizer SGD --epoch 120 --gpu 0 --npoint 8192 --weighted_loss True
```

Note: Saving the pretrained model (https://drive.google.com/file/d/1U6TJYdRg77uNhLRPg4WJ9UeTx__y3aIz/view?usp=drive_link) in log/dales_seg/3dumamba/checkpoints, to get better results.
#### I could not find such pretrained model, but it is totally ok to train from scratch and get the same result.


## Acknowledgement

We would like to express our sincere gratitude to PointMamba (https://github.com/LMD0311/PointMamba) and PVCNN (https://github.com/mit-han-lab/pvcnn/tree/master)  for their valuable work on, which has significantly contributed to the development of this project.
