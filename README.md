# 3D-UMamba
3D-UMamba: 3D U-Net with state space model for semantic segmentation of multi-source LiDAR point clouds
[DALES -> S3DIS -> TUB CSE]

This is a Pytorch implementation of 3D-UMamba.

## Abtract

 the state-of-the-art 3D-UMamba framework for in-
door LiDAR semantic segmentation and evaluates the relationship between segmentation
quality and the reliability of structural geometric assessment. A unified preprocessing
pipeline was developed for the DALES, S3DIS, and TU-CSE datasets, and the original
3D-UMamba framework was successfully reproduced and adapted for indoor semantic
segmentation. Transfer learning from the S3DIS benchmark indoor domain to the TU-
CSE residential environment was investigated using zero-shot inference, training from
scratch, and fine-tuning.
The reproduced model achieved competitive benchmark performance on the S3DIS
dataset, obtaining an mIoU of 82.8%, outperforming the methods included in the bench-
mark comparison. On the TU-CSE residential dataset, fine-tuning consistently improved
semantic segmentation performance compared with both zero-shot transfer and train-
ing from scratch, demonstrating the effectiveness of transferring learned representations
from benchmark indoor datasets to residential environments with limited labelled data.
The predicted ceiling, floor, and wall classes were subsequently extracted for automated
surface-area estimation. Validation against manually annotated point clouds and Revit-
derived models showed low relative surface-area errors, while mesh visualisations con-
firmed the successful extraction of the primary structural component


## Install
The latest codes are tested on CUDA11.3 and above, PyTorch 1.10.1 and Python 3.9.

NOTE: Encountered issues with CUDA 11.3 and 11.7 while installing some dependecies and packages. Safe with CUDA 12.3

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
