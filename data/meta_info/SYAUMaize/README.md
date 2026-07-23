Paper Link: https://doi.org/10.1016/j.isprsjprs.2024.03.025
# Introduction
The dataset used in the paper "Maize stem–leaf segmentation framework based on deformable point clouds" consists of 428 maize plants from five different maize varieties, captured at various stages of growth. The plants were scanned using a 3D laser scanner to create point clouds. The dataset was constructed from plants grown in field experiments at Shenyang Agricultural University across three years (2019, 2020, and 2021). These point clouds were manually cleaned to remove noise and extraneous elements using CloudCompare Stereo software. The point clouds were downsampled to 20,480 points to preserve the morphological characteristics while minimizing the data scale.

The labeling of the dataset was carried out using a custom toolkit called Label3DMaize. This tool allows for semi-automatic segmentation and annotation of maize point clouds. Users interactively select key areas in the point cloud, and the software automatically segments these areas using unsupervised algorithms. After segmentation, each organ (stem, leaf) is assigned an integer label, with stems labeled as 0 and leaves labeled sequentially from the bottom to the top of the plant.

Regarding the train-test split, the authors used a subset of the dataset for training and another subset for testing the models. Specifically, two labeled data items were randomly chosen from the dataset for each leaf number, resulting in 22 labeled data items in total. These labeled data items were used to generate multiple deformed point clouds for training the segmentation models (PointNet++ and HAIS). The models were then tested on a dataset comprising 406 real plant data points, demonstrating the efficiency of the deformation and segmentation methods.

## Labeling
**Dataset Description**: 0 represents the stem, while 1,2,3, etc., represent leaves, enumerated from the base upwards.


**Data Distribution**


| Number of leaves | Number of samples | XY335 data bulk | LD145 data bulk | LD502 data bulk | LD586 data bulk | LD1281 data bulk | Number of incompletedata | Number of completedata |
| ---------------- | ----------------- | --------------- | --------------- | --------------- | --------------- | ---------------- | ------------------------ | ---------------------- |
| 2                | 12                | 12              | 0               | 0               | 0               | 0                | 8                        | 4                      |
| 3                | 17                | 17              | 0               | 0               | 0               | 0                | 7                        | 10                     |
| 4                | 38                | 31              | 3               | 0               | 1               | 3                | 10                       | 27                     |
| 5                | 98                | 85              | 2               | 3               | 4               | 4                | 28                       | 70                     |
| 6                | 100               | 86              | 4               | 3               | 5               | 2                | 25                       | 75                     |
| 7                | 58                | 44              | 3               | 3               | 4               | 4                | 12                       | 47                     |
| 8                | 36                | 33              | 1               | 0               | 1               | 1                | 6                        | 30                     |
| 9                | 27                | 20              | 1               | 5               | 1               | 0                | 7                        | 20                     |
| 10               | 24                | 13              | 3               | 4               | 3               | 1                | 5                        | 19                     |
| 11               | 12                | 6               | 2               | 2               | 0               | 2                | 5                        | 7                      |
| 12               | 6                 | 2               | 0               | 0               | 3               | 1                | 4                        | 2                      |

# Model training
We tested the above methods using a workstation equipped with a 12th Gen Intel(R) Core(TM) i9-12900 K processor, 128-GB RAM and an RTX
A6000 GPU. Approximately 2.8 s were required for our physics-based deformation framework to deform a plant point cloud.
We trained the PointNet++ model with 100 epochs, each encompassing 198 batches with sizes of 100, and downsampled the plant point-cloud
data to 4096 points specifically for this model. We adopted the default hyperparameters proposed by Qi et al.(2017) and assessed semantic seg­
mentation precision using the mean intersection-over-union (mIoU) metric, as recommended in their research. In contrast, the HAIS model was
trained on a solitary GPU configuring a batch size of 100 over 100 K iterations. We utilised the default hyperparameters of Chen et al.(2021) and
evaluated the model’s performance using the mean average precision (mAP) metric per their methodology. We normalised our data and adjusted the
voxel size to 0.003 before introducing it into the model and maintained all other parameters at the default HAIS configuration.

# PSS settings
x, y, z, sem, ins \
sem: 0: stem; 1: leaf \
ins: 0: stem; 1,2,3,...: leaf \
Min Bounding Box: [16.126392, 15.014052, 54.105585] \
Max Bounding Box: [1372.037736, 1214.928592, 1949.81959]

## Train-test split (SYAUMaize)
Train: 80-428 (XY335) \
Test: 1-79 (LD145, LD502, LD586, LD1281)
