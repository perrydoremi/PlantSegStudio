# Plant Segmentation Studio

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7-green.svg)](https://developer.nvidia.com/cuda-11-7-0-download-archive)

## Overview

Plant Segmentation Studio (PSS) is a unified framework built on [mmdet3d](https://github.com/open-mmlab/mmdetection3d) for training 3D point cloud segmentation models for plant data. 

## 🚀 Key Features

- **All in One, Easy to Use**: Single codebase, simple commands for multiple plant segmentation datasets and algorithms
- **Plant-Specific**: Optimized for plant organ segmentation tasks
- **State-of-the-art Models**: Includes PTv3, OneFormer3d, SoftGroup, and more

## 🧠 Model Zoo

All models below are provided for the five benchmark datasets: **COS**,
**HR3D**, **Pheno4D**, **SoyBeanMVS**, and **SYAUMaize**.


| Semantic Segmentation                         | Instance Segmentation                              |
| --------------------------------------------- | -------------------------------------------------- |
| [PointNet++](configs/Semantic_seg/PointNet2/) | [OneFormer3D](configs/Instance_seg/Oneformer3d/)   |
| [DGCNN](configs/Semantic_seg/DGCNN/)          | [SPVFormer3d](configs/Instance_seg/SPVFormer3d/)   |
| [PAConv](configs/Semantic_seg/PAConv/)        | [Minkformer3d](configs/Instance_seg/Minkformer3d/) |
| [MinkUNet](configs/Semantic_seg/MinkUnet/)    | [PTFormer3d](configs/Instance_seg/PTFormer3d/)     |
| [SPVCNN](configs/Semantic_seg/SPVCNN/)        | [DSPVFormer3d](configs/Instance_seg/DSPVFormer3d/) |
| [SCUNet](configs/Semantic_seg/SCUNet/)        | [SoftGroup](configs/Instance_seg/SoftGroup/)       |
| [PTv3](configs/Semantic_seg/PTv3/)            | [Mink-SG](configs/Instance_seg/Mink-SG/)           |
|                                               | [SPVCNN-SG](configs/Instance_seg/SPVCNN-SG/)       |


## 🛠️ Installation

### Prerequisites

- conda environment manager
- Python 3.8+
- CUDA 11.7

### Step 1: Setup Environment

```bash
git clone https://github.com/perrydoremi/PlantSegStudio.git
cd PlantSegStudio/

conda create --name pss python=3.8 -y
conda activate pss
```

### Step 2: Verify CUDA Installation

Check if CUDA 11.7 is installed, type this in your terminal:

```bash
ls /usr/local/ | grep cuda
```

If `cuda-11.7` is not present, [download and install CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive):

```bash
chmod +x cuda_11.7.0_515.43.04_linux.run #(What you just download)
sudo sh cuda_11.7.0_515.43.04_linux.run
# Disabled the driver installation if your machine already has one
```

Configure CUDA path:

```bash
export PATH=/usr/local/cuda-11.7/bin:$PATH
```

### Step 3: Install Dependencies

#### Core Dependencies

```bash
# Install PyTorch
conda install pytorch==1.13.1 torchvision==0.14.1 cudatoolkit=11.6 -c pytorch -c nvidia

# Install MMDetection3D dependencies
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
mim install mmdet==3.3.0

# Install additional packages
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-1.13.0+cu116.html --no-deps
pip install spconv-cu116
```

#### Minkowski Engine

```bash
conda install openblas-devel -c anaconda
export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include
# Replace `${YOUR_CONDA_ENVS_DIR}` with your actual conda environment path (e.g., `/home/username/anaconda3/envs/pss`)
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --config-settings="--blas_include_dirs=/opt/conda/include" --config-settings="–blas=openblas"
```

> **Troubleshooting**: If you encounter `ImportError: Pytorch not found`, or `ModuleNotFoundError: No module named 'torch'`, add `--no-build-isolation` after `--config-settings="--blas=openblas"`

#### TorchSparse (Required for SPVConv and MinkUNetBackboneV2)

Choose one option to install

**Option 1: Without sudo (Recommended)**

```bash
conda install -c bioconda sparsehash
export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include

pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

**Option 2: With sudo**

```bash
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

### Step 4: Install PSS

```bash
cd PlantSegStudio/
pip install -v -e .
```

The *SoftGroup* extension will be automatically installed. If you want to skip it initially, comment out `ext_modules` in `setup.py`.

> 📖 For detailed installation help, visit [MMDetection3D Documentation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html)

## 📊 Data Preparation

### 1. Configure Dataset Metadata

Edit `tools/plant_info.json` to add your dataset information:

```json
{
  "COS": {
    "feature": 3,                        // Number of features (x, y, z)
    "label": {"trunk": 0, "branch": 1},  // Semantic label mapping
    "instance": ["branch"]                // Instance classes to detect
  }
}
```

### 2. Organize Dataset

Place datasets in `data/plant/`. The file structure should look like this:

```
data
└── plant
    └── COS
        └── COS
            ├── COS1.txt
            ├── COS2.txt
            └── ...
```

### 3. Process Dataset

```bash
# To avoid import error
pip install tensorboardX
```

```bash
# Collect data
python tools/collect_plant_data.py --name COS --data-dir ./data/plant/COS

# Create processed dataset
python tools/create_plant_data.py --root-path ./data/plant/COS \
--out-dir ./data/plant/COS --extra-tag COS --num-data 98  # Total number of samples in COS dataset
```

## 🎯 Training

### Semantic Segmentation

```bash
python tools/train.py configs/Semantic_seg/PointNet2/pointnet2_COS.py \
    --work-dir ./data/plant/COS/Sem_exp/pointnet2_test
```

### Instance Segmentation

```bash
python tools/train.py configs/Instance_seg/SPVFormer3d/SPVFormer3d_COS.py \
    --work-dir ./data/plant/COS/Inst_exp/SPVFormer_test
```

### Visualization with Wandb (Optional)

Configure Wandb in your config file:

```python
# Wandb settings should look like this:
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs={
            'project': f'YourProjectName', # Name your project
            'group': 'YourGroupName', # Name your group
            'name': 'YourExperimentName' # Name your current experiment
        })
]
```

## 🔍 Inference

### Preparation

1. Download pre-trained weights to `checkpoints` folder
2. Disable wandb if previously used for training (Optional):

```bash
wandb offline
wandb disabled
```

### Run Inference

**Semantic Segmentation:**

```bash
python tools/test.py  configs/Semantic_seg/PointNet2/pointnet2_COS.py \
    'checkpoints/best_miou_epoch_192.pth' --save-local --task='lidar_seg' 
```

**Instance Segmentation:**

```bash
python tools/test.py  configs/Instance_seg/SPVFormer3d/SPVFormer3d_COS.py \
    'checkpoints/best_all_ap_epoch_480.pth' --save-local --task='lidar_inst_seg' 
```

> **Note**: Remove `--save-local --task='...'` flags to skip saving inference results

## 📦 Dataset and Model Weights

### Datasets download: [PSS Data Link](https://cornell.box.com/s/6aeclfgl1jrimt8ipfep50sze4rjnigv)

The dataset collection includes:

#### 1. Benchmark Datasets

- Five benchmark datasets used in Paper Table 1

#### 2. Sim2Real Datasets (Synthetic)

- **L-TreeGen:**
  - `G4HComb` - TG4 upper bound subset
  - `Helios03G4H` - TG4 upper bound + VLS03
  - `Helios006G4H` - TG4 upper bound + VLS006
  - `Helios003G4H` - TG4 upper bound + VLS003
- **Deformation:**
  - `G4HDeform` - D4 upper bound subset

### Model Weights download: [PSS Weights Link](https://cornell.box.com/s/9gfqe42lh57kpts4rocog6y9rvkvlpyh)

Weights available for:

1. **Semantic Segmentation**
2. **Instance Segmentation**
3. **Sim2Real Learning** (TG4 and D4 upper bound experiments only)
  - L-TreeGen-based: TG4, TG4+VLS03, TG4+VLS006, TG4+VLS003
  - Deformation-based: D4

> **Note:** All datasets are provided in PSS benchmark format. For meta information, see `[data/meta_info](data/meta_info)`. For original formats, please refer to the source publications.

## 🙏 Acknowledgments

This work builds upon:

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [Pointcept](https://github.com/pointcept/pointcept)
- [Oneformer3D](https://github.com/filaPro/oneformer3d)
- [SoftGroup](https://github.com/thangvubk/SoftGroup)
- [L-TreeGen]()
- [Deformation3D](https://github.com/yangxin6/Deformation3D)

## 📝 Citation

If you use PSS in your research, please cite:

```bibtex
@misc{du2025scalableorganlevel3d,
    title={Towards Scalable Organ-Level 3D Plant Segmentation: 
           Bridging the Data-Algorithm-Computing Gap}, 
    author={Ruiming Du and Guangxun Zhai and Tian Qiu and Yu Jiang},
    year={2025},
    eprint={2509.06329},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2509.06329}
}
```

The following work used PSS:

```bibtex
@article{du2026dynamic,
    title = {Dynamic Sparse Point Voxel Transformer for 3D Point Cloud Instance Segmentation of Dormant Apple Trees},
    journal = {Plant Phenomics},
    pages = {100228},
    year = {2026},
    issn = {2643-6515},
    doi = {https://doi.org/10.1016/j.plaphe.2026.100228},
    url = {https://www.sciencedirect.com/science/article/pii/S2643651526000658}
}
```

```bibtex
@inproceedings{qiu2025joint,
    author={Qiu, Tian and Du, Ruiming and Spine, Nikolai and Cheng, Lailiang and Jiang, Yu},
    booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
    title={Joint 3D Point Cloud Segmentation Using Real-Sim Loop: From Panels to Trees and Branches}, 
    year={2025},
    volume={},
    number={},
    pages={7131-7138},
    keywords={Point cloud compression;Training;Accuracy;Annotations;Scalability;Computational modeling;Field robots;Data models;Digital twins;Robots},
    doi={10.1109/ICRA55743.2025.11128189}
}
```

*For the 3D plant segmentation community*