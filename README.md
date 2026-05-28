# Prior-Conditioned State Space Models for Change Detection (SPC-SSM)

This repository contains the official implementation of **SPC-SSM** (Selective Prior-Conditioned State Space Models), integrated with a high-performance **CGNet** baseline for Remote Sensing Change Detection (CD). 

By incorporating state-space model structures (selective 2D-Mamba and vectorized 4D-Mamba) guided by change priors, SPC-SSM achieves superior accuracy and spatial context integration for change detection tasks.

---

## 🌟 Features
- **Multiple Model Architectures**:
  - `CGNet` (Original Change Guiding Network)
  - `CGNet_SSM` (2-way Recursive Prior State Space Model)
  - `CGNet_SSM_4dir` (4-way Cross-Scan Prior State Space Model)
  - `CGNet_SSM_selective` (2D Selective State-Space / Mamba logic)
  - `CGNet_SSM_selective_4D` (4D Vectorized Selective State-Space)
- **Offline Data Augmentation**: Easily generate augmented training pairs to improve robustness.
- **Optuna Hyperparameter Search**: Built-in support for automated tuning of learning rates, weight decay, and loss weights.
- **Visualizations**: Automatic saving of predictions and SSM gate activations during training and evaluation.

---

## 📦 Requirements

- **OS**: Windows, Linux, or macOS
- **Python**: 3.8 to 3.11 (3.8 recommended)
- **PyTorch**: `>= 2.0.0` (with CUDA support recommended for training)

### Package Dependencies
All required packages are listed in `requirements.txt`:
```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
tensorboardx>=2.4
numpy
pillow
matplotlib
tqdm
```

---

## 🛠️ Setup & Installation

### Option 1: Quick Install Script (Windows / CMD)
To bypass local Execution Policy restrictions and install dependencies effortlessly, run:
```cmd
python install_deps.py
```
Or simply double-click the **`install.bat`** file in the project root.

### Option 2: Manual Installation via pip
To install manually from your terminal, run:
```bash
pip install -r requirements.txt
```
*Note: For GPU support, make sure to install PyTorch compiled with your specific CUDA version. For instance, for CUDA 11.8:*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📅 Dataset Preparation

### 1. Synthetic Test Dataset (For quick pipeline validation)
You can generate a dummy synthetic dataset consisting of random image pairs to test the training pipeline in seconds:
```bash
python create_test_dataset.py ./data/LEVIR-CD 4 256
```

### 2. Real Datasets (e.g. LEVIR-CD, WHU-CD, CDD)
For full-scale training, structure your dataset folder inside the project directory as follows:
```text
./data/LEVIR-CD/
├── train/
│   ├── A/       # Images before change
│   ├── B/       # Images after change
│   └── label/   # Binary change labels (0/255 or 0/1)
└── val/
    ├── A/
    ├── B/
    └── label/
```

To configure download helpers or guides, run:
```bash
python quickstart.py
```
And select `Option 2` for a detailed dataset download guide.

---

## 🚀 Running the Code

### 1. Pipeline Verification (Quickstart)
To verify your environment is working perfectly, generate synthetic data and run a 2-epoch training test with:
```bash
python quickstart.py
```
And select `Option 4` (or `Option 1` then `Option 3`).

### 2. Training Models
Train any model variant using the main `train_CGNet.py` script. 

#### Train CGNet (Original Baseline)
```bash
python train_CGNet.py --model_type CGNet --data_name LEVIR --epoch 50 --batchsize 8 --gpu_id 0
```

#### Train SPC-SSM (Selective Mamba-based model)
```bash
python train_CGNet.py --model_type CGNet_SSM_selective_4D --data_name LEVIR --epoch 50 --batchsize 8 --gpu_id 0
```

#### Available Parameters:
* `--model_type`: Model architecture (`CGNet`, `CGNet_SSM`, `CGNet_SSM_4dir`, `CGNet_SSM_selective`, `CGNet_SSM_selective_4D`).
* `--epoch`: Number of training epochs (default: `50`).
* `--batchsize`: Batch size (default: `8`).
* `--gpu_id`: The ID of the GPU to use (e.g., `0`, `1`, `2`).
* `--data_name`: Name of the dataset (`LEVIR`, `WHU`, `CDD`, `DSIFN`, `SYSU`, `S2Looking`).
* `--offline_aug_num`: Number of offline augmented image copies to generate before training (e.g., `1` or `2`). Set to `0` to disable.
* `--optuna_trials`: Number of hyperparameter trials to run before full training (set to `>0` to activate).

---

## 📊 Evaluation & Inference

### 1. Model Comparison
To run direct evaluation and compare your trained CGNet and SPC-SSM checkpoints side by side:
```bash
python compare_models.py --model_a_path /path/to/cgnet_weights.pth --model_b_path /path/to/spc_ssm_weights.pth --val_dir ./data/LEVIR-CD/val
```

### 2. Batch Inference
To run change detection inference on a folder of image pairs:
```bash
python batch_inference.py --model_type CGNet_SSM_selective_4D --weight_path /path/to/weights.pth --img_a_dir /path/to/A/ --img_b_dir /path/to/B/ --save_dir ./predictions/
```

### 3. Change Map Visualization
Visualize and analyze generated change maps alongside the inputs and ground truth:
```bash
python visualize_changemap.py --image_a /path/to/A.png --image_b /path/to/B.png --label /path/to/label.png --pred /path/to/pred.png
```
