# Quick Start Guide for CGNet Training

## Setup Instructions

### 1. Download LEVIR-CD Dataset

The LEVIR-CD dataset needs to be downloaded manually from one of these sources:

**Option A: Baidu Cloud (Recommended - Faster)**
- URL: https://pan.baidu.com/s/1VVry18KFl2MSWS6_IOlYRA?pwd=2023
- Password: 2023

**Option B: Official Website**
- URL: https://justchenhao.github.io/LEVIR/

### 2. Extract and Organize

After downloading, extract the zip file and organize as follows:

```
./data/
├── LEVIR-CD/
│   ├── train/
│   │   ├── A/          # Before images (temporal T1)
│   │   ├── B/          # After images (temporal T2)
│   │   └── label/      # Binary change maps (0=no change, 1=change)
│   └── val/
│       ├── A/
│       ├── B/
│       └── label/
```

### 3. Quick Test Run

Once the dataset is in place, run the training:

```bash
# LEVIR-CD dataset (default)
python train_CGNet.py --epoch 5 --batchsize 4 --gpu_id '0' --data_name 'LEVIR' --model_name 'CGNet'

# Other datasets (WHU, SYSU, S2Looking, CDD, DSIFN)
python train_CGNet.py --epoch 5 --batchsize 4 --gpu_id '0' --data_name 'WHU' --model_name 'CGNet'
```

### 4. Training Parameters

```
--epoch         : Number of epochs (default: 50, use 5 for quick test)
--batchsize     : Batch size (default: 8, use 4 for quick test)
--lr            : Learning rate (default: 5e-4)
--gpu_id        : GPU ID to use (0, 1, 2, or 3)
--data_name     : Dataset name (LEVIR, WHU, SYSU, S2Looking, CDD, DSIFN)
--model_name    : Model to train (HCGMNet, CGNet, CGNet_SSM)
--trainsize     : Input image size (default: 256)
```

### 5. Output

Training results will be saved to: `./output/{data_name}/{model_name}/`

- `*_best_iou.pth` - Best model checkpoint (based on IoU)
- TensorBoard logs - Training metrics visualization

## Dataset Structure Details

**Image Channels:**
- Image A & B files: RGB (3 channels)
- Label files: Grayscale (1 channel)
- Pixel values: [0, 255] (automatically normalized)

**Expected File Organization:**
```
train/A/: RGB images from first temporal period
train/B/: RGB images from second temporal period (same spatial location)
train/label/: Binary ground truth (0=no change, 255=change)
```

Files must have matching names across A, B, and label directories.

## Troubleshooting

**"No such file or directory" error:**
- Verify dataset is in `./data/LEVIR-CD/` (or appropriate dataset name)
- Check that train/val subdirectories exist
- Ensure A/, B/, label/ folders contain images

**GPU memory error:**
- Reduce `--batchsize` (e.g., 2 or 4)
- Reduce `--trainsize` (e.g., 128 or 192)
- Use smaller epoch count for testing

**Dataset download issues:**
- Use Baidu Cloud link (more reliable)
- Ensure stable internet connection
- Check file integrity after download

## Next Steps

1. Download LEVIR-CD dataset
2. Extract to `./data/LEVIR-CD/`
3. Run quick test: `python train_CGNet.py --epoch 5 --batchsize 4 --data_name 'LEVIR'`
4. Adjust hyperparameters and train full model
