# CGNet - Quick Test Guide

## ⚡ Fast Setup (2 minutes)

### Option A: Test with Synthetic Dataset (No Download)
```bash
# Step 1: Create test dataset (4 synthetic image pairs)
python create_test_dataset.py

# Step 2: Run training for 2 epochs to test pipeline
python train_CGNet.py --epoch 2 --batchsize 2 --data_name 'LEVIR' --model_name 'CGNet'
```

### Option B: Use Interactive Quick Start
```bash
# One-command quick start wizard
python quickstart.py
```

## 📊 Using Real LEVIR-CD Dataset

### Step 1: Download Dataset
Download from **Baidu Cloud** (recommended):
- URL: https://pan.baidu.com/s/1VVry18KFl2MSWS6_IOlYRA?pwd=2023
- Password: 2023

### Step 2: Extract and Organize
```
./data/LEVIR-CD/
├── train/
│   ├── A/          (before images)
│   ├── B/          (after images)  
│   └── label/      (change maps)
└── val/
    ├── A/
    ├── B/
    └── label/
```

### Step 3: Run Training
```bash
# Quick test (2 epochs)
python train_CGNet.py --epoch 2 --batchsize 4 --data_name 'LEVIR' --model_name 'CGNet'

# Full training (50 epochs)
python train_CGNet.py --epoch 50 --batchsize 8 --data_name 'LEVIR' --model_name 'CGNet'
```

## 🎯 Common Commands

```bash
# Different models
python train_CGNet.py --model_name 'HCGMNet'      # Hierarchical CGM
python train_CGNet.py --model_name 'CGNet'        # Main model (default)
python train_CGNet.py --model_name 'CGNet_SSM'    # SSM variant

# Different datasets
python train_CGNet.py --data_name 'LEVIR'     # LEVIR-CD
python train_CGNet.py --data_name 'WHU'       # WHU-CD
python train_CGNet.py --data_name 'SYSU'      # SYSU-CD
python train_CGNet.py --data_name 'S2Looking' # S2Looking

# GPU selection
python train_CGNet.py --gpu_id '0'  # GPU 0
python train_CGNet.py --gpu_id '1'  # GPU 1

# Adjust batch size
python train_CGNet.py --batchsize 4   # Smaller (less VRAM)
python train_CGNet.py --batchsize 16  # Larger (more VRAM)
```

## 📁 New Files Created

| File | Purpose |
|------|---------|
| `create_test_dataset.py` | Generate synthetic test dataset |
| `quickstart.py` | Interactive setup wizard |
| `SETUP_DATASET.md` | Detailed dataset setup guide |
| `train_CGNet.py` | **Modified** - uses `./data/` for local datasets |

## 🔧 Modified Files

### `train_CGNet.py`
**Change:** Updated hardcoded absolute dataset paths to relative local paths

**Before:**
```python
opt.train_root = '/data/chengxi.han/data/LEVIR CD Dataset256/train/'
opt.val_root = '/data/chengxi.han/data/LEVIR CD Dataset256/val/'
```

**After:**
```python
opt.train_root = './data/LEVIR-CD/train/'
opt.val_root = './data/LEVIR-CD/val/'
```

Now uses configurable `./data/` directory that works on any machine!

## ✅ Verification Checklist

- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] CUDA 11.x and cuDNN installed
- [ ] GPU available (`nvidia-smi` shows your GPU)
- [ ] Test dataset created OR real dataset downloaded
- [ ] Dataset structure matches expected format
- [ ] `train_CGNet.py` runs without import errors

## 🐛 Troubleshooting

**Error: "No such file or directory" for dataset**
- Solution: Run `python create_test_dataset.py` or download real dataset

**Error: "CUDA out of memory"**
- Solution: Reduce `--batchsize` (e.g., 2 or 4)

**Error: "Module not found"**
- Solution: Verify network/*.py and utils/*.py files exist
- Solution: Run `pip install -r requirements.txt`

**Slow training / low GPU usage**
- Solution: Increase `--batchsize` if VRAM allows
- Solution: Use `num_workers=4` or higher in data_loader.py

## 📊 Output

Training results saved to: `./output/{dataset}/{model}/`

Files:
- `*_best_iou.pth` - Best model checkpoint
- TensorBoard logs in runs/

View tensorboard logs:
```bash
tensorboard --logdir=runs/
```

## 🚀 Next Steps

1. **Test with synthetic data** (2 min):
   ```bash
   python create_test_dataset.py && python train_CGNet.py --epoch 2 --batchsize 2
   ```

2. **Download LEVIR-CD** and run full training:
   ```bash
   python train_CGNet.py --epoch 50 --batchsize 8 --data_name 'LEVIR'
   ```

3. **Test with other datasets**:
   ```bash
   python train_CGNet.py --data_name 'WHU' --epoch 10
   ```

## 📖 References

- [LEVIR-CD Dataset](https://justchenhao.github.io/LEVIR/)
- [Original CGNet Paper](https://ieeexplore.ieee.org/document/10234560)
- [GitHub Repository](https://github.com/ChengxiHAN/CGNet-CD)

---

**Ready to train?** Run: `python create_test_dataset.py && python train_CGNet.py --epoch 2 --batchsize 2`
