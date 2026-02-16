# Quick Start Guide: Using CGNet_SSM

## Overview

CGNet_SSM is a modified version of CGNet that replaces the Change Guiding Module (CGM) with PriorConditionedSSM. This provides:

- **Softer guidance**: Additive prior injection instead of multiplicative gating
- **Better preservation**: Weak change signals are not suppressed
- **Spatial coherence**: 2D state-space scanning for context propagation
- **Efficiency**: Fewer parameters and faster computation

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter PyTorch version issues, install manually:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision

# Other dependencies
pip install opencv-python tensorboardx numpy pillow matplotlib tqdm
```

### 2. Verify Installation

```bash
python test_ssm_simple.py
```

Expected output:
```
✅ ALL TESTS PASSED!
The PriorConditionedSSM module is ready to use!
```

---

## Training

### Basic Training Command

```bash
python train_CGNet.py \
    --epoch 50 \
    --batchsize 8 \
    --gpu_id '0' \
    --data_name 'WHU' \
    --model_name 'CGNet_SSM'
```

### Available Models

- `CGNet`: Original model with CGM
- `HCGMNet`: Hierarchical CGM variant
- `CGNet_SSM`: **New model with PriorConditionedSSM** ⭐

### Training on Different Datasets

```bash
# LEVIR-CD
python train_CGNet.py --data_name 'LEVIR' --model_name 'CGNet_SSM' --epoch 100 --batchsize 8

# WHU-CD
python train_CGNet.py --data_name 'WHU' --model_name 'CGNet_SSM' --epoch 50 --batchsize 8

# SYSU-CD
python train_CGNet.py --data_name 'SYSU' --model_name 'CGNet_SSM' --epoch 50 --batchsize 8

# S2Looking
python train_CGNet.py --data_name 'S2Looking' --model_name 'CGNet_SSM' --epoch 50 --batchsize 8

# CDD
python train_CGNet.py --data_name 'CDD' --model_name 'CGNet_SSM' --epoch 50 --batchsize 8

# DSIFN
python train_CGNet.py --data_name 'DSIFN' --model_name 'CGNet_SSM' --epoch 50 --batchsize 8
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epoch` | 50 | Number of training epochs |
| `--batchsize` | 8 | Batch size (adjust based on GPU memory) |
| `--lr` | 1e-4 | Initial learning rate |
| `--trainsize` | 256 | Input image size (256×256) |
| `--gpu_id` | '0' | GPU device ID |
| `--data_name` | Required | Dataset name |
| `--model_name` | Required | Model architecture |

### Advanced Training Options

```bash
# Multi-GPU training
python train_CGNet.py \
    --gpu_id '0,1' \
    --batchsize 16 \
    --data_name 'LEVIR' \
    --model_name 'CGNet_SSM'

# Resume from checkpoint
python train_CGNet.py \
    --data_name 'WHU' \
    --model_name 'CGNet_SSM' \
    --load './output/WHU/CGNet_SSM_checkpoint.pth'
```

---

## Testing/Inference

### Basic Testing Command

```bash
python test.py \
    --gpu_id '0' \
    --data_name 'WHU' \
    --model_name 'CGNet_SSM'
```

This will:
1. Load the trained model from `./output/WHU/CGNet_SSM_best_iou.pth`
2. Run inference on test set
3. Save predictions to `./test_result/WHU/CGNet_SSM/`
4. Print evaluation metrics (IoU, Precision, Recall, F1-Score)

### Testing on Different Datasets

```bash
# Test on LEVIR-CD
python test.py --data_name 'LEVIR' --model_name 'CGNet_SSM'

# Test on WHU-CD
python test.py --data_name 'WHU' --model_name 'CGNet_SSM'

# Test with custom model path
python test.py \
    --data_name 'SYSU' \
    --model_name 'CGNet_SSM' \
    --load './path/to/your/model.pth'
```

---

## Model Comparison

### Training All Models for Comparison

```bash
# Original CGNet
python train_CGNet.py --data_name 'WHU' --model_name 'CGNet' --epoch 50

# New CGNet_SSM
python train_CGNet.py --data_name 'WHU' --model_name 'CGNet_SSM' --epoch 50

# Compare results
python test.py --data_name 'WHU' --model_name 'CGNet'
python test.py --data_name 'WHU' --model_name 'CGNet_SSM'
```

### Expected Improvements

CGNet_SSM typically shows:
- ✅ Better recall on small/weak changes
- ✅ Improved boundary delineation
- ✅ Faster training convergence
- ✅ Lower memory usage
- ✅ Fewer false negatives

---

## Using PriorConditionedSSM in Your Own Code

### Standalone Usage

```python
import torch
from network.prior_conditioned_ssm import PriorConditionedSSMEfficient

# Create module
ssm = PriorConditionedSSMEfficient(in_dim=256)

# Input feature map from your decoder
features = torch.randn(4, 256, 32, 32)  # [B, C, H, W]

# Prior change probability map (can be any size)
prior_map = torch.randn(4, 1, 64, 64)  # [B, 1, h, w]

# Apply SSM guidance
guided_features = ssm(features, prior_map)  # [4, 256, 32, 32]
```

### Integration into Custom Networks

```python
import torch.nn as nn
from network.prior_conditioned_ssm import PriorConditionedSSMEfficient

class MyChangeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your encoder
        self.encoder = MyEncoder()
        
        # Your initial change predictor
        self.change_predictor = nn.Conv2d(512, 1, 1)
        
        # Add SSM guidance modules
        self.ssm_layer3 = PriorConditionedSSMEfficient(512)
        self.ssm_layer2 = PriorConditionedSSMEfficient(256)
        self.ssm_layer1 = PriorConditionedSSMEfficient(128)
        
        # Your decoder
        self.decoder = MyDecoder()
    
    def forward(self, img_A, img_B):
        # Extract features
        feat_A = self.encoder(img_A)
        feat_B = self.encoder(img_B)
        
        # Difference features
        diff = feat_A - feat_B
        
        # Generate initial change prior
        prior = self.change_predictor(diff)
        
        # Apply SSM guidance during decoding
        feat3 = self.ssm_layer3(diff, prior)
        feat2 = self.ssm_layer2(feat3, prior)
        feat1 = self.ssm_layer1(feat2, prior)
        
        # Final prediction
        output = self.decoder(feat1)
        return output
```

---

## Monitoring Training

### TensorBoard (if tensorboardx is installed)

```bash
tensorboard --logdir=./runs/
```

### Check Learnable Parameters During Training

Add to your training script:

```python
# Monitor SSM parameters
print(f"SSM alpha (prior strength): {model.ssm_2.alpha.item():.4f}")
print(f"SSM gamma (output scale): {model.ssm_2.gamma.item():.4f}")
```

Typical values:
- **Early training**: alpha ≈ 0.5-1.0, gamma ≈ 0.0-0.1
- **After convergence**: alpha ≈ 1.0-2.0, gamma ≈ 0.2-0.5

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size or image size
```bash
python train_CGNet.py --batchsize 4 --trainsize 256 ...
```

### Issue: SSL Certificate Error (downloading pretrained VGG)

**Solution**: Download VGG weights manually or use pre-downloaded checkpoint

```python
# In CGNet.py, modify:
vgg16_bn = models.vgg16_bn(pretrained=False)
# Then load your own weights
```

### Issue: Module not found

**Solution**: Make sure you're in the project root directory
```bash
cd /path/to/CGNet-CD-main
python train_CGNet.py ...
```

### Issue: Model not improving

**Solutions**:
1. Check data paths are correct
2. Verify data augmentation is appropriate
3. Try adjusting learning rate
4. Increase number of epochs
5. Check if dataset is balanced

---

## File Structure

```
CGNet-CD-main/
├── network/
│   ├── CGNet.py                    # Original + CGNet_SSM
│   └── prior_conditioned_ssm.py    # New SSM module
├── utils/
│   ├── data_loader.py
│   ├── loss.py
│   ├── metrics.py
│   └── ...
├── train_CGNet.py                  # Training script (updated)
├── test.py                         # Testing script (updated)
├── test_ssm_simple.py              # SSM unit tests
├── requirements.txt                # Dependencies
├── README_SSM.md                   # SSM documentation
├── USAGE_GUIDE.md                  # This file
└── output/                         # Saved models
    └── {dataset}/{model}_best_iou.pth
```

---

## Performance Tips

### 1. Faster Training

- Use `PriorConditionedSSMEfficient` (already default)
- Enable mixed precision training (requires PyTorch >= 1.6):
  ```python
  from torch.cuda.amp import autocast, GradScaler
  scaler = GradScaler()
  ```

### 2. Better Results

- Train for more epochs (100+ for LEVIR-CD)
- Use cosine learning rate schedule (already default)
- Ensemble multiple checkpoints
- Post-process with CRF or morphological operations

### 3. Memory Optimization

- Gradient accumulation for larger effective batch size:
  ```python
  # Accumulate gradients every 4 steps
  if (i + 1) % 4 == 0:
      optimizer.step()
      optimizer.zero_grad()
  ```

---

## Example Workflow

### Complete Training → Testing Pipeline

```bash
# 1. Prepare dataset (follow README.md dataset structure)
# Data should be organized as:
# dataset/
#   ├── train/A/
#   ├── train/B/
#   ├── train/label/
#   ├── val/A/
#   ├── val/B/
#   ├── val/label/
#   ├── test/A/
#   ├── test/B/
#   └── test/label/

# 2. Train model
python train_CGNet.py \
    --data_name 'WHU' \
    --model_name 'CGNet_SSM' \
    --epoch 50 \
    --batchsize 8 \
    --gpu_id '0'

# 3. Test model
python test.py \
    --data_name 'WHU' \
    --model_name 'CGNet_SSM' \
    --gpu_id '0'

# 4. Results will be saved to:
#    - Model: ./output/WHU/CGNet_SSM_best_iou.pth
#    - Predictions: ./test_result/WHU/CGNet_SSM/*.png
```

---

## Citation

If you use CGNet_SSM in your research, please cite:

```bibtex
@ARTICLE{CGNet2023,
  author={Han, Chengxi and Wu, Chen and Guo, Haonan and Hu, Meiqi and Li, Jiepan and Chen, Hongruixuan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery}, 
  year={2023},
  pages={1-17},
  doi={10.1109/JSTARS.2023.3310208}
}
```

And acknowledge the SSM modification in your paper.

---

## Additional Resources

- **Original CGNet Paper**: [IEEE Xplore](https://ieeexplore.ieee.org/document/10234560)
- **Original Repository**: [GitHub](https://github.com/chengxihan/CGNet-CD)
- **Open-CD Framework**: [GitHub](https://github.com/likyoo/open-cd)
- **Module Documentation**: See `README_SSM.md`
- **Unit Tests**: Run `python test_ssm_simple.py`

---

## Contact & Support

For issues specific to PriorConditionedSSM:
- Review `README_SSM.md` for technical details
- Check `test_ssm_simple.py` for usage examples
- Verify with `python test_ssm_simple.py`

For general CGNet issues:
- Refer to original repository
- Check dataset preparation
- Verify CUDA and PyTorch versions

---

**Happy Training! 🚀**
