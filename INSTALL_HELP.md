# Installation Troubleshooting Guide

## Issue: `ModuleNotFoundError: No module named 'tensorboardX'`

### Root Cause
PowerShell execution policy is preventing package installation. This is a Windows security feature.

### Solution: Choose One of These 4 Options

---

## **Option 1: Use Command Prompt (Recommended - Easiest)**

1. **Close VS Code Terminal**
2. **Open Command Prompt** (not PowerShell):
   - Press `Win + R`
   - Type `cmd`
   - Press Enter
3. **Navigate to project:**
   ```cmd
   cd E:\0.Recherche\Encadrement\Yosra Ben Naceur\Code\PriorConditionedSSM
   ```
4. **Install dependencies:**
   ```cmd
   python -m pip install tensorboardX opencv-python numpy pillow matplotlib tqdm
   ```
5. **Test installation:**
   ```cmd
   python -c "import tensorboardX; print('Success!')"
   ```
6. **Create test dataset:**
   ```cmd
   python create_test_dataset.py
   ```

---

## **Option 2: Use Python Script (No PowerShell)**

1. **Run the Python installer:**
   ```
   python install_deps.py
   ```
   This bypasses PowerShell execution policy issues by using Python directly.

---

## **Option 3: Use Batch File**

1. **Double-click** `install.bat` in the project folder
2. **Wait for installation to complete**
3. **Press Enter to close the window**

---

## **Option 4: Bypass PowerShell Execution Policy**

If you must use PowerShell:

1. **Run PowerShell as Administrator:**
   - Right-click PowerShell
   - Select "Run as Administrator"

2. **Set execution policy:**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Answer `Y` (Yes) when prompted**

4. **Close and reopen PowerShell**

5. **Install packages:**
   ```powershell
   pip install tensorboardX opencv-python numpy pillow matplotlib tqdm
   ```

---

## Verification

After installation, verify all modules are available:

```python
# Command Prompt or Python shell:
python -c "
import tensorboardX
import cv2
import numpy
import PIL
import matplotlib
import tqdm
print('✓ All packages installed successfully!')
"
```

---

## If All Else Fails: Minimal Setup

The `visualization.py` has been updated to work **without** tensorboardX:

1. **Create test dataset:**
   ```cmd
   python create_test_dataset.py
   ```

2. **Run training (will work with basic logging only):**
   ```cmd
   python train_CGNet.py --epoch 2 --batchsize 2 --data_name LEVIR
   ```

Training will work but won't have TensorBoard logs. Install tensorboardX later when PowerShell issues are resolved.

---

## Quick Test Command

Once installed, verify everything works:

```cmd
python create_test_dataset.py && python train_CGNet.py --epoch 2 --batchsize 2
```

This should:
1. Create test dataset (4 synthetic image pairs)
2. Run training for 2 epochs
3. Save results to `./output/LEVIR/CGNet/`

---

## Still Having Issues?

If you're still having problems:

1. **Check Python version:** `python --version` (should be 3.8+)
2. **Check pip:** `python -m pip --version`
3. **Check pip location:** `python -m pip show pip`
4. **Upgrade pip:** `python -m pip install --upgrade pip`
5. **Try without version constraints:** `pip install tensorboardX`

---

## Notes

- **Option 1 (Command Prompt)** is fastest and most reliable
- **Option 2 (Python script)** is second best
- Even without tensorboardX, most functionality works
- TensorBoard visualization is optional (training still works without it)
