@echo off
REM Install CGNet-CD dependencies
echo Installing required packages...
python -m pip install tensorboardX opencv-python numpy pillow matplotlib tqdm --no-warn-script-location
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Installation successful!
    echo.
    echo Next steps:
    echo 1. Create test dataset: python create_test_dataset.py
    echo 2. Run training: python train_CGNet.py --epoch 2 --batchsize 2 --data_name LEVIR
) else (
    echo.
    echo ✗ Installation failed. Try running in Command Prompt instead of PowerShell.
)
pause
