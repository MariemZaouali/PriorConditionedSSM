#!/usr/bin/env python3
"""
Install dependencies for CGNet-CD
Run this script if pip install fails due to permission/execution policy issues
"""

import subprocess
import sys

def install_packages():
    """Install required packages using pip"""
    packages = [
        'tensorboardX',
        'opencv-python',
        'numpy',
        'pillow',
        'matplotlib',
        'tqdm'
    ]
    
    print("=" * 60)
    print("  CGNet-CD Dependency Installer")
    print("=" * 60)
    print(f"\nInstalling {len(packages)} packages...\n")
    
    for package in packages:
        print(f"Installing {package}...", end=" ")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                package, '-q', '--no-warn-script-location'
            ])
            print("✓")
        except subprocess.CalledProcessError as e:
            print(f"✗ (Error: {e})")
            print(f"  Skipping {package}, may already be installed")
    
    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Create test dataset:")
    print("   python create_test_dataset.py")
    print("\n2. Run training:")
    print("   python train_CGNet.py --epoch 2 --batchsize 2 --data_name LEVIR")

if __name__ == '__main__':
    try:
        install_packages()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
