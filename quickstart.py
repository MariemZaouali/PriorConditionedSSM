#!/usr/bin/env python3
"""
Quick Start Script for CGNet Training
This script helps set up and run a quick test of the training pipeline
"""

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def main():
    print_header("CGNet Training - Quick Start")
    
    print("This script will help you set up and test CGNet training.")
    print("\nChoose an option:")
    print("  1. Create TEST dataset (synthetic images for pipeline testing)")
    print("  2. Download REAL dataset manually (step-by-step guide)")
    print("  3. Run training with test dataset")
    print("  4. Run full setup (1 + 3)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ['1', '4']:
        print_header("Creating Test Dataset")
        print("Creating minimal synthetic dataset for testing...")
        os.system('python create_test_dataset.py ./data/LEVIR-CD 4 256')
        
        if choice == '1':
            print("\n✓ Test dataset created!")
            print("  Next: Run training with: python train_CGNet.py --epoch 2 --batchsize 2")
            return
    
    if choice in ['2']:
        print_header("Real Dataset Setup Guide")
        print("\n1. Visit one of these sources:")
        print("   - Baidu Cloud (faster): https://pan.baidu.com/s/1VVry18KFl2MSWS6_IOlYRA?pwd=2023")
        print("     Password: 2023")
        print("   - Official: https://justchenhao.github.io/LEVIR/")
        print("\n2. Download and extract the ZIP file")
        print("\n3. Organize in your project:")
        print("   ./data/LEVIR-CD/")
        print("   ├── train/")
        print("   │   ├── A/")
        print("   │   ├── B/")
        print("   │   └── label/")
        print("   └── val/")
        print("       ├── A/")
        print("       ├── B/")
        print("       └── label/")
        return
    
    if choice in ['3', '4']:
        print_header("Training with Test Dataset")
        print("\nStarting training with test dataset...")
        print("(This will train for 2 epochs with batch size 2 for quick testing)")
        
        cmd = [
            'python', 'train_CGNet.py',
            '--epoch', '2',
            '--batchsize', '2',
            '--gpu_id', '0',
            '--data_name', 'LEVIR',
            '--model_name', 'CGNet'
        ]
        
        print(f"\nCommand: {' '.join(cmd)}\n")
        
        try:
            subprocess.run(cmd, check=True)
            print_header("✓ Training Completed!")
            print("Check './output/LEVIR/CGNet/' for saved models")
        except Exception as e:
            print(f"\n✗ Error during training: {e}")
            print("\nTroubleshooting tips:")
            print("1. Check GPU availability: nvidia-smi")
            print("2. Verify dataset exists: python create_test_dataset.py")
            print("3. Check Python dependencies: pip install -r requirements.txt")
            return 1
    else:
        print("Invalid choice")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
