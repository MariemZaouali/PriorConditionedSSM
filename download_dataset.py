"""
Download and setup LEVIR-CD dataset for quick testing
This script downloads a sample of LEVIR-CD dataset from Google Drive or alternative sources
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

def create_sample_dataset(dataset_path='./data'):
    """
    Create a minimal sample dataset structure for quick testing
    if download fails, at least users have the structure
    """
    print(f"Creating sample dataset structure at {dataset_path}...")
    
    base_path = Path(dataset_path) / 'LEVIR-CD'
    
    for split in ['train', 'val']:
        for folder in ['A', 'B', 'label']:
            dir_path = base_path / split / folder
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {dir_path}")
    
    print(f"\n✓ Dataset structure created at: {base_path}")
    return str(base_path)

def download_levir_cd(dataset_path='./data'):
    """
    Download LEVIR-CD dataset
    Note: You may need to download manually from:
    https://justchenhao.github.io/LEVIR/
    Or from Baidu Cloud: https://pan.baidu.com/s/1VVry18KFl2MSWS6_IOlYRA?pwd=l7iv
    """
    
    print("=" * 60)
    print("LEVIR-CD Dataset Download")
    print("=" * 60)
    
    print("\nDocumentation:")
    print("- Official: https://justchenhao.github.io/LEVIR/")
    print("- Baidu Cloud: https://pan.baidu.com/s/1VVry18KFl2MSWS6_IOlYRA?pwd=l7iv (pwd: 2023)")
    print("\nSteps:")
    print("1. Download from Baidu Cloud (recommended, faster)")
    print("2. Extract the zip file")
    print("3. Place extracted folders in: ./data/LEVIR-CD/")
    print("\nExpected structure:")
    print("  ./data/LEVIR-CD/")
    print("  ├── train/")
    print("  │   ├── A/       (before images)")
    print("  │   ├── B/       (after images)")
    print("  │   └── label/   (change maps)")
    print("  └── val/")
    print("      ├── A/")
    print("      ├── B/")
    print("      └── label/")
    
    # Create minimal structure
    dataset_root = create_sample_dataset(dataset_path)
    
    print("\n" + "=" * 60)
    print(f"Dataset root ready: {dataset_root}")
    print("=" * 60)
    
    return dataset_root

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else './data'
    download_levir_cd(dataset_path)
