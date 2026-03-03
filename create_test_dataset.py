"""
Create a minimal test dataset for quick validation of the training pipeline
This generates small synthetic images for testing without downloading full dataset
"""

import numpy as np
from PIL import Image
import os
from pathlib import Path

def create_test_dataset(output_dir='./data/LEVIR-CD', num_samples=4, img_size=256):
    """
    Create a minimal test dataset with synthetic images
    
    Args:
        output_dir: Root directory to save dataset
        num_samples: Number of image triplets (A, B, label) to create
        img_size: Image size (256x256)
    """
    
    print(f"Creating test dataset with {num_samples} samples at {output_dir}...")
    
    dataset_root = Path(output_dir)
    
    # Create directory structure
    for split in ['train', 'val']:
        for folder in ['A', 'B', 'label']:
            dir_path = dataset_root / split / folder
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    np.random.seed(42)
    
    for split in ['train', 'val']:
        for idx in range(num_samples):
            # Create before image (A) - random pattern
            img_a = np.random.randint(50, 200, (img_size, img_size, 3), dtype=np.uint8)
            
            # Create after image (B) - similar but with some changes
            img_b = img_a.copy()
            # Add some change areas
            change_y = np.random.randint(50, 200, 2)
            change_x = np.random.randint(50, 200, 2)
            img_b[change_y[0]:change_y[0]+50, change_x[0]:change_x[0]+50, :] = np.random.randint(100, 255, (50, 50, 3), dtype=np.uint8)
            
            # Create label (binary change map)
            label = np.zeros((img_size, img_size), dtype=np.uint8)
            label[change_y[0]:change_y[0]+50, change_x[0]:change_x[0]+50] = 255
            if len(change_y) > 1 and len(change_x) > 1:
                label[change_y[1]:change_y[1]+40, change_x[1]:change_x[1]+40] = 255
            
            # Save images
            filename = f'{idx:04d}.tif'
            
            Image.fromarray(img_a, 'RGB').save(dataset_root / split / 'A' / filename)
            Image.fromarray(img_b, 'RGB').save(dataset_root / split / 'B' / filename)
            Image.fromarray(label, 'L').save(dataset_root / split / 'label' / filename)
            
            print(f"  Created {split} sample {idx+1}/{num_samples}")
    
    print(f"\n✓ Test dataset created successfully!")
    print(f"  Location: {dataset_root}")
    print(f"  Train samples: {num_samples}")
    print(f"  Val samples: {num_samples}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"\nYou can now run training:")
    print(f"  python train_CGNet.py --epoch 2 --batchsize 2 --data_name 'LEVIR'")

if __name__ == '__main__':
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else './data/LEVIR-CD'
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    img_size = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    
    create_test_dataset(output_dir, num_samples, img_size)
