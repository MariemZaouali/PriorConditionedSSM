"""
Visualization during training - monitor change maps at each epoch
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from network.CGNet import CGNet
from network.CGNet_SSM import CGNet_SSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, np.array(img)

def visualize_during_training(checkpoint_dir, img_A_path, img_B_path, 
                             model_type='CGNet_SSM', output_dir='training_viz'):
    """
    Load checkpoints from training and visualize their outputs
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images once
    img_A, img_A_rgb = load_image(img_A_path)
    img_B, img_B_rgb = load_image(img_B_path)
    
    # Find all checkpoints
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Create model
    if model_type == 'CGNet':
        model = CGNet().to(device)
    else:
        model = CGNet_SSM().to(device)
    
    for i, checkpoint_file in enumerate(checkpoints):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        try:
            # Load checkpoint
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            
            # Inference
            with torch.no_grad():
                outputs = model(img_A, img_B)
                change_map = F.sigmoid(outputs[1]).squeeze().cpu().numpy()
            
            # Visualization
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            
            axes[0].imshow(img_A_rgb)
            axes[0].set_title('Image T1', fontsize=10)
            axes[0].axis('off')
            
            axes[1].imshow(img_B_rgb)
            axes[1].set_title('Image T2', fontsize=10)
            axes[1].axis('off')
            
            im = axes[2].imshow(change_map, cmap='jet', vmin=0, vmax=1)
            axes[2].set_title(f'Change Map\n({checkpoint_file})', fontsize=10)
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2])
            
            title = f'{model_type} - Epoch {i+1}'
            fig.suptitle(title, fontsize=12, fontweight='bold')
            
            # Save
            output_path = os.path.join(output_dir, f'epoch_{i+1:03d}.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Epoch {i+1}: {output_path}")
        
        except Exception as e:
            print(f"✗ Failed to process {checkpoint_file}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True, help='Directory with .pth checkpoints')
    parser.add_argument('--img_A', required=True, help='Image T1')
    parser.add_argument('--img_B', required=True, help='Image T2')
    parser.add_argument('--model_type', choices=['CGNet', 'CGNet_SSM'], default='CGNet_SSM')
    parser.add_argument('--output_dir', default='training_viz')
    
    args = parser.parse_args()
    
    visualize_during_training(args.checkpoint_dir, args.img_A, args.img_B, 
                             args.model_type, args.output_dir)
    print(f"\n✓ All visualizations saved to {args.output_dir}")
