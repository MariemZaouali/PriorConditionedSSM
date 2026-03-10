"""
Simple script to visualize change maps from a single image pair
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

from network.CGNet import CGNet
from network.CGNet_SSM import CGNet_SSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path):
    """Load RGB image and normalize to [0, 1]"""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(device), np.array(img)

def load_label(label_path):
    """Load label and normalize"""
    label = Image.open(label_path).convert('L')
    label_array = np.array(label).astype(np.float32)
    if label_array.max() > 1:
        label_array = label_array / 255.0
    return label_array

def inference(model, img_A, img_B):
    """Run inference"""
    model.eval()
    with torch.no_grad():
        outputs = model(img_A, img_B)
        change_map = outputs[1]  # Final output
        change_map = torch.sigmoid(change_map)
    return change_map.squeeze().cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Visualize change map')
    parser.add_argument('--img_A', required=True, help='Image T1 path')
    parser.add_argument('--img_B', required=True, help='Image T2 path')
    parser.add_argument('--label', default=None, help='Ground truth path (optional)')
    parser.add_argument('--model', required=True, help='Model checkpoint path')
    parser.add_argument('--model_type', choices=['CGNet', 'CGNet_SSM'], default='CGNet_SSM')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold')
    parser.add_argument('--save', default=None, help='Save path for visualization')
    parser.add_argument('--colormap', default='jet', help='Colormap (jet, viridis, plasma, gray, etc)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading {args.model_type}...")
    if args.model_type == 'CGNet':
        model = CGNet().to(device)
    else:
        model = CGNet_SSM().to(device)
    
    checkpoint = torch.load(args.model, map_location=device)
    
    # Handle state dict key mismatch between old and new model versions
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New checkpoint format with optimizer and metadata
        state_dict = checkpoint['model_state_dict']
    else:
        # Old checkpoint format with just model weights
        state_dict = checkpoint
    
    # Map old keys to new keys for backward compatibility
    new_state_dict = {}
    for key, value in state_dict.items():
        # Map old layer names to new layer names
        new_key = key.replace('down1.', 'd1.').replace('down2.', 'd2.').replace('down3.', 'd3.').replace('down4.', 'd4.')
        new_key = new_key.replace('conv_reduce_1.', 'red1.').replace('conv_reduce_2.', 'red2.')
        new_key = new_key.replace('conv_reduce_3.', 'red3.').replace('conv_reduce_4.', 'red4.')
        new_key = new_key.replace('decoder_module4.', 'dec_mod4.').replace('decoder_module3.', 'dec_mod3.')
        new_key = new_key.replace('decoder_module2.', 'dec_mod2.').replace('decoder_final.', 'final.')
        
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    print("✓ Model loaded")
    
    # Load images
    img_A, img_A_rgb = load_image(args.img_A)
    img_B, img_B_rgb = load_image(args.img_B)
    
    # Inference
    print("Running inference...")
    change_map = inference(model, img_A, img_B)
    
    # Binary prediction
    pred_binary = (change_map >= args.threshold).astype(np.uint8) * 255
    
    # Load label if provided
    label = None
    if args.label and os.path.exists(args.label):
        label = load_label(args.label)
        label_uint8 = (label * 255).astype(np.uint8)
    
    # Create visualization — always 4 columns:
    #   T1 | T2 | Probability Map | Ground Truth (or Binary Prediction)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Image T1
    axes[0].imshow(img_A_rgb)
    axes[0].set_title('Image T1', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Image T2
    axes[1].imshow(img_B_rgb)
    axes[1].set_title('Image T2', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Continuous change map
    im = axes[2].imshow(change_map, cmap=args.colormap, vmin=0, vmax=1)
    axes[2].set_title(f'Change Map ({args.model_type})\nContinuous [0, 1]', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], label='Change Probability')
    
    # Ground truth or binary prediction
    if label is not None:
        axes[3].imshow(label_uint8, cmap='gray', vmin=0, vmax=255)
        axes[3].set_title('Ground Truth', fontsize=12, fontweight='bold')
    else:
        axes[3].imshow(pred_binary, cmap='gray', vmin=0, vmax=255)
        axes[3].set_title(f'Binary Prediction\n(threshold={args.threshold})', 
                         fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {args.save}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*50)
    print("CHANGE MAP STATISTICS")
    print("="*50)
    print(f"Change map range: [{change_map.min():.4f}, {change_map.max():.4f}]")
    print(f"Change map mean: {change_map.mean():.4f}")
    print(f"Change map std: {change_map.std():.4f}")
    print(f"Pixels changed (threshold={args.threshold}): {pred_binary.sum() / pred_binary.size * 100:.2f}%")
    
    if label is not None:
        label_binary = (label >= 0.5).astype(np.uint8)
        tp = ((pred_binary > 0) & (label_binary > 0)).sum()
        fp = ((pred_binary > 0) & (label_binary == 0)).sum()
        fn = ((pred_binary == 0) & (label_binary > 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nWith Ground Truth:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
    
    print("="*50)

if __name__ == '__main__':
    main()
