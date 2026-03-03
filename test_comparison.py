"""
Compare change maps between CGNet and CGNet_SSM
Generates side-by-side visualizations and metrics
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import argparse
from pathlib import Path

from network.CGNet import CGNet
from network.CGNet_SSM import CGNet_SSM
from utils.metrics import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path):
    """Load and normalize image to [0, 1]"""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(device)

def load_label(label_path):
    """Load label image and normalize to [0, 1]"""
    label = Image.open(label_path).convert('L')
    label_array = np.array(label).astype(np.float32)
    # Normalize: if max value is 255, divide by 255
    if label_array.max() > 1:
        label_array = label_array / 255.0
    return label_array

def inference(model, img_A, img_B):
    """Run inference and return change map"""
    model.eval()
    with torch.no_grad():
        outputs = model(img_A, img_B)
        change_map = outputs[1]  # Final output
        change_map = F.sigmoid(change_map)
    return change_map.squeeze().cpu().numpy()

def compare_models(image_A_path, image_B_path, label_path, model_cg=None, model_ssm=None):
    """Compare CGNet and CGNet_SSM outputs"""
    
    # Load images
    img_A = load_image(image_A_path)
    img_B = load_image(image_B_path)
    label = load_label(label_path)
    
    # Get change maps
    change_map_cg = inference(model_cg, img_A, img_B)
    change_map_ssm = inference(model_ssm, img_A, img_B)
    
    # Binarize at threshold 0.5
    pred_cg_binary = (change_map_cg >= 0.5).astype(int)
    pred_ssm_binary = (change_map_ssm >= 0.5).astype(int)
    
    # Calculate metrics
    label_binary = (label >= 0.5).astype(int)
    
    eva_cg = Evaluator(num_class=2)
    eva_ssm = Evaluator(num_class=2)
    
    eva_cg.add_batch(label_binary, pred_cg_binary)
    eva_ssm.add_batch(label_binary, pred_ssm_binary)
    
    metrics_cg = {
        'IoU': eva_cg.Intersection_over_Union()[1],
        'Precision': eva_cg.Precision()[1],
        'Recall': eva_cg.Recall()[1],
        'F1': eva_cg.F1()[1]
    }
    
    metrics_ssm = {
        'IoU': eva_ssm.Intersection_over_Union()[1],
        'Precision': eva_ssm.Precision()[1],
        'Recall': eva_ssm.Recall()[1],
        'F1': eva_ssm.F1()[1]
    }
    
    return {
        'img_A': img_A.squeeze(0).permute(1, 2, 0).cpu().numpy(),
        'img_B': img_B.squeeze(0).permute(1, 2, 0).cpu().numpy(),
        'label': label,
        'change_map_cg': change_map_cg,
        'change_map_ssm': change_map_ssm,
        'pred_cg': pred_cg_binary,
        'pred_ssm': pred_ssm_binary,
        'metrics_cg': metrics_cg,
        'metrics_ssm': metrics_ssm
    }

def plot_comparison(results, save_path=None):
    """Plot side-by-side comparison"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.30)
    
    # Row 1: Input images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['img_A'])
    ax1.set_title('Image T1', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results['img_B'])
    ax2.set_title('Image T2', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2:4])
    ax3.imshow(results['label'], cmap='gray')
    ax3.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Row 2: Change maps (continuous)
    ax4 = fig.add_subplot(gs[1, 0:2])
    im4 = ax4.imshow(results['change_map_cg'], cmap='jet', vmin=0, vmax=1)
    ax4.set_title('CGNet - Continuous', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    ax5 = fig.add_subplot(gs[1, 2:4])
    im5 = ax5.imshow(results['change_map_ssm'], cmap='jet', vmin=0, vmax=1)
    ax5.set_title('CGNet_SSM - Continuous', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # Row 3: Binary predictions
    ax6 = fig.add_subplot(gs[2, 0:2])
    ax6.imshow(results['pred_cg'], cmap='gray')
    ax6.set_title('CGNet - Binary', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[2, 2:4])
    ax7.imshow(results['pred_ssm'], cmap='gray')
    ax7.set_title('CGNet_SSM - Binary', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # Add metrics text
    metrics_text_cg = f"CGNet Metrics:\nIoU: {results['metrics_cg']['IoU']:.4f}\nPrecision: {results['metrics_cg']['Precision']:.4f}\nRecall: {results['metrics_cg']['Recall']:.4f}\nF1: {results['metrics_cg']['F1']:.4f}"
    metrics_text_ssm = f"CGNet_SSM Metrics:\nIoU: {results['metrics_ssm']['IoU']:.4f}\nPrecision: {results['metrics_ssm']['Precision']:.4f}\nRecall: {results['metrics_ssm']['Recall']:.4f}\nF1: {results['metrics_ssm']['F1']:.4f}"
    
    fig.text(0.05, 0.02, metrics_text_cg, fontsize=10, family='monospace', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.text(0.52, 0.02, metrics_text_ssm, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare CGNet and CGNet_SSM change detection')
    parser.add_argument('--img_A', required=True, help='Path to image T1')
    parser.add_argument('--img_B', required=True, help='Path to image T2')
    parser.add_argument('--label', required=True, help='Path to ground truth label')
    parser.add_argument('--model_cg', required=True, help='Path to CGNet checkpoint')
    parser.add_argument('--model_ssm', required=True, help='Path to CGNet_SSM checkpoint')
    parser.add_argument('--save', default='comparison_result.png', help='Output path for comparison image')
    
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    model_cg = CGNet().to(device)
    model_ssm = CGNet_SSM().to(device)
    
    checkpoint_cg = torch.load(args.model_cg, map_location=device)
    checkpoint_ssm = torch.load(args.model_ssm, map_location=device)
    
    model_cg.load_state_dict(checkpoint_cg)
    model_ssm.load_state_dict(checkpoint_ssm)
    
    print("✓ Models loaded successfully")
    
    # Run comparison
    print("Running inference and comparison...")
    results = compare_models(args.img_A, args.img_B, args.label, model_cg, model_ssm)
    
    # Print metrics
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nCGNet Metrics:")
    for metric, value in results['metrics_cg'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nCGNet_SSM Metrics:")
    for metric, value in results['metrics_ssm'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nImprovement (SSM vs CGNet):")
    for metric in results['metrics_cg'].keys():
        improvement = results['metrics_ssm'][metric] - results['metrics_cg'][metric]
        improvement_pct = (improvement / results['metrics_cg'][metric] * 100) if results['metrics_cg'][metric] != 0 else 0
        symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
        print(f"  {metric}: {improvement:+.4f} ({improvement_pct:+.1f}%) {symbol}")
    print("="*60 + "\n")
    
    # Plot results
    plot_comparison(results, args.save)

if __name__ == '__main__':
    main()
