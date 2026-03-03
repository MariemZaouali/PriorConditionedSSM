"""
Batch inference: Generate change maps for test dataset
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm

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
    if label_array.max() > 1:
        label_array = label_array / 255.0
    return label_array

def inference(model, img_A, img_B):
    """Run inference and return continuous change map [0, 1]"""
    model.eval()
    with torch.no_grad():
        outputs = model(img_A, img_B)
        change_map = outputs[1]  # Final output
        change_map = F.sigmoid(change_map)
    return change_map.squeeze().cpu().numpy()

def save_change_map(change_map, output_path, colormap='jet'):
    """Save visualization of change map"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    plt.figure(figsize=(6, 6), dpi=100)
    im = plt.imshow(change_map, cmap=colormap, vmin=0, vmax=1)
    plt.colorbar(im, label='Change Probability')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def save_binary_prediction(pred, output_path):
    """Save binary prediction (0 or 1)"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    pred_uint8 = (pred * 255).astype(np.uint8)
    img = Image.fromarray(pred_uint8)
    img.save(output_path)

def batch_inference(data_dir, model, model_name='model', output_dir='results', threshold=0.5):
    """
    Run inference on all test samples in directory
    
    Expected structure:
    data_dir/
        A/              # Image T1
        B/              # Image T2
        label/          # Ground truth (optional)
    """
    
    img_a_dir = os.path.join(data_dir, 'A')
    img_b_dir = os.path.join(data_dir, 'B')
    label_dir = os.path.join(data_dir, 'label')
    
    # Get list of image files
    if os.path.exists(img_a_dir):
        image_files = sorted([f.replace('.tif', '') for f in os.listdir(img_a_dir) if f.endswith('.tif')])
    else:
        raise ValueError(f"Directory {img_a_dir} not found")
    
    has_labels = os.path.exists(label_dir)
    
    # Create output directories
    output_base = os.path.join(output_dir, model_name)
    os.makedirs(output_base, exist_ok=True)
    
    # Metrics tracking
    all_metrics = {}
    evaluator = Evaluator(num_class=2) if has_labels else None
    
    print(f"\nProcessing {len(image_files)} samples with {model_name}...")
    
    for sample_name in tqdm(image_files, desc=model_name):
        # Load images
        img_a_path = os.path.join(img_a_dir, sample_name + '.tif')
        img_b_path = os.path.join(img_b_dir, sample_name + '.tif')
        
        img_A = load_image(img_a_path)
        img_B = load_image(img_b_path)
        
        # Inference
        change_map = inference(model, img_A, img_B)
        pred_binary = (change_map >= threshold).astype(int)
        
        # Save outputs
        output_sample_dir = os.path.join(output_base, sample_name)
        os.makedirs(output_sample_dir, exist_ok=True)
        
        save_change_map(change_map, os.path.join(output_sample_dir, 'change_map.png'))
        save_binary_prediction(pred_binary, os.path.join(output_sample_dir, 'prediction_binary.tif'))
        
        # Save continuous prediction as numpy
        np.save(os.path.join(output_sample_dir, 'change_map.npy'), change_map)
        
        # Calculate metrics if labels available
        if has_labels:
            label_path = os.path.join(label_dir, sample_name + '.tif')
            label = load_label(label_path)
            label_binary = (label >= 0.5).astype(int)
            evaluator.add_batch(label_binary, pred_binary)
            
            # Save per-sample metrics
            eva_sample = Evaluator(num_class=2)
            eva_sample.add_batch(label_binary, pred_binary)
            all_metrics[sample_name] = {
                'IoU': float(eva_sample.Intersection_over_Union()[1]),
                'Precision': float(eva_sample.Precision()[1]),
                'Recall': float(eva_sample.Recall()[1]),
                'F1': float(eva_sample.F1()[1])
            }
    
    # Calculate overall metrics
    overall_metrics = None
    if evaluator is not None:
        overall_metrics = {
            'IoU': float(evaluator.Intersection_over_Union()[1]),
            'Precision': float(evaluator.Precision()[1]),
            'Recall': float(evaluator.Recall()[1]),
            'F1': float(evaluator.F1()[1]),
            'OA': float(evaluator.OA())
        }
    
    # Save metrics to JSON
    if all_metrics:
        metrics_summary = {
            'model': model_name,
            'threshold': threshold,
            'num_samples': len(image_files),
            'overall': overall_metrics,
            'per_sample': all_metrics
        }
        
        metrics_path = os.path.join(output_base, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"✓ Metrics saved to {metrics_path}")
    
    return overall_metrics, all_metrics

def main():
    parser = argparse.ArgumentParser(description='Batch inference on test dataset')
    parser.add_argument('--data_dir', required=True, help='Path to test data directory')
    parser.add_argument('--model_cg', required=True, help='Path to CGNet checkpoint')
    parser.add_argument('--model_ssm', required=True, help='Path to CGNet_SSM checkpoint')
    parser.add_argument('--output_dir', default='inference_results', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold (0-1)')
    
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    model_cg = CGNet().to(device)
    model_ssm = CGNet_SSM().to(device)
    
    checkpoint_cg = torch.load(args.model_cg, map_location=device)
    checkpoint_ssm = torch.load(args.model_ssm, map_location=device)
    
    model_cg.load_state_dict(checkpoint_cg)
    model_ssm.load_state_dict(checkpoint_ssm)
    
    print("✓ Models loaded")
    
    # Run batch inference
    metrics_cg, samples_cg = batch_inference(args.data_dir, model_cg, 'CGNet', args.output_dir, args.threshold)
    metrics_ssm, samples_ssm = batch_inference(args.data_dir, model_ssm, 'CGNet_SSM', args.output_dir, args.threshold)
    
    # Print summary
    if metrics_cg and metrics_ssm:
        print("\n" + "="*70)
        print("OVERALL RESULTS")
        print("="*70)
        
        print(f"\nCGNet:")
        for metric, value in metrics_cg.items():
            print(f"  {metric:10s}: {value:.4f}")
        
        print(f"\nCGNet_SSM:")
        for metric, value in metrics_ssm.items():
            print(f"  {metric:10s}: {value:.4f}")
        
        print(f"\nImprovement (SSM vs CGNet):")
        for metric in metrics_cg.keys():
            improvement = metrics_ssm[metric] - metrics_cg[metric]
            improvement_pct = (improvement / metrics_cg[metric] * 100) if metrics_cg[metric] != 0 else 0
            symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
            print(f"  {metric:10s}: {improvement:+.4f} ({improvement_pct:+.1f}%) {symbol}")
        
        print("="*70)
    
    print(f"\n✓ Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
