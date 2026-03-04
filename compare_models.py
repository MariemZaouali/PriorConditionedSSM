"""
=============================================================================
  CGNet vs CGNet_SSM — Full Comparison Pipeline
  with Step-by-Step Visualization & Comprehensive Metrics
=============================================================================

Usage (single image pair):
  python compare_models.py --mode single \
      --img_A data/LEVIR-CD/val/A/0000.tif \
      --img_B data/LEVIR-CD/val/B/0000.tif \
      --label data/LEVIR-CD/val/label/0000.tif \
      --model_cg  checkpoints/CGNet_best.pth \
      --model_ssm checkpoints/CGNet_SSM_best.pth

Usage (whole dataset directory):
  python compare_models.py --mode dataset \
      --data_dir data/LEVIR-CD/val \
      --model_cg  checkpoints/CGNet_best.pth \
      --model_ssm checkpoints/CGNet_SSM_best.pth \
      --out_dir   results/comparison

Quick demo (no checkpoints — random weights):
  python compare_models.py --mode demo \
      --data_dir data/LEVIR-CD/val \
      --out_dir  results/demo
"""

import os
import sys
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import requests
import zipfile
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')           # headless-safe; change to 'TkAgg' for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image

from network.CGNet     import CGNet
from network.CGNet_SSM import CGNet_SSM
from utils.metrics     import Evaluator

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5          # binarisation threshold

# ---------------------------------------------------------------------------
# Colour palette for error maps:  TN=black  FP=red  FN=cyan  TP=white
# ---------------------------------------------------------------------------
ERROR_CMAP = ListedColormap(['#000000', '#ff4444', '#44ddff', '#ffffff'])
ERROR_LABELS = ['TN (Correct No-Change)',
                'FP (False Alarm)',
                'FN (Missed Change)',
                'TP (Correct Change)']


# ===========================================================================
#  Dataset download
# ===========================================================================

def download_levir_cd(target_dir: str = 'data/LEVIR-CD'):
    """
    Download and extract LEVIR-CD dataset if not already present.
    """
    target_dir = Path(target_dir)
    if target_dir.exists():
        print(f"  ✓ Dataset already exists at {target_dir}")
        return

    url = 'https://justchenhao.github.io/LEVIR/LEVIR-CD.zip'
    zip_path = target_dir.parent / 'LEVIR-CD.zip'

    print(f"  Downloading LEVIR-CD dataset from {url} ...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(
            desc='Downloading', total=total_size, unit='B', unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        print(f"  ✓ Downloaded to {zip_path}")

        print("  Extracting ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir.parent)
        print(f"  ✓ Extracted to {target_dir.parent}")

        # Clean up zip
        zip_path.unlink(missing_ok=True)
        print("  ✓ Cleaned up zip file")
    except Exception as e:
        print(f"  ⚠ Download failed: {e}")
        print("  Proceeding without dataset download.")


# ===========================================================================
#  Helper functions
# ===========================================================================

def load_image(path: str) -> torch.Tensor:
    """Load RGB image → float32 tensor [1, 3, H, W] in [0, 1]."""
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def load_label(path: str) -> np.ndarray:
    """Load label image → binary uint8 [H, W] ∈ {0, 1}."""
    lbl = Image.open(path).convert('L')
    arr = np.array(lbl, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return (arr >= THRESHOLD).astype(np.uint8)


def run_inference(model: torch.nn.Module,
                  img_A: torch.Tensor,
                  img_B: torch.Tensor) -> np.ndarray:
    """Return sigmoid-activated final change map as numpy [H, W]."""
    model.eval()
    with torch.no_grad():
        _, final = model(img_A, img_B)
        prob = torch.sigmoid(final).squeeze().cpu().numpy()
    return prob


def compute_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Build a 4-class error map:
      0 = TN  1 = FP  2 = FN  3 = TP
    """
    err = np.zeros_like(gt, dtype=np.int32)
    err[(gt == 0) & (pred == 0)] = 0   # TN
    err[(gt == 0) & (pred == 1)] = 1   # FP
    err[(gt == 1) & (pred == 0)] = 2   # FN
    err[(gt == 1) & (pred == 1)] = 3   # TP
    return err


def compute_metrics(evaluator: Evaluator) -> dict:
    """Extract all scalar metrics from an Evaluator object."""
    iou  = evaluator.Intersection_over_Union()
    prec = evaluator.Precision()
    rec  = evaluator.Recall()
    f1   = evaluator.F1()
    return {
        'IoU'      : float(iou [1]),
        'Precision': float(prec[1]),
        'Recall'   : float(rec [1]),
        'F1'       : float(f1  [1]),
        'OA'       : float(evaluator.OA()),
        'Kappa'    : float(evaluator.Kappa()),
        'mIoU'     : float(evaluator.Mean_Intersection_over_Union()),
    }


# ===========================================================================
#  Step-by-step visualization for ONE image pair
# ===========================================================================

def visualize_steps(img_A_np  : np.ndarray,
                    img_B_np  : np.ndarray,
                    gt        : np.ndarray,
                    prob_cg   : np.ndarray,
                    prob_ssm  : np.ndarray,
                    metrics_cg: dict,
                    metrics_ssm: dict,
                    title     : str = 'Comparison',
                    save_path : str = None,
                    show      : bool = False) -> plt.Figure:
    """
    Create a rich figure with 5 rows of visualisation steps:

      Row 1 – Inputs          : Image T1 | Image T2 | Ground Truth
      Row 2 – Probability maps: CGNet | CGNet_SSM  (jet colormap + colorbar)
      Row 3 – Binary preds    : CGNet | CGNet_SSM
      Row 4 – Error maps      : CGNet | CGNet_SSM  (TP/FP/FN/TN)
      Row 5 – Metric bars     : grouped bar chart comparing all metrics
    """
    pred_cg  = (prob_cg  >= THRESHOLD).astype(np.uint8)
    pred_ssm = (prob_ssm >= THRESHOLD).astype(np.uint8)
    err_cg   = compute_error_map(gt, pred_cg)
    err_ssm  = compute_error_map(gt, pred_ssm)

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(5, 3, figure=fig,
                           hspace=0.45, wspace=0.30,
                           left=0.06, right=0.96,
                           top=0.95, bottom=0.05)

    # ── Row 1 : Inputs ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_A_np)
    ax.set_title('Image T1 (Before)', fontsize=11, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img_B_np)
    ax.set_title('Image T2 (After)', fontsize=11, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Ground Truth', fontsize=11, fontweight='bold')
    ax.axis('off')

    # ── Row 2 : Continuous probability maps ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(prob_cg, cmap='jet', vmin=0, vmax=1)
    ax.set_title('CGNet – Probability Map', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(prob_ssm, cmap='jet', vmin=0, vmax=1)
    ax.set_title('CGNet_SSM – Probability Map', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Difference map (SSM - CGNet)
    diff = prob_ssm - prob_cg
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title('Difference (SSM − CGNet)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 3 : Binary predictions ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(pred_cg, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'CGNet – Binary (thr={THRESHOLD})', fontsize=11, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(pred_ssm, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'CGNet_SSM – Binary (thr={THRESHOLD})', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Agreement map
    agree = np.zeros_like(gt, dtype=np.int32)
    agree[(pred_cg == pred_ssm) & (pred_cg == gt)]  = 3   # both correct
    agree[(pred_cg != pred_ssm) & (pred_ssm == gt)] = 2   # SSM wins
    agree[(pred_cg != pred_ssm) & (pred_cg  == gt)] = 1   # CGNet wins
    agree[(pred_cg == pred_ssm) & (pred_cg != gt)]  = 0   # both wrong
    agree_cmap = ListedColormap(['#333333', '#ffaa00', '#4488ff', '#44cc44'])
    ax = fig.add_subplot(gs[2, 2])
    im = ax.imshow(agree, cmap=agree_cmap, vmin=0, vmax=3)
    ax.set_title('Model Agreement Map', fontsize=11, fontweight='bold')
    ax.axis('off')
    patches = [
        mpatches.Patch(color='#333333', label='Both Wrong'),
        mpatches.Patch(color='#ffaa00', label='CGNet Only Correct'),
        mpatches.Patch(color='#4488ff', label='SSM Only Correct'),
        mpatches.Patch(color='#44cc44', label='Both Correct'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=7,
              framealpha=0.7, handlelength=1.2)

    # ── Row 4 : Error maps (TP / FP / FN / TN) ──────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    im = ax.imshow(err_cg, cmap=ERROR_CMAP, vmin=0, vmax=3)
    ax.set_title('CGNet – Error Map', fontsize=11, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[3, 1])
    im = ax.imshow(err_ssm, cmap=ERROR_CMAP, vmin=0, vmax=3)
    ax.set_title('CGNet_SSM – Error Map', fontsize=11, fontweight='bold')
    ax.axis('off')
    # Shared legend for error maps
    err_patches = [mpatches.Patch(color=c, label=l)
                   for c, l in zip(['#000000', '#ff4444', '#44ddff', '#ffffff'],
                                    ERROR_LABELS)]
    ax.legend(handles=err_patches, loc='lower right', fontsize=7,
              framealpha=0.8, handlelength=1.2)

    # Overlay on image B for context
    overlay = img_B_np.copy()
    overlay[err_ssm == 1, :] = [1.0, 0.27, 0.27]   # FP → red
    overlay[err_ssm == 2, :] = [0.27, 0.87, 1.0]   # FN → cyan
    overlay[err_ssm == 3, :] = [1.0, 1.0, 1.0]     # TP → white
    ax = fig.add_subplot(gs[3, 2])
    ax.imshow(overlay)
    ax.set_title('CGNet_SSM Errors on T2', fontsize=11, fontweight='bold')
    ax.axis('off')

    # ── Row 5 : Grouped bar chart ────────────────────────────────────────────
    ax = fig.add_subplot(gs[4, :])
    metric_keys = ['IoU', 'Precision', 'Recall', 'F1', 'OA', 'Kappa', 'mIoU']
    vals_cg  = [metrics_cg .get(k, 0.0) for k in metric_keys]
    vals_ssm = [metrics_ssm.get(k, 0.0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    w = 0.35
    bars_cg  = ax.bar(x - w/2, vals_cg,  w, label='CGNet',     color='#4488ff', alpha=0.85)
    bars_ssm = ax.bar(x + w/2, vals_ssm, w, label='CGNet_SSM', color='#ff8844', alpha=0.85)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Metric Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    for bar in bars_cg:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)
    for bar in bars_ssm:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

    # Save / show
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f"  ✓ Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ===========================================================================
#  Summary figure: dataset-level metrics
# ===========================================================================

def plot_summary(metrics_cg: dict, metrics_ssm: dict,
                 save_path: str = None, show: bool = False) -> plt.Figure:
    """
    Two panels:
      Left  – grouped bar chart of all metrics
      Right – improvement bar chart (SSM − CGNet)
    """
    metric_keys = ['IoU', 'Precision', 'Recall', 'F1', 'OA', 'Kappa', 'mIoU']
    vals_cg  = [metrics_cg .get(k, 0.0) for k in metric_keys]
    vals_ssm = [metrics_ssm.get(k, 0.0) for k in metric_keys]
    diffs    = [v2 - v1 for v1, v2 in zip(vals_cg, vals_ssm)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('CGNet vs CGNet_SSM — Dataset-Level Summary', fontsize=14, fontweight='bold')

    # Left: absolute values
    x = np.arange(len(metric_keys))
    w = 0.35
    b1 = ax1.bar(x - w/2, vals_cg,  w, label='CGNet',     color='#4488ff', alpha=0.85)
    b2 = ax1.bar(x + w/2, vals_ssm, w, label='CGNet_SSM', color='#ff8844', alpha=0.85)
    ax1.set_ylim(0, 1.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_keys, fontsize=10)
    ax1.set_ylabel('Score')
    ax1.set_title('Absolute Metrics')
    ax1.legend(fontsize=10)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax1.set_axisbelow(True)
    for b in b1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                 f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for b in b2:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                 f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    # Right: improvement
    colours = ['#44cc44' if d >= 0 else '#cc4444' for d in diffs]
    bars = ax2.bar(x, diffs, width=0.55, color=colours, alpha=0.85)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_keys, fontsize=10)
    ax2.set_ylabel('Δ Score (SSM − CGNet)')
    ax2.set_title('Improvement of CGNet_SSM over CGNet')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)
    for b, d in zip(bars, diffs):
        ax2.text(b.get_x() + b.get_width()/2,
                 d + (0.002 if d >= 0 else -0.004),
                 f'{d:+.4f}', ha='center',
                 va='bottom' if d >= 0 else 'top', fontsize=8.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f"\n✓ Summary figure saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ===========================================================================
#  Print metrics table to console
# ===========================================================================

def print_metrics_table(metrics_cg: dict, metrics_ssm: dict):
    sep = '─' * 62
    print('\n' + sep)
    print(f"{'Metric':<12} {'CGNet':>10} {'CGNet_SSM':>12} {'Δ (SSM−CG)':>12}  {'±%':>7}")
    print(sep)
    for k in ['IoU', 'Precision', 'Recall', 'F1', 'OA', 'Kappa', 'mIoU']:
        v_cg  = metrics_cg .get(k, float('nan'))
        v_ssm = metrics_ssm.get(k, float('nan'))
        delta = v_ssm - v_cg
        pct   = (delta / v_cg * 100) if v_cg != 0 else float('nan')
        arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '=')
        print(f"{k:<12} {v_cg:>10.4f} {v_ssm:>12.4f} {delta:>+12.4f}  {pct:>+6.1f}% {arrow}")
    print(sep + '\n')


# ===========================================================================
#  Model loaders
# ===========================================================================

def build_models(ckpt_cg: str = None, ckpt_ssm: str = None):
    """Instantiate models and optionally load checkpoints."""
    model_cg  = CGNet()    .to(device)
    model_ssm = CGNet_SSM().to(device)

    if ckpt_cg:
        ck = torch.load(ckpt_cg, map_location=device)
        if isinstance(ck, dict) and 'state_dict' in ck:
            ck = ck['state_dict']
        model_cg.load_state_dict(ck)
        print(f"  ✓ CGNet     loaded from {ckpt_cg}")
    else:
        print("  ⚠ CGNet     — using random weights (no checkpoint provided)")

    if ckpt_ssm:
        ck = torch.load(ckpt_ssm, map_location=device)
        if isinstance(ck, dict) and 'state_dict' in ck:
            ck = ck['state_dict']
        model_ssm.load_state_dict(ck)
        print(f"  ✓ CGNet_SSM loaded from {ckpt_ssm}")
    else:
        print("  ⚠ CGNet_SSM — using random weights (no checkpoint provided)")

    model_cg .eval()
    model_ssm.eval()
    return model_cg, model_ssm


# ===========================================================================
#  Single-pair mode
# ===========================================================================

def run_single(args):
    print('\n=== SINGLE IMAGE PAIR MODE ===')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cg, model_ssm = build_models(args.model_cg, args.model_ssm)

    img_A = load_image(args.img_A)
    img_B = load_image(args.img_B)
    gt    = load_label(args.label)

    print('\nRunning inference …')
    t0 = time.time()
    prob_cg  = run_inference(model_cg,  img_A, img_B)
    t_cg = time.time() - t0
    t0 = time.time()
    prob_ssm = run_inference(model_ssm, img_A, img_B)
    t_ssm = time.time() - t0
    print(f"  CGNet     : {t_cg*1000:.1f} ms")
    print(f"  CGNet_SSM : {t_ssm*1000:.1f} ms")

    pred_cg  = (prob_cg  >= THRESHOLD).astype(np.uint8)
    pred_ssm = (prob_ssm >= THRESHOLD).astype(np.uint8)

    eva_cg  = Evaluator(2); eva_cg .add_batch(gt, pred_cg)
    eva_ssm = Evaluator(2); eva_ssm.add_batch(gt, pred_ssm)

    metrics_cg  = compute_metrics(eva_cg)
    metrics_ssm = compute_metrics(eva_ssm)

    print_metrics_table(metrics_cg, metrics_ssm)

    img_A_np = img_A.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_B_np = img_B.squeeze(0).permute(1, 2, 0).cpu().numpy()

    stem = Path(args.img_A).stem
    visualize_steps(
        img_A_np, img_B_np, gt,
        prob_cg, prob_ssm,
        metrics_cg, metrics_ssm,
        title=f'CGNet vs CGNet_SSM  |  {stem}',
        save_path=str(out_dir / f'{stem}_comparison.png'),
        show=args.show,
    )
    plot_summary(
        metrics_cg, metrics_ssm,
        save_path=str(out_dir / f'{stem}_summary.png'),
        show=args.show,
    )
    print('Done.')


# ===========================================================================
#  Dataset mode (iterates over all samples in a split directory)
# ===========================================================================

def run_dataset(args):
    print('\n=== DATASET MODE ===')
    # Ensure dataset is present
    download_levir_cd(target_dir=str(Path(args.data_dir).parent))
    
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / 'per_sample'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Discover sample names
    label_dir = data_dir / 'label'
    if not label_dir.exists():
        sys.exit(f"ERROR: label directory not found: {label_dir}")
    sample_names = sorted([p.stem for p in label_dir.iterdir()
                            if p.suffix.lower() in ('.tif', '.png', '.jpg')])
    print(f"  Found {len(sample_names)} samples in {data_dir}")

    model_cg, model_ssm = build_models(args.model_cg, args.model_ssm)

    eva_cg  = Evaluator(2)
    eva_ssm = Evaluator(2)

    # Timing accumulators
    time_cg = 0.0; time_ssm = 0.0

    for name in tqdm(sample_names, desc='Evaluating', unit='img'):
        # Try common extensions
        def find_file(folder, stem):
            for ext in ('.tif', '.png', '.jpg', '.jpeg'):
                p = folder / (stem + ext)
                if p.exists():
                    return p
            return None

        p_A = find_file(data_dir / 'A',     name)
        p_B = find_file(data_dir / 'B',     name)
        p_L = find_file(data_dir / 'label', name)

        if p_A is None or p_B is None or p_L is None:
            tqdm.write(f"  ⚠ Skipping {name} — missing files")
            continue

        img_A = load_image(str(p_A))
        img_B = load_image(str(p_B))
        gt    = load_label(str(p_L))

        t0 = time.time()
        prob_cg  = run_inference(model_cg,  img_A, img_B)
        time_cg  += time.time() - t0
        t0 = time.time()
        prob_ssm = run_inference(model_ssm, img_A, img_B)
        time_ssm += time.time() - t0

        pred_cg  = (prob_cg  >= THRESHOLD).astype(np.uint8)
        pred_ssm = (prob_ssm >= THRESHOLD).astype(np.uint8)

        eva_cg .add_batch(gt, pred_cg)
        eva_ssm.add_batch(gt, pred_ssm)

        # Per-sample visualisation (only if requested or few samples)
        if args.save_all_vis or len(sample_names) <= 20:
            img_A_np = img_A.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_B_np = img_B.squeeze(0).permute(1, 2, 0).cpu().numpy()
            e_cg  = Evaluator(2); e_cg .add_batch(gt, pred_cg)
            e_ssm = Evaluator(2); e_ssm.add_batch(gt, pred_ssm)
            visualize_steps(
                img_A_np, img_B_np, gt,
                prob_cg, prob_ssm,
                compute_metrics(e_cg), compute_metrics(e_ssm),
                title=f'CGNet vs CGNet_SSM  |  {name}',
                save_path=str(vis_dir / f'{name}_comparison.png'),
                show=False,
            )

    # Dataset-level metrics
    metrics_cg  = compute_metrics(eva_cg)
    metrics_ssm = compute_metrics(eva_ssm)

    n = len(sample_names)
    print(f"\n  Avg inference time — CGNet    : {time_cg  / n * 1000:.1f} ms/img")
    print(f"  Avg inference time — CGNet_SSM: {time_ssm / n * 1000:.1f} ms/img")

    print_metrics_table(metrics_cg, metrics_ssm)

    plot_summary(
        metrics_cg, metrics_ssm,
        save_path=str(out_dir / 'dataset_summary.png'),
        show=args.show,
    )

    # Save numerical results
    import json
    results = {
        'num_samples'  : n,
        'CGNet'        : metrics_cg,
        'CGNet_SSM'    : metrics_ssm,
        'improvement'  : {k: metrics_ssm[k] - metrics_cg[k] for k in metrics_cg},
        'time_cg_ms'   : round(time_cg  / n * 1000, 2),
        'time_ssm_ms'  : round(time_ssm / n * 1000, 2),
    }
    json_path = out_dir / 'comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Numerical results saved → {json_path}")
    print('Done.')


# ===========================================================================
#  Demo mode — runs without checkpoints on the sample data
# ===========================================================================

def run_demo(args):
    """
    Quick demo: loads random-weight models on the val images already present
    in ./data/LEVIR-CD/val (or any other directory passed via --data_dir).
    Useful to verify the pipeline works before you have trained checkpoints.
    """
    print('\n=== DEMO MODE (random weights — for pipeline verification) ===')
    args.model_cg  = None
    args.model_ssm = None
    run_dataset(args)


# ===========================================================================
#  CLI entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='CGNet vs CGNet_SSM — Comparison with Visualization & Metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--mode', choices=['single', 'dataset', 'demo'],
                   default='demo',
                   help='Comparison mode (default: demo)')

    # Single-mode paths
    p.add_argument('--img_A',  default=None, help='Path to T1 image (single mode)')
    p.add_argument('--img_B',  default=None, help='Path to T2 image (single mode)')
    p.add_argument('--label',  default=None, help='Path to GT label (single mode)')

    # Dataset path
    p.add_argument('--data_dir', default='data/LEVIR-CD/val',
                   help='Root of dataset split: must contain A/, B/, label/')

    # Checkpoint paths
    p.add_argument('--model_cg',  default=None, help='CGNet checkpoint (.pth)')
    p.add_argument('--model_ssm', default=None, help='CGNet_SSM checkpoint (.pth)')

    # Output
    p.add_argument('--out_dir', default='results/comparison',
                   help='Output directory for figures and JSON')
    p.add_argument('--save_all_vis', action='store_true',
                   help='Save per-sample visualisations for all images '
                        '(dataset mode, auto-enabled when ≤20 samples)')
    p.add_argument('--show', action='store_true',
                   help='Display figures interactively (requires a display)')

    return p.parse_args()


def main():
    args = parse_args()

    print('=' * 62)
    print('  CGNet vs CGNet_SSM Comparison Pipeline')
    print(f'  Mode   : {args.mode}')
    print(f'  Device : {device}')
    print('=' * 62)

    if args.mode == 'single':
        for field, name in [('img_A', '--img_A'), ('img_B', '--img_B'), ('label', '--label')]:
            if getattr(args, field) is None:
                sys.exit(f"ERROR: {name} is required in single mode")
        run_single(args)

    elif args.mode == 'dataset':
        run_dataset(args)

    elif args.mode == 'demo':
        run_demo(args)


if __name__ == '__main__':
    main()
