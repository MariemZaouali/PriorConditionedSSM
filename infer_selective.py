"""
infer_selective.py
==================
Inference-only script for CGNet_SSM_selective and CGNet_SSM_selective_4D.

Supports both fusion modes:
  --fusion_mode add        → simple element-wise addition H+V (Eq. 12, default)
  --fusion_mode learnable  → channel-wise learnable weighted fusion (ablation for reviewer)

Usage example (no training):
  python infer_selective.py \\
      --model_type CGNet_SSM_selective \\
      --fusion_mode learnable \\
      --load_path ./output/LEVIR/CGNet_SSM_selective/CGNet_SSM_selective_best_iou.pth \\
      --data_name LEVIR \\
      --test_root ./data/LEVIR-CD/test/ \\
      --save_path ./test_result/
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

from utils import data_loader_original as data_loader
from utils.metrics import Evaluator


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for CGNet_SSM_selective / CGNet_SSM_selective_4D"
    )
    parser.add_argument('--model_type', type=str,
                        default='CGNet_SSM_selective',
                        choices=['CGNet_SSM_selective', 'CGNet_SSM_selective_4D'],
                        help='Model architecture to use')
    parser.add_argument('--fusion_mode', type=str,
                        default='add',
                        choices=['add', 'learnable'],
                        help=(
                            "'add'       = simple element-wise sum H+V (Eq. 12, original)\n"
                            "'learnable' = channel-wise learnable weighted fusion (ablation)"
                        ))
    parser.add_argument('--load_path', type=str, required=True,
                        help='Path to pre-trained weights (.pth)')
    parser.add_argument('--data_name', type=str, default='LEVIR',
                        choices=['LEVIR', 'WHU', 'CDD', 'DSIFN', 'SYSU', 'S2Looking'],
                        help='Dataset name (used to build default test_root if not provided)')
    parser.add_argument('--test_root', type=str, default='',
                        help='Explicit path to the test split directory '
                             '(overrides the default derived from data_name)')
    parser.add_argument('--save_path', type=str, default='./test_result/',
                        help='Directory where predicted maps are saved')
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--trainsize', type=int, default=256,
                        help='Resize dimension used during training')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold for change map')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Default dataset paths (mirrors train_CGNet.py convention)
# ---------------------------------------------------------------------------
DATASET_ROOTS = {
    'LEVIR':     './data/LEVIR-CD/test/',
    'WHU':       './data/WHU-CD/test/',
    'CDD':       './data/CDD/test/',
    'DSIFN':     './data/DSIFN/test/',
    'SYSU':      './data/SYSU-CD/test/',
    'S2Looking': './data/S2Looking/test/',
}


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def run_inference(model, test_loader, device, save_path, threshold=0.5):
    model.eval()
    Eva_test = Evaluator(num_class=2)

    os.makedirs(save_path, exist_ok=True)

    for i, (A, B, mask, filenames) in enumerate(tqdm(test_loader, desc='Inference')):
        with torch.no_grad():
            A = A.to(device)
            B = B.to(device)
            Y = mask.to(device)

            outputs = model(A, B)
            # outputs is (coarse_pred, fine_pred, gates) for selective models
            fine_pred = outputs[1]
            output = torch.sigmoid(fine_pred)

        pred_binary = (output >= threshold).float()
        pred_np = pred_binary.squeeze(1).cpu().numpy().astype(int)
        target_np = Y.cpu().numpy()
        if target_np.max() > 1:
            target_np = target_np // 255

        Eva_test.add_batch(target_np, pred_np)

        # Save binary predictions
        for j in range(output.shape[0]):
            prob_map = output[j].squeeze().cpu().numpy()
            final_mask = (prob_map * 255).astype(np.uint8)
            fname = filenames[j] if isinstance(filenames[j], str) else str(filenames[j])
            img_path = os.path.join(save_path, fname + '.png')
            os.makedirs(os.path.dirname(img_path) if os.path.dirname(img_path) else save_path,
                        exist_ok=True)
            Image.fromarray(final_mask).save(img_path)

    # Print metrics
    IoU   = Eva_test.Intersection_over_Union()
    Pre   = Eva_test.Precision()
    Rec   = Eva_test.Recall()
    F1    = Eva_test.F1()
    OA    = Eva_test.OA()
    Kappa = Eva_test.Kappa()

    print('\n' + '=' * 60)
    print(f'  Results — fusion_mode = {opt.fusion_mode}')
    print('=' * 60)
    print(f'  F1        : {F1[1]*100:.2f} %')
    print(f'  Precision : {Pre[1]*100:.2f} %')
    print(f'  Recall    : {Rec[1]*100:.2f} %')
    print(f'  OA        : {OA[1]*100:.2f} %')
    print(f'  Kappa     : {Kappa[1]*100:.2f} %')
    print(f'  IoU       : {IoU[1]*100:.2f} %')
    print('=' * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    opt = parse_args()

    # -- Device setup ---------------------------------------------------------
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id   = int(opt.gpu_id) if opt.gpu_id.isdigit() else 0
        if gpu_id >= num_gpus:
            print(f'Warning: GPU {gpu_id} not available, falling back to GPU 0')
            gpu_id = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device('cuda:0')
        print(f'Using GPU {gpu_id} ({num_gpus} GPU(s) available)')
    else:
        device = torch.device('cpu')
        print('Warning: No CUDA GPUs — using CPU (will be slow)')

    # -- Model ----------------------------------------------------------------
    if opt.model_type == 'CGNet_SSM_selective':
        from network.CGNet_SSM_selective import CGNet_SSM
        model = CGNet_SSM(fusion_mode=opt.fusion_mode).to(device)
    elif opt.model_type == 'CGNet_SSM_selective_4D':
        from network.CGNet_SSM_selective_4D import CGNet_SSM
        model = CGNet_SSM(fusion_mode=opt.fusion_mode).to(device)
    else:
        raise ValueError(f'Unknown model_type: {opt.model_type}')

    print(f'[*] Model  : {opt.model_type}  (fusion_mode={opt.fusion_mode})')

    # -- Load weights ---------------------------------------------------------
    if not os.path.exists(opt.load_path):
        raise FileNotFoundError(
            f'Pre-trained weights not found: {opt.load_path}\n'
            f'Train first with train_CGNet.py or point to an existing checkpoint.'
        )

    checkpoint = torch.load(opt.load_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'[*] Loaded checkpoint from {opt.load_path} '
              f'(epoch {checkpoint.get("epoch", "?")})')
    else:
        model.load_state_dict(checkpoint)
        print(f'[*] Loaded weights from {opt.load_path}')

    # -- Test data loader -----------------------------------------------------
    test_root = opt.test_root if opt.test_root else DATASET_ROOTS.get(opt.data_name, '')
    if not test_root or not os.path.exists(test_root):
        raise FileNotFoundError(
            f'Test directory not found: {test_root}\n'
            f'Set --test_root explicitly or ensure ./data/{opt.data_name}-CD/test/ exists.'
        )

    test_loader = data_loader.get_test_loader(
        test_root, opt.batchsize, opt.trainsize,
        num_workers=2, shuffle=False, pin_memory=True
    )
    print(f'[*] Test set: {test_root}  ({len(test_loader.dataset)} samples)')

    # -- Save path ------------------------------------------------------------
    full_save_path = os.path.join(
        opt.save_path, opt.data_name, opt.model_type,
        f'fusion_{opt.fusion_mode}'
    )

    # -- Run ------------------------------------------------------------------
    run_inference(model, test_loader, device, full_save_path, opt.threshold)
    print(f'\n[*] Predictions saved to: {full_save_path}')
