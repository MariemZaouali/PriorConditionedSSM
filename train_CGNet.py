#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
#from catalyst.contrib.nn import Lookahead
import torch.nn as nn
import numpy as np
from torch import optim
import utils.visualization as visual
from utils import data_loader_original as data_loader
from torch.optim import lr_scheduler
from tqdm import tqdm
import random
from utils.utils import clip_gradient, adjust_lr
from utils.metrics import Evaluator
import json
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from network.CGNet import HCGMNet, CGNet

import time
import sys

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, pred, target):
        # BCE Loss
        bce_loss = self.bce(pred, target)
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred_sigmoid * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)
        
        return bce_loss + dice_loss

start=time.time()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_visualizations(epoch, A, B, Y, preds, gates, save_path, filename_prefix="sample"):
    """Save visualization of predictions and gate masks"""
    # Create visualization directory
    viz_dir = os.path.join(save_path, 'visualizations', f'epoch_{epoch}')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Convert tensors to numpy for visualization
    # Handle batch dimension - take first sample from batch
    A_np = A[0].cpu().numpy().transpose(1, 2, 0)  # Remove batch dim, keep CHW -> HWC
    B_np = B[0].cpu().numpy().transpose(1, 2, 0)  # Remove batch dim, keep CHW -> HWC
    Y_np = Y[0].cpu().numpy().squeeze()  # Remove batch and channel dims
    
    # Normalize images to 0-255 range for visualization
    A_np = ((A_np - A_np.min()) / (A_np.max() - A_np.min()) * 255).astype(np.uint8)
    B_np = ((B_np - B_np.min()) / (B_np.max() - B_np.min()) * 255).astype(np.uint8)
    
    # Process predictions - index [0] to take the first sample from the batch,
    # then squeeze to remove any remaining size-1 dimensions.
    coarse_pred = F.sigmoid(preds[0])[0].detach().cpu().numpy().squeeze()
    fine_pred   = F.sigmoid(preds[1])[0].detach().cpu().numpy().squeeze()
    
    # Apply threshold for binary predictions
    coarse_binary = (coarse_pred >= 0.5).astype(np.uint8) * 255
    fine_binary   = (fine_pred   >= 0.5).astype(np.uint8) * 255
    ground_truth  = (Y_np        >= 0.5).astype(np.uint8) * 255
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Epoch {epoch} - Sample Visualization', fontsize=16)
    
    # Top row: Input images and ground truth
    axes[0, 0].imshow(A_np)
    axes[0, 0].set_title('Image A (Before)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(B_np)
    axes[0, 1].set_title('Image B (After)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(ground_truth, cmap='gray')
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(coarse_binary, cmap='gray')
    axes[0, 3].set_title('Coarse Prediction')
    axes[0, 3].axis('off')
    
    # Bottom row: Fine prediction and gate masks
    axes[1, 0].imshow(fine_binary, cmap='gray')
    axes[1, 0].set_title('Fine Prediction')
    axes[1, 0].axis('off')
    
    # Gate masks visualization
    if gates is not None:
        # Index [0] to select the first sample from the batch before squeezing.
        gate1_np = gates[0][0].detach().cpu().numpy().squeeze()
        gate2_np = gates[1][0].detach().cpu().numpy().squeeze()
        gate3_np = gates[2][0].detach().cpu().numpy().squeeze()
        
        axes[1, 1].imshow(gate1_np, cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title('Gate 1 (SSM1)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(gate2_np, cmap='hot', vmin=0, vmax=1)
        axes[1, 2].set_title('Gate 2 (SSM2)')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(gate3_np, cmap='hot', vmin=0, vmax=1)
        axes[1, 3].set_title('Gate 3 (SSM3)')
        axes[1, 3].axis('off')
    else:
        # If no gates, show fine prediction again or leave empty
        axes[1, 1].text(0.5, 0.5, 'No Gates\n(CGNet)', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Gate 1')
        axes[1, 1].axis('off')
        
        axes[1, 2].text(0.5, 0.5, 'No Gates\n(CGNet)', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Gate 2')
        axes[1, 2].axis('off')
        
        axes[1, 3].text(0.5, 0.5, 'No Gates\n(CGNet)', ha='center', va='center', transform=axes[1, 3].transAxes)
        axes[1, 3].set_title('Gate 3')
        axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{filename_prefix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization for epoch {epoch} at {viz_dir}/{filename_prefix}.png")

def train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, net, criterion, optimizer, num_epoches, device):
    vis = visual.Visualization()
    vis.create_summary(data_name)
    global best_iou, best_f1, best_precision, best_recall, best_oa, best_kappa
    global best_metrics, all_metrics
    
    # Initialize best metrics tracking
    if not hasattr(train, 'best_metrics'):
        train.best_metrics = {
            'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 
            'oa': 0.0, 'kappa': 0.0, 'epoch': 0
        }
        train.all_metrics = []
    
    epoch_loss = 0
    net.train(True)

    length = 0
    st = time.time()
    
    # Save visualizations for first few samples of each epoch
    viz_samples_saved = False
    
    for i, (A, B, mask) in enumerate(tqdm(train_loader)):
        A = A.to(device)
        B = B.to(device)
        Y = mask.to(device).float()  # The original data loader already returns proper shape
        optimizer.zero_grad()
        
        # Get predictions and gates for visualization
        if hasattr(net, 'ssm1') and opt.model_type == 'CGNet_SSM':
            # For CGNet_SSM, get predictions and gate masks
            coarse_pred, fine_pred, gates = net(A, B)
            preds = (coarse_pred, fine_pred)
        else:
            preds = net(A, B)
            gates = None
        
        # For visualization, save first few samples
        if not viz_samples_saved and i < 3:  # Save first 3 samples
            try:
                # Ensure we have the right format for CGNet_SSM
                if gates is not None:
                    # Convert tuple of gates to list for visualization function
                    gates_list = list(gates) if isinstance(gates, tuple) else gates
                else:
                    gates_list = None
                
                # Debug: Print tensor shapes and types
                print(f"Debug - A shape: {A.shape}, B shape: {B.shape}, Y shape: {Y.shape}")
                print(f"Debug - preds type: {type(preds)}, preds[0] shape: {preds[0].shape}, preds[1] shape: {preds[1].shape}")
                if gates_list is not None:
                    print(f"Debug - gates_list type: {type(gates_list)}, length: {len(gates_list)}")
                    for j, gate in enumerate(gates_list):
                        print(f"Debug - gate {j} shape: {gate.shape}")
                
                save_visualizations(epoch, A.cpu(), B.cpu(), Y.cpu(), preds, gates_list, save_path, f"train_sample_{i}")
                viz_samples_saved = True
                print(f"✓ Successfully saved visualization for epoch {epoch}, sample {i}")
            except Exception as e:
                print(f"Warning: Could not save visualization for epoch {epoch}, sample {i}: {e}")
                import traceback
                traceback.print_exc()
        # For CGNet_SSM and CGNet, use Both Maps (Deep Supervision)
        # preds[0] is coarse map, preds[1] is final refined map
        loss = criterion(preds[0].float(), Y) + criterion(preds[1].float(), Y)
        
        # Add L1 regularization for RPSS gates to encourage selectivity
        if hasattr(net, 'ssm1') and hasattr(net.ssm1, 'gate'):
            gate_l1_loss = 0.0
            for ssm_module in [net.ssm1, net.ssm2, net.ssm3]:
                # Get the gate output for regularization
                # Note: We need to compute this during forward pass
                # For now, we'll add a small L1 penalty on gate weights
                gate_params = list(ssm_module.gate.parameters())
                if gate_params:
                    gate_l1_loss += sum(p.abs().sum() for p in gate_params)
            
            # Add L1 regularization to the loss
            loss += 0.001 * gate_l1_loss
        
        # ---- loss function ----
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()

        output = F.sigmoid(preds[1])
        output[output >= 0.5] = 1  # Restored threshold to 0.5 from original paper
        output[output < 0.5] = 0
        pred = output.squeeze(1).data.cpu().numpy().astype(int)  # Remove channel dimension
        target = Y.squeeze(1).cpu().numpy().astype(int)  # Remove channel dimension
        
        # Normalize labels from 0-255 to 0-1 range if needed
        if target.max() > 1:
            target = target // 255
        
        Eva_train.add_batch(target, pred)

        length += 1
    
    # Training metrics
    train_iou = Eva_train.Intersection_over_Union()[1]
    train_pre = Eva_train.Precision()[1]
    train_recall = Eva_train.Recall()[1]
    train_f1 = Eva_train.F1()[1]
    train_oa = Eva_train.OA()
    train_kappa = Eva_train.Kappa()
    train_loss = epoch_loss / length

    # Add training metrics to visualization
    vis.add_scalar(epoch, train_iou, 'train_mIoU')
    vis.add_scalar(epoch, train_pre, 'train_Precision')
    vis.add_scalar(epoch, train_recall, 'train_Recall')
    vis.add_scalar(epoch, train_f1, 'train_F1')
    vis.add_scalar(epoch, train_oa, 'train_OA')
    vis.add_scalar(epoch, train_kappa, 'train_Kappa')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training] IoU: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, OA: %.4f, Kappa: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            train_iou, train_pre, train_recall, train_f1, train_oa, train_kappa))
    print("Starting validation!")


    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.to(device)
            B = B.to(device)
            Y = mask.to(device).float()  # The original data loader already returns proper shape
            preds = net(A,B)[1]
            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.squeeze(1).data.cpu().numpy().astype(int)  # Remove channel dimension
            target = Y.squeeze(1).cpu().numpy().astype(int)  # Remove channel dimension
            
            # Normalize labels from 0-255 to 0-1 range if needed
            if target.max() > 1:
                target = target // 255
            
            Eva_val.add_batch(target, pred)

            length += 1
    
    # Validation metrics
    val_iou = Eva_val.Intersection_over_Union()[1]
    val_pre = Eva_val.Precision()[1]
    val_recall = Eva_val.Recall()[1]
    val_f1 = Eva_val.F1()[1]
    val_oa = Eva_val.OA()
    val_kappa = Eva_val.Kappa()

    # Add validation metrics to visualization
    vis.add_scalar(epoch, val_iou, 'val_mIoU')
    vis.add_scalar(epoch, val_pre, 'val_Precision')
    vis.add_scalar(epoch, val_recall, 'val_Recall')
    vis.add_scalar(epoch, val_f1, 'val_F1')
    vis.add_scalar(epoch, val_oa, 'val_OA')
    vis.add_scalar(epoch, val_kappa, 'val_Kappa')

    print('[Validation] IoU: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, OA: %.4f, Kappa: %.4f' % (
        val_iou, val_pre, val_recall, val_f1, val_oa, val_kappa))
    
    # Create comprehensive metrics dictionary for this epoch
    current_metrics = {
        'epoch': epoch,
        'train_loss': float(train_loss),
        'train': {
            'iou': float(train_iou),
            'precision': float(train_pre),
            'recall': float(train_recall),
            'f1': float(train_f1),
            'oa': float(train_oa),
            'kappa': float(train_kappa)
        },
        'val': {
            'iou': float(val_iou),
            'precision': float(val_pre),
            'recall': float(val_recall),
            'f1': float(val_f1),
            'oa': float(val_oa),
            'kappa': float(val_kappa)
        }
    }
    
    # Store all metrics
    train.all_metrics.append(current_metrics)
    
    # Save all metrics to JSON file
    metrics_file = os.path.join(save_path, 'all_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(train.all_metrics, f, indent=2)
    
    # Check for best metrics and save checkpoints
    if val_iou >= train.best_metrics['iou']:
        train.best_metrics['iou'] = val_iou
        train.best_metrics['f1'] = val_f1
        train.best_metrics['precision'] = val_pre
        train.best_metrics['recall'] = val_recall
        train.best_metrics['oa'] = val_oa
        train.best_metrics['kappa'] = val_kappa
        train.best_metrics['epoch'] = epoch
        
        best_net = net.state_dict()
        print('New Best Model - IoU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f, OA: %.4f, Kappa: %.4f, Epoch: %d' % (
            val_iou, val_f1, val_pre, val_recall, val_oa, val_kappa, epoch))
        
        # Save best model checkpoint with model name in filename
        model_name_suffix = opt.model_type  # CGNet or CGNet_SSM
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save checkpoint with model name
        checkpoint_filename = f'best_model_checkpoint_{model_name_suffix}_{timestamp}.pth'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': best_net,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'metrics': current_metrics,
            'model_type': opt.model_type,
            'data_name': data_name,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        torch.save(checkpoint, os.path.join(save_path, checkpoint_filename))
        
        # Save weights with model name
        weights_filename = f'best_model_weights_{model_name_suffix}_{timestamp}.pth'
        torch.save(best_net, os.path.join(save_path, weights_filename))
        
        # Save best metrics summary
        best_metrics_summary = {
            'best_epoch': epoch,
            'model_type': opt.model_type,
            'data_name': data_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': train.best_metrics
        }
        with open(os.path.join(save_path, 'best_metrics.json'), 'w') as f:
            json.dump(best_metrics_summary, f, indent=2)
    
    print('Current Best - IoU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f, OA: %.4f, Kappa: %.4f, Epoch: %d' % (
        train.best_metrics['iou'], train.best_metrics['f1'], train.best_metrics['precision'], 
        train.best_metrics['recall'], train.best_metrics['oa'], train.best_metrics['kappa'], train.best_metrics['epoch']))
    
    vis.close_summary()


if __name__ == '__main__':
    seed_everything(42)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number') #修改这里！！！
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size') #修改这里！！！
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='2', help='train use gpu')  #修改这里！！！
    parser.add_argument('--data_name', type=str, default='WHU', #修改这里！！！
                        help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='CGNet',
                        help='the test rgb images root')
    parser.add_argument('--model_type', type=str, default='CGNet',
                        choices=['CGNet', 'CGNet_SSM'],
                        help='Model type: CGNet (original) or CGNet_SSM (with RecursivePriorStateSpace)')
    parser.add_argument('--save_path', type=str,
                        default='./output/')
    parser.add_argument('--offline_aug_num', type=int, default=1,
                        help='Number of offline augmentations per image (0 to disable)')
    opt = parser.parse_args()

    # set the device for training
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = int(opt.gpu_id) if opt.gpu_id.isdigit() else 0
        
        # Check if requested GPU exists, otherwise use first available
        if gpu_id >= num_gpus:
            print(f'Warning: GPU {gpu_id} not available (only {num_gpus} GPU(s) detected)')
            gpu_id = 0
            print(f'Falling back to GPU 0')
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device('cuda:0')  # After setting CUDA_VISIBLE_DEVICES, use cuda:0
        print(f'Using GPU {gpu_id} (total {num_gpus} GPU(s) available)')
    else:
        device = torch.device('cpu')
        print('Warning: No CUDA GPUs available - using CPU (training will be slow)')

    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name
    
    # Configure dataset paths (use local relative paths)
    dataset_base = './data'
    
    # Check if dataset exists and download if needed
    def check_and_download_dataset(data_name):
        dataset_path = os.path.join(dataset_base, f'{data_name}-CD')
        if not os.path.exists(dataset_path):
            print(f"Dataset {data_name}-CD not found at {dataset_path}")
            print(f"Please download the {data_name}-CD dataset and extract it to {dataset_path}")
            print("You can download it from the official source or use the download_dataset.py script")
            return False
        return True
    
    if opt.data_name == 'LEVIR':
        opt.train_root = os.path.join(dataset_base, 'LEVIR-CD', 'train') + '/'
        opt.val_root = os.path.join(dataset_base, 'LEVIR-CD', 'val') + '/'
        if not check_and_download_dataset('LEVIR'):
            sys.exit(1)
    elif opt.data_name == 'WHU':
        opt.train_root = os.path.join(dataset_base, 'WHU-CD', 'train') + '/'
        opt.val_root = os.path.join(dataset_base, 'WHU-CD', 'val') + '/'
        if not check_and_download_dataset('WHU'):
            sys.exit(1)
    elif opt.data_name == 'CDD':
        opt.train_root = os.path.join(dataset_base, 'CDD', 'train') + '/'
        opt.val_root = os.path.join(dataset_base, 'CDD', 'val') + '/'
        if not check_and_download_dataset('CDD'):
            sys.exit(1)
    elif opt.data_name == 'DSIFN':
        opt.train_root = os.path.join(dataset_base, 'DSIFN', 'train') + '/'
        opt.val_root = os.path.join(dataset_base, 'DSIFN', 'val') + '/'
        if not check_and_download_dataset('DSIFN'):
            sys.exit(1)
    elif opt.data_name == 'SYSU':
        opt.train_root = os.path.join(dataset_base, 'SYSU-CD', 'train') + '/'
        opt.val_root = os.path.join(dataset_base, 'SYSU-CD', 'val') + '/'
        if not check_and_download_dataset('SYSU'):
            sys.exit(1)
    elif opt.data_name == 'S2Looking':
        opt.train_root = os.path.join(dataset_base, 'S2Looking', 'train') + '/'
        opt.val_root = os.path.join(dataset_base, 'S2Looking', 'val') + '/'
        if not check_and_download_dataset('S2Looking'):
            sys.exit(1)

    # ---> Intégration de l'augmentation hors-ligne <---
    if hasattr(opt, 'offline_aug_num') and opt.offline_aug_num > 0:
        path_A = os.path.join(opt.train_root, 'A')
        if os.path.exists(path_A):
            # Vérifier si les augmentations "offline" ont déjà été faites
            existing_aug = len([f for f in os.listdir(path_A) if '_aug' in f])
            if existing_aug == 0:
                print(f"[*] Aucune augmentation hors-ligne détectée dans {path_A}.")
                print(f"[*] Lancement de la génération ({opt.offline_aug_num} copies / image)...")
                import subprocess
                subprocess.run(['python', 'offline_augmentation.py', 
                                '--dataset_path', opt.train_root, 
                                '--aug_num', str(opt.offline_aug_num)])
            else:
                print(f"[*] Augmentations hors-ligne déjà présentes ({existing_aug} repérées). On utilise le dataset tel quel.")

    train_loader = data_loader.get_loader(opt.train_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=True, pin_memory=True)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=False, pin_memory=True)
    Eva_train = Evaluator(num_class = 2)
    Eva_val = Evaluator(num_class=2)

    # Load model based on model_type
    if opt.model_type == 'CGNet':
        model = CGNet().to(device)
        print(f"✓ Loaded CGNet (original)")
    elif opt.model_type == 'CGNet_SSM':
        from network.CGNet_SSM import CGNet_SSM
        model = CGNet_SSM().to(device)
        print(f"Loaded CGNet_SSM (with RecursivePriorStateSpace)")
    else:
        raise ValueError(f"Unknown model_type: {opt.model_type}. Choose from: CGNet, CGNet_SSM")
    
    # Legacy support: allow model_name parameter (maps to model_type for backward compatibility)
    if opt.model_name != 'CGNet' and opt.model_name != opt.model_type:
        print(f"⚠ Warning: model_name is deprecated, use model_type instead")
        if opt.model_name == 'CGNet_SSM':
            opt.model_type = 'CGNet_SSM'


    # Weighted BCE + Dice Loss to handle class imbalance
    # Increase weight for changed pixels, but lowered to 2.0 to improve Precision
    pos_weight = torch.tensor([2.0]).to(device)  # Reduced from 5.0 to 2.0
    criterion = BCEDiceLoss(pos_weight=pos_weight).to(device)


    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    #base_optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_name = opt.data_name
    best_iou = 0.0

    print("Start train...")
    # args = parser.parse_args()
    # print('现在的数据是：',args.data_name)


    for epoch in range(1, opt.epoch):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        # cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        Eva_train.reset()
        Eva_val.reset()
        train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, model, criterion, optimizer, opt.epoch, device)
        lr_scheduler.step()
        # print('现在的数据是：', args.data_name)

end=time.time()
print('程序训练train的时间为:',end-start)