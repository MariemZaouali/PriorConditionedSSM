import torch
import numpy as np


def clip_gradient(optimizer, grad_clip):
    """
    For adaptive gradient clipping: https://arxiv.org/abs/1211.1541
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    """Decay learning rate by a factor of decay_rate every decay_epoch epochs."""
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr
    return decay * init_lr


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Calculate the computational complexity (FLOPs) and number of parameters of a model.
    Uses the thop library to compute FLOPs (Floating Point Operations).
    
    Args:
        model: The neural network model to analyze
        input_tensor: A sample input tensor to the model (e.g., torch.randn(1, 3, 256, 256))
    
    Returns:
        None (prints the results to console)
    """
    from thop import profile, clever_format
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print('flops : ', flops)
    print('params : ', params)
