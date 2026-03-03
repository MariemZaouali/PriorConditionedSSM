import torch
import torch.nn as nn
import torch.nn.functional as F


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).add_(1e-5)
            j = y_pred.sum(1).add_(1e-5)
            intersection = (y_true * y_pred).sum(1)
        l = torch.mean(intersection / (i + j - intersection + smooth))
        return l

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(torch.sigmoid(y_pred), y_true)
        b = self.soft_dice_loss(y_true, torch.sigmoid(y_pred))
        return a + b
