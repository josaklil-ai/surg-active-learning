import torch
from torch import nn
import torch.nn.functional as F

class MixedLoss(nn.Module):
    def __init__(self):
        super(MixedLoss, self).__init__()

    def forward(self, logits, targets, smooth=1.):
        N, C, H, W = targets.shape
        
        probs = F.softmax(logits, dim=1)
        
        # CE loss
        ce_loss = F.cross_entropy(logits, targets)
        
        # Dice loss
        probs = probs.permute(1, 0, 2, 3).contiguous().view(C, -1)
        targets = targets.permute(1, 0, 2, 3).contiguous().view(C, -1)
        
        intersection = (probs * targets)
        class_dice = (2. * intersection.sum(1) + smooth) / (probs.sum(1) + targets.sum(1) + smooth)
        
        dice_loss = 1 - class_dice.mean()

        # return ce_loss + dice_loss
        return ce_loss
        # return dice_loss
        
