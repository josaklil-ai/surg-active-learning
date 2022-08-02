import torch
from torch import nn
import torch.nn.functional as F

class DiceCoeff(nn.Module):
    def __init__(self, args):
        super(DiceCoeff, self).__init__()
        self.args = args
        
    def forward(self, logits, targets, smooth=1.):
        N, C, H, W = targets.shape
        
        probs = F.softmax(logits, dim=1)
        
        # Multiclass dice coefficient
        preds = F.one_hot(probs.argmax(1), C).permute(3, 0, 1, 2).contiguous().view(C, -1)
        targets = targets.permute(1, 0, 2, 3).contiguous().view(C, -1)
        
        intersection = (preds * targets)  
        class_dice = (2. * intersection.sum(1) + smooth) / (preds.sum(1) + targets.sum(1) + smooth)
        
        if self.args.dataset == 'cholecseg8k':
            present_in_val = [0, 1, 2, 3, 4, 5, 9, 10] # exclude classes not present in val for more accurate dice 
            class_dice = class_dice[present_in_val]
        
        return class_dice
    
