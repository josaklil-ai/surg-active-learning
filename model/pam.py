import torch
from torch import nn
import torch.nn.functional as F

att_map_size = {
    13: (56, 56),  # CholecSeg8k
    19: (56, 56),  # m2caiSeg
}

class PAM(nn.Module):
    """
    ref: Positional attention module in `Dual attention network for scene segmentation`.
    """
    #
    def __init__(self):
        super(PAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))  
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        B, C, H, W = x.size()
        x = x.softmax(dim=1)

        if self.gamma < -0.01:
            out = x
            attention = None
        else:
            proj_value = x.view(B, -1, W * H)  
            proj_query = x.view(B, -1, W * H).permute(0, 2, 1)
            proj_key = x.view(B, -1, W * H)
            
            energy = torch.bmm(proj_query, proj_key)  
            attention = self.softmax(energy)

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(B, C, H, W)

            out = self.gamma * out + x

        return out, attention

    
class MaskHead(nn.Module):
    """
    ref: Mask head defined in `DEAL: Difficulty-awarE Active learning for Semantic Segmentation`.
    """
    def __init__(self, num_classes, with_pam=False):
        super(MaskHead, self).__init__()
        self.att_size = att_map_size[num_classes]
        self.pam = PAM().cuda() if with_pam else None
        self.mask_head = nn.Conv2d(num_classes, 1, kernel_size=1, stride=1, bias=True).cuda()
        self.with_pam = with_pam

    def forward(self, inputs):  # 1/4 feature
        if self.with_pam:
            x = F.interpolate(inputs, size=self.att_size, mode='bilinear', align_corners=True)
            feat, attention = self.pam(x)
            mask = self.mask_head(feat)
            return mask, attention
        else:
            mask = self.mask_head(inputs)
            return mask, None
        
        