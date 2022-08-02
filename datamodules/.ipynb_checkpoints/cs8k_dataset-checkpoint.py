from PIL import Image
import torch 
from torchvision.datasets import VisionDataset

class CS8KDataset(VisionDataset):
    def __init__(self, X, Y, transforms=None):      
        self.X = X
        self.Y = Y
        self.transforms = transforms
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = Image.open(self.X[idx])
        mask = Image.open(self.Y[idx])
 
        if self.transforms:
            image, mask = self.transforms(image, mask)
        
        C = 13 # num semantic classes for cholecseg8k
        
        masks = torch.zeros((C, image.shape[1], image.shape[2]))
        for c in range(0, C):
            masks[c, :, :] = torch.where(mask == c, 1, 0)[:, :, 0].float()
 
        return image, masks


