from PIL import Image
import torch 
import torch.nn.functional as F
from torchvision.datasets import VisionDataset

m2caiseg_cmap = {
    (170, 0, 85): 0, # 'Unknown':
    (0, 85, 170): 1, # 'Grasper': 
    (0, 85, 170): 2, # 'Bipolar': 
    (0, 170, 85): 3, # 'Hook': 
    (0, 255, 85): 4, # 'Scissors': 
    (0, 255, 170): 5, # 'Clipper': 
    (85, 0, 170): 6, # 'Irrigator':
    (85, 0, 255): 7, # 'Specimen Bag':
    (170, 85, 85): 8, # 'Trocars': 
    (170, 170, 170): 9, # 'Clip':
    (85, 170, 0): 10, # 'Liver': 
    (85, 170, 255): 11, # 'Gallbladder': 
    (85, 255, 0): 12, # 'Fat':
    (85, 255, 170): 13, # 'Upper Wall':
    (170, 0, 255): 14, # 'Artery': 
    (255, 0, 255): 15, # 'Intestine': 
    (255, 255, 0): 16, # 'Bile': 
    (255, 0, 0): 17, # 'Blood': 
    (0, 0, 0): 18, # 'Black':     
}

class M2CAISEGDataset(VisionDataset):
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
        
        masks = torch.zeros(mask.shape[:2])
        for c in m2caiseg_cmap:
            masks[(mask.numpy() == c).all(axis=2)] = m2caiseg_cmap[c]
            
        masks = F.one_hot(masks.to(torch.int64))
        masks = masks.permute(2, 0, 1).float()
 
        return image, masks

