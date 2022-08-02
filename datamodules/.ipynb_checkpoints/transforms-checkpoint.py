import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import numpy as np
np.random.seed(3507)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize(object):
    def __init__(self, height, width):
        self.h = height
        self.w = width

    def __call__(self, image, target):
        image = F.resize(image, (self.h, self.w))
        target = F.resize(target, (self.h, self.w), interpolation=T.InterpolationMode.NEAREST)
        return image, target 
    
class PILToTensor(object):
    def __call__(self, image, target):
        image = F.pil_to_tensor(image).float()
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

    