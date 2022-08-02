import os
import glob
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from datamodules.m2caiseg_dataset import M2CAISEGDataset
import datamodules.transforms as T 


class M2CAISEGDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.root = '/pasteur/u/josaklil/ALSSO/M2CAISEG'
        
        # Augmentations for train data
        self.train_trans = T.Compose([
            T.Resize(224, 224),
            T.PILToTensor(),
        ])
        
        # Only resizing for validation and test augmentations
        self.val_trans = T.Compose([
            T.Resize(224, 224),
            T.PILToTensor(),
        ])
        self.test_trans = self.val_trans
        
    def prepare_data(self):
        X_trainval = np.asarray(glob.glob(f'{self.root}/train/images/**/*.jpg', recursive=True))
        Y_trainval = np.asarray([os.path.splitext(path.replace('images', 'groundtruth'))[0] + '_gt.png'
                                for path in X_trainval])
        
        self.X_te = np.asarray(glob.glob(f'{self.root}/test/images/**/*.jpg', recursive=True))
        self.Y_te = np.asarray([os.path.splitext(path.replace('images', 'groundtruth'))[0] + '_gt.png'
                                for path in self.X_te])
        
        # follow m2caiseg original train/test split, but use 1/8 of train for val
        self.X_tr, self.X_val, self.Y_tr, self.Y_val = train_test_split(
            X_trainval, Y_trainval, test_size=0.125
        )
        
        self.X_val = self.X_val[0:self.args.val_size]
        self.Y_val = self.Y_val[0:self.args.val_size]
        
        self.X_te = self.X_te[0:self.args.test_size]
        self.Y_te = self.Y_te[0:self.args.test_size]
        
    def setup(self, stage=None):
        self.all = M2CAISEGDataset(self.X_tr, self.Y_tr, self.train_trans) # train + unlb
        self.val = M2CAISEGDataset(self.X_val, self.Y_val, self.val_trans)
        self.test = M2CAISEGDataset(self.X_te, self.Y_te, self.test_trans)
        
    def update_pool(self, inc_idxs):
        # Train dataset should only include inputs for current active learning round
        self.lb = M2CAISEGDataset(self.X_tr[inc_idxs], self.Y_tr[inc_idxs], self.train_trans)
        
        # Unlabeled used for querying samples for next round
        exc_idxs = ~inc_idxs
        self.unlb = M2CAISEGDataset(self.X_tr[exc_idxs], self.Y_tr[exc_idxs], self.train_trans)   

    def train_dataloader(self):
        return DataLoader(self.lb, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, shuffle=False)
    
    def unlb_dataloader(self):
        return DataLoader(self.unlb, batch_size=1, shuffle=False)
    
    def all_dataloader(self):
        return DataLoader(self.all, batch_size=1, shuffle=False)
    
    def compute_lb_dist(self):
        dist = torch.zeros(13)
        lb_dataloader = self.train_dataloader()
        
        for i, (x, y) in enumerate(lb_dataloader):
            for j in range(13):
                dist[j] += y[:, j, :, :].sum()
            
        return dist
    
    