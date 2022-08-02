import numpy as np
import torch

class Query:
    def __init__(self, idxs_lb, model=None, dm=None):
        
        self.idxs_lb = idxs_lb
        
        self.model = model
        self.softmax = torch.nn.Softmax2d()
        
        self.all_dataloader = dm.all_dataloader()
        self.unlb_dataloader = dm.unlb_dataloader()
        
    def query(self, n):
        pass
    
    
class Naive(Query):
    def __init__(self, idxs_lb, model, dm):
        super(Naive, self).__init__(idxs_lb, model, dm)
    
    def query(self, n):
        return np.where(self.idxs_lb==False)[0][0:n]

    