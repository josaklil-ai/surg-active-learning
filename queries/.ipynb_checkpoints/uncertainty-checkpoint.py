from .query import Query
import numpy as np
import torch

class MaxEntropy(Query):
    def __init__(self, idxs_lb, model, dm):
        super(MaxEntropy, self).__init__(idxs_lb, model, dm)

    def query(self, n):
        self.model.eval().cuda()
        U = torch.zeros(len(self.unlb_dataloader))
                                                      
        with torch.no_grad():
            for i, (x, y) in enumerate(self.unlb_dataloader):
                x = x.cuda()
                out, _, _ = self.model(x)
                prob = self.softmax(out).squeeze()
                u = -(prob * torch.log(prob))
                u = u.sum(0).sum()
                U[i] = u
        
        return np.take(np.where(self.idxs_lb==False), U.sort(descending=True)[1][:n])

    
class Margins(Query):
    def __init__(self, idxs_lb, model, dm):
        super(Margins, self).__init__(idxs_lb, model, dm)

    def query(self, n):
        self.model.eval().cuda()
        U = torch.zeros(len(self.unlb_dataloader))
        
        with torch.no_grad():
            for i, (x, y) in enumerate(self.unlb_dataloader):
                x = x.cuda()
                out, _, _ = self.model(x)
                prob = self.softmax(out).squeeze()
                probs_sorted, _ = prob.sort(dim=0, descending=True)
                U[i] = (probs_sorted[0, :, :] - probs_sorted[1, :, :]).sum()
        
        return np.take(np.where(self.idxs_lb==False), U.sort()[1][:n])

    
class LeastConf(Query):
    def __init__(self, idxs_lb, model, dm):
        super(LeastConf, self).__init__(idxs_lb, model, dm)

    def query(self, n):
        self.model.eval().cuda()
        U = torch.zeros(len(self.unlb_dataloader))
        
        with torch.no_grad():
            for i, (x, y) in enumerate(self.unlb_dataloader):
                x = x.cuda()
                out, _, _ = self.model(x)
                prob = self.softmax(out).squeeze()
                u = prob.max(0)[0].sum()
                U[i] = u
        
        return np.take(np.where(self.idxs_lb==False), U.sort()[1][:n])
    
    