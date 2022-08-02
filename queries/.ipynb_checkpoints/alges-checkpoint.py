from sklearn.cluster import kmeans_plusplus
from model.loss import MixedLoss
from .query import Query
import numpy as np
import torch
import torch.nn.functional as F

class ALGESv1(Query):
    def __init__(self, idxs_lb, model, dm):
        super(ALGESv1, self).__init__(idxs_lb, model, dm)
        self.n_class = model.n_class

    def query(self, n):
        self.model.train().cuda()
        mixed_loss = MixedLoss()
        
        unlb_embs = torch.zeros((len(self.unlb_dataloader), self.n_class))
        
        with torch.enable_grad():
            for i, (x, y) in enumerate(self.unlb_dataloader):
                x = x.cuda()
                out, _, _ = self.model(x)
                
                prob = self.softmax(out)
                prob = prob.permute(0, 2, 3, 1).contiguous().view(-1, y.shape[1])
    
                y_hat = F.one_hot(prob.argmax(1), y.shape[1])
                y_hat = y_hat.view(y.shape[0], y.shape[2], y.shape[3], y.shape[1]).contiguous().permute(0, 3, 1, 2).double()
            
                loss = mixed_loss(out, y_hat)
                loss.backward()
                
                G = self.model.conv_last.weight.grad
                gx = G.norm(dim=1, p=2).squeeze()
                
                unlb_embs[i] = gx
                
        _, indices = kmeans_plusplus(unlb_embs.cpu().detach().numpy(), n_clusters=n, random_state=0)
        
        return np.take(np.where(self.idxs_lb==False), indices)
    
    
class ALGESv2(Query):
    def __init__(self, idxs_lb, model, dm):
        super(ALGESv2, self).__init__(idxs_lb, model, dm)
        self.n_class = model.n_class

    def query(self, n):
        self.model.eval().cuda()
        U = torch.zeros((len(self.unlb_dataloader), self.n_class))
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        self.model.conv_original_size2.register_forward_hook(get_activation('conv_original_size2'))
        
        with torch.no_grad():
            for i, (x, y) in enumerate(self.unlb_dataloader):
                activation = {}
                
                x = x.cuda()
                out, _, _ = self.model(x)
                
                P = self.softmax(out).squeeze()
                P_ = P.argmax(0)
                Z = activation['conv_original_size2'].squeeze()

                G = (P**2 + 1 - 2*P.max(0)[0]).sum(0) * (Z.norm(dim=0, p=2)**2)
                
                for k in range(self.n_class):
                    U[i, k] = G[P_ == k].sum()
                    
        _, indices = kmeans_plusplus(U.cpu().detach().numpy(), n_clusters=n, random_state=0)
        
        return np.take(np.where(self.idxs_lb==False), indices)
    
    