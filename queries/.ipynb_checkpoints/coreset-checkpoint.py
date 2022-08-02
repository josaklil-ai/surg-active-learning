from .query import Query
import numpy as np
import torch
import copy

class CoreSet(Query):
    def __init__(self, idxs_lb, model, dm):
        super(CoreSet, self).__init__(idxs_lb, model, dm)
    
    def query(self, n):
        self.model.eval().cuda()
        
        N = len(self.all_dataloader)
        embs = torch.zeros((N, 3136))
        
        with torch.no_grad():
            for i, (x, y) in enumerate(self.all_dataloader):
                x = x.cuda()
                _, emb, _ = self.model(x)
                embs[i] = emb.flatten()
                
        embs = embs.numpy()
        
        distances = np.matmul(embs, embs.T)
        squares = np.array(distances.diagonal()).reshape(N, 1)
        
        distances *= -2
        distances += squares
        distances += squares.T
        distances = np.sqrt(distances)
        
        # matrix of distances between unlb and lb points
        D = distances[~self.idxs_lb, :][:, self.idxs_lb]  
        
        temp_lb = copy.deepcopy(self.idxs_lb)
        for i in range(n):
            closests = D.min(axis=1) # closest point in lb to each unlb
            closest_idxs_ = closests.argmax()
            # closest_idxs = np.arange(N)[~self.idxs_lb][closest_idxs_]
            closest_idxs = np.arange(N)[~temp_lb][closest_idxs_]
            temp_lb[closest_idxs] = True

            D = np.delete(D, closest_idxs_, 0)
            D = np.append(D, distances[~temp_lb, closest_idxs][:, None], axis=1)
            
        return np.where(temp_lb ^ self.idxs_lb)[0]
    
    