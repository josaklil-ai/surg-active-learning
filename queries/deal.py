from .query import Query
import numpy as np
import torch

class DEAL(Query):
    def __init__(self, idxs_lb, model, dm):
        super(DEAL, self).__init__(idxs_lb, model, dm)
        
    def query(self, n):
        self.model.eval().cuda()
        U = torch.zeros(len(self.unlb_dataloader))

        with torch.no_grad():
            for i, (x, y) in enumerate(self.unlb_dataloader):
                x = x.cuda()
                out, _, diff_map = self.model(x)
                diff_map = diff_map.detach().cpu().numpy().squeeze(1)
                region_areas, score_ticks = np.histogram(diff_map, bins=10)
                probs = region_areas / region_areas.sum()
                entropy = -np.nansum(np.multiply(probs, np.log(probs + 1e-12)))
                U[i] = entropy
        
        return np.take(np.where(self.idxs_lb==False), U.sort(descending=True)[1][:n])
    
    