from .query import Query
import numpy as np

class Random(Query):
    def __init__(self, idxs_lb, model, dm):
        super(Random, self).__init__(idxs_lb, model, dm)
        
    def query(self, n):
        return np.random.choice(np.where(self.idxs_lb==False)[0], n, replace=False)
    
    