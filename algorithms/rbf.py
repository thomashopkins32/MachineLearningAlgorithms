import numpy as np
from tqdm import tqdm


class RBFNet:
    '''
    Radial Basis Function Network to predict
    class labels using guassian kernel
    '''
    def __init__(self, data, k):
        self.r = 2 / np.sqrt(k)
        self.data = data


    def distance(self, x1, x2):
        x2 = x2[:-1]
        return np.linalg.norm(x1 - x2, ord=2)
        
        
    def guassian_kernel(self, z):
            return (1/np.sqrt(2*np.pi))*np.exp(-0.5*(z**2))
        

    def weight(self, xn, denom):
        yn = xn[-1]
        return yn/denom


    def predict(self, xx):
        preds = []
        for x in tqdm(xx):
            full_total = 0
            total = 0
            for d in self.data:
                total += self.guassian_kernel(self.distance(x, d)/self.r)
            for d in self.data:
                full_total += self.weight(d, total)*self.guassian_kernel(self.distance(x, d)/self.r)
            preds.append(full_total)
        return np.sign(preds)


    def predict_no_tqdm(self, xx):
        preds = []
        for x in xx:
            full_total = 0
            total = 0
            for d in self.data:
                total += self.guassian_kernel(self.distance(x, d)/self.r)
            for d in self.data:
                full_total += self.weight(d, total)*self.guassian_kernel(self.distance(x, d)/self.r)
            preds.append(full_total)
        return np.sign(preds)

