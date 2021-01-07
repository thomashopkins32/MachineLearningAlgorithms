import numpy as np 
from tqdm import tqdm 


class KNN:
    '''
    Performs the k-Nearest Neighbor algorithm
    to predict class labels given new data
    '''
    def __init__(self, data, k): 
        self.data = data 
        self.k = k 


    def distance(self, x1, x2): 
        x2 = x2[:-1] 
        return np.linalg.norm(x1 - x2, ord=2) 


    def predict(self, x1): 
        counts = [] 
        for x in tqdm(x1): 
            data = sorted(self.data, key=lambda x2: self.distance(x, x2)) 
            count = 0 
            for k in range(self.k): 
                if data[k][-1] == 1.0: 
                    count += 1 
                elif data[k][-1] == -1.0: 
                    count -= 1 
            if count < 0: 
                counts.append(-1.0)
            else:
                counts.append(1.0)
        counts = np.array(counts)
        return counts


    def predict_no_tqdm(self, x1):
        counts = []
        for x in x1:
            data = sorted(self.data, key=lambda x2: self.distance(x, x2))
            count = 0
            for k in range(self.k):
                if data[k][-1] == 1.0:
                    count += 1
                elif data[k][-1] == -1.0:
                    count -= 1
            if count < 0:
                counts.append(-1.0)
            else:
                counts.append(1.0)
        counts = np.array(counts)
        return counts

