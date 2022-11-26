import numpy as np

def mixup(x1, x2, y1, y2, alpha=0.0):
    ''' Applies mixup data augmentation '''
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y
