import numpy as np
from tqdm import tqdm


class SVM:
    def __init__(self, data, c=0.0):
        '''
        Initializes the SVM and constructs the optimal hyperplane
        
        Parameters
        ----------
        data : np.array
            input data to construct hyperplane with
        c : float
            the regularization parameter (non-separable data)
        '''
        self.alphas, self.xx, self.yy = self.solve(data[:, :-1], data[:, -1], c)
        total = 0
        for i in range(self.alphas.shape[0]):
            total += self.alphas[i]*self.yy[i]*self.kernel(self.xx[i], self.xx[0])
        self.bias = self.yy[0] - total
        
    
    def kernel(self, x1, x2):
        '''
        Computes the transformation based on the 8th order polynomial kernel
        
        Parameters
        ----------
        x1 : np.array
            array of first data points
        x2 : np.array
            array of second data points
        
        Returns
        -------
        np.array
            array of pairwise of this -> (1 + x1^T x2)^8
        '''
        return (1 + np.dot(x1, x2))**8
    
    
    def solve(self, xx, yy, c):
        '''
        Solves the Quadratic Programming SVM problem for
        the data and regularization parameter
        
        Parameters
        ----------
        xx : np.array
            data points
        yy : np.array
            data labels
        c : float
            the regularization parameter (non-separable data)
        
        Returns
        -------
        np.array
            weights of the optimal hyperplane
        '''
        # gram matrix
        N = xx.shape[0]
        Q = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Q[i][j] = yy[i]*yy[j]*self.kernel(xx[i], xx[j])
        G = np.concatenate((-np.eye(N), np.eye(N)), axis=0)
        h = np.concatenate((np.zeros((N, 1)), np.full((N, 1), c)), axis=0)
        P_mat = matrix(Q, tc='d')
        q_mat = matrix(-np.ones(N), tc='d')
        G_mat = matrix(G, tc='d')
        h_mat = matrix(h, tc='d')
        yy_A = yy.reshape(1, -1)
        A_mat = matrix(yy_A, tc='d')
        b_mat = matrix(0.0, tc='d')
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        alphas = solvers.qp(P_mat, q_mat, G_mat, h_mat, A_mat, b_mat)
        sys.stdout = old_stdout
        alphas = np.array(alphas['x']).squeeze()
        mask = np.logical_and(alphas > 0.00001, alphas < c)
        alphas = alphas[mask].reshape(-1, 1)
        xx = xx[mask]
        yy = yy[mask].reshape(-1, 1)
        return alphas, xx, yy
    
    
    def predict_no_tqdm(self, xx):
        '''
        Makes class predictions based on weights established by optimal
        hyperplane
        
        Parameters
        ----------
        xx : np.array
            data to predict class labels for
        
        Returns
        -------
        np.array
            array of predictions for xx data
        '''
        preds = []
        for x in xx:
            total = 0
            for i in range(self.alphas.shape[0]):
                total += self.alphas[i]*self.yy[i]*self.kernel(self.xx[i], x)
            total += self.bias
            preds.append(np.sign(total))
        return np.array(preds)
    
    
    
    def predict(self, xx):
        '''
        Makes class predictions based on weights established by optimal
        hyperplane
        
        Parameters
        ----------
        xx : np.array
            data to predict class labels for
        
        Returns
        -------
        np.array
            array of predictions for xx data
        '''
        preds = []
        for x in tqdm(xx):
            total = 0.0
            for i in range(self.alphas.shape[0]):
                total += self.alphas[i]*self.yy[i]*self.kernel(self.xx[i], x)
            total += self.bias
            preds.append(np.sign(total))
        return np.array(preds)
    
