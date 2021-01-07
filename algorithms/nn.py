import numpy as np

class NeuralNetwork:
    def __init__(self, dims, out_transform='linear', value=None, variance=1.0):
        '''
        Constructor for a neural network with len(dims) number
        of layers (including input) each with dimensionality
        specified in dims. Hidden layer tranforms are tanh.
        
        Parameters
        ----------
        dims : list[int]
            the dimensionality of each layer in the network
        out_transform : str, optional
            the transformation to apply to the output node
        '''
        self.dims = dims
        self.out_transform = out_transform
        if not value is None:
            self.W = self.initialize_weights(dims, value=value)
        else:
            self.W = self.initialize_weights(dims, variance=variance)
        self.layer_outputs = []
        
    
    def initialize_weights(self, dims, value=None, variance=1.0):
        '''
        Initializes the weights of each layer
        in the network to random small values
        
        Parameters
        ----------
        dims : list[int]
            the dimensionality of each layer in the network
        value : float, optional
            set all the weights to this value
        
        Returns
        -------
        list[np.array]
            list of weight matrices for each layer
        '''
        weights = []
        for i in range(len(dims)-1):
            if value is None:
                weights.append(np.random.normal(scale=variance, size=(dims[i]+1, dims[i+1])))
            else:
                weights.append(np.full((dims[i]+1, dims[i+1]), value))
        return weights
    
    
    def forward(self, xx):
        '''
        Perform forward propogation of the neural network
        with a batch of data points
        
        Parameters
        ----------
        xx : np.array
            input data matrix with rows as individual points
            with augmented 1.0
        
        Returns
        -------
        np.array
            output of last layer (prediction) using self.out_transform
        '''
        self.layer_outputs = []
        for i in range(len(self.W)-1):
            xx = np.hstack((np.ones((xx.shape[0], 1)), xx))
            self.layer_outputs.append(xx)
            xx = np.matmul(xx, self.W[i])
            xx = np.tanh(xx)
        xx = np.hstack((np.ones((xx.shape[0], 1)), xx))
        self.layer_outputs.append(xx)
        xx = np.matmul(xx, self.W[-1])
        if self.out_transform == 'tanh':
            return np.tanh(xx)
        if self.out_transform == 'sign':
            return np.sign(xx)
        return xx
    
    
    def num_gradient(self, xx, peturb=0.0001):
        '''
        Calculates the numerical gradient
        for a batch of data points
        
        Parameters
        ----------
        xx : np.array
            data points to operate on
        yy : np.array
            actual labels for the input data
        preds : np.array
            predicted labels for the input data
        
        Returns
        -------
        np.array
            gradients of error for each layer
            should match shapes of matrices in self.W
        '''
        grads = []
        for l in range(0, len(self.dims)-1):
            grad = np.zeros(self.W[l].shape)
            for x in xx:
                for i in range(self.W[l].shape[0]):
                    for j in range(self.W[l].shape[1]):
                        self.W[l][i][j] -= peturb
                        preds = self.forward(np.array([x[:-1]]))
                        e1 = (preds[0] - x[-1])**2
                        self.W[l][i][j] += 2*peturb
                        preds = self.forward(np.array([x[:-1]]))
                        e2 = (preds[0] - x[-1])**2
                        self.W[l][i][j] -= peturb
                        grad[i][j] += ((1/(4*xx.shape[0]))*((e2 - e1) / (2*peturb)))
            grads.append(grad)
        return grads
                        
                        
    
    def backward(self, yy, preds, lamb=0.0):
        '''
        Peform backward propogation with gradient descent
        for a batch of data points
        
        Parameters
        ----------
        yy : np.array
            actual labels for the input data
        preds : np.array
            predicted labels for the input data
        l : float, optional
            regularization (weight decay) parameter
            
        Returns
        -------
        np.array
            gradients of error for each layer
            should match shapes of matrices in self.W
        '''
        # shape (N, 1)
        yy = yy.reshape(-1, 1)
        deltas = []
        gradients = []
        # shape (N, 1)
        deriv_L = np.ones((preds.shape[0], 1))
        if self.out_transform == 'tanh':
            deriv_L = 1 - np.multiply(preds, preds)
        # shape (N, 1) for output
        delta_L = np.multiply(2*(preds - yy), deriv_L)
        # shape (d[L-1], d[L]) for last weight matrix (contains sums)
        gradient_L = (1/(4*yy.shape[0]))*np.matmul(np.transpose(self.layer_outputs[-1]), delta_L) + (((2*lamb)/(4*yy.shape[0]))*self.W[-1])
        gradients.append(gradient_L)
        deltas.append(delta_L)
        for l in range(len(self.layer_outputs)-1, 0, -1):
            # theta'(s(l)) shape should be (N, d[l])
            deriv_l = (1- np.multiply(self.layer_outputs[l], self.layer_outputs[l]))[:, 1:]
            # sensitivity
            # shape (N, d[l])
            tmp = np.matmul(self.W[l], np.transpose(deltas[-1]))
            tmp = np.transpose(tmp)[:, 1:]
            # shape (N, d[l])
            delta_l = np.multiply(deriv_l, tmp)
            # gradient shape (d[l-1], d[l]) (contains sums over N points)
            gradient_l = (1/(4*yy.shape[0]))*np.matmul(np.transpose(self.layer_outputs[l-1]), delta_l) + (((2*lamb)/(4*yy.shape[0]))*self.W[l-1])
            gradients.append(gradient_l)
            deltas.append(delta_l)
        # ein and reverse order
        total = 0
        for w in self.W:
            total += np.sum(w**2)
        ein = np.sum((1/(4*yy.shape[0]))*((preds - yy)**2)) + ((lamb/(4*yy.shape[0]))*total)
        return ein, gradients[::-1]
    
    
    def predict(self, xx):
        preds = self.forward(xx)
        return np.sign(preds)

