import numpy as np


class Perceptron:
    '''
    Linear Perceptron for Regression with any order
    polynomial transform, data must have 2 features
    '''
    def __init__(self, data, order, learning_rate=0.0):
        self.data_t = self.transform_features(data, order, labels=True)
        self.w = self.linear_regression(data, l=learning_rate)
        self.order = order

    def linear_regression(self, data, l=0.0):
        '''
        Performs linear regression pseudo-inverse algorithm to
        obtain a weight vector

        Parameters
        ----------
        data : np.array
            data with features and label as last column
        l : float, optional
            regularization parameter lambda	

        Returns
        -------
        w : np.array
            list of weights for linear model
        '''
        x_data = data[:, :-1]
        y_data = data[:, -1]
        pseudo_x = np.dot(np.linalg.inv(np.dot(np.transpose(x_data), x_data) + l*np.eye(data.shape[1]-1)), 
                          np.transpose(x_data))
        w = np.dot(pseudo_x, y_data)
        return w


    def legendre_table(self, x, k):
        '''
        Creates dynamic programming table
        for values of legendre transform

        Parameters
        ----------
        x : float
            value to operate on
        k : int
            largest exponent to compute

        Returns
        -------
        list
            dynamic programming table of legendre tranform
        '''
        dp = [0 for _ in range(k+1)]
        dp[0] = 1
        dp[0] = x
        for i in range(2, k+1):
            dp[i] = (((2*k-1)/k)*x*dp[i-1])-(((k-1)/k)*dp[i-2])
        return dp
        

    def compute_features(self, x, order):
        '''
        Computes the new features based on the legendre
        polynomial transform

        Parameters
        ----------
        x : np.array
            data point to transform
        order : int
            largest exponent of transformed features

        Returns
        -------
        np.array
            transformed features of data point
        '''
        l_x1 = legendre_table(x[0], order)
        l_x2 = legendre_table(x[1], order)
        features = [1, l_x1[1], l_x2[1]]
        for i in range(2, order+1):
            j = i
            while j >= 0:
                features.append(l_x1[j]*l_x2[i-j])
                j -= 1
        return np.array(features)

    def transform_features(self, xx, order, labels=False):
        transformed_data = []
        for x in xx:
            features = self.compute_features([x[0], x[1]], order)
            if labels:
                features.append(x[-1])
                transformed_data.append(features)
        return np.array(transformed_data)	


    def predict(self, xx):
        '''
        Makes prediction using learned weight vector

        Parameters
        ----------
        xx : np.array
            data to predict
        
        Returns
        -------
        np.array
            class labels (-1 or 1) for each data point
        '''
        xx_t = self.transform_features(xx, self.order)
        return np.sign(np.matmul(xx_t, self.w))

