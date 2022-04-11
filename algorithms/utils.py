import numpy as np
import torch
import scipy.signal
from torch import nn


def add_length_to_shape(length, shape=None):
    '''
    Combines an arbitrary shape with a preferred length.

    Parameters
    ----------
    length : int
        size of first axis
    shape : tuple[int], optional
        size of the rest of the axes
    '''
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    # LOOK THIS UP
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class TrajectoryBuffer:
    def __init__(self, obs_shape, action_shape, size, discount=0.99, lam=0.95):
        '''
        Stores the trajectories that the agent takes up to the buffer size.

        It will store for each step in the environment:
            - observation
            - immediate reward
            - action taken
            - probability of selecting that action (according to policy)
            - perceived value of the observation
        When the trajectory is finished it will compute:
            - discounted reward to go
            - discounted lambda advantage

        The buffer can be emptied by calling the `get()` method
        '''
        self.obs_buf = np.zeros(add_length_to_shape(size, obs_shape),
                                dtype=np.float32)
        self.act_buf = np.zeros(add_length_to_shape(size, action_shape),
                                dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)

        self.ptr = 0
        self.start_ptr = 0
        self.size = size

        self.discount = discount
        self.lam = lam

    def store(self, obs, action, reward, logp, value):
        ''' Store a single step in the buffer '''
        if self.ptr == self.size:
            print('Cannot store current step. Buffer is full.')
            return
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.logp_buf[self.ptr] = logp
        self.val_buf[self.ptr] = value
        self.ptr += 1

    def finish_trajectory(self, last_val=0.0):
        ''' Computes the return and advantage per step in trajactory '''
        path_slice = slice(self.start_ptr, self.ptr)
        rewards = np.append(self.rew_buf[path_slice], last_val)
        values = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage (LOOK THIS UP!)
        deltas = rewards[:-1] + self.discount * values[1:] - values[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas,
                                                   self.discount * self.lam)
        # Rewards-to-go (LOOK THIS UP!)
        self.rew_buf[path_slice] = discount_cumsum(rewards,
                                                   self.discount)[:-1]
        self.start_ptr = self.ptr

    def get(self):
        ''' Empties the buffer into something useable for learning '''
        if self.ptr != self.size:
            print('ERROR: buffer not full')
            return
        self.ptr = 0
        self.start_ptr = 0
        # advantage normalization (LOOK THIS UP!)
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = {'obs': self.obs_buf, 'act': self.act_buf, 'ret': self.ret_buf,
                'adv': self.adv_buf, 'logp': self.logp_buf}
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class MLP(nn.Module):
    def __init__(self, nodes_per_layer, activation='relu'):
        '''
        A basic multi-layered perceptron using PyTorch.

        Parameters
        ----------
        nodes_per_layer : List[int]
            number of nodes per layer in the network
        activation : str, optional
            description of activation to use in between layers
            supports: {'sigmoid', 'relu', 'tanh'}
        '''
        super().__init__()
        activ_func = None
        if activation == 'relu':
            activ_func = nn.ReLU()
        elif activation == 'sigmoid':
            activ_func = nn.Sigmoid()
        elif activation == 'tanh':
            activ_func = nn.Tanh()
        self.mlp = nn.Sequential()
        for i in range(1, len(nodes_per_layer)):
            self.mlp.append(nn.Linear(nodes_per_layer[i-1],
                                    nodes_per_layer[i]))
            if activ_func is not None and i != len(nodes_per_layer)-1:
                self.mlp.append(activ_func)

    def forward(self, x):
        return self.mlp(x)
