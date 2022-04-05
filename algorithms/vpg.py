import torch

from utils import TrajectoryBuffer
from ac import ActorCriticMLP


class VPG:
    ''' Vanilla Policy Gradient Algorithm '''
    def __init__(self, buffer_size=100, discount=0.99, lr=0.001, lam=0.95):
        self.buffer_size = buffer_size
        self.discount = discount
        self.lr = lr
        self.lam = lam

    def update(self, data):
        ''' Updates policy and value parameters via backprop '''
        pass

    def train(self, env_func):
        ''' Train an agent on the given environment '''
        env = env_func()
        self.buffer = TrajectoryBuffer(env.observation_spec,
                                       env.action_spec,
                                       self.buffer_size,
                                       discount=self.discount,
                                       lam=self.lam)
        self.ac = ActorCriticMLP(env.observation_spec,
                                 env.action_spec)



    def eval(self, env):
        ''' Evaluate an agent on the given environment '''
        
