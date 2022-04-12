import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils import MLP


class ActorCriticMLP:
    ''' Actor/Critic that performs actions and makes value estimates '''
    def __init__(self, obs_dim, act_dim):
        self.actor = MLP([obs_dim, 128, 64, act_dim])
        self.critic = MLP([obs_dim, 128, 64, 1])

    def distribution(self, obs):
        ''' Returns the current policy distribution over the observation '''
        return Categorical(logits=self.actor(obs))

    def policy(self, obs, act=None):
        ''' Returns an action given the observation '''
        pi = self.distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a

    def value(self, obs):
        ''' Returns the perceived value of the observation '''
        return self.critic(obs)

    def step(self, obs):
        ''' Returns the action, value, and logp_a for the observation '''
        with torch.no_grad():
            pi, _ = self.policy(obs)
            a = pi.sample()
            logp = pi.log_prob(a)
            v = self.value(obs)
        return a.numpy(), v.numpy(), logp.numpy()
