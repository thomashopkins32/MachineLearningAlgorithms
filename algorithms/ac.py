import torch
import torch.nn.functional as F

from utils import MLP


class ActorCriticMLP:
    ''' Actor/Critic that performs actions and makes value estimates '''
    def __init__(self, obs_dim, act_dim):
        self.actor = MLP([obs_dim, 128, 64, act_dim])
        self.critic = MLP([obs_dim, 128, 64, 1])

    def distribution(self, obs):
        ''' Returns the current policy distribution over the observation '''
        with torch.no_grad():
            return F.softmax(self.actor(obs), dim=1)

    def act(self, obs):
        ''' Returns an action given the observation '''
        return self.actor(obs)

    def value(self, obs):
        ''' Returns the perceived value of the observation '''
        return self.critic(obs)

    def logp_a(self, obs, a):
        ''' Returns the log probability that an action was selected '''
        dist = self.distribution(obs)
        return dist[:, a]
