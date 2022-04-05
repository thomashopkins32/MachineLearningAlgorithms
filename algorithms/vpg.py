import numpy as np


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
            - discounted return for each step in trajectory
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

    def finish_trajectory(self):
        ''' Computes the return and advantage per step in trajactory '''
        pass

    def get(self):
        ''' Empties the buffer into something useable for learning '''
        pass


class ActorCritic:
    ''' Actor/Critic that performs actions and makes value estimates '''
    def __init__(self):
        pass

    def distribution(self, obs):
        ''' Returns the current policy distribution over the observation '''
        pass

    def act(self, obs):
        ''' Returns an action given the observation '''
        pass

    def value(self, obs):
        ''' Returns the perceived value of the observation '''
        pass

    def logprob_a(self, obs, a):
        ''' Returns the log probability that an action was selected '''
        pass


class VPG:
    ''' Vanilla Policy Gradient Algorithm '''
    def __init__(self):
        pass

    def update(self, data):
        ''' Updates policy and value parameters via gradient ascent/descent '''
        pass

    def train(self, env):
        ''' Train an agent on the given environment '''
        pass

    def eval(self, env):
        ''' Evaluate an agent on the given environment '''
        pass
