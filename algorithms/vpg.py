import torch
from torch.optim import Adam
import gym

from utils import TrajectoryBuffer
from ac import ActorCriticMLP


class VPG:
    ''' Vanilla Policy Gradient Algorithm '''
    def __init__(self, buffer_size=500, discount=0.99, pi_lr=0.0003, v_lr=0.001, lam=0.97):
        self.buffer_size = buffer_size
        self.discount = discount
        self.pi_lr = pi_lr
        self.v_lr = v_lr
        self.lam = lam

    def compute_loss_pi(self, data):
        obs = data['obs']
        act = data['act']
        adv = data['adv']
        logp_old = data['logp']

        pi, logp = self.ac.policy(obs, act=act)
        loss_pi = -(logp * adv).mean()

        return loss_pi

    def compute_loss_val(self, data):
        obs = data['obs']
        ret = data['ret']
        return ((self.ac.value(obs) - ret) ** 2).mean()

    def update(self):
        ''' Updates policy and value parameters via backprop '''
        data = self.buffer.get()

        self.pi_optim.zero_grad()
        pi_loss = self.compute_loss_pi(data)
        pi_loss.backward()
        self.pi_optim.step()

        for i in range(self.train_v_iters):
            self.v_optim.zero_grad()
            v_loss = self.compute_loss_val(data)
            v_loss.backward()
            self.v_optim.step()


    def train(self, env_func, epochs=250, train_v_iters=80):
        ''' Train an agent on the given environment '''
        env = env_func()
        self.buffer = TrajectoryBuffer(env.observation_space.shape,
                                       env.action_space.shape,
                                       self.buffer_size,
                                       discount=self.discount,
                                       lam=self.lam)
        self.ac = ActorCriticMLP(env.observation_space.shape[0],
                                 env.action_space.n)
        self.train_v_iters = train_v_iters
        self.pi_optim = Adam(self.ac.actor.parameters(), lr=self.pi_lr)
        self.v_optim = Adam(self.ac.critic.parameters(), lr=self.v_lr)
        o = env.reset()
        ep_ret = 0
        ep_len = 0
        for k in range(epochs):
            for t in range(self.buffer_size):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, done, _ = env.step(a)
                ep_ret += r
                ep_len += 1

                self.buffer.store(o, a, r, logp, v)
                o = next_o

                buffer_full = t == self.buffer_size - 1

                if done or buffer_full:
                    if buffer_full:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0.0
                    self.buffer.finish_trajectory(last_val=v)
                    o = env.reset()
                    ep_ret = 0
                    ep_len = 0
            self.update()

    def eval(self, env_func, load=False):
        ''' Evaluate an agent on the given environment '''
        env = env_func()
        if load:
            self.ac = ActorCriticMLP(env.observation_space.shape[0],
                                     env.action_space.n)
            self.ac.load_models('./saved_models/')
        ep_ret = 0
        ep_len = 0
        o = env.reset()
        done = False
        while not done:
            env.render()
            a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, done, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            o = next_o
        print('Completed Episode! Results:')
        print(f'Episode Length: {ep_len}')
        print(f'Episode Return: {ep_ret}')
        return ep_ret


if __name__ == '__main__':
    env_func = lambda : gym.make('CartPole-v1')
    vpg = VPG()
    vpg.train(env_func)
    total_r = 0
    for i in range(100):
        r = vpg.eval(env_func)
        total_r += r
    print(f'Average Return Over 100 Trials: {total_r / 100}')
