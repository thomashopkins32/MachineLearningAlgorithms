import torch
import torch.optim as optim
import gym

from ac import ActorCriticMLP


class REINFORCE:
    ''' Implements One-Step Actor-Critic REINFORCE '''
    def __init__(self, discount=0.99, pi_lr=0.0001, v_lr=0.0001):
        self.discount = discount
        self.pi_lr = pi_lr
        self.v_lr = v_lr

    def train(self, env_func, epochs=1000):
        ''' Train an agent on the given environment '''
        env = env_func()
        self.ac = ActorCriticMLP(env.observation_space.shape[0],
                                 env.action_space.n)
        pi_optim = optim.Adam(self.ac.actor.parameters(), lr=self.pi_lr)
        v_optim = optim.Adam(self.ac.critic.parameters(), lr=self.v_lr)

        for e in range(epochs):
            o = env.reset()
            I = 1
            done = False
            ep_rew = 0.0
            while not done:
                v_optim.zero_grad()
                pi_optim.zero_grad()
                with torch.no_grad():
                    pi, _ = self.ac.policy(torch.as_tensor(o, dtype=torch.float32))
                    a = pi.sample().item()
                next_o, r, done, _ = env.step(a)
                with torch.no_grad():
                    # value used for delta calc (no grad)
                    if done:
                        vp = 0.0
                    else:
                        vp = self.ac.value(torch.as_tensor(next_o, dtype=torch.float32))
                    v = self.ac.value(torch.as_tensor(o, dtype=torch.float32))
                    delta = r + self.discount * vp - v
                # value used for autograd
                v = self.ac.value(torch.as_tensor(o, dtype=torch.float32))
                v = -(delta * v)
                v.backward()
                v_optim.step()
                _, logp = self.ac.policy(torch.as_tensor(o, dtype=torch.float32),
                                         torch.as_tensor(a, dtype=torch.long))
                g = -(I * delta * logp)
                g.backward()
                pi_optim.step()
                I = self.discount * I
                ep_rew += r
                o = next_o

    def eval(self, env_func, epochs=100):
        ''' Evaluate an agent on the given environment '''
        env = env_func()
        total_rew = 0.0
        for e in range(epochs):
            o = env.reset()
            done = False
            ep_rew = 0.0
            while not done:
                env.render()
                with torch.no_grad():
                    pi, _ = self.ac.policy(torch.as_tensor(o, dtype=torch.float32))
                    a = pi.sample().item()
                next_o, r, done, _ = env.step(a)
                o = next_o
                ep_rew += r
            total_rew += ep_rew
        return total_rew / epochs


if __name__ == '__main__':
    env_func = lambda : gym.make('CartPole-v1')
    reinforce = REINFORCE()
    reinforce.train(env_func)
    print(reinforce.eval(env_func))
