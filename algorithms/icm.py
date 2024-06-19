import logging
from itertools import chain

import numpy as np
import scipy # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium


class PPO:
    """Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py"""

    def __init__(self, actor: nn.Module, critic: nn.Module,
                 train_actor_iters=80, train_critic_iters=80,
                 clip_ratio=0.5, target_kl=0.9, actor_lr=3e-4, critic_lr=1e-3):
        # Environment & Agent
        self.actor = actor
        self.critic = critic

        # Training duration
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters

        # Learning hyperparameters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.actor_optim = optim.Adam(
            actor.parameters(),
            lr=actor_lr,
        )
        self.critic_optim = optim.Adam(
            critic.parameters(),
            lr=critic_lr,
        )

    def _compute_actor_loss(self, data):
        env_obs, act, adv, logp_old = (
            data["observations"],
            data["actions"],
            data["advantages"],
            data["log_probs"],
        )

        action_dist = self.actor(env_obs)
        # TODO: Double check this
        logp = action_dist.gather(1, act.unsqueeze(-1)).squeeze().log()
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss = -(torch.min(ratio * adv, clip_adv)).mean()

        kl = (logp_old - logp).mean()

        return loss, kl

    def _update_actor(self, data):
        self.actor.train()
        for _ in range(self.train_actor_iters):
            self.actor_optim.zero_grad()
            loss, kl = self._compute_actor_loss(data)
            if kl > 1.5 * self.target_kl:
                # early stopping
                break
            loss.backward()
            self.actor_optim.step()
        self.actor.eval()

    def _compute_critic_loss(self, data):
        env_obs, ret = (
            data["observations"],
            data["returns"],
        )
        _, v = self.critic(env_obs)
        return ((v - ret) ** 2).mean()

    def _update_critic(self, data):
        self.critic.train()
        for _ in range(self.train_critic_iters):
            self.critic_optim.zero_grad()
            loss = self._compute_critic_loss(data)
            loss.backward()
            self.critic_optim.step()
        self.critic.eval()

    def update(self, data):
        """Updates the actor and critic models given the a dataset of trajectories"""
        self._update_actor(data)
        self._update_critic(data)


class TrajectoryBuffer:
    """
    Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
    Used to store a single trajectory.
    """

    def __init__(
        self,
        max_buffer_size: int,
        discount_factor: float = 0.99,
        gae_discount_factor: float = 0.95,
    ):
        self.max_buffer_size = max_buffer_size
        self.discount_factor = discount_factor
        self.gae_discount_factor = gae_discount_factor
        self.observations: list[torch.Tensor] = []
        self.next_observations: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[torch.Tensor] = []

    def store(
        self,
        observation: torch.Tensor,
        next_observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: float,
        log_prob: torch.Tensor,
    ) -> None:
        if len(self.observations) == self.max_buffer_size:
            logging.warn(
                f"Cannot store additional time-steps in an already full trajectory. Current size: {len(self.observations)}. Max size: {self.max_buffer_size}"
            )
            return
        self.observations.append(observation)
        self.next_observations.append(next_observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def get(self, last_value: float) -> dict[str, torch.Tensor]:
        size = len(self.observations)
        if size < self.max_buffer_size:
            logging.warn(
                f"Computing information on a potentially unfinished trajectory. Current size: {size}. Max size: {self.max_buffer_size}"
            )
        observations = torch.stack(self.observations)
        next_observations = torch.stack(self.observations)

        self.rewards.append(last_value)
        self.values.append(last_value)
        rewards = torch.tensor(self.rewards)
        values = torch.tensor(self.values)

        deltas = rewards[:-1] + self.discount_factor * values[1:] - values[:-1]
        advantages = torch.tensor(
            discount_cumsum(
                deltas.numpy(), self.discount_factor * self.gae_discount_factor
            ).copy()
        )
        returns = torch.tensor(
            discount_cumsum(rewards.numpy(), self.discount_factor)[:-1].copy()
        ).squeeze()

        # Normalize advantages
        adv_mean, adv_std = torch.mean(advantages), torch.std(advantages)
        advantages = (advantages - adv_mean) / adv_std

        return {
            "observations": observations,
            "next_observations": next_observations,
            "actions": torch.stack(self.actions),
            "returns": returns,
            "advantages": advantages,
            "log_probs": torch.stack(self.log_probs),
        }


class ICM:
    def __init__(self, phi, forward_model, inverse_model, eta=1.0):
        # feature extractor
        self.phi = phi

        self.forward_model = forward_model
        self.inverse_model = inverse_model

        self.eta = eta

        self.forward_optim = optim.Adam(
            chain(phi.parameters(), forward_model.parameters()), lr=1e-3)
        self.inverse_optim = optim.Adam(
            chain(phi.parameters(),inverse_model.parameters()), lr=1e-3)

    def forward_dynamics(self, data):
        self.forward_optim.zero_grad()
        observations, next_observations, actions = (
            data["observations"],
            data["next_observations"],
            data["actions"],
        )

        # extract features
        xt = self.phi(observations)
        xt_1 = self.phi(next_observations)

        # forward dyanmics
        xt_1_pred = self.forward_model(xt, actions)

        # forward loss
        loss = torch.nn.functional.mse_loss(xt_1_pred, xt_1)

        loss.backward()

        self.forward_optim.step()


    def inverse_dynamics(self, data):
        self.inverse_optim.zero_grad()
        observations, next_observations, actions = (
            data["observations"],
            data["next_observations"],
            data["actions"],
        )

        # extract features
        xt = self.phi(observations)
        xt_1 = self.phi(next_observations)

        # inverse dynamics
        a_pred = self.inverse_model(xt, xt_1)

        # inverse loss
        loss = torch.nn.functional.l1_loss(a_pred, actions)

        loss.backward()

        self.inverse_optim.step()


class CartPoleFeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class CartPoleActor(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CartPoleCritic(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CartPoleInverseDynamics(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, xt, xt_1):
        x = torch.cat((xt, xt_1), dim=-1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CartPoleForwardDynamics(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # feature extraction output + action
        self.fc1 = nn.Linear(input_shape + 1, 512)
        self.fc2 = nn.Linear(512, input_shape)

    def forward(self, x, a):
        x = torch.cat((x, a), dim=-1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """Taken from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L29"""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def preprocess_observation(obs):
    # Convert the observation to an image
    img = np.zeros((1, 64, 64), dtype=np.uint8)
    img[0, :, :] = np.reshape(obs, (64, 64))
    return img


if __name__ == "__main__":
    env = gymnasium.make('CartPole-v1')
    input_shape = (1, 64, 64)
    num_actions = env.action_space.n
    model = CartPoleCNN(input_shape, num_actions)
    optimizer = optim.Adam(model.parameters())
    replay_buffer = deque(maxlen=10000)

    for episode in range(num_episodes):
        obs = env.reset()
        obs = preprocess_observation(obs)
        done = False
        while not done:
            # Get the action from the model
            action = model(torch.from_numpy(obs).float().unsqueeze(0)).max(1)[1].item()
            
            # Take the action and get the next observation
            next_obs, reward, done, _ = env.step(action)
            next_obs = preprocess_observation(next_obs)
            
            # Store the transition in the replay buffer
            replay_buffer.append((obs, action, reward, next_obs, done))
            
            obs = next_obs

