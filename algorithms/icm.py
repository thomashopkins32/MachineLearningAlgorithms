import logging
from itertools import chain

import numpy as np
import scipy  # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium
from tqdm import tqdm # type: ignore
import matplotlib.pyplot as plt


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
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[torch.Tensor] = []

    def reset(self):
        self.observations = []
        self.next_observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    def store(
        self,
        observation: torch.Tensor,
        next_observation: torch.Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: torch.Tensor,
    ) -> None:
        if self.is_full():
            logging.warning(
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
        if not self.is_full():
            logging.warning(
                f"Computing information on a potentially unfinished trajectory. Current size: {len(self.observations)}. Max size: {self.max_buffer_size}"
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
            "actions": torch.tensor(self.actions, dtype=torch.int64),
            "returns": returns,
            "advantages": advantages,
            "log_probs": torch.stack(self.log_probs),
        }

    def is_full(self) -> bool:
        return len(self.observations) == self.max_buffer_size


class PPO:
    """Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py"""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        train_actor_iters=80,
        train_critic_iters=80,
        clip_ratio=0.2,
        target_kl=0.01,
        actor_lr=3e-5,
        critic_lr=1e-4,
    ):
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
        logp = action_dist.softmax(-1).gather(1, act.unsqueeze(-1)).squeeze().log()
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
        v = self.critic(env_obs)
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


class ICM:
    def __init__(self, phi, forward_model, inverse_model, eta=10.0):
        # feature extractor
        self.phi = phi

        self.forward_model = forward_model
        self.inverse_model = inverse_model

        self.eta = eta

        self.forward_optim = optim.Adam(
            chain(phi.parameters(), forward_model.parameters()), lr=1e-3
        )
        self.inverse_optim = optim.Adam(
            chain(phi.parameters(), inverse_model.parameters()), lr=1e-3
        )

    def intrinsic_reward(self, obs, next_obs, action):
        with torch.no_grad():
            # extract features
            xt = self.phi(obs)
            xt_1 = self.phi(next_obs)

            # forward dyanmics
            xt_1_pred = self.forward_model(xt, action)

            # forward loss
            loss = self.eta * torch.nn.functional.mse_loss(xt_1_pred, xt_1)

        return loss.item()

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
        xt_1_pred = self.forward_model(xt, actions.unsqueeze(-1))

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
        a_pred_dist = self.inverse_model(xt, xt_1).softmax(-1)

        # inverse loss
        loss = torch.nn.functional.cross_entropy(a_pred_dist, actions)

        loss.backward()

        self.inverse_optim.step()


class CartPoleFeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CartPoleActor(nn.Module):
    def __init__(self, input_shape: int, num_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CartPoleCritic(nn.Module):
    def __init__(self, input_shape: int):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CartPoleInverseDynamics(nn.Module):
    def __init__(self, input_shape: int, num_actions: int):
        super().__init__()
        # Double the length of a single observation
        self.fc1 = nn.Linear(input_shape * 2, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, xt: torch.Tensor, xt_1: torch.Tensor):
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


if __name__ == "__main__":
    env = gymnasium.make("CartPole-v1")
    input_shape = env.observation_space.shape[0]  # type: ignore
    num_actions = env.action_space.n  # type: ignore
    num_steps = 1000000

    trajectory_buffer = TrajectoryBuffer(
        10_000, discount_factor=0.99, gae_discount_factor=0.95
    )

    ppo_actor = CartPoleActor(input_shape, num_actions)
    ppo_critic = CartPoleCritic(input_shape)
    inverse_dynamics = CartPoleInverseDynamics(512, num_actions)
    forward_dynamics = CartPoleForwardDynamics(512)
    feature_extractor = CartPoleFeatureExtractor(input_shape)

    ppo = PPO(ppo_actor, ppo_critic)
    icm = ICM(feature_extractor, forward_dynamics, inverse_dynamics)

    env_rewards = []
    intrinsic_rewards = []

    obs, _ = env.reset()
    obs = torch.from_numpy(obs).float()
    episode_env_reward = 0.0
    episode_intrinsic_reward = 0.0
    for step in tqdm(range(num_steps)):
        # Get the action from the model
        with torch.no_grad():
            action_dist = ppo_actor(obs).softmax(-1)
            value = ppo_critic(obs).item()
        action = torch.multinomial(action_dist, 1).item()
        logp = action_dist[action].log()

        # Take the action and get the next observation
        next_obs, env_reward, done, _, _ = env.step(action) # type: ignore
        if done:
            next_obs, _ = env.reset()
        next_obs = torch.from_numpy(next_obs).float()

        # Add intrinsic reward to the environment reward
        intrinsic_reward = icm.intrinsic_reward(obs, next_obs, torch.tensor(action).unsqueeze(0))

        reward = env_reward + intrinsic_reward

        # Get and store episode results
        episode_env_reward += env_reward
        episode_intrinsic_reward += intrinsic_reward
        if done:
            env_rewards.append(episode_env_reward)
            intrinsic_rewards.append(episode_intrinsic_reward)
            episode_env_reward = 0.0
            episode_intrinsic_reward = 0.0

        # Store the transition in the replay buffer
        trajectory_buffer.store(obs, next_obs, action, reward, value, logp) # type: ignore
        obs = next_obs

        if trajectory_buffer.is_full():
            last_value = ppo_critic(obs).item()
            data = trajectory_buffer.get(last_value)
            ppo.update(data)
            icm.forward_dynamics(data)
            icm.inverse_dynamics(data)
            trajectory_buffer.reset()

    step_range = np.arange(len(env_rewards))
    rewards = [env_rewards[i] + intrinsic_rewards[i] for i in range(len(env_rewards))]
    plt.plot(step_range, rewards, label='total')
    plt.plot(step_range, intrinsic_rewards, label="intrinsic reward")
    plt.plot(step_range, env_rewards, label='environment reward')
    plt.legend()
    plt.show()

# TODO: Random minibatches from the full trajectory
# TODO: Run ppo.py in a separate file first before adding ICM. We need to verify that PPO works before adding ICM