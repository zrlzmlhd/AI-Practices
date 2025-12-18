"""
Training Utilities Module.

This module provides training loops, logging, and evaluation utilities
for policy gradient algorithms.

Core Idea:
    Standardized training infrastructure enables consistent experimentation,
    proper logging, and reproducible results across different algorithms.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch

# Conditional gymnasium import
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None

from core.config import TrainingConfig
from core.buffers import EpisodeBuffer


class TrainingLogger:
    """
    Training statistics logger with rolling window averages.

    Tracks episode rewards, lengths, and loss metrics over training.

    Attributes
    ----------
    window_size : int
        Size of rolling window for averaging.
    episode_rewards : deque
        Rolling buffer of episode rewards.
    episode_lengths : deque
        Rolling buffer of episode lengths.
    losses : Dict[str, deque]
        Rolling buffers for each loss metric.

    Examples
    --------
    >>> logger = TrainingLogger(window_size=100)
    >>> logger.log_episode(reward=200.0, length=150)
    >>> logger.log_loss({"policy_loss": 0.5, "value_loss": 0.1})
    >>> print(logger.get_summary())
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize logger with specified window size.

        Parameters
        ----------
        window_size : int, default=100
            Number of episodes to average over.
        """
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.losses: Dict[str, deque] = {}

    def log_episode(self, reward: float, length: int) -> None:
        """
        Log episode statistics.

        Parameters
        ----------
        reward : float
            Total episode reward.
        length : int
            Episode length in steps.
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

    def log_loss(self, loss_dict: Dict[str, float]) -> None:
        """
        Log training losses.

        Parameters
        ----------
        loss_dict : dict
            Dictionary of loss name to value.
        """
        for key, value in loss_dict.items():
            if key not in self.losses:
                self.losses[key] = deque(maxlen=self.window_size)
            self.losses[key].append(value)

    @property
    def mean_reward(self) -> float:
        """Mean reward over recent episodes."""
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def mean_length(self) -> float:
        """Mean episode length over recent episodes."""
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    def get_summary(self) -> str:
        """
        Get formatted summary string.

        Returns
        -------
        str
            Summary of recent training statistics.
        """
        summary = f"Reward: {self.mean_reward:.2f} | Length: {self.mean_length:.1f}"
        for key, values in self.losses.items():
            if values:
                summary += f" | {key}: {np.mean(values):.4f}"
        return summary


def train_policy_gradient(
    agent,
    env_name: str = "CartPole-v1",
    num_episodes: int = 500,
    max_steps: int = 500,
    log_interval: int = 50,
    render: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[object, List[float]]:
    """
    Train a policy gradient agent on a Gymnasium environment.

    Parameters
    ----------
    agent : BasePolicyGradient
        Policy gradient agent to train.
    env_name : str, default="CartPole-v1"
        Gymnasium environment name.
    num_episodes : int, default=500
        Number of training episodes.
    max_steps : int, default=500
        Maximum steps per episode.
    log_interval : int, default=50
        Episodes between log outputs.
    render : bool, default=False
        Whether to render the environment.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print training progress.

    Returns
    -------
    agent : BasePolicyGradient
        Trained agent.
    rewards_history : List[float]
        Episode rewards over training.

    Raises
    ------
    ImportError
        If gymnasium is not installed.

    Examples
    --------
    >>> from algorithms import REINFORCE, A2C
    >>> from core import TrainingConfig

    >>> config = TrainingConfig(gamma=0.99, lr_actor=1e-3)
    >>> agent = REINFORCE(state_dim=4, action_dim=2, config=config)
    >>> agent, rewards = train_policy_gradient(
    ...     agent, "CartPole-v1", num_episodes=300
    ... )
    """
    if not HAS_GYMNASIUM:
        raise ImportError(
            "gymnasium is required for training. "
            "Install with: pip install gymnasium"
        )

    # Create environment
    env = gym.make(env_name, render_mode="human" if render else None)

    # Set seeds
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    logger = TrainingLogger()
    rewards_history = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training {agent.__class__.__name__} on {env_name}")
        print(f"{'=' * 60}\n")

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode if seed else None)
        buffer = EpisodeBuffer()
        total_reward = 0.0

        for step in range(max_steps):
            # Select action
            action, info = agent.select_action(state)

            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience
            buffer.store(
                state=state,
                action=action,
                reward=reward,
                log_prob=info["log_prob"],
                value=info.get("value"),
                done=done,
                entropy=info.get("entropy"),
            )

            total_reward += reward
            state = next_state

            if done:
                break

        # Update agent
        # A2C needs next_state for bootstrapping
        from algorithms.a2c import A2C
        if isinstance(agent, A2C):
            loss_info = agent.update(buffer, next_state, done)
        else:
            loss_info = agent.update(buffer)

        # Log statistics
        rewards_history.append(total_reward)
        logger.log_episode(total_reward, step + 1)
        logger.log_loss(loss_info)

        # Print progress
        if verbose and (episode + 1) % log_interval == 0:
            print(f"Episode {episode + 1:4d} | {logger.get_summary()}")

    env.close()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training complete. Final avg reward: {logger.mean_reward:.2f}")
        print(f"{'=' * 60}\n")

    return agent, rewards_history


def evaluate_agent(
    agent,
    env_name: str,
    num_episodes: int = 10,
    render: bool = False,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Evaluate a trained agent.

    Parameters
    ----------
    agent : BasePolicyGradient
        Trained agent to evaluate.
    env_name : str
        Gymnasium environment name.
    num_episodes : int, default=10
        Number of evaluation episodes.
    render : bool, default=False
        Whether to render.
    seed : int, optional
        Random seed.

    Returns
    -------
    mean_reward : float
        Mean episode reward.
    std_reward : float
        Standard deviation of rewards.
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium is required")

    env = gym.make(env_name, render_mode="human" if render else None)
    rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep if seed else None)
        total_reward = 0.0
        done = False

        while not done:
            action, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def compare_algorithms(
    env_name: str = "CartPole-v1",
    num_episodes: int = 300,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Compare different policy gradient algorithms.

    Parameters
    ----------
    env_name : str, default="CartPole-v1"
        Environment to train on.
    num_episodes : int, default=300
        Training episodes per algorithm.
    seed : int, default=42
        Random seed.

    Returns
    -------
    Dict[str, List[float]]
        Dictionary mapping algorithm names to reward histories.
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium is required")

    from algorithms import REINFORCE, REINFORCEBaseline, A2C

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    results = {}
    config = TrainingConfig(
        gamma=0.99,
        lr_actor=1e-3,
        lr_critic=1e-3,
        entropy_coef=0.01,
    )

    # REINFORCE
    print("\n[1/3] Training REINFORCE...")
    agent_rf = REINFORCE(state_dim, action_dim, config)
    _, rewards_rf = train_policy_gradient(
        agent_rf, env_name, num_episodes, log_interval=100, seed=seed
    )
    results["REINFORCE"] = rewards_rf

    # REINFORCE + Baseline
    print("\n[2/3] Training REINFORCE + Baseline...")
    agent_rfb = REINFORCEBaseline(state_dim, action_dim, config)
    _, rewards_rfb = train_policy_gradient(
        agent_rfb, env_name, num_episodes, log_interval=100, seed=seed
    )
    results["REINFORCE+Baseline"] = rewards_rfb

    # A2C
    print("\n[3/3] Training A2C...")
    agent_a2c = A2C(state_dim, action_dim, config, use_gae=True)
    _, rewards_a2c = train_policy_gradient(
        agent_a2c, env_name, num_episodes, log_interval=100, seed=seed
    )
    results["A2C (GAE)"] = rewards_a2c

    return results
