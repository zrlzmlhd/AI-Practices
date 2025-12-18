"""
Training and Evaluation Utilities.

This module provides training loop and evaluation functions.

Core Idea (核心思想)
====================
生产级训练循环，包括：
- Episode迭代和step交互
- 日志记录和进度显示
- Early stopping和模型保存
- 多episode评估

Training Flow (训练流程)
========================
for episode in range(num_episodes):
    state = env.reset()
    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, done = env.step(action)
        loss = agent.train_step(state, action, reward, next_state, done)
        state = next_state
    log_episode(episode, reward, loss)
    checkpoint_if_best(agent)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.enums import DQNVariant


@dataclass
class TrainingLogger:
    """
    Structured training logger with statistics tracking.

    Core Idea (核心思想)
    --------------------
    记录训练过程中的关键指标，支持：
    - Episode奖励追踪
    - 滑动窗口平均
    - 学习曲线导出

    Attributes
    ----------
    episode_rewards : List[float]
        All episode rewards
    episode_lengths : List[int]
        All episode lengths
    losses : List[float]
        Training losses
    q_values : List[float]
        Average Q-values
    epsilons : List[float]
        Exploration rates
    window_size : int
        Window size for moving average

    Examples
    --------
    >>> logger = TrainingLogger(window_size=100)
    >>> logger.log_episode(100.0, 200, 0.01, 0.05)
    >>> logger.mean_reward
    100.0
    """

    window_size: int = 100
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    q_values: List[float] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    _start_time: Optional[float] = field(default=None, repr=False)

    def start(self) -> None:
        """Start timing."""
        self._start_time = time.time()

    def log_episode(
        self,
        reward: float,
        length: int,
        loss: Optional[float] = None,
        epsilon: Optional[float] = None,
        q_value: Optional[float] = None,
    ) -> None:
        """
        Log episode statistics.

        Parameters
        ----------
        reward : float
            Total episode reward
        length : int
            Episode length (steps)
        loss : Optional[float]
            Average episode loss
        epsilon : Optional[float]
            Current exploration rate
        q_value : Optional[float]
            Average Q-value
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        if loss is not None:
            self.losses.append(loss)
        if epsilon is not None:
            self.epsilons.append(epsilon)
        if q_value is not None:
            self.q_values.append(q_value)

        if self._start_time is not None:
            self.timestamps.append(time.time() - self._start_time)

    @property
    def num_episodes(self) -> int:
        """Number of logged episodes."""
        return len(self.episode_rewards)

    @property
    def mean_reward(self) -> float:
        """Mean reward over window."""
        if not self.episode_rewards:
            return 0.0
        window = self.episode_rewards[-self.window_size:]
        return float(np.mean(window))

    @property
    def std_reward(self) -> float:
        """Standard deviation of reward over window."""
        if not self.episode_rewards:
            return 0.0
        window = self.episode_rewards[-self.window_size:]
        return float(np.std(window))

    @property
    def max_reward(self) -> float:
        """Maximum episode reward."""
        if not self.episode_rewards:
            return 0.0
        return float(np.max(self.episode_rewards))

    @property
    def mean_length(self) -> float:
        """Mean episode length over window."""
        if not self.episode_lengths:
            return 0.0
        window = self.episode_lengths[-self.window_size:]
        return float(np.mean(window))

    @property
    def mean_loss(self) -> float:
        """Mean loss over window."""
        if not self.losses:
            return 0.0
        window = self.losses[-self.window_size:]
        return float(np.mean(window))

    def get_summary(self) -> Dict[str, Any]:
        """
        Get training summary statistics.

        Returns
        -------
        Dict[str, Any]
            Summary with all statistics
        """
        return {
            "num_episodes": self.num_episodes,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "max_reward": self.max_reward,
            "mean_length": self.mean_length,
            "mean_loss": self.mean_loss,
            "final_epsilon": self.epsilons[-1] if self.epsilons else None,
            "total_time": self.timestamps[-1] if self.timestamps else None,
        }

    def get_smoothed_rewards(
        self,
        window: Optional[int] = None,
    ) -> List[float]:
        """
        Get smoothed reward curve.

        Parameters
        ----------
        window : Optional[int]
            Smoothing window size

        Returns
        -------
        List[float]
            Smoothed rewards
        """
        window = window or self.window_size
        if len(self.episode_rewards) < window:
            return self.episode_rewards.copy()

        smoothed = []
        for i in range(len(self.episode_rewards)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(self.episode_rewards[start:i + 1]))

        return smoothed


def train_agent(
    agent: Any,
    env_name: str,
    num_episodes: int = 500,
    max_steps: int = 1000,
    target_reward: Optional[float] = None,
    eval_frequency: int = 50,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Any, TrainingLogger]:
    """
    Train DQN agent on given environment.

    Core Idea (核心思想)
    --------------------
    完整的训练循环，包括：
    - Episode交互和经验收集
    - 在线学习和目标网络同步
    - 周期性评估和Early Stopping
    - Checkpoint保存

    Training Flow (训练流程)
    ------------------------
    1. Reset environment
    2. For each step: select action → execute → store → update
    3. Log episode statistics
    4. Periodic evaluation and checkpointing

    Parameters
    ----------
    agent : Any
        DQN variant agent with train_step() method
    env_name : str
        Gymnasium environment name
    num_episodes : int, default=500
        Number of training episodes
    max_steps : int, default=1000
        Maximum steps per episode
    target_reward : Optional[float]
        Target reward for early stopping
    eval_frequency : int, default=50
        Episodes between evaluations
    checkpoint_dir : Optional[Union[str, Path]]
        Directory for saving checkpoints
    verbose : bool, default=True
        Whether to print progress
    seed : Optional[int]
        Random seed for environment

    Returns
    -------
    agent : Any
        Trained agent
    logger : TrainingLogger
        Training statistics

    Examples
    --------
    >>> agent, logger = train_agent(agent, "CartPole-v1", num_episodes=500)
    >>> print(f"Final reward: {logger.mean_reward:.1f}")

    Notes
    -----
    - Requires gymnasium package
    - Saves best model if checkpoint_dir provided
    """
    try:
        import gymnasium as gym
    except ImportError:
        raise ImportError(
            "gymnasium required for training. "
            "Install with: pip install gymnasium"
        )

    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)

    logger = TrainingLogger()
    logger.start()

    best_reward = float("-inf")
    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    agent.set_train_mode()

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        num_updates = 0

        for step in range(max_steps):
            action = agent.select_action(state, training=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            loss = agent.train_step(state, action, float(reward), next_state, done)

            if loss is not None:
                episode_loss += loss
                num_updates += 1

            episode_reward += reward
            state = next_state

            if done:
                break

        avg_loss = episode_loss / max(num_updates, 1)
        epsilon = getattr(agent, "epsilon", 0.0)

        logger.log_episode(episode_reward, step + 1, avg_loss, epsilon)

        if verbose and (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Reward: {episode_reward:.1f} | "
                f"Mean({logger.window_size}): {logger.mean_reward:.1f} | "
                f"Loss: {avg_loss:.4f} | "
                f"ε: {epsilon:.3f}"
            )

        # Checkpoint best model
        if checkpoint_path is not None and logger.mean_reward > best_reward:
            best_reward = logger.mean_reward
            agent.save(checkpoint_path / "best_model.pt")

        # Early stopping
        if target_reward is not None and logger.mean_reward >= target_reward:
            if verbose:
                print(f"Target reward {target_reward} reached at episode {episode + 1}!")
            break

    env.close()
    return agent, logger


def evaluate_agent(
    agent: Any,
    env_name: str,
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = False,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Evaluate agent performance.

    Parameters
    ----------
    agent : Any
        Trained agent with select_action() method
    env_name : str
        Gymnasium environment name
    num_episodes : int, default=10
        Number of evaluation episodes
    max_steps : int, default=1000
        Maximum steps per episode
    render : bool, default=False
        Whether to render environment
    seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    mean_reward : float
        Mean episode reward
    std_reward : float
        Standard deviation of episode rewards

    Examples
    --------
    >>> mean, std = evaluate_agent(agent, "CartPole-v1", num_episodes=100)
    >>> print(f"Performance: {mean:.1f} ± {std:.1f}")
    """
    try:
        import gymnasium as gym
    except ImportError:
        raise ImportError(
            "gymnasium required for evaluation. "
            "Install with: pip install gymnasium"
        )

    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)

    agent.set_eval_mode()
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        rewards.append(episode_reward)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))
