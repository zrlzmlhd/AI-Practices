"""
Training Loop Implementation
============================

Core Idea
---------
Unified training infrastructure for all continuous control RL algorithms.
Handles data collection, learning updates, evaluation, and logging.

Mathematical Theory
-------------------
**Off-Policy Training Loop**:

The training process alternates between:

1. **Data Collection**: :math:`\\mathcal{D} \\leftarrow \\mathcal{D} \\cup \\{(s,a,r,s',d)\\}`
2. **Learning**: :math:`\\theta \\leftarrow \\theta + \\alpha \\nabla L(\\theta; \\mathcal{D})`

For off-policy algorithms (DDPG, TD3, SAC), we can update multiple times
per environment step, improving sample efficiency:

.. math::

    \\text{Updates per step} = \\frac{\\text{Gradient steps}}{\\text{Env steps}}

**Exploration Schedule**:

Initial random exploration fills the buffer:

.. math::

    a_t = \\begin{cases}
    \\text{Uniform}(\\mathcal{A}) & t < T_{random} \\\\
    \\pi(s_t) + \\epsilon & t \\geq T_{random}
    \\end{cases}

Summary
-------
Trainer class provides:

- Configurable training loop with checkpointing
- Periodic evaluation during training
- Training history logging
- Resume from checkpoint support
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_agent import BaseAgent
from training.evaluator import evaluate_agent


@dataclass
class TrainingConfig:
    """
    Training Loop Configuration
    ============================

    Attributes
    ----------
    total_timesteps : int
        Total environment steps for training.
    start_timesteps : int
        Random exploration steps before learning.
    eval_frequency : int
        Steps between evaluations.
    eval_episodes : int
        Episodes per evaluation.
    log_frequency : int
        Steps between logging.
    save_frequency : int
        Steps between checkpoints.
    max_episode_steps : int, optional
        Maximum steps per episode.
    updates_per_step : int
        Gradient updates per environment step.
    """

    total_timesteps: int = 1_000_000
    start_timesteps: int = 25_000
    eval_frequency: int = 5_000
    eval_episodes: int = 10
    log_frequency: int = 1_000
    save_frequency: int = 50_000
    max_episode_steps: Optional[int] = None
    updates_per_step: int = 1


@dataclass
class TrainingHistory:
    """
    Training Metrics History
    ========================

    Stores training progress for analysis and visualization.

    Attributes
    ----------
    timesteps : List[int]
        Timesteps at each logging point.
    eval_rewards : List[float]
        Mean evaluation returns.
    eval_stds : List[float]
        Standard deviation of evaluation returns.
    training_metrics : Dict[str, List[float]]
        Per-update metrics (losses, Q-values, etc.).
    episode_rewards : List[float]
        Per-episode training returns.
    episode_lengths : List[int]
        Per-episode lengths.
    """

    timesteps: List[int] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    eval_stds: List[float] = field(default_factory=list)
    training_metrics: Dict[str, List[float]] = field(default_factory=dict)
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)


class Trainer:
    """
    Unified Training Loop for Continuous Control Agents
    ====================================================

    Core Idea
    ---------
    Algorithm-agnostic training infrastructure that handles:

    - Environment interaction and data collection
    - Learning updates (respecting start_timesteps)
    - Periodic evaluation and logging
    - Checkpointing and resume

    Mathematical Theory
    -------------------
    **Training Phases**:

    1. **Warm-up** (t < start_timesteps):
       Random actions to fill buffer with diverse experiences

    2. **Learning** (t ≥ start_timesteps):
       Policy-guided exploration with gradient updates

    **Evaluation Protocol**:

    Periodic evaluation uses deterministic policy:

    .. math::

        \\hat{J}(\\pi) = \\frac{1}{N} \\sum_{i=1}^{N}
        \\sum_{t=0}^{T} r_t^{(i)}

    Problem Statement
    -----------------
    Without a unified trainer:

    - Training code duplicated per algorithm
    - Inconsistent evaluation protocols
    - Manual checkpoint management
    - No standardized logging

    Trainer centralizes all infrastructure, enabling fair algorithm comparison.

    Parameters
    ----------
    agent : BaseAgent
        RL agent to train.
    env : gym.Env
        Training environment.
    eval_env : gym.Env, optional
        Separate environment for evaluation.
    config : TrainingConfig
        Training hyperparameters.
    save_path : str, optional
        Directory for checkpoints.

    Attributes
    ----------
    history : TrainingHistory
        Training progress metrics.

    Example
    -------
    >>> from popular_rl_algorithms import SACAgent, SACConfig
    >>> import gymnasium as gym
    >>>
    >>> env = gym.make("Pendulum-v1")
    >>> config = SACConfig(state_dim=3, action_dim=1, max_action=2.0)
    >>> agent = SACAgent(config)
    >>>
    >>> trainer = Trainer(agent, env)
    >>> history = trainer.train(total_timesteps=100000)
    >>> print(f"Final reward: {history.eval_rewards[-1]:.2f}")
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: Any,
        eval_env: Optional[Any] = None,
        config: Optional[TrainingConfig] = None,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize trainer.

        Parameters
        ----------
        agent : BaseAgent
            Agent to train.
        env : gym.Env
            Training environment.
        eval_env : gym.Env, optional
            Evaluation environment. Uses env if not provided.
        config : TrainingConfig, optional
            Training settings. Uses defaults if not provided.
        save_path : str, optional
            Checkpoint directory.
        verbose : bool, default=True
            Whether to print training progress.
        """
        self.agent = agent
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env
        self.config = config if config is not None else TrainingConfig()
        self.save_path = save_path
        self.verbose = verbose

        self.history = TrainingHistory()
        self._timestep = 0
        self._episode = 0

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> TrainingHistory:
        """
        Run training loop.

        Parameters
        ----------
        total_timesteps : int, optional
            Override config.total_timesteps.
        callback : Callable, optional
            Function called every log_frequency steps.
            Signature: callback(timestep, metrics)

        Returns
        -------
        TrainingHistory
            Training progress and metrics.

        Notes
        -----
        **Training Loop**:

        .. code-block:: text

            for t in range(total_timesteps):
                if t < start_timesteps:
                    action = random
                else:
                    action = agent.select_action(state)

                next_state, reward, done = env.step(action)
                agent.store_transition(...)

                if t >= start_timesteps:
                    metrics = agent.update()

                if t % eval_frequency == 0:
                    evaluate_and_log()
        """
        if total_timesteps is not None:
            self.config.total_timesteps = total_timesteps

        # Get environment dimensions
        state = self._reset_env()
        action_dim = self.env.action_space.shape[0]
        action_high = self.env.action_space.high[0]

        episode_reward = 0.0
        episode_length = 0
        start_time = time.time()

        self.agent.train_mode()

        while self._timestep < self.config.total_timesteps:
            # Select action
            if self._timestep < self.config.start_timesteps:
                # Random exploration
                action = self.env.action_space.sample()
            else:
                # Policy with exploration
                action = self.agent.select_action(np.asarray(state), deterministic=False)

            # Environment step
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, truncated, _ = step_result
                done = done or truncated
            else:
                next_state, reward, done, _ = step_result

            # Handle max episode steps
            if self.config.max_episode_steps is not None:
                if episode_length + 1 >= self.config.max_episode_steps:
                    done = True

            # Store transition
            self.agent.store_transition(
                np.asarray(state),
                np.asarray(action),
                float(reward),
                np.asarray(next_state),
                done,
            )

            episode_reward += reward
            episode_length += 1
            self._timestep += 1

            # Learning update
            if self._timestep >= self.config.start_timesteps:
                for _ in range(self.config.updates_per_step):
                    metrics = self.agent.update()

                    # Store metrics
                    for key, value in metrics.items():
                        if key not in self.history.training_metrics:
                            self.history.training_metrics[key] = []
                        self.history.training_metrics[key].append(value)

            # Episode end
            if done:
                self.history.episode_rewards.append(episode_reward)
                self.history.episode_lengths.append(episode_length)
                self._episode += 1

                state = self._reset_env()
                episode_reward = 0.0
                episode_length = 0
            else:
                state = next_state

            # Evaluation
            if self._timestep % self.config.eval_frequency == 0:
                eval_stats = evaluate_agent(
                    self.agent,
                    self.eval_env,
                    num_episodes=self.config.eval_episodes,
                )

                self.history.timesteps.append(self._timestep)
                self.history.eval_rewards.append(eval_stats["mean_reward"])
                self.history.eval_stds.append(eval_stats["std_reward"])

                if self.verbose:
                    elapsed = time.time() - start_time
                    fps = self._timestep / elapsed
                    print(
                        f"Step {self._timestep:7d} | "
                        f"Eval: {eval_stats['mean_reward']:8.2f} ± {eval_stats['std_reward']:6.2f} | "
                        f"Episodes: {self._episode:4d} | "
                        f"FPS: {fps:.0f}"
                    )

            # Logging callback
            if callback is not None and self._timestep % self.config.log_frequency == 0:
                callback(self._timestep, metrics if self._timestep >= self.config.start_timesteps else {})

            # Checkpointing
            if self.save_path is not None and self._timestep % self.config.save_frequency == 0:
                self._save_checkpoint()

        # Final evaluation
        final_stats = evaluate_agent(
            self.agent,
            self.eval_env,
            num_episodes=self.config.eval_episodes,
        )

        if self.verbose:
            print(f"\nTraining complete!")
            print(f"Final evaluation: {final_stats['mean_reward']:.2f} ± {final_stats['std_reward']:.2f}")

        return self.history

    def _reset_env(self) -> np.ndarray:
        """Reset environment and return initial state."""
        result = self.env.reset()
        if isinstance(result, tuple):
            state, _ = result
        else:
            state = result
        return np.asarray(state)

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        import os

        if self.save_path is None:
            return

        os.makedirs(self.save_path, exist_ok=True)
        checkpoint_path = os.path.join(
            self.save_path, f"checkpoint_{self._timestep}.pt"
        )
        self.agent.save(checkpoint_path)

        if self.verbose:
            print(f"  Saved checkpoint: {checkpoint_path}")


def train_agent(
    agent: BaseAgent,
    env: Any,
    total_timesteps: int = 100_000,
    start_timesteps: int = 10_000,
    eval_frequency: int = 5_000,
    eval_episodes: int = 5,
    verbose: bool = True,
) -> TrainingHistory:
    """
    Convenience function for quick training.

    Parameters
    ----------
    agent : BaseAgent
        Agent to train.
    env : gym.Env
        Training environment.
    total_timesteps : int
        Total training steps.
    start_timesteps : int
        Random exploration steps.
    eval_frequency : int
        Steps between evaluations.
    eval_episodes : int
        Episodes per evaluation.
    verbose : bool
        Print progress.

    Returns
    -------
    TrainingHistory
        Training metrics.

    Example
    -------
    >>> history = train_agent(agent, env, total_timesteps=50000)
    """
    config = TrainingConfig(
        total_timesteps=total_timesteps,
        start_timesteps=start_timesteps,
        eval_frequency=eval_frequency,
        eval_episodes=eval_episodes,
    )

    trainer = Trainer(agent, env, config=config, verbose=verbose)
    return trainer.train()


if __name__ == "__main__":
    print("Trainer module loaded successfully.")
    print("Use Trainer class or train_agent function to train agents.")
