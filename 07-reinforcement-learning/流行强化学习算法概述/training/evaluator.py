"""
Agent Evaluation Utilities
==========================

Core Idea
---------
Separate evaluation from training for unbiased performance assessment.
Evaluation uses deterministic policy without exploration noise.

Mathematical Theory
-------------------
**Unbiased Evaluation**:

During training, we collect data with exploration:

.. math::

    a_t = \\pi(s_t) + \\epsilon_t

For evaluation, we use the learned policy directly:

.. math::

    a_t = \\pi(s_t) \\quad \\text{(deterministic)}

This provides an unbiased estimate of the policy's true performance.

**Statistical Significance**:

Multiple evaluation episodes reduce variance:

.. math::

    \\bar{G} = \\frac{1}{N} \\sum_{i=1}^{N} G_i, \\quad
    \\text{SE} = \\frac{\\sigma}{\\sqrt{N}}

where N is the number of episodes and SE is standard error.

Summary
-------
- evaluate_episode: Single episode rollout
- evaluate_agent: Multiple episodes with statistics
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_agent import BaseAgent


def evaluate_episode(
    agent: BaseAgent,
    env: Any,
    max_steps: Optional[int] = None,
    render: bool = False,
) -> Tuple[float, int, List[float]]:
    """
    Evaluate agent for one episode.

    Parameters
    ----------
    agent : BaseAgent
        Agent to evaluate.
    env : gym.Env
        Environment instance.
    max_steps : int, optional
        Maximum steps per episode. If None, runs until done.
    render : bool, default=False
        Whether to render the environment.

    Returns
    -------
    total_reward : float
        Cumulative episode reward.
    episode_length : int
        Number of steps in episode.
    rewards : List[float]
        Per-step rewards for detailed analysis.

    Notes
    -----
    Uses deterministic=True for evaluation to assess learned policy
    without exploration noise.
    """
    # Handle both old and new gym API
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state, _ = reset_result
    else:
        state = reset_result

    total_reward = 0.0
    rewards: List[float] = []
    step = 0

    done = False
    truncated = False

    while not (done or truncated):
        if max_steps is not None and step >= max_steps:
            break

        if render:
            env.render()

        # Deterministic action for evaluation
        action = agent.select_action(np.asarray(state), deterministic=True)

        # Handle both old and new gym API
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, done, truncated, _ = step_result
        else:
            next_state, reward, done, _ = step_result
            truncated = False

        total_reward += reward
        rewards.append(reward)
        state = next_state
        step += 1

    return total_reward, step, rewards


def evaluate_agent(
    agent: BaseAgent,
    env: Any,
    num_episodes: int = 10,
    max_steps: Optional[int] = None,
    render: bool = False,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate agent over multiple episodes.

    Parameters
    ----------
    agent : BaseAgent
        Agent to evaluate.
    env : gym.Env
        Environment instance.
    num_episodes : int, default=10
        Number of evaluation episodes.
    max_steps : int, optional
        Maximum steps per episode.
    render : bool, default=False
        Whether to render the environment.
    verbose : bool, default=False
        Whether to print per-episode results.

    Returns
    -------
    Dict[str, float]
        Evaluation statistics:
        - mean_reward: Average episode return
        - std_reward: Standard deviation of returns
        - min_reward: Minimum episode return
        - max_reward: Maximum episode return
        - mean_length: Average episode length
        - std_length: Standard deviation of lengths

    Example
    -------
    >>> stats = evaluate_agent(agent, env, num_episodes=10)
    >>> print(f"Mean return: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    """
    # Set agent to evaluation mode
    was_training = agent.training
    agent.eval_mode()

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    for ep in range(num_episodes):
        reward, length, _ = evaluate_episode(agent, env, max_steps, render)
        episode_rewards.append(reward)
        episode_lengths.append(length)

        if verbose:
            print(f"  Episode {ep + 1}/{num_episodes}: "
                  f"Return = {reward:.2f}, Length = {length}")

    # Restore training mode
    if was_training:
        agent.train_mode()

    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
    }


if __name__ == "__main__":
    print("Evaluator module loaded successfully.")
    print("Run with a gym environment to test evaluation functions.")
