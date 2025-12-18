"""
Variant Comparison Utilities.

This module provides tools for comparing DQN variants.

Core Idea (核心思想)
====================
系统化地比较不同DQN变体的性能，支持：
- 多变体并行训练
- 统计显著性分析
- 消融研究

Comparison Metrics (对比指标)
============================
1. **Final Performance**: 最终100episode平均奖励
2. **Sample Efficiency**: 达到目标奖励所需episode数
3. **Training Stability**: 奖励方差
4. **Wall-clock Time**: 实际训练时间
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import DQNVariantConfig
from core.enums import DQNVariant
from agents.variant_agent import DQNVariantAgent
from utils.training import TrainingLogger, train_agent


@dataclass
class VariantResult:
    """
    Results from training a single variant.

    Attributes
    ----------
    variant : DQNVariant
        Algorithm variant
    logger : TrainingLogger
        Training statistics
    agent : DQNVariantAgent
        Trained agent
    final_mean : float
        Final mean reward
    final_std : float
        Final reward standard deviation
    episodes_to_solve : Optional[int]
        Episodes to reach target (if applicable)
    training_time : float
        Wall-clock training time (seconds)
    """
    variant: DQNVariant
    logger: TrainingLogger
    agent: DQNVariantAgent
    final_mean: float
    final_std: float
    episodes_to_solve: Optional[int]
    training_time: float


def compare_variants(
    env_name: str,
    variants: Optional[List[DQNVariant]] = None,
    config: Optional[DQNVariantConfig] = None,
    num_episodes: int = 500,
    num_runs: int = 1,
    target_reward: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, VariantResult]:
    """
    Compare multiple DQN variants on given environment.

    Core Idea (核心思想)
    --------------------
    在相同配置下训练多个DQN变体，比较其性能指标：

    1. **Final Performance**: 最终平均奖励
    2. **Sample Efficiency**: 达到目标所需episode数
    3. **Training Stability**: 奖励方差

    Parameters
    ----------
    env_name : str
        Gymnasium environment name
    variants : Optional[List[DQNVariant]]
        Variants to compare (default: all)
    config : Optional[DQNVariantConfig]
        Shared configuration (will infer dims from env if None)
    num_episodes : int, default=500
        Training episodes per variant
    num_runs : int, default=1
        Number of runs per variant for averaging
    target_reward : Optional[float]
        Target reward for sample efficiency
    seed : Optional[int]
        Base random seed
    verbose : bool, default=True
        Whether to print progress

    Returns
    -------
    Dict[str, VariantResult]
        Results keyed by variant name

    Examples
    --------
    >>> results = compare_variants(
    ...     "CartPole-v1",
    ...     variants=[DQNVariant.DOUBLE, DQNVariant.RAINBOW],
    ...     num_episodes=200,
    ... )
    >>> for name, result in results.items():
    ...     print(f"{name}: {result.final_mean:.1f} ± {result.final_std:.1f}")

    Notes
    -----
    - Requires gymnasium package
    - Training time scales with num_variants × num_episodes × num_runs
    """
    try:
        import gymnasium as gym
    except ImportError:
        raise ImportError(
            "gymnasium required for comparison. "
            "Install with: pip install gymnasium"
        )

    if variants is None:
        variants = list(DQNVariant)

    # Infer dimensions from environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    if config is None:
        config = DQNVariantConfig(
            state_dim=state_dim,
            action_dim=action_dim,
        )
    else:
        config = DQNVariantConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
            epsilon_decay_steps=config.epsilon_decay_steps,
        )

    results: Dict[str, VariantResult] = {}

    for variant in variants:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training {variant}")
            print(f"{'='*50}")

        run_rewards = []
        run_loggers = []
        run_times = []
        episodes_to_solve_list = []

        for run in range(num_runs):
            run_seed = seed + run if seed is not None else None

            agent = DQNVariantAgent(config, variant)

            agent_trained, logger = train_agent(
                agent,
                env_name,
                num_episodes=num_episodes,
                target_reward=target_reward,
                seed=run_seed,
                verbose=verbose and num_runs == 1,
            )

            run_rewards.append(logger.episode_rewards)
            run_loggers.append(logger)
            run_times.append(
                logger.timestamps[-1] if logger.timestamps else 0.0
            )

            # Check episodes to solve
            if target_reward is not None:
                solved_ep = None
                for i, r in enumerate(logger.get_smoothed_rewards(100)):
                    if r >= target_reward:
                        solved_ep = i
                        break
                episodes_to_solve_list.append(solved_ep)

        # Aggregate results
        final_rewards = [
            np.mean(rewards[-100:]) for rewards in run_rewards
        ]
        final_mean = float(np.mean(final_rewards))
        final_std = float(np.std(final_rewards))

        episodes_to_solve = None
        if target_reward is not None and episodes_to_solve_list:
            valid_episodes = [e for e in episodes_to_solve_list if e is not None]
            if valid_episodes:
                episodes_to_solve = int(np.mean(valid_episodes))

        results[str(variant)] = VariantResult(
            variant=variant,
            logger=run_loggers[0],
            agent=agent_trained,
            final_mean=final_mean,
            final_std=final_std,
            episodes_to_solve=episodes_to_solve,
            training_time=float(np.mean(run_times)),
        )

        if verbose:
            print(f"\n{variant} Summary:")
            print(f"  Final Reward: {final_mean:.1f} ± {final_std:.1f}")
            if episodes_to_solve is not None:
                print(f"  Episodes to Solve: {episodes_to_solve}")
            print(f"  Training Time: {np.mean(run_times):.1f}s")

    if verbose:
        print(f"\n{'='*50}")
        print("Comparison Complete")
        print(f"{'='*50}")
        _print_comparison_table(results)

    return results


def _print_comparison_table(results: Dict[str, VariantResult]) -> None:
    """Print formatted comparison table."""
    print("\n{:<20} {:>12} {:>12} {:>12}".format(
        "Variant", "Mean±Std", "Solved@", "Time(s)"
    ))
    print("-" * 60)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].final_mean,
        reverse=True,
    )

    for name, result in sorted_results:
        solved_str = (
            str(result.episodes_to_solve)
            if result.episodes_to_solve is not None
            else "N/A"
        )
        print(
            f"{name:<20} "
            f"{result.final_mean:>6.1f}±{result.final_std:<5.1f} "
            f"{solved_str:>12} "
            f"{result.training_time:>12.1f}"
        )
