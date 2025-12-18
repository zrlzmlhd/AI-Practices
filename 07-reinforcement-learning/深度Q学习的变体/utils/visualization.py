"""
Visualization Utilities.

This module provides plotting functions for training analysis.

Core Idea (核心思想)
====================
生产级可视化工具，支持：
- 训练曲线绘制
- 变体性能对比
- 消融分析可视化
- 分布式学习可视化

All plots follow publication-ready standards with:
- Clear labels and legends
- Confidence intervals
- Consistent styling
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_training_curves(
    logger: Any,
    title: str = "Training Progress",
    smoothing_window: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Plot training curves from logger.

    Parameters
    ----------
    logger : TrainingLogger
        Training logger with episode statistics
    title : str, default="Training Progress"
        Plot title
    smoothing_window : int, default=10
        Window size for moving average smoothing
    save_path : Optional[Union[str, Path]]
        Path to save figure
    show : bool, default=True
        Whether to display plot

    Examples
    --------
    >>> from utils.training import TrainingLogger
    >>> logger = TrainingLogger()
    >>> # ... training ...
    >>> plot_training_curves(logger, title="DQN Training")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. "
            "Install with: pip install matplotlib"
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode rewards
    ax1 = axes[0, 0]
    rewards = logger.episode_rewards
    smoothed = logger.get_smoothed_rewards(smoothing_window)

    ax1.plot(rewards, alpha=0.3, color="blue", label="Raw")
    ax1.plot(smoothed, color="blue", linewidth=2, label=f"Smoothed ({smoothing_window})")
    ax1.axhline(logger.mean_reward, color="red", linestyle="--", label=f"Final: {logger.mean_reward:.1f}")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Episode lengths
    ax2 = axes[0, 1]
    if logger.episode_lengths:
        lengths = logger.episode_lengths
        ax2.plot(lengths, alpha=0.5, color="green")
        ax2.axhline(logger.mean_length, color="red", linestyle="--")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Length")
        ax2.set_title("Episode Lengths")
        ax2.grid(True, alpha=0.3)

    # Training loss
    ax3 = axes[1, 0]
    if logger.losses:
        losses = logger.losses
        ax3.plot(losses, alpha=0.5, color="orange")
        ax3.set_xlabel("Update Step")
        ax3.set_ylabel("Loss")
        ax3.set_title("Training Loss")
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")

    # Exploration rate
    ax4 = axes[1, 1]
    if logger.epsilons:
        epsilons = logger.epsilons
        ax4.plot(epsilons, color="purple")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Epsilon")
        ax4.set_title("Exploration Rate")
        ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    results: Dict[str, Any],
    metric: str = "reward",
    title: str = "Variant Comparison",
    smoothing_window: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Plot comparison of multiple variants.

    Parameters
    ----------
    results : Dict[str, VariantResult]
        Results from compare_variants()
    metric : str, default="reward"
        Metric to plot ("reward", "loss", "length")
    title : str, default="Variant Comparison"
        Plot title
    smoothing_window : int, default=10
        Smoothing window size
    save_path : Optional[Union[str, Path]]
        Path to save figure
    show : bool, default=True
        Whether to display plot

    Examples
    --------
    >>> results = compare_variants("CartPole-v1", variants=[...])
    >>> plot_comparison(results, title="DQN Variants on CartPole")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. "
            "Install with: pip install matplotlib"
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    ax1 = axes[0]
    for name, result in results.items():
        logger = result.logger
        rewards = logger.episode_rewards
        smoothed = logger.get_smoothed_rewards(smoothing_window)
        ax1.plot(smoothed, label=name, linewidth=2)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Learning Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar chart comparison
    ax2 = axes[1]
    names = list(results.keys())
    means = [r.final_mean for r in results.values()]
    stds = [r.final_std for r in results.values()]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax2.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="black")

    ax2.set_ylabel("Final Mean Reward")
    ax2.set_title("Final Performance")

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.1,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_ablation_study(
    baseline_name: str,
    baseline_reward: float,
    ablations: Dict[str, float],
    title: str = "Ablation Study",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Plot ablation study results.

    Parameters
    ----------
    baseline_name : str
        Name of baseline (e.g., "Rainbow")
    baseline_reward : float
        Baseline performance
    ablations : Dict[str, float]
        Ablation results: {"- Component": reward, ...}
    title : str, default="Ablation Study"
        Plot title
    save_path : Optional[Union[str, Path]]
        Path to save figure
    show : bool, default=True
        Whether to display plot

    Examples
    --------
    >>> plot_ablation_study(
    ...     "Rainbow", 441,
    ...     {"- PER": 358, "- Multi-step": 340, "- Distributional": 315}
    ... )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. "
            "Install with: pip install matplotlib"
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    names = [baseline_name] + list(ablations.keys())
    values = [baseline_reward] + list(ablations.values())

    # Colors: baseline green, ablations red
    colors = ["green"] + ["red"] * len(ablations)

    # Horizontal bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor="black")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Performance Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_distribution(
    support: np.ndarray,
    probs: np.ndarray,
    title: str = "Value Distribution",
    expected_value: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Plot categorical value distribution (for C51).

    Parameters
    ----------
    support : np.ndarray
        Support atoms, shape (num_atoms,)
    probs : np.ndarray
        Probabilities, shape (num_atoms,)
    title : str, default="Value Distribution"
        Plot title
    expected_value : Optional[float]
        Expected value to mark
    save_path : Optional[Union[str, Path]]
        Path to save figure
    show : bool, default=True
        Whether to display plot

    Examples
    --------
    >>> support = np.linspace(-10, 10, 51)
    >>> probs = np.random.softmax(np.random.randn(51))
    >>> plot_distribution(support, probs, title="Q-Value Distribution")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. "
            "Install with: pip install matplotlib"
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    delta_z = support[1] - support[0] if len(support) > 1 else 1.0
    ax.bar(support, probs, width=delta_z * 0.8, alpha=0.7, color="blue", edgecolor="black")

    if expected_value is not None:
        ax.axvline(expected_value, color="red", linestyle="--", linewidth=2, label=f"E[Z] = {expected_value:.2f}")
        ax.legend()

    ax.set_xlabel("Return")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
