"""
Visualization Utilities Module.

This module provides plotting functions for training curves,
algorithm comparisons, and advantage estimation analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# Conditional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def plot_training_curves(
    results: Dict[str, List[float]],
    window_size: int = 50,
    save_path: Optional[str] = None,
    title: str = "Policy Gradient Methods Comparison",
    figsize: tuple = (12, 6),
) -> None:
    """
    Plot training curves with smoothing.

    Parameters
    ----------
    results : Dict[str, List[float]]
        Dictionary mapping algorithm names to reward histories.
    window_size : int, default=50
        Smoothing window size for moving average.
    save_path : str, optional
        Path to save the figure. If None, displays interactively.
    title : str, default="Policy Gradient Methods Comparison"
        Plot title.
    figsize : tuple, default=(12, 6)
        Figure size in inches.

    Examples
    --------
    >>> results = {
    ...     "REINFORCE": rewards_rf,
    ...     "A2C": rewards_a2c,
    ... }
    >>> plot_training_curves(results, save_path="comparison.png")
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib is required for plotting. Install: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (name, rewards) in enumerate(results.items()):
        color = colors[idx % len(colors)]

        # Plot raw data with transparency
        ax.plot(rewards, alpha=0.2, color=color)

        # Plot smoothed curve
        if len(rewards) > window_size:
            smoothed = np.convolve(
                rewards,
                np.ones(window_size) / window_size,
                mode="valid",
            )
            ax.plot(
                range(window_size - 1, len(rewards)),
                smoothed,
                label=name,
                color=color,
                linewidth=2,
            )
        else:
            ax.plot(rewards, label=name, color=color, linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


def plot_advantage_comparison(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    save_path: Optional[str] = None,
) -> None:
    """
    Compare different advantage estimation methods visually.

    Parameters
    ----------
    rewards : List[float]
        Episode rewards.
    values : List[float]
        Value estimates.
    gamma : float, default=0.99
        Discount factor.
    save_path : str, optional
        Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib is required")
        return

    from utils.returns import compute_gae, compute_returns

    dones = [False] * (len(rewards) - 1) + [True]

    # Compute different advantage estimates
    mc_returns = compute_returns(rewards, gamma, normalize=False)
    mc_advantages = mc_returns - np.array(values)

    lambdas = [0.0, 0.5, 0.9, 1.0]
    gae_advantages = {}

    for lam in lambdas:
        adv, _ = compute_gae(rewards, values, 0.0, dones, gamma, lam)
        gae_advantages[lam] = adv.numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, lam in zip(axes.flat, lambdas):
        ax.plot(gae_advantages[lam], label=f"GAE λ={lam}", linewidth=2)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Advantage")
        ax.set_title(f"λ={lam}: Var={np.var(gae_advantages[lam]):.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("GAE λ Parameter Effect on Advantage Variance", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_policy_gradient_intuition(save_path: Optional[str] = None) -> None:
    """
    Visualize policy gradient update intuition.

    Shows how the policy gradient increases probability of
    high-reward actions and decreases probability of low-reward actions.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib is required")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    actions = ["Left", "Right", "Jump"]
    initial_probs = [0.33, 0.33, 0.34]
    rewards = [0.1, 0.9, 0.0]
    updated_probs = [0.15, 0.70, 0.15]

    axes[0].bar(actions, initial_probs, color="steelblue", alpha=0.7)
    axes[0].set_ylim(0, 0.8)
    axes[0].set_title("Initial Policy π(a|s)", fontsize=12)
    axes[0].set_ylabel("Probability")

    axes[1].bar(actions, rewards, color="green", alpha=0.7)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Action Returns Q(s,a)", fontsize=12)
    axes[1].set_ylabel("Return")

    axes[2].bar(actions, updated_probs, color="orange", alpha=0.7)
    axes[2].set_ylim(0, 0.8)
    axes[2].set_title("Updated Policy π'(a|s)", fontsize=12)
    axes[2].set_ylabel("Probability")

    plt.suptitle(
        "Policy Gradient: Increase probability of high-return actions",
        y=1.02,
        fontsize=14,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_baseline_effect(save_path: Optional[str] = None) -> None:
    """
    Visualize how baselines reduce gradient variance.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib is required")
        return

    np.random.seed(42)

    n_samples = 1000
    base_return = 100
    returns = np.random.normal(base_return, 20, n_samples)

    gradient_no_baseline = returns
    baseline = returns.mean()
    gradient_with_baseline = returns - baseline

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(gradient_no_baseline, bins=50, alpha=0.7, color="steelblue")
    axes[0].axvline(x=0, color="red", linestyle="--", label="Zero")
    axes[0].set_title(
        f"No Baseline\nMean={np.mean(gradient_no_baseline):.1f}, "
        f"Var={np.var(gradient_no_baseline):.1f}"
    )
    axes[0].set_xlabel("Gradient Signal")
    axes[0].legend()

    axes[1].hist(gradient_with_baseline, bins=50, alpha=0.7, color="orange")
    axes[1].axvline(x=0, color="red", linestyle="--", label="Zero")
    axes[1].set_title(
        f"With Baseline\nMean={np.mean(gradient_with_baseline):.1f}, "
        f"Var={np.var(gradient_with_baseline):.1f}"
    )
    axes[1].set_xlabel("Gradient Signal")
    axes[1].legend()

    plt.suptitle("Baseline Reduces Variance While Preserving Expected Gradient", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
