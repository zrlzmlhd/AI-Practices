"""
Training Visualization Utilities
================================

Core Idea
---------
Publication-quality plots for training progress and algorithm comparison.

Summary
-------
- plot_training_curves: Single agent training progress
- plot_comparison: Multiple algorithms side-by-side
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib() -> None:
    """Verify matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_training_curves(
    timesteps: List[int],
    rewards: List[float],
    stds: Optional[List[float]] = None,
    title: str = "Training Progress",
    xlabel: str = "Timesteps",
    ylabel: str = "Average Return",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    Plot training progress with optional confidence bands.

    Parameters
    ----------
    timesteps : List[int]
        X-axis values (environment steps).
    rewards : List[float]
        Mean evaluation returns.
    stds : List[float], optional
        Standard deviations for confidence bands.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : Tuple[int, int]
        Figure size in inches.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display plot.

    Returns
    -------
    plt.Figure or None
        Figure object if show=False.

    Example
    -------
    >>> plot_training_curves(
    ...     history.timesteps,
    ...     history.eval_rewards,
    ...     history.eval_stds,
    ...     title="SAC on Pendulum-v1"
    ... )
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    timesteps = np.array(timesteps)
    rewards = np.array(rewards)

    # Main line
    ax.plot(timesteps, rewards, linewidth=2, label="Mean Return")

    # Confidence band
    if stds is not None:
        stds = np.array(stds)
        ax.fill_between(
            timesteps,
            rewards - stds,
            rewards + stds,
            alpha=0.3,
            label="±1 Std",
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Format x-axis for large numbers
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}")
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None

    return fig


def plot_comparison(
    results: Dict[str, Dict[str, List]],
    title: str = "Algorithm Comparison",
    xlabel: str = "Timesteps",
    ylabel: str = "Average Return",
    figsize: Tuple[int, int] = (12, 7),
    save_path: Optional[str] = None,
    show: bool = True,
    colors: Optional[Dict[str, str]] = None,
) -> Optional["plt.Figure"]:
    """
    Compare multiple algorithms on the same plot.

    Parameters
    ----------
    results : Dict[str, Dict[str, List]]
        Algorithm results. Format:
        {
            "DDPG": {"timesteps": [...], "rewards": [...], "stds": [...]},
            "TD3": {"timesteps": [...], "rewards": [...], "stds": [...]},
            "SAC": {"timesteps": [...], "rewards": [...], "stds": [...]},
        }
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display plot.
    colors : Dict[str, str], optional
        Custom colors per algorithm.

    Returns
    -------
    plt.Figure or None
        Figure object if show=False.

    Example
    -------
    >>> results = {
    ...     "DDPG": {"timesteps": t1, "rewards": r1, "stds": s1},
    ...     "TD3": {"timesteps": t2, "rewards": r2, "stds": s2},
    ...     "SAC": {"timesteps": t3, "rewards": r3, "stds": s3},
    ... }
    >>> plot_comparison(results, title="Pendulum-v1")
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    default_colors = {
        "DDPG": "#1f77b4",  # Blue
        "TD3": "#ff7f0e",   # Orange
        "SAC": "#2ca02c",   # Green
    }

    if colors is None:
        colors = default_colors

    for algo_name, data in results.items():
        timesteps = np.array(data["timesteps"])
        rewards = np.array(data["rewards"])
        color = colors.get(algo_name, None)

        ax.plot(
            timesteps,
            rewards,
            linewidth=2,
            label=algo_name,
            color=color,
        )

        if "stds" in data and data["stds"] is not None:
            stds = np.array(data["stds"])
            ax.fill_between(
                timesteps,
                rewards - stds,
                rewards + stds,
                alpha=0.2,
                color=color,
            )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}")
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None

    return fig


def plot_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Training Metrics",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    Plot multiple training metrics in subplots.

    Parameters
    ----------
    metrics : Dict[str, List[float]]
        Metric name to values mapping.
    title : str
        Figure title.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display.

    Returns
    -------
    plt.Figure or None
        Figure if show=False.

    Example
    -------
    >>> plot_metrics(history.training_metrics)
    """
    _check_matplotlib()

    n_metrics = len(metrics)
    if n_metrics == 0:
        return None

    # Calculate grid size
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        ax.plot(values, linewidth=0.5, alpha=0.7)

        # Rolling mean for smoother visualization
        if len(values) > 100:
            window = min(100, len(values) // 10)
            rolling_mean = np.convolve(
                values, np.ones(window) / window, mode="valid"
            )
            ax.plot(
                range(window - 1, len(values)),
                rolling_mean,
                linewidth=2,
                color="red",
                label=f"Rolling mean (w={window})",
            )
            ax.legend(fontsize=9)

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Update Steps", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None

    return fig


if __name__ == "__main__":
    _check_matplotlib()

    # Demo with synthetic data
    np.random.seed(42)

    timesteps = list(range(0, 100001, 5000))
    n_points = len(timesteps)

    # Simulate learning curves
    ddpg_rewards = [-1000 + 800 * (1 - np.exp(-t / 30000)) + np.random.randn() * 50
                    for t in timesteps]
    td3_rewards = [-1000 + 850 * (1 - np.exp(-t / 25000)) + np.random.randn() * 40
                   for t in timesteps]
    sac_rewards = [-1000 + 900 * (1 - np.exp(-t / 20000)) + np.random.randn() * 30
                   for t in timesteps]

    ddpg_stds = [100 * np.exp(-t / 50000) + 20 for t in timesteps]
    td3_stds = [80 * np.exp(-t / 50000) + 15 for t in timesteps]
    sac_stds = [60 * np.exp(-t / 50000) + 10 for t in timesteps]

    results = {
        "DDPG": {"timesteps": timesteps, "rewards": ddpg_rewards, "stds": ddpg_stds},
        "TD3": {"timesteps": timesteps, "rewards": td3_rewards, "stds": td3_stds},
        "SAC": {"timesteps": timesteps, "rewards": sac_rewards, "stds": sac_stds},
    }

    print("Generating comparison plot...")
    plot_comparison(results, title="Algorithm Comparison (Simulated)")
