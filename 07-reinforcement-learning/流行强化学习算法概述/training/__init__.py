"""
Training Module
===============

Production-grade training utilities for continuous control RL algorithms.

Components
----------
- Trainer: Main training loop with logging and checkpointing
- evaluate_agent: Evaluation utilities
- plot_training_curves: Visualization
"""

from .trainer import Trainer
from .evaluator import evaluate_agent, evaluate_episode
from .visualization import plot_training_curves, plot_comparison

__all__ = [
    "Trainer",
    "evaluate_agent",
    "evaluate_episode",
    "plot_training_curves",
    "plot_comparison",
]
