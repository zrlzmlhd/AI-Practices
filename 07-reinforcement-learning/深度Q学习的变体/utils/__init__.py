"""
Training Utilities Module.

This module provides utilities for training, evaluation, and visualization:
    - train_agent: Complete training loop with logging
    - evaluate_agent: Evaluate agent performance
    - compare_variants: Compare different DQN variants
    - TrainingLogger: Structured training logging
    - plot_training_curves: Visualize training progress
    - plot_comparison: Compare variant performance

Core Idea (核心思想)
====================
提供生产级的训练工具链，支持：

1. **训练循环**: 完整的episode循环，包含early stopping和checkpointing
2. **评估**: 多episode测试，计算均值和标准差
3. **可视化**: 训练曲线、变体对比、消融分析
4. **日志**: 结构化日志记录，支持TensorBoard

Example:
    >>> from utils import train_agent, evaluate_agent, compare_variants
    >>> agent, history = train_agent(agent, "CartPole-v1", num_episodes=500)
    >>> mean_reward, std_reward = evaluate_agent(agent, "CartPole-v1")
    >>> results = compare_variants(variants, "CartPole-v1")
"""

from utils.training import (
    TrainingLogger,
    train_agent,
    evaluate_agent,
)
from utils.comparison import compare_variants
from utils.visualization import plot_training_curves, plot_comparison

__all__ = [
    "TrainingLogger",
    "train_agent",
    "evaluate_agent",
    "compare_variants",
    "plot_training_curves",
    "plot_comparison",
]
