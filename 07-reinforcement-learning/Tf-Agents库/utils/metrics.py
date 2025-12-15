#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习训练指标与可视化工具

提供训练过程中的性能指标收集、统计分析和可视化功能。

================================================================================
核心思想 (Core Idea)
================================================================================
有效的监控和可视化是深度强化学习调试和分析的关键工具。
本模块提供：
1. 实时指标收集和滑动窗口统计
2. 训练曲线绘制和对比分析
3. 策略性能评估和置信区间估计

================================================================================
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import json
import os


@dataclass
class TrainingMetrics:
    """
    训练指标收集器

    核心思想 (Core Idea):
        收集和管理强化学习训练过程中的各类指标，
        支持滑动窗口统计、持久化存储和批量分析。

    数学原理 (Mathematical Theory):
        滑动窗口平均：
        $$\bar{x}_t = \frac{1}{W} \sum_{i=t-W+1}^{t} x_i$$

        指数移动平均 (EMA)：
        $$\text{EMA}_t = \alpha \cdot x_t + (1-\alpha) \cdot \text{EMA}_{t-1}$$

    Attributes:
        window_size: 滑动窗口大小
        episode_rewards: 完整回合奖励历史
        episode_lengths: 完整回合长度历史
        losses: 训练损失历史
        q_values: Q 值估计历史
        epsilon_values: 探索率历史
        learning_rates: 学习率历史

    Example:
        >>> metrics = TrainingMetrics(window_size=100)
        >>> metrics.add_episode(reward=200, length=150)
        >>> metrics.add_loss(0.05)
        >>> stats = metrics.get_statistics()
    """

    window_size: int = 100
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    q_values: List[float] = field(default_factory=list)
    epsilon_values: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    _recent_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    _recent_lengths: deque = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self):
        """初始化滑动窗口"""
        self._recent_rewards = deque(maxlen=self.window_size)
        self._recent_lengths = deque(maxlen=self.window_size)

    def add_episode(self, reward: float, length: int) -> None:
        """
        记录回合统计

        Args:
            reward: 回合总奖励
            length: 回合步数
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self._recent_rewards.append(reward)
        self._recent_lengths.append(length)

    def add_loss(self, loss: float) -> None:
        """记录训练损失"""
        self.losses.append(loss)

    def add_q_value(self, q_value: float) -> None:
        """记录 Q 值估计"""
        self.q_values.append(q_value)

    def add_epsilon(self, epsilon: float) -> None:
        """记录探索率"""
        self.epsilon_values.append(epsilon)

    def add_learning_rate(self, lr: float) -> None:
        """记录学习率"""
        self.learning_rates.append(lr)

    def get_recent_mean_reward(self) -> float:
        """
        获取最近回合的平均奖励

        Returns:
            滑动窗口内的平均奖励
        """
        if not self._recent_rewards:
            return 0.0
        return np.mean(self._recent_rewards)

    def get_recent_mean_length(self) -> float:
        """获取最近回合的平均长度"""
        if not self._recent_lengths:
            return 0.0
        return np.mean(self._recent_lengths)

    def get_statistics(self) -> Dict[str, float]:
        """
        获取综合统计信息

        Returns:
            包含各类统计指标的字典
        """
        stats = {
            'num_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0.0,
            'max_reward': max(self.episode_rewards) if self.episode_rewards else 0.0,
            'min_reward': min(self.episode_rewards) if self.episode_rewards else 0.0,
            'recent_mean_reward': self.get_recent_mean_reward(),
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'recent_mean_length': self.get_recent_mean_length(),
            'mean_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
        }

        if self.q_values:
            stats['mean_q'] = np.mean(self.q_values[-100:])

        return stats

    def save(self, filepath: str) -> None:
        """
        保存指标到文件

        Args:
            filepath: 保存路径（JSON 格式）
        """
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'q_values': self.q_values,
            'epsilon_values': self.epsilon_values,
            'learning_rates': self.learning_rates
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'TrainingMetrics':
        """
        从文件加载指标

        Args:
            filepath: 文件路径

        Returns:
            加载的 TrainingMetrics 实例
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        metrics = cls()
        metrics.episode_rewards = data.get('episode_rewards', [])
        metrics.episode_lengths = data.get('episode_lengths', [])
        metrics.losses = data.get('losses', [])
        metrics.q_values = data.get('q_values', [])
        metrics.epsilon_values = data.get('epsilon_values', [])
        metrics.learning_rates = data.get('learning_rates', [])

        return metrics


def smooth_curve(
    values: List[float],
    weight: float = 0.9
) -> np.ndarray:
    """
    使用指数移动平均平滑曲线

    核心思想 (Core Idea):
        EMA 对近期值赋予更高权重，产生平滑的趋势曲线，
        便于观察长期训练趋势。

    数学原理 (Mathematical Theory):
        指数移动平均：
        $$\text{EMA}_t = \alpha \cdot x_t + (1-\alpha) \cdot \text{EMA}_{t-1}$$

        其中 $\alpha = 1 - \text{weight}$ 是平滑系数。
        weight 越大，平滑程度越高。

    Args:
        values: 原始值序列
        weight: 平滑权重（0-1，越大越平滑）

    Returns:
        平滑后的值序列
    """
    smoothed = []
    last = values[0] if values else 0

    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val

    return np.array(smoothed)


def plot_training_curves(
    metrics: TrainingMetrics,
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    show_raw: bool = True,
    smoothing_weight: float = 0.9
) -> plt.Figure:
    """
    绘制训练曲线

    核心思想 (Core Idea):
        可视化训练过程中的关键指标，包括奖励曲线、损失曲线等。
        同时显示原始数据和平滑曲线，便于分析。

    Args:
        metrics: TrainingMetrics 实例
        title: 图表标题
        save_path: 保存路径（None 则不保存）
        show_raw: 是否显示原始（未平滑）数据
        smoothing_weight: 平滑权重

    Returns:
        matplotlib Figure 对象
    """
    # 确定子图数量
    num_plots = 2  # 奖励和长度
    if metrics.losses:
        num_plots += 1
    if metrics.q_values:
        num_plots += 1

    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # 回合奖励曲线
    ax = axes[plot_idx]
    episodes = range(1, len(metrics.episode_rewards) + 1)

    if show_raw:
        ax.plot(episodes, metrics.episode_rewards, alpha=0.3, color='blue', label='Raw')

    if metrics.episode_rewards:
        smoothed_rewards = smooth_curve(metrics.episode_rewards, smoothing_weight)
        ax.plot(episodes, smoothed_rewards, color='blue', linewidth=2, label='Smoothed')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # 回合长度曲线
    ax = axes[plot_idx]

    if show_raw:
        ax.plot(episodes, metrics.episode_lengths, alpha=0.3, color='green', label='Raw')

    if metrics.episode_lengths:
        smoothed_lengths = smooth_curve(
            [float(x) for x in metrics.episode_lengths],
            smoothing_weight
        )
        ax.plot(episodes, smoothed_lengths, color='green', linewidth=2, label='Smoothed')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # 损失曲线（如果有）
    if metrics.losses:
        ax = axes[plot_idx]
        steps = range(1, len(metrics.losses) + 1)

        if show_raw:
            ax.plot(steps, metrics.losses, alpha=0.3, color='red', label='Raw')

        smoothed_losses = smooth_curve(metrics.losses, smoothing_weight)
        ax.plot(steps, smoothed_losses, color='red', linewidth=2, label='Smoothed')

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Q 值曲线（如果有）
    if metrics.q_values:
        ax = axes[plot_idx]
        steps = range(1, len(metrics.q_values) + 1)

        if show_raw:
            ax.plot(steps, metrics.q_values, alpha=0.3, color='purple', label='Raw')

        smoothed_q = smooth_curve(metrics.q_values, smoothing_weight)
        ax.plot(steps, smoothed_q, color='purple', linewidth=2, label='Smoothed')

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Q Value')
        ax.set_title('Mean Q Values')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def compare_runs(
    metrics_list: List[TrainingMetrics],
    labels: List[str],
    metric_name: str = 'reward',
    title: str = "Algorithm Comparison",
    save_path: Optional[str] = None,
    smoothing_weight: float = 0.9
) -> plt.Figure:
    """
    对比多次训练运行

    核心思想 (Core Idea):
        将多个算法或配置的训练曲线绘制在同一图上，便于性能对比。

    Args:
        metrics_list: TrainingMetrics 实例列表
        labels: 对应的标签列表
        metric_name: 对比的指标名称 ('reward', 'length', 'loss')
        title: 图表标题
        save_path: 保存路径
        smoothing_weight: 平滑权重

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))

    for metrics, label, color in zip(metrics_list, labels, colors):
        if metric_name == 'reward':
            values = metrics.episode_rewards
            ylabel = 'Total Reward'
            xlabel = 'Episode'
        elif metric_name == 'length':
            values = [float(x) for x in metrics.episode_lengths]
            ylabel = 'Episode Length'
            xlabel = 'Episode'
        elif metric_name == 'loss':
            values = metrics.losses
            ylabel = 'Loss'
            xlabel = 'Training Step'
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        if not values:
            continue

        x = range(1, len(values) + 1)

        # 绘制原始数据（淡色）
        ax.plot(x, values, alpha=0.2, color=color)

        # 绘制平滑曲线
        smoothed = smooth_curve(values, smoothing_weight)
        ax.plot(x, smoothed, color=color, linewidth=2, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def compute_confidence_interval(
    rewards_matrix: np.ndarray,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算多次运行的置信区间

    核心思想 (Core Idea):
        对多次随机种子运行的结果计算均值和置信区间，
        量化算法性能的可靠性。

    数学原理 (Mathematical Theory):
        假设回合奖励服从正态分布，$(1-\alpha)$ 置信区间：
        $$\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

        其中：
        - $\bar{x}$: 样本均值
        - $s$: 样本标准差
        - $n$: 样本数（运行次数）
        - $t$: t 分布临界值

    Args:
        rewards_matrix: 奖励矩阵，形状 (num_runs, num_episodes)
        confidence: 置信水平

    Returns:
        (均值, 下界, 上界) 元组
    """
    from scipy import stats

    mean = np.mean(rewards_matrix, axis=0)
    std = np.std(rewards_matrix, axis=0)
    n = rewards_matrix.shape[0]

    # t 分布临界值
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)

    margin = t_value * std / np.sqrt(n)

    lower = mean - margin
    upper = mean + margin

    return mean, lower, upper


def plot_with_confidence(
    rewards_matrix: np.ndarray,
    label: str = "Algorithm",
    confidence: float = 0.95,
    smoothing_weight: float = 0.9,
    ax: Optional[plt.Axes] = None,
    color: str = 'blue'
) -> plt.Axes:
    """
    绘制带置信区间的曲线

    Args:
        rewards_matrix: 奖励矩阵 (num_runs, num_episodes)
        label: 曲线标签
        confidence: 置信水平
        smoothing_weight: 平滑权重
        ax: matplotlib Axes（None 则创建新图）
        color: 曲线颜色

    Returns:
        matplotlib Axes 对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    mean, lower, upper = compute_confidence_interval(rewards_matrix, confidence)

    # 平滑
    mean_smooth = smooth_curve(list(mean), smoothing_weight)
    lower_smooth = smooth_curve(list(lower), smoothing_weight)
    upper_smooth = smooth_curve(list(upper), smoothing_weight)

    episodes = range(1, len(mean) + 1)

    # 绘制置信区间
    ax.fill_between(episodes, lower_smooth, upper_smooth, alpha=0.2, color=color)

    # 绘制均值曲线
    ax.plot(episodes, mean_smooth, color=color, linewidth=2, label=label)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


if __name__ == "__main__":
    # 测试指标收集和可视化
    print("=" * 60)
    print("训练指标与可视化测试")
    print("=" * 60)

    # 创建模拟数据
    metrics = TrainingMetrics(window_size=50)

    np.random.seed(42)
    for i in range(200):
        # 模拟逐渐提升的性能
        base_reward = 50 + i * 0.5 + np.random.randn() * 20
        length = max(10, int(100 + i * 0.3 + np.random.randn() * 10))

        metrics.add_episode(reward=base_reward, length=length)
        metrics.add_loss(1.0 / (i + 1) + np.random.randn() * 0.01)

    # 打印统计信息
    stats = metrics.get_statistics()
    print("\n训练统计:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")

    # 绘制训练曲线
    print("\n绘制训练曲线...")
    fig = plot_training_curves(metrics, title="模拟训练进度")
    plt.show()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
