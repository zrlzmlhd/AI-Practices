#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线统计工具

================================================================================
核心思想 (Core Idea)
================================================================================
在线统计 (Online Statistics) 是指在数据流到达时逐步更新统计量，而无需存储所有历史数据。
这对于强化学习中的观测归一化、奖励缩放等任务至关重要。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
**Welford 算法** (在线均值和方差计算):

增量均值更新:
    $$\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}$$

方差辅助量 M 更新:
    $$M_n = M_{n-1} + (x_n - \mu_{n-1})(x_n - \mu_n)$$

方差计算:
    $$\sigma^2 = \frac{M_n}{n}$$

**并行 Welford 算法** (用于批量更新):
合并两个样本集的统计量:
    $$\mu_{total} = \frac{n_a \mu_a + n_b \mu_b}{n_a + n_b}$$
    $$\delta = \mu_b - \mu_a$$
    $$M_{total} = M_a + M_b + \delta^2 \cdot \frac{n_a n_b}{n_a + n_b}$$

**指数移动平均 (EMA)**:
    $$\text{EMA}_t = \alpha \cdot x_t + (1-\alpha) \cdot \text{EMA}_{t-1}$$

其中 $\alpha = 1 - \text{decay}$ 是平滑系数。

================================================================================
问题背景 (Problem Statement)
================================================================================
强化学习训练中需要在线统计的场景:
1. 观测归一化：使神经网络输入具有零均值和单位方差
2. 奖励缩放：稳定 value function 的学习
3. 优势函数估计：GAE 中需要估计回报的均值和方差
4. 训练监控：跟踪回合奖励等指标的移动平均

================================================================================
算法总结 (Summary)
================================================================================
- RunningStatistics: 使用 Welford 算法计算均值和方差，支持批量更新
- ExponentialMovingAverage: 计算指数加权移动平均，适合平滑时间序列
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np


@dataclass
class RunningStatistics:
    """
    在线统计量计算器 (Welford 算法)

    使用 Welford 算法在线计算均值和方差，数值稳定且内存高效。
    支持单样本更新和批量更新。

    Attributes
    ----------
    shape : tuple of int
        数据形状
    mean : np.ndarray
        当前均值估计
    var : np.ndarray
        当前方差估计
    count : int
        已处理样本数

    Example
    -------
    >>> stats = RunningStatistics(shape=(4,))
    >>> for x in data:
    ...     stats.update(x)
    >>> normalized = stats.normalize(new_data)
    """
    shape: Tuple[int, ...]
    mean: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)
    count: int = field(default=0, init=False)
    _m2: np.ndarray = field(init=False)  # 方差辅助量

    def __post_init__(self):
        """初始化统计量为零均值、单位方差"""
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self._m2 = np.zeros(self.shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """
        使用新样本更新统计量

        支持单个样本或批量样本的更新。
        使用并行 Welford 算法进行批量更新。

        Parameters
        ----------
        x : np.ndarray
            新数据点
            - 单样本: 形状为 self.shape
            - 批量: 形状为 (batch_size, *self.shape)
        """
        # 统一为批量格式
        if x.ndim == len(self.shape):
            batch = np.expand_dims(x, 0)
        else:
            batch = x

        batch_count = batch.shape[0]
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)

        # 并行 Welford 算法 (Chan et al., 1979)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # 更新均值
        self.mean = self.mean + delta * batch_count / max(1, total_count)

        # 更新 M2 (方差辅助量)
        self._m2 = (
            self._m2 +
            batch_var * batch_count +
            delta**2 * self.count * batch_count / max(1, total_count)
        )

        self.count = total_count

        # 从 M2 计算方差
        if self.count > 1:
            self.var = self._m2 / self.count
        else:
            self.var = np.ones(self.shape, dtype=np.float64)

    def normalize(
        self,
        x: np.ndarray,
        epsilon: float = 1e-8,
        clip: float = 10.0
    ) -> np.ndarray:
        """
        标准化数据

        将数据转换为均值为 0、方差为 1 的分布。

        Parameters
        ----------
        x : np.ndarray
            待标准化数据
        epsilon : float
            数值稳定性常数，防止除零
        clip : float
            标准化后的裁剪范围 [-clip, clip]

        Returns
        -------
        np.ndarray
            标准化后的数据，dtype 为 float32
        """
        normalized = (x - self.mean) / np.sqrt(self.var + epsilon)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def denormalize(self, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        反标准化数据

        将标准化后的数据恢复到原始尺度。

        Parameters
        ----------
        x : np.ndarray
            标准化后的数据
        epsilon : float
            数值稳定性常数

        Returns
        -------
        np.ndarray
            原始尺度的数据
        """
        return x * np.sqrt(self.var + epsilon) + self.mean

    def reset(self) -> None:
        """重置统计量到初始状态"""
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self._m2 = np.zeros(self.shape, dtype=np.float64)
        self.count = 0

    def get_state(self) -> dict:
        """
        获取当前状态（用于保存）

        Returns
        -------
        dict
            包含 mean, var, count 的字典
        """
        return {
            'mean': self.mean.copy(),
            'var': self.var.copy(),
            'count': self.count
        }

    def set_state(self, state: dict) -> None:
        """
        设置状态（用于加载）

        Parameters
        ----------
        state : dict
            包含 mean, var, count 的字典
        """
        self.mean = state['mean'].copy()
        self.var = state['var'].copy()
        self.count = state['count']
        # 重建 _m2
        self._m2 = self.var * self.count


@dataclass
class ExponentialMovingAverage:
    """
    指数移动平均计算器

    用于平滑时间序列数据，对近期数据赋予更高权重。

    数学公式:
        EMA_t = decay * EMA_{t-1} + (1 - decay) * x_t

    Attributes
    ----------
    decay : float
        衰减系数，越大则历史权重越高
        - decay=0.99: 约 100 个样本的窗口
        - decay=0.999: 约 1000 个样本的窗口
    value : float
        当前 EMA 值
    initialized : bool
        是否已初始化

    Example
    -------
    >>> ema = ExponentialMovingAverage(decay=0.99)
    >>> for x in data:
    ...     smoothed = ema.update(x)
    >>> print(f"平滑值: {ema.value}")
    """
    decay: float = 0.99
    value: float = field(default=0.0, init=False)
    initialized: bool = field(default=False, init=False)

    def __post_init__(self):
        if not 0 <= self.decay <= 1:
            raise ValueError(f"decay 必须在 [0, 1] 范围内，得到 {self.decay}")

    def update(self, x: float) -> float:
        """
        更新 EMA

        Parameters
        ----------
        x : float
            新数据点

        Returns
        -------
        float
            更新后的 EMA 值
        """
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1 - self.decay) * x
        return self.value

    def reset(self) -> None:
        """重置到初始状态"""
        self.value = 0.0
        self.initialized = False


# =============================================================================
#                           单元测试
# =============================================================================

def _run_tests() -> bool:
    """运行单元测试"""
    print("\n" + "=" * 60)
    print("统计工具模块单元测试")
    print("=" * 60)

    all_passed = True

    # 测试 1: RunningStatistics 基本功能
    print("\n[测试 1] RunningStatistics 基本功能...")
    try:
        stats = RunningStatistics(shape=(4,))

        # 更新 1000 个标准正态样本
        np.random.seed(42)
        for _ in range(1000):
            stats.update(np.random.randn(4))

        # 检查均值接近 0
        assert np.allclose(stats.mean, 0, atol=0.1), f"均值偏差过大: {stats.mean}"
        # 检查方差接近 1
        assert np.allclose(stats.var, 1, atol=0.2), f"方差偏差过大: {stats.var}"

        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 2: 批量更新
    print("\n[测试 2] 批量更新...")
    try:
        stats = RunningStatistics(shape=(4,))

        # 批量更新
        batch = np.random.randn(100, 4)
        stats.update(batch)

        assert stats.count == 100
        assert stats.mean.shape == (4,)
        assert stats.var.shape == (4,)

        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 3: 标准化功能
    print("\n[测试 3] 标准化功能...")
    try:
        stats = RunningStatistics(shape=(4,))

        # 训练统计量
        for _ in range(500):
            stats.update(np.random.randn(4) * 2 + 3)  # 均值 3，标准差 2

        # 标准化
        x = np.array([3.0, 3.0, 3.0, 3.0])
        normalized = stats.normalize(x)

        # 均值附近的点应该接近 0
        assert np.allclose(normalized, 0, atol=0.5), f"标准化结果异常: {normalized}"

        # 测试裁剪
        x_extreme = np.array([100.0, 100.0, 100.0, 100.0])
        normalized_extreme = stats.normalize(x_extreme, clip=10.0)
        assert np.all(np.abs(normalized_extreme) <= 10.0), "裁剪失效"

        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 4: 状态保存和加载
    print("\n[测试 4] 状态保存和加载...")
    try:
        stats1 = RunningStatistics(shape=(4,))
        for _ in range(100):
            stats1.update(np.random.randn(4))

        # 保存状态
        state = stats1.get_state()

        # 创建新实例并加载
        stats2 = RunningStatistics(shape=(4,))
        stats2.set_state(state)

        assert np.allclose(stats1.mean, stats2.mean)
        assert np.allclose(stats1.var, stats2.var)
        assert stats1.count == stats2.count

        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 5: ExponentialMovingAverage
    print("\n[测试 5] ExponentialMovingAverage...")
    try:
        ema = ExponentialMovingAverage(decay=0.9)

        # 更新常数序列
        for _ in range(100):
            ema.update(10.0)

        # EMA 应该收敛到常数
        assert abs(ema.value - 10.0) < 0.1, f"EMA 未收敛: {ema.value}"

        # 测试重置
        ema.reset()
        assert ema.initialized is False
        assert ema.value == 0.0

        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    print("\n" + "=" * 60)
    print("测试结果:", "全部通过" if all_passed else "部分失败")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _run_tests()
