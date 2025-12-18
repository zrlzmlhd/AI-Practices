#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
观测包装器

================================================================================
核心思想 (Core Idea)
================================================================================
观测包装器对环境返回的观测进行预处理，使其更适合神经网络处理。
常见操作包括归一化、帧堆叠、展平等。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
**观测归一化**:
    $$\tilde{s} = \frac{s - \mu}{\sigma + \epsilon}$$

其中 μ 和 σ 通过在线 Welford 算法估计。

**帧堆叠** (Frame Stacking):
将连续 k 帧堆叠为单个观测:
    $$s_t^{stacked} = [s_{t-k+1}, s_{t-k+2}, ..., s_t]$$

这为部分可观测环境提供时序信息。

================================================================================
问题背景 (Problem Statement)
================================================================================
原始观测可能存在的问题:
1. 不同维度尺度差异大，影响网络训练
2. 单帧观测不包含速度等动态信息
3. 多维观测需要展平为向量输入全连接网络
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from .statistics import RunningStatistics

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.core import ObservationWrapper
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None
    ObservationWrapper = object
    warnings.warn("gymnasium 未安装", ImportWarning)


class NormalizeObservationWrapper(ObservationWrapper):
    """
    观测归一化包装器

    在线估计观测的均值和方差，并将观测标准化到均值为 0、方差为 1 的分布。
    这对于使用神经网络的 RL 算法至关重要。

    Parameters
    ----------
    env : gymnasium.Env
        原始环境
    epsilon : float
        数值稳定性常数
    clip : float
        归一化后的裁剪范围

    Attributes
    ----------
    running_stats : RunningStatistics
        在线统计量计算器
    update_stats : bool
        是否更新统计量（训练/评估模式）

    Example
    -------
    >>> env = gym.make("Pendulum-v1")
    >>> env = NormalizeObservationWrapper(env)
    >>> obs, _ = env.reset()
    >>> # obs 现在是归一化的
    """

    def __init__(
        self,
        env: "gym.Env",
        epsilon: float = 1e-8,
        clip: float = 10.0,
        update_stats: bool = True
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip
        self.update_stats = update_stats

        # 获取观测形状
        obs_shape = env.observation_space.shape
        self.running_stats = RunningStatistics(shape=obs_shape)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        处理观测

        Parameters
        ----------
        observation : np.ndarray
            原始观测

        Returns
        -------
        np.ndarray
            归一化后的观测
        """
        if self.update_stats:
            self.running_stats.update(observation)

        return self.running_stats.normalize(
            observation,
            epsilon=self.epsilon,
            clip=self.clip
        )

    def set_training_mode(self, training: bool) -> None:
        """
        设置训练/评估模式

        Parameters
        ----------
        training : bool
            True 表示训练模式（更新统计量）
            False 表示评估模式（固定统计量）
        """
        self.update_stats = training

    def get_statistics(self) -> Dict[str, Any]:
        """获取当前统计量（用于保存）"""
        return self.running_stats.get_state()

    def set_statistics(self, stats: Dict[str, Any]) -> None:
        """设置统计量（用于加载）"""
        self.running_stats.set_state(stats)


class FrameStackWrapper(ObservationWrapper):
    """
    帧堆叠包装器

    将连续 k 帧观测堆叠为单个观测，为部分可观测环境提供时序信息。
    这对于需要从观测中推断速度等动态信息的场景非常有用。

    Parameters
    ----------
    env : gymnasium.Env
        原始环境
    n_frames : int
        堆叠帧数
    stack_axis : int
        堆叠维度

    Example
    -------
    >>> env = gym.make("CartPole-v1")
    >>> env = FrameStackWrapper(env, n_frames=4)
    >>> obs, _ = env.reset()  # shape: (4, 4)
    """

    def __init__(
        self,
        env: "gym.Env",
        n_frames: int = 4,
        stack_axis: int = 0
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)
        self.n_frames = n_frames
        self.stack_axis = stack_axis

        # 初始化帧缓冲区
        self.frames: Deque[np.ndarray] = deque(maxlen=n_frames)

        # 更新观测空间
        old_space = env.observation_space
        low = np.repeat(old_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(old_space.high[np.newaxis, ...], n_frames, axis=0)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=old_space.dtype
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境并初始化帧缓冲"""
        obs, info = self.env.reset(**kwargs)

        # 用初始帧填充缓冲区
        for _ in range(self.n_frames):
            self.frames.append(obs)

        return self._get_stacked_obs(), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """处理单帧观测"""
        self.frames.append(observation)
        return self._get_stacked_obs()

    def _get_stacked_obs(self) -> np.ndarray:
        """获取堆叠后的观测"""
        return np.stack(list(self.frames), axis=self.stack_axis)


class FlattenObservationWrapper(ObservationWrapper):
    """
    观测展平包装器

    将多维观测展平为一维向量，便于全连接网络处理。

    Parameters
    ----------
    env : gymnasium.Env
        原始环境
    """

    def __init__(self, env: "gym.Env"):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)

        # 计算展平后的维度
        old_space = env.observation_space
        flat_dim = int(np.prod(old_space.shape))

        self.observation_space = spaces.Box(
            low=old_space.low.flatten(),
            high=old_space.high.flatten(),
            dtype=old_space.dtype
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """展平观测"""
        return observation.flatten()


# =============================================================================
#                           单元测试
# =============================================================================

def _run_tests() -> bool:
    """运行单元测试"""
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    print("\n" + "=" * 60)
    print("观测包装器模块单元测试")
    print("=" * 60)

    all_passed = True

    # 测试 1: NormalizeObservationWrapper
    print("\n[测试 1] NormalizeObservationWrapper...")
    try:
        env = gym.make("CartPole-v1")
        env = NormalizeObservationWrapper(env)

        obs, _ = env.reset()
        assert obs.dtype == np.float32, f"dtype 错误: {obs.dtype}"

        # 运行一些步以收集统计量
        for _ in range(100):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.all(np.abs(obs) <= 10), f"观测超出裁剪范围: {obs}"
            if terminated or truncated:
                obs, _ = env.reset()

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 2: 训练/评估模式切换
    print("\n[测试 2] 训练/评估模式切换...")
    try:
        env = gym.make("CartPole-v1")
        env = NormalizeObservationWrapper(env)

        # 训练模式
        env.set_training_mode(True)
        obs, _ = env.reset()
        count_before = env.running_stats.count

        env.step(env.action_space.sample())
        count_after = env.running_stats.count
        assert count_after > count_before, "训练模式应更新统计量"

        # 评估模式
        env.set_training_mode(False)
        count_before = env.running_stats.count
        env.step(env.action_space.sample())
        count_after = env.running_stats.count
        assert count_after == count_before, "评估模式不应更新统计量"

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 3: FrameStackWrapper
    print("\n[测试 3] FrameStackWrapper...")
    try:
        env = gym.make("CartPole-v1")
        env = FrameStackWrapper(env, n_frames=4)

        obs, _ = env.reset()
        assert obs.shape == (4, 4), f"堆叠观测形状错误: {obs.shape}"

        # 检查初始帧都相同
        assert np.allclose(obs[0], obs[1]), "初始帧应相同"

        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (4, 4), f"步后形状错误: {obs.shape}"

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 4: FlattenObservationWrapper
    print("\n[测试 4] FlattenObservationWrapper...")
    try:
        # 先堆叠再展平
        env = gym.make("CartPole-v1")
        env = FrameStackWrapper(env, n_frames=4)
        env = FlattenObservationWrapper(env)

        obs, _ = env.reset()
        assert obs.shape == (16,), f"展平后形状错误: {obs.shape}"

        env.close()
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
