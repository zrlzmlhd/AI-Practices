#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励包装器

提供奖励预处理功能：归一化、裁剪、符号化等。
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import numpy as np

from .statistics import RunningStatistics, ExponentialMovingAverage

try:
    import gymnasium as gym
    from gymnasium.core import RewardWrapper
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    RewardWrapper = object


class NormalizeRewardWrapper(RewardWrapper):
    """奖励归一化包装器：通过估计回报的标准差来缩放奖励"""

    def __init__(self, env: "gym.Env", gamma: float = 0.99, epsilon: float = 1e-8, clip: float = 10.0):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.running_stats = RunningStatistics(shape=(1,))
        self.returns: float = 0.0

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.returns = reward + self.gamma * self.returns * (1 - terminated)
        self.running_stats.update(np.array([self.returns]))
        normalized_reward = self.reward(reward)
        if terminated or truncated:
            self.returns = 0.0
        return obs, normalized_reward, terminated, truncated, info

    def reward(self, reward: float) -> float:
        std = np.sqrt(self.running_stats.var[0] + self.epsilon)
        normalized = reward / max(std, self.epsilon)
        return float(np.clip(normalized, -self.clip, self.clip))


class ClipRewardWrapper(RewardWrapper):
    """奖励裁剪包装器：将奖励裁剪到指定范围"""

    def __init__(self, env: "gym.Env", min_reward: float = -1.0, max_reward: float = 1.0):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, self.min_reward, self.max_reward))


class SignRewardWrapper(RewardWrapper):
    """符号奖励包装器：将奖励转换为其符号 (-1, 0, +1)"""

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


def _run_tests() -> bool:
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    print("\n" + "=" * 60)
    print("奖励包装器模块单元测试")
    print("=" * 60)

    all_passed = True

    print("\n[测试 1] NormalizeRewardWrapper...")
    try:
        env = gym.make("CartPole-v1")
        env = NormalizeRewardWrapper(env)
        obs, _ = env.reset()
        for _ in range(100):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    print("\n[测试 2] ClipRewardWrapper...")
    try:
        env = gym.make("Pendulum-v1")
        env = ClipRewardWrapper(env, min_reward=-1.0, max_reward=1.0)
        obs, _ = env.reset()
        obs, reward, _, _, _ = env.step(env.action_space.sample())
        assert -1.0 <= reward <= 1.0
        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    print("\n" + "=" * 60)
    print("测试结果:", "全部通过" if all_passed else "部分失败")
    return all_passed


if __name__ == "__main__":
    _run_tests()
