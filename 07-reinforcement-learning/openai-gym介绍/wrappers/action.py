#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作包装器

提供动作预处理功能：裁剪、重缩放、粘性动作等。
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.core import ActionWrapper
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None
    ActionWrapper = object


class ClipActionWrapper(ActionWrapper):
    """动作裁剪包装器：将动作裁剪到有效范围内"""

    def __init__(self, env: "gym.Env"):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("ClipActionWrapper 仅适用于 Box 动作空间")

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.action_space.low, self.action_space.high)


class RescaleActionWrapper(ActionWrapper):
    """动作重缩放包装器：将 [-1, 1] 映射到环境实际范围"""

    def __init__(self, env: "gym.Env"):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("RescaleActionWrapper 仅适用于 Box 动作空间")

        self.low = env.action_space.low
        self.high = env.action_space.high

        self.action_space = spaces.Box(
            low=-np.ones_like(self.low),
            high=np.ones_like(self.high),
            dtype=env.action_space.dtype
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        return self.low + (action + 1.0) * (self.high - self.low) / 2.0

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        return 2.0 * (action - self.low) / (self.high - self.low) - 1.0


class StickyActionWrapper(ActionWrapper):
    """粘性动作包装器：以一定概率重复上一步的动作"""

    def __init__(self, env: "gym.Env", sticky_prob: float = 0.25):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")
        super().__init__(env)
        self.sticky_prob = sticky_prob
        self.last_action: Optional[Any] = None

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.last_action = None
        return self.env.reset(**kwargs)

    def action(self, action: Any) -> Any:
        if self.last_action is not None and np.random.random() < self.sticky_prob:
            return self.last_action
        self.last_action = action
        return action


def _run_tests() -> bool:
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    print("\n" + "=" * 60)
    print("动作包装器模块单元测试")
    print("=" * 60)

    all_passed = True

    print("\n[测试 1] ClipActionWrapper...")
    try:
        env = gym.make("Pendulum-v1")
        env = ClipActionWrapper(env)
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(np.array([100.0]))
        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    print("\n[测试 2] RescaleActionWrapper...")
    try:
        env = gym.make("Pendulum-v1")
        env = RescaleActionWrapper(env)
        assert np.allclose(env.action_space.low, -1)
        assert np.allclose(env.action_space.high, 1)
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(np.array([0.0]))
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
