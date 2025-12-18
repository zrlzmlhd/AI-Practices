#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
包装器工厂函数和通用包装器
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.core import Wrapper
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None
    Wrapper = object

from .observation import NormalizeObservationWrapper, FrameStackWrapper
from .action import ClipActionWrapper
from .reward import NormalizeRewardWrapper


class TimeLimitWrapper(Wrapper):
    """时间限制包装器：在指定步数后强制终止回合"""

    def __init__(self, env: "gym.Env", max_steps: int):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


class EpisodeStatisticsWrapper(Wrapper):
    """回合统计包装器：记录每个回合的奖励、长度等统计信息"""

    def __init__(self, env: "gym.Env", deque_size: int = 100):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")
        super().__init__(env)
        self.deque_size = deque_size
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_start_time = time.time()
        self.reward_history: Deque[float] = deque(maxlen=deque_size)
        self.length_history: Deque[int] = deque(maxlen=deque_size)
        self.total_steps = 0
        self.total_episodes = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_start_time = time.time()
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        self.total_steps += 1

        if terminated or truncated:
            episode_time = time.time() - self.episode_start_time
            self.reward_history.append(self.episode_reward)
            self.length_history.append(self.episode_length)
            self.total_episodes += 1
            info['episode'] = {'r': self.episode_reward, 'l': self.episode_length, 't': episode_time}

        return obs, reward, terminated, truncated, info

    def get_statistics(self) -> Dict[str, float]:
        if not self.reward_history:
            return {}
        return {
            'mean_reward': float(np.mean(self.reward_history)),
            'std_reward': float(np.std(self.reward_history)),
            'mean_length': float(np.mean(self.length_history)),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes
        }


class ActionRepeatWrapper(Wrapper):
    """动作重复包装器：重复执行同一动作多次，累积奖励"""

    def __init__(self, env: "gym.Env", n_repeats: int = 4):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")
        super().__init__(env)
        self.n_repeats = n_repeats

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        for _ in range(self.n_repeats):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_wrapped_env(
    env_id: str,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    clip_action: bool = True,
    frame_stack: int = 0,
    time_limit: Optional[int] = None,
    record_stats: bool = True,
    **kwargs
) -> "gym.Env":
    """
    创建带包装器的环境

    Parameters
    ----------
    env_id : str
        环境 ID
    normalize_obs : bool
        是否归一化观测
    normalize_reward : bool
        是否归一化奖励
    clip_action : bool
        是否裁剪动作 (仅连续动作空间)
    frame_stack : int
        帧堆叠数量，0 表示不堆叠
    time_limit : int, optional
        时间限制
    record_stats : bool
        是否记录统计信息

    Returns
    -------
    gymnasium.Env
        包装后的环境
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium 未安装")

    env = gym.make(env_id, **kwargs)

    if record_stats:
        env = EpisodeStatisticsWrapper(env)

    if time_limit is not None:
        env = TimeLimitWrapper(env, time_limit)

    if frame_stack > 0:
        env = FrameStackWrapper(env, n_frames=frame_stack)

    if normalize_obs and isinstance(env.observation_space, spaces.Box):
        env = NormalizeObservationWrapper(env)

    if clip_action and isinstance(env.action_space, spaces.Box):
        env = ClipActionWrapper(env)

    if normalize_reward:
        env = NormalizeRewardWrapper(env)

    return env


def _run_tests() -> bool:
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    print("\n" + "=" * 60)
    print("工厂函数模块单元测试")
    print("=" * 60)

    all_passed = True

    print("\n[测试 1] make_wrapped_env...")
    try:
        env = make_wrapped_env("Pendulum-v1", normalize_obs=True, normalize_reward=True, clip_action=True)
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        for _ in range(10):
            obs, reward, _, _, _ = env.step(env.action_space.sample())
        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    print("\n[测试 2] EpisodeStatisticsWrapper...")
    try:
        env = gym.make("CartPole-v1")
        env = EpisodeStatisticsWrapper(env)
        obs, _ = env.reset()
        for _ in range(500):
            obs, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                assert 'episode' in info
                obs, _ = env.reset()
        stats = env.get_statistics()
        assert 'mean_reward' in stats
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
