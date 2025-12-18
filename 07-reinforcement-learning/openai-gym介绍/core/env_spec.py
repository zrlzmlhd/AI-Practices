#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gymnasium 环境规格提取工具

================================================================================
核心思想 (Core Idea)
================================================================================
EnvironmentSpec 封装了 Gymnasium 环境的完整元信息，包括观测空间、动作空间、
奖励范围、最大步数等。通过统一的规格对象，算法可以自动适配不同环境。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
环境规格对应 MDP 的形式化定义:

    $$\text{MDP} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, T)$$

其中:
- $\mathcal{S}$: 状态空间 → observation_space
- $\mathcal{A}$: 动作空间 → action_space
- $R$: 奖励函数 → reward_range
- $T$: 最大时间步 → max_episode_steps

EnvironmentSpec 提供这些信息的结构化访问接口。

================================================================================
问题背景 (Problem Statement)
================================================================================
强化学习算法需要根据环境特性进行配置:
1. 根据观测维度设置网络输入层
2. 根据动作空间类型选择输出层（softmax vs tanh）
3. 根据奖励范围选择合适的缩放策略
4. 根据最大步数设置回放缓冲区大小

手动查询这些信息容易出错，EnvironmentSpec 提供自动化解决方案。

================================================================================
算法总结 (Summary)
================================================================================
EnvironmentSpec 的使用流程:
1. 创建环境: env = gym.make(env_id)
2. 提取规格: spec = get_env_spec(env)
3. 配置算法: 根据 spec 设置网络架构和超参数
4. 开始训练: 使用正确配置的算法与环境交互
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .spaces import SpaceInfo, SpaceType, analyze_space

try:
    import gymnasium as gym
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    warnings.warn("gymnasium 未安装", ImportWarning)


@dataclass
class EnvironmentSpec:
    """
    环境规格完整描述

    收集并组织环境的所有关键信息，便于算法适配和配置。

    Attributes
    ----------
    env_id : str
        环境唯一标识符，如 "CartPole-v1"
    obs_info : SpaceInfo
        观测空间详细信息
    action_info : SpaceInfo
        动作空间详细信息
    reward_range : tuple of float
        奖励值范围 (min, max)
    max_episode_steps : int or None
        单回合最大步数，None 表示无限制
    is_discrete_action : bool
        是否为离散动作空间
    is_image_obs : bool
        是否为图像观测（3维以上）
    metadata : dict
        环境元数据，包含渲染模式等信息

    Example
    -------
    >>> env = gym.make("CartPole-v1")
    >>> spec = get_env_spec(env)
    >>> print(f"环境: {spec.env_id}")
    环境: CartPole-v1
    >>> print(f"离散动作: {spec.is_discrete_action}")
    离散动作: True
    >>> print(f"观测维度: {spec.obs_info.flat_dim}")
    观测维度: 4
    """
    env_id: str
    obs_info: SpaceInfo
    action_info: SpaceInfo
    reward_range: Tuple[float, float] = (-float('inf'), float('inf'))
    max_episode_steps: Optional[int] = None
    is_discrete_action: bool = True
    is_image_obs: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """后初始化：自动推断派生属性"""
        self.is_discrete_action = self.action_info.space_type in (
            SpaceType.DISCRETE,
            SpaceType.MULTI_DISCRETE,
            SpaceType.MULTI_BINARY
        )

        if self.obs_info.shape is not None:
            self.is_image_obs = len(self.obs_info.shape) >= 3

    @property
    def obs_dim(self) -> int:
        """观测空间展平维度"""
        return self.obs_info.flat_dim

    @property
    def action_dim(self) -> int:
        """
        动作空间维度

        - 离散空间: 返回动作数量
        - 连续空间: 返回动作向量维度
        """
        if self.is_discrete_action:
            return self.action_info.n if self.action_info.n else self.action_info.flat_dim
        return self.action_info.flat_dim

    @property
    def n_actions(self) -> Optional[int]:
        """离散动作空间的动作数量，连续空间返回 None"""
        if self.is_discrete_action and self.action_info.n is not None:
            return self.action_info.n
        return None

    def summary(self) -> str:
        """生成环境规格摘要字符串"""
        lines = [
            f"环境规格: {self.env_id}",
            "-" * 40,
            f"观测空间:",
            f"  类型: {self.obs_info.space_type.name}",
            f"  形状: {self.obs_info.shape}",
            f"  维度: {self.obs_dim}",
            f"动作空间:",
            f"  类型: {self.action_info.space_type.name}",
            f"  维度: {self.action_dim}",
            f"  离散: {self.is_discrete_action}",
            f"奖励范围: {self.reward_range}",
            f"最大步数: {self.max_episode_steps}",
        ]
        return "\n".join(lines)


def get_env_spec(env: "gym.Env") -> EnvironmentSpec:
    """
    提取环境完整规格

    从 Gymnasium 环境实例中提取所有关键元信息，
    构建 EnvironmentSpec 对象。

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium 环境实例

    Returns
    -------
    EnvironmentSpec
        环境规格数据对象

    Example
    -------
    >>> env = gym.make("CartPole-v1")
    >>> spec = get_env_spec(env)
    >>> print(spec.summary())
    环境规格: CartPole-v1
    ----------------------------------------
    观测空间:
      类型: BOX
      形状: (4,)
      维度: 4
    动作空间:
      类型: DISCRETE
      维度: 2
      离散: True
    奖励范围: (-inf, inf)
    最大步数: 500
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium 未安装")

    env_id = env.spec.id if env.spec else "unknown"
    max_steps = None

    if env.spec is not None:
        max_steps = env.spec.max_episode_steps

    reward_range = getattr(env, 'reward_range', (-float('inf'), float('inf')))
    metadata = getattr(env, 'metadata', {})

    return EnvironmentSpec(
        env_id=env_id,
        obs_info=analyze_space(env.observation_space),
        action_info=analyze_space(env.action_space),
        reward_range=reward_range,
        max_episode_steps=max_steps,
        metadata=metadata
    )


def get_env_spec_by_id(env_id: str, **make_kwargs) -> EnvironmentSpec:
    """
    通过环境 ID 获取环境规格

    便捷函数，自动创建和关闭环境。

    Parameters
    ----------
    env_id : str
        环境 ID，如 "CartPole-v1"
    **make_kwargs
        传递给 gym.make 的参数

    Returns
    -------
    EnvironmentSpec
        环境规格数据对象
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium 未安装")

    env = gym.make(env_id, **make_kwargs)
    spec = get_env_spec(env)
    env.close()
    return spec


# =============================================================================
#                           单元测试
# =============================================================================

def _run_tests() -> bool:
    """运行单元测试"""
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    print("\n" + "=" * 60)
    print("环境规格模块单元测试")
    print("=" * 60)

    all_passed = True

    # 测试 1: CartPole 规格
    print("\n[测试 1] CartPole 环境规格...")
    try:
        env = gym.make("CartPole-v1")
        spec = get_env_spec(env)

        assert spec.env_id == "CartPole-v1"
        assert spec.is_discrete_action is True
        assert spec.max_episode_steps == 500
        assert spec.obs_dim == 4
        assert spec.action_dim == 2
        assert spec.n_actions == 2

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 2: Pendulum 规格
    print("\n[测试 2] Pendulum 环境规格...")
    try:
        env = gym.make("Pendulum-v1")
        spec = get_env_spec(env)

        assert spec.env_id == "Pendulum-v1"
        assert spec.is_discrete_action is False
        assert spec.obs_dim == 3
        assert spec.action_dim == 1
        assert spec.n_actions is None  # 连续空间

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 3: 通过 ID 获取规格
    print("\n[测试 3] 通过 ID 获取规格...")
    try:
        spec = get_env_spec_by_id("MountainCar-v0")

        assert spec.env_id == "MountainCar-v0"
        assert spec.obs_dim == 2
        assert spec.action_dim == 3

        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 4: 规格摘要
    print("\n[测试 4] 规格摘要生成...")
    try:
        spec = get_env_spec_by_id("CartPole-v1")
        summary = spec.summary()

        assert "CartPole-v1" in summary
        assert "BOX" in summary
        assert "DISCRETE" in summary

        print("  [通过]")
        print("\n摘要预览:")
        print(summary)
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    print("\n" + "=" * 60)
    print("测试结果:", "全部通过" if all_passed else "部分失败")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _run_tests()
