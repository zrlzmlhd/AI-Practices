#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gymnasium 空间分析工具

================================================================================
核心思想 (Core Idea)
================================================================================
Gymnasium 使用 Space 对象来定义观测空间和动作空间的结构。本模块提供工具函数
用于分析这些空间的属性，帮助构建兼容的神经网络架构和采样策略。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
强化学习中的空间定义:

**离散空间 (Discrete)**:
    $$\mathcal{A} = \{0, 1, \ldots, n-1\}$$

**连续空间 (Box)**:
    $$\mathcal{S} = \{s \in \mathbb{R}^d : l_i \leq s_i \leq h_i, \forall i\}$$

**多离散空间 (MultiDiscrete)**:
    $$\mathcal{A} = \mathcal{A}_1 \times \mathcal{A}_2 \times \ldots \times \mathcal{A}_k$$
    其中 $\mathcal{A}_i = \{0, 1, \ldots, n_i-1\}$

================================================================================
问题背景 (Problem Statement)
================================================================================
不同的 RL 算法需要根据空间类型选择不同的策略网络架构:
- 离散动作空间: 使用 softmax 输出层，输出动作概率分布
- 连续动作空间: 使用高斯策略，输出均值和标准差
- 混合空间: 需要组合多种输出头

本模块提供统一的空间分析接口，简化算法与环境的适配。

================================================================================
算法对比 (Comparison)
================================================================================
| 空间类型      | 网络输出    | 采样方法        | 典型环境          |
|--------------|------------|----------------|------------------|
| Discrete     | logits     | Categorical    | CartPole, Atari  |
| Box          | μ, σ       | Gaussian       | MuJoCo, Pendulum |
| MultiDiscrete| multi-head | Multi-Categorical| 多智能体       |

================================================================================
复杂度 (Complexity)
================================================================================
- analyze_space: O(1) 对于基本空间，O(n) 对于组合空间
- 空间复杂度: O(d) 其中 d 为空间维度

================================================================================
算法总结 (Summary)
================================================================================
空间分析是构建 RL 系统的第一步:
1. 识别空间类型以选择合适的网络架构
2. 提取空间维度以确定网络输入/输出大小
3. 获取边界信息以进行动作裁剪或归一化
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None
    warnings.warn(
        "gymnasium 未安装。请执行: pip install gymnasium[classic-control]",
        ImportWarning
    )


class SpaceType(Enum):
    """
    Gymnasium 空间类型枚举

    用于标识不同的空间类型，便于算法根据类型选择合适的处理策略。
    """
    DISCRETE = auto()       # 离散空间: {0, 1, ..., n-1}
    BOX = auto()            # 连续空间: [low, high]^n
    MULTI_DISCRETE = auto() # 多离散空间: 多个独立的离散空间
    MULTI_BINARY = auto()   # 多二值空间: {0, 1}^n
    TUPLE = auto()          # 元组空间: 多个空间的组合
    DICT = auto()           # 字典空间: 命名空间的组合


@dataclass(frozen=True)
class SpaceInfo:
    """
    空间信息数据类

    封装 Gymnasium 空间的关键属性，便于算法选择合适的策略网络架构。
    使用 frozen=True 确保不可变性，适合作为缓存键。

    Attributes
    ----------
    space_type : SpaceType
        空间类型枚举值
    shape : tuple of int, optional
        空间形状，用于 Box 和 MultiDiscrete 空间
    n : int, optional
        离散空间的动作数量，仅用于 Discrete 空间
    low : np.ndarray, optional
        连续空间的下界，仅用于 Box 空间
    high : np.ndarray, optional
        连续空间的上界，仅用于 Box 空间
    dtype : np.dtype, optional
        空间的数据类型
    is_bounded : bool
        空间是否有界（所有维度都有有限的上下界）
    flat_dim : int
        空间展平后的维度，用于确定网络层大小

    Example
    -------
    >>> env = gym.make("CartPole-v1")
    >>> info = analyze_space(env.observation_space)
    >>> print(f"类型: {info.space_type}, 维度: {info.flat_dim}")
    类型: SpaceType.BOX, 维度: 4
    """
    space_type: SpaceType
    shape: Optional[Tuple[int, ...]] = None
    n: Optional[int] = None
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    dtype: Optional[np.dtype] = None
    is_bounded: bool = True
    flat_dim: int = 0


def analyze_space(space: "spaces.Space") -> SpaceInfo:
    """
    分析 Gymnasium 空间属性

    根据空间类型提取关键信息，用于构建兼容的神经网络架构。
    支持所有 Gymnasium 内置空间类型。

    Parameters
    ----------
    space : gymnasium.spaces.Space
        Gymnasium 空间对象，可以是 Discrete、Box、MultiDiscrete 等

    Returns
    -------
    SpaceInfo
        包含空间关键属性的数据对象

    Raises
    ------
    ImportError
        如果 gymnasium 未安装
    ValueError
        如果遇到不支持的空间类型

    Example
    -------
    >>> import gymnasium as gym
    >>> env = gym.make("CartPole-v1")
    >>>
    >>> # 分析观测空间
    >>> obs_info = analyze_space(env.observation_space)
    >>> print(f"观测维度: {obs_info.shape}, 类型: {obs_info.space_type}")
    观测维度: (4,), 类型: SpaceType.BOX
    >>>
    >>> # 分析动作空间
    >>> act_info = analyze_space(env.action_space)
    >>> print(f"动作数量: {act_info.n}")
    动作数量: 2

    Notes
    -----
    对于组合空间 (Tuple, Dict)，flat_dim 计算为所有子空间维度之和。
    这在需要将组合观测展平为单一向量时非常有用。
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium 未安装，请执行: pip install gymnasium")

    if isinstance(space, spaces.Discrete):
        return SpaceInfo(
            space_type=SpaceType.DISCRETE,
            n=int(space.n),
            dtype=space.dtype,
            flat_dim=int(space.n)  # one-hot 编码的维度
        )

    elif isinstance(space, spaces.Box):
        is_bounded = bool(
            np.all(np.isfinite(space.low)) and
            np.all(np.isfinite(space.high))
        )
        return SpaceInfo(
            space_type=SpaceType.BOX,
            shape=space.shape,
            low=space.low.copy(),
            high=space.high.copy(),
            dtype=space.dtype,
            is_bounded=is_bounded,
            flat_dim=int(np.prod(space.shape))
        )

    elif isinstance(space, spaces.MultiDiscrete):
        return SpaceInfo(
            space_type=SpaceType.MULTI_DISCRETE,
            shape=(len(space.nvec),),
            dtype=space.dtype,
            flat_dim=int(np.sum(space.nvec))  # 所有离散空间的 one-hot 总和
        )

    elif isinstance(space, spaces.MultiBinary):
        n = space.n if isinstance(space.n, int) else int(np.prod(space.n))
        return SpaceInfo(
            space_type=SpaceType.MULTI_BINARY,
            shape=(n,) if isinstance(space.n, int) else tuple(space.n),
            dtype=space.dtype,
            flat_dim=n
        )

    elif isinstance(space, spaces.Tuple):
        total_dim = sum(analyze_space(s).flat_dim for s in space.spaces)
        return SpaceInfo(
            space_type=SpaceType.TUPLE,
            flat_dim=total_dim
        )

    elif isinstance(space, spaces.Dict):
        total_dim = sum(analyze_space(s).flat_dim for s in space.spaces.values())
        return SpaceInfo(
            space_type=SpaceType.DICT,
            flat_dim=total_dim
        )

    else:
        raise ValueError(f"不支持的空间类型: {type(space).__name__}")


def get_action_dim(space: "spaces.Space") -> int:
    """
    获取动作空间维度

    对于离散空间返回动作数量（用于 softmax 输出层），
    对于连续空间返回动作向量维度（用于高斯策略的均值输出）。

    Parameters
    ----------
    space : gymnasium.spaces.Space
        动作空间对象

    Returns
    -------
    int
        动作维度
        - Discrete(n): 返回 n
        - Box(shape): 返回 prod(shape)

    Example
    -------
    >>> env = gym.make("CartPole-v1")
    >>> print(get_action_dim(env.action_space))  # Discrete(2)
    2
    >>>
    >>> env = gym.make("Pendulum-v1")
    >>> print(get_action_dim(env.action_space))  # Box(1,)
    1
    """
    info = analyze_space(space)

    if info.space_type == SpaceType.DISCRETE:
        return info.n
    elif info.space_type == SpaceType.BOX:
        return info.flat_dim
    else:
        return info.flat_dim


def get_obs_dim(space: "spaces.Space") -> int:
    """
    获取观测空间维度

    返回观测向量的展平维度，用于确定网络输入层大小。
    对于图像观测，返回 C×H×W 的乘积。

    Parameters
    ----------
    space : gymnasium.spaces.Space
        观测空间对象

    Returns
    -------
    int
        观测维度（展平后）

    Example
    -------
    >>> env = gym.make("CartPole-v1")
    >>> print(get_obs_dim(env.observation_space))
    4
    """
    return analyze_space(space).flat_dim


def is_discrete_action_space(space: "spaces.Space") -> bool:
    """
    判断动作空间是否为离散空间

    Parameters
    ----------
    space : gymnasium.spaces.Space
        动作空间对象

    Returns
    -------
    bool
        True 表示离散动作空间，False 表示连续动作空间
    """
    info = analyze_space(space)
    return info.space_type in (SpaceType.DISCRETE, SpaceType.MULTI_DISCRETE, SpaceType.MULTI_BINARY)


def is_image_observation(space: "spaces.Space") -> bool:
    """
    判断观测空间是否为图像空间

    基于空间形状判断：三维且形状为 (H, W, C) 或 (C, H, W) 格式。

    Parameters
    ----------
    space : gymnasium.spaces.Space
        观测空间对象

    Returns
    -------
    bool
        True 表示图像观测空间
    """
    info = analyze_space(space)
    if info.space_type != SpaceType.BOX or info.shape is None:
        return False
    return len(info.shape) >= 3


# =============================================================================
#                           单元测试
# =============================================================================

def _run_tests() -> bool:
    """运行单元测试"""
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    print("\n" + "=" * 60)
    print("空间分析模块单元测试")
    print("=" * 60)

    all_passed = True

    # 测试 1: Discrete 空间
    print("\n[测试 1] Discrete 空间分析...")
    try:
        space = spaces.Discrete(5)
        info = analyze_space(space)
        assert info.space_type == SpaceType.DISCRETE
        assert info.n == 5
        assert info.flat_dim == 5
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 2: Box 空间
    print("\n[测试 2] Box 空间分析...")
    try:
        space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        info = analyze_space(space)
        assert info.space_type == SpaceType.BOX
        assert info.shape == (4,)
        assert info.flat_dim == 4
        assert info.is_bounded is True
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 3: MultiDiscrete 空间
    print("\n[测试 3] MultiDiscrete 空间分析...")
    try:
        space = spaces.MultiDiscrete([3, 4, 5])
        info = analyze_space(space)
        assert info.space_type == SpaceType.MULTI_DISCRETE
        assert info.flat_dim == 12  # 3 + 4 + 5
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 4: 实际环境
    print("\n[测试 4] CartPole 环境空间...")
    try:
        env = gym.make("CartPole-v1")

        obs_info = analyze_space(env.observation_space)
        assert obs_info.space_type == SpaceType.BOX
        assert obs_info.flat_dim == 4

        act_info = analyze_space(env.action_space)
        assert act_info.space_type == SpaceType.DISCRETE
        assert act_info.n == 2

        assert get_obs_dim(env.observation_space) == 4
        assert get_action_dim(env.action_space) == 2
        assert is_discrete_action_space(env.action_space) is True

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 5: 连续动作空间
    print("\n[测试 5] Pendulum 连续动作空间...")
    try:
        env = gym.make("Pendulum-v1")

        act_info = analyze_space(env.action_space)
        assert act_info.space_type == SpaceType.BOX
        assert act_info.flat_dim == 1
        assert is_discrete_action_space(env.action_space) is False

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
