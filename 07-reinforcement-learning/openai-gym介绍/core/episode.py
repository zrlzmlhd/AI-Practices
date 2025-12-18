#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gymnasium 回合执行工具

================================================================================
核心思想 (Core Idea)
================================================================================
强化学习的训练和评估都基于回合 (Episode) 的概念。一个回合是从环境重置开始，
到达终止状态或超时为止的完整交互序列。本模块提供标准化的回合执行和数据收集工具。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
一个回合 (Episode) 是一个轨迹 (Trajectory):

    $$\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_T, s_T)$$

其中:
- $s_t$: 时刻 t 的状态
- $a_t$: 时刻 t 的动作
- $r_{t+1}$: 执行动作 $a_t$ 后获得的奖励
- $T$: 回合终止时刻

回合总奖励 (Return):
    $$G = \sum_{t=0}^{T-1} r_{t+1}$$

折扣回报 (Discounted Return):
    $$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$$

================================================================================
问题背景 (Problem Statement)
================================================================================
回合执行是 RL 中最基础的操作，但实现中有许多细节需要处理:
1. 正确区分 terminated 和 truncated 信号
2. 高效收集轨迹数据用于训练
3. 记录统计信息用于监控
4. 支持确定性评估（固定种子）

本模块封装这些细节，提供简洁的高层接口。

================================================================================
算法总结 (Summary)
================================================================================
- StepResult: 封装单步交互结果
- EpisodeResult: 收集完整回合数据和统计
- run_episode: 执行单个回合的标准流程
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import gymnasium as gym
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    warnings.warn("gymnasium 未安装", ImportWarning)


@dataclass
class StepResult:
    """
    单步交互结果

    封装 env.step() 的返回值，提供更清晰的访问接口。
    Gymnasium 的 step 返回 5 个值，本类将其组织为语义化的属性。

    Attributes
    ----------
    observation : np.ndarray
        新状态观测
    reward : float
        即时奖励
    terminated : bool
        是否因任务完成而终止（如到达目标、任务失败）
    truncated : bool
        是否因超时等外部原因而截断
    info : dict
        附加信息字典，可能包含调试信息、统计数据等

    Properties
    ----------
    done : bool
        回合是否结束（terminated or truncated）

    Notes
    -----
    terminated 和 truncated 的区别很重要:
    - terminated=True: 环境达到自然终止状态，用于 bootstrap 时应使用 V(s')=0
    - truncated=True: 人为截断（如超时），bootstrap 时应使用 V(s') 估计值
    """
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

    @property
    def done(self) -> bool:
        """回合是否结束"""
        return self.terminated or self.truncated


@dataclass
class EpisodeResult:
    """
    完整回合结果

    记录一个完整回合的所有交互数据和统计信息。
    可用于训练数据收集、回放分析、可视化等。

    Attributes
    ----------
    observations : list of np.ndarray
        状态序列，长度为 T+1（包含初始状态和所有后续状态）
    actions : list
        动作序列，长度为 T
    rewards : list of float
        奖励序列，长度为 T
    total_reward : float
        累积奖励（回合总回报）
    length : int
        回合长度（执行的动作数）
    terminated : bool
        是否正常终止
    truncated : bool
        是否被截断
    info : dict
        最后一步的附加信息

    Methods
    -------
    append(obs, action, reward)
        添加一步数据
    get_discounted_returns(gamma)
        计算每步的折扣回报

    Example
    -------
    >>> result = EpisodeResult()
    >>> result.append(obs, action, 1.0)
    >>> print(f"当前长度: {result.length}, 累积奖励: {result.total_reward}")
    """
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    total_reward: float = 0.0
    length: int = 0
    terminated: bool = False
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def append(
        self,
        obs: np.ndarray,
        action: Any,
        reward: float
    ) -> None:
        """
        添加一步数据

        Parameters
        ----------
        obs : np.ndarray
            新观测（执行动作后的状态）
        action : Any
            执行的动作
        reward : float
            获得的奖励
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_reward += reward
        self.length += 1

    def get_discounted_returns(self, gamma: float = 0.99) -> np.ndarray:
        """
        计算每步的折扣回报

        从后向前计算:
            G_t = r_{t+1} + γ * G_{t+1}

        Parameters
        ----------
        gamma : float
            折扣因子，范围 [0, 1]

        Returns
        -------
        np.ndarray
            折扣回报数组，形状为 (length,)
        """
        returns = np.zeros(self.length)
        running_return = 0.0

        for t in reversed(range(self.length)):
            running_return = self.rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """
        将数据转换为 NumPy 数组

        Returns
        -------
        dict
            包含 'observations', 'actions', 'rewards', 'returns' 的字典
        """
        return {
            'observations': np.array(self.observations[:-1]),  # s_0 to s_{T-1}
            'next_observations': np.array(self.observations[1:]),  # s_1 to s_T
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'returns': self.get_discounted_returns(),
        }


def run_episode(
    env: "gym.Env",
    policy: Callable[[np.ndarray], Any],
    max_steps: Optional[int] = None,
    render: bool = False,
    seed: Optional[int] = None,
    collect_data: bool = True
) -> EpisodeResult:
    """
    执行一个完整回合

    使用给定策略在环境中执行交互，直到回合结束。
    这是 RL 中最基础的操作流程。

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium 环境实例
    policy : Callable[[np.ndarray], Any]
        策略函数，接收观测返回动作
        - 离散动作: 返回 int
        - 连续动作: 返回 np.ndarray
    max_steps : int, optional
        额外的最大步数限制，None 表示不限制
    render : bool, default=False
        是否在每步后渲染环境
    seed : int, optional
        环境重置的随机种子，用于可复现的评估
    collect_data : bool, default=True
        是否收集完整的轨迹数据，False 时只记录统计信息

    Returns
    -------
    EpisodeResult
        回合结果数据，包含轨迹和统计信息

    Example
    -------
    >>> env = gym.make("CartPole-v1")
    >>> # 定义简单策略
    >>> def angle_policy(obs):
    ...     return 1 if obs[2] > 0 else 0
    >>>
    >>> result = run_episode(env, angle_policy, seed=42)
    >>> print(f"回合奖励: {result.total_reward}, 长度: {result.length}")

    Notes
    -----
    - 策略函数应该是无状态的，或者在每次调用时自行管理状态
    - 如果不需要轨迹数据（如只做评估），设置 collect_data=False 可以节省内存
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium 未安装")

    result = EpisodeResult()

    # 重置环境
    obs, info = env.reset(seed=seed)
    if collect_data:
        result.observations.append(obs)

    step_count = 0
    max_steps = max_steps or float('inf')

    while step_count < max_steps:
        if render:
            env.render()

        # 获取动作
        action = policy(obs)

        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 记录数据
        if collect_data:
            result.append(next_obs, action, reward)
        else:
            result.total_reward += reward
            result.length += 1

        obs = next_obs
        step_count += 1

        # 检查终止
        if terminated or truncated:
            result.terminated = terminated
            result.truncated = truncated
            result.info = info
            break

    return result


def run_episodes(
    env: "gym.Env",
    policy: Callable[[np.ndarray], Any],
    n_episodes: int,
    seed: Optional[int] = None,
    collect_data: bool = False
) -> List[EpisodeResult]:
    """
    运行多个回合

    Parameters
    ----------
    env : gymnasium.Env
        环境实例
    policy : Callable
        策略函数
    n_episodes : int
        运行的回合数
    seed : int, optional
        基础随机种子，每个回合使用 seed+i
    collect_data : bool
        是否收集轨迹数据

    Returns
    -------
    list of EpisodeResult
        所有回合的结果
    """
    results = []

    for i in range(n_episodes):
        episode_seed = seed + i if seed is not None else None
        result = run_episode(
            env, policy,
            seed=episode_seed,
            collect_data=collect_data
        )
        results.append(result)

    return results


# =============================================================================
#                           单元测试
# =============================================================================

def _run_tests() -> bool:
    """运行单元测试"""
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    print("\n" + "=" * 60)
    print("回合执行模块单元测试")
    print("=" * 60)

    all_passed = True

    # 测试 1: EpisodeResult 数据类
    print("\n[测试 1] EpisodeResult 数据类...")
    try:
        result = EpisodeResult()

        for i in range(10):
            result.append(
                obs=np.zeros(4),
                action=i % 2,
                reward=1.0
            )

        assert result.length == 10
        assert result.total_reward == 10.0
        assert len(result.observations) == 10

        returns = result.get_discounted_returns(gamma=0.99)
        assert len(returns) == 10
        assert returns[0] > returns[-1]  # 早期回报更高

        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 2: run_episode 基本功能
    print("\n[测试 2] run_episode 基本功能...")
    try:
        env = gym.make("CartPole-v1")

        def random_policy(obs):
            return env.action_space.sample()

        result = run_episode(env, random_policy, seed=42)

        assert result.length > 0
        assert len(result.observations) == result.length + 1
        assert len(result.actions) == result.length
        assert len(result.rewards) == result.length

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 3: 可复现性
    print("\n[测试 3] 种子可复现性...")
    try:
        env = gym.make("CartPole-v1")

        def deterministic_policy(obs):
            return 1 if obs[2] > 0 else 0

        result1 = run_episode(env, deterministic_policy, seed=42)
        result2 = run_episode(env, deterministic_policy, seed=42)

        assert result1.total_reward == result2.total_reward
        assert result1.length == result2.length

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 4: 连续动作空间
    print("\n[测试 4] 连续动作空间...")
    try:
        env = gym.make("Pendulum-v1")

        def pd_policy(obs):
            cos_theta, sin_theta, theta_dot = obs
            theta = np.arctan2(sin_theta, cos_theta)
            torque = -10.0 * theta - 2.0 * theta_dot
            return np.clip([torque], -2.0, 2.0)

        result = run_episode(env, pd_policy, seed=42)

        assert result.length == 200  # Pendulum 固定 200 步
        assert len(result.actions) == 200
        assert all(isinstance(a, np.ndarray) for a in result.actions)

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 5: 多回合运行
    print("\n[测试 5] run_episodes...")
    try:
        env = gym.make("CartPole-v1")

        def policy(obs):
            return 1 if obs[2] > 0 else 0

        results = run_episodes(env, policy, n_episodes=5, seed=42)

        assert len(results) == 5
        assert all(r.length > 0 for r in results)

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
