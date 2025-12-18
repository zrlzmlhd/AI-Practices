#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gymnasium 策略评估工具

================================================================================
核心思想 (Core Idea)
================================================================================
策略评估是衡量强化学习算法性能的标准方法。通过在多个回合中运行策略，收集统计数据
（平均奖励、标准差、成功率等），可以客观比较不同策略或算法的效果。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
策略的期望回报定义为:

    $$J(\pi) = \mathbb{E}_{\tau \sim \pi}[G(\tau)] = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T-1} r_{t+1}\right]$$

实际评估中使用蒙特卡洛估计:

    $$\hat{J}(\pi) = \frac{1}{N}\sum_{i=1}^{N} G(\tau_i)$$

其中 $N$ 是评估回合数。

置信区间估计（假设正态分布）:

    $$\hat{J}(\pi) \pm z_{\alpha/2} \cdot \frac{\hat{\sigma}}{\sqrt{N}}$$

对于 95% 置信区间，$z_{0.025} \approx 1.96$。

================================================================================
问题背景 (Problem Statement)
================================================================================
策略评估需要考虑:
1. **随机性**: 环境和策略都可能是随机的，需要多回合采样
2. **统计显著性**: 回合数太少会导致估计不可靠
3. **公平比较**: 不同策略应使用相同的种子序列
4. **效率**: 评估过程应该高效，不浪费计算资源

================================================================================
算法总结 (Summary)
================================================================================
本模块提供:
- evaluate_policy: 函数式评估接口
- PolicyEvaluator: 面向对象的评估器，支持多策略比较
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .episode import run_episode, EpisodeResult

try:
    import gymnasium as gym
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    warnings.warn("gymnasium 未安装", ImportWarning)


@dataclass
class EvaluationResult:
    """
    策略评估结果

    Attributes
    ----------
    mean_reward : float
        平均回合奖励
    std_reward : float
        回合奖励标准差
    min_reward : float
        最小回合奖励
    max_reward : float
        最大回合奖励
    mean_length : float
        平均回合长度
    std_length : float
        回合长度标准差
    n_episodes : int
        评估回合数
    success_rate : float
        成功率（terminated=True 的比例）
    total_time : float
        评估总时间（秒）
    episode_rewards : list of float
        每回合的奖励列表
    episode_lengths : list of int
        每回合的长度列表
    """
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    mean_length: float = 0.0
    std_length: float = 0.0
    n_episodes: int = 0
    success_rate: float = 0.0
    total_time: float = 0.0
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    def summary(self) -> str:
        """生成评估摘要字符串"""
        return (
            f"评估结果 ({self.n_episodes} 回合):\n"
            f"  奖励: {self.mean_reward:.2f} ± {self.std_reward:.2f}\n"
            f"  范围: [{self.min_reward:.1f}, {self.max_reward:.1f}]\n"
            f"  长度: {self.mean_length:.1f} ± {self.std_length:.1f}\n"
            f"  成功率: {self.success_rate*100:.1f}%\n"
            f"  用时: {self.total_time:.2f}s"
        )

    @property
    def confidence_interval_95(self) -> tuple:
        """95% 置信区间"""
        if self.n_episodes < 2:
            return (self.mean_reward, self.mean_reward)
        se = self.std_reward / np.sqrt(self.n_episodes)
        return (self.mean_reward - 1.96 * se, self.mean_reward + 1.96 * se)


def evaluate_policy(
    env: "gym.Env",
    policy: Callable[[np.ndarray], Any],
    n_episodes: int = 10,
    seed: Optional[int] = None,
    verbose: bool = True,
    deterministic: bool = True
) -> EvaluationResult:
    """
    评估策略性能

    在多个回合中运行策略，统计性能指标。

    Parameters
    ----------
    env : gymnasium.Env
        评估环境
    policy : Callable[[np.ndarray], Any]
        待评估策略，输入观测返回动作
    n_episodes : int, default=10
        评估回合数
    seed : int, optional
        基础随机种子，每回合使用 seed+i
    verbose : bool, default=True
        是否打印评估进度
    deterministic : bool, default=True
        是否使用确定性评估（固定种子序列）

    Returns
    -------
    EvaluationResult
        包含统计信息的评估结果

    Example
    -------
    >>> env = gym.make("CartPole-v1")
    >>> def simple_policy(obs):
    ...     return 1 if obs[2] > 0 else 0
    >>>
    >>> result = evaluate_policy(env, simple_policy, n_episodes=100)
    >>> print(result.summary())
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium 未安装")

    rewards = []
    lengths = []
    successes = 0

    start_time = time.time()

    for i in range(n_episodes):
        episode_seed = seed + i if (seed is not None and deterministic) else None

        result = run_episode(
            env, policy,
            seed=episode_seed,
            collect_data=False  # 评估时不需要收集轨迹
        )

        rewards.append(result.total_reward)
        lengths.append(result.length)

        if result.terminated:
            successes += 1

        if verbose:
            print(f"回合 {i+1}/{n_episodes}: "
                  f"奖励={result.total_reward:.2f}, "
                  f"长度={result.length}")

    total_time = time.time() - start_time

    eval_result = EvaluationResult(
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        min_reward=float(np.min(rewards)),
        max_reward=float(np.max(rewards)),
        mean_length=float(np.mean(lengths)),
        std_length=float(np.std(lengths)),
        n_episodes=n_episodes,
        success_rate=successes / n_episodes,
        total_time=total_time,
        episode_rewards=rewards,
        episode_lengths=lengths
    )

    if verbose:
        print("\n" + eval_result.summary())

    return eval_result


class PolicyEvaluator:
    """
    策略评估器

    面向对象的评估接口，支持多策略比较和结果缓存。

    Attributes
    ----------
    env_id : str
        环境 ID
    n_episodes : int
        默认评估回合数
    seed : int
        基础随机种子
    results : dict
        策略名称到评估结果的映射

    Example
    -------
    >>> evaluator = PolicyEvaluator("CartPole-v1", n_episodes=50)
    >>> evaluator.add_policy("random", lambda obs: np.random.randint(2))
    >>> evaluator.add_policy("angle", lambda obs: 1 if obs[2] > 0 else 0)
    >>> evaluator.run_all()
    >>> evaluator.print_comparison()
    """

    def __init__(
        self,
        env_id: str,
        n_episodes: int = 10,
        seed: int = 42,
        **env_kwargs
    ):
        """
        初始化评估器

        Parameters
        ----------
        env_id : str
            Gymnasium 环境 ID
        n_episodes : int
            默认评估回合数
        seed : int
            基础随机种子
        **env_kwargs
            传递给 gym.make 的参数
        """
        self.env_id = env_id
        self.n_episodes = n_episodes
        self.seed = seed
        self.env_kwargs = env_kwargs

        self.policies: Dict[str, Callable] = {}
        self.results: Dict[str, EvaluationResult] = {}

    def add_policy(self, name: str, policy: Callable[[np.ndarray], Any]) -> None:
        """
        添加待评估策略

        Parameters
        ----------
        name : str
            策略名称（用于显示和比较）
        policy : Callable
            策略函数
        """
        self.policies[name] = policy

    def evaluate(
        self,
        name: str,
        n_episodes: Optional[int] = None,
        verbose: bool = False
    ) -> EvaluationResult:
        """
        评估单个策略

        Parameters
        ----------
        name : str
            策略名称
        n_episodes : int, optional
            评估回合数，默认使用初始化时的值
        verbose : bool
            是否打印进度

        Returns
        -------
        EvaluationResult
            评估结果
        """
        if name not in self.policies:
            raise ValueError(f"未找到策略: {name}")

        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        env = gym.make(self.env_id, **self.env_kwargs)
        n_eps = n_episodes or self.n_episodes

        result = evaluate_policy(
            env,
            self.policies[name],
            n_episodes=n_eps,
            seed=self.seed,
            verbose=verbose
        )

        env.close()
        self.results[name] = result
        return result

    def run_all(self, verbose: bool = True) -> Dict[str, EvaluationResult]:
        """
        评估所有策略

        Parameters
        ----------
        verbose : bool
            是否打印进度

        Returns
        -------
        dict
            策略名称到评估结果的映射
        """
        for name in self.policies:
            if verbose:
                print(f"\n{'='*50}")
                print(f"评估策略: {name}")
                print('='*50)
            self.evaluate(name, verbose=verbose)

        return self.results

    def print_comparison(self) -> None:
        """打印策略比较表"""
        if not self.results:
            print("尚未评估任何策略")
            return

        print("\n" + "=" * 70)
        print("策略比较")
        print("=" * 70)
        print(f"{'策略':<20} {'平均奖励':>12} {'标准差':>10} {'成功率':>10}")
        print("-" * 60)

        # 按平均奖励排序
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].mean_reward,
            reverse=True
        )

        for name, result in sorted_results:
            print(f"{name:<20} {result.mean_reward:>12.2f} "
                  f"{result.std_reward:>10.2f} "
                  f"{result.success_rate*100:>9.1f}%")

    def get_best_policy(self) -> Optional[str]:
        """
        获取表现最好的策略名称

        Returns
        -------
        str or None
            最佳策略名称
        """
        if not self.results:
            return None

        return max(self.results.items(), key=lambda x: x[1].mean_reward)[0]


# =============================================================================
#                           单元测试
# =============================================================================

def _run_tests() -> bool:
    """运行单元测试"""
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    print("\n" + "=" * 60)
    print("策略评估模块单元测试")
    print("=" * 60)

    all_passed = True

    # 测试 1: 基本评估
    print("\n[测试 1] 基本评估功能...")
    try:
        env = gym.make("CartPole-v1")

        def policy(obs):
            return 1 if obs[2] > 0 else 0

        result = evaluate_policy(env, policy, n_episodes=5, seed=42, verbose=False)

        assert result.n_episodes == 5
        assert len(result.episode_rewards) == 5
        assert result.mean_reward > 0
        assert 0 <= result.success_rate <= 1

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 2: PolicyEvaluator
    print("\n[测试 2] PolicyEvaluator...")
    try:
        evaluator = PolicyEvaluator("CartPole-v1", n_episodes=5, seed=42)

        evaluator.add_policy("random", lambda obs: np.random.randint(2))
        evaluator.add_policy("angle", lambda obs: 1 if obs[2] > 0 else 0)

        results = evaluator.run_all(verbose=False)

        assert len(results) == 2
        assert "random" in results
        assert "angle" in results

        # angle 策略应该比随机策略好
        assert results["angle"].mean_reward > results["random"].mean_reward

        best = evaluator.get_best_policy()
        assert best == "angle"

        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 3: 置信区间
    print("\n[测试 3] 置信区间计算...")
    try:
        env = gym.make("CartPole-v1")

        def policy(obs):
            return 1 if obs[2] > 0 else 0

        result = evaluate_policy(env, policy, n_episodes=20, seed=42, verbose=False)
        ci_low, ci_high = result.confidence_interval_95

        assert ci_low < result.mean_reward < ci_high
        assert ci_high - ci_low > 0  # 区间宽度为正

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 4: 可复现性
    print("\n[测试 4] 评估可复现性...")
    try:
        env = gym.make("CartPole-v1")

        def policy(obs):
            return 1 if obs[2] > 0 else 0

        result1 = evaluate_policy(env, policy, n_episodes=10, seed=42, verbose=False)
        result2 = evaluate_policy(env, policy, n_episodes=10, seed=42, verbose=False)

        assert result1.mean_reward == result2.mean_reward
        assert result1.episode_rewards == result2.episode_rewards

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
