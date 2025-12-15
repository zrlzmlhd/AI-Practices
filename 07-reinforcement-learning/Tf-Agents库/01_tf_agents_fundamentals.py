#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF-Agents 核心组件与基础概念

本模块系统性地介绍 TF-Agents 库的核心抽象，包括环境、智能体、策略、
经验回放等关键组件。通过理解这些基础概念，为后续深入学习各类算法奠定基础。

================================================================================
核心思想 (Core Idea)
================================================================================
TF-Agents 将强化学习问题分解为独立、可组合的模块：
- Environment: 定义状态转移和奖励函数
- Agent: 封装学习算法和策略更新
- Policy: 将观测映射到动作
- Replay Buffer: 存储和采样经验
- Driver: 自动化数据收集

这种模块化设计使得算法开发、调试和实验更加高效。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
强化学习的核心是马尔可夫决策过程 (MDP)：

MDP 定义为五元组 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$

其中：
- $\mathcal{S}$: 状态空间 (State Space)
- $\mathcal{A}$: 动作空间 (Action Space)
- $P(s'|s,a)$: 状态转移概率 (Transition Probability)
- $R(s,a,s')$: 奖励函数 (Reward Function)
- $\gamma \in [0,1]$: 折扣因子 (Discount Factor)

目标是找到最优策略 $\pi^*$ 最大化累积折扣奖励：

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1})\right]$$

================================================================================
TF-Agents 核心抽象
================================================================================

1. TimeStep: 封装环境交互的单步信息
   - step_type: FIRST (初始), MID (中间), LAST (终止)
   - reward: 即时奖励
   - discount: 折扣因子
   - observation: 当前观测

2. Trajectory: 描述完整的经验轨迹
   - observation, action, policy_info, next_observation
   - reward, discount, step_type, next_step_type

3. Spec: 定义数据规范（类型、形状、边界）
   - ArraySpec: NumPy 数组规范
   - TensorSpec: TensorFlow 张量规范
   - BoundedSpec: 带边界的规范

================================================================================
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any

# TF-Agents 核心导入
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_py_policy
from tf_agents.policies import random_tf_policy


class TFAgentsConceptDemo:
    """
    TF-Agents 核心概念演示类

    通过具体示例展示 TF-Agents 的核心抽象和使用方法。

    核心思想 (Core Idea):
        将强化学习的各个组件封装为独立模块，每个模块有清晰的接口定义，
        便于组合和扩展。

    Attributes:
        env_name: Gym 环境名称
        py_env: Python 环境实例
        tf_env: TensorFlow 环境实例（批处理支持）
    """

    def __init__(self, env_name: str = "CartPole-v1"):
        """
        初始化演示实例

        Args:
            env_name: OpenAI Gym 环境标识符
        """
        self.env_name = env_name
        self.py_env: Optional[py_environment.PyEnvironment] = None
        self.tf_env: Optional[tf_py_environment.TFPyEnvironment] = None

    def demonstrate_specs(self) -> Dict[str, Any]:
        """
        演示 TF-Agents 的 Spec 系统

        Spec 是 TF-Agents 中描述数据规范的核心抽象，定义了：
        - 数据类型 (dtype)
        - 数据形状 (shape)
        - 数值边界 (minimum, maximum) - 仅 BoundedSpec

        数学表示:
            对于有界规范 $\text{Spec}(x)$，约束 $x \in [\text{min}, \text{max}]$

        Returns:
            包含各类规范示例的字典
        """
        # ArraySpec: 基础数组规范
        observation_spec = array_spec.ArraySpec(
            shape=(4,),      # 4维观测向量
            dtype=np.float32,
            name="observation"
        )

        # BoundedArraySpec: 带边界的数组规范
        # 适用于动作空间（离散或连续）
        discrete_action_spec = array_spec.BoundedArraySpec(
            shape=(),           # 标量
            dtype=np.int32,
            minimum=0,
            maximum=1,          # 二分类动作
            name="discrete_action"
        )

        continuous_action_spec = array_spec.BoundedArraySpec(
            shape=(2,),         # 2维连续动作
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name="continuous_action"
        )

        # TensorSpec: TensorFlow 张量规范
        tensor_observation_spec = tensor_spec.TensorSpec(
            shape=(4,),
            dtype=tf.float32,
            name="tensor_observation"
        )

        specs = {
            "observation_spec": observation_spec,
            "discrete_action_spec": discrete_action_spec,
            "continuous_action_spec": continuous_action_spec,
            "tensor_spec": tensor_observation_spec
        }

        print("=" * 60)
        print("Spec 系统演示")
        print("=" * 60)
        for name, spec in specs.items():
            print(f"\n{name}:")
            print(f"  Shape: {spec.shape}")
            print(f"  Dtype: {spec.dtype}")
            if hasattr(spec, 'minimum'):
                print(f"  Minimum: {spec.minimum}")
                print(f"  Maximum: {spec.maximum}")

        return specs

    def demonstrate_timestep(self) -> ts.TimeStep:
        """
        演示 TimeStep 数据结构

        TimeStep 封装了环境交互的单步信息，是 TF-Agents 中最基础的数据单元。

        数学原理:
            TimeStep 对应 MDP 中的状态转移：$(s_t, a_t) \to (r_t, s_{t+1})$

            其中 step_type 标识轨迹边界：
            - FIRST: $t = 0$ (回合开始)
            - MID: $0 < t < T$ (回合中间)
            - LAST: $t = T$ (回合结束)

        Returns:
            示例 TimeStep 对象
        """
        # 创建不同类型的 TimeStep
        first_step = ts.restart(observation=np.array([0.0, 0.1, 0.2, 0.3]))

        mid_step = ts.transition(
            observation=np.array([0.1, 0.2, 0.3, 0.4]),
            reward=1.0,
            discount=0.99
        )

        terminal_step = ts.termination(
            observation=np.array([0.5, 0.6, 0.7, 0.8]),
            reward=0.0
        )

        print("\n" + "=" * 60)
        print("TimeStep 数据结构演示")
        print("=" * 60)

        for name, step in [("FIRST", first_step),
                           ("MID", mid_step),
                           ("LAST", terminal_step)]:
            print(f"\n{name} TimeStep:")
            print(f"  step_type: {step.step_type}")
            print(f"  reward: {step.reward}")
            print(f"  discount: {step.discount}")
            print(f"  observation shape: {step.observation.shape}")

        return mid_step

    def demonstrate_trajectory(self) -> trajectory.Trajectory:
        """
        演示 Trajectory 数据结构

        Trajectory 表示一个完整的经验片段，用于存储到 Replay Buffer 和训练。

        数学原理:
            轨迹定义为状态-动作序列：
            $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)$

            单步轨迹包含：$(s_t, a_t, r_t, s_{t+1}, \gamma_t)$

        Returns:
            示例 Trajectory 对象
        """
        # 创建单步轨迹
        obs = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        action = np.array([1], dtype=np.int32)
        policy_info = ()
        next_obs = np.array([[1.1, 2.1, 3.1, 4.1]], dtype=np.float32)
        reward = np.array([1.0], dtype=np.float32)
        discount = np.array([0.99], dtype=np.float32)

        traj = trajectory.Trajectory(
            step_type=np.array([ts.StepType.MID]),
            observation=obs,
            action=action,
            policy_info=policy_info,
            next_step_type=np.array([ts.StepType.MID]),
            reward=reward,
            discount=discount
        )

        print("\n" + "=" * 60)
        print("Trajectory 数据结构演示")
        print("=" * 60)
        print(f"  observation: {traj.observation}")
        print(f"  action: {traj.action}")
        print(f"  reward: {traj.reward}")
        print(f"  discount: {traj.discount}")
        print(f"  step_type: {traj.step_type}")

        return traj

    def demonstrate_environment(self) -> Tuple[py_environment.PyEnvironment,
                                                tf_py_environment.TFPyEnvironment]:
        """
        演示环境封装和使用

        TF-Agents 提供两种环境接口：
        1. PyEnvironment: Python 原生接口，便于调试
        2. TFPyEnvironment: TensorFlow 接口，支持批处理和 GPU 加速

        核心思想:
            通过 tf_py_environment.TFPyEnvironment 包装，可以将任何
            PyEnvironment 转换为支持批处理的 TensorFlow 环境，
            实现高效的并行数据收集。

        Returns:
            (Python环境, TensorFlow环境) 元组
        """
        # 加载 Gym 环境
        self.py_env = suite_gym.load(self.env_name)

        # 包装为 TensorFlow 环境
        self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)

        print("\n" + "=" * 60)
        print(f"环境演示: {self.env_name}")
        print("=" * 60)
        print(f"\n观测空间 (Observation Spec):")
        print(f"  {self.tf_env.observation_spec()}")
        print(f"\n动作空间 (Action Spec):")
        print(f"  {self.tf_env.action_spec()}")
        print(f"\n时间步规范 (Time Step Spec):")
        print(f"  {self.tf_env.time_step_spec()}")

        # 演示环境交互
        print("\n环境交互演示:")
        time_step = self.tf_env.reset()
        print(f"  Reset observation: {time_step.observation.numpy()}")

        # 随机动作
        action = tf.constant([0])  # CartPole 动作: 0=左, 1=右
        next_time_step = self.tf_env.step(action)
        print(f"  After action 0: reward={next_time_step.reward.numpy()}")

        return self.py_env, self.tf_env

    def demonstrate_policy(self) -> None:
        """
        演示策略 (Policy) 组件

        策略是强化学习的核心，定义了智能体的行为方式。

        数学原理:
            策略 $\pi: \mathcal{S} \to \mathcal{P}(\mathcal{A})$
            将状态映射到动作分布。

            确定性策略: $a = \pi(s)$
            随机策略: $a \sim \pi(\cdot|s)$

        TF-Agents 策略类型:
            - RandomPolicy: 均匀随机策略，用于初始探索
            - GreedyPolicy: 贪婪策略，用于评估
            - EpsilonGreedyPolicy: ε-贪婪策略，平衡探索-利用
        """
        if self.tf_env is None:
            self.demonstrate_environment()

        # 创建随机策略
        random_policy = random_tf_policy.RandomTFPolicy(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec()
        )

        print("\n" + "=" * 60)
        print("策略 (Policy) 演示")
        print("=" * 60)

        # 使用策略选择动作
        time_step = self.tf_env.reset()
        policy_step = random_policy.action(time_step)

        print(f"\n随机策略动作: {policy_step.action.numpy()}")
        print(f"策略信息: {policy_step.info}")

        # 运行一个回合
        total_reward = 0.0
        steps = 0
        time_step = self.tf_env.reset()

        while not time_step.is_last():
            action_step = random_policy.action(time_step)
            time_step = self.tf_env.step(action_step.action)
            total_reward += time_step.reward.numpy()[0]
            steps += 1

        print(f"\n随机策略回合统计:")
        print(f"  总步数: {steps}")
        print(f"  总奖励: {total_reward:.2f}")


def compute_returns(
    rewards: np.ndarray,
    discount: float = 0.99
) -> np.ndarray:
    """
    计算折扣累积回报

    核心思想 (Core Idea):
        回报 (Return) 是从某时刻开始的未来奖励的折扣和，
        是价值函数估计的目标。

    数学原理 (Mathematical Theory):
        回报定义为：
        $$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} = r_{t+1} + \gamma G_{t+1}$$

        其中：
        - $G_t$: 时刻 $t$ 的回报
        - $\gamma$: 折扣因子，$\gamma \in [0,1]$
        - $r_t$: 时刻 $t$ 的即时奖励

    算法实现:
        使用后向递推计算，时间复杂度 $O(T)$

    Args:
        rewards: 奖励序列，形状 (T,)
        discount: 折扣因子 γ

    Returns:
        回报序列，形状 (T,)

    复杂度 (Complexity):
        时间: O(T)，其中 T 为序列长度
        空间: O(T)，存储回报序列

    Example:
        >>> rewards = np.array([1.0, 1.0, 1.0, 0.0])
        >>> returns = compute_returns(rewards, discount=0.99)
        >>> print(returns)  # [2.9701, 1.99, 1.0, 0.0]
    """
    returns = np.zeros_like(rewards)
    running_return = 0.0

    # 后向递推: G_t = r_t + γ * G_{t+1}
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + discount * running_return
        returns[t] = running_return

    return returns


def compute_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    discount: float = 0.99,
    gae_lambda: float = 0.95
) -> np.ndarray:
    """
    计算广义优势估计 (Generalized Advantage Estimation, GAE)

    核心思想 (Core Idea):
        GAE 通过指数加权平均多步 TD 误差，在偏差和方差之间取得平衡，
        是现代策略梯度方法的标准优势估计器。

    数学原理 (Mathematical Theory):
        TD 误差 (Temporal Difference Error):
        $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

        GAE 定义：
        $$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

        展开形式：
        $$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + ...$$

        参数解释：
        - $\lambda = 0$: 单步 TD，低方差高偏差
        - $\lambda = 1$: Monte Carlo，高方差低偏差
        - $\lambda \in (0,1)$: 平衡偏差-方差

    问题背景 (Problem Statement):
        策略梯度方法需要估计优势函数 $A(s,a) = Q(s,a) - V(s)$，
        直接使用 Monte Carlo 回报方差过大，单步 TD 偏差过大，
        GAE 提供了可调节的折中方案。

    算法对比 (Comparison):
        - vs Monte Carlo: 更低方差，更稳定的训练
        - vs 1-step TD: 更低偏差，更准确的估计
        - vs n-step TD: 更灵活，通过 λ 自适应调节

    Args:
        rewards: 奖励序列 (T,)
        values: 价值估计 (T+1,)，包含终止状态值
        discount: 折扣因子 γ
        gae_lambda: GAE 参数 λ

    Returns:
        优势估计序列 (T,)

    复杂度 (Complexity):
        时间: O(T)
        空间: O(T)

    Reference:
        Schulman, J. et al. (2016). High-Dimensional Continuous Control
        Using Generalized Advantage Estimation. ICLR.
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0

    # 后向递推计算 GAE
    for t in reversed(range(T)):
        # TD 误差: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + discount * values[t + 1] - values[t]
        # GAE 递推: A_t = δ_t + (γλ)A_{t+1}
        gae = delta + discount * gae_lambda * gae
        advantages[t] = gae

    return advantages


class EnvironmentStats:
    """
    环境统计信息收集器

    用于收集和分析智能体与环境交互的统计数据，
    支持实时监控训练进度和性能评估。

    核心思想 (Core Idea):
        通过追踪回合奖励、长度等指标，评估智能体性能并诊断训练问题。

    Attributes:
        episode_rewards: 每回合总奖励列表
        episode_lengths: 每回合步数列表
        _current_reward: 当前回合累积奖励
        _current_length: 当前回合步数
    """

    def __init__(self):
        """初始化统计收集器"""
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._current_reward: float = 0.0
        self._current_length: int = 0

    def step(self, reward: float, done: bool) -> None:
        """
        记录单步交互

        Args:
            reward: 当前步奖励
            done: 回合是否结束
        """
        self._current_reward += reward
        self._current_length += 1

        if done:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self._current_reward = 0.0
            self._current_length = 0

    def get_stats(self, last_n: int = 100) -> Dict[str, float]:
        """
        获取统计摘要

        Args:
            last_n: 统计最近 n 个回合

        Returns:
            包含平均奖励、平均长度等的字典
        """
        if not self.episode_rewards:
            return {"mean_reward": 0.0, "mean_length": 0.0}

        recent_rewards = self.episode_rewards[-last_n:]
        recent_lengths = self.episode_lengths[-last_n:]

        return {
            "mean_reward": np.mean(recent_rewards),
            "std_reward": np.std(recent_rewards),
            "min_reward": np.min(recent_rewards),
            "max_reward": np.max(recent_rewards),
            "mean_length": np.mean(recent_lengths),
            "num_episodes": len(self.episode_rewards)
        }


def run_demo():
    """
    运行 TF-Agents 核心概念演示
    """
    print("=" * 70)
    print("TF-Agents 核心组件演示")
    print("=" * 70)

    demo = TFAgentsConceptDemo(env_name="CartPole-v1")

    # 演示各组件
    demo.demonstrate_specs()
    demo.demonstrate_timestep()
    demo.demonstrate_trajectory()
    demo.demonstrate_environment()
    demo.demonstrate_policy()

    # 演示计算函数
    print("\n" + "=" * 60)
    print("辅助函数演示")
    print("=" * 60)

    # 回报计算
    rewards = np.array([1.0, 1.0, 1.0, 0.0, -1.0])
    returns = compute_returns(rewards, discount=0.99)
    print(f"\n奖励序列: {rewards}")
    print(f"折扣回报: {returns}")

    # GAE 计算
    values = np.array([0.5, 0.6, 0.7, 0.4, 0.3, 0.0])  # T+1 个值
    advantages = compute_advantages(rewards, values)
    print(f"价值估计: {values}")
    print(f"GAE 优势: {advantages}")

    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)


if __name__ == "__main__":
    # 设置 TensorFlow 日志级别
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 运行演示
    run_demo()
