#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Q-Network (DQN) 智能体实现 - CartPole 环境

本模块实现了完整的 DQN 智能体，包括核心算法、经验回放、目标网络等组件。
通过 CartPole 环境验证算法正确性。

================================================================================
核心思想 (Core Idea)
================================================================================
DQN 使用深度神经网络逼近 Q 函数，结合经验回放和目标网络两项关键技术，
实现了端到端的离散动作空间强化学习。其核心创新在于：
1. 经验回放打破样本相关性
2. 目标网络稳定训练过程

================================================================================
数学原理 (Mathematical Theory)
================================================================================
Q-Learning 更新规则：
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

DQN 损失函数：
$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}
\left[(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a))^2\right]$$

其中：
- $\theta$: 在线网络参数
- $\theta^-$: 目标网络参数 (周期性更新)
- $\mathcal{D}$: 经验回放缓冲区
- $\gamma$: 折扣因子

目标网络软更新（可选）：
$$\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$$

================================================================================
问题背景 (Problem Statement)
================================================================================
传统 Q-Learning 使用表格方法存储 Q 值，无法处理高维或连续状态空间。
神经网络函数逼近面临两个核心挑战：
1. 样本相关性：连续采样的数据高度相关，违反 i.i.d. 假设
2. 非稳态目标：目标值随网络更新而变化，导致训练不稳定

DQN 通过经验回放和目标网络分别解决这两个问题。

================================================================================
算法对比 (Comparison)
================================================================================
| 方法          | 优点                | 缺点                    |
|--------------|---------------------|------------------------|
| Tabular Q    | 简单直观、收敛保证    | 无法处理高维状态空间      |
| DQN          | 处理高维状态、样本高效 | 仅支持离散动作、过估计问题 |
| Double DQN   | 缓解过估计           | 计算量稍增              |
| Dueling DQN  | 更好的状态值估计      | 网络结构更复杂           |
| Rainbow      | 集成多项改进          | 实现复杂、计算开销大      |

================================================================================
复杂度 (Complexity)
================================================================================
时间复杂度：
- 单次更新: O(B × D × A)，B=批大小，D=网络深度，A=动作数
- 数据收集: O(1) per step

空间复杂度：
- 回放缓冲区: O(N × S)，N=容量，S=状态维度
- 网络参数: O(L × H²)，L=层数，H=隐藏维度

================================================================================
算法总结 (Summary)
================================================================================
DQN 智能体执行以下循环：
1. 使用 ε-贪婪策略与环境交互，收集经验 (s, a, r, s')
2. 将经验存入回放缓冲区
3. 从缓冲区采样批量数据
4. 计算 TD 目标: y = r + γ max Q_target(s', a')
5. 最小化 MSE 损失更新在线网络
6. 周期性更新目标网络

================================================================================
Reference:
    Mnih, V. et al. (2015). Human-level control through deep reinforcement
    learning. Nature, 518(7540), 529-533.
================================================================================
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import tensorflow as tf

# TF-Agents imports
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.networks import encoding_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DQNConfig:
    """
    DQN 超参数配置

    核心思想 (Core Idea):
        将所有超参数集中管理，便于实验复现和参数调优。
        使用 dataclass 提供类型提示和默认值。

    数学原理 (Mathematical Theory):
        各参数对应算法中的数学符号：
        - learning_rate: 对应梯度下降步长 $\alpha$
        - discount_factor: 对应折扣因子 $\gamma$
        - epsilon_*: 对应 ε-贪婪探索参数

    Attributes:
        env_name: Gym 环境标识符
        num_iterations: 总训练迭代次数
        collect_steps_per_iteration: 每次迭代收集的交互步数
        replay_buffer_capacity: 经验回放缓冲区容量
        batch_size: 训练批大小
        learning_rate: 学习率
        discount_factor: 折扣因子 γ
        target_update_period: 目标网络更新周期（硬更新）
        target_update_tau: 目标网络软更新系数（1.0=硬更新）
        epsilon_greedy: ε-贪婪探索初始值
        fc_layer_params: Q网络全连接层维度
        log_interval: 日志打印间隔
        eval_interval: 评估间隔
        num_eval_episodes: 评估时运行的回合数
    """
    env_name: str = "CartPole-v1"
    num_iterations: int = 20000
    collect_steps_per_iteration: int = 1
    replay_buffer_capacity: int = 100000
    batch_size: int = 64
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    target_update_period: int = 200
    target_update_tau: float = 1.0
    epsilon_greedy: float = 0.1
    fc_layer_params: tuple = (100, 50)
    log_interval: int = 200
    eval_interval: int = 1000
    num_eval_episodes: int = 10


class DQNTrainer:
    """
    DQN 智能体训练器

    封装了 DQN 算法的完整训练流程，包括：
    - 环境创建和管理
    - Q 网络构建
    - 智能体实例化
    - 经验回放缓冲区
    - 训练循环
    - 评估和日志

    核心思想 (Core Idea):
        将 DQN 训练流程模块化封装，每个组件职责清晰，
        便于调试、扩展和复用。

    数学原理 (Mathematical Theory):
        训练目标是最小化 Bellman 误差：
        $$\min_\theta \mathbb{E}[(Q_\theta(s,a) - y)^2]$$

        其中 TD 目标 $y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$

    Example:
        >>> config = DQNConfig(num_iterations=1000)
        >>> trainer = DQNTrainer(config)
        >>> trainer.initialize()
        >>> returns = trainer.train()
    """

    def __init__(self, config: DQNConfig):
        """
        初始化训练器

        Args:
            config: DQN 超参数配置
        """
        self.config = config
        self.train_env: Optional[tf_py_environment.TFPyEnvironment] = None
        self.eval_env: Optional[tf_py_environment.TFPyEnvironment] = None
        self.agent: Optional[dqn_agent.DqnAgent] = None
        self.replay_buffer: Optional[tf_uniform_replay_buffer.TFUniformReplayBuffer] = None
        self.train_step_counter: Optional[tf.Variable] = None
        self.returns_history: list[float] = []

    def _create_environments(self) -> None:
        """
        创建训练和评估环境

        数学原理:
            训练环境用于数据收集，评估环境用于策略评估。
            两个环境独立，避免评估影响训练状态。
        """
        train_py_env = suite_gym.load(self.config.env_name)
        eval_py_env = suite_gym.load(self.config.env_name)

        self.train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        logger.info(f"环境创建完成: {self.config.env_name}")
        logger.info(f"  观测空间: {self.train_env.observation_spec()}")
        logger.info(f"  动作空间: {self.train_env.action_spec()}")

    def _build_q_network(self) -> q_network.QNetwork:
        """
        构建 Q 网络

        核心思想 (Core Idea):
            Q 网络将状态映射到各动作的 Q 值，使用多层感知机实现。

        数学原理 (Mathematical Theory):
            Q 网络近似 Q 函数：
            $$Q_\theta: \mathcal{S} \to \mathbb{R}^{|\mathcal{A}|}$$

            输出层维度等于动作数，每个输出对应一个动作的 Q 值。

        Returns:
            构建好的 Q 网络实例
        """
        q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=self.config.fc_layer_params,
            activation_fn=tf.keras.activations.relu
        )

        logger.info(f"Q 网络结构: {self.config.fc_layer_params}")
        return q_net

    def _create_agent(self, q_net: q_network.QNetwork) -> None:
        """
        创建 DQN 智能体

        核心思想 (Core Idea):
            DQN Agent 封装了 Q 学习算法，包括：
            - 在线网络和目标网络
            - ε-贪婪探索策略
            - TD 误差计算和梯度更新

        数学原理 (Mathematical Theory):
            Agent 执行以下计算：
            1. 动作选择: $a = \arg\max_a Q_\theta(s, a)$ (概率 $1-\epsilon$)
            2. TD 误差: $\delta = r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)$
            3. 参数更新: $\theta \leftarrow \theta - \alpha \nabla_\theta \delta^2$

        Args:
            q_net: Q 网络实例
        """
        self.train_step_counter = tf.Variable(0, dtype=tf.int64)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate
        )

        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter,
            gamma=self.config.discount_factor,
            epsilon_greedy=self.config.epsilon_greedy,
            target_update_period=self.config.target_update_period,
            target_update_tau=self.config.target_update_tau
        )

        self.agent.initialize()
        logger.info("DQN 智能体初始化完成")

    def _create_replay_buffer(self) -> None:
        """
        创建经验回放缓冲区

        核心思想 (Core Idea):
            经验回放存储历史交互经验，通过随机采样打破样本相关性，
            使训练更接近 i.i.d. 假设。

        数学原理 (Mathematical Theory):
            缓冲区 $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$

            均匀采样概率: $P(i) = \frac{1}{|\mathcal{D}|}$

            采样复杂度: O(1) 均摊
        """
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.config.replay_buffer_capacity
        )

        logger.info(f"回放缓冲区容量: {self.config.replay_buffer_capacity}")

    def _collect_step(self, environment, policy) -> None:
        """
        收集单步交互数据

        Args:
            environment: 交互环境
            policy: 数据收集策略
        """
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)

        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        self.replay_buffer.add_batch(traj)

    def _compute_avg_return(
        self,
        environment,
        policy,
        num_episodes: int = 10
    ) -> float:
        """
        计算平均回合回报

        数学原理 (Mathematical Theory):
            回报 $G = \sum_{t=0}^T r_t$（无折扣评估）

            平均回报 $\bar{G} = \frac{1}{N}\sum_{i=1}^N G_i$

        Args:
            environment: 评估环境
            policy: 评估策略
            num_episodes: 评估回合数

        Returns:
            平均回合回报
        """
        total_return = 0.0

        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward.numpy()[0]

            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return

    def initialize(self) -> None:
        """
        初始化所有组件

        执行顺序：
        1. 创建环境
        2. 构建 Q 网络
        3. 创建智能体
        4. 创建回放缓冲区
        5. 预填充缓冲区
        """
        self._create_environments()
        q_net = self._build_q_network()
        self._create_agent(q_net)
        self._create_replay_buffer()

        # 预填充回放缓冲区
        logger.info("预填充回放缓冲区...")
        random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(),
            self.train_env.action_spec()
        )

        # 收集初始数据
        self.train_env.reset()
        for _ in range(self.config.batch_size * 10):
            self._collect_step(self.train_env, random_policy)

        logger.info(f"初始缓冲区大小: {self.replay_buffer.num_frames().numpy()}")

    def train(self) -> list[float]:
        """
        执行训练循环

        核心算法流程:
            for iteration in range(num_iterations):
                1. 使用收集策略收集数据
                2. 从缓冲区采样批量数据
                3. 计算 TD 目标和损失
                4. 梯度更新
                5. 周期性评估和日志

        Returns:
            评估回报历史列表

        Raises:
            RuntimeError: 如果组件未初始化
        """
        if self.agent is None:
            raise RuntimeError("请先调用 initialize() 方法")

        # 创建数据集
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.config.batch_size,
            num_steps=2  # 连续两步用于计算 TD 目标
        ).prefetch(3)

        iterator = iter(dataset)

        # 包装训练步骤
        self.agent.train = common.function(self.agent.train)

        # 初始评估
        avg_return = self._compute_avg_return(
            self.eval_env,
            self.agent.policy,
            self.config.num_eval_episodes
        )
        self.returns_history.append(avg_return)
        logger.info(f"初始评估回报: {avg_return:.2f}")

        # 训练循环
        self.train_env.reset()

        for iteration in range(self.config.num_iterations):
            # 收集数据
            for _ in range(self.config.collect_steps_per_iteration):
                self._collect_step(self.train_env, self.agent.collect_policy)

            # 训练步骤
            experience, _ = next(iterator)
            train_loss = self.agent.train(experience).loss

            step = self.train_step_counter.numpy()

            # 日志
            if step % self.config.log_interval == 0:
                logger.info(f"Step {step}: Loss = {train_loss:.4f}")

            # 评估
            if step % self.config.eval_interval == 0:
                avg_return = self._compute_avg_return(
                    self.eval_env,
                    self.agent.policy,
                    self.config.num_eval_episodes
                )
                self.returns_history.append(avg_return)
                logger.info(f"Step {step}: Avg Return = {avg_return:.2f}")

        return self.returns_history


def create_custom_q_network(
    observation_spec,
    action_spec,
    preprocessing_layers: Optional[dict] = None,
    fc_layer_params: tuple = (256, 128, 64),
    dropout_rate: float = 0.0,
    activation_fn: Callable = tf.keras.activations.relu
) -> q_network.QNetwork:
    """
    创建自定义 Q 网络

    核心思想 (Core Idea):
        提供灵活的网络配置选项，支持不同的预处理、网络深度和正则化。

    数学原理 (Mathematical Theory):
        Q 网络结构：
        $$Q_\theta(s, \cdot) = f_L \circ f_{L-1} \circ ... \circ f_1(s)$$

        其中 $f_i$ 为带激活函数的全连接层：
        $$f_i(x) = \text{ReLU}(W_i x + b_i)$$

        Dropout 正则化（训练时）：
        $$\tilde{x} = x \odot m, \quad m_i \sim \text{Bernoulli}(1-p)$$

    Args:
        observation_spec: 观测空间规范
        action_spec: 动作空间规范
        preprocessing_layers: 预处理层字典
        fc_layer_params: 全连接层维度元组
        dropout_rate: Dropout 概率
        activation_fn: 激活函数

    Returns:
        配置好的 Q 网络

    Example:
        >>> q_net = create_custom_q_network(
        ...     obs_spec, act_spec,
        ...     fc_layer_params=(256, 128),
        ...     dropout_rate=0.1
        ... )
    """
    # 构建编码网络
    encoder = encoding_network.EncodingNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        fc_layer_params=fc_layer_params[:-1] if len(fc_layer_params) > 1 else (),
        activation_fn=activation_fn,
        dropout_layer_params=[dropout_rate] * (len(fc_layer_params) - 1)
            if dropout_rate > 0 else None
    )

    # 构建完整 Q 网络
    q_net = q_network.QNetwork(
        observation_spec,
        action_spec,
        encoding_network=encoder,
        fc_layer_params=(fc_layer_params[-1],) if len(fc_layer_params) > 0 else ()
    )

    return q_net


def run_quick_test():
    """
    快速测试 DQN 实现

    使用较小的参数验证代码正确性，适合调试和 CI/CD。
    """
    logger.info("=" * 60)
    logger.info("DQN 快速测试")
    logger.info("=" * 60)

    # 测试配置（小规模）
    config = DQNConfig(
        num_iterations=100,
        batch_size=32,
        log_interval=20,
        eval_interval=50,
        num_eval_episodes=3
    )

    trainer = DQNTrainer(config)
    trainer.initialize()
    returns = trainer.train()

    logger.info(f"测试完成，回报历史: {returns}")
    logger.info("=" * 60)


def run_full_training():
    """
    完整训练流程

    使用生产级参数训练 DQN 智能体。
    """
    logger.info("=" * 60)
    logger.info("DQN 完整训练 - CartPole-v1")
    logger.info("=" * 60)

    config = DQNConfig(
        num_iterations=20000,
        batch_size=64,
        learning_rate=1e-3,
        replay_buffer_capacity=100000,
        epsilon_greedy=0.1,
        target_update_period=200,
        fc_layer_params=(100, 50),
        log_interval=500,
        eval_interval=1000,
        num_eval_episodes=10
    )

    trainer = DQNTrainer(config)
    trainer.initialize()
    returns = trainer.train()

    # 输出训练结果
    logger.info("=" * 60)
    logger.info("训练完成")
    logger.info(f"最终平均回报: {returns[-1]:.2f}")
    logger.info(f"最高平均回报: {max(returns):.2f}")
    logger.info("=" * 60)

    return returns


if __name__ == "__main__":
    # 设置环境变量
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 可选：运行快速测试
    # run_quick_test()

    # 运行完整训练
    run_full_training()
