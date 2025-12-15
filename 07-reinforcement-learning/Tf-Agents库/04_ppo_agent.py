#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proximal Policy Optimization (PPO) 智能体实现

本模块实现了 PPO 算法，这是目前最流行的策略梯度方法之一。
PPO 通过裁剪目标函数限制策略更新幅度，实现了简单、稳定且高效的训练。

================================================================================
核心思想 (Core Idea)
================================================================================
PPO 是 TRPO (Trust Region Policy Optimization) 的简化版本，
通过在目标函数中添加裁剪项来限制策略更新幅度，避免过大的策略变化导致训练不稳定。
相比 TRPO 的二阶优化，PPO 只需一阶梯度，实现简单且同样有效。

核心创新：
1. 裁剪比率限制策略变化
2. 使用多个 epoch 重复利用数据
3. GAE 优势估计降低方差

================================================================================
数学原理 (Mathematical Theory)
================================================================================
策略梯度定理：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[
    \nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s, a)
\right]$$

重要性采样（使用旧策略数据）：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{old}}}\left[
    \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_{\theta_{old}}}(s, a)
\right]$$

定义概率比率：
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

PPO-Clip 目标函数：
$$L^{CLIP}(\theta) = \mathbb{E}_t\left[
    \min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)
\right]$$

其中：
- $\hat{A}_t$: GAE 优势估计
- $\epsilon$: 裁剪参数（通常 0.1-0.2）
- $\text{clip}(x, a, b) = \max(a, \min(x, b))$

完整目标函数（包含价值和熵）：
$$L^{CLIP+VF+S}(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta]$$

其中：
- $L^{VF} = (V_\theta(s) - V_{target})^2$: 价值函数损失
- $S[\pi_\theta] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$: 熵奖励
- $c_1, c_2$: 损失权重系数

广义优势估计 (GAE)：
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

================================================================================
问题背景 (Problem Statement)
================================================================================
策略梯度方法面临的主要挑战：
1. 高方差：需要大量样本才能获得稳定的梯度估计
2. 步长敏感：策略更新太大可能导致性能崩溃
3. 样本效率：on-policy 方法每次更新后数据失效

TRPO 通过 KL 散度约束解决步长问题，但需要二阶优化，计算复杂。
PPO 通过裁剪目标函数实现类似效果，但只需一阶梯度。

================================================================================
算法对比 (Comparison)
================================================================================
| 方法         | 类型       | 样本效率 | 稳定性 | 实现复杂度 | 超参敏感性 |
|-------------|-----------|---------|--------|-----------|-----------|
| REINFORCE   | on-policy | 低      | 低     | 简单      | 高        |
| A2C/A3C     | on-policy | 低      | 中     | 中等      | 中        |
| TRPO        | on-policy | 低      | 高     | 复杂      | 低        |
| PPO         | on-policy | 低      | 高     | 简单      | 低        |
| SAC         | off-policy| 高      | 高     | 中等      | 低        |

PPO 优势：
- 实现简单：只需裁剪操作，无需二阶优化
- 训练稳定：裁剪限制策略变化幅度
- 超参鲁棒：对超参数选择不敏感
- 通用性强：适用于离散和连续动作空间

PPO 劣势：
- 样本效率低：on-policy 方法，数据无法重复利用
- 难以并行：需要同步更新（A3C 可异步）

================================================================================
复杂度 (Complexity)
================================================================================
时间复杂度（单次更新）：
- 数据收集: O(T × N)，T=步数，N=并行环境数
- 策略更新: O(K × B × D)，K=epoch数，B=批大小，D=网络参数

空间复杂度：
- 轨迹缓冲区: O(T × N × S)
- 网络参数: O(D)（Actor + Critic）

================================================================================
算法总结 (Summary)
================================================================================
PPO 训练循环：
1. 使用当前策略收集 N 步交互数据
2. 计算 GAE 优势估计
3. 对收集的数据进行 K 个 epoch 的小批量更新：
   a. 计算概率比率 r_t(θ)
   b. 计算裁剪目标
   c. 更新策略网络
   d. 更新价值网络
4. 重复以上过程

================================================================================
Reference:
    Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms.
    arXiv:1707.06347.
================================================================================
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import numpy as np
import tensorflow as tf

# TF-Agents imports
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.drivers import dynamic_episode_driver

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """
    PPO 超参数配置

    核心思想 (Core Idea):
        集中管理 PPO 算法的所有超参数，便于实验复现和调优。

    数学原理 (Mathematical Theory):
        关键参数对应公式：
        - importance_ratio_clipping: 裁剪参数 ε
        - lambda_value: GAE 参数 λ
        - discount_factor: 折扣因子 γ
        - entropy_regularization: 熵系数 c_2
        - value_pred_loss_coef: 价值损失系数 c_1

    Attributes:
        env_name: 环境名称
        num_iterations: 训练迭代数
        num_parallel_environments: 并行环境数
        collect_episodes_per_iteration: 每迭代收集回合数
        num_epochs: 每批数据的训练 epoch 数
        num_steps: 每回合最大步数
        actor_fc_layers: Actor 网络隐藏层
        value_fc_layers: Value 网络隐藏层
        learning_rate: 学习率
        importance_ratio_clipping: PPO 裁剪参数 ε
        lambda_value: GAE λ 参数
        discount_factor: 折扣因子 γ
        entropy_regularization: 熵正则化系数
        value_pred_loss_coef: 价值函数损失系数
        use_gae: 是否使用 GAE
        normalize_advantages: 是否标准化优势
        normalize_observations: 是否标准化观测
        gradient_clipping: 梯度裁剪阈值
        log_interval: 日志间隔
        eval_interval: 评估间隔
        num_eval_episodes: 评估回合数
    """
    env_name: str = "CartPole-v1"
    num_iterations: int = 500
    num_parallel_environments: int = 4
    collect_episodes_per_iteration: int = 4
    num_epochs: int = 25
    num_steps: int = 256
    actor_fc_layers: Tuple[int, ...] = (64, 64)
    value_fc_layers: Tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    importance_ratio_clipping: float = 0.2
    lambda_value: float = 0.95
    discount_factor: float = 0.99
    entropy_regularization: float = 0.0
    value_pred_loss_coef: float = 0.5
    use_gae: bool = True
    normalize_advantages: bool = True
    normalize_observations: bool = False
    gradient_clipping: Optional[float] = 0.5
    log_interval: int = 25
    eval_interval: int = 50
    num_eval_episodes: int = 10


class PPOTrainer:
    """
    PPO 智能体训练器

    封装 PPO-Clip 算法的完整训练流程，支持并行环境数据收集。

    核心思想 (Core Idea):
        PPO 通过裁剪目标函数限制策略更新幅度，
        实现简单稳定的策略梯度优化。

    Attributes:
        config: PPO 配置对象
        train_env: 训练环境（可并行）
        eval_env: 评估环境
        agent: PPO 智能体
        replay_buffer: 轨迹缓冲区
        returns_history: 评估回报历史

    Example:
        >>> config = PPOConfig(env_name="CartPole-v1")
        >>> trainer = PPOTrainer(config)
        >>> trainer.initialize()
        >>> returns = trainer.train()
    """

    def __init__(self, config: PPOConfig):
        """
        初始化 PPO 训练器

        Args:
            config: PPO 配置对象
        """
        self.config = config
        self.train_env: Optional[tf_py_environment.TFPyEnvironment] = None
        self.eval_env: Optional[tf_py_environment.TFPyEnvironment] = None
        self.agent: Optional[ppo_clip_agent.PPOClipAgent] = None
        self.replay_buffer: Optional[tf_uniform_replay_buffer.TFUniformReplayBuffer] = None
        self.train_step_counter: Optional[tf.Variable] = None
        self.returns_history: list[float] = []

    def _create_environments(self) -> None:
        """
        创建训练和评估环境

        核心思想 (Core Idea):
            PPO 通过并行环境加速数据收集。
            多个环境同时运行，增加样本多样性。

        数学原理 (Mathematical Theory):
            并行环境相当于增加批大小：
            有效批大小 = N_env × T_steps

            这增加了梯度估计的稳定性，降低方差。
        """
        # 创建并行训练环境
        if self.config.num_parallel_environments > 1:
            train_py_env = parallel_py_environment.ParallelPyEnvironment(
                [lambda: suite_gym.load(self.config.env_name)]
                * self.config.num_parallel_environments
            )
        else:
            train_py_env = suite_gym.load(self.config.env_name)

        eval_py_env = suite_gym.load(self.config.env_name)

        self.train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        logger.info(f"环境: {self.config.env_name}")
        logger.info(f"  并行环境数: {self.config.num_parallel_environments}")
        logger.info(f"  观测空间: {self.train_env.observation_spec()}")
        logger.info(f"  动作空间: {self.train_env.action_spec()}")

    def _build_actor_network(self) -> actor_distribution_network.ActorDistributionNetwork:
        """
        构建 Actor 网络（策略网络）

        核心思想 (Core Idea):
            Actor 网络输出动作分布。对于离散动作空间，输出 Categorical 分布；
            对于连续动作空间，输出 Gaussian 分布。

        数学原理 (Mathematical Theory):
            离散动作空间：
            $$\pi_\theta(a|s) = \text{softmax}(f_\theta(s))_a$$

            连续动作空间：
            $$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$$

        Returns:
            Actor 分布网络
        """
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=self.config.actor_fc_layers,
            activation_fn=tf.keras.activations.tanh
        )

        logger.info(f"Actor 网络: {self.config.actor_fc_layers}")
        return actor_net

    def _build_value_network(self) -> value_network.ValueNetwork:
        """
        构建 Value 网络（价值网络）

        核心思想 (Core Idea):
            Value 网络估计状态价值 V(s)，用于计算优势函数和作为 baseline 降低方差。

        数学原理 (Mathematical Theory):
            状态价值函数：
            $$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s\right]$$

            用于计算 TD 误差：
            $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

        Returns:
            Value 网络
        """
        value_net = value_network.ValueNetwork(
            self.train_env.observation_spec(),
            fc_layer_params=self.config.value_fc_layers,
            activation_fn=tf.keras.activations.tanh
        )

        logger.info(f"Value 网络: {self.config.value_fc_layers}")
        return value_net

    def _create_agent(
        self,
        actor_net: actor_distribution_network.ActorDistributionNetwork,
        value_net: value_network.ValueNetwork
    ) -> None:
        """
        创建 PPO 智能体

        核心思想 (Core Idea):
            PPO Agent 实现裁剪目标函数和多 epoch 训练。

        数学原理 (Mathematical Theory):
            PPO-Clip 目标：
            $$L^{CLIP}(\theta) = \mathbb{E}_t\left[
                \min\left(r_t(\theta) \hat{A}_t,
                \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)
            \right]$$

            裁剪机制：
            - 当 $\hat{A}_t > 0$（好动作）：限制 $r_t \leq 1+\epsilon$
            - 当 $\hat{A}_t < 0$（坏动作）：限制 $r_t \geq 1-\epsilon$

        Args:
            actor_net: Actor 网络
            value_net: Value 网络
        """
        self.train_step_counter = tf.Variable(0, dtype=tf.int64)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate
        )

        self.agent = ppo_clip_agent.PPOClipAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            actor_net=actor_net,
            value_net=value_net,
            optimizer=optimizer,
            importance_ratio_clipping=self.config.importance_ratio_clipping,
            lambda_value=self.config.lambda_value,
            discount_factor=self.config.discount_factor,
            entropy_regularization=self.config.entropy_regularization,
            value_pred_loss_coef=self.config.value_pred_loss_coef,
            num_epochs=self.config.num_epochs,
            use_gae=self.config.use_gae,
            normalize_advantages=self.config.normalize_advantages,
            normalize_observations=self.config.normalize_observations,
            gradient_clipping=self.config.gradient_clipping,
            train_step_counter=self.train_step_counter
        )

        self.agent.initialize()
        logger.info("PPO 智能体初始化完成")
        logger.info(f"  裁剪参数 ε: {self.config.importance_ratio_clipping}")
        logger.info(f"  GAE λ: {self.config.lambda_value}")
        logger.info(f"  训练 epochs: {self.config.num_epochs}")

    def _create_replay_buffer(self) -> None:
        """
        创建轨迹缓冲区

        核心思想 (Core Idea):
            PPO 使用 on-policy 数据，每次更新后清空缓冲区。
            缓冲区仅存储当前迭代收集的轨迹。
        """
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.config.num_steps + 1
        )

    def _collect_episode(self, environment, policy) -> int:
        """
        收集完整回合数据

        Args:
            environment: 交互环境
            policy: 数据收集策略

        Returns:
            回合步数
        """
        time_step = environment.current_time_step()
        steps = 0

        while not time_step.is_last() and steps < self.config.num_steps:
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)

            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            self.replay_buffer.add_batch(traj)

            time_step = next_time_step
            steps += 1

        return steps

    def _compute_avg_return(
        self,
        environment,
        policy,
        num_episodes: int = 10
    ) -> float:
        """
        计算平均回合回报

        Args:
            environment: 评估环境
            policy: 评估策略
            num_episodes: 评估回合数

        Returns:
            平均回报
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

        return total_return / num_episodes

    def initialize(self) -> None:
        """
        初始化所有组件
        """
        self._create_environments()
        actor_net = self._build_actor_network()
        value_net = self._build_value_network()
        self._create_agent(actor_net, value_net)
        self._create_replay_buffer()

    def train(self) -> list[float]:
        """
        执行 PPO 训练循环

        核心算法流程:
            for iteration in range(num_iterations):
                1. 使用当前策略收集 N 个回合数据
                2. 计算 GAE 优势估计
                3. 进行 K 个 epoch 的小批量更新：
                   - 计算概率比率 r_t(θ)
                   - 计算裁剪后的目标
                   - 更新策略和价值网络
                4. 清空缓冲区
                5. 周期性评估

        Returns:
            评估回报历史列表

        Raises:
            RuntimeError: 如果组件未初始化
        """
        if self.agent is None:
            raise RuntimeError("请先调用 initialize() 方法")

        # 编译训练函数
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
        for iteration in range(self.config.num_iterations):
            # 清空缓冲区
            self.replay_buffer.clear()

            # 收集数据
            self.train_env.reset()
            total_steps = 0

            for _ in range(self.config.collect_episodes_per_iteration):
                steps = self._collect_episode(
                    self.train_env,
                    self.agent.collect_policy
                )
                total_steps += steps

            # 准备训练数据
            trajectories = self.replay_buffer.gather_all()

            # 训练步骤
            train_loss = self.agent.train(experience=trajectories)

            step = self.train_step_counter.numpy()

            # 日志
            if (iteration + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Iteration {iteration + 1}: "
                    f"Loss = {train_loss.loss:.4f}, "
                    f"Steps = {total_steps}"
                )

            # 评估
            if (iteration + 1) % self.config.eval_interval == 0:
                avg_return = self._compute_avg_return(
                    self.eval_env,
                    self.agent.policy,
                    self.config.num_eval_episodes
                )
                self.returns_history.append(avg_return)
                logger.info(
                    f"Iteration {iteration + 1}: Avg Return = {avg_return:.2f}"
                )

        return self.returns_history


def compute_ppo_loss(
    log_probs: tf.Tensor,
    old_log_probs: tf.Tensor,
    advantages: tf.Tensor,
    clip_ratio: float = 0.2
) -> tf.Tensor:
    """
    计算 PPO-Clip 损失

    核心思想 (Core Idea):
        通过裁剪概率比率，限制策略更新幅度，防止过大的策略变化。

    数学原理 (Mathematical Theory):
        概率比率：
        $$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
                      = \exp(\log \pi_\theta - \log \pi_{\theta_{old}})$$

        裁剪目标：
        $$L^{CLIP} = \min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)$$

        最大化目标等价于最小化负值：
        $$\text{Loss} = -L^{CLIP}$$

    Args:
        log_probs: 当前策略的动作对数概率
        old_log_probs: 旧策略的动作对数概率
        advantages: 优势估计
        clip_ratio: 裁剪参数 ε

    Returns:
        PPO 裁剪损失（标量）

    复杂度 (Complexity):
        时间: O(B)，B 为批大小
        空间: O(B)
    """
    # 计算概率比率
    ratio = tf.exp(log_probs - old_log_probs)

    # 裁剪后的比率
    clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # 取两者较小值（悲观估计）
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

    return policy_loss


def compute_value_loss(
    values: tf.Tensor,
    returns: tf.Tensor,
    old_values: Optional[tf.Tensor] = None,
    clip_ratio: float = 0.2,
    use_clipping: bool = True
) -> tf.Tensor:
    """
    计算价值函数损失

    核心思想 (Core Idea):
        价值网络损失用于训练状态价值估计。
        可选的裁剪防止价值函数过大变化。

    数学原理 (Mathematical Theory):
        基础 MSE 损失：
        $$L^{VF} = \frac{1}{2} (V_\theta(s) - V_{target})^2$$

        裁剪版本（类似 PPO 策略裁剪）：
        $$L^{VF}_{clip} = \max\left(
            (V_\theta - V_{target})^2,
            (\text{clip}(V_\theta, V_{old}-\epsilon, V_{old}+\epsilon) - V_{target})^2
        \right)$$

    Args:
        values: 当前价值估计
        returns: 目标回报（或 GAE + old_values）
        old_values: 旧价值估计（用于裁剪）
        clip_ratio: 裁剪参数
        use_clipping: 是否使用裁剪

    Returns:
        价值函数损失（标量）
    """
    if use_clipping and old_values is not None:
        # 裁剪版本
        clipped_values = old_values + tf.clip_by_value(
            values - old_values, -clip_ratio, clip_ratio
        )
        value_loss_unclipped = tf.square(values - returns)
        value_loss_clipped = tf.square(clipped_values - returns)
        value_loss = 0.5 * tf.reduce_mean(
            tf.maximum(value_loss_unclipped, value_loss_clipped)
        )
    else:
        # 基础 MSE
        value_loss = 0.5 * tf.reduce_mean(tf.square(values - returns))

    return value_loss


def run_quick_test():
    """
    快速测试 PPO 实现
    """
    logger.info("=" * 60)
    logger.info("PPO 快速测试")
    logger.info("=" * 60)

    config = PPOConfig(
        env_name="CartPole-v1",
        num_iterations=20,
        num_parallel_environments=2,
        collect_episodes_per_iteration=2,
        num_epochs=5,
        log_interval=5,
        eval_interval=10,
        num_eval_episodes=3
    )

    trainer = PPOTrainer(config)
    trainer.initialize()
    returns = trainer.train()

    logger.info(f"测试完成，回报历史: {returns}")
    logger.info("=" * 60)


def run_full_training():
    """
    完整 PPO 训练

    使用生产级参数训练 PPO 智能体。
    """
    logger.info("=" * 60)
    logger.info("PPO 完整训练 - CartPole-v1")
    logger.info("=" * 60)

    config = PPOConfig(
        env_name="CartPole-v1",
        num_iterations=500,
        num_parallel_environments=4,
        collect_episodes_per_iteration=4,
        num_epochs=25,
        num_steps=256,
        actor_fc_layers=(64, 64),
        value_fc_layers=(64, 64),
        learning_rate=1e-3,
        importance_ratio_clipping=0.2,
        lambda_value=0.95,
        discount_factor=0.99,
        entropy_regularization=0.01,
        log_interval=25,
        eval_interval=50,
        num_eval_episodes=10
    )

    trainer = PPOTrainer(config)
    trainer.initialize()
    returns = trainer.train()

    logger.info("=" * 60)
    logger.info("训练完成")
    logger.info(f"最终平均回报: {returns[-1]:.2f}")
    logger.info(f"最高平均回报: {max(returns):.2f}")
    logger.info("=" * 60)

    return returns


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 快速测试
    # run_quick_test()

    # 完整训练
    run_full_training()
