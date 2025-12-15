#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soft Actor-Critic (SAC) 智能体实现 - 连续控制

本模块实现了 SAC 算法，这是当前最先进的连续动作空间强化学习方法之一。
通过最大熵强化学习框架，SAC 在样本效率和稳定性上达到了优秀的平衡。

================================================================================
核心思想 (Core Idea)
================================================================================
SAC 是一种 off-policy actor-critic 算法，其核心创新是将策略熵项加入目标函数，
鼓励探索的同时保持学习稳定性。算法维护三个网络：
1. Actor (策略网络): 输出动作分布
2. Critic (双 Q 网络): 估计动作价值，取较小值防止过估计
3. Value (价值网络，可选): 估计状态价值

================================================================================
数学原理 (Mathematical Theory)
================================================================================
最大熵强化学习目标：
$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}
\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$

其中：
- $\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$ 是策略熵
- $\alpha > 0$ 是温度参数，控制探索-利用权衡

软贝尔曼方程：
$$Q(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim P}
\left[V(s')\right]$$

$$V(s) = \mathbb{E}_{a \sim \pi}\left[Q(s, a) - \alpha \log \pi(a|s)\right]$$

策略更新（KL 散度最小化）：
$$\pi^* = \arg\min_\pi D_{KL}\left(\pi(\cdot|s) \Big\|
\frac{\exp(Q(s, \cdot)/\alpha)}{Z(s)}\right)$$

等价于最大化：
$$J_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\phi}
\left[Q_\theta(s, a) - \alpha \log \pi_\phi(a|s)\right]$$

自动温度调节：
$$J(\alpha) = \mathbb{E}_{a \sim \pi}
\left[-\alpha \log \pi(a|s) - \alpha \bar{\mathcal{H}}\right]$$

其中 $\bar{\mathcal{H}}$ 是目标熵（通常设为 $-\dim(\mathcal{A})$）。

================================================================================
问题背景 (Problem Statement)
================================================================================
连续动作空间强化学习面临几个关键挑战：
1. 动作空间无限大，无法穷举搜索
2. 确定性策略容易陷入局部最优
3. 探索-利用权衡难以平衡
4. Q 值过估计导致训练不稳定

SAC 通过以下机制解决：
1. 随机策略自然支持连续动作
2. 熵正则化鼓励多样性探索
3. 自动温度调节动态平衡
4. 双 Q 网络取最小值防止过估计

================================================================================
算法对比 (Comparison)
================================================================================
| 方法    | 策略类型  | 样本效率 | 稳定性 | 探索机制     | 调参难度 |
|--------|----------|---------|--------|-------------|---------|
| DDPG   | 确定性   | 中等     | 低     | 噪声注入     | 高      |
| TD3    | 确定性   | 中等     | 中     | 噪声+延迟更新 | 中      |
| SAC    | 随机     | 高      | 高     | 熵最大化     | 低      |
| PPO    | 随机     | 低      | 高     | 策略裁剪     | 低      |

SAC 优势：
- 自动探索：熵项自动平衡探索
- 样本高效：off-policy 可重复利用数据
- 超参数鲁棒：自动温度调节减少调参

SAC 劣势：
- 计算开销：维护多个网络
- 内存占用：需要回放缓冲区
- 离散动作支持有限（需改进）

================================================================================
复杂度 (Complexity)
================================================================================
时间复杂度（单次更新）：
- Critic 更新: O(B × D)，B=批大小，D=网络参数
- Actor 更新: O(B × D)
- Alpha 更新: O(B)

空间复杂度：
- 网络参数: O(D)（Actor + 2×Critic + 2×Target）
- 回放缓冲区: O(N × (S + A))

================================================================================
算法总结 (Summary)
================================================================================
SAC 训练循环：
1. 从策略采样动作 $a \sim \pi_\phi(·|s)$
2. 环境交互获得 $(s, a, r, s', done)$
3. 存入回放缓冲区
4. 采样批量数据
5. 更新 Q 网络（软贝尔曼目标）
6. 更新策略网络（最大化 Q - α log π）
7. 更新温度参数（目标熵约束）
8. 软更新目标网络

================================================================================
Reference:
    Haarnoja, T. et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy
    Deep Reinforcement Learning with a Stochastic Actor. ICML.

    Haarnoja, T. et al. (2018). Soft Actor-Critic Algorithms and Applications.
    arXiv:1812.05905.
================================================================================
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

# TF-Agents imports
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
from tf_agents.train.utils import spec_utils

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SACConfig:
    """
    SAC 超参数配置

    核心思想 (Core Idea):
        集中管理 SAC 算法的所有超参数，便于实验和复现。

    数学原理 (Mathematical Theory):
        关键参数对应算法公式：
        - actor_learning_rate: Actor 优化器学习率
        - critic_learning_rate: Critic 优化器学习率
        - alpha_learning_rate: 温度 α 学习率
        - target_entropy: 目标熵 $\bar{\mathcal{H}}$，通常设为 $-\dim(\mathcal{A})$
        - gamma: 折扣因子 $\gamma$
        - tau: 目标网络软更新系数 $\tau$

    Attributes:
        env_name: 连续控制环境名称
        num_iterations: 总训练迭代数
        collect_steps_per_iteration: 每迭代收集步数
        replay_buffer_capacity: 回放缓冲区容量
        batch_size: 训练批大小
        actor_learning_rate: Actor 学习率
        critic_learning_rate: Critic 学习率
        alpha_learning_rate: 温度参数学习率
        target_entropy: 目标熵（None 表示自动计算）
        gamma: 折扣因子
        tau: 目标网络软更新系数
        reward_scale_factor: 奖励缩放因子
        actor_fc_layers: Actor 网络隐藏层
        critic_joint_fc_layers: Critic 联合网络层
        log_interval: 日志间隔
        eval_interval: 评估间隔
        num_eval_episodes: 评估回合数
    """
    env_name: str = "Pendulum-v1"
    num_iterations: int = 100000
    collect_steps_per_iteration: int = 1
    replay_buffer_capacity: int = 1000000
    batch_size: int = 256
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    target_entropy: Optional[float] = None  # 自动设置为 -dim(action)
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale_factor: float = 1.0
    actor_fc_layers: Tuple[int, ...] = (256, 256)
    critic_joint_fc_layers: Tuple[int, ...] = (256, 256)
    log_interval: int = 1000
    eval_interval: int = 5000
    num_eval_episodes: int = 10


class SACTrainer:
    """
    SAC 智能体训练器

    封装 SAC 算法的完整训练流程，适用于连续动作空间任务。

    核心思想 (Core Idea):
        SAC 通过最大熵目标实现稳定的连续控制学习，
        本训练器提供端到端的训练管道。

    Attributes:
        config: SAC 配置对象
        train_env: 训练环境
        eval_env: 评估环境
        agent: SAC 智能体
        replay_buffer: 经验回放缓冲区
        returns_history: 评估回报历史

    Example:
        >>> config = SACConfig(env_name="Pendulum-v1")
        >>> trainer = SACTrainer(config)
        >>> trainer.initialize()
        >>> returns = trainer.train()
    """

    def __init__(self, config: SACConfig):
        """
        初始化 SAC 训练器

        Args:
            config: SAC 配置对象
        """
        self.config = config
        self.train_env: Optional[tf_py_environment.TFPyEnvironment] = None
        self.eval_env: Optional[tf_py_environment.TFPyEnvironment] = None
        self.agent: Optional[sac_agent.SacAgent] = None
        self.replay_buffer: Optional[tf_uniform_replay_buffer.TFUniformReplayBuffer] = None
        self.train_step_counter: Optional[tf.Variable] = None
        self.returns_history: list[float] = []

    def _create_environments(self) -> None:
        """
        创建训练和评估环境

        确保环境具有连续动作空间，适合 SAC 算法。
        """
        train_py_env = suite_gym.load(self.config.env_name)
        eval_py_env = suite_gym.load(self.config.env_name)

        self.train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        # 验证连续动作空间
        action_spec = self.train_env.action_spec()
        if not action_spec.dtype.is_floating:
            raise ValueError(
                f"SAC 需要连续动作空间，但 {self.config.env_name} "
                f"的动作类型为 {action_spec.dtype}"
            )

        logger.info(f"环境: {self.config.env_name}")
        logger.info(f"  观测空间: {self.train_env.observation_spec()}")
        logger.info(f"  动作空间: {action_spec}")
        logger.info(f"  动作范围: [{action_spec.minimum}, {action_spec.maximum}]")

    def _build_actor_network(self) -> actor_distribution_network.ActorDistributionNetwork:
        """
        构建 Actor 网络（策略网络）

        核心思想 (Core Idea):
            Actor 网络输出动作分布的参数（均值和标准差），
            使用 Tanh 将输出压缩到有效动作范围。

        数学原理 (Mathematical Theory):
            策略网络输出高斯分布参数：
            $$\mu, \sigma = f_\phi(s)$$

            动作采样（重参数化技巧）：
            $$a = \tanh(\mu + \sigma \cdot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

            Tanh 压缩后的对数概率需要雅可比校正：
            $$\log \pi(a|s) = \log \mathcal{N}(\mu, \sigma) - \sum_i \log(1 - \tanh^2(u_i))$$

        Returns:
            Actor 分布网络
        """
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=self.config.actor_fc_layers,
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork
        )

        logger.info(f"Actor 网络: {self.config.actor_fc_layers}")
        return actor_net

    def _build_critic_network(self) -> critic_network.CriticNetwork:
        """
        构建 Critic 网络（Q 网络）

        核心思想 (Core Idea):
            Critic 网络估计状态-动作对的 Q 值。
            SAC 使用双 Q 网络，取较小值防止过估计。

        数学原理 (Mathematical Theory):
            Q 网络输入状态和动作，输出标量 Q 值：
            $$Q_\theta(s, a) = f_\theta([s; a])$$

            双 Q 网络取最小：
            $$Q_{target} = \min(Q_1(s', a'), Q_2(s', a')) - \alpha \log \pi(a'|s')$$

        Returns:
            Critic 网络
        """
        # 观测和动作规范
        observation_spec, action_spec = spec_utils.get_tensor_specs(
            self.train_env.action_spec().shape
        )

        critic_net = critic_network.CriticNetwork(
            (self.train_env.observation_spec(), self.train_env.action_spec()),
            joint_fc_layer_params=self.config.critic_joint_fc_layers
        )

        logger.info(f"Critic 网络: {self.config.critic_joint_fc_layers}")
        return critic_net

    def _create_agent(
        self,
        actor_net: actor_distribution_network.ActorDistributionNetwork,
        critic_net: critic_network.CriticNetwork
    ) -> None:
        """
        创建 SAC 智能体

        核心思想 (Core Idea):
            SAC Agent 整合 Actor、Critic 和自动温度调节，
            实现最大熵强化学习。

        数学原理 (Mathematical Theory):
            智能体维护以下组件：
            1. Actor $\pi_\phi$: 策略网络
            2. Critic $Q_\theta$: 双 Q 网络
            3. Target Critic $Q_{\bar{\theta}}$: 目标 Q 网络
            4. Alpha $\alpha$: 温度参数（可学习）

            更新顺序：
            1. Critic: 最小化软贝尔曼残差
            2. Actor: 最大化 Q - α log π
            3. Alpha: 满足目标熵约束

        Args:
            actor_net: Actor 网络
            critic_net: Critic 网络
        """
        self.train_step_counter = tf.Variable(0, dtype=tf.int64)

        # 计算目标熵
        action_spec = self.train_env.action_spec()
        if self.config.target_entropy is None:
            # 默认目标熵: -dim(action)
            target_entropy = -np.prod(action_spec.shape).astype(np.float32)
        else:
            target_entropy = self.config.target_entropy

        logger.info(f"目标熵: {target_entropy}")

        self.agent = sac_agent.SacAgent(
            self.train_env.time_step_spec(),
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.actor_learning_rate
            ),
            critic_optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.critic_learning_rate
            ),
            alpha_optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.alpha_learning_rate
            ),
            target_entropy=target_entropy,
            gamma=self.config.gamma,
            target_update_tau=self.config.tau,
            reward_scale_factor=self.config.reward_scale_factor,
            train_step_counter=self.train_step_counter
        )

        self.agent.initialize()
        logger.info("SAC 智能体初始化完成")

    def _create_replay_buffer(self) -> None:
        """
        创建经验回放缓冲区

        SAC 是 off-policy 算法，依赖回放缓冲区重复利用历史数据。
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

        Args:
            environment: 评估环境
            policy: 评估策略（通常是贪婪策略）
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

        执行顺序：
        1. 创建环境
        2. 构建 Actor 和 Critic 网络
        3. 创建 SAC 智能体
        4. 创建回放缓冲区
        5. 预填充缓冲区
        """
        self._create_environments()
        actor_net = self._build_actor_network()
        critic_net = self._build_critic_network()
        self._create_agent(actor_net, critic_net)
        self._create_replay_buffer()

        # 预填充回放缓冲区
        logger.info("预填充回放缓冲区...")
        random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(),
            self.train_env.action_spec()
        )

        self.train_env.reset()
        initial_collect_steps = max(1000, self.config.batch_size * 10)
        for _ in range(initial_collect_steps):
            self._collect_step(self.train_env, random_policy)

        logger.info(f"初始缓冲区大小: {self.replay_buffer.num_frames().numpy()}")

    def train(self) -> list[float]:
        """
        执行 SAC 训练循环

        核心算法流程:
            for iteration in range(num_iterations):
                1. 采样动作: a ~ π(·|s)
                2. 环境交互: (s', r, done) = env.step(a)
                3. 存储经验: buffer.add(s, a, r, s', done)
                4. 采样批量: batch = buffer.sample(batch_size)
                5. 更新 Critic: 最小化软贝尔曼残差
                6. 更新 Actor: 最大化 Q - α log π
                7. 更新 Alpha: 目标熵约束
                8. 软更新目标网络: θ' ← τθ + (1-τ)θ'

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
            num_steps=2
        ).prefetch(3)

        iterator = iter(dataset)

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
        self.train_env.reset()

        for iteration in range(self.config.num_iterations):
            # 收集数据
            for _ in range(self.config.collect_steps_per_iteration):
                self._collect_step(self.train_env, self.agent.collect_policy)

            # 训练步骤
            experience, _ = next(iterator)
            train_loss = self.agent.train(experience)

            step = self.train_step_counter.numpy()

            # 日志
            if step % self.config.log_interval == 0:
                logger.info(
                    f"Step {step}: "
                    f"Actor Loss = {train_loss.extra.actor_loss:.4f}, "
                    f"Critic Loss = {train_loss.extra.critic_loss:.4f}, "
                    f"Alpha = {train_loss.extra.alpha_loss:.4f}"
                )

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


def create_squashed_gaussian_policy(
    observation_spec,
    action_spec,
    fc_layer_params: Tuple[int, ...] = (256, 256),
    activation: str = "relu"
) -> actor_distribution_network.ActorDistributionNetwork:
    """
    创建 Squashed Gaussian 策略网络

    核心思想 (Core Idea):
        输出高斯分布参数，使用 Tanh 压缩到动作边界。
        这是 SAC 标准的策略参数化方式。

    数学原理 (Mathematical Theory):
        策略分布：
        $$\pi_\phi(a|s) = \tanh\left(\mathcal{N}(\mu_\phi(s), \sigma_\phi(s)^2)\right)$$

        重参数化梯度：
        $$\nabla_\phi J = \mathbb{E}_{s, \epsilon}\left[
            \nabla_\phi Q(s, f_\phi(s, \epsilon)) -
            \alpha \nabla_\phi \log \pi_\phi(f_\phi(s, \epsilon)|s)
        \right]$$

        其中 $f_\phi(s, \epsilon) = \tanh(\mu_\phi(s) + \sigma_\phi(s) \cdot \epsilon)$

    Args:
        observation_spec: 观测空间规范
        action_spec: 动作空间规范
        fc_layer_params: 隐藏层维度
        activation: 激活函数名称

    Returns:
        配置好的 Actor 网络
    """
    activation_fn = getattr(tf.keras.activations, activation)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=fc_layer_params,
        activation_fn=activation_fn,
        continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork
    )

    return actor_net


def run_quick_test():
    """
    快速测试 SAC 实现

    使用小规模参数验证代码正确性。
    """
    logger.info("=" * 60)
    logger.info("SAC 快速测试")
    logger.info("=" * 60)

    config = SACConfig(
        env_name="Pendulum-v1",
        num_iterations=200,
        batch_size=64,
        log_interval=50,
        eval_interval=100,
        num_eval_episodes=3
    )

    trainer = SACTrainer(config)
    trainer.initialize()
    returns = trainer.train()

    logger.info(f"测试完成，回报历史: {returns}")
    logger.info("=" * 60)


def run_full_training():
    """
    完整 SAC 训练

    使用生产级参数训练 SAC 智能体解决 Pendulum 任务。
    """
    logger.info("=" * 60)
    logger.info("SAC 完整训练 - Pendulum-v1")
    logger.info("=" * 60)

    config = SACConfig(
        env_name="Pendulum-v1",
        num_iterations=100000,
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        actor_fc_layers=(256, 256),
        critic_joint_fc_layers=(256, 256),
        log_interval=1000,
        eval_interval=5000,
        num_eval_episodes=10
    )

    trainer = SACTrainer(config)
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
