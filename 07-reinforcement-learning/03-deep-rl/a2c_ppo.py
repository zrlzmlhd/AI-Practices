#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actor-Critic 与 PPO 算法实现

本模块实现策略梯度家族的核心算法：
- A2C (Advantage Actor-Critic)：同步版本的 Actor-Critic 算法
- PPO (Proximal Policy Optimization)：带信任域约束的策略优化

核心概念：
1. Actor-Critic 架构：分离策略（Actor）和价值估计（Critic）
2. 优势函数 (Advantage)：A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)
3. 广义优势估计 (GAE)：平衡偏差与方差的优势估计方法
4. PPO-Clip：通过裁剪比率限制策略更新幅度

数学基础：
- 策略梯度定理: ∇J(θ) = E_π[∇logπ(a|s)A(s,a)]
- GAE: Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}, δ_t = r_t + γV(s_{t+1}) - V(s_t)
- PPO 目标: L^{CLIP}(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

参考文献：
[1] Mnih et al., "Asynchronous Methods for Deep RL", 2016 (A3C/A2C)
[2] Schulman et al., "Proximal Policy Optimization Algorithms", 2017
[3] Schulman et al., "High-Dimensional Continuous Control Using GAE", 2015

运行环境：
    Python >= 3.8
    PyTorch >= 1.9
    Gymnasium >= 0.28
    NumPy >= 1.20

Author: Ziming Ding
Date: 2024
"""

from __future__ import annotations
import os
import random
import warnings
from typing import Tuple, List, Optional, Dict, NamedTuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# 可选依赖
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib 未安装，绘图功能不可用")

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    warnings.warn("gymnasium 未安装，环境交互功能不可用")


# =============================================================================
# 轨迹数据结构
# =============================================================================

class RolloutBatch(NamedTuple):
    """
    轨迹批次数据

    存储一个完整回滚（rollout）的所有数据，用于策略梯度更新。
    使用 NamedTuple 确保数据不可变性和类型检查。

    Attributes:
        states: 状态序列，形状 (T, state_dim)
        actions: 动作序列，形状 (T,)
        log_probs: 动作对数概率，形状 (T,)
        rewards: 即时奖励序列，形状 (T,)
        values: 状态价值估计，形状 (T,)
        dones: 回合终止标志，形状 (T,)
        next_values: 下一状态价值，形状 (T,)
    """
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    next_values: torch.Tensor


class RolloutBuffer:
    """
    轨迹缓冲区

    收集环境交互数据用于策略梯度训练。
    与经验回放不同，策略梯度方法需要完整轨迹数据，
    并且是 on-policy 的（数据只能用一次）。

    设计考虑：
    - 支持 GAE 优势估计
    - 支持 mini-batch 训练
    - 数据使用后清空（on-policy 特性）

    Attributes:
        gamma: 折扣因子
        gae_lambda: GAE λ 参数
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """
        初始化轨迹缓冲区

        Args:
            gamma: 折扣因子，控制未来奖励衰减
            gae_lambda: GAE λ 参数，控制偏差-方差权衡
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self) -> None:
        """清空缓冲区"""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ) -> None:
        """
        添加单步转换

        Args:
            state: 当前状态
            action: 执行的动作
            log_prob: 动作的对数概率
            reward: 获得的奖励
            value: 状态价值估计
            done: 回合是否结束
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self,
        last_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算回报和 GAE 优势估计

        GAE (Generalized Advantage Estimation) 公式：
        δ_t = r_t + γV(s_{t+1}) - V(s_t)
        Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

        Args:
            last_value: 最后状态的价值估计（用于 bootstrap）

        Returns:
            (returns, advantages): 回报和优势估计
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        n_steps = len(rewards)

        # 添加最后的价值用于 bootstrap
        values = np.append(values, last_value)

        # 计算 GAE
        advantages = np.zeros(n_steps, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n_steps)):
            # δ_t = r_t + γ(1-done)V(s_{t+1}) - V(s_t)
            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )
            # Â_t = δ_t + γλ(1-done)Â_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # 回报 = 优势 + 价值
        returns = advantages + values[:-1]

        return (
            torch.FloatTensor(returns),
            torch.FloatTensor(advantages)
        )

    def get_batch(
        self,
        last_value: float,
        device: torch.device
    ) -> Tuple[RolloutBatch, torch.Tensor, torch.Tensor]:
        """
        获取完整批次数据和计算的优势

        Args:
            last_value: 最后状态价值
            device: 计算设备

        Returns:
            (batch, returns, advantages): 批次数据和计算结果
        """
        returns, advantages = self.compute_returns_and_advantages(last_value)

        batch = RolloutBatch(
            states=torch.FloatTensor(np.array(self.states)).to(device),
            actions=torch.LongTensor(self.actions).to(device),
            log_probs=torch.FloatTensor(self.log_probs).to(device),
            rewards=torch.FloatTensor(self.rewards).to(device),
            values=torch.FloatTensor(self.values).to(device),
            dones=torch.FloatTensor(self.dones).to(device),
            next_values=torch.cat([
                torch.FloatTensor(self.values[1:]),
                torch.FloatTensor([last_value])
            ]).to(device)
        )

        return batch, returns.to(device), advantages.to(device)

    def __len__(self) -> int:
        return len(self.states)


# =============================================================================
# 神经网络架构
# =============================================================================

def _init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """
    正交初始化权重

    Args:
        module: 网络模块
        gain: 初始化增益
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 共享参数网络

    网络架构采用共享特征提取层 + 独立输出头的设计，
    这种设计可以共享低层特征表示，同时让策略和价值
    函数保持独立的高层表示。

    结构:
        Input → Shared MLP → Actor Head (策略) → π(a|s)
                          → Critic Head (价值) → V(s)

    设计考虑：
    - 共享层：提取通用状态特征
    - 独立头：允许策略和价值函数有不同的高层表示
    - 策略输出：动作概率分布（离散动作空间）
    - 价值输出：状态价值标量

    Attributes:
        shared: 共享特征提取层
        actor: 策略输出头
        critic: 价值输出头
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ) -> None:
        """
        初始化 Actor-Critic 网络

        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间大小
            hidden_dim: 隐藏层单元数
        """
        super().__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # 策略头（Actor）
        self.actor = nn.Linear(hidden_dim, action_dim)

        # 价值头（Critic）
        self.critic = nn.Linear(hidden_dim, 1)

        # 初始化权重
        self.shared.apply(lambda m: _init_weights(m, gain=np.sqrt(2)))
        _init_weights(self.actor, gain=0.01)  # 策略头使用较小增益
        _init_weights(self.critic, gain=1.0)

    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            state: 状态张量，形状 (batch_size, state_dim)

        Returns:
            (action_logits, value): 动作 logits 和状态价值
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作、对数概率、熵和价值

        Args:
            state: 状态张量
            action: 可选的指定动作（用于计算旧动作的新对数概率）

        Returns:
            (action, log_prob, entropy, value): 动作及相关信息
        """
        action_logits, value = self(state)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class A2CConfig:
    """
    A2C 超参数配置

    Attributes:
        state_dim: 状态空间维度
        action_dim: 动作空间大小
        hidden_dim: 隐藏层单元数
        learning_rate: 学习率
        gamma: 折扣因子
        gae_lambda: GAE λ 参数
        value_coef: 价值损失系数
        entropy_coef: 熵损失系数（鼓励探索）
        max_grad_norm: 最大梯度范数
        n_steps: 每次更新收集的步数
        device: 计算设备
    """
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 5
    device: str = "auto"


@dataclass
class PPOConfig:
    """
    PPO 超参数配置

    PPO 特有参数：
    - clip_epsilon: 裁剪系数，限制策略更新幅度
    - n_epochs: 每批数据的训练轮数
    - mini_batch_size: 小批次大小
    - target_kl: KL 散度阈值（可选的早停条件）

    Attributes:
        state_dim: 状态空间维度
        action_dim: 动作空间大小
        hidden_dim: 隐藏层单元数
        learning_rate: 学习率
        gamma: 折扣因子
        gae_lambda: GAE λ 参数
        clip_epsilon: PPO 裁剪系数
        value_coef: 价值损失系数
        entropy_coef: 熵损失系数
        max_grad_norm: 最大梯度范数
        n_steps: 每次更新收集的步数
        n_epochs: 每批数据训练轮数
        mini_batch_size: 小批次大小
        target_kl: KL 散度阈值
        device: 计算设备
    """
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    n_epochs: int = 10
    mini_batch_size: int = 64
    target_kl: Optional[float] = None
    device: str = "auto"


# =============================================================================
# A2C 智能体
# =============================================================================

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) 智能体

    A2C 是 A3C（异步优势演员-评论家）的同步版本，
    使用多环境并行收集数据，但在一个进程内同步更新。

    核心特点：
    1. Actor-Critic 架构：策略和价值函数共享特征
    2. 优势函数：使用 A(s,a) = r + γV(s') - V(s) 减少方差
    3. 熵正则化：鼓励探索，防止策略过早收敛
    4. N-step returns：平衡偏差和方差

    损失函数：
    L = L_policy + c1 * L_value - c2 * H[π]

    其中：
    - L_policy = -E[log π(a|s) * A(s,a)]
    - L_value = E[(V(s) - R)²]
    - H[π] = -E[Σ π(a|s) log π(a|s)]

    使用示例:
        >>> config = A2CConfig(state_dim=4, action_dim=2)
        >>> agent = A2CAgent(config)
        >>> action, log_prob, value = agent.get_action(state)
        >>> agent.store_transition(state, action, log_prob, reward, value, done)
        >>> loss = agent.update(last_value)
    """

    def __init__(self, config: A2CConfig) -> None:
        """
        初始化 A2C 智能体

        Args:
            config: 超参数配置
        """
        self.config = config
        self._setup_device()
        self._setup_network()
        self._setup_buffer()

    def _setup_device(self) -> None:
        """配置计算设备"""
        if self.config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(self.config.device)

    def _setup_network(self) -> None:
        """初始化网络和优化器"""
        self.network = ActorCriticNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )

    def _setup_buffer(self) -> None:
        """初始化轨迹缓冲区"""
        self.buffer = RolloutBuffer(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )

    def get_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[int, float, float]:
        """
        根据当前策略选择动作

        Args:
            state: 当前状态
            training: 是否为训练模式

        Returns:
            (action, log_prob, value): 动作、对数概率和价值估计
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(
                state_tensor
            )

        return (
            action.item(),
            log_prob.item(),
            value.item()
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ) -> None:
        """
        存储单步转换到缓冲区

        Args:
            state: 当前状态
            action: 执行的动作
            log_prob: 动作对数概率
            reward: 即时奖励
            value: 状态价值估计
            done: 回合是否结束
        """
        self.buffer.add(state, action, log_prob, reward, value, done)

    def update(self, last_value: float) -> Dict[str, float]:
        """
        执行一次策略更新

        使用收集的轨迹数据计算策略梯度并更新网络。

        Args:
            last_value: 最后状态的价值估计

        Returns:
            包含各项损失的字典
        """
        # 获取批次数据
        batch, returns, advantages = self.buffer.get_batch(
            last_value, self.device
        )

        # 标准化优势（减少方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算新的策略分布
        _, new_log_probs, entropy, values = self.network.get_action_and_value(
            batch.states, batch.actions
        )

        # 策略损失
        policy_loss = -(new_log_probs * advantages.detach()).mean()

        # 价值损失
        value_loss = F.mse_loss(values, returns)

        # 熵损失（负号因为我们想最大化熵）
        entropy_loss = -entropy.mean()

        # 总损失
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )

        # 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()

        # 清空缓冲区
        self.buffer.reset()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item()
        }

    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config
        }, path)

    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


# =============================================================================
# PPO 智能体
# =============================================================================

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) 智能体

    PPO 是目前最流行的策略梯度算法之一，通过限制策略更新幅度
    来保证训练稳定性，同时允许多次使用同一批数据。

    核心思想：
    1. 裁剪目标：限制新旧策略比率在 [1-ε, 1+ε] 范围内
    2. 多轮更新：每批数据可以训练多个 epoch
    3. Mini-batch：将大批次分割成小批次训练
    4. 价值裁剪（可选）：类似地限制价值函数更新

    PPO-Clip 目标函数：
    L^{CLIP}(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

    其中 r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)

    优势：
    - 样本效率高于 A2C（数据可复用）
    - 训练稳定（裁剪保护）
    - 实现简单
    - 超参数鲁棒

    使用示例:
        >>> config = PPOConfig(state_dim=4, action_dim=2)
        >>> agent = PPOAgent(config)
        >>> action, log_prob, value = agent.get_action(state)
        >>> agent.store_transition(state, action, log_prob, reward, value, done)
        >>> loss_info = agent.update(last_value)
    """

    def __init__(self, config: PPOConfig) -> None:
        """
        初始化 PPO 智能体

        Args:
            config: 超参数配置
        """
        self.config = config
        self._setup_device()
        self._setup_network()
        self._setup_buffer()

    def _setup_device(self) -> None:
        """配置计算设备"""
        if self.config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(self.config.device)

    def _setup_network(self) -> None:
        """初始化网络和优化器"""
        self.network = ActorCriticNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5  # PPO 常用设置
        )

    def _setup_buffer(self) -> None:
        """初始化轨迹缓冲区"""
        self.buffer = RolloutBuffer(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )

    def get_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[int, float, float]:
        """
        根据当前策略选择动作

        Args:
            state: 当前状态
            training: 是否为训练模式

        Returns:
            (action, log_prob, value): 动作、对数概率和价值估计
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(
                state_tensor
            )

        return (
            action.item(),
            log_prob.item(),
            value.item()
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ) -> None:
        """存储转换到缓冲区"""
        self.buffer.add(state, action, log_prob, reward, value, done)

    def update(self, last_value: float) -> Dict[str, float]:
        """
        执行 PPO 更新

        PPO 的特点是可以多次使用同一批数据进行更新，
        通过裁剪机制保证策略不会更新太多。

        Args:
            last_value: 最后状态的价值估计

        Returns:
            包含各项损失的字典
        """
        # 获取批次数据
        batch, returns, advantages = self.buffer.get_batch(
            last_value, self.device
        )

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 记录损失用于返回
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        update_count = 0

        batch_size = len(batch.states)
        mini_batch_size = self.config.mini_batch_size

        # 多轮 epoch 更新
        for epoch in range(self.config.n_epochs):
            # 随机打乱索引
            indices = np.random.permutation(batch_size)

            # Mini-batch 更新
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                # 提取 mini-batch 数据
                mb_states = batch.states[mb_indices]
                mb_actions = batch.actions[mb_indices]
                mb_old_log_probs = batch.log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # 计算新策略分布
                _, new_log_probs, entropy, values = \
                    self.network.get_action_and_value(mb_states, mb_actions)

                # 计算策略比率
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                # 近似 KL 散度
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                # 早停检查
                if (self.config.target_kl is not None
                        and approx_kl > 1.5 * self.config.target_kl):
                    break

                # PPO-Clip 目标
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = F.mse_loss(values, mb_returns)

                # 熵损失
                entropy_loss = -entropy.mean()

                # 总损失
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                # 累计统计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += approx_kl
                update_count += 1

            # 早停检查
            if (self.config.target_kl is not None
                    and approx_kl > 1.5 * self.config.target_kl):
                break

        # 清空缓冲区
        self.buffer.reset()

        # 返回平均损失
        if update_count > 0:
            return {
                "policy_loss": total_policy_loss / update_count,
                "value_loss": total_value_loss / update_count,
                "entropy": total_entropy / update_count,
                "approx_kl": total_kl / update_count,
                "n_updates": update_count
            }
        else:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "n_updates": 0
            }

    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config
        }, path)

    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


# =============================================================================
# 训练函数
# =============================================================================

def train_a2c(
    env_name: str = "CartPole-v1",
    num_episodes: int = 500,
    n_steps: int = 5,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Optional[A2CAgent], List[float]]:
    """
    训练 A2C 智能体

    A2C 使用 n-step 返回，每 n_steps 步进行一次更新。

    Args:
        env_name: Gymnasium 环境名称
        num_episodes: 训练回合数
        n_steps: 每次更新收集的步数
        seed: 随机种子
        verbose: 是否打印进度

    Returns:
        (agent, rewards_history): 训练后的智能体和奖励历史
    """
    if not HAS_GYM:
        print("错误: 需要安装 gymnasium")
        return None, []

    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"训练 A2C")
        print(f"环境: {env_name}")
        print(f"状态维度: {state_dim}, 动作数: {action_dim}")
        print(f"{'=' * 60}")

    # 创建配置和智能体
    config = A2CConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        n_steps=n_steps,
        learning_rate=7e-4,
        gamma=0.99,
        gae_lambda=0.95
    )

    agent = A2CAgent(config)

    # 训练
    rewards_history: List[float] = []
    best_avg = float("-inf")

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode if seed else None)
        episode_reward = 0.0
        done = False
        step_count = 0

        while not done:
            # 选择动作
            action, log_prob, value = agent.get_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储转换
            agent.store_transition(
                state, action, log_prob, reward, value, done
            )

            state = next_state
            episode_reward += reward
            step_count += 1

            # N-step 更新
            if step_count % n_steps == 0 or done:
                if done:
                    last_value = 0.0
                else:
                    _, _, last_value = agent.get_action(state)
                agent.update(last_value)

        rewards_history.append(episode_reward)

        # 打印进度
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            best_avg = max(best_avg, avg_reward)
            print(
                f"回合 {episode + 1:4d} | "
                f"平均奖励: {avg_reward:7.2f} | "
                f"最佳: {best_avg:7.2f}"
            )

    env.close()

    # 评估
    if verbose:
        print(f"\n最终评估 (A2C):")
        eval_rewards = evaluate_policy_agent(agent, env_name, num_episodes=10)
        print(f"评估奖励: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")

    return agent, rewards_history


def train_ppo(
    env_name: str = "CartPole-v1",
    total_timesteps: int = 100000,
    n_steps: int = 2048,
    n_epochs: int = 10,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Optional[PPOAgent], List[float]]:
    """
    训练 PPO 智能体

    PPO 收集固定数量的步数后进行多轮 epoch 更新。

    Args:
        env_name: Gymnasium 环境名称
        total_timesteps: 总训练步数
        n_steps: 每次更新收集的步数
        n_epochs: 每批数据训练轮数
        seed: 随机种子
        verbose: 是否打印进度

    Returns:
        (agent, rewards_history): 训练后的智能体和奖励历史
    """
    if not HAS_GYM:
        print("错误: 需要安装 gymnasium")
        return None, []

    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"训练 PPO")
        print(f"环境: {env_name}")
        print(f"状态维度: {state_dim}, 动作数: {action_dim}")
        print(f"总步数: {total_timesteps}")
        print(f"{'=' * 60}")

    # 创建配置和智能体
    config = PPOConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        n_steps=n_steps,
        n_epochs=n_epochs,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2
    )

    agent = PPOAgent(config)

    # 训练
    rewards_history: List[float] = []
    episode_rewards: List[float] = []
    best_avg = float("-inf")

    state, _ = env.reset(seed=seed if seed else None)
    episode_reward = 0.0
    total_steps = 0
    update_count = 0

    while total_steps < total_timesteps:
        # 收集 n_steps 步数据
        for _ in range(n_steps):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(
                state, action, log_prob, reward, value, done
            )

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                episode_rewards.append(episode_reward)
                rewards_history.append(episode_reward)
                episode_reward = 0.0
                state, _ = env.reset()

            if total_steps >= total_timesteps:
                break

        # 计算最后状态价值
        _, _, last_value = agent.get_action(state)

        # 更新
        loss_info = agent.update(last_value)
        update_count += 1

        # 打印进度
        if verbose and update_count % 10 == 0 and episode_rewards:
            avg_reward = np.mean(episode_rewards[-50:])
            best_avg = max(best_avg, avg_reward)
            print(
                f"步数 {total_steps:7d} | "
                f"平均奖励: {avg_reward:7.2f} | "
                f"最佳: {best_avg:7.2f} | "
                f"KL: {loss_info['approx_kl']:.4f}"
            )

    env.close()

    # 评估
    if verbose:
        print(f"\n最终评估 (PPO):")
        eval_rewards = evaluate_policy_agent(agent, env_name, num_episodes=10)
        print(f"评估奖励: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")

    return agent, rewards_history


def evaluate_policy_agent(
    agent,
    env_name: str,
    num_episodes: int = 10,
    render: bool = False
) -> List[float]:
    """
    评估策略梯度智能体

    Args:
        agent: A2C 或 PPO 智能体
        env_name: 环境名称
        num_episodes: 评估回合数
        render: 是否渲染

    Returns:
        各回合奖励列表
    """
    if not HAS_GYM:
        return []

    env = gym.make(env_name, render_mode="human" if render else None)
    rewards: List[float] = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action, _, _ = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)

    env.close()
    return rewards


def compare_algorithms(
    env_name: str = "CartPole-v1",
    seed: int = 42
) -> Dict[str, List[float]]:
    """
    比较 A2C 和 PPO 性能

    Args:
        env_name: 环境名称
        seed: 随机种子

    Returns:
        各算法的奖励历史
    """
    results: Dict[str, List[float]] = {}

    # A2C
    print("\n训练 A2C...")
    _, rewards_a2c = train_a2c(
        env_name=env_name,
        num_episodes=300,
        seed=seed
    )
    results["A2C"] = rewards_a2c

    # PPO
    print("\n训练 PPO...")
    _, rewards_ppo = train_ppo(
        env_name=env_name,
        total_timesteps=50000,
        n_steps=256,
        seed=seed
    )
    results["PPO"] = rewards_ppo

    # 绘图
    if HAS_MATPLOTLIB and results:
        plot_comparison(results, env_name)

    return results


def plot_comparison(
    results: Dict[str, List[float]],
    env_name: str,
    window_size: int = 20
) -> None:
    """绘制算法比较图"""
    plt.figure(figsize=(12, 6))

    for name, rewards in results.items():
        if len(rewards) >= window_size:
            smoothed = np.convolve(
                rewards,
                np.ones(window_size) / window_size,
                mode="valid"
            )
            plt.plot(smoothed, label=name, alpha=0.8)

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title(f"A2C vs PPO on {env_name}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("a2c_ppo_comparison.png", dpi=150)
    print("图像已保存: a2c_ppo_comparison.png")
    plt.close()


# =============================================================================
# 单元测试
# =============================================================================

def run_unit_tests() -> bool:
    """
    运行单元测试

    Returns:
        测试是否全部通过
    """
    print("\n" + "=" * 60)
    print("运行单元测试")
    print("=" * 60)

    all_passed = True

    # 测试 1: RolloutBuffer
    print("\n[测试 1] RolloutBuffer")
    try:
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        state = np.array([1.0, 2.0, 3.0, 4.0])

        for i in range(10):
            buffer.add(state, i % 2, -0.1 * i, float(i), 0.5, i == 9)

        assert len(buffer) == 10, f"缓冲区大小错误: {len(buffer)}"

        returns, advantages = buffer.compute_returns_and_advantages(0.0)
        assert returns.shape == (10,), f"回报形状错误: {returns.shape}"
        assert advantages.shape == (10,), f"优势形状错误: {advantages.shape}"

        buffer.reset()
        assert len(buffer) == 0, "清空后缓冲区应为空"

        print("  ✓ RolloutBuffer 测试通过")
    except Exception as e:
        print(f"  ✗ RolloutBuffer 测试失败: {e}")
        all_passed = False

    # 测试 2: ActorCriticNetwork
    print("\n[测试 2] ActorCriticNetwork")
    try:
        net = ActorCriticNetwork(state_dim=4, action_dim=2, hidden_dim=64)
        x = torch.randn(32, 4)

        logits, value = net(x)
        assert logits.shape == (32, 2), f"logits 形状错误: {logits.shape}"
        assert value.shape == (32, 1), f"value 形状错误: {value.shape}"

        action, log_prob, entropy, value = net.get_action_and_value(x)
        assert action.shape == (32,), f"动作形状错误"
        assert log_prob.shape == (32,), f"对数概率形状错误"
        assert entropy.shape == (32,), f"熵形状错误"
        assert value.shape == (32,), f"价值形状错误"

        print("  ✓ ActorCriticNetwork 测试通过")
    except Exception as e:
        print(f"  ✗ ActorCriticNetwork 测试失败: {e}")
        all_passed = False

    # 测试 3: A2CAgent
    print("\n[测试 3] A2CAgent")
    try:
        config = A2CConfig(state_dim=4, action_dim=2)
        agent = A2CAgent(config)

        state = np.random.randn(4).astype(np.float32)
        action, log_prob, value = agent.get_action(state)

        assert 0 <= action < 2, f"动作超出范围: {action}"
        assert isinstance(log_prob, float), "log_prob 应为 float"
        assert isinstance(value, float), "value 应为 float"

        # 测试更新
        for i in range(10):
            s = np.random.randn(4).astype(np.float32)
            a, lp, v = agent.get_action(s)
            agent.store_transition(s, a, lp, 1.0, v, i == 9)

        loss_info = agent.update(0.0)
        assert "policy_loss" in loss_info, "缺少 policy_loss"
        assert "value_loss" in loss_info, "缺少 value_loss"

        print("  ✓ A2CAgent 测试通过")
    except Exception as e:
        print(f"  ✗ A2CAgent 测试失败: {e}")
        all_passed = False

    # 测试 4: PPOAgent
    print("\n[测试 4] PPOAgent")
    try:
        config = PPOConfig(
            state_dim=4,
            action_dim=2,
            n_steps=64,
            mini_batch_size=32
        )
        agent = PPOAgent(config)

        state = np.random.randn(4).astype(np.float32)
        action, log_prob, value = agent.get_action(state)

        assert 0 <= action < 2, f"动作超出范围: {action}"

        # 测试更新
        for i in range(64):
            s = np.random.randn(4).astype(np.float32)
            a, lp, v = agent.get_action(s)
            agent.store_transition(s, a, lp, 1.0, v, i == 63)

        loss_info = agent.update(0.0)
        assert "policy_loss" in loss_info, "缺少 policy_loss"
        assert "approx_kl" in loss_info, "缺少 approx_kl"

        print("  ✓ PPOAgent 测试通过")
    except Exception as e:
        print(f"  ✗ PPOAgent 测试失败: {e}")
        all_passed = False

    # 测试 5: 环境交互
    print("\n[测试 5] 环境交互")
    if HAS_GYM:
        try:
            env = gym.make("CartPole-v1")
            config = A2CConfig(state_dim=4, action_dim=2)
            agent = A2CAgent(config)

            state, _ = env.reset()
            total_reward = 0.0

            for _ in range(100):
                action, log_prob, value = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store_transition(
                    state, action, log_prob, reward, value, done
                )
                total_reward += reward

                if done:
                    agent.update(0.0)
                    state, _ = env.reset()
                else:
                    state = next_state

            env.close()
            print("  ✓ 环境交互测试通过")
        except Exception as e:
            print(f"  ✗ 环境交互测试失败: {e}")
            all_passed = False
    else:
        print("  - 跳过（gymnasium 未安装）")

    # 测试 6: 模型保存/加载
    print("\n[测试 6] 模型保存/加载")
    try:
        import tempfile

        config = PPOConfig(state_dim=4, action_dim=2)
        agent = PPOAgent(config)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        agent.save(temp_path)

        agent2 = PPOAgent(config)
        agent2.load(temp_path)

        os.remove(temp_path)

        # 验证参数一致
        for p1, p2 in zip(
            agent.network.parameters(),
            agent2.network.parameters()
        ):
            assert torch.allclose(p1, p2), "参数不一致"

        print("  ✓ 模型保存/加载测试通过")
    except Exception as e:
        print(f"  ✗ 模型保存/加载测试失败: {e}")
        all_passed = False

    # 测试 7: GAE 计算正确性
    print("\n[测试 7] GAE 计算验证")
    try:
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)

        # 简单测试用例：3步轨迹
        states = [np.array([1.0]) for _ in range(3)]
        for i, s in enumerate(states):
            buffer.add(
                state=s,
                action=0,
                log_prob=0.0,
                reward=1.0,
                value=0.5,
                done=(i == 2)
            )

        returns, advantages = buffer.compute_returns_and_advantages(0.0)

        # 手动验证 GAE
        gamma, lam = 0.99, 0.95
        v = [0.5, 0.5, 0.5, 0.0]  # values + last_value
        r = [1.0, 1.0, 1.0]
        d = [0, 0, 1]

        expected_adv = np.zeros(3)
        gae = 0.0
        for t in reversed(range(3)):
            delta = r[t] + gamma * v[t+1] * (1 - d[t]) - v[t]
            gae = delta + gamma * lam * (1 - d[t]) * gae
            expected_adv[t] = gae

        assert np.allclose(
            advantages.numpy(), expected_adv, atol=1e-5
        ), f"GAE 计算不正确: {advantages.numpy()} vs {expected_adv}"

        print("  ✓ GAE 计算验证通过")
    except Exception as e:
        print(f"  ✗ GAE 计算验证失败: {e}")
        all_passed = False

    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败，请检查错误信息")
    print("=" * 60)

    return all_passed


# =============================================================================
# 主程序入口
# =============================================================================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Actor-Critic 与 PPO 算法实现"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["a2c", "ppo", "compare"],
        default="ppo",
        help="选择算法"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="运行单元测试"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="A2C 训练回合数"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="PPO 总训练步数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    args = parser.parse_args()

    if args.test:
        run_unit_tests()
    elif args.algo == "a2c":
        agent, rewards = train_a2c(
            num_episodes=args.episodes,
            seed=args.seed
        )
        if rewards and HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 5))
            window = 20
            smoothed = np.convolve(
                rewards, np.ones(window)/window, mode="valid"
            )
            plt.plot(smoothed)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("A2C Training on CartPole-v1")
            plt.grid(True, alpha=0.3)
            plt.savefig("a2c_training.png", dpi=150)
            plt.close()
    elif args.algo == "ppo":
        agent, rewards = train_ppo(
            total_timesteps=args.timesteps,
            seed=args.seed
        )
        if rewards and HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 5))
            window = 20
            if len(rewards) >= window:
                smoothed = np.convolve(
                    rewards, np.ones(window)/window, mode="valid"
                )
                plt.plot(smoothed)
            else:
                plt.plot(rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("PPO Training on CartPole-v1")
            plt.grid(True, alpha=0.3)
            plt.savefig("ppo_training.png", dpi=150)
            plt.close()
    elif args.algo == "compare":
        compare_algorithms(seed=args.seed)
    else:
        run_unit_tests()


if __name__ == "__main__":
    main()
