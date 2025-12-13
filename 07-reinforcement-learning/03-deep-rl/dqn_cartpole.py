#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度 Q 网络 (DQN) 实现 - CartPole 环境

本模块实现深度强化学习中的核心价值函数方法：
- 基础 DQN (Mnih et al., 2013)
- Double DQN (van Hasselt et al., 2016)
- Dueling DQN (Wang et al., 2016)
- 优先级经验回放 (Schaul et al., 2015)

核心技术要点：
1. 经验回放 (Experience Replay)：打破样本间的时序相关性
2. 目标网络 (Target Network)：稳定训练过程中的目标值估计
3. 双网络解耦 (Double DQN)：缓解 Q 值过估计问题
4. 价值分解 (Dueling DQN)：分离状态价值 V(s) 和优势函数 A(s,a)
5. 优先级采样 (PER)：优先学习 TD 误差大的样本

数学基础：
- Q-Learning 更新: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
- DQN 损失函数: L(θ) = E[(r + γ max_a' Q(s',a';θ⁻) - Q(s,a;θ))²]
- Double DQN 目标: y = r + γ Q(s', argmax_a' Q(s',a';θ); θ⁻)

参考文献：
[1] Mnih et al., "Playing Atari with Deep Reinforcement Learning", 2013
[2] van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", 2016
[3] Wang et al., "Dueling Network Architectures for Deep RL", 2016
[4] Schaul et al., "Prioritized Experience Replay", 2015

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
import sys
import random
import warnings
from typing import Tuple, List, Optional, Dict, Any, Union
from collections import deque
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 尝试导入绘图和环境库
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib 未安装，绘图功能将不可用")

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    warnings.warn("gymnasium 未安装，无法运行环境交互")


# =============================================================================
# 经验回放模块
# =============================================================================

@dataclass
class Transition:
    """
    单步交互转换数据结构

    存储智能体与环境交互的完整信息，用于经验回放采样。
    使用 dataclass 确保数据不可变性和类型安全。

    Attributes:
        state: 当前状态观测值
        action: 执行的动作
        reward: 获得的即时奖励
        next_state: 转换后的下一状态
        done: 回合是否结束标志
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    均匀采样经验回放缓冲区

    经验回放是 DQN 的核心技术之一，解决以下问题：
    1. 打破样本间的时序相关性，满足 i.i.d. 假设
    2. 提高数据利用效率，每个样本可多次用于训练
    3. 平滑数据分布，避免灾难性遗忘

    实现细节：
    - 使用 deque 作为底层存储，自动处理容量溢出
    - 均匀随机采样，每个样本被选中概率相等
    - 批量返回 NumPy 数组，便于 PyTorch 张量转换

    Attributes:
        capacity: 缓冲区最大容量
        buffer: 存储转换数据的双端队列
    """

    def __init__(self, capacity: int = 100000) -> None:
        """
        初始化经验回放缓冲区

        Args:
            capacity: 缓冲区最大容量，超出后自动丢弃最旧数据
        """
        if capacity <= 0:
            raise ValueError(f"缓冲区容量必须为正整数，当前: {capacity}")

        self._capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        """返回缓冲区最大容量"""
        return self._capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        存储单步交互经验

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 即时奖励
            next_state: 下一状态
            done: 回合终止标志
        """
        transition = Transition(state, action, reward, next_state, done)
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        均匀随机采样一个批次的经验

        Args:
            batch_size: 采样批次大小

        Returns:
            (states, actions, rewards, next_states, dones) 元组

        Raises:
            ValueError: 当缓冲区大小不足时抛出
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"采样大小 {batch_size} 超出缓冲区当前大小 {len(self._buffer)}"
            )

        batch = random.sample(self._buffer, batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """返回当前存储的经验数量"""
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """检查是否有足够的样本进行采样"""
        return len(self._buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区 (Prioritized Experience Replay, PER)

    核心思想：优先采样 TD 误差大的经验，提高学习效率

    数学原理：
    - 采样概率: P(i) = p_i^α / Σ_k p_k^α
    - 优先级: p_i = |δ_i| + ε (δ_i 为 TD 误差)
    - 重要性采样权重: w_i = (N · P(i))^(-β)

    参数说明：
    - α: 优先级指数，α=0 退化为均匀采样，α=1 完全按优先级采样
    - β: 重要性采样指数，用于修正优先级采样引入的偏差
    - ε: 防止优先级为零的小常数

    实现采用 Sum-Tree 数据结构，实现 O(log n) 的采样和更新复杂度。
    此处使用简化版线性实现，适用于中小规模缓冲区。

    Attributes:
        capacity: 缓冲区最大容量
        alpha: 优先级指数
        beta: 重要性采样指数（可随训练逐渐增加到1）
        epsilon: 防止零优先级的小常数
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6
    ) -> None:
        """
        初始化优先级经验回放缓冲区

        Args:
            capacity: 最大容量
            alpha: 优先级指数，范围 [0, 1]
            beta: 重要性采样指数初始值，范围 [0, 1]
            epsilon: 优先级下界
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha 必须在 [0, 1] 范围内，当前: {alpha}")
        if not 0 <= beta <= 1:
            raise ValueError(f"beta 必须在 [0, 1] 范围内，当前: {beta}")

        self._capacity = capacity
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon

        self._buffer: List[Transition] = []
        self._priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._position: int = 0
        self._max_priority: float = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        存储经验，新经验赋予最大优先级

        新经验默认获得最大优先级，确保至少被采样一次

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 即时奖励
            next_state: 下一状态
            done: 回合终止标志
        """
        transition = Transition(state, action, reward, next_state, done)

        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._position] = transition

        self._priorities[self._position] = self._max_priority
        self._position = (self._position + 1) % self._capacity

    def sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """
        按优先级采样经验批次

        Args:
            batch_size: 采样大小

        Returns:
            (states, actions, rewards, next_states, dones, indices, weights) 元组
            - indices: 采样索引，用于更新优先级
            - weights: 重要性采样权重，用于加权损失
        """
        buffer_len = len(self._buffer)
        if batch_size > buffer_len:
            raise ValueError(f"采样大小超出缓冲区: {batch_size} > {buffer_len}")

        # 计算采样概率
        priorities = self._priorities[:buffer_len]
        probs = priorities ** self._alpha
        probs = probs / probs.sum()

        # 按概率采样
        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=False)

        # 计算重要性采样权重
        weights = (buffer_len * probs[indices]) ** (-self._beta)
        weights = weights / weights.max()  # 归一化

        # 提取批次数据
        batch = [self._buffer[i] for i in indices]
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return (states, actions, rewards, next_states, dones,
                indices.astype(np.int64), weights.astype(np.float32))

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray
    ) -> None:
        """
        更新指定经验的优先级

        Args:
            indices: 需要更新的经验索引
            td_errors: 对应的 TD 误差
        """
        priorities = np.abs(td_errors) + self._epsilon
        for idx, priority in zip(indices, priorities):
            self._priorities[idx] = priority

        self._max_priority = max(self._max_priority, priorities.max())

    def update_beta(self, beta: float) -> None:
        """更新重要性采样指数"""
        self._beta = min(1.0, beta)

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self._buffer) >= batch_size


# =============================================================================
# 神经网络模块
# =============================================================================

def _init_weights(module: nn.Module) -> None:
    """
    网络权重初始化

    使用正交初始化（Orthogonal Initialization），
    对于深度强化学习通常比默认初始化效果更好。

    Args:
        module: 需要初始化的网络模块
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DQNNetwork(nn.Module):
    """
    基础 DQN 网络架构

    使用多层感知机（MLP）作为函数逼近器，
    将状态映射到各动作的 Q 值。

    网络结构: State → FC → ReLU → FC → ReLU → FC → Q-values

    Attributes:
        input_dim: 输入状态维度
        output_dim: 输出动作数量（即 Q 值数量）
        hidden_dim: 隐藏层神经元数量
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128
    ) -> None:
        """
        初始化 DQN 网络

        Args:
            input_dim: 状态空间维度
            output_dim: 动作空间大小
            hidden_dim: 隐藏层单元数
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        # 应用权重初始化
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入状态张量，形状 (batch_size, input_dim)

        Returns:
            Q 值张量，形状 (batch_size, output_dim)
        """
        return self.net(x)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN 网络架构 (Wang et al., 2016)

    核心思想：将 Q(s,a) 分解为状态价值 V(s) 和优势函数 A(s,a)

    公式: Q(s,a) = V(s) + A(s,a) - mean_a'(A(s,a'))

    优势：
    1. 状态价值流可以独立学习状态好坏，无需遍历所有动作
    2. 优势流专注于比较不同动作的相对好坏
    3. 在动作影响不大的状态下学习效率更高

    注意：需要减去优势函数均值以确保可识别性，
    即确保 A 和 V 的分解唯一。

    网络结构:
        State → 共享特征层 → 价值流 → V(s)
                         → 优势流 → A(s,a)

        Q(s,a) = V(s) + (A(s,a) - mean(A))
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128
    ) -> None:
        """
        初始化 Dueling DQN 网络

        Args:
            input_dim: 状态空间维度
            output_dim: 动作空间大小
            hidden_dim: 隐藏层单元数
        """
        super().__init__()

        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 状态价值流 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # 优势函数流 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，合并价值流和优势流

        Args:
            x: 输入状态张量，形状 (batch_size, input_dim)

        Returns:
            Q 值张量，形状 (batch_size, output_dim)
        """
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # 合并公式: Q = V + (A - mean(A))
        # 减去均值确保 A 和 V 分解的唯一性
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))

        return q_values


# =============================================================================
# DQN 智能体
# =============================================================================

@dataclass
class DQNConfig:
    """
    DQN 智能体超参数配置

    集中管理所有超参数，便于实验调优和配置管理。

    Attributes:
        state_dim: 状态空间维度
        action_dim: 动作空间大小
        hidden_dim: 隐藏层单元数
        learning_rate: Adam 优化器学习率
        gamma: 折扣因子，控制对未来奖励的重视程度
        epsilon_start: ε-greedy 初始探索率
        epsilon_end: ε-greedy 最终探索率
        epsilon_decay: 探索率衰减系数（每步乘以此值）
        buffer_size: 经验回放缓冲区大小
        batch_size: 训练批次大小
        target_update_freq: 目标网络更新频率（步数）
        double_dqn: 是否使用 Double DQN
        dueling: 是否使用 Dueling DQN 架构
        prioritized: 是否使用优先级经验回放
        per_alpha: PER 优先级指数
        per_beta_start: PER 重要性采样初始值
        per_beta_frames: PER beta 退火帧数
        grad_clip: 梯度裁剪阈值
        device: 计算设备
    """
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 100
    double_dqn: bool = False
    dueling: bool = False
    prioritized: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100000
    grad_clip: float = 10.0
    device: str = "auto"


class DQNAgent:
    """
    DQN 智能体实现

    支持的算法变体：
    1. 标准 DQN：使用目标网络和经验回放
    2. Double DQN：解耦动作选择和评估，缓解过估计
    3. Dueling DQN：分解价值函数为 V(s) 和 A(s,a)
    4. PER-DQN：优先级经验回放

    核心方法：
    - get_action(): ε-greedy 策略选择动作
    - update(): 存储经验并执行梯度更新
    - sync_target_network(): 同步目标网络参数

    使用示例:
        >>> config = DQNConfig(state_dim=4, action_dim=2)
        >>> agent = DQNAgent(config)
        >>> action = agent.get_action(state)
        >>> loss = agent.update(state, action, reward, next_state, done)
    """

    def __init__(self, config: DQNConfig) -> None:
        """
        初始化 DQN 智能体

        Args:
            config: 超参数配置对象
        """
        self.config = config
        self._setup_device()
        self._setup_networks()
        self._setup_replay_buffer()

        # 训练状态
        self.epsilon = config.epsilon_start
        self.update_count = 0
        self.frame_count = 0

    def _setup_device(self) -> None:
        """配置计算设备"""
        if self.config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(self.config.device)

    def _setup_networks(self) -> None:
        """初始化神经网络和优化器"""
        # 选择网络架构
        NetworkClass = (
            DuelingDQNNetwork if self.config.dueling else DQNNetwork
        )

        self.q_network = NetworkClass(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)

        self.target_network = NetworkClass(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)

        # 同步目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 目标网络不需要梯度

        # 优化器
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )

    def _setup_replay_buffer(self) -> None:
        """初始化经验回放缓冲区"""
        if self.config.prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.config.buffer_size,
                alpha=self.config.per_alpha,
                beta=self.config.per_beta_start
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=self.config.buffer_size
            )

    def get_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> int:
        """
        使用 ε-greedy 策略选择动作

        ε-greedy 策略平衡探索与利用：
        - 以概率 ε 随机选择动作（探索）
        - 以概率 1-ε 选择当前最优动作（利用）

        Args:
            state: 当前状态观测
            training: 是否处于训练模式

        Returns:
            选择的动作索引
        """
        # 训练时使用探索
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)

        # 贪婪选择
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.argmax(dim=1).item()

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[float]:
        """
        存储经验并执行一步梯度更新

        更新步骤：
        1. 存储经验到回放缓冲区
        2. 检查缓冲区是否有足够样本
        3. 采样批次并计算 TD 目标
        4. 计算损失并反向传播
        5. 定期同步目标网络

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 回合是否结束

        Returns:
            本次更新的损失值，若未进行更新则返回 None
        """
        # 存储经验
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.frame_count += 1

        # 检查是否可以开始训练
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return None

        # 采样批次
        if self.config.prioritized:
            batch = self.replay_buffer.sample(self.config.batch_size)
            states, actions, rewards, next_states, dones, indices, weights = batch
            weights_tensor = torch.FloatTensor(weights).to(self.device)

            # 更新 PER 的 beta 值
            fraction = min(
                1.0,
                self.frame_count / self.config.per_beta_frames
            )
            beta = (
                self.config.per_beta_start +
                fraction * (1.0 - self.config.per_beta_start)
            )
            self.replay_buffer.update_beta(beta)
        else:
            states, actions, rewards, next_states, dones = (
                self.replay_buffer.sample(self.config.batch_size)
            )
            weights_tensor = None

        # 转换为张量
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 计算当前 Q 值
        current_q_values = self.q_network(states_t)
        current_q = current_q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN：用在线网络选择动作，目标网络评估
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # 标准 DQN
                next_q = self.target_network(next_states_t).max(dim=1)[0]

            # TD 目标
            target_q = rewards_t + self.config.gamma * next_q * (1 - dones_t)

        # 计算 TD 误差
        td_errors = current_q - target_q

        # 计算损失
        if self.config.prioritized:
            # PER：加权 MSE 损失
            loss = (weights_tensor * (td_errors ** 2)).mean()
            # 更新优先级
            self.replay_buffer.update_priorities(
                indices, td_errors.detach().cpu().numpy()
            )
        else:
            # 标准 MSE 损失
            loss = F.mse_loss(current_q, target_q)

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.config.grad_clip
        )

        self.optimizer.step()

        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.config.target_update_freq == 0:
            self.sync_target_network()

        return loss.item()

    def sync_target_network(self) -> None:
        """
        同步目标网络参数

        将在线网络的参数复制到目标网络。
        这是硬更新（hard update），也可以实现软更新（soft update）:
        θ_target ← τ * θ_online + (1 - τ) * θ_target
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """
        衰减探索率

        使用指数衰减策略，逐步减少探索，增加利用。
        """
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

    def save(self, path: str) -> None:
        """
        保存模型检查点

        Args:
            path: 保存路径
        """
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "frame_count": self.frame_count,
            "config": self.config
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        加载模型检查点

        Args:
            path: 检查点文件路径
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.update_count = checkpoint["update_count"]
        self.frame_count = checkpoint["frame_count"]


# =============================================================================
# 训练与评估
# =============================================================================

def train_dqn(
    env_name: str = "CartPole-v1",
    num_episodes: int = 500,
    double_dqn: bool = False,
    dueling: bool = False,
    prioritized: bool = False,
    render: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Optional[DQNAgent], List[float]]:
    """
    训练 DQN 智能体

    Args:
        env_name: Gymnasium 环境名称
        num_episodes: 训练回合数
        double_dqn: 是否使用 Double DQN
        dueling: 是否使用 Dueling DQN
        prioritized: 是否使用优先级经验回放
        render: 是否渲染环境
        seed: 随机种子
        verbose: 是否打印训练进度

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
    env = gym.make(env_name, render_mode="human" if render else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 构建算法名称
    algo_parts = []
    if double_dqn:
        algo_parts.append("Double")
    if dueling:
        algo_parts.append("Dueling")
    if prioritized:
        algo_parts.append("PER")
    algo_parts.append("DQN")
    algo_name = " ".join(algo_parts)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"训练 {algo_name}")
        print(f"环境: {env_name}")
        print(f"状态维度: {state_dim}, 动作数: {action_dim}")
        print(f"{'=' * 60}")

    # 创建配置和智能体
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=100,
        double_dqn=double_dqn,
        dueling=dueling,
        prioritized=prioritized
    )

    agent = DQNAgent(config)

    # 训练循环
    rewards_history: List[float] = []
    best_avg_reward = float("-inf")

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode if seed else None)
        episode_reward = 0.0
        done = False

        while not done:
            # 选择动作
            action = agent.get_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 更新智能体
            agent.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

        # 衰减探索率
        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        # 打印进度
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            best_avg_reward = max(best_avg_reward, avg_reward)
            print(
                f"回合 {episode + 1:4d} | "
                f"平均奖励: {avg_reward:7.2f} | "
                f"最佳: {best_avg_reward:7.2f} | "
                f"ε: {agent.epsilon:.3f}"
            )

    env.close()

    # 最终评估
    if verbose:
        print(f"\n最终评估 ({algo_name}):")
        eval_rewards = evaluate_agent(agent, env_name, num_episodes=10)
        print(f"评估奖励: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")

    return agent, rewards_history


def evaluate_agent(
    agent: DQNAgent,
    env_name: str,
    num_episodes: int = 10,
    render: bool = False
) -> List[float]:
    """
    评估训练好的智能体

    Args:
        agent: DQN 智能体
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
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)

    env.close()
    return rewards


def compare_algorithms(
    env_name: str = "CartPole-v1",
    num_episodes: int = 300,
    seed: int = 42
) -> Dict[str, List[float]]:
    """
    比较不同 DQN 变体的性能

    Args:
        env_name: 环境名称
        num_episodes: 每种算法的训练回合数
        seed: 随机种子

    Returns:
        各算法的奖励历史字典
    """
    if not HAS_GYM:
        print("错误: 需要安装 gymnasium")
        return {}

    results: Dict[str, List[float]] = {}

    algorithms = [
        ("DQN", False, False, False),
        ("Double DQN", True, False, False),
        ("Dueling DQN", False, True, False),
        ("Double Dueling DQN", True, True, False),
    ]

    for name, double, dueling, prioritized in algorithms:
        print(f"\n训练 {name}...")
        _, rewards = train_dqn(
            env_name=env_name,
            num_episodes=num_episodes,
            double_dqn=double,
            dueling=dueling,
            prioritized=prioritized,
            seed=seed,
            verbose=True
        )
        results[name] = rewards

    # 绘制比较图
    if HAS_MATPLOTLIB and results:
        plot_comparison(results, env_name)

    return results


def plot_comparison(
    results: Dict[str, List[float]],
    env_name: str,
    window_size: int = 20
) -> None:
    """
    绘制算法比较图

    Args:
        results: 各算法的奖励历史
        env_name: 环境名称
        window_size: 滑动平均窗口大小
    """
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10.colors

    for idx, (name, rewards) in enumerate(results.items()):
        if len(rewards) >= window_size:
            smoothed = np.convolve(
                rewards,
                np.ones(window_size) / window_size,
                mode="valid"
            )
            plt.plot(
                smoothed,
                label=name,
                alpha=0.8,
                color=colors[idx % len(colors)]
            )

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title(f"DQN Variants Comparison on {env_name}", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = "dqn_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图像已保存: {save_path}")
    plt.close()


def plot_training_curve(
    rewards: List[float],
    title: str = "Training Curve",
    window_size: int = 20
) -> None:
    """
    绘制单个训练曲线

    Args:
        rewards: 奖励历史
        title: 图表标题
        window_size: 滑动平均窗口
    """
    if not HAS_MATPLOTLIB:
        print("警告: matplotlib 未安装，无法绘图")
        return

    plt.figure(figsize=(10, 5))

    # 原始数据（透明）
    plt.plot(rewards, alpha=0.3, color="blue", label="Raw")

    # 滑动平均
    if len(rewards) >= window_size:
        smoothed = np.convolve(
            rewards,
            np.ones(window_size) / window_size,
            mode="valid"
        )
        plt.plot(smoothed, color="blue", linewidth=2, label="Smoothed")

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = "dqn_training.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图像已保存: {save_path}")
    plt.close()


# =============================================================================
# 单元测试
# =============================================================================

def run_unit_tests() -> bool:
    """
    运行单元测试

    测试各个组件的基本功能，确保代码正确性。

    Returns:
        测试是否全部通过
    """
    print("\n" + "=" * 60)
    print("运行单元测试")
    print("=" * 60)

    all_passed = True

    # 测试 1: ReplayBuffer
    print("\n[测试 1] ReplayBuffer")
    try:
        buffer = ReplayBuffer(capacity=100)
        state = np.array([1.0, 2.0, 3.0, 4.0])
        for i in range(50):
            buffer.push(state, i % 2, float(i), state, False)

        assert len(buffer) == 50, f"缓冲区大小错误: {len(buffer)}"
        assert buffer.is_ready(32), "缓冲区应该准备就绪"

        batch = buffer.sample(32)
        assert len(batch) == 5, f"批次元组大小错误: {len(batch)}"
        assert batch[0].shape == (32, 4), f"状态形状错误: {batch[0].shape}"

        print("  ✓ ReplayBuffer 测试通过")
    except Exception as e:
        print(f"  ✗ ReplayBuffer 测试失败: {e}")
        all_passed = False

    # 测试 2: PrioritizedReplayBuffer
    print("\n[测试 2] PrioritizedReplayBuffer")
    try:
        per_buffer = PrioritizedReplayBuffer(capacity=100)
        state = np.array([1.0, 2.0, 3.0, 4.0])
        for i in range(50):
            per_buffer.push(state, i % 2, float(i), state, False)

        assert len(per_buffer) == 50, f"PER 缓冲区大小错误"

        batch = per_buffer.sample(32)
        assert len(batch) == 7, f"PER 批次元组大小错误: {len(batch)}"

        indices, weights = batch[5], batch[6]
        per_buffer.update_priorities(indices, np.random.randn(32))

        print("  ✓ PrioritizedReplayBuffer 测试通过")
    except Exception as e:
        print(f"  ✗ PrioritizedReplayBuffer 测试失败: {e}")
        all_passed = False

    # 测试 3: DQNNetwork
    print("\n[测试 3] DQNNetwork")
    try:
        net = DQNNetwork(input_dim=4, output_dim=2, hidden_dim=64)
        x = torch.randn(32, 4)
        out = net(x)

        assert out.shape == (32, 2), f"网络输出形状错误: {out.shape}"
        assert not torch.isnan(out).any(), "输出包含 NaN"

        print("  ✓ DQNNetwork 测试通过")
    except Exception as e:
        print(f"  ✗ DQNNetwork 测试失败: {e}")
        all_passed = False

    # 测试 4: DuelingDQNNetwork
    print("\n[测试 4] DuelingDQNNetwork")
    try:
        net = DuelingDQNNetwork(input_dim=4, output_dim=2, hidden_dim=64)
        x = torch.randn(32, 4)
        out = net(x)

        assert out.shape == (32, 2), f"Dueling 网络输出形状错误: {out.shape}"
        assert not torch.isnan(out).any(), "输出包含 NaN"

        print("  ✓ DuelingDQNNetwork 测试通过")
    except Exception as e:
        print(f"  ✗ DuelingDQNNetwork 测试失败: {e}")
        all_passed = False

    # 测试 5: DQNAgent 基本功能
    print("\n[测试 5] DQNAgent")
    try:
        config = DQNConfig(state_dim=4, action_dim=2, batch_size=32)
        agent = DQNAgent(config)

        state = np.random.randn(4).astype(np.float32)
        action = agent.get_action(state, training=True)

        assert 0 <= action < 2, f"动作超出范围: {action}"

        # 填充缓冲区并测试更新
        for _ in range(100):
            s = np.random.randn(4).astype(np.float32)
            a = random.randint(0, 1)
            r = random.random()
            ns = np.random.randn(4).astype(np.float32)
            d = random.random() > 0.9
            agent.update(s, a, r, ns, d)

        # 检查是否能正常获取损失
        loss = agent.update(state, 0, 1.0, state, False)
        assert loss is not None, "损失不应为 None"

        print("  ✓ DQNAgent 测试通过")
    except Exception as e:
        print(f"  ✗ DQNAgent 测试失败: {e}")
        all_passed = False

    # 测试 6: Double DQN
    print("\n[测试 6] Double DQN Agent")
    try:
        config = DQNConfig(
            state_dim=4, action_dim=2, batch_size=32, double_dqn=True
        )
        agent = DQNAgent(config)

        for _ in range(100):
            s = np.random.randn(4).astype(np.float32)
            agent.update(s, 0, 1.0, s, False)

        print("  ✓ Double DQN 测试通过")
    except Exception as e:
        print(f"  ✗ Double DQN 测试失败: {e}")
        all_passed = False

    # 测试 7: Dueling DQN
    print("\n[测试 7] Dueling DQN Agent")
    try:
        config = DQNConfig(
            state_dim=4, action_dim=2, batch_size=32, dueling=True
        )
        agent = DQNAgent(config)

        for _ in range(100):
            s = np.random.randn(4).astype(np.float32)
            agent.update(s, 0, 1.0, s, False)

        print("  ✓ Dueling DQN 测试通过")
    except Exception as e:
        print(f"  ✗ Dueling DQN 测试失败: {e}")
        all_passed = False

    # 测试 8: PER DQN
    print("\n[测试 8] PER DQN Agent")
    try:
        config = DQNConfig(
            state_dim=4, action_dim=2, batch_size=32, prioritized=True
        )
        agent = DQNAgent(config)

        for _ in range(100):
            s = np.random.randn(4).astype(np.float32)
            agent.update(s, 0, 1.0, s, False)

        print("  ✓ PER DQN 测试通过")
    except Exception as e:
        print(f"  ✗ PER DQN 测试失败: {e}")
        all_passed = False

    # 测试 9: 环境交互
    print("\n[测试 9] 环境交互")
    if HAS_GYM:
        try:
            env = gym.make("CartPole-v1")
            config = DQNConfig(state_dim=4, action_dim=2, batch_size=32)
            agent = DQNAgent(config)

            state, _ = env.reset()
            for _ in range(100):
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(state, action, reward, next_state, terminated or truncated)
                if terminated or truncated:
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

    # 测试 10: 模型保存/加载
    print("\n[测试 10] 模型保存/加载")
    try:
        import tempfile
        config = DQNConfig(state_dim=4, action_dim=2)
        agent = DQNAgent(config)

        # 保存
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        agent.save(temp_path)

        # 加载
        agent2 = DQNAgent(config)
        agent2.load(temp_path)

        # 清理
        os.remove(temp_path)

        # 验证参数一致
        for p1, p2 in zip(
            agent.q_network.parameters(),
            agent2.q_network.parameters()
        ):
            assert torch.allclose(p1, p2), "参数加载不一致"

        print("  ✓ 模型保存/加载测试通过")
    except Exception as e:
        print(f"  ✗ 模型保存/加载测试失败: {e}")
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
    """
    主函数

    支持命令行参数或直接运行：
    1. 无参数：运行单元测试
    2. --train：运行完整训练
    3. --compare：比较不同算法
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="DQN 深度强化学习实现"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="运行训练"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="比较不同算法变体"
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
        help="训练回合数"
    )
    parser.add_argument(
        "--double",
        action="store_true",
        help="使用 Double DQN"
    )
    parser.add_argument(
        "--dueling",
        action="store_true",
        help="使用 Dueling DQN"
    )
    parser.add_argument(
        "--per",
        action="store_true",
        help="使用优先级经验回放"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    args = parser.parse_args()

    # 根据参数执行相应功能
    if args.compare:
        compare_algorithms(
            num_episodes=args.episodes,
            seed=args.seed
        )
    elif args.train:
        agent, rewards = train_dqn(
            num_episodes=args.episodes,
            double_dqn=args.double,
            dueling=args.dueling,
            prioritized=args.per,
            seed=args.seed
        )

        if rewards and HAS_MATPLOTLIB:
            algo_name = "DQN"
            if args.double:
                algo_name = "Double " + algo_name
            if args.dueling:
                algo_name = "Dueling " + algo_name
            plot_training_curve(
                rewards,
                title=f"{algo_name} Training on CartPole-v1"
            )
    else:
        # 默认运行测试
        run_unit_tests()


if __name__ == "__main__":
    main()
