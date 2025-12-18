"""
Replay Buffers Module.

This module provides experience replay buffer implementations for DQN training:
    - ReplayBuffer: Uniform random sampling (O(1) push, O(B) sample)
    - PrioritizedReplayBuffer: TD-error weighted sampling with SumTree
    - NStepReplayBuffer: Multi-step returns for faster credit assignment

Core Idea (核心思想)
====================
经验回放是DQN的核心创新之一，解决了两个关键问题：

1. **打破时序相关性**: 连续采样的样本高度相关，违反SGD的i.i.d.假设。
   随机采样打破时序关联，提供更稳定的梯度估计。

2. **提高数据效率**: 每个经验可被多次使用，而非"用后即弃"。
   这对于数据收集成本高的RL场景尤为重要。

Mathematical Foundation (数学基础)
==================================
Uniform sampling:
    P(i) = 1/|D|, ∀i ∈ D

Prioritized sampling:
    P(i) = p_i^α / Σ_k p_k^α
    where p_i = |δ_i| + ε (TD error + small constant)

Importance sampling correction:
    w_i = (1 / (N · P(i)))^β / max_j w_j

Example:
    >>> from buffers import ReplayBuffer, PrioritizedReplayBuffer
    >>> buffer = ReplayBuffer(capacity=10000)
    >>> buffer.push(state, action, reward, next_state, done)
    >>> states, actions, rewards, next_states, dones = buffer.sample(64)

References:
    [1] Mnih et al. (2015). Human-level control through deep RL.
    [2] Schaul et al. (2016). Prioritized Experience Replay.
    [3] Sutton (1988). Learning to predict by temporal differences.
"""

from buffers.base import ReplayBuffer
from buffers.sum_tree import SumTree
from buffers.prioritized import PrioritizedReplayBuffer
from buffers.n_step import NStepReplayBuffer

__all__ = [
    "ReplayBuffer",
    "SumTree",
    "PrioritizedReplayBuffer",
    "NStepReplayBuffer",
]
