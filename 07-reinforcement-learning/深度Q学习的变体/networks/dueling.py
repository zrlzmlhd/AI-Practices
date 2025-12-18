"""
Dueling Network Architecture.

This module implements the Dueling DQN architecture (Wang et al., 2016).

Core Idea (核心思想)
====================
将Q函数分解为状态价值V(s)和动作优势A(s,a)：

    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

这种分解使网络能够独立学习状态的价值，提高泛化能力。

Mathematical Foundation (数学基础)
==================================
Decomposition:
    Q(s, a) = V(s) + A(s, a) - (1/|A|) Σ_a' A(s, a')

Identifiability constraint:
    Σ_a A(s, a) = 0  =>  V(s) = max_a Q(s, a)

Benefits:
    - Learn state values without evaluating every action
    - Better generalization in states where action choice matters less

References:
    Wang, Z. et al. (2016). Dueling Network Architectures for Deep
    Reinforcement Learning. ICML.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from networks.base import init_weights


class DuelingNetwork(nn.Module):
    """
    Dueling DQN architecture (Wang et al., 2016).

    Core Idea (核心思想)
    --------------------
    将Q函数分解为**状态价值**和**动作优势**：

    .. math::
        Q(s, a) = V(s) + A(s, a) - \\frac{1}{|\\mathcal{A}|} \\sum_{a'} A(s, a')

    Intuition (直觉理解)
    --------------------
    - **V(s)**: 这个状态有多好？（与动作无关）
    - **A(s,a)**: 动作a比平均动作好多少？

    Identifiability (可辨识性)
    -------------------------
    **问题**: Q = V + A 有无穷多分解方式

    **解决方案**: 强制 Σ_a A(s,a) = 0，通过减去均值实现

    Benefits (优势)
    ---------------
    1. 更好的泛化：可以在不评估每个动作的情况下学习状态价值
    2. 对于动作选择不重要的状态，更快学习
    3. 提高样本效率

    Network Architecture (网络架构)
    ------------------------------
    ::

        Input → Shared Layers → Value Stream  → V(s) [1维]
                             ↘
                               Advantage Stream → A(s,a) [|A|维]
                             ↘
                               聚合 → Q(s,a) = V + (A - mean(A))

    Parameters
    ----------
    state_dim : int
        Dimension of state space
    action_dim : int
        Number of discrete actions
    hidden_dim : int, default=128
        Hidden layer size for both streams

    Examples
    --------
    >>> net = DuelingNetwork(state_dim=4, action_dim=2)
    >>> q_values = net(torch.randn(1, 4))
    >>> q_values.shape
    torch.Size([1, 2])

    References
    ----------
    Wang, Z. et al. (2016). Dueling Network Architectures for Deep
    Reinforcement Learning. ICML.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        """
        Initialize Dueling network.

        Parameters
        ----------
        state_dim : int
            Dimension of state space
        action_dim : int
            Number of discrete actions
        hidden_dim : int, default=128
            Hidden layer size
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Value stream: V(s) ∈ ℝ
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream: A(s, ·) ∈ ℝ^{|A|}
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

        self.apply(init_weights)

    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass with value-advantage aggregation.

        Computes Q(s, a) = V(s) + [A(s, a) - mean_a' A(s, a')]

        Parameters
        ----------
        state : Tensor
            Batch of states, shape (batch_size, state_dim)

        Returns
        -------
        Tensor
            Q-values for all actions, shape (batch_size, action_dim)

        Notes
        -----
        Mean subtraction ensures identifiability:
            Σ_a A(s, a) = 0  =>  V(s) = max_a Q(s, a)

        Examples
        --------
        >>> net = DuelingNetwork(state_dim=4, action_dim=2)
        >>> q_values = net(torch.randn(32, 4))
        >>> q_values.shape
        torch.Size([32, 2])
        """
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Mean subtraction for identifiability
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values
