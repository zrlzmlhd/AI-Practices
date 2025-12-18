"""
Base Network Architecture for DQN.

This module provides the standard DQN network and initialization utilities.

Core Idea (核心思想)
====================
标准DQN网络将状态映射到所有动作的Q值：

    f_θ: ℝ^d → ℝ^{|A|}

使用正交初始化提高训练稳定性，ReLU激活函数引入非线性。

Mathematical Foundation (数学基础)
==================================
Forward computation:
    h_1 = ReLU(W_1 x + b_1)
    h_2 = ReLU(W_2 h_1 + b_2)
    ...
    Q(s,·) = W_out h_L + b_out

Orthogonal initialization ensures:
    W^T W ≈ I

which preserves gradient norms during backpropagation.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


def init_weights(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    """
    Orthogonal initialization for stable deep RL training.

    Parameters
    ----------
    module : nn.Module
        Module to initialize
    gain : float, default=sqrt(2)
        Scaling factor (sqrt(2) is optimal for ReLU)

    Notes
    -----
    Orthogonal initialization preserves gradient norms:
        W^T W ≈ I

    This prevents vanishing/exploding gradients in deep networks.

    Examples
    --------
    >>> linear = nn.Linear(64, 64)
    >>> init_weights(linear)
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DQNNetwork(nn.Module):
    """
    Standard DQN network architecture.

    Core Idea (核心思想)
    --------------------
    将状态映射到所有动作的Q值：f_θ: ℝ^d → ℝ^{|A|}

    使用多层全连接网络近似Q函数，ReLU激活函数引入非线性。

    Mathematical Foundation (数学基础)
    ----------------------------------
    Forward computation through L hidden layers:

    .. math::
        h_1 = \\text{ReLU}(W_1 s + b_1)
        h_l = \\text{ReLU}(W_l h_{l-1} + b_l), \\quad l = 2, ..., L
        Q(s, \\cdot) = W_{out} h_L + b_{out}

    Parameters
    ----------
    state_dim : int
        Dimension of state/observation space
    action_dim : int
        Number of discrete actions
    hidden_dim : int, default=128
        Size of hidden layers
    num_layers : int, default=2
        Number of hidden layers

    Attributes
    ----------
    state_dim : int
        Input state dimension
    action_dim : int
        Output action dimension
    network : nn.Sequential
        The neural network layers

    Examples
    --------
    >>> net = DQNNetwork(state_dim=4, action_dim=2)
    >>> state = torch.randn(1, 4)
    >>> q_values = net(state)
    >>> q_values.shape
    torch.Size([1, 2])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        """
        Initialize DQN network.

        Parameters
        ----------
        state_dim : int
            Dimension of state space
        action_dim : int
            Number of discrete actions
        hidden_dim : int, default=128
            Hidden layer size
        num_layers : int, default=2
            Number of hidden layers
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        layers: List[nn.Module] = []
        prev_dim = state_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass: state → Q-values.

        Parameters
        ----------
        state : Tensor
            Batch of states, shape (batch_size, state_dim)

        Returns
        -------
        Tensor
            Q-values for all actions, shape (batch_size, action_dim)

        Examples
        --------
        >>> net = DQNNetwork(state_dim=4, action_dim=2)
        >>> q_values = net(torch.randn(32, 4))
        >>> q_values.shape
        torch.Size([32, 2])
        """
        return self.network(state)
