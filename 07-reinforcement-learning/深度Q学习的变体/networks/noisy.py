"""
Noisy Network Components for Learned Exploration.

This module implements NoisyNet (Fortunato et al., 2017) for parametric exploration.

Core Idea (核心思想)
====================
用可学习的参数化噪声替代ε-greedy探索，实现状态依赖的探索策略。

Noisy Linear Layer:
    y = (μ^w + σ^w ⊙ ε^w) x + (μ^b + σ^b ⊙ ε^b)

- μ: 可学习的均值参数
- σ: 可学习的噪声尺度参数（自动调节探索程度）
- ε: 每次前向传播时采样的随机噪声

Mathematical Foundation (数学基础)
==================================
Factorized Gaussian Noise (efficient parameterization):
    ε_{ij} = f(ε_i) · f(ε_j), where f(x) = sign(x)√|x|

This reduces noise parameters from O(pq) to O(p+q) for (p×q) weight matrix.

Key Properties:
- σ parameters are learned end-to-end via backpropagation
- Network automatically reduces exploration as it becomes more certain
- Exploration is inherently state-dependent

References:
    Fortunato, M. et al. (2017). Noisy Networks for Exploration. ICLR.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration (Fortunato et al., 2017).

    Core Idea (核心思想)
    --------------------
    用可学习的参数化噪声替代ε-greedy探索：

    .. math::
        y = (\\mu^w + \\sigma^w \\odot \\varepsilon^w) x + (\\mu^b + \\sigma^b \\odot \\varepsilon^b)

    其中：
    - μ: 可学习的均值参数
    - σ: 可学习的噪声尺度参数
    - ε: 随机噪声

    Factorized Noise (因式分解噪声)
    ------------------------------
    为减少参数量，使用因式分解高斯噪声：

    .. math::
        \\varepsilon_{ij} = f(\\varepsilon_i) \\cdot f(\\varepsilon_j), \\quad f(x) = \\text{sign}(x)\\sqrt{|x|}

    参数量从 O(pq) 降到 O(p+q)

    Properties (特性)
    -----------------
    - **状态依赖探索**: 不同状态有不同的探索程度
    - **自动退火**: 随着学习进行，σ自然减小
    - **端到端学习**: 不需要手动调整ε衰减

    Parameters
    ----------
    in_features : int
        Input dimension
    out_features : int
        Output dimension
    sigma_init : float, default=0.5
        Initial noise scale

    Examples
    --------
    >>> layer = NoisyLinear(64, 32)
    >>> x = torch.randn(16, 64)
    >>> layer.reset_noise()  # Sample new noise
    >>> y = layer(x)
    >>> y.shape
    torch.Size([16, 32])

    References
    ----------
    Fortunato, M. et al. (2017). Noisy Networks for Exploration. ICLR.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
    ) -> None:
        """
        Initialize noisy linear layer.

        Parameters
        ----------
        in_features : int
            Input dimension
        out_features : int
            Output dimension
        sigma_init : float, default=0.5
            Initial noise scale
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters: μ and σ for weights and biases
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not parameters, not saved in state_dict)
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features),
        )
        self.register_buffer(
            "bias_epsilon",
            torch.empty(out_features),
        )

        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self) -> None:
        """
        Initialize μ and σ parameters.

        μ initialized uniformly in [-1/√fan_in, 1/√fan_in]
        σ initialized to sigma_init / √fan_in
        """
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        sigma = self.sigma_init / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma)
        self.bias_sigma.data.fill_(sigma)

    def _factorized_noise(self, size: int) -> Tensor:
        """
        Generate factorized Gaussian noise: f(x) = sign(x)√|x|.

        Parameters
        ----------
        size : int
            Number of noise samples

        Returns
        -------
        Tensor
            Factorized noise, shape (size,)
        """
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """
        Sample new noise for exploration.

        Should be called before each forward pass during training
        to ensure exploration diversity.

        Examples
        --------
        >>> layer = NoisyLinear(64, 32)
        >>> layer.reset_noise()
        >>> out1 = layer(x)
        >>> layer.reset_noise()
        >>> out2 = layer(x)  # Different due to new noise
        """
        epsilon_in = self._factorized_noise(self.in_features)
        epsilon_out = self._factorized_noise(self.out_features)

        # Outer product for factorized noise: ε_{ij} = ε_i · ε_j
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with noisy weights.

        During training:
            w = μ^w + σ^w ⊙ ε^w
            b = μ^b + σ^b ⊙ ε^b
            y = wx + b

        During evaluation:
            Uses mean parameters only (deterministic)

        Parameters
        ----------
        x : Tensor
            Input tensor, shape (batch_size, in_features)

        Returns
        -------
        Tensor
            Output tensor, shape (batch_size, out_features)
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Evaluation: use mean parameters only
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        """Return string representation of layer configuration."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"sigma_init={self.sigma_init}"
        )


class NoisyNetwork(nn.Module):
    """
    Noisy DQN network for learned exploration (Fortunato et al., 2017).

    Replaces ε-greedy with parametric noise layers that learn
    state-dependent exploration through gradient descent.

    Parameters
    ----------
    state_dim : int
        Dimension of state space
    action_dim : int
        Number of discrete actions
    hidden_dim : int, default=128
        Hidden layer size
    sigma_init : float, default=0.5
        Initial noise scale for noisy layers

    Examples
    --------
    >>> net = NoisyNetwork(state_dim=4, action_dim=2)
    >>> net.reset_noise()  # Sample new noise
    >>> q_values = net(torch.randn(1, 4))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Standard layers for feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Noisy layers for exploration
        self.noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.noisy2 = NoisyLinear(hidden_dim, action_dim, sigma_init)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass: state → Q-values."""
        features = self.feature(state)
        x = F.relu(self.noisy1(features))
        return self.noisy2(x)

    def reset_noise(self) -> None:
        """Sample new noise for all noisy layers."""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class NoisyDuelingNetwork(nn.Module):
    """
    Combines Dueling architecture with Noisy Networks.

    Features:
    - Value-advantage decomposition for stable learning
    - Parametric noise for learned exploration

    Parameters
    ----------
    state_dim : int
        Dimension of state space
    action_dim : int
        Number of discrete actions
    hidden_dim : int, default=128
        Hidden layer size
    sigma_init : float, default=0.5
        Initial noise scale

    Examples
    --------
    >>> net = NoisyDuelingNetwork(state_dim=4, action_dim=2)
    >>> net.reset_noise()
    >>> q_values = net(torch.randn(1, 4))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Noisy value stream
        self.value_noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.value_noisy2 = NoisyLinear(hidden_dim, 1, sigma_init)

        # Noisy advantage stream
        self.adv_noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.adv_noisy2 = NoisyLinear(hidden_dim, action_dim, sigma_init)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass with dueling aggregation."""
        features = self.feature(state)

        # Value stream
        value = F.relu(self.value_noisy1(features))
        value = self.value_noisy2(value)

        # Advantage stream
        advantage = F.relu(self.adv_noisy1(features))
        advantage = self.adv_noisy2(advantage)

        # Aggregate with mean subtraction
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values

    def reset_noise(self) -> None:
        """Reset noise for all noisy layers."""
        self.value_noisy1.reset_noise()
        self.value_noisy2.reset_noise()
        self.adv_noisy1.reset_noise()
        self.adv_noisy2.reset_noise()
