"""
Categorical (Distributional) Network Architectures.

This module implements Categorical DQN / C51 (Bellemare et al., 2017).

Core Idea (核心思想)
====================
从建模期望值转向建模完整的回报分布：

Standard DQN: Q(s, a) = E[Z(s, a)]
Categorical DQN: Z(s, a) ~ Categorical(z_1, ..., z_N; p_1, ..., p_N)

分布视角提供更丰富的学习信号，使训练更加稳定。

Mathematical Foundation (数学基础)
==================================
Distribution Representation:
    Z(s, a) ~ Categorical(z_1, ..., z_N; p_1(s,a), ..., p_N(s,a))

Support Atoms:
    z_i = V_min + i · Δz, where Δz = (V_max - V_min) / (N - 1)

Distributional Bellman Equation:
    T Z(s, a) = R + γ Z(S', A')  (equality in distribution)

Categorical Projection:
    Projects shifted distribution back onto fixed support

KL Divergence Loss:
    L = D_KL(Φ T Z(s, a) || Z(s, a; θ))

References:
    Bellemare, M. et al. (2017). A Distributional Perspective on
    Reinforcement Learning. ICML.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from networks.base import init_weights


class CategoricalNetwork(nn.Module):
    """
    Categorical DQN / C51 network (Bellemare et al., 2017).

    Core Idea (核心思想)
    --------------------
    从建模**期望值**转向建模**完整回报分布**：

    - Standard DQN: Q(s, a) = E[Z(s, a)]
    - Categorical DQN: Z(s, a) ~ Categorical(z_1, ..., z_N; p_1, ..., p_N)

    Mathematical Foundation (数学基础)
    ----------------------------------
    Distribution Representation:

    .. math::
        Z(s, a) \\sim \\text{Categorical}(z_1, ..., z_N; p_1(s,a), ..., p_N(s,a))

    Support Atoms (固定支撑点):

    .. math::
        z_i = V_{\\min} + i \\cdot \\Delta z, \\quad \\Delta z = \\frac{V_{\\max} - V_{\\min}}{N - 1}

    Expected Q-value from distribution:

    .. math::
        Q(s, a) = \\mathbb{E}[Z(s, a)] = \\sum_i z_i \\cdot p_i(s, a)

    Hyperparameters (超参数)
    -----------------------
    | Parameter | Typical Value | Description |
    |-----------|---------------|-------------|
    | N (atoms) | 51            | Number of support atoms |
    | V_min     | -10           | Minimum return value |
    | V_max     | 10            | Maximum return value |

    Parameters
    ----------
    state_dim : int
        Dimension of state space
    action_dim : int
        Number of discrete actions
    hidden_dim : int, default=128
        Hidden layer size
    num_atoms : int, default=51
        Number of atoms in distribution
    v_min : float, default=-10.0
        Minimum value of support
    v_max : float, default=10.0
        Maximum value of support

    Attributes
    ----------
    support : Tensor
        Fixed support atoms, shape (num_atoms,)
    delta_z : float
        Spacing between adjacent atoms

    Examples
    --------
    >>> net = CategoricalNetwork(state_dim=4, action_dim=2)
    >>> log_probs = net(torch.randn(1, 4))
    >>> log_probs.shape
    torch.Size([1, 2, 51])
    >>> q_values = net.get_q_values(torch.randn(1, 4))
    >>> q_values.shape
    torch.Size([1, 2])

    References
    ----------
    Bellemare, M. et al. (2017). A Distributional Perspective on
    Reinforcement Learning. ICML.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support atoms: z_i = V_min + i · Δz
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, num_atoms),
        )

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Output: |A| × N_atoms logits
        self.output = nn.Linear(hidden_dim, action_dim * num_atoms)

        self.apply(init_weights)

    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass returning log probabilities.

        Parameters
        ----------
        state : Tensor
            Batch of states, shape (batch_size, state_dim)

        Returns
        -------
        Tensor
            Log probabilities, shape (batch_size, action_dim, num_atoms)

        Notes
        -----
        Returns log-softmax for numerical stability when computing
        KL divergence loss.
        """
        batch_size = state.shape[0]
        features = self.feature(state)
        logits = self.output(features)

        # Reshape to (batch, actions, atoms) and apply log-softmax
        logits = logits.view(batch_size, self.action_dim, self.num_atoms)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def get_q_values(self, state: Tensor) -> Tensor:
        """
        Compute expected Q-values from distribution.

        Q(s, a) = Σ_i z_i · p_i(s, a)

        Parameters
        ----------
        state : Tensor
            Batch of states, shape (batch_size, state_dim)

        Returns
        -------
        Tensor
            Q-values, shape (batch_size, action_dim)
        """
        log_probs = self.forward(state)
        probs = log_probs.exp()
        q_values = (probs * self.support).sum(dim=-1)
        return q_values


class CategoricalDuelingNetwork(nn.Module):
    """
    Categorical Dueling network combining distributional RL with value-advantage decomposition.

    Extends C51 with Dueling architecture:
    - Separate value and advantage distributions
    - Combines benefits of both approaches

    Parameters
    ----------
    state_dim : int
        Dimension of state space
    action_dim : int
        Number of discrete actions
    hidden_dim : int, default=128
        Hidden layer size
    num_atoms : int, default=51
        Number of atoms in distribution
    v_min : float, default=-10.0
        Minimum value of support
    v_max : float, default=10.0
        Maximum value of support

    Examples
    --------
    >>> net = CategoricalDuelingNetwork(state_dim=4, action_dim=2)
    >>> log_probs = net(torch.randn(1, 4))
    >>> q_values = net.get_q_values(torch.randn(1, 4))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, num_atoms),
        )

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Value stream: outputs atom probabilities
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_atoms),
        )

        # Advantage stream: outputs atom probabilities for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim * num_atoms),
        )

        self.apply(init_weights)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass with distributional dueling aggregation."""
        batch_size = state.shape[0]
        features = self.feature(state)

        value = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(
            batch_size, self.action_dim, self.num_atoms
        )

        # Aggregate: Q = V + (A - mean(A)) in logit space
        logits = value + (advantage - advantage.mean(dim=1, keepdim=True))
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def get_q_values(self, state: Tensor) -> Tensor:
        """Compute expected Q-values from distribution."""
        log_probs = self.forward(state)
        probs = log_probs.exp()
        q_values = (probs * self.support).sum(dim=-1)
        return q_values
