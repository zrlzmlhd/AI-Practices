"""
Rainbow Network Architecture.

This module implements the Rainbow network combining all DQN improvements.

Core Idea (核心思想)
====================
Rainbow (Hessel et al., 2018) 组合了所有DQN改进：

    Rainbow = Double + Dueling + Noisy + Categorical + PER + N-step

Each component addresses orthogonal issues, and their combination
achieves state-of-the-art on Atari (441% human-normalized median score).

Network Architecture (网络架构)
==============================
Combines:
1. **Dueling**: Separate value and advantage streams
2. **Noisy**: Parametric noise for exploration
3. **Categorical**: Distributional output over atoms

The network outputs log-probabilities over N atoms for each action,
using noisy layers in both value and advantage streams.

Performance (性能)
==================
Median human-normalized score on Atari:
- DQN: 79%
- Double DQN: 117%
- Dueling DQN: 151%
- Categorical DQN: 235%
- Rainbow: 441%

References:
    Hessel, M. et al. (2018). Rainbow: Combining Improvements in Deep
    Reinforcement Learning. AAAI.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from networks.noisy import NoisyLinear


class RainbowNetwork(nn.Module):
    """
    Rainbow network combining all DQN improvements (Hessel et al., 2018).

    Core Idea (核心思想)
    --------------------
    Rainbow组合了所有DQN改进，每个组件解决正交的问题：

    +------------------+--------------------------------------------------+
    | Component        | Contribution                                     |
    +==================+==================================================+
    | Double           | Fixes overestimation via decoupled evaluation    |
    +------------------+--------------------------------------------------+
    | Dueling          | Better generalization via V/A decomposition      |
    +------------------+--------------------------------------------------+
    | Noisy            | Learned, state-dependent exploration             |
    +------------------+--------------------------------------------------+
    | Categorical      | Full return distribution modeling                |
    +------------------+--------------------------------------------------+
    | PER              | Efficient sampling of informative transitions    |
    +------------------+--------------------------------------------------+
    | Multi-step       | Faster credit assignment                         |
    +------------------+--------------------------------------------------+

    Network Architecture (网络架构)
    ------------------------------
    ::

        Input → Shared Features → Noisy Value Stream  → V distribution [N atoms]
                               ↘
                                 Noisy Adv Stream    → A distribution [|A| × N atoms]
                               ↘
                                 Aggregate (logit space) → Q distribution [|A| × N atoms]

    Mathematical Foundation (数学基础)
    ----------------------------------
    Output representation:

    .. math::
        Z(s, a; \\theta) \\sim \\text{Categorical}(z_1, ..., z_N; p_1(s,a;\\theta), ..., p_N(s,a;\\theta))

    Dueling in distribution space:

    .. math::
        \\text{logits}(s, a) = V_\\theta(s) + A_\\theta(s, a) - \\text{mean}_a A_\\theta(s, a)

    Expected Q-value:

    .. math::
        Q(s, a) = \\sum_i z_i \\cdot p_i(s, a)

    Parameters
    ----------
    state_dim : int
        Dimension of state space
    action_dim : int
        Number of discrete actions
    hidden_dim : int, default=128
        Hidden layer size
    num_atoms : int, default=51
        Number of atoms for distributional output
    v_min : float, default=-10.0
        Minimum value of support
    v_max : float, default=10.0
        Maximum value of support
    sigma_init : float, default=0.5
        Initial noise scale for noisy layers

    Attributes
    ----------
    support : Tensor
        Fixed support atoms, shape (num_atoms,)
    delta_z : float
        Spacing between adjacent atoms

    Examples
    --------
    >>> net = RainbowNetwork(state_dim=4, action_dim=2)
    >>> net.reset_noise()  # Sample new noise
    >>> log_probs = net(torch.randn(1, 4))
    >>> log_probs.shape
    torch.Size([1, 2, 51])
    >>> q_values = net.get_q_values(torch.randn(1, 4))
    >>> q_values.shape
    torch.Size([1, 2])

    References
    ----------
    Hessel, M. et al. (2018). Rainbow: Combining Improvements in Deep
    Reinforcement Learning. AAAI.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        sigma_init: float = 0.5,
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

        # Shared feature extraction (standard layers)
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Noisy value stream
        self.value_noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.value_noisy2 = NoisyLinear(hidden_dim, num_atoms, sigma_init)

        # Noisy advantage stream
        self.adv_noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.adv_noisy2 = NoisyLinear(
            hidden_dim, action_dim * num_atoms, sigma_init
        )

    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass with all Rainbow components.

        Combines:
        1. Shared feature extraction
        2. Noisy value stream → V distribution over atoms
        3. Noisy advantage stream → A distribution over atoms for each action
        4. Dueling aggregation in logit space
        5. Log-softmax for distributional output

        Parameters
        ----------
        state : Tensor
            Batch of states, shape (batch_size, state_dim)

        Returns
        -------
        Tensor
            Log probabilities over atoms for each action,
            shape (batch_size, action_dim, num_atoms)
        """
        batch_size = state.shape[0]
        features = self.feature(state)

        # Value stream
        value = F.relu(self.value_noisy1(features))
        value = self.value_noisy2(value).view(batch_size, 1, self.num_atoms)

        # Advantage stream
        advantage = F.relu(self.adv_noisy1(features))
        advantage = self.adv_noisy2(advantage).view(
            batch_size, self.action_dim, self.num_atoms
        )

        # Dueling aggregation in logit space
        logits = value + (advantage - advantage.mean(dim=1, keepdim=True))
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
            Q-values for all actions, shape (batch_size, action_dim)

        Examples
        --------
        >>> net = RainbowNetwork(state_dim=4, action_dim=2)
        >>> q_values = net.get_q_values(torch.randn(32, 4))
        >>> q_values.shape
        torch.Size([32, 2])
        """
        log_probs = self.forward(state)
        probs = log_probs.exp()
        q_values = (probs * self.support).sum(dim=-1)
        return q_values

    def reset_noise(self) -> None:
        """
        Reset noise for exploration.

        Should be called before each forward pass during training
        to ensure exploration diversity.

        Examples
        --------
        >>> net = RainbowNetwork(state_dim=4, action_dim=2)
        >>> net.reset_noise()
        >>> q1 = net.get_q_values(state)
        >>> net.reset_noise()
        >>> q2 = net.get_q_values(state)  # Different due to noise
        """
        self.value_noisy1.reset_noise()
        self.value_noisy2.reset_noise()
        self.adv_noisy1.reset_noise()
        self.adv_noisy2.reset_noise()
