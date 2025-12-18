"""
Configuration for DQN Variant Agents.

This module provides centralized hyperparameter management with validation.

Core Idea (核心思想)
====================
使用dataclass集中管理所有超参数，通过__post_init__进行验证，
确保参数在有效范围内。支持自动设备检测和可复现的随机种子设置。

Hyperparameter Categories (超参数分类)
======================================
1. **Environment**: state_dim, action_dim
2. **Architecture**: hidden_dim, num_layers, num_atoms
3. **Learning**: learning_rate, gamma, batch_size
4. **Exploration**: epsilon_start, epsilon_end, epsilon_decay_steps
5. **Buffer**: buffer_size, min_buffer_size, n_steps
6. **PER**: use_per, per_alpha, per_beta_start, per_beta_frames
7. **Target Network**: target_update_freq, soft_update_tau
8. **Training**: grad_clip, device, seed

Example:
    >>> config = DQNVariantConfig(state_dim=4, action_dim=2)
    >>> config.get_device()
    device(type='cuda')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DQNVariantConfig:
    """
    Configuration for DQN variant agents.

    Core Idea (核心思想)
    --------------------
    集中管理所有DQN变体的超参数，提供类型安全的配置接口。
    通过dataclass实现不可变配置，支持IDE自动补全和类型检查。

    Mathematical Foundation (数学基础)
    ----------------------------------
    Key hyperparameters and their roles:

    - **γ (gamma)**: Discount factor in Bellman equation
      Q(s,a) = E[R + γ max_a' Q(s',a')]

    - **α (learning_rate)**: Step size for gradient descent
      θ ← θ - α ∇L(θ)

    - **ε (epsilon)**: Exploration probability in ε-greedy
      a = argmax_a Q(s,a) with prob 1-ε, random with prob ε

    - **τ (soft_update_tau)**: Polyak averaging coefficient
      θ⁻ ← τθ + (1-τ)θ⁻

    Attributes
    ----------
    state_dim : int
        Dimension of state/observation space
    action_dim : int
        Number of discrete actions
    hidden_dim : int, default=128
        Hidden layer size for neural networks
    num_layers : int, default=2
        Number of hidden layers (for standard DQN)
    num_atoms : int, default=51
        Number of atoms for distributional RL (C51)
    v_min : float, default=-10.0
        Minimum value of support for C51
    v_max : float, default=10.0
        Maximum value of support for C51
    sigma_init : float, default=0.5
        Initial noise scale for NoisyNet layers
    learning_rate : float, default=1e-3
        Optimizer learning rate
    gamma : float, default=0.99
        Discount factor for future rewards
    batch_size : int, default=64
        Mini-batch size for training
    epsilon_start : float, default=1.0
        Initial exploration rate
    epsilon_end : float, default=0.01
        Final exploration rate
    epsilon_decay_steps : int, default=10000
        Steps to decay ε from start to end
    buffer_size : int, default=100000
        Maximum replay buffer capacity
    min_buffer_size : int, default=1000
        Minimum buffer size before training starts
    use_per : bool, default=False
        Whether to use Prioritized Experience Replay
    per_alpha : float, default=0.6
        PER prioritization exponent (0=uniform, 1=full)
    per_beta_start : float, default=0.4
        Initial importance sampling exponent
    per_beta_frames : int, default=100000
        Frames to anneal β from start to 1.0
    n_steps : int, default=1
        Number of steps for n-step returns
    target_update_freq : int, default=100
        Steps between hard target network updates
    soft_update_tau : Optional[float], default=None
        Polyak averaging coefficient (None=hard update)
    grad_clip : float, default=10.0
        Maximum gradient norm for clipping
    device : str, default="auto"
        Compute device ("auto", "cpu", "cuda", "mps")
    seed : Optional[int], default=None
        Random seed for reproducibility

    Raises
    ------
    ValueError
        If any parameter is outside its valid range

    Examples
    --------
    Basic configuration:

    >>> config = DQNVariantConfig(state_dim=4, action_dim=2)
    >>> config.gamma
    0.99

    Custom configuration:

    >>> config = DQNVariantConfig(
    ...     state_dim=84*84*4,
    ...     action_dim=18,
    ...     hidden_dim=512,
    ...     use_per=True,
    ...     n_steps=3,
    ... )

    Get device:

    >>> device = config.get_device()
    >>> isinstance(device, torch.device)
    True
    """

    # Environment
    state_dim: int
    action_dim: int

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 2

    # Distributional RL (C51)
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0

    # Noisy Networks
    sigma_init: float = 0.5

    # Learning
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64

    # Exploration (for non-noisy variants)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 10000

    # Replay buffer
    buffer_size: int = 100000
    min_buffer_size: int = 1000

    # Prioritized replay
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100000

    # Multi-step learning
    n_steps: int = 1

    # Target network
    target_update_freq: int = 100
    soft_update_tau: Optional[float] = None

    # Training
    grad_clip: float = 10.0
    device: str = "auto"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """
        Validate configuration parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        # Environment validation
        if self.state_dim <= 0:
            raise ValueError(
                f"state_dim must be positive, got {self.state_dim}"
            )
        if self.action_dim <= 0:
            raise ValueError(
                f"action_dim must be positive, got {self.action_dim}"
            )

        # Architecture validation
        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {self.hidden_dim}"
            )
        if self.num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {self.num_layers}"
            )

        # Distributional RL validation
        if self.num_atoms < 2:
            raise ValueError(
                f"num_atoms must be >= 2, got {self.num_atoms}"
            )
        if self.v_min >= self.v_max:
            raise ValueError(
                f"v_min ({self.v_min}) must be < v_max ({self.v_max})"
            )

        # Learning validation
        if not 0 < self.learning_rate <= 1:
            raise ValueError(
                f"learning_rate must be in (0, 1], got {self.learning_rate}"
            )
        if not 0 <= self.gamma <= 1:
            raise ValueError(
                f"gamma must be in [0, 1], got {self.gamma}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}"
            )

        # Exploration validation
        if not 0 <= self.epsilon_start <= 1:
            raise ValueError(
                f"epsilon_start must be in [0, 1], got {self.epsilon_start}"
            )
        if not 0 <= self.epsilon_end <= 1:
            raise ValueError(
                f"epsilon_end must be in [0, 1], got {self.epsilon_end}"
            )
        if self.epsilon_decay_steps <= 0:
            raise ValueError(
                f"epsilon_decay_steps must be positive, got {self.epsilon_decay_steps}"
            )

        # Buffer validation
        if self.buffer_size <= 0:
            raise ValueError(
                f"buffer_size must be positive, got {self.buffer_size}"
            )
        if self.min_buffer_size <= 0:
            raise ValueError(
                f"min_buffer_size must be positive, got {self.min_buffer_size}"
            )

        # PER validation
        if not 0 <= self.per_alpha <= 1:
            raise ValueError(
                f"per_alpha must be in [0, 1], got {self.per_alpha}"
            )
        if not 0 <= self.per_beta_start <= 1:
            raise ValueError(
                f"per_beta_start must be in [0, 1], got {self.per_beta_start}"
            )

        # Multi-step validation
        if self.n_steps < 1:
            raise ValueError(
                f"n_steps must be >= 1, got {self.n_steps}"
            )

        # Target network validation
        if self.target_update_freq <= 0:
            raise ValueError(
                f"target_update_freq must be positive, got {self.target_update_freq}"
            )
        if self.soft_update_tau is not None and not 0 < self.soft_update_tau <= 1:
            raise ValueError(
                f"soft_update_tau must be in (0, 1], got {self.soft_update_tau}"
            )

        # Training validation
        if self.grad_clip <= 0:
            raise ValueError(
                f"grad_clip must be positive, got {self.grad_clip}"
            )

    def get_device(self) -> torch.device:
        """
        Get compute device based on configuration.

        Automatically detects available hardware if device="auto".

        Returns
        -------
        torch.device
            The compute device to use

        Examples
        --------
        >>> config = DQNVariantConfig(state_dim=4, action_dim=2, device="auto")
        >>> device = config.get_device()
        >>> device.type in ("cpu", "cuda", "mps")
        True
        """
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary for serialization
        """
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_atoms": self.num_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "sigma_init": self.sigma_init,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "buffer_size": self.buffer_size,
            "min_buffer_size": self.min_buffer_size,
            "use_per": self.use_per,
            "per_alpha": self.per_alpha,
            "per_beta_start": self.per_beta_start,
            "per_beta_frames": self.per_beta_frames,
            "n_steps": self.n_steps,
            "target_update_freq": self.target_update_freq,
            "soft_update_tau": self.soft_update_tau,
            "grad_clip": self.grad_clip,
            "device": self.device,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DQNVariantConfig":
        """
        Create configuration from dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary

        Returns
        -------
        DQNVariantConfig
            New configuration instance
        """
        return cls(**config_dict)
