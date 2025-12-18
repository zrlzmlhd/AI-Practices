"""
Training Configuration Module.

This module defines the hyperparameter configuration for policy gradient algorithms,
encapsulating all tunable parameters in a structured dataclass.

Core Idea:
    Centralized configuration management enables reproducible experiments and
    systematic hyperparameter tuning. Each parameter is documented with its
    theoretical motivation and practical guidelines.

Mathematical Background:
    Policy gradient algorithms optimize:
        J(θ) = E_{τ~π_θ}[∑_t γ^t r_t]

    Key hyperparameters affect:
        - Bias-variance trade-off (γ, λ_GAE)
        - Optimization stability (lr, grad_norm)
        - Exploration behavior (entropy_coef)

References:
    [1] Schulman et al. (2016). High-dimensional continuous control using GAE.
    [2] Mnih et al. (2016). Asynchronous methods for deep reinforcement learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Hyperparameter configuration for policy gradient training.

    This dataclass encapsulates all configurable parameters for policy gradient
    algorithms including REINFORCE, REINFORCE with Baseline, and A2C.

    Attributes
    ----------
    gamma : float, default=0.99
        Discount factor for future rewards. Controls the agent's farsightedness.

        Mathematical Role:
            G_t = ∑_{k=0}^{∞} γ^k r_{t+k}

        Guidelines:
            - 0.99: Long-horizon tasks (control, games)
            - 0.95: Medium-horizon tasks
            - 0.90: Short-horizon tasks

    lr_actor : float, default=3e-4
        Learning rate for the policy network (actor).

        Guidelines:
            - Should be smaller than lr_critic for stable training
            - Too high: Policy collapse, unstable updates
            - Too low: Slow convergence

    lr_critic : float, default=1e-3
        Learning rate for the value network (critic).

        Guidelines:
            - Can be 3-10x larger than lr_actor
            - Value function should converge faster than policy

    entropy_coef : float, default=0.01
        Coefficient for entropy regularization term.

        Mathematical Role:
            L_entropy = -c_ent * H(π) = c_ent * ∑_a π(a|s) log π(a|s)

        Guidelines:
            - Higher (0.05-0.1): More exploration, stochastic policies
            - Lower (0.001-0.01): Less exploration, deterministic policies
            - Should decay during training for many tasks

    value_coef : float, default=0.5
        Coefficient for value function loss in combined loss.

        Mathematical Role:
            L_total = L_policy + c_v * L_value - c_ent * H(π)

        Guidelines:
            - 0.5 is standard for A2C/PPO
            - Increase if value estimates are poor

    max_grad_norm : float, default=0.5
        Maximum gradient norm for gradient clipping.

        Purpose:
            - Prevents gradient explosion
            - Stabilizes training in early stages

        Guidelines:
            - 0.5: Conservative, very stable
            - 1.0: Standard choice
            - 5.0: Permissive, faster but riskier

    gae_lambda : float, default=0.95
        Lambda parameter for Generalized Advantage Estimation (GAE).

        Mathematical Role:
            A_t^GAE(γ,λ) = ∑_{l=0}^{∞} (γλ)^l δ_{t+l}
            where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Bias-Variance Trade-off:
            - λ=0: TD(0), high bias, low variance
            - λ=1: Monte Carlo, no bias, high variance
            - λ=0.95: Good balance for most tasks

    n_steps : int, default=5
        Number of steps for n-step returns (if not using GAE).

        Mathematical Role:
            G_t^(n) = ∑_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n})

        Guidelines:
            - 5: Good for episodic tasks
            - 20: Better for long trajectories
            - 128+: Approaching Monte Carlo

    normalize_advantage : bool, default=True
        Whether to normalize advantage estimates.

        Mathematical Operation:
            A_norm = (A - μ_A) / (σ_A + ε)

        Benefits:
            - Reduces variance
            - Consistent gradient magnitudes
            - Almost always beneficial

    device : str, default="cpu"
        Computation device ("cpu" or "cuda").

    seed : Optional[int], default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> # Default configuration
    >>> config = TrainingConfig()

    >>> # Custom configuration for CartPole
    >>> config = TrainingConfig(
    ...     gamma=0.99,
    ...     lr_actor=1e-3,
    ...     lr_critic=1e-3,
    ...     entropy_coef=0.01,
    ...     gae_lambda=0.95
    ... )

    >>> # Configuration for continuous control
    >>> config = TrainingConfig(
    ...     gamma=0.99,
    ...     lr_actor=3e-4,
    ...     lr_critic=3e-4,
    ...     entropy_coef=0.001,  # Lower for continuous
    ...     gae_lambda=0.97
    ... )

    Notes
    -----
    Complexity Analysis:
        - Memory: O(1) - Fixed number of parameters
        - Access: O(1) - Direct attribute access

    Algorithm Comparison:
        | Algorithm          | Key Parameters              |
        |--------------------|-----------------------------|
        | REINFORCE          | gamma, lr_actor             |
        | REINFORCE+Baseline | gamma, lr_actor, lr_critic  |
        | A2C                | All parameters              |
        | PPO                | All + clip_epsilon          |
    """

    # Discount and return computation
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 5

    # Learning rates
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3

    # Loss coefficients
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # Optimization
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    # Environment
    device: str = "cpu"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate()

    def _validate(self) -> None:
        """
        Validate parameter ranges and relationships.

        Raises
        ------
        ValueError
            If any parameter is out of valid range.
        """
        if not 0 < self.gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")

        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")

        if self.lr_actor <= 0:
            raise ValueError(f"lr_actor must be positive, got {self.lr_actor}")

        if self.lr_critic <= 0:
            raise ValueError(f"lr_critic must be positive, got {self.lr_critic}")

        if self.entropy_coef < 0:
            raise ValueError(f"entropy_coef must be non-negative, got {self.entropy_coef}")

        if self.value_coef < 0:
            raise ValueError(f"value_coef must be non-negative, got {self.value_coef}")

        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")

        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {self.device}")

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict
            Dictionary representation of configuration.
        """
        return {
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "n_steps": self.n_steps,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "normalize_advantage": self.normalize_advantage,
            "device": self.device,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """
        Create configuration from dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary containing configuration parameters.

        Returns
        -------
        TrainingConfig
            Configuration instance.
        """
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
