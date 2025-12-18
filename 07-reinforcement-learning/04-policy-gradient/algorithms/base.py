"""
Base Policy Gradient Algorithm.

This module defines the abstract base class for all policy gradient algorithms,
establishing a common interface and shared functionality.

Core Idea:
    Policy gradient methods directly optimize a parameterized policy by
    gradient ascent on the expected return objective. The base class
    provides common infrastructure for training state management and
    tensor operations.

Mathematical Background:
    Policy Gradient Objective:
        J(θ) = E_{τ~π_θ}[R(τ)] = E_{τ~π_θ}[∑_t γ^t r_t]

    Gradient Estimation:
        ∇_θ J(θ) = E_{π_θ}[∇_θ log π_θ(a|s) · Ψ_t]

    Where Ψ_t varies by algorithm:
        | Algorithm          | Ψ_t                        |
        |--------------------|----------------------------|
        | REINFORCE          | G_t (MC return)            |
        | REINFORCE+Baseline | G_t - V(s_t)               |
        | A2C                | r + γV(s') - V(s) (TD)     |
        | A2C + GAE          | Σ(γλ)^k δ_{t+k}           |

References:
    [1] Sutton et al. (1999). Policy gradient methods for RL with FA.
    [2] Williams (1992). Simple statistical gradient-following algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from core.config import TrainingConfig
from core.buffers import EpisodeBuffer


class BasePolicyGradient(ABC):
    """
    Abstract base class for policy gradient algorithms.

    Defines the common interface and shared utilities for all policy
    gradient implementations.

    Core Idea:
        Establish a consistent API for policy gradient algorithms:
        1. select_action(): Sample actions from policy
        2. update(): Compute gradients and update parameters

    Attributes
    ----------
    config : TrainingConfig
        Hyperparameter configuration.
    device : torch.device
        Computation device (CPU/CUDA).
    training_info : dict
        Dictionary tracking training statistics.

    Methods
    -------
    select_action(state, deterministic=False)
        Sample an action from the policy.
    update(buffer)
        Update policy parameters using collected experience.

    Examples
    --------
    >>> class MyAlgorithm(BasePolicyGradient):
    ...     def select_action(self, state, deterministic=False):
    ...         # Implementation
    ...         pass
    ...
    ...     def update(self, buffer):
    ...         # Implementation
    ...         pass

    Notes
    -----
    Design Pattern:
        Template Method pattern - base class defines algorithm skeleton,
        subclasses implement specific steps.

    Thread Safety:
        Not thread-safe. For parallel training, use separate instances
        or synchronization mechanisms.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize base policy gradient algorithm.

        Parameters
        ----------
        config : TrainingConfig
            Training hyperparameters and settings.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.training_info = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
        }

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Select an action given the current state.

        Parameters
        ----------
        state : np.ndarray
            Current state observation.
        deterministic : bool, default=False
            If True, return the most likely action (no sampling).
            Useful for evaluation/testing.

        Returns
        -------
        action : Any
            Selected action (int for discrete, np.ndarray for continuous).
        info : Dict[str, torch.Tensor]
            Additional information including:
            - "log_prob": Log probability of selected action
            - "entropy": Entropy of action distribution
            - "value": Value estimate (if available)

        Notes
        -----
        Implementation Guidelines:
            1. Convert state to tensor and add batch dimension
            2. Forward pass through policy network
            3. Sample or select action based on deterministic flag
            4. Return action and auxiliary info for gradient computation
        """
        pass

    @abstractmethod
    def update(self, buffer: EpisodeBuffer) -> Dict[str, float]:
        """
        Update policy parameters using collected experience.

        Parameters
        ----------
        buffer : EpisodeBuffer
            Buffer containing trajectory data:
            - states, actions, rewards
            - log_probs (for gradient computation)
            - values (for baseline, if applicable)
            - dones (episode termination flags)

        Returns
        -------
        Dict[str, float]
            Dictionary of training metrics:
            - "policy_loss": Policy gradient loss
            - "value_loss": Value function loss (if applicable)
            - "entropy": Mean policy entropy
            - Additional algorithm-specific metrics

        Notes
        -----
        Implementation Guidelines:
            1. Compute returns/advantages from buffer data
            2. Calculate policy gradient loss
            3. Add entropy regularization if configured
            4. Perform backpropagation and optimization
            5. Return metrics for logging
        """
        pass

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to tensor on the correct device.

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        torch.Tensor
            Tensor on configured device.
        """
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _normalize_advantages(
        self,
        advantages: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Normalize advantages to zero mean and unit variance.

        Advantage normalization is a variance reduction technique that
        helps stabilize training by ensuring consistent gradient magnitudes.

        Parameters
        ----------
        advantages : torch.Tensor
            Raw advantage estimates.
        eps : float, default=1e-8
            Small constant for numerical stability.

        Returns
        -------
        torch.Tensor
            Normalized advantages.

        Notes
        -----
        Mathematical Operation:
            A_norm = (A - mean(A)) / (std(A) + eps)

        Benefits:
            - Consistent gradient scale across batches
            - Reduced sensitivity to reward magnitude
            - More stable optimization
        """
        if len(advantages) > 1 and self.config.normalize_advantage:
            return (advantages - advantages.mean()) / (advantages.std() + eps)
        return advantages

    def save(self, path: str) -> None:
        """
        Save model parameters to file.

        Parameters
        ----------
        path : str
            File path for saving.
        """
        state = {
            "config": self.config.to_dict(),
            "training_info": self.training_info,
        }

        # Add model-specific state
        if hasattr(self, "policy"):
            state["policy_state_dict"] = self.policy.state_dict()
        if hasattr(self, "value_net"):
            state["value_net_state_dict"] = self.value_net.state_dict()
        if hasattr(self, "model"):
            state["model_state_dict"] = self.model.state_dict()

        torch.save(state, path)

    def load(self, path: str) -> None:
        """
        Load model parameters from file.

        Parameters
        ----------
        path : str
            File path to load from.
        """
        state = torch.load(path, map_location=self.device)

        if hasattr(self, "policy") and "policy_state_dict" in state:
            self.policy.load_state_dict(state["policy_state_dict"])
        if hasattr(self, "value_net") and "value_net_state_dict" in state:
            self.value_net.load_state_dict(state["value_net_state_dict"])
        if hasattr(self, "model") and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])

        self.training_info = state.get("training_info", self.training_info)
