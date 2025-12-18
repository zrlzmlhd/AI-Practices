"""
Base Configuration Module
=========================

Core Idea
---------
Centralized configuration management with automatic device selection,
hyperparameter validation, and serialization support.

Mathematical Theory
-------------------
Configuration parameters directly influence optimization dynamics:

.. math::

    \\theta_{t+1} = \\theta_t + \\alpha \\nabla_\\theta J(\\theta)

where :math:`\\alpha` (learning rate) critically affects convergence.

The discount factor :math:`\\gamma` determines the horizon of value estimation:

.. math::

    G_t = \\sum_{k=0}^{\\infty} \\gamma^k r_{t+k+1}

Higher :math:`\\gamma` (closer to 1) makes the agent more far-sighted.

Problem Statement
-----------------
Deep RL algorithms require numerous hyperparameters that interact in complex
ways. Without proper management:

1. Hyperparameters scatter across code, making reproduction difficult
2. No validation leads to silent failures (e.g., negative learning rates)
3. Device management repeats in every component

This module provides a single source of truth for all configuration.

Summary
-------
BaseConfig consolidates common RL hyperparameters with validation.
DeviceMixin provides automatic hardware acceleration detection.
"""

from abc import ABC
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union
import json
import torch


class DeviceMixin:
    """
    Automatic Device Selection
    ==========================

    Core Idea
    ---------
    Automatically detect and utilize the best available compute device
    (CUDA GPU > Apple MPS > CPU) without manual configuration.

    Problem Statement
    -----------------
    PyTorch requires explicit device specification. Users often:

    - Hardcode 'cuda' and crash on CPU-only machines
    - Forget to move tensors to GPU, wasting compute resources
    - Write repetitive device detection code

    This mixin centralizes device management with a single 'auto' option.

    Complexity
    ----------
    - Time: O(1) - Simple hardware availability checks
    - Space: O(1) - Only stores device reference

    Example
    -------
    >>> config = BaseConfig(state_dim=4, action_dim=2, device='auto')
    >>> device = config.get_device()
    >>> device  # Returns cuda:0 if available, else mps or cpu
    """

    device: str = "auto"

    def get_device(self) -> torch.device:
        """
        Resolve device string to torch.device.

        Returns
        -------
        torch.device
            Best available compute device.
            Priority: CUDA > MPS (Apple Silicon) > CPU

        Raises
        ------
        ValueError
            If specified device is invalid.
        """
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        valid_devices = {"cpu", "cuda", "mps"}
        device_type = self.device.split(":")[0]
        if device_type not in valid_devices:
            raise ValueError(
                f"Invalid device '{self.device}'. "
                f"Expected one of: {valid_devices} or 'auto'"
            )

        return torch.device(self.device)


@dataclass
class BaseConfig(DeviceMixin):
    """
    Base Configuration for All RL Algorithms
    =========================================

    Core Idea
    ---------
    Single source of truth for hyperparameters shared across DDPG, TD3, and SAC.
    Provides validation, serialization, and sensible defaults.

    Mathematical Theory
    -------------------
    Key parameters and their theoretical roles:

    **Learning Rates** (:math:`\\alpha_{actor}`, :math:`\\alpha_{critic}`):

    - Actor: :math:`\\theta \\leftarrow \\theta + \\alpha_{actor} \\nabla_\\theta J(\\theta)`
    - Critic: :math:`\\phi \\leftarrow \\phi - \\alpha_{critic} \\nabla_\\phi L(\\phi)`

    **Discount Factor** (:math:`\\gamma`):

    .. math::

        Q(s, a) = r + \\gamma \\mathbb{E}[Q(s', a')]

    - :math:`\\gamma = 0`: Myopic, only considers immediate reward
    - :math:`\\gamma \\to 1`: Far-sighted, considers long-term returns

    **Soft Update Coefficient** (:math:`\\tau`):

    .. math::

        \\theta_{target} \\leftarrow \\tau \\theta + (1 - \\tau) \\theta_{target}

    - Small :math:`\\tau` (0.001-0.01): Stable but slow target updates
    - Large :math:`\\tau`: Fast adaptation but potential instability

    Problem Statement
    -----------------
    Each RL algorithm requires:

    - Environment specifications (state/action dimensions)
    - Network architecture (hidden layer sizes)
    - Optimization settings (learning rates, batch size)
    - Algorithm-specific parameters

    Without centralization, these scatter across constructors, making
    hyperparameter tuning and reproduction error-prone.

    Algorithm Comparison
    --------------------
    +--------------+-------------+-------------+-------------+
    | Parameter    | DDPG        | TD3         | SAC         |
    +==============+=============+=============+=============+
    | lr_actor     | 1e-4        | 3e-4        | 3e-4        |
    | lr_critic    | 1e-3        | 3e-4        | 3e-4        |
    | tau          | 0.005       | 0.005       | 0.005       |
    | gamma        | 0.99        | 0.99        | 0.99        |
    | buffer_size  | 1e6         | 1e6         | 1e6         |
    +--------------+-------------+-------------+-------------+

    Complexity
    ----------
    - Time: O(1) for all operations (validation, device selection)
    - Space: O(n) where n is number of parameters

    Summary
    -------
    Inherit from BaseConfig to create algorithm-specific configurations.
    Override defaults as needed while maintaining a consistent interface.

    Attributes
    ----------
    state_dim : int
        Dimensionality of observation space.
    action_dim : int
        Dimensionality of action space.
    max_action : float
        Maximum absolute value for action clipping.
    hidden_dims : List[int]
        Sizes of hidden layers in neural networks.
    lr_actor : float
        Learning rate for policy network.
    lr_critic : float
        Learning rate for value network(s).
    gamma : float
        Discount factor for future rewards.
    tau : float
        Soft update coefficient for target networks.
    buffer_size : int
        Maximum capacity of experience replay buffer.
    batch_size : int
        Number of transitions per training batch.
    start_timesteps : int
        Steps of random exploration before learning begins.
    device : str
        Compute device: 'auto', 'cpu', 'cuda', or 'mps'.
    seed : Optional[int]
        Random seed for reproducibility.

    Example
    -------
    >>> config = BaseConfig(
    ...     state_dim=11,
    ...     action_dim=3,
    ...     max_action=1.0,
    ...     hidden_dims=[256, 256],
    ...     lr_actor=3e-4,
    ...     gamma=0.99
    ... )
    >>> config.validate()
    >>> device = config.get_device()
    """

    # Environment specifications
    state_dim: int = 0
    action_dim: int = 0
    max_action: float = 1.0

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # Optimization
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    # Experience replay
    buffer_size: int = 1_000_000
    batch_size: int = 256

    # Exploration
    start_timesteps: int = 25000

    # Device and reproducibility
    device: str = "auto"
    seed: Optional[int] = None

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid.

        Example
        -------
        >>> config = BaseConfig(state_dim=-1, action_dim=2)
        >>> config.validate()
        ValueError: state_dim must be positive, got -1
        """
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")

        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")

        if self.max_action <= 0:
            raise ValueError(f"max_action must be positive, got {self.max_action}")

        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")

        if any(h <= 0 for h in self.hidden_dims):
            raise ValueError(f"All hidden_dims must be positive: {self.hidden_dims}")

        if self.lr_actor <= 0:
            raise ValueError(f"lr_actor must be positive, got {self.lr_actor}")

        if self.lr_critic <= 0:
            raise ValueError(f"lr_critic must be positive, got {self.lr_critic}")

        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")

        if not 0 < self.tau <= 1:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")

        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {self.buffer_size}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.batch_size > self.buffer_size:
            raise ValueError(
                f"batch_size ({self.batch_size}) cannot exceed "
                f"buffer_size ({self.buffer_size})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Configuration as dictionary for serialization.
        """
        return asdict(self)

    def to_json(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """
        Create configuration from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        BaseConfig
            Instantiated configuration.
        """
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "BaseConfig":
        """
        Load configuration from JSON file.

        Parameters
        ----------
        path : str
            Input file path.

        Returns
        -------
        BaseConfig
            Loaded configuration.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __post_init__(self) -> None:
        """Set random seed if specified."""
        if self.seed is not None:
            self._set_seed(self.seed)

    def _set_seed(self, seed: int) -> None:
        """
        Set random seeds for reproducibility.

        Parameters
        ----------
        seed : int
            Random seed value.
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Unit tests for configuration module
    import tempfile
    import os

    print("Testing BaseConfig...")

    # Test 1: Basic instantiation
    config = BaseConfig(state_dim=4, action_dim=2)
    assert config.state_dim == 4
    assert config.action_dim == 2
    assert config.gamma == 0.99
    print("  [PASS] Basic instantiation")

    # Test 2: Device selection
    device = config.get_device()
    assert isinstance(device, torch.device)
    print(f"  [PASS] Device selection: {device}")

    # Test 3: Validation - valid config
    valid_config = BaseConfig(state_dim=10, action_dim=3, max_action=2.0)
    valid_config.validate()
    print("  [PASS] Valid config validation")

    # Test 4: Validation - invalid config
    try:
        invalid_config = BaseConfig(state_dim=-1, action_dim=2)
        invalid_config.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "state_dim" in str(e)
        print("  [PASS] Invalid config raises ValueError")

    # Test 5: Serialization
    config = BaseConfig(state_dim=8, action_dim=4, gamma=0.95)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        config.to_json(temp_path)
        loaded = BaseConfig.from_json(temp_path)
        assert loaded.state_dim == 8
        assert loaded.action_dim == 4
        assert loaded.gamma == 0.95
        print("  [PASS] JSON serialization/deserialization")
    finally:
        os.unlink(temp_path)

    # Test 6: Dictionary conversion
    d = config.to_dict()
    assert isinstance(d, dict)
    assert d["state_dim"] == 8
    reconstructed = BaseConfig.from_dict(d)
    assert reconstructed.state_dim == 8
    print("  [PASS] Dictionary conversion")

    print("\nAll BaseConfig tests passed!")
