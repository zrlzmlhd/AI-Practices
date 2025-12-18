"""
Core Module
===========

Foundational components shared across all reinforcement learning algorithms.

Components
----------
- BaseConfig: Base configuration class with common hyperparameters
- DeviceMixin: Automatic device selection (CUDA/MPS/CPU)
- ReplayBuffer: Experience replay memory
- Networks: Actor and Critic neural network architectures
- BaseAgent: Abstract base class defining the agent interface
"""

from .config import BaseConfig, DeviceMixin
from .buffer import ReplayBuffer, Transition
from .networks import (
    DeterministicActor,
    GaussianActor,
    QNetwork,
    TwinQNetwork,
    ValueNetwork,
    orthogonal_init,
    create_mlp,
)
from .base_agent import BaseAgent

__all__ = [
    "BaseConfig",
    "DeviceMixin",
    "ReplayBuffer",
    "Transition",
    "DeterministicActor",
    "GaussianActor",
    "QNetwork",
    "TwinQNetwork",
    "ValueNetwork",
    "orthogonal_init",
    "create_mlp",
    "BaseAgent",
]
