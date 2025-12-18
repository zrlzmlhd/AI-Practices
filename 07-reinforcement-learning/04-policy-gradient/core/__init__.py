"""
Core Module - Policy Gradient Methods

Provides fundamental data structures and configurations for policy gradient algorithms.

Exports:
    - TrainingConfig: Hyperparameter configuration dataclass
    - EpisodeBuffer: Episode trajectory storage
    - Transition: Single-step transition tuple
"""

from core.config import TrainingConfig
from core.buffers import EpisodeBuffer, Transition

__all__ = [
    "TrainingConfig",
    "EpisodeBuffer",
    "Transition",
]
