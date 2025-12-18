"""
Core Module - Configuration and Data Structures.

This module provides foundational components for DQN variants:
    - DQNVariantConfig: Hyperparameter configuration with validation
    - DQNVariant: Enumeration of available algorithm variants
    - Transition: Single-step transition data structure
    - NStepTransition: Multi-step transition for n-step learning

Core Idea (核心思想)
====================
集中管理所有DQN变体的配置参数，确保类型安全和参数验证。
使用dataclass和enum实现强类型约束，提高代码可维护性。

Example:
    >>> from core import DQNVariantConfig, DQNVariant
    >>> config = DQNVariantConfig(state_dim=4, action_dim=2)
    >>> config.get_device()
    device(type='cuda')
"""

from core.config import DQNVariantConfig
from core.types import Transition, NStepTransition, FloatArray, IntArray
from core.enums import DQNVariant

__all__ = [
    "DQNVariantConfig",
    "DQNVariant",
    "Transition",
    "NStepTransition",
    "FloatArray",
    "IntArray",
]
