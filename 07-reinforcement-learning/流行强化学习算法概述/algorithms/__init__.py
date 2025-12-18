"""
Algorithms Module
=================

State-of-the-art continuous control reinforcement learning algorithms.

Algorithms
----------
- DDPG: Deep Deterministic Policy Gradient
- TD3: Twin Delayed DDPG
- SAC: Soft Actor-Critic

All algorithms share a common interface defined by BaseAgent,
enabling polymorphic training and fair comparison.
"""

from .ddpg import DDPGConfig, DDPGAgent
from .td3 import TD3Config, TD3Agent
from .sac import SACConfig, SACAgent

__all__ = [
    "DDPGConfig",
    "DDPGAgent",
    "TD3Config",
    "TD3Agent",
    "SACConfig",
    "SACAgent",
]
