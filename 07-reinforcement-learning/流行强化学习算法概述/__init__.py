"""
流行强化学习算法概述 (Popular Reinforcement Learning Algorithms)
================================================================

Production-grade implementations of state-of-the-art continuous control
reinforcement learning algorithms.

Algorithms
----------
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed DDPG)
- SAC (Soft Actor-Critic)

Architecture
------------
::

    流行强化学习算法概述/
    ├── core/               # Core components: buffers, networks, base classes
    │   ├── config.py       # Base configuration
    │   ├── buffer.py       # Experience replay buffer
    │   ├── networks.py     # Neural network architectures
    │   └── base_agent.py   # Abstract agent interface
    │
    ├── algorithms/         # Algorithm implementations
    │   ├── ddpg.py        # Deep Deterministic Policy Gradient
    │   ├── td3.py         # Twin Delayed DDPG
    │   └── sac.py         # Soft Actor-Critic
    │
    ├── training/          # Training utilities
    │   ├── trainer.py     # Training loop
    │   ├── evaluator.py   # Evaluation utilities
    │   └── visualization.py # Plotting utilities
    │
    ├── tests/             # Unit tests
    └── notebooks/         # Interactive tutorials

Example
-------
>>> from 流行强化学习算法概述 import SACAgent, SACConfig, Trainer
>>> config = SACConfig(state_dim=3, action_dim=1, max_action=2.0)
>>> agent = SACAgent(config)
>>> trainer = Trainer(agent, env_name="Pendulum-v1")
>>> trainer.train(total_timesteps=100000)

References
----------
.. [1] Lillicrap et al., "Continuous control with deep reinforcement learning"
       ICLR 2016. https://arxiv.org/abs/1509.02971
.. [2] Fujimoto et al., "Addressing Function Approximation Error in
       Actor-Critic Methods" ICML 2018. https://arxiv.org/abs/1802.09477
.. [3] Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
       Deep Reinforcement Learning" ICML 2018. https://arxiv.org/abs/1801.01290
"""

from .core.config import BaseConfig, DeviceMixin
from .core.buffer import ReplayBuffer, Transition
from .core.networks import (
    DeterministicActor,
    GaussianActor,
    QNetwork,
    TwinQNetwork,
    ValueNetwork,
)
from .core.base_agent import BaseAgent

from .algorithms.ddpg import DDPGConfig, DDPGAgent
from .algorithms.td3 import TD3Config, TD3Agent
from .algorithms.sac import SACConfig, SACAgent

from .training.trainer import Trainer
from .training.evaluator import evaluate_agent, evaluate_episode
from .training.visualization import plot_training_curves, plot_comparison

__version__ = "1.0.0"
__author__ = "AI-Practices"

__all__ = [
    # Core
    "BaseConfig",
    "DeviceMixin",
    "ReplayBuffer",
    "Transition",
    "DeterministicActor",
    "GaussianActor",
    "QNetwork",
    "TwinQNetwork",
    "ValueNetwork",
    "BaseAgent",
    # Algorithms
    "DDPGConfig",
    "DDPGAgent",
    "TD3Config",
    "TD3Agent",
    "SACConfig",
    "SACAgent",
    # Training
    "Trainer",
    "evaluate_agent",
    "evaluate_episode",
    "plot_training_curves",
    "plot_comparison",
]
