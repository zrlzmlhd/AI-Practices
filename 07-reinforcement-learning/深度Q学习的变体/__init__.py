"""
Deep Q-Network Variants Module.

This package provides production-ready implementations of DQN variants
for reinforcement learning research and applications.

Algorithms Implemented (已实现算法)
===================================
+------------------+--------------------------------------------------+
| Variant          | Key Innovation                                   |
+==================+==================================================+
| VANILLA          | Original DQN (Mnih et al., 2015)                 |
+------------------+--------------------------------------------------+
| DOUBLE           | Decoupled action selection/evaluation            |
+------------------+--------------------------------------------------+
| DUELING          | Value-advantage decomposition architecture       |
+------------------+--------------------------------------------------+
| NOISY            | Parametric exploration via noisy layers          |
+------------------+--------------------------------------------------+
| CATEGORICAL      | Full return distribution modeling (C51)          |
+------------------+--------------------------------------------------+
| RAINBOW          | All improvements combined                        |
+------------------+--------------------------------------------------+

Module Structure (模块结构)
===========================
::

    深度Q学习的变体/
    ├── core/           Configuration and data structures
    │   ├── config.py   DQNVariantConfig
    │   ├── enums.py    DQNVariant enumeration
    │   └── types.py    Transition, NStepTransition
    ├── buffers/        Replay buffer implementations
    │   ├── base.py     ReplayBuffer (uniform)
    │   ├── sum_tree.py SumTree data structure
    │   ├── prioritized.py  PrioritizedReplayBuffer
    │   └── n_step.py   NStepReplayBuffer
    ├── networks/       Neural network architectures
    │   ├── base.py     DQNNetwork
    │   ├── dueling.py  DuelingNetwork
    │   ├── noisy.py    NoisyLinear, NoisyNetwork
    │   ├── categorical.py  CategoricalNetwork (C51)
    │   └── rainbow.py  RainbowNetwork
    ├── agents/         Agent implementations
    │   └── variant_agent.py  DQNVariantAgent
    ├── utils/          Training utilities
    │   ├── training.py     train_agent, evaluate_agent
    │   ├── comparison.py   compare_variants
    │   └── visualization.py  plotting functions
    └── tests/          Unit tests

Quick Start (快速开始)
======================
>>> from dqn_variants import DQNVariantAgent, DQNVariantConfig, DQNVariant
>>> from dqn_variants import train_agent, evaluate_agent
>>>
>>> # Create agent
>>> config = DQNVariantConfig(state_dim=4, action_dim=2)
>>> agent = DQNVariantAgent(config, DQNVariant.RAINBOW)
>>>
>>> # Train
>>> agent, logger = train_agent(agent, "CartPole-v1", num_episodes=500)
>>>
>>> # Evaluate
>>> mean_reward, std_reward = evaluate_agent(agent, "CartPole-v1")
>>> print(f"Performance: {mean_reward:.1f} ± {std_reward:.1f}")

References
==========
[1] Mnih, V. et al. (2015). Human-level control through deep RL. Nature.
[2] van Hasselt, H. et al. (2016). Deep RL with Double Q-learning. AAAI.
[3] Wang, Z. et al. (2016). Dueling Network Architectures. ICML.
[4] Fortunato, M. et al. (2017). Noisy Networks for Exploration. ICLR.
[5] Bellemare, M. et al. (2017). A Distributional Perspective on RL. ICML.
[6] Schaul, T. et al. (2016). Prioritized Experience Replay. ICLR.
[7] Hessel, M. et al. (2018). Rainbow: Combining Improvements. AAAI.
"""

# Core components
from core.config import DQNVariantConfig
from core.enums import DQNVariant
from core.types import Transition, NStepTransition

# Buffers
from buffers.base import ReplayBuffer
from buffers.sum_tree import SumTree
from buffers.prioritized import PrioritizedReplayBuffer
from buffers.n_step import NStepReplayBuffer

# Networks
from networks.base import DQNNetwork, init_weights
from networks.dueling import DuelingNetwork
from networks.noisy import NoisyLinear, NoisyNetwork, NoisyDuelingNetwork
from networks.categorical import CategoricalNetwork, CategoricalDuelingNetwork
from networks.rainbow import RainbowNetwork

# Agents
from agents.variant_agent import DQNVariantAgent

# Utilities
from utils.training import TrainingLogger, train_agent, evaluate_agent
from utils.comparison import compare_variants
from utils.visualization import (
    plot_training_curves,
    plot_comparison,
    plot_ablation_study,
    plot_distribution,
)

__version__ = "1.0.0"
__author__ = "Reinforcement Learning Research Group"

__all__ = [
    # Core
    "DQNVariantConfig",
    "DQNVariant",
    "Transition",
    "NStepTransition",
    # Buffers
    "ReplayBuffer",
    "SumTree",
    "PrioritizedReplayBuffer",
    "NStepReplayBuffer",
    # Networks
    "DQNNetwork",
    "init_weights",
    "DuelingNetwork",
    "NoisyLinear",
    "NoisyNetwork",
    "NoisyDuelingNetwork",
    "CategoricalNetwork",
    "CategoricalDuelingNetwork",
    "RainbowNetwork",
    # Agents
    "DQNVariantAgent",
    # Utilities
    "TrainingLogger",
    "train_agent",
    "evaluate_agent",
    "compare_variants",
    "plot_training_curves",
    "plot_comparison",
    "plot_ablation_study",
    "plot_distribution",
]
