"""
Policy Gradient Methods Module.

This package provides production-ready implementations of policy gradient
algorithms for reinforcement learning research and applications.

Algorithms Implemented:
    - REINFORCE: Monte Carlo policy gradient (Williams, 1992)
    - REINFORCEBaseline: REINFORCE with learned value baseline
    - A2C: Advantage Actor-Critic with GAE (Mnih et al., 2016)

Module Structure:
    core/           Configuration and data structures
    networks/       Neural network architectures
    algorithms/     Algorithm implementations
    utils/          Training and visualization utilities
    tests/          Unit tests

Quick Start:
    >>> from policy_gradient import REINFORCE, A2C, TrainingConfig
    >>> from policy_gradient import train_policy_gradient

    >>> config = TrainingConfig(gamma=0.99, lr_actor=1e-3)
    >>> agent = A2C(state_dim=4, action_dim=2, config=config)
    >>> agent, rewards = train_policy_gradient(agent, "CartPole-v1")

References:
    [1] Williams (1992). Simple statistical gradient-following algorithms.
    [2] Sutton et al. (1999). Policy gradient methods for RL with FA.
    [3] Schulman et al. (2016). High-dimensional continuous control using GAE.
    [4] Mnih et al. (2016). Asynchronous methods for deep RL.
"""

# Core components
from core.config import TrainingConfig
from core.buffers import EpisodeBuffer, Transition

# Neural networks
from networks.base import MLP, init_weights
from networks.policy import DiscretePolicy, ContinuousPolicy
from networks.value import ValueNetwork, ActorCriticNetwork

# Algorithms
from algorithms.base import BasePolicyGradient
from algorithms.reinforce import REINFORCE
from algorithms.reinforce_baseline import REINFORCEBaseline
from algorithms.a2c import A2C

# Utilities
from utils.returns import compute_returns, compute_gae, compute_n_step_returns
from utils.training import TrainingLogger, train_policy_gradient, evaluate_agent
from utils.visualization import plot_training_curves

__version__ = "1.0.0"
__author__ = "Reinforcement Learning Research Group"

__all__ = [
    # Core
    "TrainingConfig",
    "EpisodeBuffer",
    "Transition",
    # Networks
    "MLP",
    "init_weights",
    "DiscretePolicy",
    "ContinuousPolicy",
    "ValueNetwork",
    "ActorCriticNetwork",
    # Algorithms
    "BasePolicyGradient",
    "REINFORCE",
    "REINFORCEBaseline",
    "A2C",
    # Utilities
    "compute_returns",
    "compute_gae",
    "compute_n_step_returns",
    "TrainingLogger",
    "train_policy_gradient",
    "evaluate_agent",
    "plot_training_curves",
]
