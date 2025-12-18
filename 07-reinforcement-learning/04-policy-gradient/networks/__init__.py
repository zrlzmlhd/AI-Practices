"""
Networks Module - Policy Gradient Methods

Provides neural network architectures for policy gradient algorithms.

Exports:
    - init_weights: Orthogonal weight initialization
    - MLP: Multi-layer perceptron base module
    - DiscretePolicy: Softmax policy for discrete actions
    - ContinuousPolicy: Gaussian policy for continuous actions
    - ValueNetwork: State value function V(s)
    - ActorCriticNetwork: Shared-feature actor-critic architecture
"""

from networks.base import init_weights, MLP
from networks.policy import DiscretePolicy, ContinuousPolicy
from networks.value import ValueNetwork, ActorCriticNetwork

__all__ = [
    "init_weights",
    "MLP",
    "DiscretePolicy",
    "ContinuousPolicy",
    "ValueNetwork",
    "ActorCriticNetwork",
]
