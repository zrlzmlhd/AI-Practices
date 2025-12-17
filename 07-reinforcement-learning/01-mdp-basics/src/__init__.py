"""
MDP Basics: Markov Decision Process and Dynamic Programming

Production-grade implementation of classical dynamic programming algorithms
for solving Markov Decision Processes.

Modules:
    environment: MDP environment abstractions and GridWorld implementation
    algorithms: Policy Evaluation, Policy Iteration, Value Iteration
    executor: Policy execution and evaluation utilities
    visualization: Rendering utilities for policies and value functions

References:
    [1] Sutton & Barto, "Reinforcement Learning: An Introduction", 2018
    [2] Bellman, R. "Dynamic Programming", Princeton University Press, 1957
    [3] Howard, R. "Dynamic Programming and Markov Processes", MIT Press, 1960
"""

from .environment import (
    MDPEnvironment,
    GridWorld,
    GridWorldConfig,
    State,
    Action,
    Policy,
    ValueFunction,
    QFunction,
)
from .algorithms import DynamicProgrammingSolver, AlgorithmResult
from .executor import PolicyExecutor

__version__ = "1.0.0"
__author__ = "AI-Practices"

__all__ = [
    "MDPEnvironment",
    "GridWorld",
    "GridWorldConfig",
    "DynamicProgrammingSolver",
    "AlgorithmResult",
    "PolicyExecutor",
    "State",
    "Action",
    "Policy",
    "ValueFunction",
    "QFunction",
]
