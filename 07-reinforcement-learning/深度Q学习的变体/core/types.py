"""
Type Definitions for DQN Variants.

This module defines type aliases and data structures used across the package.

Core Idea (核心思想)
====================
使用NamedTuple实现不可变的transition数据结构，结合NumPy类型别名
提供清晰的类型注解，支持静态类型检查和IDE自动补全。

Mathematical Definition (数学定义)
==================================
Transition表示MDP中的单步交互：

    τ_t = (s_t, a_t, r_t, s_{t+1}, d_t)

where:
    - s_t ∈ S: Current state
    - a_t ∈ A: Action taken
    - r_t ∈ ℝ: Immediate reward
    - s_{t+1} ∈ S: Next state
    - d_t ∈ {0, 1}: Terminal flag
"""

from __future__ import annotations

from typing import Any, NamedTuple
import numpy as np
from numpy.typing import NDArray


# Type Aliases
FloatArray = NDArray[np.floating[Any]]
"""Float-valued NumPy array for states, rewards, etc."""

IntArray = NDArray[np.int64]
"""Integer-valued NumPy array for actions, indices, etc."""


class Transition(NamedTuple):
    """
    Single-step transition tuple for experience replay.

    Core Idea (核心思想)
    --------------------
    表示MDP中单步交互的基本数据单元：τ = (s_t, a_t, r_t, s_{t+1}, done_t)。
    使用NamedTuple实现内存高效且不可变的数据存储。

    Mathematical Definition (数学定义)
    ----------------------------------
    A transition represents one step of the MDP:

    .. math::
        \\tau_t = (s_t, a_t, r_t, s_{t+1}, d_t)

    where:
        - s_t ∈ S: Current state
        - a_t ∈ A: Action taken
        - r_t ∈ ℝ: Immediate reward r(s_t, a_t)
        - s_{t+1} ∈ S: Next state from transition P(·|s_t, a_t)
        - d_t ∈ {0, 1}: Terminal flag

    Attributes
    ----------
    state : FloatArray
        Current observation s_t ∈ ℝ^d, shape (state_dim,)
    action : int
        Discrete action index a_t ∈ {0, ..., |A|-1}
    reward : float
        Scalar immediate reward r_t
    next_state : FloatArray
        Next observation s_{t+1} ∈ ℝ^d, shape (state_dim,)
    done : bool
        Episode termination flag (True if terminal state)

    Notes
    -----
    - NamedTuple provides immutability and memory efficiency
    - Fields are stored contiguously for cache-friendly access
    - Used as the atomic unit for all replay buffer implementations

    Examples
    --------
    >>> state = np.array([1.0, 2.0, 3.0, 4.0])
    >>> transition = Transition(state, 0, 1.0, state, False)
    >>> transition.action
    0
    """
    state: FloatArray
    action: int
    reward: float
    next_state: FloatArray
    done: bool


class NStepTransition(NamedTuple):
    """
    N-step transition for multi-step learning.

    Core Idea (核心思想)
    --------------------
    存储n步累积回报用于多步学习。相比单步TD，n步方法在偏差与方差之间
    提供了更灵活的权衡，通常可以加速学习。

    Mathematical Definition (数学定义)
    ----------------------------------
    N-step return from time t:

    .. math::
        G_t^{(n)} = \\sum_{k=0}^{n-1} \\gamma^k R_{t+k+1} + \\gamma^n V(S_{t+n})

    The transition stores:
        - Initial state s_t and action a_t
        - Cumulative discounted reward: Σ_{k=0}^{n-1} γ^k r_{t+k+1}
        - Bootstrap state s_{t+n} for value estimation

    Attributes
    ----------
    state : FloatArray
        Initial observation s_t ∈ ℝ^d at start of n-step sequence
    action : int
        Action a_t taken at initial state
    n_step_return : float
        Cumulative n-step discounted return: Σ_{k=0}^{n-1} γ^k r_{t+k+1}
    next_state : FloatArray
        State s_{t+n} for bootstrapping (n steps ahead)
    done : bool
        True if episode terminated within n steps
    n_steps : int
        Actual number of steps (may be < n if episode ended early)

    Notes
    -----
    - n_steps can be less than configured n when episode terminates early
    - When done=True, no bootstrapping is needed (terminal state)
    - Provides faster credit assignment compared to 1-step TD

    See Also
    --------
    NStepReplayBuffer : Buffer implementation for n-step transitions
    """
    state: FloatArray
    action: int
    n_step_return: float
    next_state: FloatArray
    done: bool
    n_steps: int
