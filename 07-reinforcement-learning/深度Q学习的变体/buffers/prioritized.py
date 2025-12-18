"""
Prioritized Experience Replay Buffer.

This module implements PER (Schaul et al., 2016) using SumTree.

Core Idea (核心思想)
====================
优先经验回放通过TD误差大小对样本进行优先采样，使得学习更加高效：

1. **重要性采样**: 高TD误差的样本包含更多学习信号
2. **偏差校正**: 使用重要性采样权重保持梯度无偏

Mathematical Foundation (数学基础)
==================================
Priority Definition:
    p_i = |δ_i| + ε

where δ_i is TD error and ε prevents zero priority.

Sampling Probability:
    P(i) = p_i^α / Σ_k p_k^α

- α = 0: Uniform sampling (ignores priorities)
- α = 1: Full prioritization

Importance Sampling Weights (unbias gradients):
    w_i = (1 / (N · P(i)))^β / max_j w_j

β anneals from β₀ to 1 over training to fully correct bias.

References:
    Schaul, T. et al. (2016). Prioritized Experience Replay. ICLR.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import Transition, FloatArray, IntArray
from buffers.sum_tree import SumTree


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.

    Core Idea (核心思想)
    --------------------
    优先经验回放通过TD误差大小对样本进行优先采样，使得学习更加高效：

    1. **重要性采样**: 高TD误差的样本包含更多学习信号
    2. **偏差校正**: 使用重要性采样权重保持梯度无偏

    Mathematical Foundation (数学基础)
    ----------------------------------
    **Priority Definition**:

    .. math::
        p_i = |\\delta_i| + \\epsilon

    where δ_i is TD error and ε prevents zero priority.

    **Sampling Probability**:

    .. math::
        P(i) = \\frac{p_i^\\alpha}{\\sum_k p_k^\\alpha}

    - α = 0: Uniform sampling (ignores priorities)
    - α = 1: Full prioritization

    **Importance Sampling Weights** (unbias gradients):

    .. math::
        w_i = \\left( \\frac{1}{N \\cdot P(i)} \\right)^\\beta / \\max_j w_j

    β anneals from β₀ to 1 over training to fully correct bias.

    Implementation Details (实现细节)
    ---------------------------------
    - Uses SumTree for O(log N) proportional sampling
    - Stratified sampling: divides priority range into equal segments
    - New transitions get max priority for guaranteed initial sampling

    Complexity Analysis (复杂度分析)
    --------------------------------
    +------------------+------------+----------------------------------+
    | Operation        | Complexity | Notes                            |
    +==================+============+==================================+
    | push()           | O(log N)   | Add to sum tree                  |
    +------------------+------------+----------------------------------+
    | sample()         | O(B log N) | B stratified samples             |
    +------------------+------------+----------------------------------+
    | update_priorities| O(B log N) | Update B priorities              |
    +------------------+------------+----------------------------------+

    Parameters
    ----------
    capacity : int
        Maximum buffer size
    alpha : float, default=0.6
        Prioritization exponent. 0 = uniform, 1 = full priority.
    beta_start : float, default=0.4
        Initial importance sampling exponent
    beta_frames : int, default=100000
        Frames over which to anneal β from beta_start to 1.0
    epsilon : float, default=1e-6
        Small constant preventing zero priority

    Attributes
    ----------
    capacity : int
        Maximum buffer size
    beta : float
        Current importance sampling exponent

    Raises
    ------
    ValueError
        If alpha or beta_start not in [0, 1]

    Examples
    --------
    >>> buffer = PrioritizedReplayBuffer(capacity=10000)
    >>> buffer.push(state, action, reward, next_state, done)
    >>> batch = buffer.sample(64)
    >>> states, actions, rewards, next_states, dones, indices, weights = batch
    >>> buffer.update_priorities(indices, td_errors)

    References
    ----------
    Schaul, T. et al. (2016). Prioritized Experience Replay. ICLR.

    See Also
    --------
    ReplayBuffer : Uniform sampling baseline
    SumTree : Underlying data structure
    """

    __slots__ = (
        "_capacity", "_alpha", "_beta", "_beta_start", "_beta_frames",
        "_epsilon", "_sum_tree", "_max_priority", "_frame",
    )

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
    ) -> None:
        """
        Initialize prioritized replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store
        alpha : float, default=0.6
            Prioritization exponent in [0, 1].
            0 = uniform sampling, 1 = full prioritization.
        beta_start : float, default=0.4
            Initial importance sampling exponent in [0, 1].
            Anneals to 1.0 over training.
        beta_frames : int, default=100000
            Number of frames to anneal β from beta_start to 1.0
        epsilon : float, default=1e-6
            Small constant added to priorities to prevent zero priority

        Raises
        ------
        ValueError
            If alpha or beta_start not in [0, 1], or capacity not positive
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(
                f"capacity must be positive integer, got {capacity!r}"
            )
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 <= beta_start <= 1:
            raise ValueError(f"beta_start must be in [0, 1], got {beta_start}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self._capacity = capacity
        self._alpha = alpha
        self._beta_start = beta_start
        self._beta = beta_start
        self._beta_frames = max(1, beta_frames)
        self._epsilon = epsilon

        self._sum_tree = SumTree(capacity)
        self._max_priority = 1.0
        self._frame = 0

    @property
    def capacity(self) -> int:
        """Maximum buffer size (read-only)."""
        return self._capacity

    @property
    def beta(self) -> float:
        """Current importance sampling exponent."""
        return self._beta

    @property
    def alpha(self) -> float:
        """Prioritization exponent."""
        return self._alpha

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self._sum_tree)

    def push(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> None:
        """
        Store transition with maximum priority.

        New transitions receive max priority to guarantee they are
        sampled at least once before priority updates.

        Parameters
        ----------
        state : FloatArray
            Current state observation
        action : int
            Discrete action index
        reward : float
            Immediate reward
        next_state : FloatArray
            Next state observation
        done : bool
            Episode termination flag

        Examples
        --------
        >>> buffer = PrioritizedReplayBuffer(capacity=1000)
        >>> buffer.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
        >>> len(buffer)
        1
        """
        transition = Transition(state, action, reward, next_state, done)
        priority = self._max_priority ** self._alpha
        self._sum_tree.add(priority, transition)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[FloatArray, IntArray, FloatArray, FloatArray, FloatArray, IntArray, FloatArray]:
        """
        Sample batch with prioritized probabilities.

        Uses stratified sampling: divides total priority into B equal
        segments and samples one transition from each segment.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample

        Returns
        -------
        states : FloatArray
            Batch of states, shape (batch_size, state_dim)
        actions : IntArray
            Batch of actions, shape (batch_size,)
        rewards : FloatArray
            Batch of rewards, shape (batch_size,)
        next_states : FloatArray
            Batch of next states, shape (batch_size, state_dim)
        dones : FloatArray
            Batch of done flags, shape (batch_size,)
        indices : IntArray
            Tree indices for priority updates, shape (batch_size,)
        weights : FloatArray
            Importance sampling weights, shape (batch_size,)

        Raises
        ------
        ValueError
            If batch_size exceeds buffer size

        Examples
        --------
        >>> buffer = PrioritizedReplayBuffer(capacity=1000)
        >>> for _ in range(100):
        ...     buffer.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
        >>> batch = buffer.sample(32)
        >>> states, actions, rewards, next_states, dones, indices, weights = batch
        >>> weights.shape
        (32,)
        """
        buffer_len = len(self._sum_tree)
        if batch_size > buffer_len:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({buffer_len})"
            )

        self._anneal_beta()

        indices = np.empty(batch_size, dtype=np.int64)
        weights = np.empty(batch_size, dtype=np.float32)
        batch: List[Transition] = []

        total = self._sum_tree.total_priority
        segment = total / batch_size

        min_prob = self._sum_tree.min_priority() ** self._alpha / total
        max_weight = (buffer_len * min_prob) ** (-self._beta) if min_prob > 0 else 1.0

        for i in range(batch_size):
            low, high = segment * i, segment * (i + 1)
            cumsum = random.uniform(low, high)
            tree_idx, priority, data = self._sum_tree.get(cumsum)

            prob = priority / total
            weight = (buffer_len * prob) ** (-self._beta) / max_weight

            indices[i] = tree_idx
            weights[i] = weight
            batch.append(data)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def _anneal_beta(self) -> None:
        """
        Linearly anneal β from beta_start to 1.0 over beta_frames.

        This gradually increases the importance of the IS correction
        as training progresses, ensuring unbiased gradients at convergence.
        """
        self._frame += 1
        fraction = min(1.0, self._frame / self._beta_frames)
        self._beta = self._beta_start + fraction * (1.0 - self._beta_start)

    def update_priorities(
        self,
        indices: IntArray,
        td_errors: FloatArray,
    ) -> None:
        """
        Update priorities based on TD errors.

        Priority formula: p_i = (|δ_i| + ε)^α

        Parameters
        ----------
        indices : IntArray
            Tree indices from sample(), shape (batch_size,)
        td_errors : FloatArray
            TD errors for each transition, shape (batch_size,)

        Examples
        --------
        >>> buffer = PrioritizedReplayBuffer(capacity=1000)
        >>> # ... push transitions and sample ...
        >>> td_errors = np.random.randn(32)
        >>> buffer.update_priorities(indices, td_errors)
        """
        priorities = (np.abs(td_errors) + self._epsilon) ** self._alpha

        for idx, priority in zip(indices, priorities):
            self._sum_tree.update_priority(int(idx), float(priority))

        self._max_priority = max(
            self._max_priority,
            float(np.max(priorities)) ** (1.0 / self._alpha)
        )

    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has sufficient samples.

        Parameters
        ----------
        min_size : int
            Minimum required buffer size

        Returns
        -------
        bool
            True if buffer size >= min_size
        """
        return len(self._sum_tree) >= min_size
