"""
Uniform Experience Replay Buffer.

This module implements a standard replay buffer with O(1) operations.

Core Idea (核心思想)
====================
经验回放通过存储和随机采样历史交互数据，打破样本间的时序相关性，
使得神经网络训练更加稳定。

Mathematical Foundation (数学基础)
==================================
Uniform sampling probability:
    P(i) = 1/|D|, ∀i ∈ D

Expected gradient with replay buffer:
    ∇_θ L = E_{(s,a,r,s')~U(D)} [(Q(s,a;θ) - y)²]

Complexity Analysis (复杂度分析)
================================
+------------+------------+----------------------------------+
| Operation  | Complexity | Notes                            |
+============+============+==================================+
| push()     | O(1)*      | Amortized; may trigger eviction  |
+------------+------------+----------------------------------+
| sample()   | O(B)       | B = batch_size                   |
+------------+------------+----------------------------------+
| __len__()  | O(1)       | Cached by deque                  |
+------------+------------+----------------------------------+
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import Transition, FloatArray, IntArray


class ReplayBuffer:
    """
    Uniform experience replay buffer with O(1) operations.

    Core Idea (核心思想)
    --------------------
    经验回放是DQN的核心创新之一，解决了两个关键问题：

    1. **打破时序相关性**: 连续采样的样本高度相关，违反SGD的i.i.d.假设。
       随机采样打破时序关联，提供更稳定的梯度估计。

    2. **提高数据效率**: 每个经验可被多次使用，而非"用后即弃"。
       这对于数据收集成本高的RL场景尤为重要。

    Mathematical Foundation (数学基础)
    ----------------------------------
    Uniform sampling probability:

    .. math::
        P(i) = \\frac{1}{|\\mathcal{D}|}, \\quad \\forall i \\in \\mathcal{D}

    Expected gradient with replay buffer:

    .. math::
        \\nabla_\\theta L = \\mathbb{E}_{(s,a,r,s') \\sim \\mathcal{U}(\\mathcal{D})}
        \\left[ (Q(s,a;\\theta) - y)^2 \\right]

    Implementation Details (实现细节)
    ---------------------------------
    - Uses ``collections.deque`` with ``maxlen`` for automatic FIFO eviction
    - ``__slots__`` reduces memory footprint per instance
    - Vectorized NumPy operations for batch sampling

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store. Must be positive.
        When full, oldest transitions are evicted (FIFO policy).

    Attributes
    ----------
    capacity : int
        Maximum buffer size (read-only property)

    Raises
    ------
    ValueError
        If ``capacity <= 0``

    Examples
    --------
    Basic usage:

    >>> buffer = ReplayBuffer(capacity=10000)
    >>> state = np.array([1.0, 2.0, 3.0, 4.0])
    >>> buffer.push(state, 0, 1.0, state, False)
    >>> len(buffer)
    1

    Batch sampling:

    >>> for _ in range(100):
    ...     buffer.push(state, 0, 1.0, state, False)
    >>> states, actions, rewards, next_states, dones = buffer.sample(32)
    >>> states.shape
    (32, 4)

    See Also
    --------
    PrioritizedReplayBuffer : Priority-based sampling for improved efficiency
    NStepReplayBuffer : Multi-step returns for faster credit assignment
    """

    __slots__ = ("_capacity", "_buffer")

    def __init__(self, capacity: int) -> None:
        """
        Initialize uniform replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store. Must be positive.

        Raises
        ------
        ValueError
            If capacity is not a positive integer.
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(
                f"capacity must be a positive integer, got {capacity!r} "
                f"(type: {type(capacity).__name__})"
            )
        self._capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        """
        Maximum buffer capacity (read-only).

        Returns
        -------
        int
            Maximum number of transitions the buffer can hold
        """
        return self._capacity

    def __len__(self) -> int:
        """
        Return current number of stored transitions.

        Returns
        -------
        int
            Number of transitions currently in buffer
        """
        return len(self._buffer)

    def push(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> None:
        """
        Store a single transition in the buffer.

        Parameters
        ----------
        state : FloatArray
            Current state observation, shape (state_dim,)
        action : int
            Discrete action index
        reward : float
            Immediate reward received
        next_state : FloatArray
            Next state observation, shape (state_dim,)
        done : bool
            Whether episode terminated after this transition

        Notes
        -----
        - O(1) amortized time complexity
        - Automatically evicts oldest transition when at capacity (FIFO)

        Examples
        --------
        >>> buffer = ReplayBuffer(capacity=100)
        >>> state = np.zeros(4)
        >>> buffer.push(state, 0, 1.0, state, False)
        >>> len(buffer)
        1
        """
        self._buffer.append(
            Transition(state, action, reward, next_state, done)
        )

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[FloatArray, IntArray, FloatArray, FloatArray, FloatArray]:
        """
        Sample a uniform random mini-batch of transitions.

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
            Batch of done flags (0.0 or 1.0), shape (batch_size,)

        Raises
        ------
        ValueError
            If batch_size exceeds current buffer size

        Notes
        -----
        - Samples without replacement within each call
        - Returns NumPy arrays for efficient tensor conversion

        Examples
        --------
        >>> buffer = ReplayBuffer(capacity=1000)
        >>> for _ in range(100):
        ...     buffer.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
        >>> states, actions, rewards, next_states, dones = buffer.sample(32)
        >>> states.shape
        (32, 4)
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size "
                f"({len(self._buffer)}). Ensure buffer has at least "
                f"{batch_size} transitions before sampling."
            )

        batch = random.sample(list(self._buffer), batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has sufficient samples for training.

        Parameters
        ----------
        min_size : int
            Minimum required number of transitions

        Returns
        -------
        bool
            True if len(buffer) >= min_size

        Examples
        --------
        >>> buffer = ReplayBuffer(capacity=1000)
        >>> buffer.is_ready(100)
        False
        >>> for _ in range(100):
        ...     buffer.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
        >>> buffer.is_ready(100)
        True
        """
        return len(self._buffer) >= min_size

    def clear(self) -> None:
        """
        Remove all transitions from buffer.

        Examples
        --------
        >>> buffer = ReplayBuffer(capacity=100)
        >>> buffer.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
        >>> buffer.clear()
        >>> len(buffer)
        0
        """
        self._buffer.clear()
