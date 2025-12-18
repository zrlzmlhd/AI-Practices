"""
N-Step Experience Replay Buffer.

This module implements multi-step learning for faster credit assignment.

Core Idea (核心思想)
====================
N步回报通过在多步之后再进行bootstrap，实现了TD(0)和Monte Carlo之间的折中：

- **更快的信用分配**: 奖励信号传播更远
- **降低偏差**: 使用更多真实回报，减少bootstrap偏差
- **增加方差**: 引入更多随机性

Mathematical Foundation (数学基础)
==================================
N-step Return:
    G_t^{(n)} = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n})

N-step TD Target:
    y_t^{(n)} = G_t^{(n)} = Σ_{k=0}^{n-1} γ^k r_{t+k+1} + γ^n max_a Q(s_{t+n}, a)

Bias-Variance Trade-off:
+-------+------------------+------------------+
| n     | Bias             | Variance         |
+=======+==================+==================+
| 1     | High (bootstrap) | Low              |
+-------+------------------+------------------+
| 3-5   | Medium           | Medium           |
+-------+------------------+------------------+
| ∞     | Zero (MC)        | High             |
+-------+------------------+------------------+

References:
    Sutton, R. S. (1988). Learning to predict by temporal differences.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import Transition, NStepTransition, FloatArray, IntArray


class NStepReplayBuffer:
    """
    N-step experience replay buffer for multi-step learning.

    Core Idea (核心思想)
    --------------------
    N步回报通过在多步之后再进行bootstrap，实现了TD(0)和Monte Carlo之间的折中：

    - **更快的信用分配**: 奖励信号传播更远
    - **降低偏差**: 使用更多真实回报，减少bootstrap偏差
    - **增加方差**: 引入更多随机性

    Mathematical Foundation (数学基础)
    ----------------------------------
    **N-step Return**:

    .. math::
        G_t^{(n)} = \\sum_{k=0}^{n-1} \\gamma^k R_{t+k+1} + \\gamma^n V(S_{t+n})

    **N-step TD Target**:

    .. math::
        y_t^{(n)} = G_t^{(n)} = \\sum_{k=0}^{n-1} \\gamma^k r_{t+k+1} + \\gamma^n \\max_{a'} Q(s_{t+n}, a')

    **Bias-Variance Trade-off**:

    +-------+------------------+------------------+
    | n     | Bias             | Variance         |
    +=======+==================+==================+
    | 1     | High (bootstrap) | Low              |
    +-------+------------------+------------------+
    | 3-5   | Medium           | Medium           |
    +-------+------------------+------------------+
    | ∞     | Zero (MC)        | High             |
    +-------+------------------+------------------+

    Parameters
    ----------
    capacity : int
        Maximum number of n-step transitions to store
    n_steps : int, default=3
        Number of steps for returns calculation
    gamma : float, default=0.99
        Discount factor

    Attributes
    ----------
    capacity : int
        Maximum buffer size
    n_steps : int
        Number of steps for multi-step returns

    Examples
    --------
    >>> buffer = NStepReplayBuffer(capacity=10000, n_steps=3, gamma=0.99)
    >>> buffer.push(state, action, reward, next_state, done)
    >>> batch = buffer.sample(64)
    >>> states, actions, returns, next_states, dones, actual_n_steps = batch

    References
    ----------
    Sutton, R. S. (1988). Learning to predict by the methods of temporal
    differences. Machine learning, 3(1):9-44.
    """

    __slots__ = (
        "_capacity", "_n_steps", "_gamma", "_buffer", "_n_step_buffer",
    )

    def __init__(
        self,
        capacity: int,
        n_steps: int = 3,
        gamma: float = 0.99,
    ) -> None:
        """
        Initialize n-step replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of n-step transitions to store
        n_steps : int, default=3
            Number of steps for return calculation. Must be >= 1.
        gamma : float, default=0.99
            Discount factor for return calculation. Must be in [0, 1].

        Raises
        ------
        ValueError
            If n_steps < 1 or gamma not in [0, 1]
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(
                f"capacity must be positive integer, got {capacity!r}"
            )
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be integer >= 1, got {n_steps}")
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        self._capacity = capacity
        self._n_steps = n_steps
        self._gamma = gamma

        self._buffer: Deque[NStepTransition] = deque(maxlen=capacity)
        self._n_step_buffer: Deque[Transition] = deque(maxlen=n_steps)

    @property
    def capacity(self) -> int:
        """Maximum buffer size (read-only)."""
        return self._capacity

    @property
    def n_steps(self) -> int:
        """Number of steps for multi-step returns."""
        return self._n_steps

    @property
    def gamma(self) -> float:
        """Discount factor."""
        return self._gamma

    def __len__(self) -> int:
        """Current number of stored n-step transitions."""
        return len(self._buffer)

    def push(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> Optional[NStepTransition]:
        """
        Add transition and compute n-step return when ready.

        Accumulates single-step transitions until n steps collected,
        then computes discounted n-step return and stores the result.

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

        Returns
        -------
        Optional[NStepTransition]
            N-step transition if n steps accumulated, else None.
            Also returns early if episode terminates.

        Examples
        --------
        >>> buffer = NStepReplayBuffer(capacity=100, n_steps=3)
        >>> result = buffer.push(state, 0, 1.0, next_state, False)
        >>> result is None  # Not enough steps yet
        True
        >>> for _ in range(2):
        ...     result = buffer.push(state, 0, 1.0, next_state, False)
        >>> result is not None  # Now we have 3 steps
        True
        """
        self._n_step_buffer.append(
            Transition(state, action, reward, next_state, done)
        )

        if len(self._n_step_buffer) < self._n_steps:
            if done:
                return self._make_n_step_transition()
            return None

        return self._make_n_step_transition()

    def _make_n_step_transition(self) -> NStepTransition:
        """
        Compute n-step return and create transition.

        Returns
        -------
        NStepTransition
            Computed n-step transition
        """
        n_step_return = 0.0
        gamma_power = 1.0

        for t in self._n_step_buffer:
            n_step_return += gamma_power * t.reward
            gamma_power *= self._gamma
            if t.done:
                break

        first = self._n_step_buffer[0]
        last = self._n_step_buffer[-1]

        n_step = NStepTransition(
            state=first.state,
            action=first.action,
            n_step_return=n_step_return,
            next_state=last.next_state,
            done=last.done,
            n_steps=len(self._n_step_buffer),
        )

        self._buffer.append(n_step)

        if last.done:
            self._n_step_buffer.clear()
        else:
            self._n_step_buffer.popleft()

        return n_step

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[FloatArray, IntArray, FloatArray, FloatArray, FloatArray, IntArray]:
        """
        Sample n-step transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample

        Returns
        -------
        states : FloatArray
            Batch of initial states, shape (batch_size, state_dim)
        actions : IntArray
            Batch of actions, shape (batch_size,)
        n_step_returns : FloatArray
            Batch of n-step returns, shape (batch_size,)
        next_states : FloatArray
            Batch of states n steps ahead, shape (batch_size, state_dim)
        dones : FloatArray
            Batch of done flags, shape (batch_size,)
        n_steps : IntArray
            Actual number of steps for each transition, shape (batch_size,)

        Raises
        ------
        ValueError
            If batch_size exceeds buffer size

        Examples
        --------
        >>> buffer = NStepReplayBuffer(capacity=1000, n_steps=3)
        >>> # ... push transitions ...
        >>> batch = buffer.sample(32)
        >>> states, actions, returns, next_states, dones, actual_steps = batch
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size {batch_size} exceeds buffer size {len(self._buffer)}"
            )

        batch = random.sample(list(self._buffer), batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        returns = np.array([t.n_step_return for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        n_steps = np.array([t.n_steps for t in batch], dtype=np.int64)

        return states, actions, returns, next_states, dones, n_steps

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
        return len(self._buffer) >= min_size

    def reset_episode(self) -> None:
        """
        Clear n-step buffer at episode end.

        Should be called when an episode terminates to ensure the
        n-step buffer doesn't carry over partial sequences to the
        next episode.

        Examples
        --------
        >>> buffer = NStepReplayBuffer(capacity=100, n_steps=3)
        >>> # ... run episode ...
        >>> buffer.reset_episode()  # Call at episode end
        """
        self._n_step_buffer.clear()

    def clear(self) -> None:
        """Remove all transitions from buffer."""
        self._buffer.clear()
        self._n_step_buffer.clear()
