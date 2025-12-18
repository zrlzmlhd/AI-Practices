"""
Experience Replay Buffer
========================

Core Idea
---------
Store and sample transitions (s, a, r, s', done) to break temporal
correlation in sequential data and enable efficient off-policy learning.

Mathematical Theory
-------------------
Experience replay addresses two critical issues in RL:

**1. Correlation Breaking**

Sequential samples from environment are temporally correlated:

.. math::

    \\text{Corr}(s_t, s_{t+1}) \\neq 0

This violates the i.i.d. assumption required by SGD. Random sampling from
a large buffer approximates i.i.d. data:

.. math::

    \\{(s_i, a_i, r_i, s'_i)\\}_{i=1}^{B} \\sim \\mathcal{D} \\quad \\text{uniformly}

**2. Sample Efficiency**

Each transition can be reused multiple times:

.. math::

    \\text{Data Efficiency} = \\frac{\\text{Gradient Updates}}{\\text{Environment Steps}} \\gg 1

A single environment step can contribute to many weight updates.

Problem Statement
-----------------
Without experience replay:

1. **Correlated Updates**: Sequential samples cause oscillating gradients
2. **Data Waste**: Each transition used only once then discarded
3. **Catastrophic Forgetting**: Recent experiences overwrite old knowledge

Experience replay solves all three by maintaining a diverse memory pool.

Algorithm Comparison
--------------------
+-----------------------+----------------+------------------+
| Buffer Type           | Complexity     | Use Case         |
+=======================+================+==================+
| Uniform (this)        | O(1) sample    | DDPG, TD3, SAC   |
| Prioritized (PER)     | O(log n)       | Rainbow, DQN     |
| HER (Hindsight)       | O(1) + goals   | Sparse rewards   |
+-----------------------+----------------+------------------+

Complexity
----------
- Push: O(1) amortized (circular buffer)
- Sample: O(B) where B is batch_size
- Space: O(capacity × transition_size)

Summary
-------
ReplayBuffer provides a simple, efficient circular buffer for storing
transitions and sampling random mini-batches. It's the foundation of
all off-policy deep RL algorithms.
"""

from typing import NamedTuple, List, Tuple, Optional, Union
import numpy as np
import torch


class Transition(NamedTuple):
    """
    Single Experience Transition
    ============================

    Represents one step of interaction with the environment:
    (state, action, reward, next_state, done)

    Mathematical Definition
    -----------------------
    A transition :math:`\\tau = (s, a, r, s', d)` where:

    - :math:`s \\in \\mathcal{S}`: Current state
    - :math:`a \\in \\mathcal{A}`: Action taken
    - :math:`r \\in \\mathbb{R}`: Immediate reward
    - :math:`s' \\in \\mathcal{S}`: Resulting next state
    - :math:`d \\in \\{0, 1\\}`: Terminal flag (1 if episode ended)

    The transition follows from the MDP dynamics:

    .. math::

        s' \\sim P(\\cdot | s, a), \\quad r = R(s, a, s')

    Attributes
    ----------
    state : np.ndarray
        Observation at time t.
    action : np.ndarray
        Action executed at time t.
    reward : float
        Scalar reward received.
    next_state : np.ndarray
        Observation at time t+1.
    done : bool
        Whether episode terminated after this transition.
    """

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Uniform Experience Replay Buffer
    =================================

    Core Idea
    ---------
    Circular buffer that stores transitions and provides uniform random
    sampling. Enables off-policy learning by decoupling data collection
    from policy optimization.

    Mathematical Theory
    -------------------
    The buffer :math:`\\mathcal{D}` stores transitions from behavior policy
    :math:`\\beta`. During training, we sample uniformly:

    .. math::

        (s, a, r, s', d) \\sim \\text{Uniform}(\\mathcal{D})

    This enables training target policy :math:`\\pi` using data from
    :math:`\\beta \\neq \\pi` (off-policy).

    The importance of uniform sampling:

    .. math::

        \\mathbb{E}_{\\tau \\sim \\mathcal{D}}[\\nabla L(\\tau)]
        \\approx \\frac{1}{|\\mathcal{D}|} \\sum_{\\tau \\in \\mathcal{D}} \\nabla L(\\tau)

    Problem Statement
    -----------------
    RL algorithms need to:

    1. Store experiences without memory overflow
    2. Sample random batches efficiently
    3. Handle variable-length episodes

    ReplayBuffer addresses all with a simple circular array implementation.

    Algorithm Comparison
    --------------------
    vs. Prioritized Experience Replay (PER):

    - Uniform: Simple, O(1) complexity, no hyperparameters
    - PER: Better sample efficiency but O(log n), needs α and β tuning

    vs. Hindsight Experience Replay (HER):

    - Uniform: General purpose
    - HER: Specialized for sparse reward / goal-conditioned tasks

    Complexity
    ----------
    - push(): O(1) - Single array assignment
    - sample(): O(B) - B independent random accesses
    - Space: O(C × D) where C=capacity, D=transition dimension

    Implementation Notes
    --------------------
    Uses numpy arrays for memory efficiency and fast indexing.
    Transitions stored as separate arrays (SoA) rather than array of
    structs (AoS) for better cache locality during batch sampling.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    state_dim : int, optional
        State space dimensionality (for pre-allocation).
    action_dim : int, optional
        Action space dimensionality (for pre-allocation).

    Attributes
    ----------
    capacity : int
        Buffer size limit.
    position : int
        Current write position (circular).
    size : int
        Current number of stored transitions.

    Example
    -------
    >>> buffer = ReplayBuffer(capacity=10000)
    >>> buffer.push(state, action, reward, next_state, done)
    >>> if buffer.is_ready(batch_size=256):
    ...     batch = buffer.sample(256)
    ...     states, actions, rewards, next_states, dones = batch

    References
    ----------
    .. [1] Lin, L.J. (1992). "Self-Improving Reactive Agents Based On
           Reinforcement Learning, Planning and Teaching"
    .. [2] Mnih et al. (2015). "Human-level control through deep
           reinforcement learning". Nature.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ) -> None:
        """
        Initialize replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum buffer size. Must be positive.
        state_dim : int, optional
            State dimensionality for pre-allocation.
            If None, buffer uses dynamic lists (slower but flexible).
        action_dim : int, optional
            Action dimensionality for pre-allocation.

        Raises
        ------
        ValueError
            If capacity is not positive.
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.position = 0
        self.size = 0

        # Pre-allocate arrays if dimensions known
        self._preallocated = state_dim is not None and action_dim is not None

        if self._preallocated:
            self._states = np.zeros((capacity, state_dim), dtype=np.float32)
            self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
            self._rewards = np.zeros((capacity, 1), dtype=np.float32)
            self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
            self._dones = np.zeros((capacity, 1), dtype=np.float32)
        else:
            self._buffer: List[Transition] = []

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition in the buffer.

        Uses circular buffer semantics: when full, overwrites oldest entry.

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        action : np.ndarray
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Next observation.
        done : bool
            Episode termination flag.

        Notes
        -----
        For pre-allocated buffers, arrays are stored by reference without
        copying. Ensure input arrays are not modified after calling push().
        """
        if self._preallocated:
            self._states[self.position] = state
            self._actions[self.position] = action
            self._rewards[self.position] = reward
            self._next_states[self.position] = next_state
            self._dones[self.position] = float(done)
        else:
            transition = Transition(
                state=np.asarray(state, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )

            if len(self._buffer) < self.capacity:
                self._buffer.append(transition)
            else:
                self._buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.
        device : torch.device, optional
            Device to place tensors on. Defaults to CPU.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Batch of (states, actions, rewards, next_states, dones).
            Each tensor has shape (batch_size, dim).

        Raises
        ------
        ValueError
            If batch_size exceeds current buffer size.

        Example
        -------
        >>> states, actions, rewards, next_states, dones = buffer.sample(32)
        >>> states.shape
        torch.Size([32, state_dim])
        """
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer "
                f"with only {self.size} entries"
            )

        indices = np.random.randint(0, self.size, size=batch_size)

        if self._preallocated:
            states = self._states[indices]
            actions = self._actions[indices]
            rewards = self._rewards[indices]
            next_states = self._next_states[indices]
            dones = self._dones[indices]
        else:
            batch = [self._buffer[i] for i in indices]
            states = np.array([t.state for t in batch])
            actions = np.array([t.action for t in batch])
            rewards = np.array([[t.reward] for t in batch])
            next_states = np.array([t.next_state for t in batch])
            dones = np.array([[float(t.done)] for t in batch])

        device = device or torch.device("cpu")

        return (
            torch.as_tensor(states, dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.float32, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(next_states, dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device),
        )

    def sample_numpy(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch without converting to PyTorch tensors.

        Useful for debugging or non-PyTorch workflows.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        Tuple[np.ndarray, ...]
            Batch of numpy arrays.
        """
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} from buffer with {self.size} entries"
            )

        indices = np.random.randint(0, self.size, size=batch_size)

        if self._preallocated:
            return (
                self._states[indices].copy(),
                self._actions[indices].copy(),
                self._rewards[indices].copy(),
                self._next_states[indices].copy(),
                self._dones[indices].copy(),
            )
        else:
            batch = [self._buffer[i] for i in indices]
            return (
                np.array([t.state for t in batch]),
                np.array([t.action for t in batch]),
                np.array([[t.reward] for t in batch]),
                np.array([t.next_state for t in batch]),
                np.array([[float(t.done)] for t in batch]),
            )

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples for training.

        Parameters
        ----------
        batch_size : int
            Required number of samples.

        Returns
        -------
        bool
            True if size >= batch_size.
        """
        return self.size >= batch_size

    def clear(self) -> None:
        """Reset buffer to empty state."""
        self.position = 0
        self.size = 0

        if not self._preallocated:
            self._buffer.clear()

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self.size

    def __repr__(self) -> str:
        """String representation showing buffer status."""
        fill_pct = 100 * self.size / self.capacity
        return (
            f"ReplayBuffer(capacity={self.capacity}, size={self.size}, "
            f"fill={fill_pct:.1f}%)"
        )


if __name__ == "__main__":
    # Unit tests for ReplayBuffer
    print("Testing ReplayBuffer...")

    # Test 1: Basic push and sample
    buffer = ReplayBuffer(capacity=100, state_dim=4, action_dim=2)

    for i in range(50):
        state = np.random.randn(4).astype(np.float32)
        action = np.random.randn(2).astype(np.float32)
        reward = float(np.random.randn())
        next_state = np.random.randn(4).astype(np.float32)
        done = i % 10 == 9
        buffer.push(state, action, reward, next_state, done)

    assert len(buffer) == 50
    print("  [PASS] Push operations")

    # Test 2: Sampling
    states, actions, rewards, next_states, dones = buffer.sample(16)
    assert states.shape == (16, 4)
    assert actions.shape == (16, 2)
    assert rewards.shape == (16, 1)
    assert next_states.shape == (16, 4)
    assert dones.shape == (16, 1)
    print("  [PASS] Sample shapes")

    # Test 3: is_ready
    assert buffer.is_ready(50)
    assert not buffer.is_ready(51)
    print("  [PASS] is_ready check")

    # Test 4: Circular buffer behavior
    for i in range(100):
        buffer.push(
            np.zeros(4, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            0.0,
            np.zeros(4, dtype=np.float32),
            False,
        )

    assert len(buffer) == 100  # Should be at capacity
    print("  [PASS] Circular buffer (capacity limit)")

    # Test 5: Dynamic buffer (no pre-allocation)
    dynamic_buffer = ReplayBuffer(capacity=50)
    for i in range(30):
        dynamic_buffer.push(
            np.random.randn(3),
            np.random.randn(1),
            0.5,
            np.random.randn(3),
            False,
        )

    s, a, r, ns, d = dynamic_buffer.sample(10)
    assert s.shape[0] == 10
    print("  [PASS] Dynamic buffer")

    # Test 6: Clear
    buffer.clear()
    assert len(buffer) == 0
    print("  [PASS] Clear operation")

    # Test 7: Error handling
    try:
        empty_buffer = ReplayBuffer(capacity=10)
        empty_buffer.sample(5)
        assert False, "Should raise ValueError"
    except ValueError:
        print("  [PASS] Sample from empty buffer raises error")

    try:
        ReplayBuffer(capacity=0)
        assert False, "Should raise ValueError"
    except ValueError:
        print("  [PASS] Zero capacity raises error")

    # Test 8: Device placement
    if torch.cuda.is_available():
        s, a, r, ns, d = buffer.sample(1, device=torch.device("cuda"))
        assert s.device.type == "cuda"
        print("  [PASS] CUDA device placement")
    else:
        print("  [SKIP] CUDA device test (no GPU)")

    print("\nAll ReplayBuffer tests passed!")
