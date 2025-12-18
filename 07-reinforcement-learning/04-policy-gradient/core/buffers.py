"""
Experience Buffers Module.

This module provides data structures for storing and managing trajectory data
collected during policy gradient training.

Core Idea:
    Policy gradient methods require complete or partial trajectories to compute
    gradient estimates. Efficient buffer management is crucial for memory usage
    and computational performance.

Mathematical Background:
    Trajectory τ = (s_0, a_0, r_1, s_1, a_1, r_2, ..., s_T)

    For REINFORCE:
        ∇_θ J(θ) ≈ (1/N) ∑_i ∑_t ∇_θ log π_θ(a_t|s_t) G_t

    Storage requirement: O(T) per episode for:
        - States: s_t ∈ R^d
        - Actions: a_t (discrete or continuous)
        - Rewards: r_t ∈ R
        - Log probabilities: log π_θ(a_t|s_t) ∈ R
        - Values (optional): V(s_t) ∈ R

References:
    [1] Sutton & Barto (2018). Reinforcement Learning: An Introduction.
    [2] Schulman et al. (2016). High-dimensional continuous control using GAE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union, NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    """
    Single-step transition data.

    Represents one step of interaction: (s, a, r, s', done).

    Attributes
    ----------
    state : np.ndarray
        Current state observation.
    action : Union[int, np.ndarray]
        Action taken (discrete index or continuous vector).
    reward : float
        Reward received after taking action.
    next_state : np.ndarray
        Resulting state observation.
    done : bool
        Whether the episode terminated.
    log_prob : Optional[torch.Tensor]
        Log probability of the action under current policy.
    value : Optional[torch.Tensor]
        Value estimate of current state (for actor-critic methods).

    Examples
    --------
    >>> transition = Transition(
    ...     state=np.array([1.0, 2.0, 3.0, 4.0]),
    ...     action=1,
    ...     reward=1.0,
    ...     next_state=np.array([1.1, 2.1, 3.1, 4.1]),
    ...     done=False,
    ...     log_prob=torch.tensor(-0.693),
    ...     value=torch.tensor(5.0)
    ... )
    """

    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None


@dataclass
class EpisodeBuffer:
    """
    Episode trajectory buffer for policy gradient methods.

    Stores complete trajectory data from one or more episodes for use in
    policy gradient updates. Supports both REINFORCE (episodic) and
    Actor-Critic (n-step) training patterns.

    Core Idea:
        Efficient trajectory storage enables batch processing for gradient
        computation. The buffer maintains separate lists for each data type
        to allow vectorized operations during updates.

    Mathematical Role:
        For policy gradient update:
            θ ← θ + α ∑_t ∇_θ log π_θ(a_t|s_t) · A_t

        The buffer stores:
            - log π_θ(a_t|s_t): For gradient computation
            - r_t: For return/advantage calculation
            - V(s_t): For baseline subtraction (Actor-Critic)

    Attributes
    ----------
    states : List[np.ndarray]
        Sequence of state observations.
    actions : List[Union[int, np.ndarray]]
        Sequence of actions taken.
    rewards : List[float]
        Sequence of rewards received.
    log_probs : List[torch.Tensor]
        Sequence of action log probabilities.
    values : List[torch.Tensor]
        Sequence of value estimates (for Actor-Critic).
    dones : List[bool]
        Sequence of episode termination flags.
    entropies : List[torch.Tensor]
        Sequence of policy entropy values (for entropy regularization).

    Examples
    --------
    >>> buffer = EpisodeBuffer()
    >>> for step in range(100):
    ...     action, log_prob, entropy = policy.sample(state)
    ...     value = critic(state)
    ...     next_state, reward, done, _ = env.step(action)
    ...     buffer.store(
    ...         state=state,
    ...         action=action,
    ...         reward=reward,
    ...         log_prob=log_prob,
    ...         value=value,
    ...         done=done,
    ...         entropy=entropy
    ...     )
    ...     if done:
    ...         break
    >>> print(f"Episode length: {len(buffer)}")
    >>> print(f"Total reward: {buffer.total_reward}")

    Notes
    -----
    Complexity Analysis:
        - Storage: O(T) where T is episode length
        - store(): O(1) amortized (list append)
        - clear(): O(1) (list reassignment)
        - total_reward: O(T) (sum over rewards)

    Memory Layout:
        Each list grows independently, allowing efficient append operations.
        For large-scale training, consider using pre-allocated arrays.
    """

    states: List[np.ndarray] = field(default_factory=list)
    actions: List[Union[int, np.ndarray]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    entropies: List[torch.Tensor] = field(default_factory=list)

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        log_prob: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        done: bool = False,
        entropy: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Store a single transition.

        Parameters
        ----------
        state : np.ndarray
            Current state observation.
        action : Union[int, np.ndarray]
            Action taken in current state.
        reward : float
            Reward received after action.
        log_prob : torch.Tensor
            Log probability of action under current policy.
        value : Optional[torch.Tensor]
            Value estimate of current state.
        done : bool
            Whether episode terminated after this step.
        entropy : Optional[torch.Tensor]
            Entropy of policy distribution at current state.

        Notes
        -----
        Time Complexity: O(1) amortized (Python list append)
        Space Complexity: O(d) where d is state/action dimension
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

        if value is not None:
            self.values.append(value)
        if entropy is not None:
            self.entropies.append(entropy)

    def clear(self) -> None:
        """
        Clear all stored data.

        Resets the buffer to empty state for next episode collection.

        Notes
        -----
        Time Complexity: O(1) (list reassignment, old lists garbage collected)
        """
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.entropies.clear()

    def __len__(self) -> int:
        """
        Return number of stored transitions.

        Returns
        -------
        int
            Number of transitions in buffer.
        """
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        """
        Compute sum of all rewards in buffer.

        Returns
        -------
        float
            Total undiscounted reward.

        Notes
        -----
        Time Complexity: O(T) where T is number of transitions
        """
        return sum(self.rewards)

    def get_batch(self) -> dict:
        """
        Convert buffer contents to batched tensors.

        Returns
        -------
        dict
            Dictionary containing:
                - "states": Stacked state tensor
                - "actions": Stacked action tensor
                - "rewards": Reward list
                - "log_probs": Stacked log probability tensor
                - "values": Stacked value tensor (if available)
                - "dones": Done flag list
                - "entropies": Stacked entropy tensor (if available)

        Notes
        -----
        Time Complexity: O(T) for stacking operations
        Space Complexity: O(T * d) for output tensors
        """
        batch = {
            "states": np.array(self.states),
            "actions": self.actions,
            "rewards": self.rewards,
            "log_probs": torch.stack(self.log_probs) if self.log_probs else None,
            "dones": self.dones,
        }

        if self.values:
            batch["values"] = torch.stack(self.values) if isinstance(self.values[0], torch.Tensor) else torch.tensor(self.values)

        if self.entropies:
            batch["entropies"] = torch.stack(self.entropies)

        return batch


@dataclass
class RolloutBuffer:
    """
    Fixed-size rollout buffer for n-step methods.

    Unlike EpisodeBuffer which grows dynamically, RolloutBuffer has a
    fixed capacity and supports efficient batch sampling. Suitable for
    A2C/PPO style algorithms with fixed rollout lengths.

    Attributes
    ----------
    capacity : int
        Maximum number of transitions to store.
    states : np.ndarray
        Pre-allocated state storage.
    actions : np.ndarray
        Pre-allocated action storage.
    rewards : np.ndarray
        Pre-allocated reward storage.
    log_probs : np.ndarray
        Pre-allocated log probability storage.
    values : np.ndarray
        Pre-allocated value storage.
    dones : np.ndarray
        Pre-allocated done flag storage.
    ptr : int
        Current write position.
    size : int
        Current number of stored transitions.

    Notes
    -----
    Complexity Analysis:
        - Memory: O(capacity * d) pre-allocated
        - store(): O(1) guaranteed
        - Vectorized operations enabled by contiguous memory
    """

    capacity: int
    state_dim: int
    action_dim: int = 1
    continuous: bool = False
    device: str = "cpu"

    def __post_init__(self):
        """Initialize pre-allocated arrays."""
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        if self.continuous:
            self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.entropies = np.zeros(self.capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        entropy: float = 0.0,
    ) -> None:
        """
        Store a transition at current pointer position.

        Parameters
        ----------
        state : np.ndarray
            State observation.
        action : Union[int, np.ndarray]
            Action taken.
        reward : float
            Reward received.
        log_prob : float
            Log probability of action.
        value : float
            Value estimate.
        done : bool
            Episode termination flag.
        entropy : float
            Policy entropy.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.entropies[self.ptr] = entropy

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self) -> dict:
        """
        Get all stored data as tensors.

        Returns
        -------
        dict
            Dictionary of tensors ready for training.
        """
        return {
            "states": torch.tensor(self.states[: self.size], dtype=torch.float32, device=self.device),
            "actions": torch.tensor(self.actions[: self.size], device=self.device),
            "rewards": self.rewards[: self.size],
            "log_probs": torch.tensor(self.log_probs[: self.size], dtype=torch.float32, device=self.device),
            "values": torch.tensor(self.values[: self.size], dtype=torch.float32, device=self.device),
            "dones": self.dones[: self.size],
            "entropies": torch.tensor(self.entropies[: self.size], dtype=torch.float32, device=self.device),
        }

    def clear(self) -> None:
        """Reset buffer to empty state."""
        self.ptr = 0
        self.size = 0
