#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Q-Network (DQN) Implementation for Continuous Control

This module implements production-grade deep value-based reinforcement learning algorithms
with comprehensive mathematical foundations and engineering optimizations suitable for
both academic publication and industrial deployment.

========================================
Mathematical Foundations
========================================

Deep Q-Network extends tabular Q-Learning to high-dimensional state spaces through
function approximation with deep neural networks.

Core Bellman Optimality Equation:
    Q*(s, a) = E[r + γ max_{a'} Q*(s', a') | s, a]

DQN Loss Function (Temporal Difference Learning):
    L(θ) = E_{(s,a,r,s')~D}[(r + γ max_{a'} Q(s', a'; θ⁻) - Q(s, a; θ))²]

where:
    - θ: Online Q-network parameters
    - θ⁻: Target network parameters (periodically synced)
    - D: Experience replay buffer
    - γ ∈ [0, 1]: Discount factor

Double DQN (van Hasselt et al., 2016):
    y^{DoubleDQN} = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ⁻)

Decouples action selection (online network) from evaluation (target network)
to eliminate maximization bias: E[max Q] ≥ max E[Q]

Dueling DQN (Wang et al., 2016):
    Q(s, a) = V(s) + (A(s, a) - 1/|A| Σ_{a'} A(s, a'))

where:
    - V(s): State value function
    - A(s, a): Advantage function

Prioritized Experience Replay (Schaul et al., 2015):
    P(i) = p_i^α / Σ_k p_k^α
    p_i = |δ_i| + ε
    w_i = (N · P(i))^{-β}

where:
    - δ_i: TD error for transition i
    - α ∈ [0, 1]: Prioritization exponent
    - β ∈ [0, 1]: Importance sampling correction
    - ε > 0: Small constant preventing zero priority

========================================
Convergence Guarantees
========================================

DQN converges to optimal Q* under conditions:
1. Function approximation: Universal approximator (e.g., neural networks)
2. Exploration: ε-greedy with lim_{t→∞} ε_t = 0
3. Learning rate: Robbins-Monro conditions
   - Σ α_t = ∞
   - Σ α_t² < ∞
4. Experience replay: Breaks temporal correlations

Sample Complexity (PAC bound):
    O(|S||A| / ((1-γ)⁴ε²) · poly(feature_dim, 1/δ))

for ε-optimal policy with probability 1-δ

========================================
Algorithmic Innovations
========================================

1. Experience Replay (Lin, 1992; Mnih et al., 2013)
   - Breaks temporal correlations in sequential data
   - Enables i.i.d. sampling assumption
   - Improves data efficiency through reuse

2. Target Network (Mnih et al., 2015)
   - Stabilizes training by fixing TD target
   - Prevents oscillations from moving targets
   - Update frequency: Hard update every C steps or soft update τθ + (1-τ)θ⁻

3. Double Q-Learning (van Hasselt, 2010; van Hasselt et al., 2016)
   - Eliminates positive bias from max operator
   - Empirically shown to improve stability

4. Dueling Architecture (Wang et al., 2016)
   - Separates state value from action advantages
   - More stable gradient flow for value function
   - Better generalization in states where actions matter less

5. Prioritized Experience Replay (Schaul et al., 2015)
   - Focuses learning on high-error transitions
   - Accelerates convergence by 2x in Atari
   - Requires importance sampling correction

========================================
Complexity Analysis
========================================

Space Complexity:
- Replay Buffer: O(B) for buffer size B
- Q-Network: O(|θ|) for parameter count
- Total: O(B + |θ|)

Time Complexity (per update):
- Experience storage: O(1)
- Sampling: O(N) uniform, O(log B) with sum-tree for PER
- Forward pass: O(|θ|) network evaluation
- Backward pass: O(|θ|) gradient computation
- Total: O(N + |θ|) for batch size N

References:
-----------
[1] Mnih et al., "Playing Atari with Deep Reinforcement Learning", NIPS Workshop 2013
[2] Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015
[3] van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI 2016
[4] Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016
[5] Schaul et al., "Prioritized Experience Replay", ICLR 2016
[6] Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning", AAAI 2018

Environment Requirements:
-------------------------
    Python >= 3.8
    PyTorch >= 1.9
    Gymnasium >= 0.28
    NumPy >= 1.20

Author: Ziming Ding
Date: 2024
Version: 2.0.0
License: MIT
"""

from __future__ import annotations
import os
import sys
import random
import warnings
from typing import Tuple, List, Optional, Dict, Any, Union
from collections import deque
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not installed, plotting unavailable")

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    warnings.warn("gymnasium not installed, environment interaction unavailable")


# =============================================================================
# Experience Replay Module
# =============================================================================

@dataclass
class Transition:
    """
    Atomic transition data structure for experience replay.

    Core Idea:
    ---------
    Encapsulates a single (s, a, r, s', done) tuple representing one step
    of environment interaction. Uses dataclass for immutability and type safety.

    Mathematical Representation:
    ---------------------------
    τ = (s_t, a_t, r_t, s_{t+1}, d_t) where:
    - s_t ∈ S: Current state
    - a_t ∈ A: Action taken
    - r_t ∈ ℝ: Immediate reward
    - s_{t+1} ∈ S: Next state
    - d_t ∈ {0, 1}: Episode termination flag

    Complexity:
    ----------
    - Space: O(|s|) for state dimension |s|
    - Access: O(1)

    Attributes:
        state: Current state observation vector
        action: Discrete action index
        reward: Scalar immediate reward
        next_state: Subsequent state observation
        done: Boolean episode termination indicator
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Uniform experience replay buffer for off-policy learning.

    Core Idea:
    ---------
    Store past transitions and sample uniformly at random to break temporal
    correlations, enabling i.i.d. assumption for gradient descent.

    Mathematical Principle:
    ----------------------
    Standard DQN samples mini-batches B ~ U(D) uniformly from buffer D.
    This decorrelates consecutive transitions which would otherwise violate
    i.i.d. assumptions of stochastic gradient descent.

    Sampling distribution: P(τ_i) = 1/|D| for all τ_i ∈ D

    Problem Statement:
    -----------------
    Raw online learning suffers from:
    1. Catastrophic forgetting: New experiences overwrite old learnings
    2. Temporal correlation: Sequential samples violate i.i.d. assumption
    3. Poor sample efficiency: Each transition used only once

    Algorithm Comparison:
    --------------------
    vs. Online Learning:
        + Breaks temporal correlations → Stable training
        + Reuses samples → Higher data efficiency
        - Requires memory: O(buffer_size)
        - Off-policy: Cannot use on-policy algorithms (e.g., vanilla SARSA)

    vs. Prioritized Replay:
        + Simpler implementation: O(1) sampling
        + Unbiased: Equal probability for all samples
        - No focus on important transitions
        - Slower convergence in sparse reward environments

    Complexity:
    ----------
    - Space: O(B × |s|) for buffer size B and state dimension |s|
    - push(): O(1) amortized (deque with maxlen)
    - sample(N): O(N) for batch size N
    - Storage: Circular buffer via collections.deque

    Theoretical Properties:
    ----------------------
    1. Convergence: Maintains i.i.d. sampling assumption
    2. Bias: Unbiased uniform sampling
    3. Variance: Higher variance than prioritized methods
    4. Stability: High stability due to decorrelation

    Implementation Details:
    ----------------------
    - Uses deque with maxlen for automatic FIFO eviction
    - Returns NumPy arrays for seamless PyTorch conversion
    - Thread-unsafe: Requires external synchronization for parallel envs

    Summary:
    -------
    Foundational component of DQN enabling off-policy learning through
    uniformly sampled experience replay. Trades memory for stability and
    sample efficiency. Essential for deep RL convergence.
    """

    def __init__(self, capacity: int = 100000) -> None:
        """
        Initialize uniform experience replay buffer.

        Args:
            capacity: Maximum buffer size. Oldest transitions evicted when full.

        Raises:
            ValueError: If capacity <= 0
        """
        if capacity <= 0:
            raise ValueError(f"Buffer capacity must be positive, got: {capacity}")

        self._capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        """Return maximum buffer capacity."""
        return self._capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store single transition to buffer.

        Complexity: O(1) amortized

        Args:
            state: Current state vector
            action: Executed action index
            reward: Observed reward
            next_state: Resulting state
            done: Episode termination flag
        """
        transition = Transition(state, action, reward, next_state, done)
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Uniformly sample mini-batch from buffer.

        Sampling: τ_i ~ U(D) independently for i ∈ [1, batch_size]

        Complexity: O(batch_size)

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple (states, actions, rewards, next_states, dones) as NumPy arrays
            - states: (batch_size, state_dim)
            - actions: (batch_size,) dtype=int64
            - rewards: (batch_size,) dtype=float32
            - next_states: (batch_size, state_dim)
            - dones: (batch_size,) dtype=float32

        Raises:
            ValueError: If batch_size > current buffer size
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Requested batch_size {batch_size} exceeds buffer size {len(self._buffer)}"
            )

        batch = random.sample(self._buffer, batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer contains enough samples for training.

        Args:
            batch_size: Required minimum samples

        Returns:
            True if len(buffer) >= batch_size
        """
        return len(self._buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay (PER) for importance-weighted sampling.

    Core Idea:
    ---------
    Sample transitions proportional to their TD error magnitude, focusing
    learning on "surprising" experiences where predictions are most wrong.

    Mathematical Principle:
    ----------------------
    Sampling probability:
        P(i) = p_i^α / Σ_k p_k^α

    Priority assignment:
        p_i = |δ_i| + ε
        δ_i = r + γ max_{a'} Q(s', a'; θ⁻) - Q(s, a; θ)  [TD error]

    Importance sampling weights:
        w_i = (N · P(i))^{-β} / max_j w_j

    where:
        - α ∈ [0, 1]: Prioritization exponent
          * α = 0: Uniform sampling (standard replay)
          * α = 1: Full prioritization (greedy)
        - β ∈ [0, 1]: Importance sampling exponent
          * β = 0: No correction (biased)
          * β = 1: Full correction (unbiased)
        - ε > 0: Small constant preventing zero priority
        - N: Buffer size

    Problem Statement:
    -----------------
    Uniform sampling treats all transitions equally, ignoring that some
    experiences contain more valuable learning signals. In sparse reward
    environments, rare successful trajectories may be under-sampled.

    PER addresses this by:
    1. Prioritizing high-error transitions (model uncertainty)
    2. Ensuring rare but important experiences are replayed more
    3. Correcting sampling bias via importance sampling weights

    Algorithm Comparison:
    --------------------
    vs. Uniform Replay:
        + 2x faster convergence (Atari experiments)
        + Better in sparse reward environments
        + Focuses on model blind spots
        - Computational overhead: O(log N) sampling with sum-tree
        - Hyperparameter sensitivity (α, β)
        - Implementation complexity

    vs. Hindsight Experience Replay:
        + Directly uses TD error signal
        + Works with any reward structure
        - Doesn't create synthetic goals
        - Less effective in multi-goal environments

    Complexity:
    ----------
    This simplified linear implementation:
    - Space: O(B) for buffer size B
    - push(): O(1)
    - sample(N): O(B + N) for buffer size B, batch size N
    - update_priorities(): O(N)

    Optimal sum-tree implementation:
    - Space: O(B)
    - push(): O(log B)
    - sample(N): O(N log B)
    - update_priorities(): O(N log B)

    Theoretical Properties:
    ----------------------
    1. Convergence: Provably converges with importance sampling correction
    2. Bias-Variance Trade-off: α controls bias (prioritization) vs variance
    3. Annealing Schedule: Typically β: 0.4 → 1.0 over training
    4. Stale Priorities: Priorities updated asynchronously may be outdated

    Implementation Details:
    ----------------------
    - Linear array implementation (simple but O(N) sampling)
    - Production systems should use sum-tree for O(log N)
    - New experiences assigned max_priority to guarantee sampling
    - β typically annealed from 0.4 to 1.0 over training

    Summary:
    -------
    Advanced replay mechanism that accelerates learning by focusing on
    high-error transitions. Critical for sample efficiency in sparse reward
    domains. Adds computational cost but provides significant performance gains.
    Essential component of Rainbow DQN.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6
    ) -> None:
        """
        Initialize prioritized experience replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent, α ∈ [0, 1]
                   α=0: uniform sampling, α=1: greedy prioritization
            beta: Importance sampling exponent, β ∈ [0, 1]
                  Typically annealed from 0.4 to 1.0
            epsilon: Small constant preventing zero priority

        Raises:
            ValueError: If alpha or beta not in [0, 1]
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got: {alpha}")
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got: {beta}")

        self._capacity = capacity
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon

        self._buffer: List[Transition] = []
        self._priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._position: int = 0
        self._max_priority: float = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store transition with maximum priority.

        New experiences assigned max_priority to ensure at least one sampling
        before priority update.

        Complexity: O(1)

        Args:
            state: Current state vector
            action: Executed action
            reward: Observed reward
            next_state: Resulting state
            done: Episode termination flag
        """
        transition = Transition(state, action, reward, next_state, done)

        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._position] = transition

        self._priorities[self._position] = self._max_priority
        self._position = (self._position + 1) % self._capacity

    def sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """
        Sample mini-batch proportional to priorities with importance weights.

        Complexity: O(|D| + N) for buffer size |D| and batch size N

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple (states, actions, rewards, next_states, dones, indices, weights)
            - states: (batch_size, state_dim)
            - actions: (batch_size,)
            - rewards: (batch_size,)
            - next_states: (batch_size, state_dim)
            - dones: (batch_size,)
            - indices: (batch_size,) - Buffer indices for priority updates
            - weights: (batch_size,) - Importance sampling weights

        Raises:
            ValueError: If batch_size > buffer size
        """
        buffer_len = len(self._buffer)
        if batch_size > buffer_len:
            raise ValueError(f"batch_size {batch_size} exceeds buffer size {buffer_len}")

        # Compute sampling probabilities: P(i) = p_i^α / Σ p_k^α
        priorities = self._priorities[:buffer_len]
        probs = priorities ** self._alpha
        probs = probs / probs.sum()

        # Sample indices proportional to priorities
        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=False)

        # Compute importance sampling weights: w_i = (N·P(i))^{-β} / max w
        weights = (buffer_len * probs[indices]) ** (-self._beta)
        weights = weights / weights.max()  # Normalize by max for stability

        # Extract transitions
        batch = [self._buffer[i] for i in indices]
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return (states, actions, rewards, next_states, dones,
                indices.astype(np.int64), weights.astype(np.float32))

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray
    ) -> None:
        """
        Update priorities based on TD errors.

        Priority formula: p_i = |δ_i| + ε

        Complexity: O(N) for N updates

        Args:
            indices: Buffer indices to update
            td_errors: Corresponding TD errors δ_i
        """
        priorities = np.abs(td_errors) + self._epsilon
        for idx, priority in zip(indices, priorities):
            self._priorities[idx] = priority

        self._max_priority = max(self._max_priority, priorities.max())

    def update_beta(self, beta: float) -> None:
        """
        Update importance sampling exponent.

        Typical annealing schedule: β: 0.4 → 1.0 over training

        Args:
            beta: New β value, clamped to [0, 1]
        """
        self._beta = min(1.0, beta)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if sufficient samples available."""
        return len(self._buffer) >= batch_size


# =============================================================================
# Neural Network Module
# =============================================================================

def _init_weights(module: nn.Module) -> None:
    """
    Orthogonal weight initialization for deep RL networks.

    Orthogonal initialization (Saxe et al., 2013) preserves gradient magnitudes
    through layers, critical for deep networks. Uses gain=√2 for ReLU activations
    following He et al., 2015.

    Complexity: O(fan_in × fan_out) for linear layers

    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DQNNetwork(nn.Module):
    """
    Standard Deep Q-Network architecture for value function approximation.

    Core Idea:
    ---------
    Multi-layer perceptron (MLP) mapping states to action-value estimates.
    Universal approximation theorem guarantees convergence to Q* with sufficient
    capacity.

    Mathematical Principle:
    ----------------------
    Function approximation:
        Q(s, a; θ) = f_θ(s)[a]

    where f_θ: S → ℝ^{|A|} is a neural network parameterized by θ.

    Architecture:
        s ∈ ℝ^d → Linear(d, h) → ReLU → Linear(h, h) → ReLU → Linear(h, |A|) → Q-values

    Loss (MSE TD error):
        L(θ) = E[(y - Q(s, a; θ))²]
        y = r + γ max_{a'} Q(s', a'; θ⁻)  [TD target]

    Problem Statement:
    -----------------
    Tabular Q-Learning requires storing Q(s,a) for every state-action pair,
    infeasible for high-dimensional or continuous state spaces. Neural networks
    provide compact function approximation with generalization.

    Algorithm Comparison:
    --------------------
    vs. Linear Approximation:
        + Automatically learns features (no manual feature engineering)
        + Handles high-dimensional inputs (images, sensor data)
        + Universal approximation capacity
        - Slower training
        - Requires more data
        - Hyperparameter tuning needed

    vs. Dueling DQN:
        + Simpler architecture
        + Faster forward pass
        - Less stable value learning
        - No explicit advantage decomposition

    Complexity:
    ----------
    - Parameters: O(d·h + h² + h·|A|) for state dim d, hidden dim h, actions |A|
    - Forward pass: O(d·h + h² + h·|A|)
    - Backward pass: Same as forward (automatic differentiation)
    - Memory: O(parameters + batch_size·d) during training

    Theoretical Properties:
    ----------------------
    1. Universal Approximation: Can represent any continuous Q* with sufficient width
    2. Convergence: Provably converges under Robbins-Monro conditions
    3. Generalization: Shares information across similar states
    4. Stability: Requires target networks and replay buffer

    Implementation Details:
    ----------------------
    - ReLU activations for non-linearity
    - Orthogonal initialization for stable gradients
    - No activation on output (Q-values can be negative)
    - Typically 2-3 hidden layers for tabular-like tasks

    Summary:
    -------
    Foundation DQN architecture enabling RL in high-dimensional spaces.
    Simple MLP design balances expressiveness with training stability.
    Sufficient for most continuous control and grid-world tasks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128
    ) -> None:
        """
        Initialize standard DQN network.

        Args:
            input_dim: State space dimensionality
            output_dim: Action space size (number of Q-values)
            hidden_dim: Hidden layer width
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        # Apply orthogonal initialization
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values.

        Complexity: O(batch_size · parameters)

        Args:
            x: State tensor (batch_size, input_dim)

        Returns:
            Q-values (batch_size, output_dim)
        """
        return self.net(x)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture with value-advantage decomposition.

    Core Idea:
    ---------
    Decompose Q(s,a) into state value V(s) and action advantage A(s,a),
    enabling independent learning of "how good is this state" vs
    "how good is this action relative to others."

    Mathematical Principle:
    ----------------------
    Dueling architecture decomposition:
        Q(s, a) = V(s) + A(s, a) - 1/|A| Σ_{a'} A(s, a')

    where:
        - V(s): State value function
        - A(s, a): Advantage function
        - Mean subtraction ensures identifiability

    Network structure:
        s → Shared Features → Value Stream → V(s)
                           → Advantage Stream → A(s, a)
        Q(s, a) = V(s) + [A(s, a) - mean_a A(s, a')]

    Identifiability constraint:
        Without mean subtraction, decomposition is non-unique:
        Q(s, a) = [V(s) + c] + [A(s, a) - c] for any constant c
        Mean subtraction enforces Σ_a A(s, a) = 0, making V and A unique.

    Problem Statement:
    -----------------
    Standard DQN learns Q(s,a) jointly, but in many states, action choice
    has minimal impact (e.g., safe states far from goals). Learning V(s)
    separately allows faster value propagation independent of action effects.

    Algorithm Comparison:
    --------------------
    vs. Standard DQN:
        + Faster convergence: V(s) learned from all actions simultaneously
        + More stable: Value stream less noisy than Q-values
        + Better in states where actions don't matter
        - Slightly more parameters: ~1.5x network size
        - Minimal overhead: Negligible computation increase

    vs. A3C (Actor-Critic):
        + Off-policy: Works with experience replay
        + No separate networks: Single network outputs both
        - Discrete actions only (no continuous control)
        - Still value-based (no explicit policy)

    Complexity:
    ----------
    Let d = state_dim, h = hidden_dim, |A| = action_dim
    - Parameters: O(d·h + 2h² + h·|A| + h) ≈ 1.5× standard DQN
    - Forward pass: O(d·h + 2h² + h·|A|)
    - Additional ops: O(|A|) for mean subtraction
    - Memory: Comparable to standard DQN

    Theoretical Properties:
    ----------------------
    1. Identifiability: Mean subtraction ensures unique V, A decomposition
    2. Variance Reduction: Value stream has lower variance than Q
    3. Generalization: Advantage function focuses on relative action quality
    4. Convergence: Same guarantees as standard DQN

    Empirical Results (Wang et al., 2016):
    -------------------------------------
    - +20% improvement over DQN on Atari
    - Faster convergence in most environments
    - Particularly effective in:
      * Environments with similar-valued actions
      * States where action choice less critical
      * Sparse reward settings

    Implementation Details:
    ----------------------
    - Shared feature extraction for efficiency
    - Separate streams: 2 hidden layers each
    - Value stream outputs scalar V(s)
    - Advantage stream outputs vector A(s, ·)
    - Mean advantage subtracted for identifiability

    Summary:
    -------
    Advanced DQN architecture decomposing Q-values into state value and
    action advantages. Provides faster, more stable learning with minimal
    computational overhead. Standard choice for modern value-based deep RL.
    Core component of Rainbow DQN.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128
    ) -> None:
        """
        Initialize Dueling DQN network.

        Args:
            input_dim: State space dimensionality
            output_dim: Action space size
            hidden_dim: Hidden layer width for each stream
        """
        super().__init__()

        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Value stream: V(s) ∈ ℝ
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream: A(s, ·) ∈ ℝ^{|A|}
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with value-advantage aggregation.

        Aggregation formula:
            Q(s, a) = V(s) + [A(s, a) - mean_a' A(s, a')]

        Complexity: O(batch_size · parameters + batch_size · |A|)

        Args:
            x: State tensor (batch_size, input_dim)

        Returns:
            Q-values (batch_size, output_dim)
        """
        features = self.feature_layer(x)

        value = self.value_stream(features)          # (B, 1)
        advantage = self.advantage_stream(features)  # (B, |A|)

        # Aggregate: Q = V + (A - mean(A))
        # Mean subtraction ensures identifiability: Σ_a A(s,a) = 0
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))

        return q_values


# =============================================================================
# DQN Agent
# =============================================================================

@dataclass
class DQNConfig:
    """
    DQN hyperparameter configuration with validation.

    Centralized configuration management for reproducibility and hyperparameter
    optimization. All parameters validated at initialization.

    Attributes:
        state_dim: State space dimensionality
        action_dim: Discrete action space size
        hidden_dim: Neural network hidden layer width
        learning_rate: Adam optimizer learning rate
        gamma: Discount factor γ ∈ [0, 1]
        epsilon_start: Initial ε-greedy exploration rate
        epsilon_end: Final ε-greedy exploration rate
        epsilon_decay: Exponential decay factor per episode
        buffer_size: Experience replay capacity
        batch_size: Mini-batch size for SGD updates
        target_update_freq: Target network sync frequency (steps)
        double_dqn: Enable Double DQN (action selection decoupling)
        dueling: Use Dueling architecture (value-advantage decomposition)
        prioritized: Enable prioritized experience replay
        per_alpha: PER prioritization exponent α ∈ [0, 1]
        per_beta_start: PER initial importance sampling β
        per_beta_frames: Frames for β annealing to 1.0
        grad_clip: Gradient norm clipping threshold
        device: Compute device ('auto', 'cpu', 'cuda')
    """
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 100
    double_dqn: bool = False
    dueling: bool = False
    prioritized: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100000
    grad_clip: float = 10.0
    device: str = "auto"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters at initialization.

        Raises:
            ValueError: If any parameter violates domain constraints
        """
        # Dimension constraints
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")

        # Learning rate
        if not 0 < self.learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")

        # Discount factor
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")

        # Exploration parameters
        if not 0 <= self.epsilon_start <= 1:
            raise ValueError(f"epsilon_start must be in [0, 1], got {self.epsilon_start}")
        if not 0 <= self.epsilon_end <= 1:
            raise ValueError(f"epsilon_end must be in [0, 1], got {self.epsilon_end}")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError(
                f"epsilon_end ({self.epsilon_end}) cannot exceed "
                f"epsilon_start ({self.epsilon_start})"
            )
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {self.epsilon_decay}")

        # Buffer parameters
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {self.buffer_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.batch_size > self.buffer_size:
            raise ValueError(
                f"batch_size ({self.batch_size}) cannot exceed "
                f"buffer_size ({self.buffer_size})"
            )

        # Target network update
        if self.target_update_freq <= 0:
            raise ValueError(
                f"target_update_freq must be positive, got {self.target_update_freq}"
            )

        # PER parameters
        if not 0 <= self.per_alpha <= 1:
            raise ValueError(f"per_alpha must be in [0, 1], got {self.per_alpha}")
        if not 0 <= self.per_beta_start <= 1:
            raise ValueError(
                f"per_beta_start must be in [0, 1], got {self.per_beta_start}"
            )
        if self.per_beta_frames <= 0:
            raise ValueError(
                f"per_beta_frames must be positive, got {self.per_beta_frames}"
            )

        # Gradient clipping
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive, got {self.grad_clip}")


class DQNAgent:
    """
    Deep Q-Network agent with algorithmic variants.

    Core Idea:
    ---------
    Off-policy value-based agent learning optimal action-value function Q*(s,a)
    through temporal difference learning with neural network function approximation.

    Mathematical Principle:
    ----------------------
    Bellman optimality operator:
        T*Q(s, a) = E[r + γ max_{a'} Q(s', a')]

    DQN iteratively applies T* via gradient descent:
        θ_{t+1} = θ_t - α ∇_θ L(θ_t)
        L(θ) = E[(y - Q(s, a; θ))²]
        y = r + γ max_{a'} Q(s', a'; θ⁻)  [Target network]

    Double DQN modification:
        y^{Double} = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ⁻)

    Action selection (ε-greedy):
        π(s) = argmax_a Q(s, a; θ)  with prob 1-ε
               random action           with prob ε

    Problem Statement:
    -----------------
    Q-Learning with neural networks faces:
    1. Moving targets: TD target y changes as θ updates
    2. Correlated samples: Sequential states violate i.i.d. assumption
    3. Overestimation bias: max operator leads to positive bias
    4. Sample inefficiency: On-policy methods waste data

    DQN innovations:
    1. Target network θ⁻: Stabilizes TD target
    2. Experience replay: Decorrelates samples
    3. Double Q-Learning: Reduces overestimation
    4. Dueling architecture: Decomposes value function

    Algorithm Comparison:
    --------------------
    vs. Tabular Q-Learning:
        + Handles high-dimensional states (images, sensors)
        + Generalization across similar states
        - Requires more samples
        - Unstable without stabilization techniques

    vs. Policy Gradient (A2C, PPO):
        + Higher sample efficiency (off-policy replay)
        + Simpler credit assignment (no variance reduction needed)
        - Discrete actions only (no continuous control)
        - Deterministic policy (no stochastic exploration)

    vs. Actor-Critic (DDPG, TD3, SAC):
        + Simpler: Single network for discrete actions
        + More stable: No policy collapse issues
        - Discrete only
        - No inherent exploration bonus

    Complexity:
    ----------
    Per update step:
    - Forward pass: O(batch_size · |θ|) for parameters |θ|
    - Backward pass: O(batch_size · |θ|)
    - Replay sampling: O(batch_size) uniform, O(batch_size log B) PER
    - Total: O(batch_size · |θ|)

    Per episode:
    - Training: O(T · (|θ| + |A|)) for T-step episode
    - Memory: O(B · |s|) for buffer size B

    Sample Complexity (theoretical bound):
        Õ(|S||A| / ((1-γ)⁴ε²))

    for ε-optimal policy with PAC guarantee

    Theoretical Properties:
    ----------------------
    1. Convergence: Provably converges to Q* with:
       - Linear approximation (tabular limit)
       - Sufficient exploration
       - Robbins-Monro learning rate schedule
    2. Optimality: Finds optimal policy π*(s) = argmax_a Q*(s, a)
    3. Off-policy: Learns from any behavior policy (data efficiency)
    4. Overestimation Bias: Standard DQN overestimates, Double DQN corrects

    Empirical Performance (Nature 2015):
    -----------------------------------
    - Human-level performance on 49/57 Atari games
    - Rainbow DQN (all extensions): State-of-art on Atari
    - Sample efficiency: 100M-200M frames to convergence
    - Typical hyperparameters:
      * Learning rate: 1e-4 to 1e-3
      * Replay buffer: 1M frames
      * Batch size: 32-64
      * Target update: 1k-10k steps
      * Exploration: ε decay over 1M steps

    Implementation Details:
    ----------------------
    - Adam optimizer: Adaptive learning rates
    - Gradient clipping: Prevents exploding gradients
    - Reward clipping (Atari): Stabilizes learning
    - Frame stacking (Atari): Provides temporal context
    - Target network: Hard update every C steps

    Recommended Usage:
    -----------------
    ```python
    config = DQNConfig(
        state_dim=4,
        action_dim=2,
        learning_rate=1e-3,
        double_dqn=True,
        dueling=True
    )
    agent = DQNAgent(config)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()
    ```

    Summary:
    -------
    Foundational deep RL algorithm enabling learning in high-dimensional
    spaces through neural function approximation. Combines Q-Learning with
    deep networks, experience replay, and target networks for stable training.
    Achieves human-level performance on complex tasks with discrete actions.
    """

    def __init__(self, config: DQNConfig) -> None:
        """
        Initialize DQN agent with validated configuration.

        Args:
            config: Hyperparameter configuration object (validated at init)
        """
        self.config = config
        self._setup_device()
        self._setup_networks()
        self._setup_replay_buffer()

        # Training state
        self.epsilon = config.epsilon_start
        self.update_count = 0
        self.frame_count = 0

    def _setup_device(self) -> None:
        """Configure compute device (CPU/CUDA)."""
        if self.config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(self.config.device)

    def _setup_networks(self) -> None:
        """
        Initialize Q-networks and optimizer.

        Creates:
        - Online Q-network (θ): Updated every step
        - Target Q-network (θ⁻): Synced periodically for stable targets
        """
        # Select architecture: Dueling or standard
        NetworkClass = (
            DuelingDQNNetwork if self.config.dueling else DQNNetwork
        )

        # Online network: Updated via gradient descent
        self.q_network = NetworkClass(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)

        # Target network: Fixed for C steps, then synced
        self.target_network = NetworkClass(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)

        # Initialize target with online weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # No gradient computation needed

        # Adam optimizer with configured learning rate
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )

    def _setup_replay_buffer(self) -> None:
        """Initialize experience replay buffer (uniform or prioritized)."""
        if self.config.prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.config.buffer_size,
                alpha=self.config.per_alpha,
                beta=self.config.per_beta_start
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=self.config.buffer_size
            )

    def get_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> int:
        """
        Select action using ε-greedy policy.

        Policy:
            π(s) = argmax_a Q(s, a; θ)  with probability 1-ε (exploitation)
                   uniform random         with probability ε (exploration)

        Complexity: O(|A| · |θ|) for action space |A| and parameters |θ|

        Args:
            state: Current state observation
            training: If True, use ε-greedy; if False, greedy only

        Returns:
            Selected action index ∈ [0, action_dim)
        """
        # Exploration: Random action with probability ε
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)

        # Exploitation: Greedy action argmax_a Q(s, a)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.argmax(dim=1).item()

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[float]:
        """
        Store transition and perform one gradient descent step.

        Update procedure:
        1. Store (s, a, r, s', done) in replay buffer
        2. Sample mini-batch from buffer (if sufficient samples)
        3. Compute TD target: y = r + γ max_{a'} Q(s', a'; θ⁻)
        4. Compute loss: L = (y - Q(s, a; θ))²
        5. Gradient descent: θ ← θ - α ∇_θ L
        6. Periodically sync target network: θ⁻ ← θ

        Complexity: O(batch_size · |θ|)

        Args:
            state: Current state s_t
            action: Executed action a_t
            reward: Observed reward r_t
            next_state: Resulting state s_{t+1}
            done: Episode termination flag

        Returns:
            Training loss if update performed, None if buffer insufficient
        """
        # Step 1: Store transition in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.frame_count += 1

        # Step 2: Check if sufficient samples for training
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return None

        # Step 3: Sample mini-batch
        if self.config.prioritized:
            # Prioritized sampling with importance weights
            batch = self.replay_buffer.sample(self.config.batch_size)
            states, actions, rewards, next_states, dones, indices, weights = batch
            weights_tensor = torch.FloatTensor(weights).to(self.device)

            # Anneal β: 0.4 → 1.0 over training
            fraction = min(
                1.0,
                self.frame_count / self.config.per_beta_frames
            )
            beta = (
                self.config.per_beta_start +
                fraction * (1.0 - self.config.per_beta_start)
            )
            self.replay_buffer.update_beta(beta)
        else:
            # Uniform sampling
            states, actions, rewards, next_states, dones = (
                self.replay_buffer.sample(self.config.batch_size)
            )
            weights_tensor = None

        # Convert to PyTorch tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Step 4: Compute current Q-values Q(s, a; θ)
        current_q_values = self.q_network(states_t)
        current_q = current_q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Step 5: Compute TD target y
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: Decouple action selection and evaluation
                # Select action: a* = argmax_a Q(s', a; θ) [online network]
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                # Evaluate action: Q(s', a*; θ⁻) [target network]
                next_q = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # Standard DQN: max_a Q(s', a; θ⁻)
                next_q = self.target_network(next_states_t).max(dim=1)[0]

            # TD target: y = r + γ max_{a'} Q(s', a'; θ⁻) (1 - done)
            target_q = rewards_t + self.config.gamma * next_q * (1 - dones_t)

        # Step 6: Compute TD error δ = Q(s, a; θ) - y
        td_errors = current_q - target_q

        # Step 7: Compute loss
        if self.config.prioritized:
            # Weighted MSE loss for importance sampling correction
            loss = (weights_tensor * (td_errors ** 2)).mean()
            # Update priorities: p_i = |δ_i| + ε
            self.replay_buffer.update_priorities(
                indices, td_errors.detach().cpu().numpy()
            )
        else:
            # Standard MSE loss
            loss = F.mse_loss(current_q, target_q)

        # Step 8: Gradient descent
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: Prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.config.grad_clip
        )

        self.optimizer.step()

        # Step 9: Periodically sync target network
        self.update_count += 1
        if self.update_count % self.config.target_update_freq == 0:
            self.sync_target_network()

        return loss.item()

    def sync_target_network(self) -> None:
        """
        Synchronize target network parameters.

        Hard update: θ⁻ ← θ (copy all weights)

        Alternative soft update (not implemented):
            θ⁻ ← τθ + (1-τ)θ⁻ for τ ∈ (0, 1)

        Complexity: O(|θ|)
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """
        Decay exploration rate.

        Exponential decay: ε ← max(ε_end, ε · decay)

        Annealing schedule ensures eventual greedy exploitation while
        maintaining minimum exploration.
        """
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

    def save(self, path: str) -> None:
        """
        Save agent checkpoint.

        Saves:
        - Q-network parameters
        - Target network parameters
        - Optimizer state (momentum, etc.)
        - Training state (epsilon, counters)
        - Configuration

        Args:
            path: Checkpoint file path (.pt extension recommended)
        """
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "frame_count": self.frame_count,
            "config": self.config
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load agent checkpoint.

        Restores full training state for seamless resumption.

        Args:
            path: Checkpoint file path
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.update_count = checkpoint["update_count"]
        self.frame_count = checkpoint["frame_count"]


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_dqn(
    env_name: str = "CartPole-v1",
    num_episodes: int = 500,
    double_dqn: bool = False,
    dueling: bool = False,
    prioritized: bool = False,
    render: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Optional[DQNAgent], List[float]]:
    """
    Train DQN agent on Gymnasium environment.

    Args:
        env_name: Gymnasium environment identifier
        num_episodes: Training episodes
        double_dqn: Enable Double DQN
        dueling: Enable Dueling architecture
        prioritized: Enable PER
        render: Visualize environment
        seed: Random seed for reproducibility
        verbose: Print training progress

    Returns:
        Tuple (trained_agent, episode_rewards)
    """
    if not HAS_GYM:
        print("Error: gymnasium not installed")
        return None, []

    # Reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create environment
    env = gym.make(env_name, render_mode="human" if render else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Build algorithm name
    algo_parts = []
    if double_dqn:
        algo_parts.append("Double")
    if dueling:
        algo_parts.append("Dueling")
    if prioritized:
        algo_parts.append("PER")
    algo_parts.append("DQN")
    algo_name = " ".join(algo_parts)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training {algo_name}")
        print(f"Environment: {env_name}")
        print(f"State dim: {state_dim}, Actions: {action_dim}")
        print(f"{'=' * 60}")

    # Initialize agent
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=100,
        double_dqn=double_dqn,
        dueling=dueling,
        prioritized=prioritized
    )

    agent = DQNAgent(config)

    # Training loop
    rewards_history: List[float] = []
    best_avg_reward = float("-inf")

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode if seed else None)
        episode_reward = 0.0
        done = False

        while not done:
            # Select and execute action
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Train agent
            agent.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

        # Decay exploration
        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        # Progress reporting
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            best_avg_reward = max(best_avg_reward, avg_reward)
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg Reward: {avg_reward:7.2f} | "
                f"Best: {best_avg_reward:7.2f} | "
                f"ε: {agent.epsilon:.3f}"
            )

    env.close()

    # Final evaluation
    if verbose:
        print(f"\nFinal Evaluation ({algo_name}):")
        eval_rewards = evaluate_agent(agent, env_name, num_episodes=10)
        print(f"Eval Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")

    return agent, rewards_history


def evaluate_agent(
    agent: DQNAgent,
    env_name: str,
    num_episodes: int = 10,
    render: bool = False
) -> List[float]:
    """
    Evaluate trained agent (greedy policy, no exploration).

    Args:
        agent: Trained DQN agent
        env_name: Environment identifier
        num_episodes: Evaluation episodes
        render: Visualize evaluation

    Returns:
        List of episode rewards
    """
    if not HAS_GYM:
        return []

    env = gym.make(env_name, render_mode="human" if render else None)
    rewards: List[float] = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)

    env.close()
    return rewards


def compare_algorithms(
    env_name: str = "CartPole-v1",
    num_episodes: int = 300,
    seed: int = 42
) -> Dict[str, List[float]]:
    """
    Benchmark DQN variants.

    Compares: DQN, Double DQN, Dueling DQN, Double Dueling DQN

    Args:
        env_name: Environment identifier
        num_episodes: Training episodes per algorithm
        seed: Random seed

    Returns:
        Dictionary mapping algorithm names to reward histories
    """
    if not HAS_GYM:
        print("Error: gymnasium not installed")
        return {}

    results: Dict[str, List[float]] = {}

    algorithms = [
        ("DQN", False, False, False),
        ("Double DQN", True, False, False),
        ("Dueling DQN", False, True, False),
        ("Double Dueling DQN", True, True, False),
    ]

    for name, double, dueling, prioritized in algorithms:
        print(f"\nTraining {name}...")
        _, rewards = train_dqn(
            env_name=env_name,
            num_episodes=num_episodes,
            double_dqn=double,
            dueling=dueling,
            prioritized=prioritized,
            seed=seed,
            verbose=True
        )
        results[name] = rewards

    # Plot comparison
    if HAS_MATPLOTLIB and results:
        plot_comparison(results, env_name)

    return results


def plot_comparison(
    results: Dict[str, List[float]],
    env_name: str,
    window_size: int = 20
) -> None:
    """
    Plot algorithm comparison with smoothing.

    Args:
        results: Algorithm name → reward history
        env_name: Environment name for title
        window_size: Moving average window
    """
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10.colors

    for idx, (name, rewards) in enumerate(results.items()):
        if len(rewards) >= window_size:
            smoothed = np.convolve(
                rewards,
                np.ones(window_size) / window_size,
                mode="valid"
            )
            plt.plot(
                smoothed,
                label=name,
                alpha=0.8,
                color=colors[idx % len(colors)]
            )

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title(f"DQN Variants Comparison on {env_name}", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = "dqn_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.close()


def plot_training_curve(
    rewards: List[float],
    title: str = "Training Curve",
    window_size: int = 20
) -> None:
    """
    Plot single training curve with smoothing.

    Args:
        rewards: Episode rewards
        title: Plot title
        window_size: Moving average window
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed")
        return

    plt.figure(figsize=(10, 5))

    # Raw data (transparent)
    plt.plot(rewards, alpha=0.3, color="blue", label="Raw")

    # Smoothed curve
    if len(rewards) >= window_size:
        smoothed = np.convolve(
            rewards,
            np.ones(window_size) / window_size,
            mode="valid"
        )
        plt.plot(smoothed, color="blue", linewidth=2, label="Smoothed")

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = "dqn_training.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.close()


# =============================================================================
# Unit Tests
# =============================================================================

def run_unit_tests() -> bool:
    """
    Comprehensive unit test suite.

    Tests:
    1. ReplayBuffer correctness
    2. PrioritizedReplayBuffer correctness
    3. DQNNetwork architecture
    4. DuelingDQNNetwork architecture
    5. DQNAgent basic functionality
    6. Double DQN variant
    7. Dueling DQN variant
    8. PER DQN variant
    9. Environment interaction
    10. Model serialization

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 60)
    print("Running Unit Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: ReplayBuffer
    print("\n[Test 1] ReplayBuffer")
    try:
        buffer = ReplayBuffer(capacity=100)
        state = np.array([1.0, 2.0, 3.0, 4.0])
        for i in range(50):
            buffer.push(state, i % 2, float(i), state, False)

        assert len(buffer) == 50, f"Buffer size error: {len(buffer)}"
        assert buffer.is_ready(32), "Buffer should be ready"

        batch = buffer.sample(32)
        assert len(batch) == 5, f"Batch tuple size error: {len(batch)}"
        assert batch[0].shape == (32, 4), f"State shape error: {batch[0].shape}"

        print("  ✓ ReplayBuffer test passed")
    except Exception as e:
        print(f"  ✗ ReplayBuffer test failed: {e}")
        all_passed = False

    # Test 2: PrioritizedReplayBuffer
    print("\n[Test 2] PrioritizedReplayBuffer")
    try:
        per_buffer = PrioritizedReplayBuffer(capacity=100)
        state = np.array([1.0, 2.0, 3.0, 4.0])
        for i in range(50):
            per_buffer.push(state, i % 2, float(i), state, False)

        assert len(per_buffer) == 50, "PER buffer size error"

        batch = per_buffer.sample(32)
        assert len(batch) == 7, f"PER batch tuple size error: {len(batch)}"

        indices, weights = batch[5], batch[6]
        per_buffer.update_priorities(indices, np.random.randn(32))

        print("  ✓ PrioritizedReplayBuffer test passed")
    except Exception as e:
        print(f"  ✗ PrioritizedReplayBuffer test failed: {e}")
        all_passed = False

    # Test 3: DQNNetwork
    print("\n[Test 3] DQNNetwork")
    try:
        net = DQNNetwork(input_dim=4, output_dim=2, hidden_dim=64)
        x = torch.randn(32, 4)
        out = net(x)

        assert out.shape == (32, 2), f"Network output shape error: {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"

        print("  ✓ DQNNetwork test passed")
    except Exception as e:
        print(f"  ✗ DQNNetwork test failed: {e}")
        all_passed = False

    # Test 4: DuelingDQNNetwork
    print("\n[Test 4] DuelingDQNNetwork")
    try:
        net = DuelingDQNNetwork(input_dim=4, output_dim=2, hidden_dim=64)
        x = torch.randn(32, 4)
        out = net(x)

        assert out.shape == (32, 2), f"Dueling network output shape error: {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"

        print("  ✓ DuelingDQNNetwork test passed")
    except Exception as e:
        print(f"  ✗ DuelingDQNNetwork test failed: {e}")
        all_passed = False

    # Test 5: DQNAgent
    print("\n[Test 5] DQNAgent")
    try:
        config = DQNConfig(state_dim=4, action_dim=2, batch_size=32)
        agent = DQNAgent(config)

        state = np.random.randn(4).astype(np.float32)
        action = agent.get_action(state, training=True)

        assert 0 <= action < 2, f"Action out of range: {action}"

        # Fill buffer and test update
        for _ in range(100):
            s = np.random.randn(4).astype(np.float32)
            a = random.randint(0, 1)
            r = random.random()
            ns = np.random.randn(4).astype(np.float32)
            d = random.random() > 0.9
            agent.update(s, a, r, ns, d)

        # Check loss computation
        loss = agent.update(state, 0, 1.0, state, False)
        assert loss is not None, "Loss should not be None"

        print("  ✓ DQNAgent test passed")
    except Exception as e:
        print(f"  ✗ DQNAgent test failed: {e}")
        all_passed = False

    # Test 6: Double DQN
    print("\n[Test 6] Double DQN Agent")
    try:
        config = DQNConfig(
            state_dim=4, action_dim=2, batch_size=32, double_dqn=True
        )
        agent = DQNAgent(config)

        for _ in range(100):
            s = np.random.randn(4).astype(np.float32)
            agent.update(s, 0, 1.0, s, False)

        print("  ✓ Double DQN test passed")
    except Exception as e:
        print(f"  ✗ Double DQN test failed: {e}")
        all_passed = False

    # Test 7: Dueling DQN
    print("\n[Test 7] Dueling DQN Agent")
    try:
        config = DQNConfig(
            state_dim=4, action_dim=2, batch_size=32, dueling=True
        )
        agent = DQNAgent(config)

        for _ in range(100):
            s = np.random.randn(4).astype(np.float32)
            agent.update(s, 0, 1.0, s, False)

        print("  ✓ Dueling DQN test passed")
    except Exception as e:
        print(f"  ✗ Dueling DQN test failed: {e}")
        all_passed = False

    # Test 8: PER DQN
    print("\n[Test 8] PER DQN Agent")
    try:
        config = DQNConfig(
            state_dim=4, action_dim=2, batch_size=32, prioritized=True
        )
        agent = DQNAgent(config)

        for _ in range(100):
            s = np.random.randn(4).astype(np.float32)
            agent.update(s, 0, 1.0, s, False)

        print("  ✓ PER DQN test passed")
    except Exception as e:
        print(f"  ✗ PER DQN test failed: {e}")
        all_passed = False

    # Test 9: Environment interaction
    print("\n[Test 9] Environment Interaction")
    if HAS_GYM:
        try:
            env = gym.make("CartPole-v1")
            config = DQNConfig(state_dim=4, action_dim=2, batch_size=32)
            agent = DQNAgent(config)

            state, _ = env.reset()
            for _ in range(100):
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(state, action, reward, next_state, terminated or truncated)
                if terminated or truncated:
                    state, _ = env.reset()
                else:
                    state = next_state

            env.close()
            print("  ✓ Environment interaction test passed")
        except Exception as e:
            print(f"  ✗ Environment interaction test failed: {e}")
            all_passed = False
    else:
        print("  - Skipped (gymnasium not installed)")

    # Test 10: Model save/load
    print("\n[Test 10] Model Save/Load")
    try:
        import tempfile
        config = DQNConfig(state_dim=4, action_dim=2)
        agent = DQNAgent(config)

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        agent.save(temp_path)

        # Load
        agent2 = DQNAgent(config)
        agent2.load(temp_path)

        # Cleanup
        os.remove(temp_path)

        # Verify parameters match
        for p1, p2 in zip(
            agent.q_network.parameters(),
            agent2.q_network.parameters()
        ):
            assert torch.allclose(p1, p2), "Parameters mismatch"

        print("  ✓ Model save/load test passed")
    except Exception as e:
        print(f"  ✗ Model save/load test failed: {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed")
    else:
        print("Some tests failed")
    print("=" * 60)

    return all_passed


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Command-line interface for DQN training and evaluation.

    Usage:
        python dqn_cartpole.py --train --episodes 500
        python dqn_cartpole.py --compare --episodes 300
        python dqn_cartpole.py --test
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep Q-Network Implementation"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare algorithm variants"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run unit tests"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Training episodes"
    )
    parser.add_argument(
        "--double",
        action="store_true",
        help="Enable Double DQN"
    )
    parser.add_argument(
        "--dueling",
        action="store_true",
        help="Enable Dueling DQN"
    )
    parser.add_argument(
        "--per",
        action="store_true",
        help="Enable prioritized experience replay"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Execute requested operation
    if args.compare:
        compare_algorithms(
            num_episodes=args.episodes,
            seed=args.seed
        )
    elif args.train:
        agent, rewards = train_dqn(
            num_episodes=args.episodes,
            double_dqn=args.double,
            dueling=args.dueling,
            prioritized=args.per,
            seed=args.seed
        )

        if rewards and HAS_MATPLOTLIB:
            algo_name = "DQN"
            if args.double:
                algo_name = "Double " + algo_name
            if args.dueling:
                algo_name = "Dueling " + algo_name
            plot_training_curve(
                rewards,
                title=f"{algo_name} Training on CartPole-v1"
            )
    else:
        # Default: Run tests
        run_unit_tests()


if __name__ == "__main__":
    main()
