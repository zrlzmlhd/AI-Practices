"""Tabular Q-Learning and SARSA Implementation.

This module provides production-ready implementations of classic temporal-difference
control algorithms for discrete state-action spaces.

Core Algorithms:
    - Q-Learning: Off-policy TD control learning optimal action-value function
    - SARSA: On-policy TD control learning policy's value function
    - Expected SARSA: On-policy variant using expected Q-values
    - Double Q-Learning: Reduces overestimation bias via action decoupling

Mathematical Foundations:

    Q-Learning Update (Off-Policy):
        Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]

        Core Idea: Directly approximates optimal action-value function Q* by using
        max operator in TD target, independent of behavior policy.

        Complexity: O(|A|) per update for max operation

    SARSA Update (On-Policy):
        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

        Core Idea: Learns value of policy being followed by using actual next
        action a' in TD target, accounting for exploration risks.

        Complexity: O(1) per update, no max needed

    Double Q-Learning Update:
        Q₁(s,a) ← Q₁(s,a) + α[r + γ Q₂(s', argmax_{a'} Q₁(s',a')) - Q₁(s,a)]
        Q₂(s,a) ← Q₂(s,a) + α[r + γ Q₁(s', argmax_{a'} Q₂(s',a')) - Q₂(s,a)]

        Core Idea: Decouples action selection and evaluation using two Q-tables,
        mitigating positive bias from max operator in standard Q-Learning.

        Proof Sketch: E[Q₂(s', argmax Q₁)] ≤ max E[Q₂] under unbiased estimation.

Problem Context:
    Traditional dynamic programming requires full environment model P(s'|s,a) and
    R(s,a,s'). Model-free TD methods learn directly from experience trajectories,
    enabling application to:
    - Unknown transition dynamics
    - Large/continuous state spaces (via function approximation)
    - Online learning scenarios

Algorithm Comparison:

    | Property        | Q-Learning | SARSA   | Expected SARSA | Double Q |
    |-----------------|------------|---------|----------------|----------|
    | Policy Type     | Off        | On      | On             | Off      |
    | Variance        | Medium     | High    | Low            | Medium   |
    | Bias            | High       | Medium  | Medium         | Low      |
    | Convergence     | To Q*      | To Q^π  | To Q^π         | To Q*    |
    | Safety          | Risky      | Safe    | Moderate       | Risky    |
    | Sample Reuse    | Yes        | No      | No             | Yes      |

    Convergence Guarantees (Robbins-Monro Conditions):
        Σ αₜ = ∞ and Σ αₜ² < ∞
        With appropriate decay (e.g., αₜ = 1/(1+visits)), Q-Learning and
        Double Q-Learning converge to Q* w.p. 1 under tabular representation.

References:
    [1] Watkins, C.J.C.H. (1989). Learning from Delayed Rewards. PhD Thesis.
    [2] Rummery & Niranjan (1994). On-Line Q-Learning Using Connectionist Systems.
    [3] Van Hasselt (2010). Double Q-learning. NeurIPS.
    [4] Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.

Dependencies:
    pip install numpy matplotlib gymnasium

Author: Research Implementation for Production & Academic Use
License: MIT
"""

from __future__ import annotations

import json
import pickle
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

# Type aliases for generic state-action spaces
State = TypeVar('State')
Action = TypeVar('Action', bound=int)


class ExplorationStrategy(Enum):
    """Enumeration of exploration strategies for action selection.

    Attributes:
        EPSILON_GREEDY: ε-greedy with uniform random exploration
        SOFTMAX: Boltzmann exploration with temperature parameter
        UCB: Upper Confidence Bound balancing exploitation and uncertainty
    """
    EPSILON_GREEDY = "epsilon_greedy"
    SOFTMAX = "softmax"
    UCB = "ucb"


@dataclass
class AgentConfig:
    """Configuration parameters for tabular RL agents.

    Attributes:
        n_actions: Size of discrete action space |A|
        learning_rate: Step size α ∈ (0,1] controlling update magnitude
        discount_factor: Discount γ ∈ [0,1] balancing immediate vs future rewards
        epsilon: Initial exploration rate for ε-greedy (typically 1.0)
        epsilon_decay: Multiplicative decay rate per episode (e.g., 0.995)
        epsilon_min: Lower bound on exploration to maintain some stochasticity
        exploration: Strategy for balancing exploration-exploitation tradeoff
        temperature: Softmax temperature τ > 0 (τ→0 greedy, τ→∞ uniform)
        ucb_c: UCB exploration coefficient c ≥ 0 weighting uncertainty bonus
    """
    n_actions: int
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    exploration: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    temperature: float = 1.0
    ucb_c: float = 2.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {self.n_actions}")
        if not 0 < self.learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0,1], got {self.learning_rate}")
        if not 0 <= self.discount_factor <= 1:
            raise ValueError(f"discount_factor must be in [0,1], got {self.discount_factor}")
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f"epsilon must be in [0,1], got {self.epsilon}")
        if self.epsilon_decay <= 0:
            raise ValueError(f"epsilon_decay must be positive, got {self.epsilon_decay}")
        if not 0 <= self.epsilon_min <= self.epsilon:
            raise ValueError(f"epsilon_min must be in [0,epsilon], got {self.epsilon_min}")


@dataclass
class TrainingMetrics:
    """Container for training statistics and performance metrics.

    Attributes:
        episode_rewards: Cumulative reward Σr per episode
        episode_lengths: Number of timesteps until termination per episode
        epsilon_history: Exploration rate trajectory over training
        td_errors: Temporal-difference errors δₜ = target - Q(s,a)
    """
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)
    td_errors: List[float] = field(default_factory=list)

    def get_moving_average(self, window: int = 100) -> np.ndarray:
        """Compute moving average of episode rewards for smoothed visualization.

        Args:
            window: Sliding window size for averaging

        Returns:
            Smoothed reward trajectory of length max(0, len(rewards) - window + 1)

        Complexity: O(n) via convolution
        """
        if len(self.episode_rewards) < window:
            return np.array(self.episode_rewards, dtype=np.float32)

        return np.convolve(
            self.episode_rewards,
            np.ones(window, dtype=np.float32) / window,
            mode='valid'
        )


class ExplorationMixin:
    """Mixin providing exploration strategies for action selection.

    Implements three classic exploration methods with numerical stability:
        1. ε-greedy: Simple random exploration with probability ε
        2. Softmax (Boltzmann): Probabilistic selection via Gibbs distribution
        3. UCB: Optimistic exploration via confidence bound on Q-values
    """

    def _epsilon_greedy(self, q_values: np.ndarray, epsilon: float) -> int:
        """ε-greedy action selection with tie-breaking.

        Mathematical Formulation:
            π(a|s) = { ε/|A| + (1-ε)     if a = argmax Q(s,a)
                     { ε/|A|              otherwise

        Args:
            q_values: Action values Q(s,·) of shape (n_actions,)
            epsilon: Exploration probability ∈ [0,1]

        Returns:
            Selected action index

        Complexity: O(|A|) for argmax operation

        Implementation Notes:
            - Uses random tie-breaking when multiple actions have max Q-value
            - Ensures uniform exploration probability across all actions
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0,1], got {epsilon}")

        if np.random.random() < epsilon:
            return np.random.randint(len(q_values))

        # Tie-breaking: randomly select among actions with maximum Q-value
        max_q = np.max(q_values)
        max_actions = np.where(np.isclose(q_values, max_q, rtol=1e-9))[0]
        return int(np.random.choice(max_actions))

    def _softmax(self, q_values: np.ndarray, temperature: float) -> int:
        """Softmax (Boltzmann) exploration via Gibbs distribution.

        Mathematical Formulation:
            π(a|s) = exp(Q(s,a)/τ) / Σ_{a'} exp(Q(s,a')/τ)

        Properties:
            - τ → 0: Approaches greedy policy (max exploitation)
            - τ → ∞: Approaches uniform random policy (max exploration)
            - τ = 1: Standard Boltzmann distribution

        Args:
            q_values: Action values Q(s,·) of shape (n_actions,)
            temperature: Temperature parameter τ > 0 controlling exploration

        Returns:
            Sampled action from Gibbs distribution

        Complexity: O(|A|) for normalization

        Numerical Stability:
            Subtracts max Q-value before exponentiation to prevent overflow:
            exp(Q/τ) / Σexp(Q/τ) = exp((Q-max)/τ) / Σexp((Q-max)/τ)
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        # Numerical stability: subtract max to prevent overflow
        q_scaled = (q_values - np.max(q_values)) / max(temperature, 1e-8)
        exp_q = np.exp(np.clip(q_scaled, -700, 700))  # Prevent exp overflow

        # Normalize to probability distribution
        probs = exp_q / (np.sum(exp_q) + 1e-10)

        return int(np.random.choice(len(q_values), p=probs))

    def _ucb(
        self,
        q_values: np.ndarray,
        action_counts: np.ndarray,
        total_count: int,
        c: float
    ) -> int:
        """Upper Confidence Bound (UCB) exploration strategy.

        Mathematical Formulation:
            A_t = argmax_a [Q(s,a) + c√(ln t / N(s,a))]

        where:
            Q(s,a): Estimated value (exploitation term)
            c√(ln t / N(s,a)): Confidence bound (exploration bonus)
            N(s,a): Visit count for state-action pair
            t: Total timesteps

        Theoretical Foundation:
            Based on Hoeffding inequality, confidence bound decreases as O(1/√n)
            with visitation count. Guarantees logarithmic regret in multi-armed
            bandits: R_T = O(√(K T ln T)) for K actions.

        Args:
            q_values: Current Q-value estimates
            action_counts: Per-action visit counts N(s,·)
            total_count: Total visits to state
            c: Exploration coefficient controlling tradeoff

        Returns:
            Action with maximum UCB value

        Complexity: O(|A|) for computing all UCB values

        Implementation Notes:
            - Prioritizes unvisited actions (infinite confidence bound)
            - Adds small constant to denominator for numerical stability
        """
        if c < 0:
            raise ValueError(f"UCB coefficient must be non-negative, got {c}")

        # Unvisited actions have infinite confidence bound -> explore first
        if np.any(action_counts == 0):
            unvisited = np.where(action_counts == 0)[0]
            return int(np.random.choice(unvisited))

        # Compute UCB values for all actions
        exploration_bonus = c * np.sqrt(
            np.log(total_count + 1) / (action_counts + 1e-10)
        )
        ucb_values = q_values + exploration_bonus

        return int(np.argmax(ucb_values))


class BaseAgent(ABC, ExplorationMixin):
    """Abstract base class for tabular temporal-difference learning agents.

    Provides common infrastructure for Q-table management, exploration policies,
    and model persistence. Subclasses implement specific update rules (Q-Learning,
    SARSA, etc.) via the update() method.

    Architecture:
        - Hash-based Q-table using defaultdict for lazy state initialization
        - Pluggable exploration strategies (ε-greedy, softmax, UCB)
        - JSON/pickle serialization for model checkpointing

    Memory Complexity: O(|S| × |A|) for Q-table storage
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize agent with configuration parameters.

        Args:
            config: Validated configuration object
        """
        self.config = config
        self.n_actions = config.n_actions
        self.lr = config.learning_rate
        self.gamma = config.discount_factor
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.exploration = config.exploration
        self.temperature = config.temperature
        self.ucb_c = config.ucb_c

        # Q-table: state → action-value vector
        self.q_table: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )

        # UCB bookkeeping: per-state action selection counts
        self.action_counts: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.int32)
        )
        self.total_steps = 0

        # Training metrics
        self.metrics = TrainingMetrics()

    def get_action(self, state: Any, training: bool = True) -> int:
        """Select action according to current policy and exploration strategy.

        Args:
            state: Current environment state (must be hashable)
            training: If True, uses exploration strategy; if False, pure greedy

        Returns:
            Selected action index ∈ {0, ..., |A|-1}

        Raises:
            ValueError: If state is not hashable or exploration strategy invalid
        """
        try:
            q_values = self.q_table[state]
        except TypeError:
            raise TypeError(f"State must be hashable, got {type(state)}")

        # Evaluation mode: greedy policy with tie-breaking
        if not training:
            max_q = np.max(q_values)
            max_actions = np.where(np.isclose(q_values, max_q, rtol=1e-9))[0]
            return int(np.random.choice(max_actions))

        # Training mode: apply exploration strategy
        if self.exploration == ExplorationStrategy.EPSILON_GREEDY:
            action = self._epsilon_greedy(q_values, self.epsilon)
        elif self.exploration == ExplorationStrategy.SOFTMAX:
            action = self._softmax(q_values, self.temperature)
        elif self.exploration == ExplorationStrategy.UCB:
            action = self._ucb(
                q_values,
                self.action_counts[state],
                self.total_steps,
                self.ucb_c
            )
        else:
            warnings.warn(f"Unknown exploration strategy {self.exploration}, using ε-greedy")
            action = self._epsilon_greedy(q_values, self.epsilon)

        # Update statistics
        self.action_counts[state][action] += 1
        self.total_steps += 1

        return int(action)

    @abstractmethod
    def update(self, *args, **kwargs) -> float:
        """Update Q-values based on experience tuple.

        Must be implemented by subclasses to define specific TD update rule.

        Returns:
            TD error δ = target - Q(s,a) for convergence diagnostics
        """
        pass

    def decay_epsilon(self) -> None:
        """Exponentially decay exploration rate.

        Update Rule:
            ε ← max(ε_min, ε × decay_rate)

        Ensures ε remains above minimum threshold for continued exploration.
        """
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def get_greedy_policy(self) -> Dict[Any, int]:
        """Extract greedy policy π*(s) = argmax_a Q(s,a) from Q-table.

        Returns:
            Dictionary mapping states to optimal actions

        Complexity: O(|S| × |A|) for extracting all state policies
        """
        policy = {}
        for state, q_vals in self.q_table.items():
            policy[state] = int(np.argmax(q_vals))
        return policy

    def get_value_function(self) -> Dict[Any, float]:
        """Compute state-value function V(s) = max_a Q(s,a).

        Returns:
            Dictionary mapping states to optimal values

        Complexity: O(|S| × |A|) for computing all state values
        """
        return {
            state: float(np.max(q_values))
            for state, q_values in self.q_table.items()
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize agent to disk for checkpointing.

        Supports JSON (human-readable) and pickle (efficient binary) formats.

        Args:
            filepath: Output path with extension (.json or .pkl)

        Raises:
            IOError: If file write fails
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'config': {
                'n_actions': self.n_actions,
                'learning_rate': self.lr,
                'discount_factor': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
            },
            'q_table': {str(k): v.tolist() for k, v in self.q_table.items()},
            'total_steps': self.total_steps,
        }

        try:
            if filepath.suffix == '.json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except IOError as e:
            raise IOError(f"Failed to save model to {filepath}: {e}")

    def load(self, filepath: Union[str, Path]) -> None:
        """Load agent state from checkpoint file.

        Args:
            filepath: Input path to serialized model

        Raises:
            IOError: If file read fails
            ValueError: If checkpoint format is incompatible
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        try:
            if filepath.suffix == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to load model from {filepath}: {e}")

        # Restore Q-table
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
        for k, v in data['q_table'].items():
            try:
                key = eval(k)  # Attempt to restore tuple keys
            except:
                key = int(k) if k.isdigit() else k
            self.q_table[key] = np.array(v, dtype=np.float32)

        self.total_steps = data.get('total_steps', 0)

    def reset(self) -> None:
        """Reset agent to initial state for new training run.

        Clears Q-table, statistics, and resets exploration rate.
        """
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
        self.action_counts = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.int32))
        self.total_steps = 0
        self.epsilon = self.config.epsilon
        self.metrics = TrainingMetrics()


class QLearningAgent(BaseAgent):
    """Q-Learning: Off-policy temporal-difference control.

    Core Idea:
        Directly learns optimal action-value function Q* by using max operator
        in TD target, independent of behavior policy. Converges to optimal policy
        even when following exploratory policy.

    Mathematical Principle:
        Update Rule:
            Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]

        Where:
            α: Learning rate (step size)
            γ: Discount factor
            r: Immediate reward
            max_{a'}: Maximum over next actions (key difference from SARSA)

        TD Target:
            y = r + γ max_{a'} Q(s',a')

        TD Error:
            δ = y - Q(s,a)

    Problem Context:
        Introduced by Watkins (1989) to learn optimal policies without requiring
        environment model. Breakthrough: off-policy learning enables data reuse
        and learning from demonstrations.

    Comparison with Alternatives:
        vs SARSA:
            + Learns optimal policy regardless of exploration
            + Allows experience replay and importance sampling
            - Higher variance due to max operator
            - Can be unsafe in exploration (cliff-walking paradox)

        vs Double Q-Learning:
            - Suffers from positive bias: E[max Q] ≥ max E[Q]
            + Simpler implementation with single Q-table

        vs DQN:
            + Guaranteed convergence in tabular case
            - Cannot handle large/continuous state spaces

    Complexity Analysis:
        Time per update: O(|A|) for max operation
        Space: O(|S| × |A|) for Q-table storage
        Sample complexity: O(|S||A|/(1-γ)²ε²) for ε-optimal policy (PAC bound)

    Theoretical Properties:
        Convergence: To Q* w.p. 1 if:
            1. All state-action pairs visited infinitely often
            2. Learning rate satisfies Robbins-Monro: Σαₜ=∞, Σαₜ²<∞
        Optimality: Extracted greedy policy is optimal: π*(s) = argmax_a Q*(s,a)

    Summary:
        Q-Learning revolutionized RL by proving optimal policies can be learned
        through pure experience without environment models. Its off-policy nature
        enables efficient use of data and forms foundation for modern deep RL
        (DQN, Rainbow). Trade-off: optimistic learning may be unsafe during training.

    Example:
        >>> config = AgentConfig(n_actions=4, learning_rate=0.1, discount_factor=0.99)
        >>> agent = QLearningAgent(config)
        >>>
        >>> state = env.reset()
        >>> action = agent.get_action(state)
        >>> next_state, reward, done, _ = env.step(action)
        >>> td_error = agent.update(state, action, reward, next_state, done)
        >>> agent.decay_epsilon()
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        double_q: bool = False
    ) -> None:
        """Initialize Q-Learning agent.

        Args:
            config: Configuration object (overrides individual parameters if provided)
            n_actions: Size of action space
            learning_rate: TD learning rate α
            discount_factor: Reward discount γ
            epsilon: Initial exploration rate
            epsilon_decay: Multiplicative decay per episode
            epsilon_min: Minimum exploration rate
            double_q: If True, uses Double Q-Learning variant
        """
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            )
        super().__init__(config)

        self.double_q = double_q
        if double_q:
            self.q_table2: Dict[Any, np.ndarray] = defaultdict(
                lambda: np.zeros(self.n_actions, dtype=np.float32)
            )

    def get_action(self, state: Any, training: bool = True) -> int:
        """Select action using combined Q-values for Double Q-Learning.

        For standard Q-Learning, uses single Q-table.
        For Double Q-Learning, uses sum Q₁(s,a) + Q₂(s,a) for action selection.
        """
        if self.double_q:
            q_values = self.q_table[state] + self.q_table2[state]
        else:
            q_values = self.q_table[state]

        if not training:
            max_q = np.max(q_values)
            max_actions = np.where(np.isclose(q_values, max_q, rtol=1e-9))[0]
            return int(np.random.choice(max_actions))

        if self.exploration == ExplorationStrategy.EPSILON_GREEDY:
            action = self._epsilon_greedy(q_values, self.epsilon)
        elif self.exploration == ExplorationStrategy.SOFTMAX:
            action = self._softmax(q_values, self.temperature)
        else:
            action = self._ucb(
                q_values,
                self.action_counts[state],
                self.total_steps,
                self.ucb_c
            )

        self.action_counts[state][action] += 1
        self.total_steps += 1
        return int(action)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ) -> float:
        """Apply Q-Learning update rule.

        Standard Q-Learning:
            Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]

        Uses TD target with max operator over next actions, regardless of
        actual action taken (off-policy property).

        Args:
            state: Current state s
            action: Action taken a
            reward: Immediate reward r
            next_state: Resulting state s'
            done: True if s' is terminal

        Returns:
            TD error δ for convergence monitoring

        Raises:
            ValueError: If action is out of bounds
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} out of range [0, {self.n_actions})")

        if self.double_q:
            return self._double_q_update(state, action, reward, next_state, done)

        current_q = self.q_table[state][action]

        # Compute TD target
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD error
        td_error = target - current_q

        # Q-value update
        self.q_table[state][action] += self.lr * td_error

        return float(td_error)

    def _double_q_update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ) -> float:
        """Double Q-Learning update to reduce overestimation bias.

        Mathematical Formulation:
            With probability 0.5:
                Q₁(s,a) ← Q₁(s,a) + α[r + γ Q₂(s', argmax_{a'} Q₁(s',a')) - Q₁(s,a)]
            Else:
                Q₂(s,a) ← Q₂(s,a) + α[r + γ Q₁(s', argmax_{a'} Q₂(s',a')) - Q₂(s,a)]

        Key Insight:
            Decouples action selection (using Q₁) from value estimation (using Q₂)
            in alternating updates. Reduces positive bias because:
            E[Q₂(s', argmax Q₁)] ≤ max E[Q₂] ≈ Q*(s',a*)

        Args:
            state, action, reward, next_state, done: Standard transition tuple

        Returns:
            TD error from updated Q-table
        """
        if np.random.random() < 0.5:
            # Update Q₁: select with Q₁, evaluate with Q₂
            current_q = self.q_table[state][action]
            if done:
                target = reward
            else:
                best_action = int(np.argmax(self.q_table[next_state]))
                target = reward + self.gamma * self.q_table2[next_state][best_action]
            td_error = target - current_q
            self.q_table[state][action] += self.lr * td_error
        else:
            # Update Q₂: select with Q₂, evaluate with Q₁
            current_q = self.q_table2[state][action]
            if done:
                target = reward
            else:
                best_action = int(np.argmax(self.q_table2[next_state]))
                target = reward + self.gamma * self.q_table[next_state][best_action]
            td_error = target - current_q
            self.q_table2[state][action] += self.lr * td_error

        return float(td_error)


class SARSAAgent(BaseAgent):
    """SARSA: On-policy temporal-difference control.

    Core Idea:
        Learns value function of the policy being executed, accounting for
        exploration risks. Uses actual next action A' instead of max, making
        it more conservative but safer during training.

    Mathematical Principle:
        Update Rule:
            Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

        Where a' is the action actually taken in state s' (not max).

        TD Target:
            y = r + γ Q(s',a')

        Key Difference from Q-Learning:
            Q-Learning: y = r + γ max_{a'} Q(s',a')  [off-policy, optimistic]
            SARSA:      y = r + γ Q(s',a')            [on-policy, realistic]

    Problem Context:
        Addresses safety concerns in Q-Learning where agent learns optimal
        policy but behaves dangerously during training due to exploration.
        SARSA accounts for exploration risk in learned values.

    Comparison with Q-Learning:
        Cliff Walking Example:
            - Q-Learning: Learns path near cliff (optimal) but falls often during training
            - SARSA: Learns safer path far from cliff (suboptimal but accounts for ε-greedy)

        Properties:
            + More stable training (lower variance)
            + Safer exploration in risky environments
            - Converges to Q^π, not Q* (suboptimal if policy remains exploratory)
            - Cannot use experience replay (needs on-policy data)

    Complexity Analysis:
        Time per update: O(1) - no max operation needed
        Space: O(|S| × |A|) for Q-table
        Sample complexity: Similar to Q-Learning but may require more samples
                          due to higher variance per update

    Theoretical Properties:
        Convergence: To Q^π (value of current policy) under Robbins-Monro conditions
        GLIE Condition: If policy becomes greedy in limit (ε→0), converges to Q*
        Variance: Higher than Q-Learning due to sampling single next action

    Summary:
        SARSA represents conservative alternative to Q-Learning, trading off
        optimality for safety. More appropriate for real-world applications where
        exploration mistakes have consequences (robotics, healthcare). Learns
        "what will I actually achieve with this policy" rather than "what could
        I achieve if I acted optimally."

    Example:
        >>> agent = SARSAAgent(config)
        >>> state = env.reset()
        >>> action = agent.get_action(state)
        >>>
        >>> while not done:
        >>>     next_state, reward, done, _ = env.step(action)
        >>>     next_action = agent.get_action(next_state)  # Sample next action first
        >>>     agent.update(state, action, reward, next_state, next_action, done)
        >>>     state, action = next_state, next_action
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ) -> None:
        """Initialize SARSA agent.

        Args:
            config: Configuration object (overrides individual parameters)
            n_actions: Action space size
            learning_rate: TD step size α
            discount_factor: Reward discount γ
            epsilon: Initial exploration rate
            epsilon_decay: Multiplicative decay per episode
            epsilon_min: Minimum exploration rate
        """
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            )
        super().__init__(config)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        next_action: int,
        done: bool
    ) -> float:
        """Apply SARSA update rule using actual next action.

        Update Formula:
            Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

        Critical: Requires next_action as argument (sampled from policy before update).
        This is the defining characteristic of on-policy learning.

        Args:
            state: Current state s
            action: Action taken a
            reward: Immediate reward r
            next_state: Resulting state s'
            next_action: Next action a' sampled from policy π(·|s')
            done: True if s' is terminal

        Returns:
            TD error δ = target - Q(s,a)

        Raises:
            ValueError: If actions are out of bounds
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} out of range [0, {self.n_actions})")
        if not 0 <= next_action < self.n_actions:
            raise ValueError(f"Next action {next_action} out of range [0, {self.n_actions})")

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            # SARSA: use actual next action (on-policy)
            target = reward + self.gamma * self.q_table[next_state][next_action]

        td_error = target - current_q
        self.q_table[state][action] += self.lr * td_error

        return float(td_error)


class ExpectedSARSAAgent(BaseAgent):
    """Expected SARSA: Hybrid on-policy control with reduced variance.

    Core Idea:
        Instead of sampling single next action (SARSA) or taking max (Q-Learning),
        uses expected value over next actions under current policy. Reduces variance
        while maintaining on-policy properties.

    Mathematical Principle:
        Update Rule:
            Q(s,a) ← Q(s,a) + α[r + γ E_π[Q(s',·)] - Q(s,a)]

        Where expectation is taken under policy π:
            E_π[Q(s',·)] = Σ_{a'} π(a'|s') Q(s',a')

        For ε-greedy policy:
            π(a'|s') = { ε/|A| + (1-ε)  if a' = argmax Q(s',a')
                       { ε/|A|           otherwise

        Expected Q-value:
            E[Q] = (ε/|A|) Σ_a Q(s',a) + (1-ε) max_a Q(s',a)

    Problem Context:
        Addresses high variance of SARSA while maintaining on-policy learning.
        Particularly effective when action space is large, as expectation provides
        more stable gradient estimate than single-action sample.

    Comparison:
        vs SARSA:
            + Lower variance (uses expectation instead of sample)
            + Faster convergence in practice
            = Same convergence guarantees (both on-policy)
            - Computational overhead: O(|A|) instead of O(1)

        vs Q-Learning:
            + On-policy (can account for exploration)
            + Lower variance than SARSA
            - Still converges to Q^π, not Q* (unless GLIE)

        vs Q-Learning (special case):
            When ε=0 (greedy), Expected SARSA reduces to Q-Learning

    Complexity Analysis:
        Time per update: O(|A|) for computing expectation
        Space: O(|S| × |A|) for Q-table
        Variance: Lower than SARSA, comparable to Q-Learning

    Theoretical Properties:
        Convergence: To Q^π under standard conditions
        Variance reduction: Var[E[Q]] = 0 vs Var[Q(s',A')] > 0 for SARSA
        Bias: Same as SARSA (both on-policy)

    Summary:
        Expected SARSA represents best of both worlds: on-policy learning with
        Q-Learning's low variance. Particularly useful in continuous control and
        large action spaces where sampling variance is problematic. Slightly more
        computationally expensive but often converges faster in practice.

    Example:
        >>> agent = ExpectedSARSAAgent(config)
        >>> state = env.reset()
        >>>
        >>> while not done:
        >>>     action = agent.get_action(state)
        >>>     next_state, reward, done, _ = env.step(action)
        >>>     agent.update(state, action, reward, next_state, done)  # No next_action needed
        >>>     state = next_state
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ) -> None:
        """Initialize Expected SARSA agent.

        Args:
            config: Configuration object
            n_actions: Action space size
            learning_rate: TD step size α
            discount_factor: Discount factor γ
            epsilon: Exploration rate
            epsilon_decay: Decay rate per episode
            epsilon_min: Minimum exploration
        """
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            )
        super().__init__(config)

    def _get_expected_q(self, state: Any) -> float:
        """Compute expected Q-value under ε-greedy policy.

        Mathematical Derivation:
            E_π[Q(s,·)] = Σ_a π(a|s) Q(s,a)

            For ε-greedy:
                = (ε/|A|) Σ_a Q(s,a) + (1-ε) Q(s, a*)

            where a* = argmax_a Q(s,a)

        Args:
            state: State to compute expected value for

        Returns:
            Expected Q-value as scalar

        Complexity: O(|A|) for computing weighted sum
        """
        q_values = self.q_table[state]
        n_actions = len(q_values)

        # Construct ε-greedy action probabilities
        probs = np.ones(n_actions, dtype=np.float32) * (self.epsilon / n_actions)
        best_action = int(np.argmax(q_values))
        probs[best_action] += (1.0 - self.epsilon)

        # Compute expectation: E[Q] = Σ π(a) Q(a)
        expected_q = np.dot(probs, q_values)

        return float(expected_q)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ) -> float:
        """Apply Expected SARSA update rule.

        Update Formula:
            Q(s,a) ← Q(s,a) + α[r + γ E_π[Q(s',·)] - Q(s,a)]

        Computes expected Q-value over all next actions weighted by policy
        probabilities. Provides lower variance than sampling.

        Args:
            state: Current state s
            action: Action taken a
            reward: Immediate reward r
            next_state: Resulting state s'
            done: True if s' is terminal

        Returns:
            TD error δ

        Raises:
            ValueError: If action is out of bounds
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} out of range [0, {self.n_actions})")

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            # Expected SARSA: use expected Q under current policy
            expected_q = self._get_expected_q(next_state)
            target = reward + self.gamma * expected_q

        td_error = target - current_q
        self.q_table[state][action] += self.lr * td_error

        return float(td_error)


class CliffWalkingEnv:
    """Cliff Walking environment for evaluating exploration-exploitation tradeoff.

    Problem Description:
        Classic gridworld demonstrating stark behavioral differences between
        on-policy (SARSA) and off-policy (Q-Learning) methods. Agent must navigate
        from start to goal while avoiding cliff. Falling incurs severe penalty.

    Environment Layout (4×12 grid):
        ┌─────────────────────────────────────────────┐
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 0
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 1
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 2
        │ S  C  C  C  C  C  C  C  C  C  C  G  │  row 3
        └─────────────────────────────────────────────┘
          0  1  2  3  4  5  6  7  8  9 10 11  columns

        Legend:
            S: Start position (3,0)
            G: Goal position (3,11)
            C: Cliff cells (3,1) to (3,10)
            .: Safe cells

    Dynamics:
        State Space: 4×12 = 48 discrete states (row, col)
        Action Space: {0:Up, 1:Right, 2:Down, 3:Left}
        Transition: Deterministic grid movement, clipped at boundaries

    Reward Structure:
        - Step cost: -1 (encourages short paths)
        - Cliff penalty: -100 and reset to start (severe punishment)
        - Goal reward: 0 and episode terminates

    Optimal Policy:
        Shortest path along cliff edge: 13 steps, return = -13
        Safe path avoiding cliff: ~30 steps, return = -30

    Behavioral Analysis:
        Q-Learning (Off-Policy):
            - Learns optimal risky path (along cliff)
            - Updates toward max Q regardless of exploration
            - High performance once converged
            - Dangerous during training (frequent cliff falls)

        SARSA (On-Policy):
            - Learns safe conservative path (far from cliff)
            - Updates account for ε-greedy exploration risk
            - Lower asymptotic performance
            - Safer training (fewer cliff falls)

        This dichotomy demonstrates fundamental difference: Q-Learning learns
        "what's optimal to do" while SARSA learns "what will I actually do given
        my exploration strategy."

    Implementation Notes:
        - Falling resets to start but episode continues (done=False)
        - Goal terminates episode (done=True)
        - Boundary collisions result in no movement

    Mathematical Properties:
        State-value of optimal path: V*(s_start) = -13
        State-value of safe path: V^ε-greedy(s_start) ≈ -30 for small ε
        Cliff cells: Strongly negative Q-values Q(s,a) << 0 for adjacent actions

    Example:
        >>> env = CliffWalkingEnv()
        >>> state = env.reset()
        >>>
        >>> # Take action Right (move along cliff edge)
        >>> next_state, reward, done = env.step(1)
        >>>
        >>> # If next_state in cliff:
        >>> #   reward = -100, next_state = start, done = False
        >>> # Else:
        >>> #   reward = -1, next_state = new position
    """

    # Action space definition
    ACTIONS = {
        0: (-1, 0),   # Up: decrease row
        1: (0, 1),    # Right: increase column
        2: (1, 0),    # Down: increase row
        3: (0, -1)    # Left: decrease column
    }
    ACTION_NAMES = ['Up', 'Right', 'Down', 'Left']

    def __init__(self, height: int = 4, width: int = 12) -> None:
        """Initialize cliff walking gridworld.

        Args:
            height: Number of rows (default 4)
            width: Number of columns (default 12)

        Raises:
            ValueError: If dimensions are too small
        """
        if height < 2 or width < 3:
            raise ValueError(f"Grid too small: {height}x{width}, need at least 2x3")

        self.height = height
        self.width = width

        # Special cell positions
        self.start = (height - 1, 0)          # Bottom-left corner
        self.goal = (height - 1, width - 1)   # Bottom-right corner
        self.cliff = [
            (height - 1, j) for j in range(1, width - 1)
        ]  # Bottom row excluding start and goal

        # Current state
        self.state = self.start

        # Environment properties
        self.n_states = height * width
        self.n_actions = 4

    def reset(self) -> Tuple[int, int]:
        """Reset environment to start state.

        Returns:
            Initial state coordinates (row, col)
        """
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Execute action and return transition.

        Dynamics:
            1. Compute next position via action displacement
            2. Clip to grid boundaries
            3. Check for cliff collision (severe penalty + reset)
            4. Check for goal (episode termination)
            5. Otherwise, standard step cost

        Args:
            action: Action index ∈ {0,1,2,3}

        Returns:
            Tuple of (next_state, reward, done)

        Raises:
            ValueError: If action is invalid
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}, must be in {list(self.ACTIONS.keys())}")

        # Apply action with boundary clipping
        di, dj = self.ACTIONS[action]
        new_i = int(np.clip(self.state[0] + di, 0, self.height - 1))
        new_j = int(np.clip(self.state[1] + dj, 0, self.width - 1))
        next_state = (new_i, new_j)

        # Check cliff collision
        if next_state in self.cliff:
            self.state = self.start
            return self.state, -100.0, False  # Severe penalty, episode continues

        self.state = next_state

        # Check goal reached
        if self.state == self.goal:
            return self.state, 0.0, True  # Episode terminates

        # Standard step
        return self.state, -1.0, False

    def render(self, path: Optional[List[Tuple[int, int]]] = None) -> str:
        """Render current environment state as ASCII art.

        Args:
            path: Optional trajectory to visualize with '*' markers

        Returns:
            String representation of gridworld
        """
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]

        # Mark cliff cells
        for pos in self.cliff:
            grid[pos[0]][pos[1]] = 'C'

        # Mark start and goal
        grid[self.start[0]][self.start[1]] = 'S'
        grid[self.goal[0]][self.goal[1]] = 'G'

        # Mark trajectory
        if path:
            for pos in path[1:-1]:  # Exclude start and goal
                if pos not in self.cliff and pos != self.start and pos != self.goal:
                    grid[pos[0]][pos[1]] = '*'

        # Mark current agent position
        if self.state != self.start and self.state != self.goal:
            if self.state not in self.cliff:
                grid[self.state[0]][self.state[1]] = '@'

        # Build bordered output
        border_h = "┌" + "─" * (self.width * 2 + 1) + "┐"
        border_b = "└" + "─" * (self.width * 2 + 1) + "┘"

        lines = [border_h]
        for row in grid:
            lines.append("│ " + " ".join(row) + " │")
        lines.append(border_b)

        output = "\n".join(lines)
        print(output)
        return output

    def get_optimal_path(self) -> List[Tuple[int, int]]:
        """Return optimal path (risky route along cliff).

        Returns:
            Sequence of states from start to goal via shortest path
        """
        path = []
        for j in range(self.width):
            path.append((self.height - 1, j))
        return path

    def get_safe_path(self) -> List[Tuple[int, int]]:
        """Return conservative path avoiding cliff.

        Returns:
            Sequence of states taking detour far from cliff
        """
        path = [self.start]

        # Move up to top row
        for i in range(self.height - 2, -1, -1):
            path.append((i, 0))

        # Move right along top row
        for j in range(1, self.width):
            path.append((0, j))

        # Move down to goal
        for i in range(1, self.height):
            path.append((i, self.width - 1))

        return path


def train_q_learning(
    env,
    agent: QLearningAgent,
    episodes: int = 500,
    max_steps: int = 200,
    verbose: bool = True,
    log_interval: int = 100
) -> TrainingMetrics:
    """Train Q-Learning agent on given environment.

    Implements standard off-policy training loop with epsilon decay.

    Args:
        env: Environment with reset() and step(action) methods
        agent: Q-Learning agent instance
        episodes: Number of training episodes
        max_steps: Maximum timesteps per episode (prevents infinite loops)
        verbose: If True, prints progress logs
        log_interval: Episodes between log outputs

    Returns:
        TrainingMetrics object with episode rewards, lengths, epsilon history

    Complexity:
        Time: O(episodes × max_steps × |A|) for max operations
        Space: O(|S| × |A|) for Q-table
    """
    metrics = TrainingMetrics()

    for episode in range(episodes):
        # Reset environment (handle both gym and custom interfaces)
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result

        total_reward = 0.0
        steps = 0

        for step in range(max_steps):
            # Select action using current policy
            action = agent.get_action(state, training=True)

            # Execute action in environment
            result = env.step(action)
            if len(result) == 3:
                next_state, reward, done = result
            else:
                # Gymnasium interface: (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            # Q-Learning update
            td_error = agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Decay exploration rate
        agent.decay_epsilon()

        # Record metrics
        metrics.episode_rewards.append(total_reward)
        metrics.episode_lengths.append(steps)
        metrics.epsilon_history.append(agent.epsilon)

        # Logging
        if verbose and (episode + 1) % log_interval == 0:
            avg_reward = np.mean(metrics.episode_rewards[-log_interval:])
            avg_steps = np.mean(metrics.episode_lengths[-log_interval:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Steps: {avg_steps:6.1f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    agent.metrics = metrics
    return metrics


def train_sarsa(
    env,
    agent: SARSAAgent,
    episodes: int = 500,
    max_steps: int = 200,
    verbose: bool = True,
    log_interval: int = 100
) -> TrainingMetrics:
    """Train SARSA agent on given environment.

    Implements on-policy training loop with action-before-update pattern.
    Critical: Must sample next_action before update (SARSA requirement).

    Args:
        env: Environment instance
        agent: SARSA agent
        episodes: Number of training episodes
        max_steps: Maximum timesteps per episode
        verbose: If True, prints progress
        log_interval: Episodes between logs

    Returns:
        TrainingMetrics with episode statistics

    Complexity:
        Time: O(episodes × max_steps) per update
        Space: O(|S| × |A|) for Q-table
    """
    metrics = TrainingMetrics()

    for episode in range(episodes):
        # Reset environment
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result

        # SARSA: Sample initial action before loop
        action = agent.get_action(state, training=True)

        total_reward = 0.0
        steps = 0

        for step in range(max_steps):
            # Execute action
            result = env.step(action)
            if len(result) == 3:
                next_state, reward, done = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            # SARSA: Sample next action before update
            next_action = agent.get_action(next_state, training=True)

            # SARSA update using actual next action
            td_error = agent.update(
                state, action, reward, next_state, next_action, done
            )

            # Transition to next step
            state = next_state
            action = next_action  # Critical: carry action forward
            total_reward += reward
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        metrics.episode_rewards.append(total_reward)
        metrics.episode_lengths.append(steps)
        metrics.epsilon_history.append(agent.epsilon)

        if verbose and (episode + 1) % log_interval == 0:
            avg_reward = np.mean(metrics.episode_rewards[-log_interval:])
            avg_steps = np.mean(metrics.episode_lengths[-log_interval:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Steps: {avg_steps:6.1f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    agent.metrics = metrics
    return metrics


def extract_path(
    agent: BaseAgent,
    env: CliffWalkingEnv,
    max_steps: int = 50
) -> List[Tuple[int, int]]:
    """Extract greedy policy trajectory from trained agent.

    Args:
        agent: Trained agent with Q-table
        env: Environment instance
        max_steps: Maximum path length (prevents infinite loops)

    Returns:
        Sequence of states following greedy policy π(s) = argmax_a Q(s,a)
    """
    state = env.reset()
    path = [state]

    for _ in range(max_steps):
        action = agent.get_action(state, training=False)  # Greedy
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
        if done:
            break

    return path


def plot_learning_curves(
    metrics_dict: Dict[str, TrainingMetrics],
    window: int = 10,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """Plot comparative learning curves for multiple algorithms.

    Args:
        metrics_dict: Mapping from algorithm name to training metrics
        window: Moving average window for smoothing
        figsize: Figure dimensions (width, height)
        save_path: If provided, saves figure to this path

    Requires: matplotlib
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Reward curve
    ax1 = axes[0]
    for name, metrics in metrics_dict.items():
        if len(metrics.episode_rewards) >= window:
            smoothed = np.convolve(
                metrics.episode_rewards,
                np.ones(window) / window,
                mode='valid'
            )
            ax1.plot(smoothed, label=name, alpha=0.8, linewidth=2)

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Learning Curve: Episode Rewards', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Steps curve
    ax2 = axes[1]
    for name, metrics in metrics_dict.items():
        if len(metrics.episode_lengths) >= window:
            smoothed = np.convolve(
                metrics.episode_lengths,
                np.ones(window) / window,
                mode='valid'
            )
            ax2.plot(smoothed, label=name, alpha=0.8, linewidth=2)

    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Steps', fontsize=12)
    ax2.set_title('Learning Curve: Episode Length', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_q_table(
    agent: BaseAgent,
    env: CliffWalkingEnv,
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None
) -> None:
    """Visualize Q-table value function and policy.

    Args:
        agent: Trained agent
        env: Cliff walking environment
        figsize: Figure dimensions
        save_path: Optional file path for saving

    Requires: matplotlib
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Prepare value function grid
    v_table = np.zeros((env.height, env.width), dtype=np.float32)
    policy_arrows = np.zeros((env.height, env.width), dtype=int)

    for i in range(env.height):
        for j in range(env.width):
            state = (i, j)
            if state in agent.q_table:
                v_table[i, j] = np.max(agent.q_table[state])
                policy_arrows[i, j] = int(np.argmax(agent.q_table[state]))

    # Value function heatmap
    ax1 = axes[0]
    im = ax1.imshow(v_table, cmap='RdYlGn', aspect='auto')
    ax1.set_title('Value Function V(s) = max_a Q(s,a)', fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    plt.colorbar(im, ax=ax1)

    # Mark cliff
    for pos in env.cliff:
        ax1.add_patch(plt.Rectangle(
            (pos[1]-0.5, pos[0]-0.5), 1, 1,
            fill=True, color='black', alpha=0.5
        ))

    # Policy arrows
    ax2 = axes[1]
    arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    for i in range(env.height):
        for j in range(env.width):
            if (i, j) in env.cliff:
                ax2.text(j, i, 'X', ha='center', va='center', fontsize=12, color='red')
            elif (i, j) == env.goal:
                ax2.text(j, i, 'G', ha='center', va='center', fontsize=12,
                        color='green', fontweight='bold')
            elif (i, j) == env.start:
                ax2.text(j, i, 'S', ha='center', va='center', fontsize=12,
                        color='blue', fontweight='bold')
            else:
                ax2.text(j, i, arrow_map[policy_arrows[i, j]],
                        ha='center', va='center', fontsize=14)

    ax2.set_xlim(-0.5, env.width - 0.5)
    ax2.set_ylim(env.height - 0.5, -0.5)
    ax2.set_title('Greedy Policy π(s)', fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.grid(True, alpha=0.3)

    # Q-value distribution
    ax3 = axes[2]
    q_max_values = [
        np.max(q) for q in agent.q_table.values() if np.any(q != 0)
    ]
    if q_max_values:
        ax3.hist(q_max_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.set_xlabel('Max Q-Value', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Q-Value Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def compare_cliff_walking(
    episodes: int = 500,
    learning_rate: float = 0.5,
    epsilon: float = 0.1,
    show_plots: bool = True
) -> Tuple[QLearningAgent, SARSAAgent]:
    """Compare Q-Learning and SARSA on Cliff Walking environment.

    Demonstrates fundamental behavioral difference between off-policy and
    on-policy learning in presence of exploration.

    Args:
        episodes: Training episodes
        learning_rate: TD step size α
        epsilon: Fixed exploration rate (no decay for clearer comparison)
        show_plots: Whether to display matplotlib plots

    Returns:
        Tuple of (trained_q_agent, trained_sarsa_agent)
    """
    print("=" * 60)
    print("Cliff Walking: Q-Learning vs SARSA Comparison")
    print("=" * 60)

    env = CliffWalkingEnv()

    # Create agents with fixed epsilon for clear behavioral comparison
    q_agent = QLearningAgent(
        n_actions=4,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_decay=1.0,  # No decay
        epsilon_min=epsilon
    )

    sarsa_agent = SARSAAgent(
        n_actions=4,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_decay=1.0,
        epsilon_min=epsilon
    )

    # Train both agents
    print("\nTraining Q-Learning...")
    q_metrics = train_q_learning(
        env, q_agent, episodes=episodes, verbose=True, log_interval=100
    )

    print("\nTraining SARSA...")
    sarsa_metrics = train_sarsa(
        env, sarsa_agent, episodes=episodes, verbose=True, log_interval=100
    )

    # Display learned policies
    print("\n" + "=" * 60)
    print("Learned Policies")
    print("=" * 60)

    print("\nQ-Learning (learns optimal risky path along cliff):")
    q_path = extract_path(q_agent, env)
    env.render(q_path)
    print(f"Path length: {len(q_path) - 1} steps")

    env.reset()
    print("\nSARSA (learns safe conservative path away from cliff):")
    sarsa_path = extract_path(sarsa_agent, env)
    env.render(sarsa_path)
    print(f"Path length: {len(sarsa_path) - 1} steps")

    # Statistics
    print("\n" + "=" * 60)
    print("Training Statistics")
    print("=" * 60)
    q_final = np.mean(q_metrics.episode_rewards[-100:])
    sarsa_final = np.mean(sarsa_metrics.episode_rewards[-100:])
    print(f"Q-Learning final 100-episode avg reward: {q_final:.2f}")
    print(f"SARSA final 100-episode avg reward:      {sarsa_final:.2f}")
    print(f"\nDifference: Q-Learning is {q_final - sarsa_final:.2f} better asymptotically")
    print("(Expected: Q-Learning learns riskier optimal path)")

    if show_plots:
        plot_learning_curves(
            {'Q-Learning': q_metrics, 'SARSA': sarsa_metrics},
            window=10,
            save_path='cliff_walking_comparison.png'
        )

    return q_agent, sarsa_agent


def train_taxi(
    episodes: int = 2000,
    show_plots: bool = True
) -> Optional[QLearningAgent]:
    """Train Q-Learning on Taxi-v3 environment.

    Args:
        episodes: Number of training episodes
        show_plots: Whether to display training curves

    Returns:
        Trained agent, or None if gymnasium not available

    Requires: gymnasium package
    """
    try:
        import gymnasium as gym
    except ImportError:
        print("Gymnasium not installed: pip install gymnasium")
        return None

    print("\n" + "=" * 60)
    print("Taxi-v3 Q-Learning Training")
    print("=" * 60)

    env = gym.make('Taxi-v3')

    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    metrics = train_q_learning(
        env, agent, episodes=episodes, verbose=True, log_interval=200
    )

    env.close()

    print(f"\nFinal 100-episode average reward: {np.mean(metrics.episode_rewards[-100:]):.2f}")

    if show_plots:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            window = 50

            # Reward curve
            smoothed_rewards = np.convolve(
                metrics.episode_rewards,
                np.ones(window) / window,
                mode='valid'
            )
            axes[0].plot(smoothed_rewards, color='steelblue', linewidth=2)
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Total Reward')
            axes[0].set_title('Taxi-v3: Reward per Episode')
            axes[0].grid(True, alpha=0.3)

            # Steps curve
            smoothed_steps = np.convolve(
                metrics.episode_lengths,
                np.ones(window) / window,
                mode='valid'
            )
            axes[1].plot(smoothed_steps, color='coral', linewidth=2)
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Steps')
            axes[1].set_title('Taxi-v3: Steps per Episode')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('taxi_training.png', dpi=150)
            plt.show()
        except ImportError:
            pass

    return agent


def main() -> None:
    """Command-line interface for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Tabular Q-Learning and SARSA Experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--exp', type=str, default='cliff',
        choices=['cliff', 'taxi', 'all'],
        help='Experiment type'
    )
    parser.add_argument(
        '--episodes', type=int, default=500,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Disable matplotlib plots'
    )

    args = parser.parse_args()

    if args.exp in ['cliff', 'all']:
        compare_cliff_walking(
            episodes=args.episodes,
            show_plots=not args.no_plot
        )

    if args.exp in ['taxi', 'all']:
        train_taxi(
            episodes=args.episodes * 4 if args.exp == 'all' else args.episodes,
            show_plots=not args.no_plot
        )


if __name__ == "__main__":
    # Unit tests
    print("Running unit tests...")
    print("-" * 60)

    # Test 1: Configuration validation
    try:
        config = AgentConfig(n_actions=4)
        assert config.n_actions == 4
        print("✓ Test 1 passed: Configuration validation")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")

    # Test 2: Q-Learning update
    try:
        agent = QLearningAgent(n_actions=4, learning_rate=0.5, discount_factor=0.9)
        state, next_state = (0, 0), (0, 1)
        agent.q_table[next_state] = np.array([1.0, 2.0, 0.5, 0.3])

        td_error = agent.update(state, 0, -1.0, next_state, False)
        expected_q = 0.5 * (-1.0 + 0.9 * 2.0)  # α[r + γ max Q']

        assert np.isclose(agent.q_table[state][0], expected_q, atol=1e-6)
        print("✓ Test 2 passed: Q-Learning update correctness")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")

    # Test 3: SARSA update
    try:
        agent = SARSAAgent(n_actions=4, learning_rate=0.5, discount_factor=0.9)
        state, next_state = (0, 0), (0, 1)
        agent.q_table[next_state] = np.array([1.0, 2.0, 0.5, 0.3])

        td_error = agent.update(state, 0, -1.0, next_state, 1, False)  # next_action=1
        expected_q = 0.5 * (-1.0 + 0.9 * 2.0)  # α[r + γ Q(s',a')]

        assert np.isclose(agent.q_table[state][0], expected_q, atol=1e-6)
        print("✓ Test 3 passed: SARSA update correctness")
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")

    # Test 4: Environment dynamics
    try:
        env = CliffWalkingEnv()
        state = env.reset()
        assert state == (3, 0), "Start position incorrect"

        # Move right into cliff
        next_state, reward, done = env.step(1)
        assert reward == -100.0, "Cliff penalty incorrect"
        assert next_state == env.start, "Should reset to start after cliff"
        print("✓ Test 4 passed: Environment dynamics")
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")

    # Test 5: Exploration strategies
    try:
        agent = QLearningAgent(n_actions=4, epsilon=0.0, epsilon_min=0.0)
        state = (0, 0)
        agent.q_table[state] = np.array([1.0, 3.0, 2.0, 0.5])

        # With ε=0, should always pick action 1 (max Q)
        actions = [agent.get_action(state, training=True) for _ in range(20)]
        assert all(a == 1 for a in actions), "Greedy policy violated"
        print("✓ Test 5 passed: Exploration strategy")
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")

    # Test 6: Double Q-Learning
    try:
        agent = QLearningAgent(n_actions=4, double_q=True)
        state = (0, 0)

        for _ in range(10):
            agent.update(state, 0, -1.0, (0, 1), False)

        # Verify both Q-tables are updated
        updated = (np.any(agent.q_table[state] != 0) or
                  np.any(agent.q_table2[state] != 0))
        assert updated, "Double Q-tables not updated"
        print("✓ Test 6 passed: Double Q-Learning")
    except Exception as e:
        print(f"✗ Test 6 failed: {e}")

    print("-" * 60)
    print("Unit tests completed\n")

    # Run main experiments
    main()
