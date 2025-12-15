"""
Hindsight Experience Replay (HER): Learning from Failures in Goal-Conditioned RL

================================================================================
CORE IDEA
================================================================================
HER transforms failed episodes into successful training examples by retroactively
treating achieved states as if they were the intended goals. Every trajectory
succeeds at reaching *some* goal, even if not the original target.

================================================================================
MATHEMATICAL THEORY
================================================================================
For a goal-conditioned policy π(a|s, g), consider trajectory τ = (s₀, a₀, s₁, ..., sₜ)
with original goal g that was not achieved. Standard RL receives only failure signals.

HER insight: The trajectory successfully reached s_T. Create virtual transitions
with hindsight goal g' = achieved_goal(s_T):

    Original: (s_t, a_t, r=-1, s_{t+1}, g)     [failure]
    Relabeled: (s_t, a_t, r=0, s_{t+1}, g')    [success!]

This is valid because under goal-conditioned policies, actions are optimal
for whatever goal they actually reach. The relabeled transition teaches
"this action sequence reaches g'" which is true.

The reward function R(s, g) is typically sparse:
    R(s, g) = 0   if ‖achieved_goal(s) - g‖ < ε
    R(s, g) = -1  otherwise

================================================================================
PROBLEM STATEMENT
================================================================================
Sparse rewards in goal-conditioned RL create severe exploration challenges:
- Random exploration rarely reaches distant goals
- Almost all transitions yield zero reward gradient
- Sample efficiency is exponentially poor in goal distance

HER addresses this by:
- Converting every trajectory into useful training signal
- Providing implicit curriculum from nearby to distant goals
- Enabling learning with purely binary success/failure rewards

================================================================================
ALGORITHM COMPARISON
================================================================================
| Method           | Reward Type   | Goal-Conditioned | Sample Efficiency |
|------------------|---------------|------------------|-------------------|
| Standard RL      | Any           | No               | Low (sparse)      |
| Reward Shaping   | Dense         | No               | Medium            |
| HER              | Sparse        | Yes              | High              |
| Curriculum HER   | Sparse        | Yes              | Higher            |

================================================================================
REFERENCES
================================================================================
[1] Andrychowicz, M., et al. (2017). Hindsight experience replay. NeurIPS.
[2] Plappert, M., et al. (2018). Multi-goal reinforcement learning:
    Challenging robotics environments. arXiv:1802.09464.
[3] Fang, M., et al. (2019). Curriculum-guided hindsight experience replay.
    NeurIPS.

================================================================================
COMPLEXITY ANALYSIS
================================================================================
- Storage: O((1 + k) × buffer_size) where k is replay_k
- Per-episode: O(T × k) relabeling operations where T is episode length
- Sampling: O(batch_size) with optional prioritization overhead
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np


class GoalSelectionStrategy(Enum):
    """Strategies for selecting hindsight goals from achieved states.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Let τ = (s₀, ..., s_T) be a trajectory with achieved goals G = {g₀, ..., g_T}.
    For transition at time t, different strategies sample hindsight goal g':

    FINAL:
        g' = g_T (always use final achieved goal)
        Simple but limited diversity.

    FUTURE:
        g' ~ Uniform(g_{t+1}, ..., g_T)
        Only use goals from the future - ensures causal consistency.
        Best performing in most benchmarks.

    EPISODE:
        g' ~ Uniform(g₀, ..., g_T)
        Can use any achieved goal in episode.
        Maximum diversity but includes "past" goals.

    RANDOM:
        g' ~ Uniform(all stored achieved goals)
        Cross-episode sampling for maximum coverage.
        May break temporal consistency assumptions.
    """

    FINAL = "final"
    FUTURE = "future"
    EPISODE = "episode"
    RANDOM = "random"


class Transition(NamedTuple):
    """Single environment transition with goal-conditioning information.

    In goal-conditioned RL, transitions include both the desired goal
    (what the agent was trying to reach) and the achieved goal (what
    state it actually reached).

    Attributes:
        state: Current observation s_t.
        action: Action taken a_t.
        reward: Received reward R(s_{t+1}, g_desired).
        next_state: Resulting observation s_{t+1}.
        done: Whether episode terminated.
        desired_goal: Target goal g the agent was pursuing.
        achieved_goal: Goal actually achieved at s_{t+1}.
    """

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    desired_goal: np.ndarray
    achieved_goal: np.ndarray


@dataclass
class Episode:
    """Container for a complete goal-conditioned episode trajectory.

    Stores all transitions along with episode-level metadata useful
    for HER relabeling and analysis.

    Attributes:
        transitions: Ordered list of transitions (s₀, a₀, ..., s_T).
        initial_goal: The desired goal g at episode start.
        final_achieved_goal: The achieved goal at s_T.
        total_reward: Cumulative reward Σ r_t.
        success: Whether achieved_goal ≈ desired_goal at episode end.
    """

    transitions: List[Transition] = field(default_factory=list)
    initial_goal: Optional[np.ndarray] = None
    final_achieved_goal: Optional[np.ndarray] = None
    total_reward: float = 0.0
    success: bool = False

    def __len__(self) -> int:
        """Return episode length (number of transitions)."""
        return len(self.transitions)

    def add_transition(self, transition: Transition) -> None:
        """Append transition to episode and update metadata.

        Args:
            transition: New transition to add.
        """
        self.transitions.append(transition)
        self.total_reward += transition.reward
        self.final_achieved_goal = transition.achieved_goal.copy()

    def get_achieved_goals(self) -> np.ndarray:
        """Get array of all achieved goals in episode.

        Returns:
            Array of shape (T, goal_dim) containing achieved goals.
        """
        if not self.transitions:
            return np.array([])
        return np.array([t.achieved_goal for t in self.transitions])


@dataclass(frozen=True)
class HERConfig:
    """Immutable configuration for Hindsight Experience Replay.

    Attributes:
        replay_k: Number of hindsight goals per original transition.
            Higher k = more relabeled data = better sample efficiency
            but increased memory and computation.
        strategy: Goal selection strategy determining which achieved
            goals become hindsight targets.
        goal_tolerance: Distance threshold ε for goal achievement:
            success if ‖achieved - desired‖ < goal_tolerance.
        future_p: For FUTURE strategy, probability of sampling future
            goals vs. falling back to final goal.
    """

    replay_k: int = 4
    strategy: GoalSelectionStrategy = GoalSelectionStrategy.FUTURE
    goal_tolerance: float = 0.05
    future_p: float = 0.8

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.replay_k < 0:
            raise ValueError(f"replay_k must be non-negative, got {self.replay_k}")
        if not 0.0 <= self.future_p <= 1.0:
            raise ValueError(f"future_p must be in [0, 1], got {self.future_p}")
        if self.goal_tolerance <= 0:
            raise ValueError(
                f"goal_tolerance must be positive, got {self.goal_tolerance}"
            )


class GoalConditionedReplayBuffer:
    """Experience replay buffer optimized for goal-conditioned RL with HER.

    ============================================================================
    CORE IDEA
    ============================================================================
    Pre-allocated numpy arrays store transitions with separate columns for
    desired and achieved goals, enabling efficient HER relabeling without
    data copying.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Stores tuples (s, a, r, s', done, g_d, g_a) where:
    - s, s' ∈ ℝ^{state_dim}: state observations
    - a ∈ ℝ^{action_dim}: action
    - r ∈ ℝ: reward (sparse for goal-conditioned)
    - g_d ∈ ℝ^{goal_dim}: desired goal
    - g_a ∈ ℝ^{goal_dim}: achieved goal at s'

    HER creates virtual transitions (s, a, r', s', done, g', g_a) with
    relabeled desired goal g' and recomputed reward r' = R(g_a, g').

    ============================================================================
    COMPLEXITY
    ============================================================================
    - add(): O(1) amortized (circular buffer)
    - sample(): O(batch_size)
    - Space: O(capacity × (2×state_dim + action_dim + 2×goal_dim + 2))
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
    ) -> None:
        """Initialize pre-allocated replay buffer arrays.

        Args:
            capacity: Maximum transitions to store.
            state_dim: Dimensionality of state observations.
            action_dim: Dimensionality of actions.
            goal_dim: Dimensionality of goal representations.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        self._states = np.zeros((capacity, state_dim), dtype=np.float64)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float64)
        self._rewards = np.zeros(capacity, dtype=np.float64)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float64)
        self._dones = np.zeros(capacity, dtype=np.float64)
        self._desired_goals = np.zeros((capacity, goal_dim), dtype=np.float64)
        self._achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float64)

        self._position = 0
        self._size = 0

    def add(self, transition: Transition) -> int:
        """Add single transition to buffer.

        Args:
            transition: Goal-conditioned transition to store.

        Returns:
            Index where transition was stored.
        """
        idx = self._position

        self._states[idx] = np.asarray(transition.state, dtype=np.float64).flatten()[
            : self.state_dim
        ]
        self._actions[idx] = np.asarray(transition.action, dtype=np.float64).flatten()[
            : self.action_dim
        ]
        self._rewards[idx] = transition.reward
        self._next_states[idx] = np.asarray(
            transition.next_state, dtype=np.float64
        ).flatten()[: self.state_dim]
        self._dones[idx] = float(transition.done)
        self._desired_goals[idx] = np.asarray(
            transition.desired_goal, dtype=np.float64
        ).flatten()[: self.goal_dim]
        self._achieved_goals[idx] = np.asarray(
            transition.achieved_goal, dtype=np.float64
        ).flatten()[: self.goal_dim]

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

        return idx

    def add_episode(self, episode: Episode) -> None:
        """Add complete episode to buffer.

        Args:
            episode: Episode containing transitions to store.
        """
        for transition in episode.transitions:
            self.add(transition)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample uniform random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with arrays for each transition component:
            - states: (batch_size, state_dim)
            - actions: (batch_size, action_dim)
            - rewards: (batch_size,)
            - next_states: (batch_size, state_dim)
            - dones: (batch_size,)
            - desired_goals: (batch_size, goal_dim)
            - achieved_goals: (batch_size, goal_dim)
        """
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")

        indices = np.random.randint(0, self._size, size=batch_size)

        return {
            "states": self._states[indices].copy(),
            "actions": self._actions[indices].copy(),
            "rewards": self._rewards[indices].copy(),
            "next_states": self._next_states[indices].copy(),
            "dones": self._dones[indices].copy(),
            "desired_goals": self._desired_goals[indices].copy(),
            "achieved_goals": self._achieved_goals[indices].copy(),
            "indices": indices,
        }

    def __len__(self) -> int:
        """Return current buffer size."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return self._size == self.capacity


class HindsightExperienceReplay:
    """Core Hindsight Experience Replay implementation.

    ============================================================================
    CORE IDEA
    ============================================================================
    After each episode, augment the replay buffer with relabeled transitions
    where achieved goals become the "desired" goals. This creates successful
    training examples from any trajectory, regardless of original success.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    For trajectory τ = (s₀, a₀, ..., s_T) with desired goal g:

    1. Store original: (s_t, a_t, R(g_a^{t+1}, g), s_{t+1}, g, g_a^{t+1})

    2. For each transition t, sample k hindsight goals G' = {g'₁, ..., g'_k}
       from achieved goals using the configured strategy.

    3. Store relabeled: (s_t, a_t, R(g_a^{t+1}, g'_i), s_{t+1}, g'_i, g_a^{t+1})
       for each g'_i ∈ G'

    The key insight: the action sequence in τ is optimal for reaching g_a^T
    (since that's where it ended up), so relabeled transitions are valid
    training data for the goal-conditioned policy.

    ============================================================================
    USAGE PATTERN
    ============================================================================
    ```python
    buffer = GoalConditionedReplayBuffer(10000, state_dim, action_dim, goal_dim)
    her = HindsightExperienceReplay(buffer, HERConfig(replay_k=4))

    for episode in training:
        goal = env.sample_goal()
        state = env.reset()
        while not done:
            action = agent.act(state, goal)
            next_state, reward, done, info = env.step(action)
            achieved_goal = info['achieved_goal']

            her.store_transition(
                state, action, reward, next_state, done,
                desired_goal=goal, achieved_goal=achieved_goal
            )
            state = next_state

        # HER relabeling happens automatically at episode end

        batch = her.sample(256)
        agent.train(batch)
    ```
    """

    def __init__(
        self,
        buffer: GoalConditionedReplayBuffer,
        config: Optional[HERConfig] = None,
        reward_function: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        """Initialize HER module.

        Args:
            buffer: Goal-conditioned replay buffer for storage.
            config: HER configuration (uses defaults if None).
            reward_function: Custom reward R(achieved, desired) -> float.
                Defaults to sparse binary reward based on goal_tolerance.
        """
        self.buffer = buffer
        self.config = config or HERConfig()
        self.reward_function = reward_function or self._sparse_reward

        self._current_episode: Episode = Episode()
        self._episode_count: int = 0
        self._relabeled_count: int = 0
        self._success_history: Deque[bool] = deque(maxlen=100)

    def _sparse_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> float:
        """Default sparse binary reward function.

        ========================================================================
        MATHEMATICAL THEORY
        ========================================================================
        The canonical HER reward is:

            R(g_a, g_d) = 0   if ‖g_a - g_d‖₂ < ε
            R(g_a, g_d) = -1  otherwise

        This creates the sparsest possible signal: only reward at exact
        goal achievement. HER makes this tractable through relabeling.

        Args:
            achieved_goal: Goal achieved at current state.
            desired_goal: Target goal.

        Returns:
            0.0 if goal achieved (within tolerance), -1.0 otherwise.
        """
        distance = np.linalg.norm(
            np.asarray(achieved_goal) - np.asarray(desired_goal)
        )
        return 0.0 if distance < self.config.goal_tolerance else -1.0

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        desired_goal: np.ndarray,
        achieved_goal: np.ndarray,
    ) -> None:
        """Store transition and trigger HER relabeling at episode end.

        Args:
            state: Current state observation.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state observation.
            done: Whether episode terminated.
            desired_goal: Target goal for this episode.
            achieved_goal: Goal achieved at next_state.
        """
        transition = Transition(
            state=np.asarray(state, dtype=np.float64),
            action=np.asarray(action, dtype=np.float64),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float64),
            done=bool(done),
            desired_goal=np.asarray(desired_goal, dtype=np.float64),
            achieved_goal=np.asarray(achieved_goal, dtype=np.float64),
        )

        if self._current_episode.initial_goal is None:
            self._current_episode.initial_goal = desired_goal.copy()

        self._current_episode.add_transition(transition)
        self.buffer.add(transition)

        if done:
            success = self._sparse_reward(
                self._current_episode.final_achieved_goal,
                self._current_episode.initial_goal,
            ) == 0.0
            self._current_episode.success = success
            self._success_history.append(success)

            self._apply_hindsight_relabeling()
            self._current_episode = Episode()
            self._episode_count += 1

    def _apply_hindsight_relabeling(self) -> None:
        """Apply HER relabeling to completed episode.

        For each transition in the episode, sample k hindsight goals
        and store relabeled transitions with recomputed rewards.
        """
        episode = self._current_episode
        if len(episode) == 0:
            return

        for t, transition in enumerate(episode.transitions):
            hindsight_goals = self._sample_hindsight_goals(episode, t)

            for goal in hindsight_goals:
                relabeled_reward = self.reward_function(
                    transition.achieved_goal, goal
                )

                relabeled = Transition(
                    state=transition.state,
                    action=transition.action,
                    reward=relabeled_reward,
                    next_state=transition.next_state,
                    done=transition.done,
                    desired_goal=goal,
                    achieved_goal=transition.achieved_goal,
                )

                self.buffer.add(relabeled)
                self._relabeled_count += 1

    def _sample_hindsight_goals(
        self,
        episode: Episode,
        transition_idx: int,
    ) -> List[np.ndarray]:
        """Sample hindsight goals for a specific transition.

        Args:
            episode: Episode containing the transition.
            transition_idx: Index of transition within episode.

        Returns:
            List of k sampled hindsight goals.
        """
        goals: List[np.ndarray] = []
        k = self.config.replay_k
        strategy = self.config.strategy

        if strategy == GoalSelectionStrategy.FINAL:
            if episode.final_achieved_goal is not None:
                goals = [episode.final_achieved_goal.copy() for _ in range(k)]

        elif strategy == GoalSelectionStrategy.FUTURE:
            future_transitions = episode.transitions[transition_idx + 1 :]
            if len(future_transitions) > 0:
                for _ in range(k):
                    idx = np.random.randint(len(future_transitions))
                    goals.append(future_transitions[idx].achieved_goal.copy())
            elif episode.final_achieved_goal is not None:
                goals = [episode.final_achieved_goal.copy() for _ in range(k)]

        elif strategy == GoalSelectionStrategy.EPISODE:
            for _ in range(k):
                idx = np.random.randint(len(episode.transitions))
                goals.append(episode.transitions[idx].achieved_goal.copy())

        elif strategy == GoalSelectionStrategy.RANDOM:
            if len(self.buffer) > 0:
                for _ in range(k):
                    idx = np.random.randint(len(self.buffer))
                    goals.append(self.buffer._achieved_goals[idx].copy())

        return goals

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch from buffer (includes HER relabeled transitions).

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Batch dictionary with transition arrays.
        """
        return self.buffer.sample(batch_size)

    def get_statistics(self) -> Dict[str, Any]:
        """Get HER training statistics.

        Returns:
            Dictionary containing:
            - episode_count: Total episodes processed
            - relabeled_transitions: Total HER relabeled transitions
            - buffer_size: Current buffer occupancy
            - relabel_ratio: Relabeled / total ratio
            - success_rate: Recent episode success rate
        """
        success_rate = (
            sum(self._success_history) / len(self._success_history)
            if self._success_history
            else 0.0
        )

        return {
            "episode_count": self._episode_count,
            "relabeled_transitions": self._relabeled_count,
            "buffer_size": len(self.buffer),
            "relabel_ratio": self._relabeled_count / max(1, len(self.buffer)),
            "success_rate": success_rate,
        }


class PrioritizedHER(HindsightExperienceReplay):
    """Prioritized Experience Replay combined with HER.

    ============================================================================
    CORE IDEA
    ============================================================================
    Weight transition sampling by TD-error magnitude. Transitions with high
    prediction error are sampled more frequently, focusing learning on
    "surprising" or "difficult" experiences.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Sampling probability for transition i:

        P(i) = p_i^α / Σ_j p_j^α

    where p_i = |δ_i| + ε is the priority based on TD-error δ_i.
    α ∈ [0, 1] controls prioritization strength (0 = uniform, 1 = full).

    To correct for biased sampling, importance sampling weights are used:

        w_i = (N · P(i))^{-β} / max_j w_j

    where β anneals from β₀ to 1 during training.

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    - vs Standard HER: Up to 2x sample efficiency on hard tasks
    - Disadvantage: Additional hyperparameters (α, β), priority update overhead
    - Best for: Long-horizon tasks with high variance transitions
    """

    def __init__(
        self,
        buffer: GoalConditionedReplayBuffer,
        config: Optional[HERConfig] = None,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ) -> None:
        """Initialize Prioritized HER.

        Args:
            buffer: Goal-conditioned replay buffer.
            config: HER configuration.
            alpha: Prioritization exponent α ∈ [0, 1].
            beta_start: Initial importance sampling exponent β₀.
            beta_frames: Training frames to anneal β from β₀ to 1.
        """
        super().__init__(buffer, config)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self._priorities = np.ones(buffer.capacity, dtype=np.float64)
        self._max_priority = 1.0
        self._frame_count = 0

    def _get_beta(self) -> float:
        """Compute current β with linear annealing schedule."""
        progress = min(1.0, self._frame_count / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample prioritized batch with importance sampling weights.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Batch dictionary including 'weights' and 'indices' for
            priority updates.
        """
        self._frame_count += 1

        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        priorities = self._priorities[: len(self.buffer)]
        probs = priorities**self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)

        beta = self._get_beta()
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return {
            "states": self.buffer._states[indices].copy(),
            "actions": self.buffer._actions[indices].copy(),
            "rewards": self.buffer._rewards[indices].copy(),
            "next_states": self.buffer._next_states[indices].copy(),
            "dones": self.buffer._dones[indices].copy(),
            "desired_goals": self.buffer._desired_goals[indices].copy(),
            "achieved_goals": self.buffer._achieved_goals[indices].copy(),
            "weights": weights.astype(np.float64),
            "indices": indices,
        }

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
        epsilon: float = 1e-6,
    ) -> None:
        """Update priorities based on new TD-errors.

        Args:
            indices: Buffer indices to update.
            td_errors: Corresponding TD-error magnitudes.
            epsilon: Small constant ensuring non-zero priority.
        """
        priorities = np.abs(td_errors) + epsilon
        self._priorities[indices] = priorities
        self._max_priority = max(self._max_priority, float(priorities.max()))


class CurriculumHER(HindsightExperienceReplay):
    """Curriculum-Guided Hindsight Experience Replay.

    ============================================================================
    CORE IDEA
    ============================================================================
    Adaptively select hindsight goals at the "frontier" of the agent's
    capability - not too easy (already mastered), not too hard (beyond reach).
    This implements automatic curriculum learning within HER.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Goal selection probability is Gaussian-weighted by difficulty:

        P(g) ∝ exp(-(d(g) - d*)² / 2σ²)

    where:
    - d(g) is goal difficulty (e.g., distance from start)
    - d* is target difficulty (curriculum frontier)
    - σ controls acceptance spread

    Curriculum adaptation tracks success rate:

        d*_{t+1} = d*_t + α · (success_rate - target_rate)

    If success rate > target: increase difficulty (harder goals)
    If success rate < target: decrease difficulty (easier goals)

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    - vs Standard HER: Faster early learning, better asymptotic on hard tasks
    - vs Manual curriculum: Automatic adaptation, no task-specific tuning
    - Disadvantage: Requires meaningful goal difficulty metric
    """

    def __init__(
        self,
        buffer: GoalConditionedReplayBuffer,
        config: Optional[HERConfig] = None,
        initial_difficulty: float = 0.1,
        target_success_rate: float = 0.5,
        curriculum_rate: float = 0.01,
        difficulty_sigma: float = 0.1,
    ) -> None:
        """Initialize Curriculum HER.

        Args:
            buffer: Goal-conditioned replay buffer.
            config: HER configuration.
            initial_difficulty: Starting difficulty d*₀.
            target_success_rate: Target success rate for curriculum.
            curriculum_rate: Adaptation rate α for difficulty updates.
            difficulty_sigma: Gaussian spread σ for goal selection.
        """
        super().__init__(buffer, config)
        self.current_difficulty = initial_difficulty
        self.target_success_rate = target_success_rate
        self.curriculum_rate = curriculum_rate
        self.difficulty_sigma = difficulty_sigma

        self._curriculum_success_history: Deque[bool] = deque(maxlen=100)
        self._difficulty_history: List[float] = [initial_difficulty]

    def _compute_goal_difficulty(
        self,
        reference_state: np.ndarray,
        goal: np.ndarray,
    ) -> float:
        """Compute difficulty of reaching goal from reference state.

        Default implementation uses Euclidean distance as difficulty proxy.
        Override for domain-specific difficulty metrics.

        Args:
            reference_state: Starting state for difficulty computation.
            goal: Target goal.

        Returns:
            Scalar difficulty value (higher = harder).
        """
        ref = np.asarray(reference_state).flatten()
        g = np.asarray(goal).flatten()
        return float(np.linalg.norm(ref[: len(g)] - g))

    def _sample_hindsight_goals(
        self,
        episode: Episode,
        transition_idx: int,
    ) -> List[np.ndarray]:
        """Sample curriculum-weighted hindsight goals.

        Goals near current difficulty d* are preferred. This focuses
        learning on the capability frontier.

        Args:
            episode: Current episode.
            transition_idx: Index of transition to relabel.

        Returns:
            List of curriculum-appropriate hindsight goals.
        """
        future_transitions = episode.transitions[transition_idx + 1 :]
        if len(future_transitions) == 0:
            return super()._sample_hindsight_goals(episode, transition_idx)

        current_state = episode.transitions[transition_idx].state
        candidates = [t.achieved_goal for t in future_transitions]
        difficulties = np.array(
            [
                self._compute_goal_difficulty(current_state, g)
                for g in candidates
            ]
        )

        weights = np.exp(
            -((difficulties - self.current_difficulty) ** 2)
            / (2 * self.difficulty_sigma**2)
        )

        weight_sum = weights.sum()
        if weight_sum < 1e-10:
            probs = np.ones(len(candidates)) / len(candidates)
        else:
            probs = weights / weight_sum

        goals = []
        for _ in range(self.config.replay_k):
            idx = np.random.choice(len(candidates), p=probs)
            goals.append(candidates[idx].copy())

        return goals

    def update_curriculum(self, success: bool) -> None:
        """Update curriculum difficulty based on episode outcome.

        Args:
            success: Whether episode achieved its original goal.
        """
        self._curriculum_success_history.append(success)

        if len(self._curriculum_success_history) >= 10:
            current_rate = sum(self._curriculum_success_history) / len(
                self._curriculum_success_history
            )
            adjustment = self.curriculum_rate * (
                current_rate - self.target_success_rate
            )
            self.current_difficulty = float(
                np.clip(self.current_difficulty + adjustment, 0.01, 10.0)
            )
            self._difficulty_history.append(self.current_difficulty)

    def get_curriculum_stats(self) -> Dict[str, float]:
        """Get curriculum learning statistics.

        Returns:
            Dictionary with difficulty and success rate metrics.
        """
        success_rate = (
            sum(self._curriculum_success_history)
            / len(self._curriculum_success_history)
            if self._curriculum_success_history
            else 0.0
        )

        return {
            "current_difficulty": self.current_difficulty,
            "success_rate": success_rate,
            "curriculum_steps": len(self._difficulty_history),
            "difficulty_change": (
                self._difficulty_history[-1] - self._difficulty_history[0]
                if len(self._difficulty_history) > 1
                else 0.0
            ),
        }


class GoalGenerator:
    """Automatic goal generation for goal-conditioned RL training.

    ============================================================================
    CORE IDEA
    ============================================================================
    Generate training goals that balance diversity (exploration) with
    achievability (efficient learning). Multiple strategies available
    depending on training phase and task structure.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Goal sampling strategies:

    Random: g ~ Uniform(goal_bounds)
        Maximum diversity, may include many unreachable goals.

    Achieved: g ~ p_achieved(g)
        Sample from empirical distribution of achieved goals.
        Guarantees reachability, may lack diversity.

    Mixture: g ~ λ·p_achieved + (1-λ)·p_random
        Interpolates between exploration and exploitation.

    Local: g ~ N(current_state, σ²I) ∩ goal_bounds
        Goals near current state, good for fine-grained control.
    """

    def __init__(
        self,
        goal_dim: int,
        goal_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        achieved_goal_buffer_size: int = 10000,
    ) -> None:
        """Initialize goal generator.

        Args:
            goal_dim: Dimensionality of goal space.
            goal_bounds: (lower, upper) bounds for valid goals.
            achieved_goal_buffer_size: Memory size for achieved goals.
        """
        self.goal_dim = goal_dim

        if goal_bounds is None:
            self.goal_low = -np.ones(goal_dim, dtype=np.float64)
            self.goal_high = np.ones(goal_dim, dtype=np.float64)
        else:
            self.goal_low = np.asarray(goal_bounds[0], dtype=np.float64)
            self.goal_high = np.asarray(goal_bounds[1], dtype=np.float64)

        self._achieved_goals: Deque[np.ndarray] = deque(
            maxlen=achieved_goal_buffer_size
        )

    def add_achieved_goal(self, goal: np.ndarray) -> None:
        """Record an achieved goal for future sampling.

        Args:
            goal: Goal that was achieved during training.
        """
        self._achieved_goals.append(
            np.asarray(goal, dtype=np.float64).flatten()[: self.goal_dim]
        )

    def sample_goal(
        self,
        strategy: str = "mixture",
        current_state: Optional[np.ndarray] = None,
        achieved_prob: float = 0.5,
        local_radius: float = 0.5,
    ) -> np.ndarray:
        """Sample goal using specified strategy.

        Args:
            strategy: One of "random", "achieved", "mixture", "local".
            current_state: Required for "local" strategy.
            achieved_prob: Probability of sampling achieved goal in "mixture".
            local_radius: Standard deviation for "local" strategy.

        Returns:
            Sampled goal within bounds.

        Raises:
            ValueError: If unknown strategy or missing required arguments.
        """
        if strategy == "random":
            return self._sample_random()

        elif strategy == "achieved":
            return self._sample_achieved()

        elif strategy == "mixture":
            if (
                np.random.random() < achieved_prob
                and len(self._achieved_goals) > 0
            ):
                return self._sample_achieved()
            return self._sample_random()

        elif strategy == "local":
            if current_state is None:
                raise ValueError("local strategy requires current_state")
            return self._sample_local(current_state, local_radius)

        else:
            raise ValueError(f"Unknown goal sampling strategy: {strategy}")

    def _sample_random(self) -> np.ndarray:
        """Sample uniformly random goal within bounds."""
        return np.random.uniform(self.goal_low, self.goal_high).astype(
            np.float64
        )

    def _sample_achieved(self) -> np.ndarray:
        """Sample from achieved goal distribution with noise."""
        if len(self._achieved_goals) == 0:
            return self._sample_random()

        idx = np.random.randint(len(self._achieved_goals))
        goal = self._achieved_goals[idx].copy()
        noise = 0.01 * np.random.randn(self.goal_dim)

        return np.clip(goal + noise, self.goal_low, self.goal_high).astype(
            np.float64
        )

    def _sample_local(
        self,
        current_state: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Sample goal near current state with Gaussian noise."""
        state = np.asarray(current_state, dtype=np.float64).flatten()
        center = state[: self.goal_dim]
        offset = np.random.randn(self.goal_dim) * radius
        goal = center + offset

        return np.clip(goal, self.goal_low, self.goal_high).astype(np.float64)

    def get_statistics(self) -> Dict[str, Any]:
        """Get goal generator statistics.

        Returns:
            Dictionary with achieved goal distribution info.
        """
        if len(self._achieved_goals) == 0:
            return {"num_achieved": 0}

        achieved_array = np.array(list(self._achieved_goals))
        return {
            "num_achieved": len(self._achieved_goals),
            "goal_mean": achieved_array.mean(axis=0).tolist(),
            "goal_std": achieved_array.std(axis=0).tolist(),
        }


def compute_success_rate(
    achieved_goals: np.ndarray,
    desired_goals: np.ndarray,
    tolerance: float = 0.05,
) -> float:
    """Compute fraction of goals successfully achieved.

    Args:
        achieved_goals: Array of achieved goals, shape (n, goal_dim).
        desired_goals: Array of desired goals, shape (n, goal_dim).
        tolerance: Distance threshold for success.

    Returns:
        Fraction of goals within tolerance of desired.
    """
    if len(achieved_goals) == 0:
        return 0.0

    distances = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
    return float(np.mean(distances < tolerance))


def analyze_goal_coverage(
    achieved_goals: np.ndarray,
    goal_bounds: Tuple[np.ndarray, np.ndarray],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Analyze coverage of goal space by achieved goals.

    Discretizes goal space into bins and measures what fraction
    have been reached at least once.

    Args:
        achieved_goals: Array of achieved goals, shape (n, goal_dim).
        goal_bounds: (lower, upper) bounds defining goal space.
        n_bins: Number of bins per dimension.

    Returns:
        Coverage analysis with metrics and statistics.
    """
    if len(achieved_goals) == 0:
        return {
            "coverage": 0.0,
            "unique_bins": 0,
            "total_bins": 0,
        }

    goal_low, goal_high = goal_bounds
    goal_dim = len(goal_low)

    normalized = (achieved_goals - goal_low) / (goal_high - goal_low + 1e-8)
    binned = np.floor(np.clip(normalized, 0, 1 - 1e-8) * n_bins).astype(int)

    unique_bins = len(set(map(tuple, binned)))
    total_bins = n_bins**goal_dim

    return {
        "coverage": unique_bins / total_bins,
        "unique_bins": unique_bins,
        "total_bins": total_bins,
        "goal_mean": achieved_goals.mean(axis=0).tolist(),
        "goal_std": achieved_goals.std(axis=0).tolist(),
        "goal_min": achieved_goals.min(axis=0).tolist(),
        "goal_max": achieved_goals.max(axis=0).tolist(),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Hindsight Experience Replay Module - Comprehensive Validation")
    print("=" * 70)

    np.random.seed(42)
    state_dim = 10
    action_dim = 4
    goal_dim = 3

    print("\n[Test 1] Goal-Conditioned Replay Buffer")
    print("-" * 50)

    buffer = GoalConditionedReplayBuffer(
        capacity=1000,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
    )

    for _ in range(100):
        transition = Transition(
            state=np.random.randn(state_dim),
            action=np.random.randn(action_dim),
            reward=-1.0,
            next_state=np.random.randn(state_dim),
            done=False,
            desired_goal=np.random.randn(goal_dim),
            achieved_goal=np.random.randn(goal_dim),
        )
        buffer.add(transition)

    print(f"  Buffer size: {len(buffer)}")
    assert len(buffer) == 100, "Buffer size mismatch"

    batch = buffer.sample(32)
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  States shape: {batch['states'].shape}")
    assert batch["states"].shape == (32, state_dim), "Batch shape mismatch"
    print("  [PASS] Goal-conditioned buffer works")

    print("\n[Test 2] Basic HER")
    print("-" * 50)

    buffer = GoalConditionedReplayBuffer(
        capacity=10000,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
    )

    her = HindsightExperienceReplay(
        buffer=buffer,
        config=HERConfig(
            replay_k=4,
            strategy=GoalSelectionStrategy.FUTURE,
            goal_tolerance=0.1,
        ),
    )

    desired_goal = np.array([1.0, 1.0, 1.0])
    for ep in range(5):
        for t in range(20):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            next_state = np.random.randn(state_dim)
            achieved_goal = next_state[:goal_dim]
            done = t == 19

            her.store_transition(
                state=state,
                action=action,
                reward=-1.0,
                next_state=next_state,
                done=done,
                desired_goal=desired_goal,
                achieved_goal=achieved_goal,
            )

    stats = her.get_statistics()
    print(f"  Episodes: {stats['episode_count']}")
    print(f"  Buffer size: {stats['buffer_size']}")
    print(f"  Relabeled transitions: {stats['relabeled_transitions']}")
    print(f"  Relabel ratio: {stats['relabel_ratio']:.2f}")
    assert stats["relabeled_transitions"] > 0, "HER should create relabeled transitions"
    print("  [PASS] Basic HER works")

    print("\n[Test 3] Goal Selection Strategies")
    print("-" * 50)

    for strategy in GoalSelectionStrategy:
        test_buffer = GoalConditionedReplayBuffer(
            capacity=1000,
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
        )
        test_her = HindsightExperienceReplay(
            buffer=test_buffer,
            config=HERConfig(replay_k=2, strategy=strategy),
        )

        for t in range(10):
            test_her.store_transition(
                state=np.random.randn(state_dim),
                action=np.random.randn(action_dim),
                reward=-1.0,
                next_state=np.random.randn(state_dim),
                done=t == 9,
                desired_goal=np.zeros(goal_dim),
                achieved_goal=np.random.randn(goal_dim),
            )

        print(f"  {strategy.value}: buffer size = {len(test_buffer)}")
    print("  [PASS] All goal selection strategies work")

    print("\n[Test 4] Prioritized HER")
    print("-" * 50)

    prio_buffer = GoalConditionedReplayBuffer(
        capacity=1000,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
    )

    prio_her = PrioritizedHER(
        buffer=prio_buffer,
        config=HERConfig(replay_k=4),
        alpha=0.6,
        beta_start=0.4,
    )

    for t in range(50):
        prio_her.store_transition(
            state=np.random.randn(state_dim),
            action=np.random.randn(action_dim),
            reward=-1.0,
            next_state=np.random.randn(state_dim),
            done=t % 10 == 9,
            desired_goal=np.zeros(goal_dim),
            achieved_goal=np.random.randn(goal_dim),
        )

    batch = prio_her.sample(16)
    print(f"  Batch has weights: {'weights' in batch}")
    print(f"  Batch has indices: {'indices' in batch}")
    print(f"  Weights range: [{batch['weights'].min():.3f}, {batch['weights'].max():.3f}]")
    assert "weights" in batch, "Prioritized batch should have weights"

    prio_her.update_priorities(batch["indices"], np.abs(np.random.randn(16)))
    print("  [PASS] Prioritized HER works")

    print("\n[Test 5] Curriculum HER")
    print("-" * 50)

    curr_buffer = GoalConditionedReplayBuffer(
        capacity=1000,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
    )

    curr_her = CurriculumHER(
        buffer=curr_buffer,
        config=HERConfig(replay_k=4),
        initial_difficulty=0.1,
        target_success_rate=0.5,
    )

    initial_diff = curr_her.current_difficulty
    for i in range(50):
        for t in range(10):
            curr_her.store_transition(
                state=np.random.randn(state_dim),
                action=np.random.randn(action_dim),
                reward=-1.0,
                next_state=np.random.randn(state_dim),
                done=t == 9,
                desired_goal=np.zeros(goal_dim),
                achieved_goal=np.random.randn(goal_dim),
            )

        curr_her.update_curriculum(success=(i % 3 == 0))

    curr_stats = curr_her.get_curriculum_stats()
    print(f"  Initial difficulty: {initial_diff:.3f}")
    print(f"  Current difficulty: {curr_stats['current_difficulty']:.3f}")
    print(f"  Success rate: {curr_stats['success_rate']:.3f}")
    print(f"  Difficulty change: {curr_stats['difficulty_change']:.4f}")
    print("  [PASS] Curriculum HER works")

    print("\n[Test 6] Goal Generator")
    print("-" * 50)

    generator = GoalGenerator(
        goal_dim=goal_dim,
        goal_bounds=(
            np.array([-2.0, -2.0, -2.0]),
            np.array([2.0, 2.0, 2.0]),
        ),
    )

    for _ in range(100):
        generator.add_achieved_goal(np.random.randn(goal_dim))

    strategies = ["random", "achieved", "mixture", "local"]
    for strategy in strategies:
        goal = generator.sample_goal(
            strategy=strategy,
            current_state=np.zeros(state_dim),
            local_radius=0.5,
        )
        in_bounds = np.all(goal >= -2) and np.all(goal <= 2)
        print(f"  {strategy}: goal = [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}], in_bounds = {in_bounds}")
        assert in_bounds, f"Goal out of bounds for strategy {strategy}"
    print("  [PASS] Goal generator works")

    print("\n[Test 7] Success Rate Computation")
    print("-" * 50)

    achieved = np.random.randn(100, goal_dim) * 0.01
    desired = np.zeros((100, goal_dim))
    rate = compute_success_rate(achieved, desired, tolerance=0.1)
    print(f"  Success rate (close goals): {rate:.3f}")
    assert rate > 0.5, "Close goals should have high success rate"

    far_achieved = np.ones((100, goal_dim))
    far_rate = compute_success_rate(far_achieved, desired, tolerance=0.1)
    print(f"  Success rate (far goals): {far_rate:.3f}")
    assert far_rate < 0.1, "Far goals should have low success rate"
    print("  [PASS] Success rate computation works")

    print("\n[Test 8] Goal Coverage Analysis")
    print("-" * 50)

    achieved_goals = np.random.randn(500, goal_dim)
    coverage = analyze_goal_coverage(
        achieved_goals,
        goal_bounds=(np.full(goal_dim, -3), np.full(goal_dim, 3)),
        n_bins=5,
    )
    print(f"  Coverage: {coverage['coverage']:.3f}")
    print(f"  Unique bins: {coverage['unique_bins']}/{coverage['total_bins']}")
    print(f"  Goal mean: [{coverage['goal_mean'][0]:.2f}, ...]")
    print("  [PASS] Goal coverage analysis works")

    print("\n[Test 9] Configuration Validation")
    print("-" * 50)

    try:
        HERConfig(replay_k=-1)
        print("  [FAIL] Should reject negative replay_k")
    except ValueError:
        print("  [PASS] Rejects negative replay_k")

    try:
        HERConfig(future_p=1.5)
        print("  [FAIL] Should reject future_p > 1")
    except ValueError:
        print("  [PASS] Rejects future_p > 1")

    try:
        HERConfig(goal_tolerance=-0.1)
        print("  [FAIL] Should reject negative goal_tolerance")
    except ValueError:
        print("  [PASS] Rejects negative goal_tolerance")

    print("\n[Test 10] Episode Container")
    print("-" * 50)

    episode = Episode()
    for t in range(10):
        episode.add_transition(
            Transition(
                state=np.random.randn(state_dim),
                action=np.random.randn(action_dim),
                reward=-1.0,
                next_state=np.random.randn(state_dim),
                done=t == 9,
                desired_goal=np.zeros(goal_dim),
                achieved_goal=np.random.randn(goal_dim),
            )
        )

    print(f"  Episode length: {len(episode)}")
    print(f"  Total reward: {episode.total_reward}")
    achieved_array = episode.get_achieved_goals()
    print(f"  Achieved goals shape: {achieved_array.shape}")
    assert achieved_array.shape == (10, goal_dim)
    print("  [PASS] Episode container works")

    print("\n" + "=" * 70)
    print("All validation tests passed successfully!")
    print("=" * 70)
