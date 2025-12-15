"""
Potential-Based Reward Shaping (PBRS): Accelerating RL Without Altering Optimality

================================================================================
CORE IDEA
================================================================================
Potential-Based Reward Shaping augments sparse environment rewards with dense
guidance signals derived from a potential function Φ(s). The key insight is that
shaping rewards computed as the temporal difference of potentials preserve the
optimal policy while dramatically accelerating learning.

================================================================================
MATHEMATICAL THEORY
================================================================================
Given an MDP M = (S, A, P, R, γ), the shaped MDP M' = (S, A, P, R', γ) uses:

    R'(s, a, s') = R(s, a, s') + F(s, a, s')

where F is the shaping function. The Policy Invariance Theorem (Ng et al., 1999)
states that the optimal policies of M and M' coincide if and only if:

    F(s, a, s') = γ · Φ(s') - Φ(s)

for some potential function Φ: S → ℝ.

Proof Sketch (Telescoping Sum):
    For trajectory τ = (s₀, a₀, s₁, ..., sₜ), cumulative shaping reward:

    Σₜ γᵗ F(sₜ, aₜ, sₜ₊₁) = Σₜ γᵗ [γΦ(sₜ₊₁) - Φ(sₜ)]
                          = γᵀ⁺¹Φ(sₜ₊₁) - Φ(s₀)  (telescopes)

    Since |Φ(s)| is bounded, this term vanishes as T→∞, leaving V* unchanged.

================================================================================
PROBLEM STATEMENT
================================================================================
Sparse rewards create the credit assignment problem: agents struggle to identify
which actions led to eventual success or failure. Manual reward engineering risks
introducing bias that changes the optimal policy. PBRS provides a principled way
to incorporate domain knowledge without this risk.

================================================================================
ALGORITHM COMPARISON
================================================================================
| Method            | Policy Invariance | Domain Knowledge | Computation |
|-------------------|-------------------|------------------|-------------|
| PBRS              | Guaranteed        | Medium           | O(1)        |
| Naive Shaping     | Not guaranteed    | High             | O(1)        |
| Curiosity (ICM)   | Not guaranteed    | Low              | O(network)  |
| HER               | Guaranteed        | Low              | O(k)        |

================================================================================
REFERENCES
================================================================================
[1] Ng, A.Y., Harada, D., & Russell, S. (1999). Policy invariance under reward
    transformations: Theory and application to reward shaping. ICML.
[2] Wiewiora, E. (2003). Potential-based shaping and Q-value initialization
    are equivalent. JAIR.
[3] Devlin, S., & Kudenko, D. (2012). Dynamic potential-based reward shaping.
    AAMAS.

================================================================================
COMPLEXITY ANALYSIS
================================================================================
- Time: O(1) per transition for shaping computation
- Space: O(|S|) for tabular potentials, O(|θ|) for function approximation
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")


class PotentialFunction(Protocol[StateType]):
    """Protocol defining the interface for state potential functions.

    A potential function Φ: S → ℝ assigns a scalar value to each state,
    representing its "promise" or estimated value toward achieving the goal.
    Higher potential indicates states that are more favorable for task completion.

    The choice of potential function directly impacts learning speed:
    - Ideal: Φ(s) ≈ V*(s), the optimal value function
    - Practical: Domain-specific heuristics (negative distance, progress metrics)

    Implementation Requirements:
        - Must be bounded: ∃M such that |Φ(s)| ≤ M for all s
        - Should be smooth to avoid discontinuous shaping rewards
        - Terminal states should have Φ(s_terminal) = 0 by convention
    """

    def __call__(self, state: StateType) -> float:
        """Compute potential value for the given state.

        Args:
            state: Current state observation.

        Returns:
            Scalar potential value Φ(s).
        """
        ...


@dataclass(frozen=True)
class ShapedRewardConfig:
    """Immutable configuration for potential-based reward shaping.

    Attributes:
        discount_factor: γ ∈ [0, 1] matching the MDP's discount factor.
            Critical: Must equal the learning algorithm's discount for
            policy invariance to hold.
        shaping_weight: Scaling factor λ ≥ 0 for the shaping bonus.
            F_scaled(s, s') = λ · F(s, s'). Larger values provide stronger
            guidance but may slow convergence if potential is imperfect.
        clip_range: Optional (min, max) bounds for shaped rewards.
            Prevents extreme values that could destabilize learning.
    """

    discount_factor: float = 0.99
    shaping_weight: float = 1.0
    clip_range: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError(
                f"discount_factor must be in [0, 1], got {self.discount_factor}"
            )
        if self.shaping_weight < 0:
            raise ValueError(
                f"shaping_weight must be non-negative, got {self.shaping_weight}"
            )
        if self.clip_range is not None:
            if len(self.clip_range) != 2:
                raise ValueError("clip_range must be a tuple of (min, max)")
            if self.clip_range[0] >= self.clip_range[1]:
                raise ValueError(
                    f"clip_range[0] must be < clip_range[1], got {self.clip_range}"
                )


class RewardShaper(Generic[StateType], abc.ABC):
    """Abstract base class for potential-based reward shaping implementations.

    ============================================================================
    CORE IDEA
    ============================================================================
    Transform sparse environment rewards into denser learning signals while
    mathematically guaranteeing that the optimal policy remains unchanged.
    The agent learns faster without being misled.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The shaped reward decomposes as:

        R'(s, a, s') = R(s, a, s') + F(s, a, s')
                     = R(s, a, s') + γΦ(s') - Φ(s)

    For value functions, this implies:

        V'_π(s) = V_π(s) + Φ(s)     (shifted by potential)
        Q'_π(s, a) = Q_π(s, a) + Φ(s)

    Crucially, for any two policies π₁, π₂:
        V_π₁(s) ≥ V_π₂(s) ⟺ V'_π₁(s) ≥ V'_π₂(s)

    Thus policy rankings (and hence π*) are preserved.

    ============================================================================
    USAGE PATTERN
    ============================================================================
    ```python
    shaper = DistanceBasedShaper(goal_position=goal)

    for episode in training:
        state = env.reset()
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # Apply shaping
            shaped_reward = shaper.shape_reward(
                state, action, next_state, reward, done
            )

            agent.learn(state, action, shaped_reward, next_state, done)
            state = next_state
    ```
    """

    def __init__(self, config: Optional[ShapedRewardConfig] = None) -> None:
        """Initialize the reward shaper.

        Args:
            config: Shaping configuration. Uses defaults if not provided.
        """
        self.config = config or ShapedRewardConfig()
        self._episode_shaping_sum: float = 0.0
        self._total_transitions: int = 0

    @abc.abstractmethod
    def potential(self, state: StateType) -> float:
        """Compute the potential value Φ(s) for a given state.

        This method defines the core heuristic that guides exploration.
        Subclasses implement domain-specific potential functions.

        Args:
            state: The state to evaluate.

        Returns:
            Scalar potential value. Higher values indicate more "promising"
            states that are closer to the goal or have higher expected value.
        """
        raise NotImplementedError

    def compute_shaping_bonus(
        self,
        state: StateType,
        next_state: StateType,
        done: bool = False,
    ) -> float:
        """Compute the potential-based shaping bonus F(s, a, s').

        ========================================================================
        MATHEMATICAL THEORY
        ========================================================================
            F(s, a, s') = γ · Φ(s') - Φ(s)

        When the episode terminates (done=True), we set Φ(s') = 0 following
        the convention that terminal states have zero potential. This ensures:
        1. The telescoping sum property holds
        2. No artificial bonus/penalty at episode end

        Args:
            state: Current state s.
            next_state: Next state s' (result of taking action).
            done: Whether next_state is terminal.

        Returns:
            The shaping bonus F(s, a, s'), optionally scaled and clipped.
        """
        current_potential = self.potential(state)
        next_potential = 0.0 if done else self.potential(next_state)

        bonus = (
            self.config.discount_factor * next_potential - current_potential
        ) * self.config.shaping_weight

        if self.config.clip_range is not None:
            bonus = float(
                np.clip(bonus, self.config.clip_range[0], self.config.clip_range[1])
            )

        self._episode_shaping_sum += bonus
        self._total_transitions += 1

        return bonus

    def shape_reward(
        self,
        state: StateType,
        action: ActionType,
        next_state: StateType,
        reward: float,
        done: bool = False,
    ) -> float:
        """Apply reward shaping to transform the original reward.

        This is the main entry point for integrating shaping into training.

        Args:
            state: Current state.
            action: Action taken (unused in PBRS but included for interface
                compatibility with action-dependent shaping extensions).
            next_state: Resulting state.
            reward: Original environment reward R(s, a, s').
            done: Whether episode terminated.

        Returns:
            Shaped reward R'(s, a, s') = R(s, a, s') + F(s, a, s').
        """
        bonus = self.compute_shaping_bonus(state, next_state, done)
        return reward + bonus

    def reset_episode_stats(self) -> Dict[str, float]:
        """Reset episode statistics and return summary.

        Call this at the end of each episode to track shaping behavior.

        Returns:
            Dictionary containing:
            - episode_shaping_sum: Total shaping bonus in episode
            - episode_transitions: Number of transitions
            - avg_shaping_bonus: Mean shaping bonus per step
        """
        stats = {
            "episode_shaping_sum": self._episode_shaping_sum,
            "episode_transitions": float(self._total_transitions),
            "avg_shaping_bonus": (
                self._episode_shaping_sum / max(1, self._total_transitions)
            ),
        }
        self._episode_shaping_sum = 0.0
        self._total_transitions = 0
        return stats


class DistanceBasedShaper(RewardShaper[np.ndarray]):
    """Distance-based potential function for goal-reaching tasks.

    ============================================================================
    CORE IDEA
    ============================================================================
    States closer to the goal have higher potential. The negative distance
    to the goal serves as a natural, interpretable potential function that
    provides a dense gradient toward the target.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The potential function is defined as:

        Φ(s) = -α · ‖s_pos - g‖_p

    where:
        - s_pos: Position component extracted from state s
        - g: Goal position
        - ‖·‖_p: Lp norm (p=1 for Manhattan, p=2 for Euclidean)
        - α: Scaling factor

    The resulting shaping bonus encourages movement toward the goal:

        F(s, s') = γ·(-α·‖s'_pos - g‖) - (-α·‖s_pos - g‖)
                 = α·(‖s_pos - g‖ - γ·‖s'_pos - g‖)

    If the agent moves toward the goal (reducing distance), F > 0 (reward).
    If the agent moves away (increasing distance), F < 0 (penalty).

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    - Advantages: Simple, interpretable, computationally cheap O(d)
    - Disadvantages: May create local optima in maze-like environments
    - Best for: Open navigation, continuous control, robotics

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(d) where d is state/goal dimensionality
    - Space: O(d) for storing goal position
    """

    def __init__(
        self,
        goal_position: np.ndarray,
        position_indices: Optional[np.ndarray] = None,
        norm_order: int = 2,
        scale: float = 1.0,
        config: Optional[ShapedRewardConfig] = None,
    ) -> None:
        """Initialize distance-based reward shaper.

        Args:
            goal_position: Target position coordinates.
            position_indices: Indices of position components in state vector.
                If None, the entire state is treated as position.
            norm_order: Order of Lp norm (1=Manhattan, 2=Euclidean, np.inf=Chebyshev).
            scale: Scaling factor α for potential values.
            config: Reward shaping configuration.
        """
        super().__init__(config)
        self.goal_position = np.asarray(goal_position, dtype=np.float64)
        self.position_indices = position_indices
        self.norm_order = norm_order
        self.scale = scale

    def _extract_position(self, state: np.ndarray) -> np.ndarray:
        """Extract position components from state vector."""
        state = np.asarray(state, dtype=np.float64).flatten()
        if self.position_indices is not None:
            return state[self.position_indices]
        return state[: len(self.goal_position)]

    def potential(self, state: np.ndarray) -> float:
        """Compute negative distance to goal as potential.

        Args:
            state: Current state vector.

        Returns:
            Potential value Φ(s) = -α · ‖pos - goal‖_p
        """
        position = self._extract_position(state)
        distance = np.linalg.norm(position - self.goal_position, ord=self.norm_order)
        return -self.scale * float(distance)


class SubgoalBasedShaper(RewardShaper[np.ndarray]):
    """Hierarchical potential using ordered subgoal waypoints.

    ============================================================================
    CORE IDEA
    ============================================================================
    For long-horizon tasks, define intermediate milestones (subgoals) and
    assign potential based on:
    1. Number of subgoals already achieved
    2. Progress toward the next unachieved subgoal

    This decomposes difficult navigation into manageable segments while
    maintaining the PBRS policy invariance guarantee.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Let G = {g₁, g₂, ..., gₙ} be an ordered sequence of subgoals. Define:

        Φ(s) = Σᵢ wᵢ · 𝟙[reached(s, gᵢ)] - d(s, g_next)

    where:
        - wᵢ: Reward weight for achieving subgoal i
        - 𝟙[reached(s, gᵢ)]: 1 if gᵢ achieved, 0 otherwise
        - g_next: Next unachieved subgoal
        - d(s, g_next): Distance to next subgoal

    The potential increases as the agent:
    1. Reaches new subgoals (discrete jumps in Σ wᵢ term)
    2. Approaches the next subgoal (continuous decrease in distance term)

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    vs Distance shaping: Handles non-convex paths, obstacles, maze navigation
    vs HER: Explicit subgoal guidance vs implicit goal relabeling
    Disadvantage: Requires domain knowledge to design subgoal sequence

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(n) where n is number of subgoals
    - Space: O(n × d) for storing subgoal positions
    """

    def __init__(
        self,
        subgoals: np.ndarray,
        position_indices: Optional[np.ndarray] = None,
        subgoal_radius: float = 0.5,
        subgoal_weights: Optional[np.ndarray] = None,
        config: Optional[ShapedRewardConfig] = None,
    ) -> None:
        """Initialize subgoal-based reward shaper.

        Args:
            subgoals: Array of shape (n_subgoals, dim) containing waypoints
                in the order they should be visited.
            position_indices: Indices of position in state vector.
            subgoal_radius: Distance threshold for subgoal achievement.
            subgoal_weights: Importance weights for each subgoal.
                Defaults to uniform weights of 1.0.
            config: Reward shaping configuration.
        """
        super().__init__(config)
        self.subgoals = np.asarray(subgoals, dtype=np.float64)
        self.position_indices = position_indices
        self.subgoal_radius = subgoal_radius

        if subgoal_weights is None:
            self.subgoal_weights = np.ones(len(subgoals), dtype=np.float64)
        else:
            self.subgoal_weights = np.asarray(subgoal_weights, dtype=np.float64)

        self._achieved_subgoals: set = set()

    def _extract_position(self, state: np.ndarray) -> np.ndarray:
        """Extract position components from state."""
        state = np.asarray(state, dtype=np.float64).flatten()
        if self.position_indices is not None:
            return state[self.position_indices]
        return state[: self.subgoals.shape[1]]

    def _update_achieved_subgoals(self, position: np.ndarray) -> None:
        """Check and update subgoal achievement status."""
        for i, subgoal in enumerate(self.subgoals):
            if i not in self._achieved_subgoals:
                distance = np.linalg.norm(position - subgoal)
                if distance < self.subgoal_radius:
                    self._achieved_subgoals.add(i)

    def _get_next_subgoal_index(self) -> int:
        """Get index of next unachieved subgoal."""
        for i in range(len(self.subgoals)):
            if i not in self._achieved_subgoals:
                return i
        return len(self.subgoals)

    def potential(self, state: np.ndarray) -> float:
        """Compute potential based on subgoal progress.

        Args:
            state: Current state vector.

        Returns:
            Potential combining achieved subgoal rewards and distance to next.
        """
        position = self._extract_position(state)
        self._update_achieved_subgoals(position)
        next_idx = self._get_next_subgoal_index()

        achieved_potential = sum(
            self.subgoal_weights[i] for i in self._achieved_subgoals
        )

        if next_idx < len(self.subgoals):
            distance_to_next = np.linalg.norm(position - self.subgoals[next_idx])
            progress_potential = -distance_to_next
        else:
            progress_potential = 0.0

        return float(achieved_potential + progress_potential)

    def reset(self) -> None:
        """Reset achieved subgoals for new episode."""
        self._achieved_subgoals.clear()
        self._episode_shaping_sum = 0.0
        self._total_transitions = 0


class LearnedPotentialShaper(RewardShaper[np.ndarray]):
    """Neural network-based learned potential function.

    ============================================================================
    CORE IDEA
    ============================================================================
    Instead of hand-crafting Φ(s), learn it from data. The ideal potential
    equals the optimal value function: Φ*(s) = V*(s). This can be approximated
    by training on expert demonstrations or pre-training with value estimation.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The potential network Φ_θ: S → ℝ is parameterized by weights θ.
    Training objective from demonstration data D:

        L(θ) = E_{(s,a,s')∼D} [(Φ_θ(s') - Φ_θ(s) - A*(s,a))²]

    where A*(s,a) is the advantage function. Intuitively, we want:

        γΦ_θ(s') - Φ_θ(s) ≈ Q*(s,a) - V*(s) = A*(s,a)

    This aligns the shaping bonus with the true advantage of each action.

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    vs Hand-crafted: Automatic, can capture complex value landscapes
    vs IRL: Simpler, doesn't require full reward recovery
    Disadvantage: Requires demonstrations or pre-training data

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(forward_pass) ≈ O(Σ layer_sizes)
    - Space: O(|θ|) network parameters
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        config: Optional[ShapedRewardConfig] = None,
    ) -> None:
        """Initialize learned potential network.

        Args:
            state_dim: Dimensionality of state space.
            hidden_dims: Sizes of hidden layers.
            activation: Activation function ("relu", "tanh", "sigmoid").
            config: Reward shaping configuration.
        """
        super().__init__(config)
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize network parameters with Xavier/Glorot initialization."""
        dims = [self.state_dim] + list(self.hidden_dims) + [1]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function element-wise."""
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x

    def _forward(self, state: np.ndarray) -> float:
        """Forward pass through potential network."""
        x = np.asarray(state, dtype=np.float64).flatten()

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = self._activation_fn(x)

        return float(x[0])

    def potential(self, state: np.ndarray) -> float:
        """Compute learned potential value.

        Args:
            state: Current state vector.

        Returns:
            Learned potential Φ_θ(s).
        """
        return self._forward(state)

    def update_weights(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
        """Update network parameters from external training.

        Args:
            weights: List of weight matrices matching network architecture.
            biases: List of bias vectors matching network architecture.

        Raises:
            ValueError: If shapes don't match network architecture.
        """
        if len(weights) != len(self._weights):
            raise ValueError(
                f"Expected {len(self._weights)} weight matrices, got {len(weights)}"
            )
        self._weights = [np.asarray(w, dtype=np.float64) for w in weights]
        self._biases = [np.asarray(b, dtype=np.float64) for b in biases]

    def get_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get current network parameters.

        Returns:
            Tuple of (weights, biases) lists.
        """
        return (
            [w.copy() for w in self._weights],
            [b.copy() for b in self._biases],
        )


@dataclass
class DynamicShapingConfig:
    """Configuration for adaptive/decaying reward shaping.

    Attributes:
        initial_weight: Starting shaping weight λ₀.
        decay_rate: Rate of weight decay (interpretation depends on method).
        min_weight: Minimum floor for shaping weight.
        adaptation_method: Strategy for weight adaptation:
            - "exponential": λₜ = λ₀ · αᵗ
            - "linear": λₜ = λ₀ · (1 - t/T)
    """

    initial_weight: float = 1.0
    decay_rate: float = 0.995
    min_weight: float = 0.01
    adaptation_method: str = "exponential"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.initial_weight < 0:
            raise ValueError(
                f"initial_weight must be non-negative, got {self.initial_weight}"
            )
        if not 0.0 < self.decay_rate <= 1.0:
            raise ValueError(
                f"decay_rate must be in (0, 1], got {self.decay_rate}"
            )
        if self.min_weight < 0:
            raise ValueError(f"min_weight must be non-negative, got {self.min_weight}")
        if self.adaptation_method not in ("exponential", "linear"):
            raise ValueError(
                f"Unknown adaptation_method: {self.adaptation_method}"
            )


class AdaptiveRewardShaper(RewardShaper[np.ndarray]):
    """Adaptive reward shaping with decaying influence over training.

    ============================================================================
    CORE IDEA
    ============================================================================
    Start with strong shaping guidance to bootstrap learning, then gradually
    reduce the shaping weight so the agent eventually optimizes only the true
    reward. This provides "training wheels" that are progressively removed.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The adaptive shaping weight follows a decay schedule:

        Exponential: λₜ = max(λ_min, λ₀ · αᵗ)
        Linear:      λₜ = max(λ_min, λ₀ · (1 - t/T))

    The effective shaped reward becomes:

        R'ₜ(s, a, s') = R(s, a, s') + λₜ · F(s, a, s')

    As λₜ → 0, the agent's objective converges to the original reward R.

    ============================================================================
    PROBLEM STATEMENT
    ============================================================================
    Fixed shaping weights can cause issues:
    - Too high: Agent over-relies on shaping, slow convergence
    - Too low: Shaping provides insufficient guidance
    - Imperfect potential: Fixed shaping may bias toward suboptimal behavior

    Adaptive decay solves this by providing strong initial guidance that
    fades, ensuring eventual convergence to the true optimal policy.

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    vs Fixed PBRS: Better asymptotic performance, handles imperfect potentials
    vs Curriculum learning: Focuses on reward rather than task difficulty
    Disadvantage: Introduces additional hyperparameters (decay schedule)

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(base_shaper) + O(1) for weight update
    - Space: O(base_shaper) + O(1) for weight tracking
    """

    def __init__(
        self,
        base_shaper: RewardShaper,
        dynamic_config: Optional[DynamicShapingConfig] = None,
    ) -> None:
        """Initialize adaptive shaper wrapping a base shaper.

        Args:
            base_shaper: Underlying reward shaper providing potential function.
            dynamic_config: Configuration for weight adaptation.
        """
        super().__init__(base_shaper.config)
        self.base_shaper = base_shaper
        self.dynamic_config = dynamic_config or DynamicShapingConfig()

        self._current_weight = self.dynamic_config.initial_weight
        self._step_count = 0
        self._episode_count = 0

    def potential(self, state: np.ndarray) -> float:
        """Delegate potential computation to base shaper."""
        return self.base_shaper.potential(state)

    def compute_shaping_bonus(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        done: bool = False,
    ) -> float:
        """Compute shaping bonus with adaptive weight.

        Args:
            state: Current state.
            next_state: Next state.
            done: Whether episode terminated.

        Returns:
            Adaptively weighted shaping bonus λₜ · F(s, s').
        """
        base_bonus = self.base_shaper.compute_shaping_bonus(state, next_state, done)
        adaptive_bonus = self._current_weight * base_bonus

        self._step_count += 1
        self._update_weight()

        return adaptive_bonus

    def _update_weight(self) -> None:
        """Update adaptive shaping weight according to decay schedule."""
        if self.dynamic_config.adaptation_method == "exponential":
            self._current_weight = max(
                self.dynamic_config.min_weight,
                self.dynamic_config.initial_weight
                * (self.dynamic_config.decay_rate ** self._step_count),
            )
        elif self.dynamic_config.adaptation_method == "linear":
            decay_steps = 1.0 / (1.0 - self.dynamic_config.decay_rate + 1e-8)
            progress = min(1.0, self._step_count / decay_steps)
            self._current_weight = max(
                self.dynamic_config.min_weight,
                self.dynamic_config.initial_weight * (1.0 - progress),
            )

    def on_episode_end(self) -> Dict[str, float]:
        """Handle episode end and return statistics.

        Returns:
            Dictionary with adaptive shaping statistics.
        """
        self._episode_count += 1
        stats = self.base_shaper.reset_episode_stats()
        stats["current_shaping_weight"] = self._current_weight
        stats["total_steps"] = float(self._step_count)
        stats["episode_count"] = float(self._episode_count)
        return stats

    @property
    def current_weight(self) -> float:
        """Get current adaptive shaping weight."""
        return self._current_weight

    def reset_weight(self) -> None:
        """Reset weight to initial value (useful for new training runs)."""
        self._current_weight = self.dynamic_config.initial_weight
        self._step_count = 0
        self._episode_count = 0


def compute_optimal_potential_from_value(
    value_function: Callable[[np.ndarray], float],
    states: np.ndarray,
) -> np.ndarray:
    """Compute optimal potential values from a value function.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The ideal potential is the optimal value function:

        Φ*(s) = V*(s)

    This provides "perfect" shaping that exactly captures the long-term value
    of each state. Using V* as potential gives the maximum possible learning
    speedup while preserving policy invariance.

    In practice, V* is unknown, but we can use:
    - Value function from pre-training
    - Fitted value from demonstrations
    - Hand-crafted approximation

    Args:
        value_function: Callable mapping states to value estimates V(s).
        states: Array of states to evaluate, shape (n_states, state_dim).

    Returns:
        Array of potential values, shape (n_states,).
    """
    return np.array(
        [value_function(s) for s in states],
        dtype=np.float64,
    )


def verify_policy_invariance(
    original_returns: np.ndarray,
    shaped_returns: np.ndarray,
    tolerance: float = 0.01,
) -> Tuple[bool, Dict[str, float]]:
    """Verify that reward shaping preserves policy return rankings.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    PBRS guarantees that for any two policies π₁, π₂:

        V^{π₁}(s) ≥ V^{π₂}(s) ⟺ V'^{π₁}(s) ≥ V'^{π₂}(s)

    This function empirically tests this by checking that the rank correlation
    between original and shaped returns is near 1.0.

    Note: Perfect correlation is expected for true PBRS. Deviations indicate:
    - Implementation bugs
    - Non-PBRS shaping (violates policy invariance)
    - Numerical precision issues

    Args:
        original_returns: Episode returns under original rewards, shape (n,).
        shaped_returns: Episode returns under shaped rewards, shape (n,).
        tolerance: Allowed deviation from perfect rank correlation.

    Returns:
        Tuple of:
        - bool: Whether policy invariance approximately holds
        - dict: Statistics including rank correlation
    """
    if len(original_returns) != len(shaped_returns):
        raise ValueError("Return arrays must have same length")

    original_ranks = np.argsort(np.argsort(original_returns))
    shaped_ranks = np.argsort(np.argsort(shaped_returns))

    if np.std(original_ranks) < 1e-8 or np.std(shaped_ranks) < 1e-8:
        rank_correlation = 1.0
    else:
        rank_correlation = float(np.corrcoef(original_ranks, shaped_ranks)[0, 1])

    invariance_holds = rank_correlation >= (1.0 - tolerance)

    stats = {
        "rank_correlation": rank_correlation,
        "original_mean": float(np.mean(original_returns)),
        "original_std": float(np.std(original_returns)),
        "shaped_mean": float(np.mean(shaped_returns)),
        "shaped_std": float(np.std(shaped_returns)),
        "return_difference_mean": float(
            np.mean(shaped_returns - original_returns)
        ),
    }

    return invariance_holds, stats


if __name__ == "__main__":
    print("=" * 70)
    print("Reward Shaping Module - Comprehensive Validation")
    print("=" * 70)

    np.random.seed(42)

    print("\n[Test 1] Distance-Based Potential Shaper")
    print("-" * 50)

    goal = np.array([10.0, 10.0])
    shaper = DistanceBasedShaper(
        goal_position=goal,
        norm_order=2,
        scale=1.0,
        config=ShapedRewardConfig(discount_factor=0.99),
    )

    test_states = [
        (np.array([0.0, 0.0]), "origin"),
        (np.array([5.0, 5.0]), "midpoint"),
        (np.array([9.0, 9.0]), "near goal"),
        (np.array([10.0, 10.0]), "at goal"),
    ]

    print("\nPotential values (higher = closer to goal):")
    potentials = []
    for state, name in test_states:
        p = shaper.potential(state)
        potentials.append(p)
        dist = np.linalg.norm(state - goal)
        print(f"  {name:12s}: Φ(s) = {p:8.4f}, distance = {dist:.4f}")

    assert potentials[-1] > potentials[0], "Potential should increase toward goal"
    assert all(
        potentials[i] <= potentials[i + 1] for i in range(len(potentials) - 1)
    ), "Potential should monotonically increase toward goal"
    print("\n  [PASS] Potential increases monotonically toward goal")

    print("\nShaping bonus for movement toward goal:")
    bonus = shaper.compute_shaping_bonus(
        np.array([0.0, 0.0]), np.array([1.0, 1.0]), done=False
    )
    print(f"  (0,0) → (1,1): F = {bonus:+.4f}")
    assert bonus > 0, "Bonus should be positive for movement toward goal"

    bonus_away = shaper.compute_shaping_bonus(
        np.array([5.0, 5.0]), np.array([4.0, 4.0]), done=False
    )
    print(f"  (5,5) → (4,4): F = {bonus_away:+.4f}")
    assert bonus_away < 0, "Bonus should be negative for movement away from goal"
    print("  [PASS] Shaping bonus correctly signed")

    print("\n[Test 2] Subgoal-Based Shaper")
    print("-" * 50)

    subgoals = np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0], [10.0, 10.0]])
    subgoal_shaper = SubgoalBasedShaper(
        subgoals=subgoals,
        subgoal_radius=0.5,
        config=ShapedRewardConfig(discount_factor=0.99),
    )

    trajectory = [
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        np.array([3.0, 3.0]),
        np.array([5.0, 5.0]),
    ]

    print("Following trajectory through subgoals:")
    for i, state in enumerate(trajectory):
        p = subgoal_shaper.potential(state)
        achieved = len(subgoal_shaper._achieved_subgoals)
        print(f"  Step {i}: pos={state}, achieved={achieved}, Φ(s)={p:.4f}")

    assert len(subgoal_shaper._achieved_subgoals) == 2, "Should achieve 2 subgoals"
    print("  [PASS] Subgoal achievement tracking correct")

    subgoal_shaper.reset()
    assert len(subgoal_shaper._achieved_subgoals) == 0, "Reset should clear subgoals"
    print("  [PASS] Reset clears achieved subgoals")

    print("\n[Test 3] Adaptive Reward Shaper")
    print("-" * 50)

    base_shaper = DistanceBasedShaper(goal_position=np.array([10.0, 10.0]))
    adaptive_shaper = AdaptiveRewardShaper(
        base_shaper=base_shaper,
        dynamic_config=DynamicShapingConfig(
            initial_weight=1.0,
            decay_rate=0.99,
            min_weight=0.01,
            adaptation_method="exponential",
        ),
    )

    print("Weight decay over 500 steps:")
    initial_weight = adaptive_shaper.current_weight
    for _ in range(500):
        adaptive_shaper.compute_shaping_bonus(
            np.array([0.0, 0.0]), np.array([1.0, 1.0])
        )
    final_weight = adaptive_shaper.current_weight

    print(f"  Initial weight: {initial_weight:.4f}")
    print(f"  After 500 steps: {final_weight:.4f}")
    print(f"  Decay ratio: {final_weight / initial_weight:.4f}")

    assert final_weight < initial_weight, "Weight should decay over time"
    assert final_weight >= 0.01, "Weight should respect minimum"
    print("  [PASS] Adaptive weight decay functions correctly")

    print("\n[Test 4] Learned Potential Shaper")
    print("-" * 50)

    learned_shaper = LearnedPotentialShaper(
        state_dim=4,
        hidden_dims=(32, 32),
        activation="relu",
    )

    test_states_4d = [np.random.randn(4) for _ in range(5)]
    potentials_learned = [learned_shaper.potential(s) for s in test_states_4d]

    print("Network outputs for random states:")
    for i, (s, p) in enumerate(zip(test_states_4d, potentials_learned)):
        print(f"  State {i}: Φ(s) = {p:.4f}")

    assert all(
        isinstance(p, float) for p in potentials_learned
    ), "Potentials should be floats"
    print("  [PASS] Learned potential network produces valid outputs")

    print("\n[Test 5] Policy Invariance Verification")
    print("-" * 50)

    original_returns = np.random.randn(100)
    shaped_returns = original_returns + np.random.randn(100) * 0.01

    invariant, stats = verify_policy_invariance(
        original_returns, shaped_returns, tolerance=0.05
    )

    print(f"  Rank correlation: {stats['rank_correlation']:.4f}")
    print(f"  Policy invariance holds: {invariant}")

    adversarial_shaped = -original_returns
    invariant_adv, _ = verify_policy_invariance(
        original_returns, adversarial_shaped
    )
    print(f"\n  Adversarial test (should fail): {not invariant_adv}")
    assert not invariant_adv, "Reversed returns should violate invariance"
    print("  [PASS] Invariance verification detects violations")

    print("\n[Test 6] Configuration Validation")
    print("-" * 50)

    try:
        ShapedRewardConfig(discount_factor=1.5)
        print("  [FAIL] Should reject discount_factor > 1")
    except ValueError:
        print("  [PASS] Rejects invalid discount_factor")

    try:
        ShapedRewardConfig(shaping_weight=-1.0)
        print("  [FAIL] Should reject negative shaping_weight")
    except ValueError:
        print("  [PASS] Rejects negative shaping_weight")

    try:
        ShapedRewardConfig(clip_range=(5.0, 1.0))
        print("  [FAIL] Should reject invalid clip_range")
    except ValueError:
        print("  [PASS] Rejects invalid clip_range")

    print("\n[Test 7] Episode Statistics")
    print("-" * 50)

    stats_shaper = DistanceBasedShaper(
        goal_position=np.array([10.0, 10.0]),
        config=ShapedRewardConfig(discount_factor=0.99),
    )

    for _ in range(20):
        state = np.random.rand(2) * 10
        next_state = state + np.random.randn(2) * 0.5
        stats_shaper.compute_shaping_bonus(state, next_state)

    episode_stats = stats_shaper.reset_episode_stats()
    print(f"  Episode shaping sum: {episode_stats['episode_shaping_sum']:.4f}")
    print(f"  Episode transitions: {episode_stats['episode_transitions']:.0f}")
    print(f"  Avg shaping bonus: {episode_stats['avg_shaping_bonus']:.4f}")

    assert episode_stats["episode_transitions"] == 20
    print("  [PASS] Episode statistics tracked correctly")

    print("\n" + "=" * 70)
    print("All validation tests passed successfully!")
    print("=" * 70)
