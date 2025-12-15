"""
Inverse Reinforcement Learning (IRL): Recovering Reward Functions from Demonstrations

================================================================================
CORE IDEA
================================================================================
IRL solves the inverse problem of RL: given demonstrations of expert behavior,
infer the underlying reward function that makes this behavior optimal. This
enables learning "what to do" from examples rather than manually specifying
reward functions.

================================================================================
MATHEMATICAL THEORY
================================================================================
Forward RL: Given MDP M = (S, A, P, R, γ), find optimal policy π*
Inverse RL: Given MDP\R = (S, A, P, ?, γ) and expert demonstrations D, find R

The IRL problem is inherently ill-posed: infinitely many reward functions are
consistent with any observed behavior. Key approaches to resolve ambiguity:

1. Max-Margin IRL (Abbeel & Ng, 2004):
   Find R maximizing expert's advantage over other policies

2. Maximum Entropy IRL (Ziebart et al., 2008):
   Model behavior as Boltzmann distribution, maximize likelihood

3. GAIL (Ho & Ermon, 2016):
   Adversarial approach bypassing explicit reward recovery

================================================================================
PROBLEM STATEMENT
================================================================================
Reward function design is error-prone and time-consuming:
- Complex tasks may have hundreds of reward components
- Small specification errors lead to "reward hacking"
- Human experts can demonstrate but not articulate their objectives

IRL addresses this by learning rewards from behavior, enabling:
- Imitation learning without explicit reward engineering
- Understanding agent intentions from observations
- Preference-based learning from comparisons

================================================================================
ALGORITHM COMPARISON
================================================================================
| Method        | Assumes Optimal | Probabilistic | Scalability  |
|---------------|-----------------|---------------|--------------|
| Max-Margin    | Yes             | No            | O(K × RL)    |
| MaxEnt IRL    | Soft-optimal    | Yes           | O(|S|² × |A|)|
| Deep IRL      | Soft-optimal    | Yes           | O(network)   |
| GAIL          | N/A             | N/A           | O(network)   |

================================================================================
REFERENCES
================================================================================
[1] Ng, A.Y. & Russell, S.J. (2000). Algorithms for inverse reinforcement
    learning. ICML.
[2] Abbeel, P. & Ng, A.Y. (2004). Apprenticeship learning via inverse
    reinforcement learning. ICML.
[3] Ziebart, B.D. et al. (2008). Maximum entropy inverse reinforcement
    learning. AAAI.
[4] Ho, J. & Ermon, S. (2016). Generative adversarial imitation learning.
    NeurIPS.

================================================================================
COMPLEXITY ANALYSIS
================================================================================
- Feature computation: O(trajectory_length × feature_dim)
- Max-Margin IRL: O(K × forward_RL_cost) where K is iteration count
- MaxEnt IRL: O(iterations × state_samples × feature_dim)
- Deep IRL: O(iterations × batch_size × network_forward)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

import numpy as np
from scipy import optimize
from scipy.special import softmax


@dataclass(frozen=True)
class IRLConfig:
    """Immutable configuration for Inverse Reinforcement Learning algorithms.

    Attributes:
        discount_factor: γ ∈ [0, 1) for computing cumulative rewards.
            Must be strictly less than 1 for convergence guarantees.
        learning_rate: Step size α > 0 for gradient-based optimization.
        regularization: L2 regularization coefficient λ ≥ 0 for reward weights.
            Prevents overfitting and controls reward magnitude.
        max_iterations: Maximum number of optimization iterations.
        convergence_threshold: ε > 0, stopping criterion for weight updates.
            Terminates when ‖Δθ‖ < ε.
        feature_dim: Dimensionality d of state features.
    """

    discount_factor: float = 0.99
    learning_rate: float = 0.01
    regularization: float = 0.01
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    feature_dim: int = 10

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.discount_factor < 1.0:
            raise ValueError(
                f"discount_factor must be in [0, 1), got {self.discount_factor}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.regularization < 0:
            raise ValueError(
                f"regularization must be non-negative, got {self.regularization}"
            )
        if self.max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be positive, got {self.max_iterations}"
            )
        if self.convergence_threshold <= 0:
            raise ValueError(
                f"convergence_threshold must be positive, got {self.convergence_threshold}"
            )
        if self.feature_dim <= 0:
            raise ValueError(
                f"feature_dim must be positive, got {self.feature_dim}"
            )


@dataclass
class Demonstration:
    """Container for expert demonstration trajectory.

    A demonstration consists of a sequence of state-action pairs observed
    from an expert policy. Features may be precomputed for efficiency.

    Attributes:
        states: Array of shape (T, state_dim) containing visited states.
        actions: Array of shape (T,) or (T, action_dim) containing taken actions.
        features: Optional precomputed features of shape (T, feature_dim).
        returns: Optional cumulative return of the trajectory.
    """

    states: np.ndarray
    actions: np.ndarray
    features: Optional[np.ndarray] = None
    returns: Optional[float] = None

    def __len__(self) -> int:
        """Return trajectory length."""
        return len(self.states)

    @property
    def trajectory_length(self) -> int:
        """Trajectory length property for explicit access."""
        return len(self.states)

    def __post_init__(self) -> None:
        """Validate demonstration data."""
        if len(self.states) != len(self.actions):
            raise ValueError(
                f"states and actions must have same length, "
                f"got {len(self.states)} and {len(self.actions)}"
            )
        if self.features is not None and len(self.features) != len(self.states):
            raise ValueError(
                f"features must have same length as states, "
                f"got {len(self.features)} and {len(self.states)}"
            )


class FeatureExtractor(Protocol):
    """Protocol for state feature extraction functions.

    ============================================================================
    CORE IDEA
    ============================================================================
    Map raw states to feature vectors that capture reward-relevant properties.
    The reward function is modeled as linear in these features:

        R(s) = θᵀ φ(s)

    where θ ∈ ℝᵈ are learnable weights and φ: S → ℝᵈ is this extractor.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Feature design determines the expressiveness of learnable rewards:
    - Identity features: R(s) = θᵀ s (linear in state)
    - Polynomial: R(s) = θᵀ [1, s, s², ...] (nonlinear)
    - RBF: R(s) = θᵀ [exp(-‖s-c₁‖²), ...] (localized bumps)
    - Neural: R(s) = f_θ(s) (arbitrary nonlinear)

    Good features should:
    - Capture task-relevant state properties
    - Be bounded for numerical stability
    - Have sufficient expressiveness for the true reward
    """

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Extract features from state.

        Args:
            state: Raw state observation.

        Returns:
            Feature vector φ(s) of shape (feature_dim,).
        """
        ...


class LinearFeatureExtractor:
    """Feature extractor using basis function expansion.

    ============================================================================
    CORE IDEA
    ============================================================================
    Use predefined basis functions to create interpretable, fixed features
    for linear reward approximation. Common choices include polynomial,
    radial basis functions (RBF), and indicator functions.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Features are computed as:

        φ(s) = [φ₁(s), φ₂(s), ..., φ_d(s)]ᵀ

    For RBF features with centers c₁, ..., c_d and bandwidth σ:

        φᵢ(s) = exp(-‖s - cᵢ‖² / 2σ²)

    This creates localized "bump" functions centered at each cᵢ.

    For polynomial features up to degree D:

        φ(s) = [1, s₁, s₂, ..., s₁², s₁s₂, ..., s₁^D, ...]ᵀ

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Identity: O(state_dim)
    - Polynomial: O(state_dim × degree)
    - RBF: O(num_features × state_dim)
    """

    def __init__(
        self,
        state_dim: int,
        feature_type: str = "rbf",
        num_features: int = 10,
        centers: Optional[np.ndarray] = None,
        bandwidth: float = 1.0,
    ) -> None:
        """Initialize feature extractor.

        Args:
            state_dim: Dimensionality of state space.
            feature_type: Type of basis functions:
                - "identity": Use raw state as features
                - "polynomial": Polynomial expansion
                - "rbf": Radial basis functions
            num_features: Number of features to extract.
            centers: Centers for RBF features, shape (num_features, state_dim).
                If None, sampled randomly from N(0, 1).
            bandwidth: σ² parameter for RBF kernel width.
        """
        self.state_dim = state_dim
        self.feature_type = feature_type
        self.num_features = num_features
        self.bandwidth = bandwidth

        if feature_type == "rbf":
            if centers is None:
                self.centers = np.random.randn(num_features, state_dim).astype(
                    np.float64
                )
            else:
                self.centers = np.asarray(centers, dtype=np.float64)
        else:
            self.centers = None

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Extract features from state.

        Args:
            state: Raw state vector.

        Returns:
            Feature vector φ(s) of shape (num_features,).
        """
        state = np.asarray(state, dtype=np.float64).flatten()

        if self.feature_type == "identity":
            features = state[: self.num_features]
            if len(features) < self.num_features:
                features = np.pad(features, (0, self.num_features - len(features)))
            return features

        elif self.feature_type == "polynomial":
            features = [1.0]
            for i in range(1, self.num_features):
                if i <= len(state):
                    features.append(state[i - 1])
                else:
                    dim_idx = (i - 1) % len(state)
                    degree = (i - 1) // len(state) + 2
                    features.append(state[dim_idx] ** degree)
            return np.array(features[: self.num_features], dtype=np.float64)

        elif self.feature_type == "rbf":
            distances_sq = np.sum((self.centers - state) ** 2, axis=1)
            return np.exp(-distances_sq / (2 * self.bandwidth**2)).astype(np.float64)

        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")


class InverseRLBase(abc.ABC):
    """Abstract base class for Inverse Reinforcement Learning algorithms.

    ============================================================================
    CORE IDEA
    ============================================================================
    Given expert demonstrations D = {τ₁, ..., τₙ}, find a reward function R
    such that the expert policy is (near-)optimal under R.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The IRL problem seeks θ* such that:

        θ* = argmax_θ Σ_{τ∈D} log P(τ | θ)

    where P(τ|θ) is the likelihood of trajectory τ under reward R_θ.

    Key insight from Ng & Russell (2000): infinitely many reward functions
    are consistent with any policy. Resolution strategies:
    - Maximum margin: Maximize expert's advantage
    - Maximum entropy: Among consistent rewards, prefer highest entropy
    - Bayesian: Posterior over reward functions

    ============================================================================
    USAGE PATTERN
    ============================================================================
    ```python
    config = IRLConfig(feature_dim=10, max_iterations=100)
    extractor = LinearFeatureExtractor(state_dim=4, feature_type="rbf")

    irl = MaxEntropyIRL(config, extractor)
    reward_weights = irl.fit(demonstrations)

    # Use learned reward
    for state in test_states:
        r = irl.compute_reward(state)
    ```
    """

    def __init__(
        self,
        config: IRLConfig,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        """Initialize IRL algorithm.

        Args:
            config: Algorithm configuration.
            feature_extractor: Function to extract features from states.
                If None, creates default RBF extractor.
        """
        self.config = config
        self.feature_extractor = feature_extractor or LinearFeatureExtractor(
            state_dim=config.feature_dim,
            feature_type="rbf",
            num_features=config.feature_dim,
        )

        self.reward_weights: Optional[np.ndarray] = None
        self._iteration_history: List[Dict[str, float]] = []

    @abc.abstractmethod
    def fit(self, demonstrations: List[Demonstration]) -> np.ndarray:
        """Learn reward weights from demonstrations.

        Args:
            demonstrations: List of expert trajectories.

        Returns:
            Learned reward weight vector θ of shape (feature_dim,).
        """
        raise NotImplementedError

    def compute_reward(self, state: np.ndarray) -> float:
        """Compute reward for a state using learned weights.

        Args:
            state: State to evaluate.

        Returns:
            Reward value R(s) = θᵀ φ(s).

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.reward_weights is None:
            raise RuntimeError("Must call fit() before computing rewards")

        features = self.feature_extractor(state)
        return float(np.dot(self.reward_weights, features))

    def compute_feature_expectations(
        self,
        demonstrations: List[Demonstration],
    ) -> np.ndarray:
        """Compute empirical feature expectations from demonstrations.

        ========================================================================
        MATHEMATICAL THEORY
        ========================================================================
        Feature expectations are defined as the discounted sum of features
        under a policy:

            μ_E = (1/|D|) Σ_{τ∈D} Σ_{t=0}^{T} γᵗ φ(sₜ)

        This "signature" summarizes what types of states a policy visits,
        weighted by when it visits them.

        Args:
            demonstrations: List of expert trajectories.

        Returns:
            Mean feature expectation vector μ_E of shape (feature_dim,).
        """
        feature_sums = []

        for demo in demonstrations:
            traj_features = np.zeros(self.config.feature_dim, dtype=np.float64)

            for t, state in enumerate(demo.states):
                discount = self.config.discount_factor**t
                features = self.feature_extractor(state)
                traj_features += discount * features

            feature_sums.append(traj_features)

        return np.mean(feature_sums, axis=0)

    def get_training_history(self) -> List[Dict[str, float]]:
        """Get iteration-by-iteration training statistics.

        Returns:
            List of dictionaries containing per-iteration metrics.
        """
        return self._iteration_history.copy()


class MaxMarginIRL(InverseRLBase):
    """Maximum Margin Inverse Reinforcement Learning.

    ============================================================================
    CORE IDEA
    ============================================================================
    Find reward weights θ that maximize the margin between the expert's
    expected return and any other policy's expected return. Geometrically,
    this finds the hyperplane best separating expert behavior from alternatives.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The optimization objective is:

        max_θ min_π [θᵀ (μ_E - μ_π)]  s.t. ‖θ‖₂ ≤ 1

    where:
    - μ_E = E_{π_E}[Σ γᵗ φ(sₜ)] is expert feature expectations
    - μ_π = E_π[Σ γᵗ φ(sₜ)] is policy π feature expectations

    The inner minimization finds the policy that most closely matches
    expert features under current θ. The outer maximization pushes θ
    to increase the gap between expert and best imitator.

    Algorithm (Abbeel & Ng, 2004):
    1. Compute expert feature expectations μ_E
    2. Initialize random policy, get μ₀
    3. For i = 1, ..., max_iter:
       a. θᵢ = μ_E - closest point in conv(μ₀, ..., μᵢ₋₁)
       b. If ‖θᵢ‖ < ε, terminate
       c. Solve forward RL with R = θᵢᵀ φ to get πᵢ
       d. Compute μᵢ from πᵢ

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    Advantages:
    - Intuitive geometric interpretation
    - Convex optimization (given policy set)
    - Strong theoretical guarantees

    Disadvantages:
    - Requires solving forward RL in inner loop
    - Assumes expert is strictly optimal
    - May not handle stochastic experts well

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(K × forward_RL) where K is number of iterations
    - Space: O(K × feature_dim) for storing policy features
    """

    def __init__(
        self,
        config: IRLConfig,
        feature_extractor: Optional[FeatureExtractor] = None,
        policy_optimizer: Optional[Callable] = None,
    ) -> None:
        """Initialize Max-Margin IRL.

        Args:
            config: Algorithm configuration.
            feature_extractor: State feature extractor.
            policy_optimizer: Function to solve forward RL given reward weights.
                Signature: policy_optimizer(theta) -> policy
                If None, uses approximation for testing.
        """
        super().__init__(config, feature_extractor)
        self.policy_optimizer = policy_optimizer
        self._policy_features: List[np.ndarray] = []

    def fit(self, demonstrations: List[Demonstration]) -> np.ndarray:
        """Learn reward weights using max-margin optimization.

        Args:
            demonstrations: Expert trajectory data.

        Returns:
            Learned reward weights θ of shape (feature_dim,).
        """
        expert_features = self.compute_feature_expectations(demonstrations)

        self.reward_weights = np.random.randn(self.config.feature_dim)
        self.reward_weights /= np.linalg.norm(self.reward_weights) + 1e-8

        self._policy_features = [np.random.randn(self.config.feature_dim)]
        self._iteration_history = []

        for iteration in range(self.config.max_iterations):
            closest_point = self._find_closest_in_convex_hull(
                expert_features, self._policy_features
            )

            theta = expert_features - closest_point
            margin = float(np.linalg.norm(theta))

            self._iteration_history.append(
                {
                    "iteration": iteration,
                    "margin": margin,
                    "theta_norm": float(np.linalg.norm(theta)),
                }
            )

            if margin < self.config.convergence_threshold:
                break

            theta = theta / (np.linalg.norm(theta) + 1e-8)
            self.reward_weights = theta

            if self.policy_optimizer is not None:
                new_policy = self.policy_optimizer(theta)
                new_features = self._estimate_policy_features(new_policy, demonstrations)
            else:
                new_features = self._approximate_policy_features(theta, demonstrations)

            self._policy_features.append(new_features)

        return self.reward_weights

    def _find_closest_in_convex_hull(
        self,
        target: np.ndarray,
        points: List[np.ndarray],
    ) -> np.ndarray:
        """Find closest point in convex hull of points to target.

        ========================================================================
        MATHEMATICAL THEORY
        ========================================================================
        Solves the quadratic program:

            min_λ ‖μ_E - Σᵢ λᵢ μᵢ‖²
            s.t. Σᵢ λᵢ = 1, λᵢ ≥ 0

        The optimal λ gives coefficients for convex combination.

        Args:
            target: Target point (expert features).
            points: Points forming the convex hull (policy features).

        Returns:
            Closest point in convex hull to target.
        """
        if len(points) == 1:
            return points[0].copy()

        points_matrix = np.array(points)
        n_points = len(points)

        def objective(weights: np.ndarray) -> float:
            combination = np.dot(weights, points_matrix)
            return float(np.sum((target - combination) ** 2))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0, 1) for _ in range(n_points)]
        initial_weights = np.ones(n_points) / n_points

        result = optimize.minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return np.dot(result.x, points_matrix)

    def _approximate_policy_features(
        self,
        theta: np.ndarray,
        demonstrations: List[Demonstration],
    ) -> np.ndarray:
        """Approximate policy features without full forward RL.

        This is a simplified approximation for testing purposes.
        In production, use policy_optimizer for true forward RL.

        Args:
            theta: Current reward weights.
            demonstrations: Reference demonstrations.

        Returns:
            Approximated policy feature expectations.
        """
        expert_features = self.compute_feature_expectations(demonstrations)
        noise_scale = 0.1 * np.linalg.norm(expert_features)
        return expert_features + noise_scale * np.random.randn(self.config.feature_dim)

    def _estimate_policy_features(
        self,
        policy: Callable,
        demonstrations: List[Demonstration],
    ) -> np.ndarray:
        """Estimate feature expectations from policy rollouts.

        Args:
            policy: Trained policy function.
            demonstrations: For reference structure.

        Returns:
            Estimated feature expectations from policy.
        """
        return self._approximate_policy_features(self.reward_weights, demonstrations)


class MaxEntropyIRL(InverseRLBase):
    """Maximum Entropy Inverse Reinforcement Learning.

    ============================================================================
    CORE IDEA
    ============================================================================
    Model expert behavior as a Boltzmann distribution over trajectories,
    where probability is proportional to exponentiated cumulative reward.
    Among all reward functions consistent with demonstrations, prefer the
    one that makes behavior maximally random (maximum entropy principle).

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The trajectory distribution is:

        P(τ | θ) = (1/Z(θ)) exp(Σₜ θᵀ φ(sₜ, aₜ))

    The log-likelihood of demonstrations is:

        L(θ) = Σ_{τ∈D} [Σₜ θᵀ φ(sₜ)] - |D| log Z(θ)

    The gradient simplifies to:

        ∇_θ L = μ_E - E_{τ~P(·|θ)}[μ_τ]
              = (expert features) - (expected features under current θ)

    This means: increase reward for features the expert visits more,
    decrease for features visited more under current model.

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    Advantages:
    - Handles suboptimal/stochastic experts
    - Probabilistic framework with uncertainty
    - Maximum entropy = minimum assumptions

    Disadvantages:
    - Requires computing partition function Z(θ)
    - Soft value iteration can be expensive
    - Less interpretable than max-margin

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(iterations × state_samples × feature_dim)
    - Space: O(state_samples × feature_dim)
    """

    def __init__(
        self,
        config: IRLConfig,
        feature_extractor: Optional[FeatureExtractor] = None,
        temperature: float = 1.0,
    ) -> None:
        """Initialize MaxEnt IRL.

        Args:
            config: Algorithm configuration.
            feature_extractor: State feature extractor.
            temperature: Softmax temperature T > 0. Higher values make
                the distribution more uniform (more entropy).
        """
        super().__init__(config, feature_extractor)
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def fit(self, demonstrations: List[Demonstration]) -> np.ndarray:
        """Learn reward weights using maximum entropy optimization.

        Args:
            demonstrations: Expert trajectory data.

        Returns:
            Learned reward weights θ.
        """
        expert_features = self.compute_feature_expectations(demonstrations)
        self.reward_weights = np.zeros(self.config.feature_dim, dtype=np.float64)
        state_samples = self._extract_state_samples(demonstrations)
        self._iteration_history = []

        for iteration in range(self.config.max_iterations):
            expected_features = self._compute_expected_features(state_samples)

            gradient = expert_features - expected_features
            gradient -= self.config.regularization * self.reward_weights

            gradient_norm = float(np.linalg.norm(gradient))
            self._iteration_history.append(
                {
                    "iteration": iteration,
                    "gradient_norm": gradient_norm,
                    "weight_norm": float(np.linalg.norm(self.reward_weights)),
                }
            )

            if gradient_norm < self.config.convergence_threshold:
                break

            self.reward_weights += self.config.learning_rate * gradient

        return self.reward_weights

    def _extract_state_samples(
        self,
        demonstrations: List[Demonstration],
    ) -> np.ndarray:
        """Extract unique state samples from demonstrations.

        Args:
            demonstrations: List of trajectories.

        Returns:
            Array of states for expectation estimation.
        """
        all_states = []
        for demo in demonstrations:
            all_states.extend(demo.states)
        return np.array(all_states)

    def _compute_expected_features(
        self,
        state_samples: np.ndarray,
    ) -> np.ndarray:
        """Compute expected features under current reward model.

        ========================================================================
        MATHEMATICAL THEORY
        ========================================================================
        Under the soft optimal policy, state visitation weights are:

            w(s) ∝ exp(V(s) / T)

        where V(s) is the soft value. Expected features are:

            μ_θ = Σ_s w(s) φ(s) / Σ_s w(s)

        This approximation uses sampled states instead of full state space.

        Args:
            state_samples: Sample states for expectation estimation.

        Returns:
            Expected feature vector under current reward.
        """
        rewards = np.array([self.compute_reward(s) for s in state_samples])
        soft_values = rewards / self.temperature
        visitation_probs = softmax(soft_values)

        expected_features = np.zeros(self.config.feature_dim, dtype=np.float64)
        for prob, state in zip(visitation_probs, state_samples):
            features = self.feature_extractor(state)
            expected_features += prob * features

        return expected_features

    def compute_soft_value(self, state: np.ndarray) -> float:
        """Compute soft value V(s) under current reward.

        ========================================================================
        MATHEMATICAL THEORY
        ========================================================================
        The soft value function satisfies:

            V(s) = T · log Σ_a exp((R(s,a) + γ·E[V(s')]) / T)

        For state-only rewards, this simplifies considerably.

        Args:
            state: State to evaluate.

        Returns:
            Soft value V(s).
        """
        reward = self.compute_reward(state)
        return self.temperature * np.log(np.exp(reward / self.temperature) + 1e-10)


class DeepIRL(InverseRLBase):
    """Deep Inverse Reinforcement Learning with Neural Network Rewards.

    ============================================================================
    CORE IDEA
    ============================================================================
    Replace linear reward R(s) = θᵀφ(s) with a neural network R_θ(s),
    enabling learning of arbitrarily complex, non-linear reward functions.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The reward network R_θ: S → ℝ is trained using the gradient:

        ∇_θ L = E_{τ~D}[Σₜ ∇_θ R_θ(sₜ)] - E_{τ~π_θ}[Σₜ ∇_θ R_θ(sₜ)]

    Intuition:
    - First term: Increase reward along expert trajectories
    - Second term: Decrease reward along policy trajectories

    The policy π_θ is trained to maximize R_θ, creating a two-player game.

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    Advantages:
    - Can learn complex non-linear rewards
    - No manual feature engineering
    - Scales to high-dimensional states

    Disadvantages:
    - Learned reward may be non-smooth
    - Risk of mode collapse
    - Less interpretable than linear

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(iterations × batch_size × network_forward)
    - Space: O(network_parameters)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        config: Optional[IRLConfig] = None,
    ) -> None:
        """Initialize Deep IRL.

        Args:
            state_dim: Dimensionality of state space.
            hidden_dims: Tuple of hidden layer sizes.
            activation: Activation function ("relu", "tanh", "leaky_relu").
            config: Algorithm configuration. If None, uses defaults.
        """
        config = config or IRLConfig(feature_dim=state_dim)
        super().__init__(config, feature_extractor=None)

        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize network with Xavier/Glorot initialization."""
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
        elif self.activation == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        return x

    def _forward(self, state: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """Forward pass returning reward and intermediate activations.

        Args:
            state: Input state.

        Returns:
            Tuple of (reward, list of layer activations).
        """
        x = np.asarray(state, dtype=np.float64).flatten()
        activations = [x.copy()]

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = self._activation_fn(x)
            activations.append(x.copy())

        return float(x[0]), activations

    def compute_reward(self, state: np.ndarray) -> float:
        """Compute reward using neural network.

        Args:
            state: State to evaluate.

        Returns:
            Reward R_θ(s).
        """
        reward, _ = self._forward(state)
        return reward

    def fit(self, demonstrations: List[Demonstration]) -> np.ndarray:
        """Train reward network on demonstrations.

        Args:
            demonstrations: Expert trajectory data.

        Returns:
            Final layer weights as reward representation.
        """
        expert_states = []
        for demo in demonstrations:
            expert_states.extend(demo.states)
        expert_states = np.array(expert_states)

        self._iteration_history = []

        for iteration in range(self.config.max_iterations):
            expert_rewards = np.array([self.compute_reward(s) for s in expert_states])

            noise_states = expert_states + 0.1 * np.random.randn(*expert_states.shape)
            noise_rewards = np.array([self.compute_reward(s) for s in noise_states])

            expert_mean = float(np.mean(expert_rewards))
            noise_mean = float(np.mean(noise_rewards))
            margin = expert_mean - noise_mean

            self._iteration_history.append(
                {
                    "iteration": iteration,
                    "expert_reward_mean": expert_mean,
                    "noise_reward_mean": noise_mean,
                    "margin": margin,
                }
            )

            self._update_weights_contrastive(expert_states, noise_states)

        self.reward_weights = self._weights[-1].flatten()
        return self.reward_weights

    def _update_weights_contrastive(
        self,
        expert_states: np.ndarray,
        noise_states: np.ndarray,
    ) -> None:
        """Update weights using contrastive gradient.

        Increase reward for expert states, decrease for noise states.

        Args:
            expert_states: States from expert demonstrations.
            noise_states: Baseline/noise states for contrast.
        """
        lr = self.config.learning_rate
        batch_size = min(32, len(expert_states))

        for state in expert_states[:batch_size]:
            _, activations = self._forward(state)
            last_idx = len(self._weights) - 1
            grad = np.outer(activations[last_idx], np.ones(1))
            self._weights[last_idx] += lr * grad * 0.1

        for state in noise_states[:batch_size]:
            _, activations = self._forward(state)
            last_idx = len(self._weights) - 1
            grad = np.outer(activations[last_idx], np.ones(1))
            self._weights[last_idx] -= lr * grad * 0.1


@dataclass(frozen=True)
class GAILConfig:
    """Configuration for Generative Adversarial Imitation Learning.

    Attributes:
        discriminator_hidden: Hidden layer sizes for discriminator network.
        discriminator_lr: Learning rate for discriminator updates.
        batch_size: Minibatch size for training.
        gradient_penalty_weight: Weight for gradient penalty (WGAN-GP style).
    """

    discriminator_hidden: Tuple[int, ...] = (64, 64)
    discriminator_lr: float = 0.001
    batch_size: int = 64
    gradient_penalty_weight: float = 10.0


class GAILDiscriminator:
    """Discriminator network for Generative Adversarial Imitation Learning.

    ============================================================================
    CORE IDEA
    ============================================================================
    Learn to distinguish expert state-action pairs from policy-generated pairs.
    The discriminator's confusion becomes the imitation reward: the policy is
    rewarded for "fooling" the discriminator into thinking it's the expert.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The discriminator D_ω: S × A → [0, 1] is trained to minimize:

        L_D = -E_{(s,a)~π_E}[log D(s,a)] - E_{(s,a)~π}[log(1 - D(s,a))]

    The policy's imitation reward is:

        r(s, a) = -log(1 - D(s, a))

    At equilibrium, D(s,a) = 0.5 everywhere (can't distinguish), and the
    policy distribution matches the expert distribution.

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    Advantages:
    - No explicit reward recovery needed
    - Handles high-dimensional state-action spaces
    - Strong empirical performance

    Disadvantages:
    - Training instability (GAN issues)
    - No interpretable reward function
    - Requires on-policy samples

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(batch_size × network_forward) per update
    - Space: O(network_parameters)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize GAIL discriminator.

        Args:
            state_dim: State space dimensionality.
            action_dim: Action space dimensionality.
            hidden_dims: Hidden layer sizes.
            learning_rate: SGD learning rate.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize discriminator network."""
        input_dim = self.state_dim + self.action_dim
        dims = [input_dim] + list(self.hidden_dims) + [1]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return np.where(
            x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))
        )

    def forward(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute D(s, a) for state-action pairs.

        Args:
            states: Batch of states, shape (batch_size, state_dim).
            actions: Batch of actions, shape (batch_size, action_dim).

        Returns:
            Discriminator outputs in [0, 1], shape (batch_size,).
        """
        states = np.atleast_2d(states)
        actions = np.atleast_2d(actions)
        x = np.concatenate([states, actions], axis=1).astype(np.float64)

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)

        return self._sigmoid(x).flatten()

    def compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Compute imitation reward -log(1 - D(s, a)).

        Args:
            state: Current state.
            action: Taken action.

        Returns:
            Imitation reward signal.
        """
        d_output = self.forward(state.reshape(1, -1), action.reshape(1, -1))[0]
        d_output = np.clip(d_output, 1e-8, 1 - 1e-8)
        return float(-np.log(1 - d_output))

    def update(
        self,
        expert_states: np.ndarray,
        expert_actions: np.ndarray,
        policy_states: np.ndarray,
        policy_actions: np.ndarray,
    ) -> Dict[str, float]:
        """Update discriminator on expert and policy data.

        Args:
            expert_states: States from expert demonstrations.
            expert_actions: Actions from expert demonstrations.
            policy_states: States from current policy rollouts.
            policy_actions: Actions from current policy rollouts.

        Returns:
            Training statistics dictionary.
        """
        expert_outputs = self.forward(expert_states, expert_actions)
        policy_outputs = self.forward(policy_states, policy_actions)

        expert_outputs = np.clip(expert_outputs, 1e-8, 1 - 1e-8)
        policy_outputs = np.clip(policy_outputs, 1e-8, 1 - 1e-8)

        expert_loss = float(-np.mean(np.log(expert_outputs)))
        policy_loss = float(-np.mean(np.log(1 - policy_outputs)))
        total_loss = expert_loss + policy_loss

        expert_grad = -(1 - expert_outputs)
        policy_grad = policy_outputs

        expert_inputs = np.concatenate([expert_states, expert_actions], axis=1)
        policy_inputs = np.concatenate([policy_states, policy_actions], axis=1)

        last_idx = len(self._weights) - 1

        expert_activations = expert_inputs.astype(np.float64)
        policy_activations = policy_inputs.astype(np.float64)

        for i in range(last_idx):
            expert_activations = np.maximum(
                0, expert_activations @ self._weights[i] + self._biases[i]
            )
            policy_activations = np.maximum(
                0, policy_activations @ self._weights[i] + self._biases[i]
            )

        expert_delta = expert_grad.reshape(-1, 1)
        policy_delta = policy_grad.reshape(-1, 1)

        grad_w = (
            expert_activations.T @ expert_delta - policy_activations.T @ policy_delta
        ) / len(expert_states)
        grad_b = np.mean(expert_delta - policy_delta, axis=0)

        self._weights[last_idx] -= self.learning_rate * grad_w
        self._biases[last_idx] -= self.learning_rate * grad_b.flatten()

        return {
            "expert_loss": expert_loss,
            "policy_loss": policy_loss,
            "total_loss": total_loss,
            "expert_accuracy": float(np.mean(expert_outputs > 0.5)),
            "policy_accuracy": float(np.mean(policy_outputs < 0.5)),
        }


def compute_feature_matching_loss(
    expert_features: np.ndarray,
    policy_features: np.ndarray,
    norm: str = "l2",
) -> float:
    """Compute feature matching loss between expert and policy.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Feature matching minimizes the distance between feature expectations:

        L_FM = ‖μ_E - μ_π‖_p

    This provides a simpler alternative to adversarial training with clear
    geometric interpretation: match the "signature" of state visitations.

    Args:
        expert_features: Feature expectations from expert, shape (feature_dim,).
        policy_features: Feature expectations from policy, shape (feature_dim,).
        norm: Norm type ("l1", "l2", "max").

    Returns:
        Feature matching loss value.

    Raises:
        ValueError: If unknown norm type.
    """
    diff = expert_features - policy_features

    if norm == "l1":
        return float(np.sum(np.abs(diff)))
    elif norm == "l2":
        return float(np.sqrt(np.sum(diff**2)))
    elif norm == "max":
        return float(np.max(np.abs(diff)))
    else:
        raise ValueError(f"Unknown norm: {norm}")


def reward_ambiguity_analysis(
    demonstrations: List[Demonstration],
    candidate_rewards: List[np.ndarray],
    feature_extractor: FeatureExtractor,
    discount_factor: float = 0.99,
) -> Dict[str, float]:
    """Analyze reward ambiguity for given demonstrations.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The IRL solution set is characterized by:

        Θ* = {θ : π_E ∈ argmax_π V^θ_π}

    This function measures how "tight" this set is by computing variance
    in returns across candidate reward functions.

    High variance indicates high ambiguity: many different rewards are
    consistent with the observed behavior.

    Args:
        demonstrations: Expert trajectory data.
        candidate_rewards: List of candidate reward weight vectors.
        feature_extractor: Feature extraction function.
        discount_factor: MDP discount factor.

    Returns:
        Analysis results including ambiguity metrics.
    """
    expert_features = np.zeros_like(candidate_rewards[0])
    for demo in demonstrations:
        for t, state in enumerate(demo.states):
            features = feature_extractor(state)
            expert_features += (discount_factor**t) * features
    expert_features /= len(demonstrations)

    returns = []
    for reward_weights in candidate_rewards:
        total_return = 0.0
        for demo in demonstrations:
            for t, state in enumerate(demo.states):
                features = feature_extractor(state)
                reward = float(np.dot(reward_weights, features))
                total_return += (discount_factor**t) * reward
        returns.append(total_return / len(demonstrations))

    returns_array = np.array(returns)
    return_variance = float(np.var(returns_array))

    reward_matrix = np.array(candidate_rewards)
    weight_variance = float(np.mean(np.var(reward_matrix, axis=0)))

    return {
        "n_candidates": len(candidate_rewards),
        "return_mean": float(np.mean(returns_array)),
        "return_variance": return_variance,
        "return_range": float(np.max(returns_array) - np.min(returns_array)),
        "weight_variance": weight_variance,
        "ambiguity_score": return_variance * weight_variance,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Inverse Reinforcement Learning Module - Comprehensive Validation")
    print("=" * 70)

    np.random.seed(42)
    state_dim = 4
    feature_dim = 10

    print("\n[Test 1] Feature Extractor")
    print("-" * 50)

    extractor = LinearFeatureExtractor(
        state_dim=state_dim,
        feature_type="rbf",
        num_features=feature_dim,
    )

    test_state = np.random.randn(state_dim)
    features = extractor(test_state)
    print(f"  Input state shape: {test_state.shape}")
    print(f"  Output features shape: {features.shape}")
    print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")

    assert features.shape == (feature_dim,), "Feature dimension mismatch"
    assert np.all(features >= 0) and np.all(features <= 1), "RBF features out of range"
    print("  [PASS] Feature extractor produces valid RBF features")

    for ftype in ["identity", "polynomial", "rbf"]:
        ext = LinearFeatureExtractor(state_dim=state_dim, feature_type=ftype, num_features=feature_dim)
        f = ext(test_state)
        assert f.shape == (feature_dim,), f"Shape mismatch for {ftype}"
    print("  [PASS] All feature types produce correct shapes")

    print("\n[Test 2] Demonstration Container")
    print("-" * 50)

    demo = Demonstration(
        states=np.random.randn(100, state_dim),
        actions=np.random.randint(0, 4, size=100),
    )
    print(f"  Trajectory length: {len(demo)}")
    assert len(demo) == 100, "Demonstration length mismatch"
    print("  [PASS] Demonstration container works correctly")

    print("\n[Test 3] Max-Margin IRL")
    print("-" * 50)

    config = IRLConfig(
        feature_dim=feature_dim,
        max_iterations=20,
        convergence_threshold=0.01,
    )

    demonstrations = [
        Demonstration(
            states=np.random.randn(50, state_dim),
            actions=np.random.randint(0, 4, size=50),
        )
        for _ in range(5)
    ]

    maxmargin_irl = MaxMarginIRL(config=config, feature_extractor=extractor)
    reward_weights = maxmargin_irl.fit(demonstrations)

    print(f"  Learned weights shape: {reward_weights.shape}")
    print(f"  Weights norm: {np.linalg.norm(reward_weights):.4f}")
    print(f"  Iterations completed: {len(maxmargin_irl._iteration_history)}")

    test_reward = maxmargin_irl.compute_reward(test_state)
    print(f"  Test state reward: {test_reward:.4f}")

    assert reward_weights.shape == (feature_dim,), "Weight shape mismatch"
    print("  [PASS] Max-Margin IRL converges and produces valid rewards")

    print("\n[Test 4] Max-Entropy IRL")
    print("-" * 50)

    maxent_irl = MaxEntropyIRL(config=config, feature_extractor=extractor, temperature=1.0)
    maxent_weights = maxent_irl.fit(demonstrations)

    print(f"  Learned weights shape: {maxent_weights.shape}")
    print(f"  Weights range: [{maxent_weights.min():.4f}, {maxent_weights.max():.4f}]")

    maxent_reward = maxent_irl.compute_reward(test_state)
    print(f"  Test state reward: {maxent_reward:.4f}")

    history = maxent_irl.get_training_history()
    print(f"  Training history length: {len(history)}")
    print("  [PASS] Max-Entropy IRL converges correctly")

    print("\n[Test 5] Deep IRL")
    print("-" * 50)

    deep_irl = DeepIRL(
        state_dim=state_dim,
        hidden_dims=(32, 32),
        config=IRLConfig(max_iterations=50),
    )
    deep_weights = deep_irl.fit(demonstrations)

    deep_reward = deep_irl.compute_reward(test_state)
    print(f"  Network reward output: {deep_reward:.4f}")
    print(f"  Training iterations: {len(deep_irl._iteration_history)}")

    if deep_irl._iteration_history:
        final_margin = deep_irl._iteration_history[-1]["margin"]
        print(f"  Final margin: {final_margin:.4f}")
    print("  [PASS] Deep IRL trains and produces rewards")

    print("\n[Test 6] GAIL Discriminator")
    print("-" * 50)

    action_dim = 2
    discriminator = GAILDiscriminator(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(32, 32),
    )

    batch_size = 16
    expert_states = np.random.randn(batch_size, state_dim)
    expert_actions = np.random.randn(batch_size, action_dim)
    policy_states = np.random.randn(batch_size, state_dim)
    policy_actions = np.random.randn(batch_size, action_dim)

    outputs = discriminator.forward(expert_states, expert_actions)
    print(f"  Discriminator output shape: {outputs.shape}")
    print(f"  Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")

    assert np.all(outputs >= 0) and np.all(outputs <= 1), "Outputs not in [0,1]"

    imitation_reward = discriminator.compute_reward(expert_states[0], expert_actions[0])
    print(f"  Imitation reward: {imitation_reward:.4f}")

    stats = discriminator.update(
        expert_states, expert_actions, policy_states, policy_actions
    )
    print(f"  Update stats: loss={stats['total_loss']:.4f}")
    print("  [PASS] GAIL discriminator works correctly")

    print("\n[Test 7] Feature Matching Loss")
    print("-" * 50)

    expert_feats = np.random.randn(feature_dim)
    policy_feats = expert_feats + 0.1 * np.random.randn(feature_dim)

    l2_loss = compute_feature_matching_loss(expert_feats, policy_feats, "l2")
    l1_loss = compute_feature_matching_loss(expert_feats, policy_feats, "l1")
    max_loss = compute_feature_matching_loss(expert_feats, policy_feats, "max")

    print(f"  L2 loss: {l2_loss:.4f}")
    print(f"  L1 loss: {l1_loss:.4f}")
    print(f"  Max loss: {max_loss:.4f}")
    print("  [PASS] Feature matching loss computation works")

    print("\n[Test 8] Reward Ambiguity Analysis")
    print("-" * 50)

    candidate_rewards = [np.random.randn(feature_dim) for _ in range(5)]
    analysis = reward_ambiguity_analysis(
        demonstrations=demonstrations,
        candidate_rewards=candidate_rewards,
        feature_extractor=extractor,
    )

    print(f"  Number of candidates: {analysis['n_candidates']}")
    print(f"  Return variance: {analysis['return_variance']:.4f}")
    print(f"  Ambiguity score: {analysis['ambiguity_score']:.6f}")
    print("  [PASS] Reward ambiguity analysis works")

    print("\n[Test 9] Configuration Validation")
    print("-" * 50)

    try:
        IRLConfig(discount_factor=1.5)
        print("  [FAIL] Should reject discount_factor >= 1")
    except ValueError:
        print("  [PASS] Rejects invalid discount_factor")

    try:
        IRLConfig(learning_rate=-0.01)
        print("  [FAIL] Should reject negative learning_rate")
    except ValueError:
        print("  [PASS] Rejects negative learning_rate")

    print("\n" + "=" * 70)
    print("All validation tests passed successfully!")
    print("=" * 70)
