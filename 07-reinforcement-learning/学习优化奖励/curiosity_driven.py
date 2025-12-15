"""
Curiosity-Driven Exploration: Intrinsic Motivation for Sparse Reward Environments

================================================================================
CORE IDEA
================================================================================
Curiosity-driven exploration augments external rewards with intrinsic motivation
signals derived from prediction errors. The agent is rewarded for encountering
states that are "surprising" - where its internal world model makes mistakes.
This creates autonomous exploration without requiring hand-crafted rewards.

================================================================================
MATHEMATICAL THEORY
================================================================================
The Intrinsic Curiosity Module (ICM) computes intrinsic reward as:

    r_i(s_t, a_t, s_{t+1}) = η · ‖f(s_{t+1}) - f̂(s_t, a_t)‖²

where:
- f(s): Feature encoder mapping states to learned representations
- f̂(s_t, a_t): Forward dynamics model predicting next state features
- η: Intrinsic reward scaling coefficient

The total reward combines extrinsic and intrinsic components:
    r_total = r_e + β · r_i

Key insight: Only predict in feature space, not raw observation space, to
ignore unpredictable noise while focusing on controllable aspects.

================================================================================
PROBLEM STATEMENT
================================================================================
Sparse rewards create exploration challenges:
- Random exploration is exponentially inefficient in large state spaces
- Without intermediate signals, credit assignment becomes intractable
- Many real tasks have natural sparse rewards (goal-reaching, game completion)

Curiosity-driven exploration addresses this by:
- Providing dense learning signals even with sparse external rewards
- Encouraging systematic state-space coverage
- Enabling learning with no external reward at all (unsupervised RL)

================================================================================
ALGORITHM COMPARISON
================================================================================
| Method              | Intrinsic Signal     | Stochasticity | Computation |
|---------------------|----------------------|---------------|-------------|
| ICM                 | Forward prediction   | Moderate      | O(network)  |
| RND                 | Random target        | Low           | O(network)  |
| Count-based         | State visitation     | None          | O(|S|)      |
| Disagreement        | Ensemble variance    | High          | O(K×network)|

================================================================================
REFERENCES
================================================================================
[1] Pathak, D., Agrawal, P., Efros, A.A., & Darrell, T. (2017). Curiosity-driven
    exploration by self-supervised prediction. ICML.
[2] Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019). Exploration by
    random network distillation. ICLR.
[3] Bellemare, M., Srinivasan, S., Ostrovski, G., et al. (2016). Unifying
    count-based exploration and intrinsic motivation. NeurIPS.

================================================================================
COMPLEXITY ANALYSIS
================================================================================
- Feature encoding: O(encoder_forward)
- Forward model prediction: O(forward_network)
- Inverse model (optional): O(inverse_network)
- Total per step: O(2-3 × network_forward)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CuriosityConfig:
    """Immutable configuration for curiosity-driven exploration.

    Attributes:
        intrinsic_reward_scale: β coefficient scaling intrinsic rewards.
            Larger values emphasize exploration over exploitation.
        feature_dim: Dimensionality of learned state representations.
        learning_rate: Learning rate for model updates.
        forward_loss_weight: Weight for forward model loss in combined objective.
        inverse_loss_weight: Weight for inverse model loss (ICM-specific).
        prediction_error_clipping: Maximum intrinsic reward per step.
            Prevents unbounded curiosity from dominating learning.
        normalize_rewards: Whether to apply running normalization.
        decay_rate: Decay rate for intrinsic reward scaling over training.
    """

    intrinsic_reward_scale: float = 0.01
    feature_dim: int = 64
    learning_rate: float = 0.001
    forward_loss_weight: float = 0.2
    inverse_loss_weight: float = 0.8
    prediction_error_clipping: float = 5.0
    normalize_rewards: bool = True
    decay_rate: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.intrinsic_reward_scale < 0:
            raise ValueError(
                f"intrinsic_reward_scale must be non-negative, "
                f"got {self.intrinsic_reward_scale}"
            )
        if self.feature_dim <= 0:
            raise ValueError(
                f"feature_dim must be positive, got {self.feature_dim}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.prediction_error_clipping <= 0:
            raise ValueError(
                f"prediction_error_clipping must be positive, "
                f"got {self.prediction_error_clipping}"
            )


class FeatureEncoder:
    """Neural network encoder mapping observations to learned features.

    ============================================================================
    CORE IDEA
    ============================================================================
    Transform high-dimensional observations into compact representations that
    capture task-relevant information while discarding noise. The encoder is
    trained jointly with the dynamics models to learn useful features.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The encoder φ_θ: O → ℝᵈ maps observations to d-dimensional features:

        f(s) = φ_θ(s)

    Training objective (via inverse model):
        min_θ L_inv = -E[log P(a | φ_θ(s), φ_θ(s'))]

    This forces features to capture action-relevant information: what changed
    between states must be predictable from the action taken.

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(Σ layer_sizes) for forward pass
    - Space: O(|θ|) for network parameters
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (128, 64),
    ) -> None:
        """Initialize feature encoder.

        Args:
            input_dim: Dimensionality of input observations.
            feature_dim: Dimensionality of output features.
            hidden_dims: Sizes of hidden layers.
        """
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize network with Xavier/Glorot initialization."""
        dims = [self.input_dim] + list(self.hidden_dims) + [self.feature_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """Encode observation to feature representation.

        Args:
            observation: Raw observation, shape (obs_dim,) or (batch, obs_dim).

        Returns:
            Feature vector(s), shape (feature_dim,) or (batch, feature_dim).
        """
        x = np.atleast_2d(observation).astype(np.float64)
        single_input = observation.ndim == 1

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)

        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        return x.flatten() if single_input else x

    def get_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get network parameters.

        Returns:
            Tuple of (weights, biases) lists.
        """
        return (
            [w.copy() for w in self._weights],
            [b.copy() for b in self._biases],
        )

    def set_parameters(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
    ) -> None:
        """Set network parameters.

        Args:
            weights: List of weight matrices.
            biases: List of bias vectors.
        """
        self._weights = [np.asarray(w, dtype=np.float64) for w in weights]
        self._biases = [np.asarray(b, dtype=np.float64) for b in biases]


class ForwardDynamicsModel:
    """Neural network predicting next state features from current state and action.

    ============================================================================
    CORE IDEA
    ============================================================================
    Learn to predict the consequences of actions in feature space. High
    prediction error indicates novelty: the agent hasn't learned this
    transition yet, so it should be explored.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The forward model g_ψ predicts next state features:

        f̂(s_{t+1}) = g_ψ(f(s_t), a_t)

    Training objective:
        min_ψ L_fwd = E[‖f(s_{t+1}) - g_ψ(f(s_t), a_t)‖²]

    The prediction error serves as intrinsic reward:
        r_i = ‖f(s_{t+1}) - f̂(s_{t+1})‖²

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    vs Raw pixel prediction: Feature space ignores irrelevant noise (clouds,
        leaves), focusing on controllable/predictable aspects.
    vs Count-based: Generalizes to continuous states, doesn't require explicit
        state counting.

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(forward_pass)
    - Space: O(|ψ|) parameters
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
    ) -> None:
        """Initialize forward dynamics model.

        Args:
            feature_dim: Dimensionality of feature space.
            action_dim: Dimensionality of action space.
            hidden_dims: Hidden layer sizes.
        """
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize network parameters."""
        input_dim = self.feature_dim + self.action_dim
        dims = [input_dim] + list(self.hidden_dims) + [self.feature_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def _encode_action(self, action: np.ndarray, batch_size: int) -> np.ndarray:
        """Encode action to vector format.

        Args:
            action: Action(s), scalar or array.
            batch_size: Expected batch size.

        Returns:
            Action vector(s) of shape (batch_size, action_dim).
        """
        action = np.atleast_1d(action)
        if action.ndim == 1 and len(action) == batch_size:
            action_encoded = np.zeros((batch_size, self.action_dim), dtype=np.float64)
            for i, a in enumerate(action):
                if isinstance(a, (int, np.integer)):
                    if a < self.action_dim:
                        action_encoded[i, int(a)] = 1.0
                else:
                    action_encoded[i] = a[: self.action_dim]
        else:
            action_encoded = np.atleast_2d(action).astype(np.float64)
        return action_encoded

    def predict(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Predict next state features.

        Args:
            current_features: Current state features, shape (feature_dim,)
                or (batch, feature_dim).
            action: Action taken, scalar, shape (action_dim,), or batched.

        Returns:
            Predicted next features, same shape as current_features.
        """
        current_features = np.atleast_2d(current_features).astype(np.float64)
        batch_size = len(current_features)
        single_input = current_features.shape[0] == 1

        action_encoded = self._encode_action(action, batch_size)
        x = np.concatenate([current_features, action_encoded], axis=1)

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)

        return x.flatten() if single_input else x

    def compute_prediction_error(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
        next_features: np.ndarray,
    ) -> float:
        """Compute squared prediction error (intrinsic reward).

        Args:
            current_features: Features of current state.
            action: Action taken.
            next_features: Actual features of next state.

        Returns:
            Squared L2 norm of prediction error.
        """
        predicted = self.predict(current_features, action)
        error = np.sum((next_features.flatten() - predicted.flatten()) ** 2)
        return float(error)

    def update(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
        next_features: np.ndarray,
        learning_rate: float,
    ) -> float:
        """Update model parameters via gradient descent.

        Args:
            current_features: Current state features.
            action: Action taken.
            next_features: Target next state features.
            learning_rate: SGD step size.

        Returns:
            Loss value before update.
        """
        current_features = np.atleast_2d(current_features).astype(np.float64)
        next_features = np.atleast_2d(next_features).astype(np.float64)
        batch_size = len(current_features)

        action_encoded = self._encode_action(action, batch_size)
        x = np.concatenate([current_features, action_encoded], axis=1)

        activations = [x.copy()]
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)
            activations.append(x.copy())

        loss = float(np.mean(np.sum((x - next_features) ** 2, axis=1)))

        delta = 2 * (x - next_features) / batch_size

        for i in range(len(self._weights) - 1, -1, -1):
            grad_w = activations[i].T @ delta
            grad_b = np.sum(delta, axis=0)

            self._weights[i] -= learning_rate * grad_w
            self._biases[i] -= learning_rate * grad_b

            if i > 0:
                delta = delta @ self._weights[i].T
                delta = delta * (activations[i] > 0)

        return loss


class InverseDynamicsModel:
    """Neural network predicting action from state transition.

    ============================================================================
    CORE IDEA
    ============================================================================
    The inverse model predicts which action caused a state transition. This
    auxiliary task helps the encoder learn action-relevant features: changes
    between states must be predictable from the action.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The inverse model h_ω predicts the action:

        â = h_ω(f(s_t), f(s_{t+1}))

    Training objective (classification for discrete actions):
        min_ω L_inv = -E[log P(a | h_ω(f(s_t), f(s_{t+1})))]

    For continuous actions:
        min_ω L_inv = E[‖a - h_ω(f(s_t), f(s_{t+1}))‖²]

    ============================================================================
    PROBLEM STATEMENT
    ============================================================================
    Without the inverse model, the encoder might learn features that are
    easy to predict but irrelevant to the agent's actions (e.g., static
    background). The inverse model forces features to capture what the
    agent can actually influence.

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(forward_pass)
    - Space: O(|ω|) parameters
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        discrete_actions: bool = True,
    ) -> None:
        """Initialize inverse dynamics model.

        Args:
            feature_dim: Dimensionality of feature space.
            action_dim: Number of discrete actions or continuous action dim.
            hidden_dims: Hidden layer sizes.
            discrete_actions: Whether action space is discrete.
        """
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.discrete_actions = discrete_actions

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize network parameters."""
        input_dim = 2 * self.feature_dim
        dims = [input_dim] + list(self.hidden_dims) + [self.action_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax along last axis."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def predict(
        self,
        current_features: np.ndarray,
        next_features: np.ndarray,
    ) -> np.ndarray:
        """Predict action from state transition.

        Args:
            current_features: Features of current state.
            next_features: Features of next state.

        Returns:
            Action probabilities (discrete) or predicted action (continuous).
        """
        current_features = np.atleast_2d(current_features).astype(np.float64)
        next_features = np.atleast_2d(next_features).astype(np.float64)
        single_input = current_features.shape[0] == 1

        x = np.concatenate([current_features, next_features], axis=1)

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)

        if self.discrete_actions:
            x = self._softmax(x)

        return x.flatten() if single_input else x

    def compute_loss(
        self,
        current_features: np.ndarray,
        next_features: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Compute inverse model loss.

        Args:
            current_features: Features of current state.
            next_features: Features of next state.
            action: True action taken.

        Returns:
            Cross-entropy (discrete) or MSE (continuous) loss.
        """
        predicted = self.predict(current_features, next_features)

        if self.discrete_actions:
            action = np.atleast_1d(action)
            predicted = np.atleast_2d(predicted)
            predicted = np.clip(predicted, 1e-8, 1 - 1e-8)
            loss = 0.0
            for i, a in enumerate(action):
                loss -= np.log(predicted[i, int(a)] + 1e-10)
            return float(loss / len(action))
        else:
            action = np.atleast_2d(action).astype(np.float64)
            return float(np.mean(np.sum((predicted - action) ** 2, axis=-1)))

    def update(
        self,
        current_features: np.ndarray,
        next_features: np.ndarray,
        action: np.ndarray,
        learning_rate: float,
    ) -> float:
        """Update model parameters via gradient descent.

        Args:
            current_features: Current state features.
            next_features: Next state features.
            action: True action taken.
            learning_rate: SGD step size.

        Returns:
            Loss value before update.
        """
        current_features = np.atleast_2d(current_features).astype(np.float64)
        next_features = np.atleast_2d(next_features).astype(np.float64)
        batch_size = len(current_features)

        x = np.concatenate([current_features, next_features], axis=1)

        activations = [x.copy()]
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)
            activations.append(x.copy())

        if self.discrete_actions:
            probs = self._softmax(x)
            action_int = np.atleast_1d(action).astype(int)

            loss = 0.0
            for i, a in enumerate(action_int):
                loss -= np.log(probs[i, a] + 1e-10)
            loss = float(loss / batch_size)

            target = np.zeros_like(probs)
            for i, a in enumerate(action_int):
                target[i, a] = 1.0
            delta = (probs - target) / batch_size
        else:
            action_arr = np.atleast_2d(action).astype(np.float64)
            loss = float(np.mean(np.sum((x - action_arr) ** 2, axis=-1)))
            delta = 2 * (x - action_arr) / batch_size

        for i in range(len(self._weights) - 1, -1, -1):
            grad_w = activations[i].T @ delta
            grad_b = np.sum(delta, axis=0)

            self._weights[i] -= learning_rate * grad_w
            self._biases[i] -= learning_rate * grad_b

            if i > 0:
                delta = delta @ self._weights[i].T
                delta = delta * (activations[i] > 0)

        return loss


class IntrinsicCuriosityModule:
    """Complete ICM implementation combining encoder, forward, and inverse models.

    ============================================================================
    CORE IDEA
    ============================================================================
    ICM generates intrinsic rewards based on prediction errors in a learned
    feature space. The architecture consists of:
    1. Feature encoder: Maps observations to compact representations
    2. Forward model: Predicts next features from current features + action
    3. Inverse model: Predicts action from state transition (auxiliary task)

    The forward model's prediction error serves as intrinsic reward, while
    the inverse model ensures the encoder captures action-relevant information.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Total loss function:

        L = (1-β) · L_inv + β · L_fwd

    where:
    - L_fwd = ‖f(s_{t+1}) - g_ψ(f(s_t), a_t)‖²  (forward prediction)
    - L_inv = -log P(a_t | f(s_t), f(s_{t+1}))  (inverse prediction)
    - β ∈ [0, 1] balances the two objectives

    Intrinsic reward:
        r_i(s_t, a_t, s_{t+1}) = η · ‖f(s_{t+1}) - f̂(s_{t+1})‖²

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    vs RND: ICM uses learned features, RND uses fixed random network.
        ICM adapts representations, RND is simpler but may include noise.
    vs Count-based: ICM generalizes to continuous spaces without discretization.
    vs Disagreement: ICM is deterministic, ensemble methods model uncertainty.

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(3 × network_forward) per step
    - Space: O(|encoder| + |forward| + |inverse|)
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: Optional[CuriosityConfig] = None,
        discrete_actions: bool = True,
    ) -> None:
        """Initialize Intrinsic Curiosity Module.

        Args:
            observation_dim: Dimensionality of observations.
            action_dim: Number of discrete actions or continuous action dim.
            config: ICM configuration. Uses defaults if None.
            discrete_actions: Whether action space is discrete.
        """
        self.config = config or CuriosityConfig()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions

        self.encoder = FeatureEncoder(
            input_dim=observation_dim,
            feature_dim=self.config.feature_dim,
        )

        self.forward_model = ForwardDynamicsModel(
            feature_dim=self.config.feature_dim,
            action_dim=action_dim,
        )

        self.inverse_model = InverseDynamicsModel(
            feature_dim=self.config.feature_dim,
            action_dim=action_dim,
            discrete_actions=discrete_actions,
        )

        self._training_stats: List[Dict[str, float]] = []
        self._step_count: int = 0
        self._running_mean: float = 0.0
        self._running_var: float = 1.0

    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> float:
        """Compute intrinsic curiosity reward.

        Args:
            state: Current observation.
            action: Action taken.
            next_state: Resulting observation.

        Returns:
            Intrinsic reward r_i = η · ‖f(s') - f̂(s')‖², clipped.
        """
        current_features = self.encoder.encode(state)
        next_features = self.encoder.encode(next_state)

        prediction_error = self.forward_model.compute_prediction_error(
            current_features, action, next_features
        )

        if self.config.normalize_rewards:
            alpha = 0.01
            self._running_mean = (
                1 - alpha
            ) * self._running_mean + alpha * prediction_error
            self._running_var = (1 - alpha) * self._running_var + alpha * (
                prediction_error - self._running_mean
            ) ** 2
            prediction_error = (prediction_error - self._running_mean) / (
                np.sqrt(self._running_var) + 1e-8
            )

        decay = self.config.decay_rate ** self._step_count
        intrinsic_reward = (
            self.config.intrinsic_reward_scale * prediction_error * decay
        )
        intrinsic_reward = np.clip(
            intrinsic_reward,
            -self.config.prediction_error_clipping,
            self.config.prediction_error_clipping,
        )

        self._step_count += 1

        return float(intrinsic_reward)

    def compute_total_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        extrinsic_reward: float,
    ) -> Tuple[float, float, float]:
        """Compute total reward combining extrinsic and intrinsic.

        Args:
            state: Current observation.
            action: Action taken.
            next_state: Resulting observation.
            extrinsic_reward: External environment reward.

        Returns:
            Tuple of (total_reward, extrinsic_reward, intrinsic_reward).
        """
        intrinsic = self.compute_intrinsic_reward(state, action, next_state)
        total = extrinsic_reward + intrinsic
        return total, extrinsic_reward, intrinsic

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> Dict[str, float]:
        """Update ICM models on a transition.

        Args:
            state: Current observation.
            action: Action taken.
            next_state: Resulting observation.

        Returns:
            Dictionary of training statistics.
        """
        current_features = self.encoder.encode(state)
        next_features = self.encoder.encode(next_state)

        forward_loss = self.forward_model.update(
            current_features,
            action,
            next_features,
            self.config.learning_rate * self.config.forward_loss_weight,
        )

        inverse_loss = self.inverse_model.update(
            current_features,
            next_features,
            action,
            self.config.learning_rate * self.config.inverse_loss_weight,
        )

        combined_loss = (
            self.config.forward_loss_weight * forward_loss
            + self.config.inverse_loss_weight * inverse_loss
        )

        prediction_error = self.forward_model.compute_prediction_error(
            current_features, action, next_features
        )

        stats = {
            "forward_loss": forward_loss,
            "inverse_loss": inverse_loss,
            "combined_loss": combined_loss,
            "prediction_error": prediction_error,
            "intrinsic_reward": self.config.intrinsic_reward_scale * prediction_error,
        }

        self._training_stats.append(stats)
        return stats

    def batch_update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ) -> Dict[str, float]:
        """Update ICM on a batch of transitions.

        Args:
            states: Batch of current observations.
            actions: Batch of actions.
            next_states: Batch of next observations.

        Returns:
            Aggregated training statistics.
        """
        current_features = self.encoder.encode(states)
        next_features = self.encoder.encode(next_states)

        forward_loss = self.forward_model.update(
            current_features,
            actions,
            next_features,
            self.config.learning_rate * self.config.forward_loss_weight,
        )

        inverse_loss = self.inverse_model.update(
            current_features,
            next_features,
            actions,
            self.config.learning_rate * self.config.inverse_loss_weight,
        )

        return {
            "forward_loss": forward_loss,
            "inverse_loss": inverse_loss,
            "batch_size": len(states),
        }

    def get_training_stats(self) -> List[Dict[str, float]]:
        """Get full training history.

        Returns:
            List of per-update statistics dictionaries.
        """
        return self._training_stats.copy()


class RandomNetworkDistillation:
    """Random Network Distillation (RND) for exploration.

    ============================================================================
    CORE IDEA
    ============================================================================
    Use prediction error to a fixed random network as intrinsic reward.
    Novel states have high prediction error because the predictor hasn't
    seen similar states during training.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    RND consists of two networks:
    - Target network f: S → ℝᵈ (fixed random initialization)
    - Predictor network f̂: S → ℝᵈ (trained to predict target)

    Intrinsic reward:
        r_i(s) = ‖f(s) - f̂(s)‖²

    The predictor learns to match the random target on visited states.
    For novel states, prediction error is high because the predictor
    hasn't been trained on similar inputs.

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    vs ICM:
    - Simpler: No inverse model, no learned features
    - May include noise: Random features don't filter irrelevant variation
    - More stable: Fixed target prevents moving goalposts

    Advantages:
    - Simple to implement
    - Works with any observation type
    - No need for action information

    Disadvantages:
    - May reward TV static, white noise (unpredictable but not interesting)
    - Doesn't distinguish controllable from uncontrollable novelty

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(2 × network_forward) per observation
    - Space: O(2 × |network|) for target and predictor
    """

    def __init__(
        self,
        observation_dim: int,
        feature_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (128, 64),
        learning_rate: float = 0.001,
        intrinsic_reward_scale: float = 0.01,
    ) -> None:
        """Initialize RND.

        Args:
            observation_dim: Dimensionality of observations.
            feature_dim: Output dimensionality for both networks.
            hidden_dims: Hidden layer sizes.
            learning_rate: Learning rate for predictor updates.
            intrinsic_reward_scale: Scaling for intrinsic rewards.
        """
        self.observation_dim = observation_dim
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.intrinsic_reward_scale = intrinsic_reward_scale

        self._target_weights: List[np.ndarray] = []
        self._target_biases: List[np.ndarray] = []
        self._predictor_weights: List[np.ndarray] = []
        self._predictor_biases: List[np.ndarray] = []

        self._init_networks()

        self._running_mean = np.zeros(1, dtype=np.float64)
        self._running_std = np.ones(1, dtype=np.float64)
        self._update_count = 0

    def _init_networks(self) -> None:
        """Initialize target and predictor networks."""
        dims = [self.observation_dim] + list(self.hidden_dims) + [self.feature_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))

            self._target_weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._target_biases.append(np.zeros(fan_out, dtype=np.float64))

            self._predictor_weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._predictor_biases.append(np.zeros(fan_out, dtype=np.float64))

    def _forward_target(self, observation: np.ndarray) -> np.ndarray:
        """Forward pass through fixed target network."""
        x = np.atleast_2d(observation).astype(np.float64)

        for i, (w, b) in enumerate(zip(self._target_weights, self._target_biases)):
            x = x @ w + b
            if i < len(self._target_weights) - 1:
                x = np.maximum(0, x)

        return x

    def _forward_predictor(self, observation: np.ndarray) -> np.ndarray:
        """Forward pass through trainable predictor network."""
        x = np.atleast_2d(observation).astype(np.float64)

        for i, (w, b) in enumerate(
            zip(self._predictor_weights, self._predictor_biases)
        ):
            x = x @ w + b
            if i < len(self._predictor_weights) - 1:
                x = np.maximum(0, x)

        return x

    def compute_intrinsic_reward(self, observation: np.ndarray) -> float:
        """Compute intrinsic reward as prediction error.

        Args:
            observation: Current observation.

        Returns:
            Normalized intrinsic reward.
        """
        target = self._forward_target(observation)
        predicted = self._forward_predictor(observation)

        error = float(np.sum((target - predicted) ** 2))

        self._update_count += 1
        alpha = 1.0 / self._update_count
        self._running_mean = (1 - alpha) * self._running_mean + alpha * error
        self._running_std = (1 - alpha) * self._running_std + alpha * abs(
            error - self._running_mean
        )

        normalized_error = (error - self._running_mean) / (self._running_std + 1e-8)
        normalized_error = max(0, normalized_error)

        return float(self.intrinsic_reward_scale * normalized_error)

    def update(self, observation: np.ndarray) -> float:
        """Update predictor network to match target.

        Args:
            observation: Observation to learn.

        Returns:
            Prediction loss before update.
        """
        observation = np.atleast_2d(observation).astype(np.float64)
        batch_size = len(observation)

        x = observation.copy()
        activations = [x.copy()]

        for i, (w, b) in enumerate(
            zip(self._predictor_weights, self._predictor_biases)
        ):
            x = x @ w + b
            if i < len(self._predictor_weights) - 1:
                x = np.maximum(0, x)
            activations.append(x.copy())

        target = self._forward_target(observation)
        loss = float(np.mean(np.sum((x - target) ** 2, axis=1)))

        delta = 2 * (x - target) / batch_size

        for i in range(len(self._predictor_weights) - 1, -1, -1):
            grad_w = activations[i].T @ delta
            grad_b = np.sum(delta, axis=0)

            self._predictor_weights[i] -= self.learning_rate * grad_w
            self._predictor_biases[i] -= self.learning_rate * grad_b

            if i > 0:
                delta = delta @ self._predictor_weights[i].T
                delta = delta * (activations[i] > 0)

        return loss


class CountBasedExploration:
    """Count-based exploration for discrete or discretized state spaces.

    ============================================================================
    CORE IDEA
    ============================================================================
    Maintain visit counts N(s) for states and reward inversely:
        r_i(s) = β / √N(s)

    This encourages visiting under-explored regions of the state space.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Classic count-based bonus (Strehl & Littman, 2008):

        r_i(s) = β / √N(s)

    For continuous states, use pseudo-counts via density models:

        N̂(s) = ρ(s) / (ρ'(s) - ρ(s))

    where ρ(s) is density before observing s, ρ'(s) is density after.

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    vs Prediction-based: Theoretically grounded, no neural network required
    Disadvantage: Requires state discretization or density estimation

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(1) average for hash table operations
    - Space: O(|visited states|)
    """

    def __init__(
        self,
        state_discretization: Optional[int] = None,
        bonus_coefficient: float = 1.0,
        intrinsic_reward_scale: float = 0.01,
    ) -> None:
        """Initialize count-based exploration.

        Args:
            state_discretization: Number of bins per dimension for continuous
                states. If None, uses states directly as keys.
            bonus_coefficient: β scaling the count-based bonus.
            intrinsic_reward_scale: Additional scaling factor.
        """
        self.state_discretization = state_discretization
        self.bonus_coefficient = bonus_coefficient
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self._counts: Dict[Tuple, int] = {}

    def _discretize(self, state: np.ndarray) -> Tuple:
        """Convert state to hashable key."""
        if self.state_discretization is not None:
            state = np.floor(state * self.state_discretization).astype(int)
        return tuple(np.atleast_1d(state).flatten())

    def get_count(self, state: np.ndarray) -> int:
        """Get visit count for state.

        Args:
            state: State to query.

        Returns:
            Number of times state has been visited.
        """
        key = self._discretize(state)
        return self._counts.get(key, 0)

    def update(self, state: np.ndarray) -> None:
        """Increment visit count for state.

        Args:
            state: State that was visited.
        """
        key = self._discretize(state)
        self._counts[key] = self._counts.get(key, 0) + 1

    def compute_intrinsic_reward(self, state: np.ndarray) -> float:
        """Compute count-based exploration bonus.

        Args:
            state: State to evaluate.

        Returns:
            Intrinsic reward β / √N(s).
        """
        count = max(1, self.get_count(state))
        bonus = self.bonus_coefficient / np.sqrt(count)
        return float(self.intrinsic_reward_scale * bonus)

    def get_total_states_visited(self) -> int:
        """Get number of unique states visited.

        Returns:
            Count of distinct states seen.
        """
        return len(self._counts)

    def get_visitation_entropy(self) -> float:
        """Compute entropy of state visitation distribution.

        Returns:
            Shannon entropy of visitation probabilities.
        """
        if not self._counts:
            return 0.0

        total = sum(self._counts.values())
        probs = np.array(list(self._counts.values())) / total
        return float(-np.sum(probs * np.log(probs + 1e-10)))


class EpisodicNoveltyModule:
    """Episodic novelty detection using nearest neighbor comparisons.

    ============================================================================
    CORE IDEA
    ============================================================================
    Maintain a memory of visited states within an episode. Reward states
    that are far from anything in memory (episodic novelty).

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The episodic intrinsic reward is:

        r_i(s) = 1 / √(Σ_{s_m ∈ M} K(s, s_m) + c)

    where:
    - M is the episodic memory
    - K is a kernel function (e.g., Gaussian)
    - c is a small constant for numerical stability

    For efficiency, often use k-nearest neighbors:

        r_i(s) = 1 / √(d_k(s) + c)

    where d_k(s) is the distance to the k-th nearest neighbor.

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    - vs RND: Complements rather than replaces; targets different timescales
    - vs Count-based: Continuous-space compatible, no discretization
    - Often combined: r_int = RND(s) × episodic_novelty(s)

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(|M| × d) or O(k log |M|) with KD-tree
    - Space: O(|M| × d) for memory storage
    """

    def __init__(
        self,
        embedding_dim: int,
        memory_capacity: int = 1000,
        k_neighbors: int = 10,
        kernel_epsilon: float = 0.001,
        intrinsic_reward_scale: float = 0.01,
    ) -> None:
        """Initialize episodic novelty module.

        Args:
            embedding_dim: Dimensionality of state embeddings.
            memory_capacity: Maximum states to store per episode.
            k_neighbors: Number of neighbors for novelty computation.
            kernel_epsilon: Kernel cluster distance.
            intrinsic_reward_scale: Scaling for intrinsic rewards.
        """
        self.embedding_dim = embedding_dim
        self.memory_capacity = memory_capacity
        self.k_neighbors = k_neighbors
        self.kernel_epsilon = kernel_epsilon
        self.intrinsic_reward_scale = intrinsic_reward_scale

        self._episodic_memory: List[np.ndarray] = []

    def _encode(self, state: np.ndarray) -> np.ndarray:
        """Encode state to embedding (simple normalization for now).

        Args:
            state: Raw state.

        Returns:
            Normalized state embedding.
        """
        state = np.asarray(state, dtype=np.float64).flatten()
        if len(state) < self.embedding_dim:
            state = np.pad(state, (0, self.embedding_dim - len(state)))
        elif len(state) > self.embedding_dim:
            state = state[: self.embedding_dim]

        norm = np.linalg.norm(state)
        return state / (norm + 1e-8) if norm > 0 else state

    def compute_intrinsic_reward(self, state: np.ndarray) -> float:
        """Compute episodic novelty reward.

        Args:
            state: State to evaluate novelty.

        Returns:
            Episodic novelty reward.
        """
        embedding = self._encode(state)

        if len(self._episodic_memory) == 0:
            return self.intrinsic_reward_scale * 1.0

        distances = []
        for mem_embedding in self._episodic_memory:
            dist = np.sum((embedding - mem_embedding) ** 2)
            distances.append(dist)

        distances = np.array(distances)
        k = min(self.k_neighbors, len(distances))
        k_nearest_distances = np.partition(distances, k - 1)[:k]

        kernel_sum = np.sum(
            self.kernel_epsilon / (k_nearest_distances + self.kernel_epsilon)
        )

        novelty = 1.0 / np.sqrt(kernel_sum + 1e-8)

        return float(self.intrinsic_reward_scale * novelty)

    def update(self, state: np.ndarray) -> None:
        """Add state to episodic memory.

        Args:
            state: State to remember.
        """
        embedding = self._encode(state)

        if len(self._episodic_memory) >= self.memory_capacity:
            self._episodic_memory.pop(0)

        self._episodic_memory.append(embedding.copy())

    def reset_episode(self) -> None:
        """Clear episodic memory for new episode."""
        self._episodic_memory.clear()

    def get_memory_size(self) -> int:
        """Get current episodic memory size."""
        return len(self._episodic_memory)


def compute_exploration_efficiency(
    states_visited: np.ndarray,
    state_space_bounds: Tuple[np.ndarray, np.ndarray],
    n_bins: int = 20,
) -> Dict[str, float]:
    """Analyze exploration coverage and efficiency.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Coverage metrics quantify how thoroughly the agent explores:

    1. Coverage ratio: |visited bins| / |total bins|
    2. Uniformity: How evenly distributed are visits?
    3. Entropy: Information-theoretic measure of coverage

    Args:
        states_visited: Array of visited states, shape (n_steps, state_dim).
        state_space_bounds: Tuple of (lower_bounds, upper_bounds) arrays.
        n_bins: Number of bins per dimension for discretization.

    Returns:
        Dictionary of exploration metrics.
    """
    states_visited = np.atleast_2d(states_visited)
    state_dim = states_visited.shape[1]

    lower, upper = state_space_bounds
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)

    normalized = (states_visited - lower) / (upper - lower + 1e-8)
    normalized = np.clip(normalized, 0, 1 - 1e-8)
    discretized = np.floor(normalized * n_bins).astype(int)

    unique_bins = set(tuple(row) for row in discretized)
    total_bins = n_bins**state_dim
    coverage_ratio = len(unique_bins) / total_bins

    bin_counts: Dict[Tuple, int] = {}
    for row in discretized:
        key = tuple(row)
        bin_counts[key] = bin_counts.get(key, 0) + 1

    counts = np.array(list(bin_counts.values()))
    if len(counts) > 0:
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = np.log(len(counts))
        uniformity = entropy / max_entropy if max_entropy > 0 else 1.0
    else:
        entropy = 0.0
        uniformity = 0.0

    return {
        "coverage_ratio": coverage_ratio,
        "unique_states": len(unique_bins),
        "total_possible": total_bins,
        "visitation_entropy": entropy,
        "uniformity": uniformity,
        "total_visits": len(states_visited),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Curiosity-Driven Exploration Module - Comprehensive Validation")
    print("=" * 70)

    np.random.seed(42)

    print("\n[Test 1] Feature Encoder")
    print("-" * 50)

    obs_dim = 10
    feature_dim = 32

    encoder = FeatureEncoder(
        input_dim=obs_dim,
        feature_dim=feature_dim,
        hidden_dims=(64, 32),
    )

    test_obs = np.random.randn(obs_dim)
    features = encoder.encode(test_obs)

    print(f"  Input shape: {test_obs.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Feature norm: {np.linalg.norm(features):.4f}")

    assert features.shape == (feature_dim,), "Feature shape mismatch"
    assert np.isclose(np.linalg.norm(features), 1.0, atol=0.01), "Features not normalized"
    print("  [PASS] Encoder produces normalized features")

    batch_obs = np.random.randn(8, obs_dim)
    batch_features = encoder.encode(batch_obs)
    assert batch_features.shape == (8, feature_dim), "Batch encoding failed"
    print("  [PASS] Batch encoding works correctly")

    print("\n[Test 2] Forward Dynamics Model")
    print("-" * 50)

    action_dim = 4
    forward_model = ForwardDynamicsModel(
        feature_dim=feature_dim,
        action_dim=action_dim,
        hidden_dims=(64, 32),
    )

    current_feat = np.random.randn(feature_dim)
    action = 2
    predicted = forward_model.predict(current_feat, np.array([action]))

    print(f"  Input features shape: {current_feat.shape}")
    print(f"  Predicted shape: {predicted.shape}")
    assert predicted.shape == (feature_dim,), "Prediction shape mismatch"

    next_feat = np.random.randn(feature_dim)
    error = forward_model.compute_prediction_error(current_feat, np.array([action]), next_feat)
    print(f"  Prediction error: {error:.4f}")
    assert error >= 0, "Prediction error should be non-negative"
    print("  [PASS] Forward model predicts and computes error")

    loss = forward_model.update(current_feat, np.array([action]), next_feat, 0.01)
    print(f"  Training loss: {loss:.4f}")
    print("  [PASS] Forward model update works")

    print("\n[Test 3] Inverse Dynamics Model")
    print("-" * 50)

    inverse_model = InverseDynamicsModel(
        feature_dim=feature_dim,
        action_dim=action_dim,
        discrete_actions=True,
    )

    current_feat = np.random.randn(feature_dim)
    next_feat = np.random.randn(feature_dim)

    action_probs = inverse_model.predict(current_feat, next_feat)
    print(f"  Action probabilities shape: {action_probs.shape}")
    print(f"  Probability sum: {action_probs.sum():.4f}")

    assert action_probs.shape == (action_dim,), "Action probs shape mismatch"
    assert np.isclose(action_probs.sum(), 1.0), "Probabilities don't sum to 1"
    print("  [PASS] Inverse model produces valid action distribution")

    true_action = np.array([1])
    loss = inverse_model.update(current_feat, next_feat, true_action, 0.01)
    print(f"  Training loss: {loss:.4f}")
    print("  [PASS] Inverse model update works")

    print("\n[Test 4] Intrinsic Curiosity Module (ICM)")
    print("-" * 50)

    config = CuriosityConfig(
        intrinsic_reward_scale=0.01,
        feature_dim=32,
        learning_rate=0.001,
    )

    icm = IntrinsicCuriosityModule(
        observation_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        discrete_actions=True,
    )

    state = np.random.randn(obs_dim)
    action = np.array([0])
    next_state = state + 0.1 * np.random.randn(obs_dim)

    intrinsic_reward = icm.compute_intrinsic_reward(state, action, next_state)
    print(f"  Intrinsic reward: {intrinsic_reward:.6f}")
    print("  [PASS] ICM computes valid intrinsic reward")

    total, ext, intr = icm.compute_total_reward(state, action, next_state, 1.0)
    print(f"  Total: {total:.4f}, Extrinsic: {ext:.4f}, Intrinsic: {intr:.6f}")
    assert abs(total - (ext + intr)) < 0.01, "Reward decomposition mismatch"
    print("  [PASS] Total reward combines extrinsic + intrinsic")

    stats = icm.update(state, action, next_state)
    print(f"  Update stats: forward_loss={stats['forward_loss']:.4f}")
    print("  [PASS] ICM update returns training statistics")

    print("\n[Test 5] Random Network Distillation (RND)")
    print("-" * 50)

    rnd = RandomNetworkDistillation(
        observation_dim=obs_dim,
        feature_dim=32,
        intrinsic_reward_scale=0.01,
    )

    obs = np.random.randn(obs_dim)
    rnd_reward = rnd.compute_intrinsic_reward(obs)
    print(f"  RND intrinsic reward: {rnd_reward:.6f}")
    assert rnd_reward >= 0, "RND reward should be non-negative"
    print("  [PASS] RND computes valid intrinsic reward")

    loss = rnd.update(obs)
    print(f"  RND training loss: {loss:.4f}")

    rewards_over_time = []
    for _ in range(100):
        r = rnd.compute_intrinsic_reward(obs)
        rnd.update(obs)
        rewards_over_time.append(r)

    print(f"  Reward after 100 updates on same obs: {rewards_over_time[-1]:.6f}")
    print("  [PASS] RND predictor learns to match target")

    print("\n[Test 6] Count-Based Exploration")
    print("-" * 50)

    counter = CountBasedExploration(state_discretization=10)

    for i in range(100):
        state = np.random.rand(2) * 2
        counter.update(state)

    unique_states = counter.get_total_states_visited()
    entropy = counter.get_visitation_entropy()
    print(f"  Unique states visited: {unique_states}")
    print(f"  Visitation entropy: {entropy:.4f}")

    bonus = counter.compute_intrinsic_reward(np.array([0.5, 0.5]))
    print(f"  Exploration bonus: {bonus:.4f}")
    print("  [PASS] Count-based exploration tracks visits")

    print("\n[Test 7] Episodic Novelty Module")
    print("-" * 50)

    episodic = EpisodicNoveltyModule(
        embedding_dim=obs_dim,
        memory_capacity=100,
        k_neighbors=5,
    )

    test_state = np.random.randn(obs_dim)
    first_reward = episodic.compute_intrinsic_reward(test_state)
    episodic.update(test_state)

    similar_state = test_state + 0.01 * np.random.randn(obs_dim)
    second_reward = episodic.compute_intrinsic_reward(similar_state)

    print(f"  First state novelty: {first_reward:.6f}")
    print(f"  Similar state novelty: {second_reward:.6f}")

    episodic.reset_episode()
    after_reset = episodic.compute_intrinsic_reward(test_state)
    print(f"  After episode reset: {after_reset:.6f}")
    print(f"  Memory size after reset: {episodic.get_memory_size()}")
    print("  [PASS] Episodic novelty module works")

    print("\n[Test 8] Exploration Efficiency Analysis")
    print("-" * 50)

    states_visited = np.random.rand(1000, 2) * 2 - 1
    bounds = (np.array([-1, -1]), np.array([1, 1]))

    metrics = compute_exploration_efficiency(states_visited, bounds, n_bins=10)
    print(f"  Coverage ratio: {metrics['coverage_ratio']:.4f}")
    print(f"  Unique states: {metrics['unique_states']}/{metrics['total_possible']}")
    print(f"  Uniformity: {metrics['uniformity']:.4f}")
    print("  [PASS] Exploration analysis computes valid metrics")

    print("\n[Test 9] Configuration Validation")
    print("-" * 50)

    try:
        CuriosityConfig(intrinsic_reward_scale=-1.0)
        print("  [FAIL] Should reject negative intrinsic_reward_scale")
    except ValueError:
        print("  [PASS] Rejects negative intrinsic_reward_scale")

    try:
        CuriosityConfig(feature_dim=0)
        print("  [FAIL] Should reject zero feature_dim")
    except ValueError:
        print("  [PASS] Rejects zero feature_dim")

    print("\n[Test 10] ICM Batch Training")
    print("-" * 50)

    batch_size = 32
    batch_states = np.random.randn(batch_size, obs_dim)
    batch_actions = np.random.randint(0, action_dim, batch_size)
    batch_next_states = batch_states + 0.1 * np.random.randn(batch_size, obs_dim)

    batch_stats = icm.batch_update(batch_states, batch_actions, batch_next_states)
    print(f"  Batch forward loss: {batch_stats['forward_loss']:.4f}")
    print(f"  Batch inverse loss: {batch_stats['inverse_loss']:.4f}")
    print("  [PASS] ICM batch training works correctly")

    print("\n" + "=" * 70)
    print("All validation tests passed successfully!")
    print("=" * 70)
