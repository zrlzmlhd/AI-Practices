"""
Neural Network Architectures for Continuous Control RL
======================================================

Core Idea
---------
Modular neural network components for actor-critic architectures:

- **Actor Networks**: Map states to actions (deterministic or stochastic)
- **Critic Networks**: Estimate action-value Q(s,a) or state-value V(s)

Mathematical Theory
-------------------
**Deterministic Policy Gradient (DPG)**:

.. math::

    \\nabla_\\theta J(\\theta) = \\mathbb{E}_{s \\sim \\rho^\\mu}
    [\\nabla_a Q(s,a)|_{a=\\mu(s)} \\nabla_\\theta \\mu_\\theta(s)]

**Stochastic Policy (Maximum Entropy)**:

.. math::

    \\pi_\\theta(a|s) = \\mathcal{N}(\\mu_\\theta(s), \\sigma_\\theta(s))

with reparameterization:

.. math::

    a = \\tanh(\\mu + \\sigma \\cdot \\epsilon), \\quad \\epsilon \\sim \\mathcal{N}(0, I)

Problem Statement
-----------------
Deep RL requires carefully designed neural networks that:

1. **Stable Gradients**: Avoid vanishing/exploding gradients
2. **Proper Initialization**: Enable effective learning from the start
3. **Bounded Outputs**: Respect action space constraints

This module provides production-ready implementations with orthogonal
initialization, proper activation functions, and bounded action outputs.

Summary
-------
- DeterministicActor: For DDPG/TD3 - outputs bounded deterministic actions
- GaussianActor: For SAC - outputs mean and log_std for stochastic policy
- QNetwork: Single Q-value estimator
- TwinQNetwork: Double Q-learning (min of two Q-values)
- ValueNetwork: State value V(s) estimator (optional for SAC)
"""

from typing import List, Tuple, Optional, Sequence
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# =============================================================================
# Initialization Utilities
# =============================================================================


def orthogonal_init(
    layer: nn.Linear,
    gain: float = 1.0,
    bias_const: float = 0.0,
) -> nn.Linear:
    """
    Orthogonal Weight Initialization
    =================================

    Core Idea
    ---------
    Initialize weights as orthogonal matrices scaled by gain factor.
    Preserves gradient magnitude through deep networks.

    Mathematical Theory
    -------------------
    For weight matrix :math:`W`, orthogonal initialization satisfies:

    .. math::

        W^T W = I \\cdot \\text{gain}^2

    This ensures that during forward pass:

    .. math::

        ||W x||_2 = \\text{gain} \\cdot ||x||_2

    preventing exponential gradient growth/decay in deep networks.

    For actor networks, gain=0.01 produces near-zero initial actions,
    enabling stable early exploration.

    Parameters
    ----------
    layer : nn.Linear
        Linear layer to initialize.
    gain : float, default=1.0
        Scaling factor for weight magnitude.
        - 1.0: Standard orthogonal (hidden layers)
        - 0.01: Near-zero output (final actor layer)
    bias_const : float, default=0.0
        Constant value for bias initialization.

    Returns
    -------
    nn.Linear
        Initialized layer (modified in-place).

    References
    ----------
    .. [1] Saxe et al. (2013). "Exact solutions to the nonlinear dynamics
           of learning in deep linear neural networks"
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def create_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: nn.Module = nn.ReLU,
    output_activation: Optional[nn.Module] = None,
    use_orthogonal: bool = True,
    final_gain: float = 1.0,
) -> nn.Sequential:
    """
    Create Multi-Layer Perceptron
    =============================

    Factory function for creating feedforward networks with consistent
    architecture and initialization.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : Sequence[int]
        Sizes of hidden layers.
    output_dim : int
        Output dimension.
    activation : nn.Module, default=nn.ReLU
        Activation function class for hidden layers.
    output_activation : nn.Module, optional
        Activation for output layer. None for linear output.
    use_orthogonal : bool, default=True
        Whether to use orthogonal initialization.
    final_gain : float, default=1.0
        Gain for final layer initialization.

    Returns
    -------
    nn.Sequential
        Constructed MLP network.

    Example
    -------
    >>> mlp = create_mlp(4, [256, 256], 2)
    >>> x = torch.randn(32, 4)
    >>> y = mlp(x)  # Shape: (32, 2)
    """
    layers: List[nn.Module] = []
    dims = [input_dim] + list(hidden_dims)

    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        if use_orthogonal:
            orthogonal_init(linear, gain=math.sqrt(2))
        layers.append(linear)
        layers.append(activation())

    # Output layer
    output_layer = nn.Linear(dims[-1], output_dim)
    if use_orthogonal:
        orthogonal_init(output_layer, gain=final_gain)
    layers.append(output_layer)

    if output_activation is not None:
        layers.append(output_activation())

    return nn.Sequential(*layers)


# =============================================================================
# Actor Networks
# =============================================================================


class DeterministicActor(nn.Module):
    """
    Deterministic Policy Network
    ============================

    Core Idea
    ---------
    Maps state directly to a single action: :math:`a = \\mu_\\theta(s)`.
    Used in DDPG and TD3 where exploration is handled externally via noise.

    Mathematical Theory
    -------------------
    The deterministic policy :math:`\\mu_\\theta: \\mathcal{S} \\to \\mathcal{A}`
    outputs bounded actions via tanh squashing:

    .. math::

        a = a_{max} \\cdot \\tanh(\\text{MLP}(s))

    The policy gradient uses the chain rule through the Q-function:

    .. math::

        \\nabla_\\theta J = \\mathbb{E}_s[\\nabla_a Q(s,a)|_{a=\\mu(s)}
        \\cdot \\nabla_\\theta \\mu_\\theta(s)]

    Problem Statement
    -----------------
    Continuous control requires:

    1. Actions within physical bounds (e.g., motor torques)
    2. Smooth policy for stable learning
    3. Expressive function approximation

    DeterministicActor uses tanh output with MLP backbone to satisfy all three.

    Algorithm Comparison
    --------------------
    vs. GaussianActor (SAC):

    - Deterministic: Simpler, faster inference, external exploration
    - Gaussian: Built-in exploration, entropy regularization, more stable

    Complexity
    ----------
    - Forward: O(d × h × L) where d=input_dim, h=hidden_dim, L=num_layers
    - Parameters: O(d×h + h² × (L-1) + h×action_dim)

    Parameters
    ----------
    state_dim : int
        Observation space dimension.
    action_dim : int
        Action space dimension.
    max_action : float
        Maximum action magnitude for output scaling.
    hidden_dims : List[int], default=[256, 256]
        Hidden layer sizes.

    Attributes
    ----------
    network : nn.Sequential
        MLP backbone.
    max_action : float
        Action scaling factor.

    Example
    -------
    >>> actor = DeterministicActor(state_dim=11, action_dim=3, max_action=1.0)
    >>> state = torch.randn(32, 11)
    >>> action = actor(state)
    >>> action.shape
    torch.Size([32, 3])
    >>> (action.abs() <= 1.0).all()
    True

    References
    ----------
    .. [1] Lillicrap et al. (2016). "Continuous control with deep
           reinforcement learning". ICLR.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.max_action = max_action

        # Build network with orthogonal init, small final layer gain
        self.network = create_mlp(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activation=nn.ReLU,
            output_activation=None,
            use_orthogonal=True,
            final_gain=0.01,  # Small initial actions for stable exploration
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute deterministic action.

        Parameters
        ----------
        state : torch.Tensor
            Batch of states, shape (batch_size, state_dim).

        Returns
        -------
        torch.Tensor
            Bounded actions, shape (batch_size, action_dim).
            Values in [-max_action, max_action].
        """
        return self.max_action * torch.tanh(self.network(state))


class GaussianActor(nn.Module):
    """
    Stochastic Gaussian Policy Network
    ===================================

    Core Idea
    ---------
    Outputs a Gaussian distribution over actions parameterized by
    mean :math:`\\mu(s)` and log standard deviation :math:`\\log\\sigma(s)`.
    Enables principled exploration through entropy maximization (SAC).

    Mathematical Theory
    -------------------
    **Policy Distribution**:

    .. math::

        \\pi_\\theta(a|s) = \\mathcal{N}(a; \\mu_\\theta(s), \\sigma_\\theta(s)^2 I)

    **Reparameterization Trick** (for backpropagation through sampling):

    .. math::

        a = \\tanh(\\mu_\\theta(s) + \\sigma_\\theta(s) \\odot \\epsilon),
        \\quad \\epsilon \\sim \\mathcal{N}(0, I)

    **Log Probability with Tanh Correction**:

    The tanh squashing changes the distribution. The corrected log-prob:

    .. math::

        \\log \\pi(a|s) = \\log \\mathcal{N}(u; \\mu, \\sigma^2)
        - \\sum_{i=1}^{d} \\log(1 - \\tanh^2(u_i))

    where :math:`u` is the pre-tanh sample and :math:`a = \\tanh(u) \\cdot a_{max}`.

    Problem Statement
    -----------------
    Maximum entropy RL requires:

    1. Stochastic policy for exploration
    2. Differentiable sampling (reparameterization)
    3. Correct log-probability computation under action bounds

    GaussianActor implements all three with numerical stability safeguards.

    Algorithm Comparison
    --------------------
    vs. DeterministicActor:

    - Gaussian: Built-in exploration, entropy bonus, more robust
    - Deterministic: Simpler, requires external noise (Ornstein-Uhlenbeck)

    vs. Beta Distribution Policy:

    - Gaussian + tanh: Simple, well-understood, slightly biased at bounds
    - Beta: Naturally bounded, but less common in practice

    Complexity
    ----------
    - Forward: O(d × h × L) - Same as deterministic
    - Sample: Additional O(action_dim) for reparameterization
    - Parameters: ~2× deterministic (separate mean/std outputs)

    Parameters
    ----------
    state_dim : int
        Observation space dimension.
    action_dim : int
        Action space dimension.
    max_action : float
        Maximum action magnitude.
    hidden_dims : List[int], default=[256, 256]
        Hidden layer sizes.
    log_std_min : float, default=-20
        Minimum log standard deviation (numerical stability).
    log_std_max : float, default=2
        Maximum log standard deviation (prevents excessive exploration).

    Attributes
    ----------
    shared_net : nn.Sequential
        Shared feature extraction layers.
    mean_layer : nn.Linear
        Output layer for mean.
    log_std_layer : nn.Linear
        Output layer for log standard deviation.

    Example
    -------
    >>> actor = GaussianActor(state_dim=11, action_dim=3, max_action=1.0)
    >>> state = torch.randn(32, 11)
    >>> action, log_prob = actor.sample(state)
    >>> action.shape
    torch.Size([32, 3])
    >>> log_prob.shape
    torch.Size([32, 1])

    References
    ----------
    .. [1] Haarnoja et al. (2018). "Soft Actor-Critic: Off-Policy Maximum
           Entropy Deep Reinforcement Learning". ICML.
    """

    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0
    EPS: float = 1e-6  # Numerical stability constant

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        hidden_dims: List[int] = None,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim

        # Shared feature extraction
        layers: List[nn.Module] = []
        dims = [state_dim] + list(hidden_dims)

        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            orthogonal_init(linear, gain=math.sqrt(2))
            layers.append(linear)
            layers.append(nn.ReLU())

        self.shared_net = nn.Sequential(*layers)

        # Separate heads for mean and log_std
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        # Initialize with small weights for stable initial policy
        orthogonal_init(self.mean_layer, gain=0.01)
        orthogonal_init(self.log_std_layer, gain=0.01)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distribution parameters.

        Parameters
        ----------
        state : torch.Tensor
            Batch of states.

        Returns
        -------
        mean : torch.Tensor
            Mean of Gaussian, shape (batch_size, action_dim).
        log_std : torch.Tensor
            Log standard deviation, clamped to [log_std_min, log_std_max].
        """
        features = self.shared_net(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick.

        Parameters
        ----------
        state : torch.Tensor
            Batch of states.
        deterministic : bool, default=False
            If True, return mean action without sampling.

        Returns
        -------
        action : torch.Tensor
            Sampled (or deterministic) bounded action.
        log_prob : torch.Tensor
            Log probability of sampled action, shape (batch_size, 1).
            Includes tanh correction for proper probability under bounds.

        Notes
        -----
        The reparameterization trick enables gradient flow through sampling:

        .. math::

            a = \\tanh(\\mu + \\sigma \\cdot \\epsilon), \\quad
            \\epsilon \\sim \\mathcal{N}(0, I)

        This allows backpropagation through the expectation.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            # Evaluation mode: use mean action
            action = torch.tanh(mean) * self.max_action
            # Log prob undefined for deterministic, return zeros
            return action, torch.zeros(state.shape[0], 1, device=state.device)

        # Reparameterization: sample from N(0,1) then transform
        normal = Normal(mean, std)
        u = normal.rsample()  # rsample enables gradient flow

        # Squash through tanh
        action = torch.tanh(u) * self.max_action

        # Compute log probability with tanh correction
        log_prob = normal.log_prob(u)

        # Correction for tanh squashing (Jacobian adjustment)
        # log(1 - tanh^2(u)) = log(1 - (action/max_action)^2)
        log_prob -= torch.log(
            self.max_action * (1 - (action / self.max_action).pow(2)) + self.EPS
        )

        # Sum over action dimensions
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given action.

        Used during policy update when computing the entropy bonus.

        Parameters
        ----------
        state : torch.Tensor
            Batch of states.
        action : torch.Tensor
            Batch of actions to evaluate.

        Returns
        -------
        log_prob : torch.Tensor
            Log probability of actions.
        entropy : torch.Tensor
            Entropy of the policy distribution.
        mean : torch.Tensor
            Mean action (for logging).
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)

        # Inverse tanh to get pre-squashed action
        # Clamp to avoid numerical issues at boundaries
        action_normalized = action / self.max_action
        action_normalized = torch.clamp(action_normalized, -1 + self.EPS, 1 - self.EPS)
        u = torch.atanh(action_normalized)

        log_prob = normal.log_prob(u)
        log_prob -= torch.log(self.max_action * (1 - action_normalized.pow(2)) + self.EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Entropy of Gaussian: 0.5 * log(2 * pi * e * sigma^2) per dimension
        entropy = normal.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, mean


# =============================================================================
# Critic Networks
# =============================================================================


class QNetwork(nn.Module):
    """
    Action-Value (Q) Network
    ========================

    Core Idea
    ---------
    Estimates the expected return when taking action a in state s and
    following policy thereafter: :math:`Q(s, a) = \\mathbb{E}[G_t | s_t=s, a_t=a]`

    Mathematical Theory
    -------------------
    The Q-function satisfies the Bellman equation:

    .. math::

        Q^\\pi(s, a) = r(s, a) + \\gamma \\mathbb{E}_{s', a'}[Q^\\pi(s', a')]

    We minimize the temporal difference (TD) error:

    .. math::

        L(\\phi) = \\mathbb{E}[(Q_\\phi(s, a) - y)^2]

    where the target is:

    .. math::

        y = r + \\gamma Q_{\\phi'}(s', \\mu_{\\theta'}(s'))

    Problem Statement
    -----------------
    Value function approximation requires:

    1. Joint state-action encoding
    2. Accurate value prediction across state space
    3. Stable training targets (addressed by target networks)

    QNetwork concatenates state and action as input, enabling the critic
    to learn action preferences conditioned on state.

    Algorithm Comparison
    --------------------
    Single Q-Network (DDPG):

    - Prone to overestimation bias
    - Simpler architecture

    Twin Q-Networks (TD3/SAC):

    - Uses min(Q1, Q2) to reduce overestimation
    - More parameters but significantly more stable

    Complexity
    ----------
    - Forward: O((state_dim + action_dim) × h × L)
    - Parameters: O((s+a)×h + h² × (L-1) + h)

    Parameters
    ----------
    state_dim : int
        State space dimension.
    action_dim : int
        Action space dimension.
    hidden_dims : List[int], default=[256, 256]
        Hidden layer sizes.

    Example
    -------
    >>> q_net = QNetwork(state_dim=11, action_dim=3)
    >>> state = torch.randn(32, 11)
    >>> action = torch.randn(32, 3)
    >>> q_value = q_net(state, action)
    >>> q_value.shape
    torch.Size([32, 1])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.network = create_mlp(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=nn.ReLU,
            use_orthogonal=True,
            final_gain=1.0,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-value for state-action pair.

        Parameters
        ----------
        state : torch.Tensor
            Batch of states, shape (batch_size, state_dim).
        action : torch.Tensor
            Batch of actions, shape (batch_size, action_dim).

        Returns
        -------
        torch.Tensor
            Q-values, shape (batch_size, 1).
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class TwinQNetwork(nn.Module):
    """
    Twin Q-Networks for Double Q-Learning
    =====================================

    Core Idea
    ---------
    Two independent Q-networks that provide pessimistic value estimates
    via :math:`\\min(Q_1, Q_2)`, addressing Q-value overestimation.

    Mathematical Theory
    -------------------
    **Overestimation Problem**:

    Standard Q-learning maximizes over noisy value estimates:

    .. math::

        \\max_a Q(s', a) \\geq Q^*(s', a^*)

    due to Jensen's inequality. Error accumulates through bootstrapping.

    **Clipped Double Q-Learning** (TD3/SAC):

    Use two independent networks and take the minimum:

    .. math::

        y = r + \\gamma \\min_{i=1,2} Q_{\\phi'_i}(s', a')

    This provides a lower bound that counteracts overestimation.

    **Theoretical Justification**:

    If :math:`Q_1, Q_2` have independent, zero-mean errors:

    .. math::

        \\mathbb{E}[\\min(Q_1, Q_2)] \\leq \\mathbb{E}[Q^*]

    The bias is negative (underestimation), which is safer than
    overestimation that causes policy divergence.

    Problem Statement
    -----------------
    Q-learning with function approximation suffers from:

    1. Maximization bias (always picking optimistic estimates)
    2. Error propagation through temporal difference bootstrapping
    3. Policy overfitting to inaccurate Q-values

    TwinQNetwork addresses all three with independent, pessimistic critics.

    Algorithm Comparison
    --------------------
    Used by:

    - TD3: min(Q1, Q2) for target value computation
    - SAC: min(Q1, Q2) - α log π(a|s) for soft value target

    Not used by:

    - DDPG: Single Q-network (prone to overestimation)

    Complexity
    ----------
    - Forward: 2× single Q-network
    - Parameters: 2× (same architecture duplicated)

    Parameters
    ----------
    state_dim : int
        State space dimension.
    action_dim : int
        Action space dimension.
    hidden_dims : List[int], default=[256, 256]
        Hidden layer sizes.

    Example
    -------
    >>> twin_q = TwinQNetwork(state_dim=11, action_dim=3)
    >>> state = torch.randn(32, 11)
    >>> action = torch.randn(32, 3)
    >>> q1, q2 = twin_q(state, action)
    >>> q_min = twin_q.min_q(state, action)
    >>> (q_min == torch.min(q1, q2)).all()
    True

    References
    ----------
    .. [1] Fujimoto et al. (2018). "Addressing Function Approximation
           Error in Actor-Critic Methods". ICML.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Two independent Q-networks
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both Q-values.

        Parameters
        ----------
        state : torch.Tensor
            Batch of states.
        action : torch.Tensor
            Batch of actions.

        Returns
        -------
        q1 : torch.Tensor
            Q-values from first network.
        q2 : torch.Tensor
            Q-values from second network.
        """
        return self.q1(state, action), self.q2(state, action)

    def min_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pessimistic Q-value estimate.

        Parameters
        ----------
        state : torch.Tensor
            Batch of states.
        action : torch.Tensor
            Batch of actions.

        Returns
        -------
        torch.Tensor
            min(Q1, Q2) for each sample.
        """
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def q1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward through Q1 only (for policy gradient).

        Using only Q1 for policy updates reduces computation and
        provides consistent gradient estimates.

        Parameters
        ----------
        state : torch.Tensor
            Batch of states.
        action : torch.Tensor
            Batch of actions.

        Returns
        -------
        torch.Tensor
            Q1 values.
        """
        return self.q1(state, action)


class ValueNetwork(nn.Module):
    """
    State Value Network
    ===================

    Core Idea
    ---------
    Estimates expected return from state s under current policy:
    :math:`V(s) = \\mathbb{E}_{a \\sim \\pi}[Q(s, a)]`

    Mathematical Theory
    -------------------
    The state value function:

    .. math::

        V^\\pi(s) = \\mathbb{E}_{a \\sim \\pi}[Q^\\pi(s, a)]
        = \\mathbb{E}_{a \\sim \\pi}[r + \\gamma V^\\pi(s')]

    In SAC, the soft value function includes entropy:

    .. math::

        V(s) = \\mathbb{E}_{a \\sim \\pi}[Q(s,a) - \\alpha \\log \\pi(a|s)]

    Problem Statement
    -----------------
    Some algorithms (older SAC formulation) separate V and Q networks:

    - V(s): State value, policy-dependent
    - Q(s,a): Action value, for policy improvement

    Modern SAC computes V implicitly from Q and π, making ValueNetwork
    optional. Included here for completeness and educational purposes.

    Complexity
    ----------
    - Forward: O(state_dim × h × L)
    - Parameters: O(s×h + h² × (L-1) + h)

    Parameters
    ----------
    state_dim : int
        State space dimension.
    hidden_dims : List[int], default=[256, 256]
        Hidden layer sizes.

    Example
    -------
    >>> v_net = ValueNetwork(state_dim=11)
    >>> state = torch.randn(32, 11)
    >>> value = v_net(state)
    >>> value.shape
    torch.Size([32, 1])
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.network = create_mlp(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=nn.ReLU,
            use_orthogonal=True,
            final_gain=1.0,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.

        Parameters
        ----------
        state : torch.Tensor
            Batch of states.

        Returns
        -------
        torch.Tensor
            State values, shape (batch_size, 1).
        """
        return self.network(state)


# =============================================================================
# Unit Tests
# =============================================================================


if __name__ == "__main__":
    print("Testing Neural Network Architectures...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    state_dim = 11
    action_dim = 3

    # Test 1: Orthogonal initialization
    layer = nn.Linear(64, 64)
    orthogonal_init(layer, gain=1.0)
    weight = layer.weight.data
    # Check approximate orthogonality
    gram = weight @ weight.T
    identity = torch.eye(64)
    orthogonality_error = (gram - identity).abs().mean().item()
    assert orthogonality_error < 0.1, f"Orthogonality error: {orthogonality_error}"
    print("  [PASS] Orthogonal initialization")

    # Test 2: MLP factory
    mlp = create_mlp(state_dim, [256, 256], action_dim)
    x = torch.randn(batch_size, state_dim)
    y = mlp(x)
    assert y.shape == (batch_size, action_dim)
    print("  [PASS] MLP factory")

    # Test 3: DeterministicActor
    actor = DeterministicActor(state_dim, action_dim, max_action=2.0).to(device)
    state = torch.randn(batch_size, state_dim, device=device)
    action = actor(state)
    assert action.shape == (batch_size, action_dim)
    assert (action.abs() <= 2.0).all(), "Actions should be bounded"
    print("  [PASS] DeterministicActor")

    # Test 4: GaussianActor sampling
    gaussian_actor = GaussianActor(state_dim, action_dim, max_action=1.0).to(device)
    action, log_prob = gaussian_actor.sample(state)
    assert action.shape == (batch_size, action_dim)
    assert log_prob.shape == (batch_size, 1)
    assert (action.abs() <= 1.0).all(), "Actions should be bounded"
    print("  [PASS] GaussianActor sampling")

    # Test 5: GaussianActor deterministic mode
    action_det, log_prob_det = gaussian_actor.sample(state, deterministic=True)
    assert (log_prob_det == 0).all(), "Deterministic mode should have zero log prob"
    print("  [PASS] GaussianActor deterministic mode")

    # Test 6: QNetwork
    q_net = QNetwork(state_dim, action_dim).to(device)
    action_q = torch.randn(batch_size, action_dim, device=device)
    q_value = q_net(state, action_q)
    assert q_value.shape == (batch_size, 1)
    print("  [PASS] QNetwork")

    # Test 7: TwinQNetwork
    twin_q = TwinQNetwork(state_dim, action_dim).to(device)
    q1, q2 = twin_q(state, action_q)
    assert q1.shape == (batch_size, 1)
    assert q2.shape == (batch_size, 1)
    q_min = twin_q.min_q(state, action_q)
    assert (q_min == torch.min(q1, q2)).all()
    print("  [PASS] TwinQNetwork")

    # Test 8: ValueNetwork
    v_net = ValueNetwork(state_dim).to(device)
    value = v_net(state)
    assert value.shape == (batch_size, 1)
    print("  [PASS] ValueNetwork")

    # Test 9: Gradient flow
    actor.zero_grad()
    action = actor(state)
    loss = action.sum()
    loss.backward()
    grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in actor.parameters())
    assert grad_exists, "Gradients should flow through actor"
    print("  [PASS] Gradient flow")

    # Test 10: Parameter count
    actor_params = sum(p.numel() for p in actor.parameters())
    print(f"  [INFO] DeterministicActor parameters: {actor_params:,}")

    twin_q_params = sum(p.numel() for p in twin_q.parameters())
    print(f"  [INFO] TwinQNetwork parameters: {twin_q_params:,}")

    print("\nAll network tests passed!")
