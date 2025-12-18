"""
Policy Networks Module.

This module implements policy network architectures for both discrete and
continuous action spaces.

Core Idea:
    Policy networks parameterize the policy π_θ(a|s) as a neural network that
    outputs a probability distribution over actions. The choice of distribution
    depends on the action space structure.

Mathematical Background:
    Policy Gradient Theorem (Sutton et al., 1999):
        ∇_θ J(θ) = E_{π_θ}[∇_θ log π_θ(a|s) · Q^π(s,a)]

    For discrete actions (Softmax):
        π_θ(a|s) = exp(f_θ(s)_a) / Σ_a' exp(f_θ(s)_a')
        ∇_θ log π_θ(a|s) = ∇_θ f_θ(s)_a - E_{a'~π}[∇_θ f_θ(s)_a']

    For continuous actions (Gaussian):
        π_θ(a|s) = N(a | μ_θ(s), σ_θ(s)²)
        ∇_θ log π_θ(a|s) = (a - μ_θ(s))/σ² · ∇_θ μ_θ(s) + ...

Problem Context:
    | Action Space | Distribution | Network Output       | Sampling     |
    |--------------|--------------|----------------------|--------------|
    | Discrete     | Categorical  | Logits (unnorm)      | Gumbel-max   |
    | Continuous   | Gaussian     | μ and log(σ)         | Reparam trick|
    | Bounded Cont.| Squashed Gaus| μ, log(σ), then tanh | Reparam+tanh |

References:
    [1] Williams (1992). Simple statistical gradient-following algorithms.
    [2] Sutton et al. (1999). Policy gradient methods for RL.
    [3] Haarnoja et al. (2018). Soft Actor-Critic (squashed Gaussian).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from networks.base import init_weights, MLP


class DiscretePolicy(nn.Module):
    """
    Policy network for discrete action spaces using Softmax distribution.

    Core Idea:
        Output logits (unnormalized log-probabilities) and convert to a
        categorical distribution. The Categorical distribution handles
        normalization and sampling efficiently.

    Mathematical Theory:
        Softmax Policy:
            π_θ(a|s) = softmax(f_θ(s))_a = exp(z_a) / Σ_a' exp(z_a')

        Log-probability:
            log π_θ(a|s) = z_a - log(Σ_a' exp(z_a'))  (numerically stable)

        Entropy:
            H(π) = -Σ_a π(a|s) log π(a|s)

    Parameters
    ----------
    state_dim : int
        Dimension of state observation space.
    action_dim : int
        Number of discrete actions.
    hidden_dims : List[int], default=[128, 128]
        Dimensions of hidden layers.

    Attributes
    ----------
    net : MLP
        Neural network mapping states to action logits.

    Examples
    --------
    >>> policy = DiscretePolicy(state_dim=4, action_dim=2)
    >>> state = torch.randn(32, 4)  # Batch of states

    >>> # Get distribution
    >>> dist = policy.get_distribution(state)
    >>> print(f"Probabilities: {dist.probs[0]}")

    >>> # Sample actions
    >>> action, log_prob, entropy = policy.sample(state)
    >>> print(f"Actions shape: {action.shape}")  # (32,)

    >>> # Evaluate specific actions
    >>> log_prob, entropy = policy.evaluate(state, action)

    Notes
    -----
    Complexity Analysis:
        - Forward pass: O(batch * (state_dim * h + h² * (L-1) + h * action_dim))
        - Sampling: O(batch * action_dim) for categorical sampling
        - Memory: O(batch * max_hidden_dim)

    Numerical Stability:
        Using logits instead of probabilities avoids:
        - Underflow in probability computation
        - Log-of-zero in log_prob computation
        PyTorch's Categorical handles log-sum-exp internally.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.ReLU(),
            output_activation=None,  # Output logits, not probabilities
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits for given states.

        Parameters
        ----------
        state : torch.Tensor
            State observations of shape (batch_size, state_dim).

        Returns
        -------
        torch.Tensor
            Action logits of shape (batch_size, action_dim).
        """
        return self.net(state)

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """
        Get categorical distribution over actions.

        Parameters
        ----------
        state : torch.Tensor
            State observations.

        Returns
        -------
        Categorical
            PyTorch Categorical distribution.
        """
        logits = self.forward(state)
        return Categorical(logits=logits)

    def sample(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy distribution.

        Parameters
        ----------
        state : torch.Tensor
            State observations of shape (batch_size, state_dim).

        Returns
        -------
        action : torch.Tensor
            Sampled actions of shape (batch_size,).
        log_prob : torch.Tensor
            Log probabilities of sampled actions, shape (batch_size,).
        entropy : torch.Tensor
            Entropy of the distribution, shape (batch_size,).

        Notes
        -----
        The sampling process:
            1. Compute logits: z = f_θ(s)
            2. Create distribution: π = Categorical(logits=z)
            3. Sample: a ~ π
            4. Compute log_prob: log π(a|s)
            5. Compute entropy: H(π)
        """
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability of given state-action pairs.

        Used for importance sampling in PPO and other off-policy corrections.

        Parameters
        ----------
        state : torch.Tensor
            State observations of shape (batch_size, state_dim).
        action : torch.Tensor
            Actions to evaluate, shape (batch_size,).

        Returns
        -------
        log_prob : torch.Tensor
            Log probabilities of actions, shape (batch_size,).
        entropy : torch.Tensor
            Entropy of distributions, shape (batch_size,).
        """
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities (not logits).

        Parameters
        ----------
        state : torch.Tensor
            State observations.

        Returns
        -------
        torch.Tensor
            Action probabilities summing to 1, shape (batch_size, action_dim).
        """
        dist = self.get_distribution(state)
        return dist.probs


class ContinuousPolicy(nn.Module):
    """
    Policy network for continuous action spaces using Gaussian distribution.

    Core Idea:
        Output mean and log-standard-deviation of a Gaussian distribution.
        For bounded action spaces, apply tanh squashing and correct the
        log-probability accordingly.

    Mathematical Theory:
        Gaussian Policy:
            π_θ(a|s) = N(a | μ_θ(s), σ_θ(s)²)

        Log-probability:
            log π(a|s) = -1/2 [(a-μ)²/σ² + log(2πσ²)]

        For Tanh Squashing (bounded actions):
            u ~ N(μ, σ²)
            a = tanh(u)  ∈ (-1, 1)

            Log-prob correction (change of variables):
            log π(a|s) = log N(u|μ,σ²) - Σ_i log(1 - tanh²(u_i))
                       = log N(u|μ,σ²) - Σ_i log(1 - a_i²)

        Reparameterization Trick (Kingma & Welling, 2014):
            u = μ + σ · ε,  where ε ~ N(0, I)
            Allows gradients to flow through sampling.

    Parameters
    ----------
    state_dim : int
        Dimension of state observation space.
    action_dim : int
        Dimension of continuous action space.
    hidden_dims : List[int], default=[256, 256]
        Dimensions of hidden layers.
    state_dependent_std : bool, default=False
        If True, std depends on state. If False, std is a learnable parameter.
    log_std_min : float, default=-20.0
        Minimum value for log standard deviation (prevents collapse).
    log_std_max : float, default=2.0
        Maximum value for log standard deviation (prevents explosion).

    Attributes
    ----------
    feature_net : MLP
        Shared feature extraction network.
    mean_layer : nn.Linear
        Output layer for action mean.
    log_std : nn.Parameter or nn.Linear
        Log standard deviation (parameter or state-dependent).

    Examples
    --------
    >>> policy = ContinuousPolicy(state_dim=3, action_dim=2)
    >>> state = torch.randn(32, 3)

    >>> # Sample actions (bounded to [-1, 1])
    >>> action, log_prob, entropy = policy.sample(state)
    >>> print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")

    >>> # Deterministic action (for evaluation)
    >>> action, _, _ = policy.sample(state, deterministic=True)

    Notes
    -----
    Complexity Analysis:
        - Forward: O(batch * (state_dim * h + h² + h * action_dim))
        - Sampling: O(batch * action_dim) for Gaussian sampling
        - Tanh + log-prob correction: O(batch * action_dim)

    Design Choices:
        - State-independent std: Fewer parameters, stable training
        - State-dependent std: More expressive, can adapt exploration
        - Tanh squashing: Essential for bounded action spaces
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        state_dependent_std: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std

        # Shared feature extraction
        self.feature_net = MLP(
            input_dim=state_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=nn.ReLU(),
            output_activation=nn.ReLU(),
        )

        # Mean output layer
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        init_weights(self.mean_layer, gain=0.01)

        # Standard deviation
        if state_dependent_std:
            self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
            init_weights(self.log_std_layer, gain=0.01)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Gaussian distribution parameters.

        Parameters
        ----------
        state : torch.Tensor
            State observations of shape (batch_size, state_dim).

        Returns
        -------
        mean : torch.Tensor
            Action means of shape (batch_size, action_dim).
        std : torch.Tensor
            Action standard deviations of shape (batch_size, action_dim).
        """
        features = self.feature_net(state)
        mean = self.mean_layer(features)

        if self.state_dependent_std:
            log_std = self.log_std_layer(features)
        else:
            log_std = self.log_std.expand_as(mean)

        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        return mean, std

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """
        Get Gaussian distribution over actions.

        Parameters
        ----------
        state : torch.Tensor
            State observations.

        Returns
        -------
        Normal
            PyTorch Normal distribution.
        """
        mean, std = self.forward(state)
        return Normal(mean, std)

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy distribution.

        Uses reparameterization trick for differentiable sampling and
        tanh squashing for bounded actions.

        Parameters
        ----------
        state : torch.Tensor
            State observations of shape (batch_size, state_dim).
        deterministic : bool, default=False
            If True, return mean action (no sampling).

        Returns
        -------
        action : torch.Tensor
            Actions bounded to [-1, 1], shape (batch_size, action_dim).
        log_prob : torch.Tensor
            Corrected log probabilities, shape (batch_size,).
        entropy : torch.Tensor
            Gaussian entropy, shape (batch_size,).

        Notes
        -----
        Log-probability correction for tanh:
            log π(a) = log N(u|μ,σ²) - Σ log(1 - tanh²(u))
                     = log N(u|μ,σ²) - Σ log(1 - a²)

        This correction accounts for the change of variables from u to a=tanh(u).
        """
        mean, std = self.forward(state)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(state.shape[0], device=state.device)
            entropy = torch.zeros(state.shape[0], device=state.device)
        else:
            dist = Normal(mean, std)
            u = dist.rsample()  # Reparameterization trick
            action = torch.tanh(u)

            # Log probability with Jacobian correction
            log_prob = dist.log_prob(u).sum(dim=-1)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

            entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability of given actions.

        Parameters
        ----------
        state : torch.Tensor
            State observations of shape (batch_size, state_dim).
        action : torch.Tensor
            Actions to evaluate, shape (batch_size, action_dim).
            Should be in [-1, 1] (tanh-squashed).

        Returns
        -------
        log_prob : torch.Tensor
            Log probabilities, shape (batch_size,).
        entropy : torch.Tensor
            Entropy values, shape (batch_size,).
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        # Inverse tanh to get u from action
        action_clipped = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(action_clipped)

        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy
