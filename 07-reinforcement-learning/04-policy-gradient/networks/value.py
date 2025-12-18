"""
Value Networks Module.

This module implements value function approximators and shared actor-critic
network architectures.

Core Idea:
    Value networks estimate expected future returns, serving as baselines for
    variance reduction or as critics in actor-critic methods. The shared
    architecture enables efficient feature reuse between policy and value
    learning.

Mathematical Background:
    State Value Function:
        V^π(s) = E_{π}[∑_{t=0}^∞ γ^t r_t | s_0 = s]

    Used as baseline in policy gradient:
        ∇_θ J(θ) = E[∇_θ log π_θ(a|s) · (Q^π(s,a) - V^π(s))]
                 = E[∇_θ log π_θ(a|s) · A^π(s,a)]

    Advantage function:
        A^π(s,a) = Q^π(s,a) - V^π(s)
                 ≈ r + γV(s') - V(s)  (TD error approximation)

Problem Context:
    Variance Reduction:
        - Without baseline: Var[∇J] ∝ E[Q²]
        - With V as baseline: Var[∇J] ∝ E[A²] << E[Q²]

    Shared vs Separate Networks:
        | Architecture | Parameters | Features    | Training   |
        |--------------|------------|-------------|------------|
        | Separate     | 2x         | Independent | Stable     |
        | Shared       | ~1.2x      | Reused      | Faster     |

References:
    [1] Sutton & Barto (2018). Reinforcement Learning: An Introduction.
    [2] Mnih et al. (2016). A3C - Asynchronous Advantage Actor-Critic.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from networks.base import init_weights, MLP


class ValueNetwork(nn.Module):
    """
    State value function network V(s).

    Estimates the expected cumulative return from a given state under
    the current policy.

    Core Idea:
        Learn V(s) to predict expected future returns, enabling:
        1. Baseline for variance reduction in REINFORCE
        2. Critic in Actor-Critic methods
        3. Target computation for TD learning

    Mathematical Theory:
        Value Function:
            V^π(s) = E_{a~π}[Q^π(s,a)] = E_{π}[∑_t γ^t r_t | s_0 = s]

        Training Target (Monte Carlo):
            L = (V_θ(s) - G_t)²  where G_t = Σ_k γ^k r_{t+k}

        Training Target (TD):
            L = (V_θ(s) - (r + γV_θ(s')))²

    Parameters
    ----------
    state_dim : int
        Dimension of state observation space.
    hidden_dims : List[int], default=[128, 128]
        Dimensions of hidden layers.

    Attributes
    ----------
    net : MLP
        Neural network mapping states to scalar values.

    Examples
    --------
    >>> value_net = ValueNetwork(state_dim=4)
    >>> states = torch.randn(32, 4)

    >>> # Get value estimates
    >>> values = value_net(states)
    >>> print(f"Values shape: {values.shape}")  # (32, 1)

    >>> # Training
    >>> targets = compute_returns(rewards, gamma)
    >>> loss = F.mse_loss(values.squeeze(), targets)

    Notes
    -----
    Complexity Analysis:
        - Forward: O(batch * (state_dim * h + h² + h))
        - Parameters: O(state_dim * h + h² + h)
        - Output: Scalar per state

    Architecture Guidelines:
        - Same hidden structure as policy often works well
        - Can use larger network than policy (value learning is supervised)
        - Output layer gain = 1.0 (values can be large)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim = state_dim

        self.net = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=nn.ReLU(),
            output_activation=None,
        )

        # Value output can be large, use gain=1.0
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                init_weights(module, gain=np.sqrt(2))

        # Find and reinitialize output layer with gain=1.0
        for module in reversed(list(self.net.modules())):
            if isinstance(module, nn.Linear):
                init_weights(module, gain=1.0)
                break

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate state values.

        Parameters
        ----------
        state : torch.Tensor
            State observations of shape (batch_size, state_dim).

        Returns
        -------
        torch.Tensor
            Value estimates of shape (batch_size, 1).
        """
        return self.net(state)


class ActorCriticNetwork(nn.Module):
    """
    Shared-feature Actor-Critic network architecture.

    Core Idea:
        Actor and Critic share a common feature extraction backbone, then
        branch into separate heads. This architecture enables:
        1. Parameter efficiency through feature reuse
        2. Multi-task learning benefits
        3. Faster training convergence

    Mathematical Theory:
        Network Structure:
            features = f_shared(s)
            π(a|s) = Actor_head(features)
            V(s) = Critic_head(features)

        Combined Loss:
            L = L_policy + c_v · L_value - c_ent · H(π)

        Where:
            L_policy = -E[log π(a|s) · A]
            L_value = E[(V(s) - G)²]
            H(π) = -E[π log π]  (entropy bonus)

    Architecture:
        ```
        state ─► [shared_net] ─► features
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                   [actor_head]           [critic_head]
                        │                       │
                        ▼                       ▼
                     policy                   value
        ```

    Parameters
    ----------
    state_dim : int
        Dimension of state observation space.
    action_dim : int
        Dimension of action space.
    hidden_dim : int, default=256
        Dimension of hidden layers.
    continuous : bool, default=False
        If True, use Gaussian policy for continuous actions.
        If False, use Categorical policy for discrete actions.

    Attributes
    ----------
    shared_net : nn.Sequential
        Shared feature extraction layers.
    actor_head : nn.Linear
        Policy output layer (logits for discrete).
    actor_mean : nn.Linear
        Mean output for continuous policy.
    actor_log_std : nn.Parameter
        Log std for continuous policy.
    critic_head : nn.Linear
        Value output layer.

    Examples
    --------
    >>> # Discrete action space
    >>> ac_net = ActorCriticNetwork(state_dim=4, action_dim=2)
    >>> state = torch.randn(32, 4)
    >>> action, log_prob, entropy, value = ac_net.get_action_and_value(state)

    >>> # Continuous action space
    >>> ac_net = ActorCriticNetwork(state_dim=3, action_dim=2, continuous=True)
    >>> action, log_prob, entropy, value = ac_net.get_action_and_value(state)

    >>> # Evaluate existing actions (for PPO)
    >>> action, log_prob, entropy, value = ac_net.get_action_and_value(state, action=old_actions)

    Notes
    -----
    Complexity Analysis:
        - Shared features: O(batch * (state_dim * h + h²))
        - Actor head: O(batch * h * action_dim)
        - Critic head: O(batch * h)
        - Total parameters: ~60% of separate networks

    Design Trade-offs:
        Shared Networks:
            + Fewer parameters
            + Feature transfer between tasks
            + Faster initial learning
            - Potential interference between objectives
            - May need careful loss weighting

        Separate Networks:
            + Independent optimization
            + Can use different architectures
            - More parameters
            - Slower training
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # Shared feature extraction
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply appropriate initialization to all layers."""
        # Shared layers with ReLU gain
        for module in self.shared_net.modules():
            if isinstance(module, nn.Linear):
                init_weights(module, gain=np.sqrt(2))

        # Actor head with small gain
        if self.continuous:
            init_weights(self.actor_mean, gain=0.01)
        else:
            init_weights(self.actor_head, gain=0.01)

        # Critic head with unit gain
        init_weights(self.critic_head, gain=1.0)

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Forward pass computing policy parameters and value.

        Parameters
        ----------
        state : torch.Tensor
            State observations of shape (batch_size, state_dim).

        Returns
        -------
        policy_output : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            For discrete: action logits of shape (batch_size, action_dim)
            For continuous: (mean, log_std) tuple
        value : torch.Tensor
            Value estimates of shape (batch_size, 1)
        """
        features = self.shared_net(state)
        value = self.critic_head(features)

        if self.continuous:
            mean = self.actor_mean(features)
            return (mean, self.actor_log_std), value
        else:
            logits = self.actor_head(features)
            return logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        If action is provided, evaluates that action (for importance sampling).
        Otherwise, samples a new action from the policy.

        Parameters
        ----------
        state : torch.Tensor
            State observations of shape (batch_size, state_dim).
        action : Optional[torch.Tensor]
            Actions to evaluate. If None, samples new actions.

        Returns
        -------
        action : torch.Tensor
            Sampled or provided actions.
        log_prob : torch.Tensor
            Log probabilities of actions, shape (batch_size,).
        entropy : torch.Tensor
            Policy entropy, shape (batch_size,).
        value : torch.Tensor
            Value estimates, shape (batch_size,).
        """
        features = self.shared_net(state)
        value = self.critic_head(features).squeeze(-1)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp()
            dist = Normal(mean, std)

            if action is None:
                u = dist.rsample()
                action = torch.tanh(u)
            else:
                action_clipped = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
                u = torch.atanh(action_clipped)

            log_prob = dist.log_prob(u).sum(dim=-1)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self.actor_head(features)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimates only (no policy computation).

        Parameters
        ----------
        state : torch.Tensor
            State observations.

        Returns
        -------
        torch.Tensor
            Value estimates.
        """
        features = self.shared_net(state)
        return self.critic_head(features).squeeze(-1)
