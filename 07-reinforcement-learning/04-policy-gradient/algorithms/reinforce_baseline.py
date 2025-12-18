"""
REINFORCE with Baseline Implementation.

This module implements REINFORCE with a learned state-value baseline for
variance reduction.

Core Idea:
    Adding a baseline b(s) to the return does not change the expected gradient
    but can significantly reduce variance. Using V(s) as baseline transforms
    the return into an advantage estimate: A(s,a) = G - V(s).

Mathematical Theory:
    Modified Policy Gradient:
        ∇_θ J(θ) = E_{π_θ}[∇_θ log π_θ(a|s) · (Q(s,a) - b(s))]

    Key Insight - Baseline doesn't change expectation:
        E_{π_θ}[∇_θ log π_θ(a|s) · b(s)] = 0

    Proof:
        Σ_a π_θ(a|s) · ∇_θ log π_θ(a|s) · b(s)
        = b(s) · Σ_a ∇_θ π_θ(a|s)
        = b(s) · ∇_θ Σ_a π_θ(a|s)
        = b(s) · ∇_θ 1
        = 0

    Optimal Baseline:
        The variance-minimizing baseline is:
        b*(s) = E[G² · (∇log π)²] / E[(∇log π)²]

        In practice, V(s) ≈ E[G|s] is a good approximation.

    Advantage Function:
        A(s,a) = Q(s,a) - V(s) = G - V(s)  (MC estimate)

        Properties:
        - E_a[A(s,a)] = 0  (zero mean over actions)
        - A > 0: action better than average
        - A < 0: action worse than average

Problem Statement:
    Reduce variance of REINFORCE gradient estimate while maintaining
    unbiasedness by learning an auxiliary value function.

Algorithm Comparison:
    | Component | REINFORCE      | REINFORCE+Baseline    |
    |-----------|----------------|----------------------|
    | Networks  | Policy only    | Policy + Value       |
    | Ψ_t       | G_t           | G_t - V(s_t)         |
    | Variance  | High          | Medium               |
    | Bias      | None          | None                 |
    | Training  | Simpler       | Two optimization objectives |

References:
    [1] Williams (1992). REINFORCE algorithm.
    [2] Sutton et al. (1999). Policy gradient with function approximation.
    [3] Greensmith et al. (2004). Variance reduction techniques for
        gradient estimates in RL.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.config import TrainingConfig
from core.buffers import EpisodeBuffer
from networks.policy import DiscretePolicy, ContinuousPolicy
from networks.value import ValueNetwork
from algorithms.base import BasePolicyGradient
from utils.returns import compute_returns


class REINFORCEBaseline(BasePolicyGradient):
    """
    REINFORCE with Learned Value Baseline.

    Extends REINFORCE by learning a state-value function V(s) to use as
    a baseline, reducing gradient variance while maintaining unbiasedness.

    Core Idea:
        Instead of using raw returns G_t, use advantages A_t = G_t - V(s_t).
        This centers the gradient signal around zero, reducing variance
        while preserving the expected gradient direction.

    Mathematical Theory:
        Policy Gradient with Baseline:
            ∇_θ J(θ) = E[∇_θ log π_θ(a|s) · (G - V(s))]

        Advantage Estimate:
            A_t = G_t - V_φ(s_t)

        Two-Objective Optimization:
            Policy: max_θ E[log π_θ(a|s) · A]
            Value:  min_φ E[(V_φ(s) - G)²]

        Variance Reduction:
            Var[G - V] < Var[G]  when V ≈ E[G|s]

    Parameters
    ----------
    state_dim : int
        Dimension of state observation space.
    action_dim : int
        Dimension of action space.
    config : TrainingConfig, optional
        Training configuration with lr_actor, lr_critic, etc.
    continuous : bool, default=False
        If True, use Gaussian policy for continuous actions.

    Attributes
    ----------
    policy : DiscretePolicy or ContinuousPolicy
        Policy network π_θ(a|s).
    value_net : ValueNetwork
        Value network V_φ(s) serving as baseline.
    policy_optimizer : torch.optim.Optimizer
        Optimizer for policy parameters.
    value_optimizer : torch.optim.Optimizer
        Optimizer for value function parameters.

    Examples
    --------
    >>> config = TrainingConfig(
    ...     gamma=0.99,
    ...     lr_actor=1e-3,
    ...     lr_critic=1e-3
    ... )
    >>> agent = REINFORCEBaseline(state_dim=4, action_dim=2, config=config)

    >>> # Training loop
    >>> buffer = EpisodeBuffer()
    >>> state = env.reset()
    >>> while not done:
    ...     action, info = agent.select_action(state)
    ...     next_state, reward, done, _ = env.step(action)
    ...     buffer.store(
    ...         state, action, reward, info["log_prob"],
    ...         value=info["value"], entropy=info["entropy"]
    ...     )
    ...     state = next_state
    >>> loss_info = agent.update(buffer)

    Notes
    -----
    Complexity Analysis:
        - select_action: O(policy_forward + value_forward)
        - update: O(T * (policy_forward + value_forward))
        - Parameters: policy_params + value_params
        - Memory: O(T) for episode storage

    Implementation Details:
        1. Value estimates are detached when computing advantages to prevent
           gradients from flowing through the baseline.
        2. Policy and value networks are updated with separate optimizers,
           allowing different learning rates.
        3. Advantage normalization further reduces variance.

    Practical Considerations:
        - lr_critic can be larger than lr_actor (value learning is supervised)
        - Initialize value network to output ~0 to avoid initial bias
        - Monitor value loss to ensure baseline is learning

    Summary:
        REINFORCE with Baseline provides a significant variance reduction over
        vanilla REINFORCE by subtracting a learned state-dependent baseline.
        This makes training more stable while keeping the gradient unbiased.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TrainingConfig = None,
        continuous: bool = False,
    ):
        if config is None:
            config = TrainingConfig()
        super().__init__(config)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # Policy network (Actor)
        if continuous:
            self.policy = ContinuousPolicy(state_dim, action_dim).to(self.device)
        else:
            self.policy = DiscretePolicy(state_dim, action_dim).to(self.device)

        # Value network (Baseline/Critic)
        self.value_net = ValueNetwork(state_dim).to(self.device)

        # Separate optimizers for different learning rates
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.lr_actor,
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=config.lr_critic,
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[Union[int, np.ndarray], Dict[str, torch.Tensor]]:
        """
        Select action and compute value estimate.

        Parameters
        ----------
        state : np.ndarray
            Current state observation.
        deterministic : bool, default=False
            If True, return most likely action.

        Returns
        -------
        action : int or np.ndarray
            Selected action.
        info : dict
            Dictionary containing:
            - "log_prob": Log probability of action
            - "value": Value estimate V(s)
            - "entropy": Policy entropy
        """
        state_t = self._to_tensor(state).unsqueeze(0)

        # Get value estimate (no gradient needed for value during action selection)
        with torch.no_grad():
            value = self.value_net(state_t)

        # Get action - need gradient for log_prob in non-deterministic mode
        if deterministic:
            with torch.no_grad():
                if self.continuous:
                    action, log_prob, entropy = self.policy.sample(state_t, deterministic=True)
                    action = action.squeeze(0).cpu().numpy()
                else:
                    logits = self.policy(state_t)
                    action = logits.argmax(dim=-1).item()
                    log_prob = torch.zeros(1)
                    entropy = torch.zeros(1)
        else:
            # Keep gradients for policy update
            if self.continuous:
                action, log_prob, entropy = self.policy.sample(state_t, deterministic=False)
                action = action.squeeze(0).detach().cpu().numpy()
            else:
                action, log_prob, entropy = self.policy.sample(state_t)
                action = action.item()

        return action, {
            "log_prob": log_prob,
            "value": value,
            "entropy": entropy,
        }

    def update(self, buffer: EpisodeBuffer) -> Dict[str, float]:
        """
        Update policy and value networks.

        Computes advantages using learned value baseline and updates
        both networks.

        Parameters
        ----------
        buffer : EpisodeBuffer
            Episode data containing rewards, log_probs, and values.

        Returns
        -------
        Dict[str, float]
            Training metrics:
            - "policy_loss": Policy gradient loss
            - "value_loss": Value function MSE loss
            - "entropy": Mean policy entropy
            - "mean_advantage": Mean advantage estimate
            - "mean_return": Mean Monte Carlo return

        Notes
        -----
        Update Procedure:
            1. Compute MC returns from rewards
            2. Compute advantages: A = G - V(s)
            3. Normalize advantages
            4. Update policy: θ ← θ + α·A·∇log π
            5. Update value: φ ← φ - β·∇(V - G)²

        Important:
            Values are detached when computing advantages to prevent
            gradient flow through the baseline into the policy loss.
        """
        # Compute Monte Carlo returns (unnormalized for value target)
        returns = compute_returns(
            buffer.rewards,
            self.config.gamma,
            normalize=False,
        )

        # Recompute values with gradient for value network update
        states = torch.FloatTensor(np.array(buffer.states)).to(self.device)
        values_for_loss = self.value_net(states).squeeze()

        # Get stored values for advantage (no gradient needed)
        with torch.no_grad():
            values_detached = values_for_loss.detach()

        # Get log probs (already have gradient from select_action)
        log_probs = torch.stack(buffer.log_probs)

        # Compute advantages: A = G - V
        # Use detached values to prevent gradient flow through baseline into policy
        advantages = returns - values_detached

        # Normalize advantages
        advantages = self._normalize_advantages(advantages)

        # Policy loss
        policy_loss = -(log_probs * advantages).mean()

        # Value loss (MSE) - uses values_for_loss which has gradients
        value_loss = F.mse_loss(values_for_loss, returns)

        # Entropy bonus
        if buffer.entropies:
            entropies = torch.stack(buffer.entropies)
            entropy_bonus = entropies.mean()
        else:
            entropy_bonus = torch.tensor(0.0)

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_total_loss = policy_loss - self.config.entropy_coef * entropy_bonus
        policy_total_loss.backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm,
            )
        self.policy_optimizer.step()

        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.value_net.parameters(),
                self.config.max_grad_norm,
            )
        self.value_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_bonus.item() if isinstance(entropy_bonus, torch.Tensor) else 0.0,
            "mean_advantage": advantages.mean().item(),
            "mean_return": returns.mean().item(),
        }
