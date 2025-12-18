"""
REINFORCE Algorithm Implementation.

This module implements the REINFORCE algorithm (Williams, 1992), the foundational
Monte Carlo policy gradient method.

Core Idea:
    REINFORCE uses complete episode returns as unbiased estimates of the
    action-value function Q(s,a). While simple and theoretically elegant,
    it suffers from high variance due to the stochasticity in trajectory
    sampling.

Mathematical Theory:
    Policy Gradient Theorem:
        ∇_θ J(θ) = E_{π_θ}[∇_θ log π_θ(a|s) · Q^π(s,a)]

    REINFORCE Estimator (using MC return):
        ∇_θ J(θ) ≈ (1/N) Σ_i Σ_t ∇_θ log π_θ(a_t^i | s_t^i) · G_t^i

    Where G_t = Σ_{k=t}^T γ^{k-t} r_k is the discounted return from time t.

    Properties:
        - Unbiased: E[G_t] = Q^π(s_t, a_t)
        - High Variance: Var[G_t] can be large due to trajectory stochasticity

Problem Statement:
    Given: MDP (S, A, P, R, γ), parameterized policy π_θ
    Find: θ* = argmax_θ J(θ) = argmax_θ E_{τ~π_θ}[R(τ)]

Algorithm Comparison:
    | Method             | Bias | Variance | Update Timing |
    |--------------------|------|----------|---------------|
    | REINFORCE          | None | High     | Episode end   |
    | REINFORCE+Baseline | None | Medium   | Episode end   |
    | Actor-Critic (TD)  | Some | Low      | Each step     |

References:
    [1] Williams, R.J. (1992). Simple statistical gradient-following
        algorithms for connectionist reinforcement learning.
        Machine Learning, 8(3-4), 229-256.

    [2] Sutton, R.S., & Barto, A.G. (2018). Reinforcement Learning:
        An Introduction. Chapter 13.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.config import TrainingConfig
from core.buffers import EpisodeBuffer
from networks.policy import DiscretePolicy, ContinuousPolicy
from algorithms.base import BasePolicyGradient
from utils.returns import compute_returns


class REINFORCE(BasePolicyGradient):
    """
    REINFORCE Algorithm (Williams, 1992).

    The foundational Monte Carlo policy gradient algorithm that uses complete
    episode returns to estimate policy gradients.

    Core Idea:
        Sample complete trajectories, compute discounted returns, and update
        policy in the direction that increases probability of high-return
        actions.

    Mathematical Theory:
        Gradient Estimate:
            ∇_θ J(θ) ≈ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t

        Loss Function (for gradient descent):
            L(θ) = -E[log π_θ(a|s) · G]

        Update Rule:
            θ ← θ + α · G_t · ∇_θ log π_θ(a_t|s_t)

    Parameters
    ----------
    state_dim : int
        Dimension of state observation space.
    action_dim : int
        Dimension of action space (number of actions for discrete,
        action vector dimension for continuous).
    config : TrainingConfig, optional
        Training configuration. Uses defaults if not provided.
    continuous : bool, default=False
        If True, use Gaussian policy for continuous action space.
        If False, use Categorical policy for discrete action space.

    Attributes
    ----------
    policy : DiscretePolicy or ContinuousPolicy
        Policy network π_θ(a|s).
    optimizer : torch.optim.Optimizer
        Optimizer for policy parameters.
    continuous : bool
        Whether using continuous action space.

    Examples
    --------
    >>> # Discrete action space (e.g., CartPole)
    >>> config = TrainingConfig(gamma=0.99, lr_actor=1e-3)
    >>> agent = REINFORCE(state_dim=4, action_dim=2, config=config)

    >>> # Training loop
    >>> buffer = EpisodeBuffer()
    >>> state = env.reset()
    >>> while not done:
    ...     action, info = agent.select_action(state)
    ...     next_state, reward, done, _ = env.step(action)
    ...     buffer.store(state, action, reward, info["log_prob"],
    ...                  entropy=info["entropy"])
    ...     state = next_state
    >>> loss_info = agent.update(buffer)

    >>> # Continuous action space (e.g., Pendulum)
    >>> agent = REINFORCE(state_dim=3, action_dim=1, config=config, continuous=True)

    Notes
    -----
    Complexity Analysis:
        - select_action: O(policy_forward_pass)
        - update: O(T * policy_forward_pass) for episode length T
        - Memory: O(T) for storing episode trajectory

    Practical Considerations:
        - High variance makes training unstable
        - Requires many episodes for reliable gradient estimates
        - Works well for simple problems with short episodes
        - Consider using baseline (REINFORCEBaseline) for better performance

    Summary:
        REINFORCE is the simplest policy gradient algorithm, providing an
        unbiased but high-variance gradient estimate. It serves as the
        foundation for understanding more advanced methods that add variance
        reduction techniques.
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

        # Build policy network
        if continuous:
            self.policy = ContinuousPolicy(state_dim, action_dim).to(self.device)
        else:
            self.policy = DiscretePolicy(state_dim, action_dim).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.lr_actor,
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[Union[int, np.ndarray], Dict[str, torch.Tensor]]:
        """
        Select action according to current policy.

        Parameters
        ----------
        state : np.ndarray
            Current state observation of shape (state_dim,).
        deterministic : bool, default=False
            If True, return most likely action (for evaluation).

        Returns
        -------
        action : int or np.ndarray
            Selected action. Integer for discrete space, array for continuous.
        info : dict
            Dictionary containing:
            - "log_prob": Log probability of selected action
            - "entropy": Entropy of policy distribution

        Notes
        -----
        For discrete actions:
            - Stochastic: Sample from Categorical(logits)
            - Deterministic: argmax over logits

        For continuous actions:
            - Stochastic: Sample from N(μ, σ²), apply tanh
            - Deterministic: tanh(μ)
        """
        state_t = self._to_tensor(state).unsqueeze(0)

        with torch.no_grad() if deterministic else torch.enable_grad():
            if self.continuous:
                action, log_prob, entropy = self.policy.sample(state_t, deterministic)
                action = action.squeeze(0).cpu().numpy()
            else:
                if deterministic:
                    logits = self.policy(state_t)
                    action = logits.argmax(dim=-1).item()
                    log_prob = torch.zeros(1)
                    entropy = torch.zeros(1)
                else:
                    action, log_prob, entropy = self.policy.sample(state_t)
                    action = action.item()

        return action, {"log_prob": log_prob, "entropy": entropy}

    def update(self, buffer: EpisodeBuffer) -> Dict[str, float]:
        """
        Update policy using REINFORCE gradient estimate.

        Computes Monte Carlo returns and updates policy to increase
        probability of high-return actions.

        Parameters
        ----------
        buffer : EpisodeBuffer
            Episode trajectory containing:
            - rewards: List of rewards [r_1, ..., r_T]
            - log_probs: List of action log probabilities

        Returns
        -------
        Dict[str, float]
            Training metrics:
            - "policy_loss": Policy gradient loss value
            - "entropy": Mean policy entropy
            - "mean_return": Mean of computed returns

        Notes
        -----
        Algorithm Steps:
            1. Compute MC returns: G_t = Σ_{k=t}^T γ^{k-t} r_k
            2. Normalize returns (optional, for variance reduction)
            3. Compute loss: L = -Σ_t log π(a_t|s_t) · G_t
            4. Add entropy bonus: L -= c_ent · H(π)
            5. Backpropagate and update

        Mathematical Details:
            The negative sign converts gradient ascent to gradient descent:
                max J(θ) ⟺ min -J(θ)
        """
        # Compute Monte Carlo returns
        returns = compute_returns(
            buffer.rewards,
            self.config.gamma,
            normalize=self.config.normalize_advantage,
        )

        # Stack log probabilities
        log_probs = torch.stack(buffer.log_probs)

        # Policy gradient loss: -E[log π(a|s) · G]
        policy_loss = -(log_probs * returns).mean()

        # Entropy regularization
        if buffer.entropies and self.config.entropy_coef > 0:
            entropies = torch.stack(buffer.entropies)
            entropy_bonus = entropies.mean()
            total_loss = policy_loss - self.config.entropy_coef * entropy_bonus
        else:
            total_loss = policy_loss
            entropy_bonus = torch.tensor(0.0)

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy_bonus.item() if isinstance(entropy_bonus, torch.Tensor) else entropy_bonus,
            "mean_return": returns.mean().item(),
        }
