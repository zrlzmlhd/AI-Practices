"""
Advantage Actor-Critic (A2C) Algorithm Implementation.

This module implements A2C with Generalized Advantage Estimation (GAE),
a synchronous actor-critic method that balances bias and variance in
advantage estimation.

Core Idea:
    A2C combines policy gradient (actor) with value function learning (critic)
    using temporal difference (TD) methods. GAE provides a principled way to
    interpolate between TD(0) and Monte Carlo estimates, controlling the
    bias-variance trade-off.

Mathematical Theory:
    TD Error (1-step advantage):
        δ_t = r_t + γV(s_{t+1}) - V(s_t)

    Generalized Advantage Estimation (Schulman et al., 2016):
        A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

    GAE as exponentially-weighted average of n-step advantages:
        A_t^GAE = (1-λ)(A_t^(1) + λA_t^(2) + λ²A_t^(3) + ...)

        where A_t^(n) = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n}) - V(s_t)

    Bias-Variance Trade-off:
        λ = 0: A_t = δ_t (TD(0), high bias, low variance)
        λ = 1: A_t = Σ_k γ^k δ_{t+k} = G_t - V(s_t) (MC, low bias, high variance)
        λ = 0.95: Good balance for most tasks

    Combined Loss Function:
        L = L_policy + c_v · L_value - c_ent · H(π)

        L_policy = -E[log π(a|s) · A^GAE]
        L_value = E[(V(s) - V_target)²]
        H(π) = -E[π log π]  (entropy for exploration)

Problem Statement:
    Learn policy π_θ and value function V_φ that maximize expected return
    while maintaining training stability through advantage normalization
    and gradient clipping.

Algorithm Comparison:
    | Method    | Advantage Est.  | Update  | Networks     | Use Case      |
    |-----------|-----------------|---------|--------------|---------------|
    | REINFORCE | G_t             | Episode | Policy only  | Simple tasks  |
    | +Baseline | G_t - V(s)      | Episode | Separate P+V | Medium tasks  |
    | A2C       | GAE(δ_t)        | n-step  | Shared P+V   | Complex tasks |
    | PPO       | GAE(δ_t)+clip   | n-step  | Shared P+V   | Production    |

References:
    [1] Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement
        Learning. ICML.
    [2] Schulman et al. (2016). High-Dimensional Continuous Control Using
        Generalized Advantage Estimation. ICLR.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.config import TrainingConfig
from core.buffers import EpisodeBuffer
from networks.policy import DiscretePolicy, ContinuousPolicy
from networks.value import ValueNetwork, ActorCriticNetwork
from algorithms.base import BasePolicyGradient
from utils.returns import compute_gae, compute_n_step_returns


class A2C(BasePolicyGradient):
    """
    Advantage Actor-Critic with GAE.

    A synchronous policy gradient method that uses TD-based advantage
    estimation with the GAE framework for stable training.

    Core Idea:
        Use a critic (value function) to estimate advantages, enabling
        more frequent updates and lower variance compared to Monte Carlo
        methods. GAE provides fine-grained control over the bias-variance
        trade-off.

    Mathematical Theory:
        Actor Update (Policy Gradient):
            θ ← θ + α_actor · ∇_θ E[log π_θ(a|s) · A^GAE]

        Critic Update (TD Learning):
            φ ← φ - α_critic · ∇_φ E[(V_φ(s) - V_target)²]

        Where V_target = r + γV(s') for TD(0)
              V_target = A^GAE + V(s) for GAE

        Total Loss (shared network):
            L = -log π(a|s)·A + c_v·(V - V_target)² - c_ent·H(π)

    Parameters
    ----------
    state_dim : int
        Dimension of state observation space.
    action_dim : int
        Dimension of action space.
    config : TrainingConfig, optional
        Training configuration including:
        - gamma: Discount factor
        - gae_lambda: GAE λ parameter
        - lr_actor, lr_critic: Learning rates
        - entropy_coef: Entropy bonus coefficient
        - value_coef: Value loss coefficient
    continuous : bool, default=False
        If True, use Gaussian policy for continuous actions.
    use_gae : bool, default=True
        If True, use GAE for advantage estimation.
        If False, use n-step returns.
    shared_network : bool, default=True
        If True, actor and critic share feature layers.

    Attributes
    ----------
    model : ActorCriticNetwork
        Shared actor-critic network (if shared_network=True).
    policy : DiscretePolicy or ContinuousPolicy
        Separate policy network (if shared_network=False).
    value_net : ValueNetwork
        Separate value network (if shared_network=False).
    optimizer : torch.optim.Optimizer
        Optimizer for network parameters.
    use_gae : bool
        Whether using GAE for advantages.
    shared_network : bool
        Whether using shared network architecture.

    Examples
    --------
    >>> # Standard A2C with GAE
    >>> config = TrainingConfig(
    ...     gamma=0.99,
    ...     gae_lambda=0.95,
    ...     lr_actor=3e-4,
    ...     entropy_coef=0.01,
    ...     value_coef=0.5
    ... )
    >>> agent = A2C(state_dim=4, action_dim=2, config=config)

    >>> # Training loop (n-step update)
    >>> buffer = EpisodeBuffer()
    >>> state = env.reset()
    >>> for step in range(n_steps):
    ...     action, info = agent.select_action(state)
    ...     next_state, reward, done, _ = env.step(action)
    ...     buffer.store(
    ...         state, action, reward, info["log_prob"],
    ...         value=info["value"], done=done, entropy=info["entropy"]
    ...     )
    ...     state = next_state
    ...     if done:
    ...         break
    >>> loss_info = agent.update(buffer, next_state, done)

    >>> # Continuous control with separate networks
    >>> agent = A2C(
    ...     state_dim=3, action_dim=1, config=config,
    ...     continuous=True, shared_network=False
    ... )

    Notes
    -----
    Complexity Analysis:
        - select_action: O(forward_pass)
        - update: O(T * forward_pass + T * backward_pass)
        - Memory: O(T * state_dim + network_params)

    GAE Computation (compute_gae):
        Time: O(T) - single backward pass
        Space: O(T) - storing advantages

    Hyperparameter Guidelines:
        | Parameter      | Typical Range | Effect                     |
        |----------------|---------------|----------------------------|
        | gae_lambda     | 0.9-0.99      | Bias-variance trade-off    |
        | entropy_coef   | 0.001-0.05    | Exploration encouragement  |
        | value_coef     | 0.25-1.0      | Critic learning speed      |
        | max_grad_norm  | 0.5-1.0       | Training stability         |

    Shared vs Separate Networks:
        Shared:
            + Fewer parameters, faster training
            + Feature transfer between tasks
            - Potential task interference

        Separate:
            + Independent optimization
            + Easier hyperparameter tuning
            - More parameters

    Summary:
        A2C is a practical, stable policy gradient algorithm suitable for
        a wide range of tasks. With GAE, it provides flexible control over
        the bias-variance trade-off. The shared network architecture enables
        efficient training while entropy regularization maintains exploration.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TrainingConfig = None,
        continuous: bool = False,
        use_gae: bool = True,
        shared_network: bool = True,
    ):
        if config is None:
            config = TrainingConfig()
        super().__init__(config)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.use_gae = use_gae
        self.shared_network = shared_network

        if shared_network:
            # Shared actor-critic network
            self.model = ActorCriticNetwork(
                state_dim,
                action_dim,
                hidden_dim=256,
                continuous=continuous,
            ).to(self.device)

            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.lr_actor,
            )
        else:
            # Separate networks
            if continuous:
                self.policy = ContinuousPolicy(state_dim, action_dim).to(self.device)
            else:
                self.policy = DiscretePolicy(state_dim, action_dim).to(self.device)

            self.value_net = ValueNetwork(state_dim).to(self.device)

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
            Dictionary containing log_prob, value, entropy.
        """
        state_t = self._to_tensor(state).unsqueeze(0)

        if self.shared_network:
            with torch.no_grad() if deterministic else torch.enable_grad():
                action, log_prob, entropy, value = self.model.get_action_and_value(
                    state_t
                )

                if self.continuous:
                    action_out = action.squeeze(0).cpu().numpy()
                else:
                    action_out = action.item()
        else:
            with torch.no_grad():
                value = self.value_net(state_t)

            with torch.no_grad() if deterministic else torch.enable_grad():
                if self.continuous:
                    action, log_prob, entropy = self.policy.sample(state_t, deterministic)
                    action_out = action.squeeze(0).cpu().numpy()
                else:
                    if deterministic:
                        logits = self.policy(state_t)
                        action_out = logits.argmax(dim=-1).item()
                        log_prob = torch.zeros(1)
                        entropy = torch.zeros(1)
                    else:
                        action, log_prob, entropy = self.policy.sample(state_t)
                        action_out = action.item()

        return action_out, {
            "log_prob": log_prob,
            "value": value if self.shared_network else value.squeeze(),
            "entropy": entropy,
        }

    def update(
        self,
        buffer: EpisodeBuffer,
        next_state: Optional[np.ndarray] = None,
        done: bool = True,
    ) -> Dict[str, float]:
        """
        Update actor and critic using collected experience.

        Parameters
        ----------
        buffer : EpisodeBuffer
            Collected trajectory data.
        next_state : np.ndarray, optional
            Final state for bootstrapping (if not terminal).
        done : bool, default=True
            Whether episode ended.

        Returns
        -------
        Dict[str, float]
            Training metrics including policy_loss, value_loss, entropy.

        Notes
        -----
        Update Procedure:
            1. Compute next state value for bootstrapping
            2. Compute advantages (GAE or n-step)
            3. Normalize advantages
            4. Compute combined loss
            5. Backpropagate and update
        """
        # Bootstrap value for non-terminal states
        if next_state is not None and not done:
            next_state_t = self._to_tensor(next_state).unsqueeze(0)
            with torch.no_grad():
                if self.shared_network:
                    next_value = self.model.get_value(next_state_t).item()
                else:
                    next_value = self.value_net(next_state_t).item()
        else:
            next_value = 0.0

        # Extract values from buffer
        values_list = []
        for v in buffer.values:
            if isinstance(v, torch.Tensor):
                values_list.append(v.item() if v.dim() == 0 else v.squeeze().item())
            else:
                values_list.append(float(v))

        # Compute advantages
        if self.use_gae:
            advantages, returns = compute_gae(
                buffer.rewards,
                values_list,
                next_value,
                buffer.dones,
                self.config.gamma,
                self.config.gae_lambda,
            )
        else:
            returns = compute_n_step_returns(
                buffer.rewards,
                values_list,
                next_value,
                buffer.dones,
                self.config.gamma,
                self.config.n_steps,
            )
            values_t = torch.tensor(values_list, dtype=torch.float32)
            advantages = returns - values_t

        # Normalize advantages
        advantages = self._normalize_advantages(advantages)

        # Get log_probs and values tensors
        log_probs = torch.stack(buffer.log_probs)

        if self.shared_network:
            values = torch.stack([v.squeeze() if isinstance(v, torch.Tensor) else torch.tensor(v)
                                  for v in buffer.values])
        else:
            values = torch.cat(buffer.values).squeeze()

        # Ensure values has correct shape
        if values.dim() > 1:
            values = values.squeeze()

        # Compute losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        if buffer.entropies:
            entropies = torch.stack(buffer.entropies)
            entropy_bonus = entropies.mean()
        else:
            entropy_bonus = torch.tensor(0.0)

        # Combined loss
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy_bonus
        )

        # Optimization
        if self.shared_network:
            self.optimizer.zero_grad()
            total_loss.backward()

            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
            self.optimizer.step()
        else:
            # Update both networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()

            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )
                nn.utils.clip_grad_norm_(
                    self.value_net.parameters(),
                    self.config.max_grad_norm,
                )

            self.policy_optimizer.step()
            self.value_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_bonus.item() if isinstance(entropy_bonus, torch.Tensor) else 0.0,
            "total_loss": total_loss.item(),
            "mean_advantage": advantages.mean().item(),
        }
