"""
Soft Actor-Critic (SAC)
=======================

Core Idea
---------
Maximum entropy reinforcement learning with automatic temperature tuning.
Learn a stochastic policy that maximizes both expected return AND policy entropy.

Mathematical Theory
-------------------
**Maximum Entropy Objective**:

Standard RL maximizes expected return. SAC adds an entropy bonus:

.. math::

    J(\\pi) = \\sum_{t=0}^{T} \\mathbb{E}_{(s_t, a_t) \\sim \\rho_\\pi}
    [r(s_t, a_t) + \\alpha \\mathcal{H}(\\pi(\\cdot|s_t))]

where :math:`\\mathcal{H}(\\pi) = -\\mathbb{E}[\\log \\pi(a|s)]` is entropy.

**Intuition**: High entropy = more randomness = more exploration.
The temperature :math:`\\alpha` controls the exploration-exploitation tradeoff.

**Soft Value Functions**:

Soft Q-function includes entropy of future policies:

.. math::

    Q^\\pi(s, a) = r + \\gamma \\mathbb{E}_{s'}[V^\\pi(s')]

Soft state value:

.. math::

    V^\\pi(s) = \\mathbb{E}_{a \\sim \\pi}[Q^\\pi(s, a) - \\alpha \\log \\pi(a|s)]

**Soft Bellman Backup**:

.. math::

    y = r + \\gamma (1-d) (\\min_{i=1,2} Q_{\\phi'_i}(s', a') - \\alpha \\log \\pi(a'|s'))

Note: Unlike TD3, the target includes the entropy term.

**Actor Update** (entropy-regularized policy gradient):

.. math::

    J_\\pi(\\theta) = \\mathbb{E}_{s \\sim \\mathcal{D}, a \\sim \\pi_\\theta}
    [\\alpha \\log \\pi_\\theta(a|s) - Q_\\phi(s, a)]

Minimize log-probability (encourage diversity) while maximizing Q.

**Automatic Temperature Adjustment**:

Learn :math:`\\alpha` to maintain target entropy :math:`\\bar{\\mathcal{H}}`:

.. math::

    J(\\alpha) = \\mathbb{E}_{a \\sim \\pi}[-\\alpha (\\log \\pi(a|s) + \\bar{\\mathcal{H}})]

Target entropy typically set to :math:`-\\dim(\\mathcal{A})` (negative action dim).

Problem Statement
-----------------
DDPG and TD3 use deterministic policies requiring external exploration noise,
which is sensitive to tuning. SAC addresses this by:

1. **Stochastic Policy**: Natural exploration without noise schedules
2. **Entropy Maximization**: Principled exploration-exploitation balance
3. **Automatic Temperature**: No manual α tuning required

The maximum entropy framework also provides:

- Better robustness to model errors (captures multiple near-optimal solutions)
- Improved transferability (policy doesn't overfit to single trajectory)
- Faster learning (entropy prevents premature convergence)

Algorithm Comparison
--------------------
+----------------------+-----------+------------------+------------------+
| Feature              | DDPG      | TD3              | SAC              |
+======================+===========+==================+==================+
| Policy Type          | Determ.   | Deterministic    | Stochastic       |
| Exploration          | OU/Gauss  | Gaussian         | Entropy          |
| Q-Networks           | 1         | 2                | 2                |
| Entropy Regularized  | No        | No               | Yes              |
| Temperature Tuning   | N/A       | N/A              | Automatic        |
| Stability            | Low       | Medium           | High             |
| Sample Efficiency    | Good      | Good             | Best             |
+----------------------+-----------+------------------+------------------+

**Advantages**:

- Most stable of the three algorithms
- Automatic exploration via entropy
- No exploration noise hyperparameters
- Better generalization

**Disadvantages**:

- More compute per step (reparameterized sampling)
- Temperature learning can be unstable initially
- Slightly more complex implementation

Complexity
----------
- Time per update: Similar to TD3 + entropy computation
- Space: 2 critics + 1 actor + optional log_alpha parameter
- Sample complexity: Often lower than TD3 due to better exploration

Summary
-------
SAC is the current state-of-the-art for continuous control. The maximum
entropy framework provides principled exploration, the stochastic policy
naturally handles multimodal action distributions, and automatic temperature
tuning eliminates a critical hyperparameter. SAC should be the default choice
for continuous control unless there's a specific reason to use TD3 (e.g.,
when deterministic behavior is required).

References
----------
.. [1] Haarnoja et al. (2018). "Soft Actor-Critic: Off-Policy Maximum
       Entropy Deep Reinforcement Learning with a Stochastic Actor".
       ICML. https://arxiv.org/abs/1801.01290
.. [2] Haarnoja et al. (2018). "Soft Actor-Critic Algorithms and
       Applications". https://arxiv.org/abs/1812.05905
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import copy
import numpy as np
import torch
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import BaseConfig
from core.buffer import ReplayBuffer
from core.networks import GaussianActor, TwinQNetwork
from core.base_agent import BaseAgent


@dataclass
class SACConfig(BaseConfig):
    """
    SAC Algorithm Configuration
    ============================

    Core Idea
    ---------
    Configuration for Soft Actor-Critic with automatic temperature adjustment.

    Mathematical Theory
    -------------------
    **Temperature Parameter** (alpha / initial_alpha):

    Controls entropy bonus weight in the objective:

    .. math::

        J = \\mathbb{E}[r + \\alpha \\mathcal{H}(\\pi)]

    - :math:`\\alpha \\to 0`: Pure return maximization (like TD3)
    - :math:`\\alpha \\to \\infty`: Maximum entropy (random policy)

    **Target Entropy** (target_entropy):

    Desired entropy level, typically :math:`-\\dim(\\mathcal{A})`:

    .. math::

        \\bar{\\mathcal{H}} = -|\\mathcal{A}|

    The automatic tuning adjusts α to maintain this entropy.

    **Learning Rate for Temperature** (lr_alpha):

    Separate learning rate for temperature optimization.
    Often set equal to critic learning rate.

    Attributes
    ----------
    initial_alpha : float
        Initial temperature value. Default: 0.2
    auto_alpha : bool
        Whether to automatically tune temperature. Default: True
    target_entropy : Optional[float]
        Target entropy. If None, set to -action_dim.
    lr_alpha : float
        Learning rate for temperature. Default: 3e-4

    Example
    -------
    >>> config = SACConfig(
    ...     state_dim=11,
    ...     action_dim=3,
    ...     max_action=1.0,
    ...     auto_alpha=True,
    ...     target_entropy=-3.0,  # -action_dim
    ... )
    >>> agent = SACAgent(config)
    """

    # SAC-specific parameters
    initial_alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: Optional[float] = None
    lr_alpha: float = 3e-4

    def validate(self) -> None:
        """Validate SAC-specific parameters."""
        super().validate()

        if self.initial_alpha < 0:
            raise ValueError(
                f"initial_alpha must be non-negative, got {self.initial_alpha}"
            )

        if self.lr_alpha <= 0:
            raise ValueError(f"lr_alpha must be positive, got {self.lr_alpha}")


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic Agent
    =======================

    Core Idea
    ---------
    Maximum entropy actor-critic for stable, sample-efficient continuous control.
    Learns a stochastic policy that maximizes both return and entropy.

    Mathematical Theory
    -------------------
    **Network Architecture**:

    - Actor :math:`\\pi_\\theta(a|s)`: Gaussian policy outputting mean and std
    - Twin Critics :math:`Q_{\\phi_1}, Q_{\\phi_2}`: Independent Q-networks
    - Temperature :math:`\\alpha`: Learnable entropy weight

    **Soft Policy Evaluation** (Critic Update):

    .. math::

        L(\\phi_i) = \\mathbb{E}[(Q_{\\phi_i}(s,a) - y)^2]

    where:

    .. math::

        y = r + \\gamma(1-d)(\\min_{j} Q_{\\phi'_j}(s', a') - \\alpha \\log \\pi(a'|s'))

    **Soft Policy Improvement** (Actor Update):

    .. math::

        L(\\theta) = \\mathbb{E}_{s, a \\sim \\pi}[\\alpha \\log \\pi_\\theta(a|s)
        - \\min_{i} Q_{\\phi_i}(s, a)]

    **Temperature Update** (if auto_alpha=True):

    .. math::

        L(\\alpha) = \\mathbb{E}_a[-\\alpha(\\log \\pi(a|s) + \\bar{\\mathcal{H}})]

    Problem Statement
    -----------------
    SAC addresses key limitations of deterministic policies:

    | Issue                | DDPG/TD3          | SAC Solution          |
    |----------------------|-------------------|-----------------------|
    | Exploration          | External noise    | Entropy maximization  |
    | Mode collapse        | Possible          | Prevented by entropy  |
    | Hyperparameter sens. | High (noise σ)    | Low (auto α)          |
    | Robustness           | Brittle           | More robust           |

    Algorithm Comparison
    --------------------
    **vs. TD3**:

    - SAC: Stochastic, better exploration, auto temperature
    - TD3: Deterministic, simpler, explicit noise control

    **vs. PPO** (on-policy):

    - SAC: Off-policy, more sample efficient
    - PPO: On-policy, more stable for some tasks

    Complexity
    ----------
    Per update step:

    - Critics: 2× forward + backward (same as TD3)
    - Actor: 1× forward + backward with reparameterization
    - Temperature: 1× scalar update (negligible)

    Parameters
    ----------
    config : SACConfig
        Algorithm configuration.

    Attributes
    ----------
    actor : GaussianActor
        Stochastic Gaussian policy.
    critic : TwinQNetwork
        Twin Q-value networks.
    critic_target : TwinQNetwork
        Target Q-networks.
    log_alpha : torch.Tensor
        Log of temperature parameter (for stable optimization).
    target_entropy : float
        Target entropy for automatic tuning.

    Example
    -------
    >>> config = SACConfig(state_dim=3, action_dim=1, max_action=2.0)
    >>> agent = SACAgent(config)
    >>> state = np.random.randn(3)
    >>> action = agent.select_action(state)  # Sampled from Gaussian
    >>> action_eval = agent.select_action(state, deterministic=True)  # Mean action
    >>> agent.store_transition(state, action, reward=1.0, next_state, done=False)
    >>> metrics = agent.update()
    >>> print(f"Alpha: {metrics['alpha']:.4f}")

    References
    ----------
    .. [1] Haarnoja et al. (2018). "Soft Actor-Critic: Off-Policy Maximum
           Entropy Deep Reinforcement Learning". ICML.
    """

    def __init__(self, config: SACConfig) -> None:
        """
        Initialize SAC agent.

        Parameters
        ----------
        config : SACConfig
            Algorithm configuration.
        """
        super().__init__()

        config.validate()
        self.config = config
        self.device = config.get_device()

        # Initialize networks
        self._init_networks()

        # Initialize optimizers
        self._init_optimizers()

        # Initialize temperature (alpha)
        self._init_temperature()

        # Initialize replay buffer
        self.buffer = ReplayBuffer(
            capacity=config.buffer_size,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
        )

        self.total_updates = 0

    def _init_networks(self) -> None:
        """Initialize actor and twin critic networks."""
        cfg = self.config

        # Stochastic actor (Gaussian)
        self.actor = GaussianActor(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            max_action=cfg.max_action,
            hidden_dims=cfg.hidden_dims,
        ).to(self.device)

        # Twin critic networks
        self.critic = TwinQNetwork(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_dims=cfg.hidden_dims,
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)

        # Freeze target networks
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def _init_optimizers(self) -> None:
        """Initialize optimizers for actor and critics."""
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.lr_actor,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.lr_critic,
        )

    def _init_temperature(self) -> None:
        """Initialize temperature parameter and its optimizer."""
        # Target entropy: -dim(A) by default
        if self.config.target_entropy is None:
            self.target_entropy = -float(self.config.action_dim)
        else:
            self.target_entropy = self.config.target_entropy

        # Log alpha for stable optimization (alpha = exp(log_alpha) > 0)
        self.log_alpha = torch.tensor(
            np.log(self.config.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.config.auto_alpha,
        )

        if self.config.auto_alpha:
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=self.config.lr_alpha,
            )

    @property
    def alpha(self) -> float:
        """Current temperature value."""
        return self.log_alpha.exp().item()

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action from stochastic policy.

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        deterministic : bool, default=False
            If True, return mean action (for evaluation).
            If False, sample from Gaussian policy.

        Returns
        -------
        np.ndarray
            Action bounded by tanh to [-max_action, max_action].

        Notes
        -----
        During training, actions are sampled for natural exploration:

        .. math::

            a \\sim \\pi_\\theta(\\cdot|s) = \\mathcal{N}(\\mu_\\theta(s), \\sigma_\\theta(s))

        During evaluation, use the mean for consistent behavior:

        .. math::

            a = \\mu_\\theta(s)
        """
        state_tensor = self._to_tensor(state).unsqueeze(0)

        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor, deterministic=deterministic)

        return action.cpu().numpy()[0]

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """
        Perform one gradient update step.

        Returns
        -------
        Dict[str, float]
            Training metrics:
            - critic_loss: TD error for Q-networks
            - actor_loss: Entropy-regularized policy loss
            - alpha: Current temperature value
            - alpha_loss: Temperature optimization loss (if auto_alpha)
            - entropy: Current policy entropy estimate

        Notes
        -----
        **Update Sequence**:

        1. **Critic Update**: Minimize soft Bellman error
           :math:`y = r + \\gamma(\\min Q' - \\alpha \\log \\pi')`

        2. **Actor Update**: Maximize Q minus entropy penalty
           :math:`L = \\alpha \\log \\pi - Q`

        3. **Temperature Update** (if auto_alpha):
           Adjust α to maintain target entropy

        4. **Target Update**: Soft update target critics
        """
        if not self.buffer.is_ready(self.config.batch_size):
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size, device=self.device
        )

        self.total_updates += 1
        metrics: Dict[str, float] = {}
        alpha = self.log_alpha.exp()

        # ====== Critic Update ======
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Target Q-values with entropy penalty
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # Soft Bellman backup (includes entropy term)
            target_q = rewards + self.config.gamma * (1 - dones) * (
                target_q - alpha * next_log_probs
            )

        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics["critic_loss"] = critic_loss.item()
        metrics["q1_mean"] = current_q1.mean().item()
        metrics["q2_mean"] = current_q2.mean().item()

        # ====== Actor Update ======
        # Sample actions from current policy (with gradients)
        sampled_actions, log_probs = self.actor.sample(states)

        # Q-values for sampled actions
        q1, q2 = self.critic(states, sampled_actions)
        min_q = torch.min(q1, q2)

        # Actor loss: maximize Q, minimize log_prob (maximize entropy)
        actor_loss = (alpha.detach() * log_probs - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics["actor_loss"] = actor_loss.item()
        metrics["entropy"] = -log_probs.mean().item()

        # ====== Temperature Update ======
        if self.config.auto_alpha:
            # Alpha loss: adjust to maintain target entropy
            alpha_loss = -(
                self.log_alpha * (log_probs.detach() + self.target_entropy)
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            metrics["alpha_loss"] = alpha_loss.item()

        metrics["alpha"] = alpha.item()

        # ====== Target Network Update ======
        self._soft_update(self.critic, self.critic_target, self.config.tau)

        return metrics

    def save(self, path: str) -> None:
        """Save agent state to disk."""
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "config": self.config.to_dict(),
            "total_updates": self.total_updates,
        }

        if self.config.auto_alpha:
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load agent state from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        self.log_alpha = checkpoint["log_alpha"].to(self.device)
        self.log_alpha.requires_grad = self.config.auto_alpha

        if self.config.auto_alpha and "alpha_optimizer" in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

        self.total_updates = checkpoint.get("total_updates", 0)

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return self.config.to_dict()


if __name__ == "__main__":
    print("Testing SAC Agent...")

    # Test 1: Initialization
    config = SACConfig(
        state_dim=4,
        action_dim=2,
        max_action=1.0,
        hidden_dims=[64, 64],
        buffer_size=1000,
        batch_size=32,
        auto_alpha=True,
    )
    agent = SACAgent(config)
    print("  [PASS] Agent initialization")

    # Test 2: Action selection (stochastic)
    state = np.random.randn(4).astype(np.float32)
    action1 = agent.select_action(state)
    action2 = agent.select_action(state)
    assert action1.shape == (2,)
    assert not np.allclose(action1, action2), "Stochastic actions should differ"
    print("  [PASS] Stochastic action selection")

    # Test 3: Deterministic action
    action_det1 = agent.select_action(state, deterministic=True)
    action_det2 = agent.select_action(state, deterministic=True)
    assert np.allclose(action_det1, action_det2), "Deterministic actions should match"
    print("  [PASS] Deterministic action selection")

    # Test 4: Store transitions and update
    for _ in range(100):
        s = np.random.randn(4).astype(np.float32)
        a = np.random.randn(2).astype(np.float32)
        r = float(np.random.randn())
        ns = np.random.randn(4).astype(np.float32)
        d = np.random.random() < 0.1
        agent.store_transition(s, a, r, ns, d)

    metrics = agent.update()
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert "alpha" in metrics
    assert "alpha_loss" in metrics  # auto_alpha=True
    assert "entropy" in metrics
    print(f"  [PASS] Update (alpha={metrics['alpha']:.4f}, entropy={metrics['entropy']:.4f})")

    # Test 5: Temperature bounds
    assert agent.alpha > 0, "Alpha should be positive"
    print("  [PASS] Alpha positivity")

    # Test 6: Save and load
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        agent.save(temp_path)
        agent2 = SACAgent(config)
        agent2.load(temp_path)

        for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
            assert torch.allclose(p1, p2)
        print("  [PASS] Save and load")
    finally:
        os.unlink(temp_path)

    # Test 7: Disabled auto_alpha
    config_fixed = SACConfig(
        state_dim=4,
        action_dim=2,
        max_action=1.0,
        hidden_dims=[64, 64],
        buffer_size=1000,
        batch_size=32,
        auto_alpha=False,
        initial_alpha=0.5,
    )
    agent_fixed = SACAgent(config_fixed)

    for _ in range(100):
        agent_fixed.store_transition(
            np.random.randn(4).astype(np.float32),
            np.random.randn(2).astype(np.float32),
            float(np.random.randn()),
            np.random.randn(4).astype(np.float32),
            False,
        )

    metrics_fixed = agent_fixed.update()
    assert "alpha_loss" not in metrics_fixed
    assert abs(agent_fixed.alpha - 0.5) < 1e-5, "Alpha should stay fixed"
    print("  [PASS] Fixed alpha mode")

    print("\nAll SAC tests passed!")
