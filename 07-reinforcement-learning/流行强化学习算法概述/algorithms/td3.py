"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
=====================================================

Core Idea
---------
Address DDPG's overestimation and instability through three techniques:

1. **Clipped Double Q-Learning**: Use min(Q1, Q2) to combat overestimation
2. **Delayed Policy Updates**: Update actor less frequently than critic
3. **Target Policy Smoothing**: Add noise to target actions for regularization

Mathematical Theory
-------------------
**1. Clipped Double Q-Learning**

Standard Q-learning overestimates due to max operator:

.. math::

    \\mathbb{E}[\\max_a Q(s', a)] \\geq \\max_a \\mathbb{E}[Q(s', a)]

TD3 uses two independent critics and takes the minimum:

.. math::

    y = r + \\gamma \\min_{i=1,2} Q_{\\phi'_i}(s', \\tilde{a}')

This provides a lower bound that counteracts overestimation bias.

**2. Delayed Policy Updates**

Actor updates are delayed by factor d (typically 2):

.. math::

    \\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta J(\\theta)
    \\quad \\text{every } d \\text{ critic updates}

This allows the critic to stabilize before policy improvement,
reducing variance in the actor gradient.

**3. Target Policy Smoothing**

Noise is added to target actions to smooth the Q-function:

.. math::

    \\tilde{a}' = \\text{clip}(\\mu_{\\theta'}(s') + \\text{clip}(\\epsilon, -c, c),
    -a_{max}, a_{max})

where :math:`\\epsilon \\sim \\mathcal{N}(0, \\sigma^2)`.

This prevents the policy from exploiting Q-function errors at
specific action points by regularizing the value landscape.

**Critic Update**:

.. math::

    L(\\phi_i) = \\mathbb{E}[(Q_{\\phi_i}(s, a) - y)^2], \\quad i \\in \\{1, 2\\}

**Actor Update** (every d steps):

.. math::

    \\nabla_\\theta J \\approx \\frac{1}{N} \\sum_i
    \\nabla_a Q_{\\phi_1}(s_i, a)|_{a=\\mu_\\theta(s_i)}
    \\nabla_\\theta \\mu_\\theta(s_i)

Note: Only Q1 is used for the actor gradient (arbitrary choice, both valid).

Problem Statement
-----------------
DDPG suffers from:

1. **Overestimation**: Q-values drift upward, causing poor policies
2. **Instability**: High variance actor gradients from noisy Q-estimates
3. **Brittleness**: Policy exploits errors in Q-function approximation

TD3's three modifications directly target each problem:

1. Clipped double-Q → Lower bound on Q-estimates
2. Delayed updates → Reduced actor gradient variance
3. Target smoothing → Robust Q-function landscape

Algorithm Comparison
--------------------
+-----------------------+------------+------------------+
| Feature               | DDPG       | TD3              |
+=======================+============+==================+
| Q-Networks            | 1          | 2 (twin)         |
| Target Q computation  | Q(s', μ')  | min(Q1, Q2) - ε  |
| Policy update freq    | Every step | Every d steps    |
| Target action noise   | None       | Clipped Gaussian |
| Overestimation        | High       | Low              |
| Stability             | Low        | Medium-High      |
+-----------------------+------------+------------------+

**Advantages over DDPG**:

- More stable training (reduced Q-overestimation)
- Better final performance (less policy overfitting)
- More hyperparameter robust

**Disadvantages**:

- 2× critic parameters and computation
- Delayed actor updates slow initial learning
- Still requires external exploration noise

Complexity
----------
- Time per update: ~2× DDPG (two Q-networks)
- Space: O(6 × network_params) - actor, 2 critics, 3 targets
- Sample complexity: Similar to DDPG (~1M steps for MuJoCo)

Summary
-------
TD3 is DDPG done right. The three simple modifications—clipped double-Q,
delayed policy updates, and target smoothing—dramatically improve
stability and performance. TD3 is the recommended baseline for
deterministic continuous control before considering SAC.

References
----------
.. [1] Fujimoto et al. (2018). "Addressing Function Approximation Error
       in Actor-Critic Methods". ICML. https://arxiv.org/abs/1802.09477
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import copy
import numpy as np
import torch
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import BaseConfig
from core.buffer import ReplayBuffer
from core.networks import DeterministicActor, TwinQNetwork
from core.base_agent import BaseAgent


@dataclass
class TD3Config(BaseConfig):
    """
    TD3 Algorithm Configuration
    ============================

    Core Idea
    ---------
    Extends BaseConfig with TD3-specific parameters for the three
    key improvements: policy delay, target noise, and noise clipping.

    Mathematical Theory
    -------------------
    **Policy Delay** (policy_delay):

    Update actor every :math:`d` critic updates. Default :math:`d=2`
    balances learning speed with stability.

    **Target Noise** (target_noise):

    Smoothing noise added to target actions:

    .. math::

        \\epsilon \\sim \\mathcal{N}(0, \\sigma^2), \\quad
        \\sigma = \\text{target\\_noise}

    **Noise Clipping** (noise_clip):

    Bound on smoothing noise magnitude:

    .. math::

        \\tilde{\\epsilon} = \\text{clip}(\\epsilon, -c, c), \\quad
        c = \\text{noise\\_clip}

    Attributes
    ----------
    policy_delay : int
        Actor update frequency relative to critic. Default: 2
    target_noise : float
        Std of Gaussian noise for target policy smoothing. Default: 0.2
    noise_clip : float
        Maximum magnitude of target noise. Default: 0.5
    exploration_noise : float
        Std of exploration noise during data collection. Default: 0.1

    Example
    -------
    >>> config = TD3Config(
    ...     state_dim=11,
    ...     action_dim=3,
    ...     max_action=1.0,
    ...     policy_delay=2,
    ...     target_noise=0.2,
    ...     noise_clip=0.5,
    ... )
    >>> agent = TD3Agent(config)
    """

    # TD3-specific parameters
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1

    def validate(self) -> None:
        """Validate TD3-specific parameters."""
        super().validate()

        if self.policy_delay < 1:
            raise ValueError(
                f"policy_delay must be at least 1, got {self.policy_delay}"
            )

        if self.target_noise < 0:
            raise ValueError(
                f"target_noise must be non-negative, got {self.target_noise}"
            )

        if self.noise_clip < 0:
            raise ValueError(
                f"noise_clip must be non-negative, got {self.noise_clip}"
            )

        if self.exploration_noise < 0:
            raise ValueError(
                f"exploration_noise must be non-negative, got {self.exploration_noise}"
            )


class TD3Agent(BaseAgent):
    """
    Twin Delayed DDPG Agent
    =======================

    Core Idea
    ---------
    Enhanced DDPG with three modifications for stable continuous control:
    twin Q-networks, delayed policy updates, and target policy smoothing.

    Mathematical Theory
    -------------------
    **Network Architecture**:

    - Actor :math:`\\mu_\\theta`: Deterministic policy
    - Twin Critics :math:`Q_{\\phi_1}, Q_{\\phi_2}`: Independent Q-networks

    **Update Procedure**:

    Every step:

    1. Compute target with smoothed action:
       :math:`\\tilde{a}' = \\text{clip}(\\mu'(s') + \\text{clip}(\\epsilon, -c, c))`
    2. Compute target Q: :math:`y = r + \\gamma(1-d)\\min(Q'_1, Q'_2)`
    3. Update both critics: minimize :math:`(Q_i - y)^2`

    Every d steps:

    4. Update actor: maximize :math:`Q_1(s, \\mu(s))`
    5. Soft update all target networks

    Problem Statement
    -----------------
    TD3 systematically addresses DDPG's failure modes:

    | Problem            | Cause                  | TD3 Solution         |
    |--------------------|------------------------|----------------------|
    | Overestimation     | max over noisy Q       | min(Q1, Q2)          |
    | High variance      | Frequent actor updates | Delayed updates      |
    | Q exploitation     | Sharp Q landscape      | Target smoothing     |

    Algorithm Comparison
    --------------------
    **vs. DDPG**:

    - TD3: More stable, better final performance
    - DDPG: Simpler, faster per-step computation

    **vs. SAC**:

    - TD3: Deterministic, requires exploration noise tuning
    - SAC: Stochastic, automatic exploration via entropy

    Complexity
    ----------
    Per update step:

    - Critics: 2× forward + backward passes
    - Actor: 1× forward + backward (every d steps)
    - Time: O(2B × D × H) for critics

    Parameters
    ----------
    config : TD3Config
        Algorithm configuration.

    Attributes
    ----------
    actor : DeterministicActor
        Policy network.
    actor_target : DeterministicActor
        Target policy network.
    critic : TwinQNetwork
        Twin Q-value networks.
    critic_target : TwinQNetwork
        Target twin Q-networks.
    buffer : ReplayBuffer
        Experience replay memory.

    Example
    -------
    >>> config = TD3Config(state_dim=3, action_dim=1, max_action=2.0)
    >>> agent = TD3Agent(config)
    >>> state = np.random.randn(3)
    >>> action = agent.select_action(state)
    >>> agent.store_transition(state, action, reward=1.0, next_state, done=False)
    >>> metrics = agent.update()

    References
    ----------
    .. [1] Fujimoto et al. (2018). "Addressing Function Approximation Error
           in Actor-Critic Methods". ICML.
    """

    def __init__(self, config: TD3Config) -> None:
        """
        Initialize TD3 agent.

        Parameters
        ----------
        config : TD3Config
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

        # Actor network
        self.actor = DeterministicActor(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            max_action=cfg.max_action,
            hidden_dims=cfg.hidden_dims,
        ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)

        # Twin critic networks
        self.critic = TwinQNetwork(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_dims=cfg.hidden_dims,
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)

        # Freeze target networks
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def _init_optimizers(self) -> None:
        """Initialize optimizers for actor and both critics."""
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.lr_actor,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.lr_critic,
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action using deterministic policy with optional exploration noise.

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        deterministic : bool, default=False
            If True, return policy output without noise.

        Returns
        -------
        np.ndarray
            Action clipped to [-max_action, max_action].
        """
        state_tensor = self._to_tensor(state).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if not deterministic and self.training:
            noise = np.random.normal(
                0,
                self.config.exploration_noise * self.config.max_action,
                size=action.shape,
            )
            action = action + noise
            action = np.clip(action, -self.config.max_action, self.config.max_action)

        return action

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
            Training metrics including critic losses, actor loss (when updated),
            and Q-value statistics.

        Notes
        -----
        **Critic Update** (every step):

        1. Add clipped noise to target actions for smoothing
        2. Compute pessimistic target: :math:`y = r + \\gamma \\min(Q'_1, Q'_2)`
        3. Update both critics via MSE loss

        **Actor Update** (every policy_delay steps):

        4. Maximize :math:`Q_1(s, \\mu(s))` (use only Q1 for gradient)
        5. Soft update all target networks
        """
        if not self.buffer.is_ready(self.config.batch_size):
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size, device=self.device
        )

        self.total_updates += 1
        metrics: Dict[str, float] = {}

        # ====== Critic Update ======
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = torch.randn_like(actions) * self.config.target_noise
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)

            next_actions = self.actor_target(next_states)
            next_actions = (next_actions + noise).clamp(
                -self.config.max_action, self.config.max_action
            )

            # Clipped double Q-learning: use minimum of two target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # TD target
            target_q = rewards + self.config.gamma * (1 - dones) * target_q

        # Current Q-values from both critics
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss (sum of MSE for both networks)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics["critic_loss"] = critic_loss.item()
        metrics["q1_mean"] = current_q1.mean().item()
        metrics["q2_mean"] = current_q2.mean().item()

        # ====== Delayed Actor Update ======
        if self.total_updates % self.config.policy_delay == 0:
            # Actor loss: negative Q1 value
            actor_actions = self.actor(states)
            actor_loss = -self.critic.q1_forward(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            metrics["actor_loss"] = actor_loss.item()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target, self.config.tau)
            self._soft_update(self.critic, self.critic_target, self.config.tau)

        return metrics

    def save(self, path: str) -> None:
        """Save agent state to disk."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "config": self.config.to_dict(),
                "total_updates": self.total_updates,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_updates = checkpoint.get("total_updates", 0)

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return self.config.to_dict()


if __name__ == "__main__":
    print("Testing TD3 Agent...")

    # Test 1: Initialization
    config = TD3Config(
        state_dim=4,
        action_dim=2,
        max_action=1.0,
        hidden_dims=[64, 64],
        buffer_size=1000,
        batch_size=32,
        policy_delay=2,
    )
    agent = TD3Agent(config)
    print("  [PASS] Agent initialization")

    # Test 2: Action selection
    state = np.random.randn(4).astype(np.float32)
    action = agent.select_action(state)
    assert action.shape == (2,)
    assert (np.abs(action) <= 1.0).all()
    print("  [PASS] Action selection")

    # Test 3: Store transitions and update
    for _ in range(100):
        s = np.random.randn(4).astype(np.float32)
        a = np.random.randn(2).astype(np.float32)
        r = float(np.random.randn())
        ns = np.random.randn(4).astype(np.float32)
        d = np.random.random() < 0.1
        agent.store_transition(s, a, r, ns, d)

    # Update twice to trigger actor update (policy_delay=2)
    metrics1 = agent.update()
    assert "critic_loss" in metrics1
    assert "actor_loss" not in metrics1  # First update, no actor
    print("  [PASS] First update (critic only)")

    metrics2 = agent.update()
    assert "actor_loss" in metrics2  # Second update includes actor
    print("  [PASS] Second update (with delayed actor)")

    # Test 4: Twin Q-network produces different values
    states = torch.randn(32, 4)
    actions = torch.randn(32, 2)
    q1, q2 = agent.critic(states, actions)
    assert not torch.allclose(q1, q2), "Twin Qs should differ"
    print("  [PASS] Twin Q-networks")

    # Test 5: Save and load
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        agent.save(temp_path)
        agent2 = TD3Agent(config)
        agent2.load(temp_path)

        for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
            assert torch.allclose(p1, p2)
        print("  [PASS] Save and load")
    finally:
        os.unlink(temp_path)

    # Test 6: Config validation
    try:
        invalid_config = TD3Config(
            state_dim=4, action_dim=2, policy_delay=0
        )
        invalid_config.validate()
        assert False, "Should raise ValueError"
    except ValueError:
        print("  [PASS] Config validation")

    print("\nAll TD3 tests passed!")
