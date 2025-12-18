"""
Deep Deterministic Policy Gradient (DDPG)
=========================================

Core Idea
---------
Extend DQN to continuous action spaces using deterministic policy gradient.
Actor outputs actions directly; critic evaluates state-action values.

Mathematical Theory
-------------------
DDPG combines three key insights:

**1. Deterministic Policy Gradient Theorem** [Silver et al., 2014]:

For deterministic policy :math:`\\mu_\\theta(s)`:

.. math::

    \\nabla_\\theta J(\\theta) = \\mathbb{E}_{s \\sim \\rho^\\mu}
    [\\nabla_a Q(s, a)|_{a=\\mu(s)} \\nabla_\\theta \\mu_\\theta(s)]

Unlike stochastic PG, no integration over actions—just chain rule through Q.

**2. Off-Policy Learning with Replay Buffer**:

Store transitions :math:`(s, a, r, s', d)` from behavior policy :math:`\\beta`:

.. math::

    \\mathcal{D} = \\{(s_i, a_i, r_i, s'_i, d_i)\\}_{i=1}^{N}

Train on random batches, enabling sample reuse.

**3. Target Networks for Stability**:

Slowly-updated target networks provide stable TD targets:

.. math::

    y = r + \\gamma Q_{\\phi'}(s', \\mu_{\\theta'}(s'))

Soft update: :math:`\\phi' \\leftarrow \\tau\\phi + (1-\\tau)\\phi'`

**Critic Update** (minimize TD error):

.. math::

    L(\\phi) = \\mathbb{E}_{(s,a,r,s',d) \\sim \\mathcal{D}}
    [(Q_\\phi(s, a) - y)^2]

**Actor Update** (maximize Q-value):

.. math::

    \\nabla_\\theta J \\approx \\frac{1}{N} \\sum_i
    \\nabla_a Q_\\phi(s_i, a)|_{a=\\mu_\\theta(s_i)}
    \\nabla_\\theta \\mu_\\theta(s_i)

Problem Statement
-----------------
Before DDPG, continuous control required:

1. Discretization (exponential action space)
2. Policy gradient with high variance
3. On-policy methods (sample inefficient)

DDPG enables efficient continuous control by:

- Deterministic policy: No integration over actions
- Off-policy learning: Reuse all collected data
- Neural function approximation: Scalable to high dimensions

Algorithm Comparison
--------------------
+------------------+------------+------------------+---------------+
| Feature          | DDPG       | TD3              | SAC           |
+==================+============+==================+===============+
| Policy           | Determ.    | Deterministic    | Stochastic    |
| Q-Networks       | 1          | 2 (twin)         | 2 (twin)      |
| Exploration      | OU/Gauss   | Gaussian + clip  | Entropy       |
| Policy Update    | Every step | Delayed          | Every step    |
| Stability        | Low        | Medium           | High          |
+------------------+------------+------------------+---------------+

**Advantages**:
- Simple implementation
- Good sample efficiency (off-policy)
- Works for continuous control

**Disadvantages**:
- Q-value overestimation → unstable training
- Sensitive to hyperparameters
- Exploration noise requires tuning

Complexity
----------
- Time per update: O(B × (state_dim + action_dim) × hidden_dim × L)
  where B=batch_size, L=number of layers
- Space: O(4 × network_params) for actor, critic, and their targets
- Sample complexity: ~1M steps for MuJoCo locomotion tasks

Summary
-------
DDPG pioneered deep RL for continuous control. While superseded by TD3/SAC
in performance, it remains foundational for understanding actor-critic
methods. Key insight: deterministic policies enable efficient gradient
computation via the chain rule through the Q-function.

References
----------
.. [1] Lillicrap et al. (2016). "Continuous control with deep reinforcement
       learning". ICLR. https://arxiv.org/abs/1509.02971
.. [2] Silver et al. (2014). "Deterministic Policy Gradient Algorithms".
       ICML. http://proceedings.mlr.press/v32/silver14.pdf
"""

from dataclasses import dataclass, field
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
from core.networks import DeterministicActor, QNetwork
from core.base_agent import BaseAgent


@dataclass
class DDPGConfig(BaseConfig):
    """
    DDPG Algorithm Configuration
    =============================

    Core Idea
    ---------
    Hyperparameters specific to DDPG, extending the base configuration
    with exploration noise settings.

    Mathematical Theory
    -------------------
    **Exploration Noise**:

    DDPG uses additive Gaussian noise for exploration:

    .. math::

        a_t = \\mu_\\theta(s_t) + \\mathcal{N}(0, \\sigma^2)

    The noise scale :math:`\\sigma` (exploration_noise) controls the
    exploration-exploitation tradeoff.

    Attributes
    ----------
    exploration_noise : float
        Standard deviation of Gaussian exploration noise.
        Higher values encourage exploration but may hurt performance.
        Typical range: [0.1, 0.3]

    Example
    -------
    >>> config = DDPGConfig(
    ...     state_dim=11,
    ...     action_dim=3,
    ...     max_action=1.0,
    ...     exploration_noise=0.1,
    ... )
    >>> agent = DDPGAgent(config)
    """

    exploration_noise: float = 0.1

    def validate(self) -> None:
        """Validate DDPG-specific parameters."""
        super().validate()

        if self.exploration_noise < 0:
            raise ValueError(
                f"exploration_noise must be non-negative, got {self.exploration_noise}"
            )


class DDPGAgent(BaseAgent):
    """
    Deep Deterministic Policy Gradient Agent
    =========================================

    Core Idea
    ---------
    Actor-critic agent for continuous control with deterministic policy
    and off-policy learning via experience replay.

    Mathematical Theory
    -------------------
    **Network Architecture**:

    - Actor :math:`\\mu_\\theta`: MLP mapping state to deterministic action
    - Critic :math:`Q_\\phi`: MLP mapping (state, action) to Q-value

    **Training Loop**:

    1. Collect: :math:`a = \\mu_\\theta(s) + \\epsilon`, store :math:`(s,a,r,s',d)`
    2. Sample: Random batch from replay buffer
    3. Critic: Minimize :math:`L = (Q(s,a) - y)^2`
    4. Actor: Maximize :math:`Q(s, \\mu(s))`
    5. Targets: Soft update :math:`\\theta' \\leftarrow \\tau\\theta + (1-\\tau)\\theta'`

    **TD Target**:

    .. math::

        y = r + \\gamma (1 - d) Q_{\\phi'}(s', \\mu_{\\theta'}(s'))

    The :math:`(1-d)` term masks the next-state value at terminal states.

    Problem Statement
    -----------------
    DDPG addresses the challenge of applying deep RL to continuous action
    spaces where DQN's argmax is intractable. By learning a deterministic
    policy, the argmax becomes trivial: just evaluate :math:`\\mu(s)`.

    Algorithm Comparison
    --------------------
    **vs. TD3**:

    - DDPG: Single Q-network, no policy delay, no target smoothing
    - TD3: Twin Q-networks, delayed updates, target policy smoothing
    - Result: TD3 is more stable but DDPG is simpler

    **vs. SAC**:

    - DDPG: Deterministic policy, external exploration
    - SAC: Stochastic policy, entropy bonus
    - Result: SAC explores more effectively but requires temperature tuning

    Complexity
    ----------
    Per update step:

    - Forward passes: 4 (actor, critic for current + target networks)
    - Backward passes: 2 (actor + critic)
    - Time: O(B × D × H) where B=batch, D=input_dim, H=hidden_dim

    Parameters
    ----------
    config : DDPGConfig
        Algorithm configuration.

    Attributes
    ----------
    actor : DeterministicActor
        Policy network.
    actor_target : DeterministicActor
        Target policy network.
    critic : QNetwork
        Q-value network.
    critic_target : QNetwork
        Target Q-value network.
    buffer : ReplayBuffer
        Experience replay memory.

    Example
    -------
    >>> config = DDPGConfig(state_dim=3, action_dim=1, max_action=2.0)
    >>> agent = DDPGAgent(config)
    >>> state = np.random.randn(3)
    >>> action = agent.select_action(state)
    >>> agent.store_transition(state, action, reward=1.0, next_state, done=False)
    >>> metrics = agent.update()

    References
    ----------
    .. [1] Lillicrap et al. (2016). "Continuous control with deep
           reinforcement learning". ICLR.
    """

    def __init__(self, config: DDPGConfig) -> None:
        """
        Initialize DDPG agent.

        Parameters
        ----------
        config : DDPGConfig
            Algorithm configuration. Must be validated before use.
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
        """Initialize actor and critic networks with target copies."""
        cfg = self.config

        # Actor network
        self.actor = DeterministicActor(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            max_action=cfg.max_action,
            hidden_dims=cfg.hidden_dims,
        ).to(self.device)

        # Target actor (copy of actor)
        self.actor_target = copy.deepcopy(self.actor)

        # Critic network
        self.critic = QNetwork(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_dims=cfg.hidden_dims,
        ).to(self.device)

        # Target critic (copy of critic)
        self.critic_target = copy.deepcopy(self.critic)

        # Freeze target networks (no gradient computation)
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def _init_optimizers(self) -> None:
        """Initialize Adam optimizers for actor and critic."""
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
        Select action using deterministic policy with optional noise.

        Parameters
        ----------
        state : np.ndarray
            Current observation, shape (state_dim,).
        deterministic : bool, default=False
            If True, return policy output without exploration noise.

        Returns
        -------
        np.ndarray
            Action to execute, shape (action_dim,).
            Clipped to [-max_action, max_action].

        Notes
        -----
        During training, Gaussian noise is added for exploration:

        .. math::

            a = \\text{clip}(\\mu_\\theta(s) + \\mathcal{N}(0, \\sigma^2),
            -a_{max}, a_{max})
        """
        state_tensor = self._to_tensor(state).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if not deterministic and self.training:
            # Add exploration noise
            noise = np.random.normal(
                0, self.config.exploration_noise * self.config.max_action,
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
        """
        Store transition in replay buffer.

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        action : np.ndarray
            Action executed.
        reward : float
            Reward received.
        next_state : np.ndarray
            Next observation.
        done : bool
            Episode termination flag.
        """
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """
        Perform one gradient update step.

        Returns
        -------
        Dict[str, float]
            Training metrics:
            - critic_loss: TD error
            - actor_loss: Negative Q-value
            - q_mean: Mean Q-value estimate
            - q_max: Maximum Q-value in batch

        Notes
        -----
        Update procedure:

        1. Sample batch from replay buffer
        2. Compute TD target: :math:`y = r + \\gamma(1-d)Q'(s', \\mu'(s'))`
        3. Update critic: minimize :math:`(Q(s,a) - y)^2`
        4. Update actor: maximize :math:`Q(s, \\mu(s))`
        5. Soft update target networks
        """
        if not self.buffer.is_ready(self.config.batch_size):
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size, device=self.device
        )

        # ====== Critic Update ======
        with torch.no_grad():
            # Target actions from target actor
            next_actions = self.actor_target(next_states)

            # Target Q-values
            target_q = self.critic_target(next_states, next_actions)

            # TD target: y = r + γ(1-d)Q'(s', a')
            target_q = rewards + self.config.gamma * (1 - dones) * target_q

        # Current Q-values
        current_q = self.critic(states, actions)

        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q, target_q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ====== Actor Update ======
        # Actor loss: negative Q-value (we want to maximize Q)
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ====== Target Network Update ======
        self._soft_update(self.actor, self.actor_target, self.config.tau)
        self._soft_update(self.critic, self.critic_target, self.config.tau)

        self.total_updates += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_mean": current_q.mean().item(),
            "q_max": current_q.max().item(),
        }

    def save(self, path: str) -> None:
        """
        Save agent state to disk.

        Parameters
        ----------
        path : str
            Output file path.

        Saves
        -----
        - Actor and critic networks (online + target)
        - Optimizer states
        - Configuration
        - Training statistics
        """
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
        """
        Load agent state from disk.

        Parameters
        ----------
        path : str
            Input file path.
        """
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
    print("Testing DDPG Agent...")

    # Test 1: Initialization
    config = DDPGConfig(
        state_dim=4,
        action_dim=2,
        max_action=1.0,
        hidden_dims=[64, 64],
        buffer_size=1000,
        batch_size=32,
    )
    agent = DDPGAgent(config)
    print("  [PASS] Agent initialization")

    # Test 2: Action selection
    state = np.random.randn(4).astype(np.float32)
    action = agent.select_action(state)
    assert action.shape == (2,)
    assert (np.abs(action) <= 1.0).all()
    print("  [PASS] Action selection")

    # Test 3: Deterministic action
    action_det = agent.select_action(state, deterministic=True)
    action_det2 = agent.select_action(state, deterministic=True)
    assert np.allclose(action_det, action_det2)
    print("  [PASS] Deterministic action")

    # Test 4: Store transitions
    for _ in range(100):
        s = np.random.randn(4).astype(np.float32)
        a = np.random.randn(2).astype(np.float32)
        r = float(np.random.randn())
        ns = np.random.randn(4).astype(np.float32)
        d = np.random.random() < 0.1
        agent.store_transition(s, a, r, ns, d)

    assert len(agent.buffer) == 100
    print("  [PASS] Store transitions")

    # Test 5: Update step
    metrics = agent.update()
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert "q_mean" in metrics
    print(f"  [PASS] Update step (critic_loss={metrics['critic_loss']:.4f})")

    # Test 6: Save and load
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        agent.save(temp_path)
        assert os.path.exists(temp_path)

        # Create new agent and load
        agent2 = DDPGAgent(config)
        agent2.load(temp_path)

        # Verify parameters match
        for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
            assert torch.allclose(p1, p2)

        print("  [PASS] Save and load")
    finally:
        os.unlink(temp_path)

    # Test 7: Config validation
    try:
        invalid_config = DDPGConfig(state_dim=-1, action_dim=2)
        invalid_config.validate()
        assert False, "Should raise ValueError"
    except ValueError:
        print("  [PASS] Config validation")

    print("\nAll DDPG tests passed!")
