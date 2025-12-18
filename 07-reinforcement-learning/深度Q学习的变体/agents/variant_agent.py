"""
Unified DQN Variant Agent.

This module implements a unified agent supporting all DQN variants.

Core Idea (核心思想)
====================
通过配置和枚举选择不同的DQN变体，提供统一的训练和推理接口。

Supported Variants (支持的变体)
==============================
+------------------+--------------------------------------------------+
| Variant          | Key Innovation                                   |
+==================+==================================================+
| VANILLA          | Original DQN (Mnih et al., 2015)                 |
+------------------+--------------------------------------------------+
| DOUBLE           | Decoupled action selection/evaluation            |
+------------------+--------------------------------------------------+
| DUELING          | Value-advantage decomposition                    |
+------------------+--------------------------------------------------+
| NOISY            | Parametric exploration via noisy layers          |
+------------------+--------------------------------------------------+
| CATEGORICAL      | Full return distribution modeling (C51)          |
+------------------+--------------------------------------------------+
| DOUBLE_DUELING   | Combines Double and Dueling                      |
+------------------+--------------------------------------------------+
| RAINBOW          | All improvements combined                        |
+------------------+--------------------------------------------------+

Training Loop (训练循环)
========================
1. Select action: ε-greedy or noisy network
2. Store transition in replay buffer
3. Sample batch and compute TD target
4. Update Q-network via gradient descent
5. Periodically sync target network
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import DQNVariantConfig
from core.enums import DQNVariant
from core.types import FloatArray

from buffers.base import ReplayBuffer
from buffers.prioritized import PrioritizedReplayBuffer
from buffers.n_step import NStepReplayBuffer

from networks.base import DQNNetwork
from networks.dueling import DuelingNetwork
from networks.noisy import NoisyNetwork, NoisyDuelingNetwork
from networks.categorical import CategoricalNetwork
from networks.rainbow import RainbowNetwork


class DQNVariantAgent:
    """
    Unified DQN variant agent supporting all algorithmic improvements.

    Core Idea (核心思想)
    --------------------
    统一的Agent接口支持所有DQN变体，包括：

    - **Double DQN**: 解耦动作选择与评估，消除过估计
    - **Dueling DQN**: 分离状态价值与动作优势
    - **Noisy DQN**: 参数化噪声实现状态依赖探索
    - **Categorical DQN**: 建模完整回报分布
    - **Rainbow**: 组合所有改进

    Mathematical Foundation (数学基础)
    ----------------------------------
    **Standard TD Target**:

    .. math::
        y = r + \\gamma \\max_{a'} Q(s', a'; \\theta^-)

    **Double DQN Target**:

    .. math::
        y = r + \\gamma Q(s', \\arg\\max_{a'} Q(s', a'; \\theta); \\theta^-)

    **Distributional Target (C51)**:

    .. math::
        \\mathcal{T} Z(s, a) \\stackrel{D}{=} R + \\gamma Z(S', A')

    API Overview (API概览)
    ----------------------
    - ``select_action(state, training)``: 选择动作
    - ``train_step(state, action, reward, next_state, done)``: 完整训练步骤
    - ``update()``: 执行一次梯度更新
    - ``save(path)`` / ``load(path)``: 模型持久化

    Parameters
    ----------
    config : DQNVariantConfig
        Hyperparameter configuration
    variant : DQNVariant, default=DOUBLE_DUELING
        Algorithm variant to use

    Attributes
    ----------
    config : DQNVariantConfig
        Configuration object
    variant : DQNVariant
        Current variant
    device : torch.device
        Compute device
    q_network : nn.Module
        Online Q-network
    target_network : nn.Module
        Target Q-network
    optimizer : torch.optim.Optimizer
        Network optimizer
    buffer : Union[ReplayBuffer, PrioritizedReplayBuffer, NStepReplayBuffer]
        Experience replay buffer
    epsilon : float
        Current exploration rate (for non-noisy variants)

    Examples
    --------
    Basic training:

    >>> config = DQNVariantConfig(state_dim=4, action_dim=2)
    >>> agent = DQNVariantAgent(config, DQNVariant.RAINBOW)
    >>> action = agent.select_action(state, training=True)
    >>> loss = agent.train_step(state, action, reward, next_state, done)

    Model persistence:

    >>> agent.save("checkpoint.pt")
    >>> agent.load("checkpoint.pt")

    References
    ----------
    [1] Mnih et al. (2015). Human-level control through deep RL.
    [2] van Hasselt et al. (2016). Deep RL with Double Q-learning.
    [3] Wang et al. (2016). Dueling Network Architectures.
    [4] Fortunato et al. (2017). Noisy Networks for Exploration.
    [5] Bellemare et al. (2017). A Distributional Perspective on RL.
    [6] Hessel et al. (2018). Rainbow: Combining Improvements.
    """

    def __init__(
        self,
        config: DQNVariantConfig,
        variant: DQNVariant = DQNVariant.DOUBLE_DUELING,
    ) -> None:
        """
        Initialize agent with specified variant.

        Parameters
        ----------
        config : DQNVariantConfig
            Hyperparameter configuration
        variant : DQNVariant, default=DOUBLE_DUELING
            Algorithm variant to use
        """
        self.config = config
        self.variant = variant
        self.device = config.get_device()

        if config.seed is not None:
            self._set_seed(config.seed)

        self._init_networks()
        self._init_buffer()
        self._init_training_state()

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_networks(self) -> None:
        """Initialize online and target networks based on variant."""
        cfg = self.config

        # Select network architecture based on variant
        if self.variant == DQNVariant.VANILLA:
            self.q_network = DQNNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.num_layers
            )
        elif self.variant == DQNVariant.DOUBLE:
            self.q_network = DQNNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.num_layers
            )
        elif self.variant == DQNVariant.DUELING:
            self.q_network = DuelingNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim
            )
        elif self.variant == DQNVariant.NOISY:
            self.q_network = NoisyNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.sigma_init
            )
        elif self.variant == DQNVariant.CATEGORICAL:
            self.q_network = CategoricalNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim,
                cfg.num_atoms, cfg.v_min, cfg.v_max
            )
        elif self.variant == DQNVariant.DOUBLE_DUELING:
            self.q_network = DuelingNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim
            )
        elif self.variant == DQNVariant.RAINBOW:
            self.q_network = RainbowNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim,
                cfg.num_atoms, cfg.v_min, cfg.v_max, cfg.sigma_init
            )
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        self.q_network = self.q_network.to(self.device)

        # Create target network (same architecture)
        self.target_network = type(self.q_network)(
            *self._get_network_args()
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Freeze target network parameters
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=cfg.learning_rate,
        )

    def _get_network_args(self) -> tuple:
        """Get constructor arguments for network based on variant."""
        cfg = self.config

        if self.variant in (DQNVariant.VANILLA, DQNVariant.DOUBLE):
            return (cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.num_layers)
        elif self.variant in (DQNVariant.DUELING, DQNVariant.DOUBLE_DUELING):
            return (cfg.state_dim, cfg.action_dim, cfg.hidden_dim)
        elif self.variant == DQNVariant.NOISY:
            return (cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.sigma_init)
        elif self.variant == DQNVariant.CATEGORICAL:
            return (
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim,
                cfg.num_atoms, cfg.v_min, cfg.v_max
            )
        elif self.variant == DQNVariant.RAINBOW:
            return (
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim,
                cfg.num_atoms, cfg.v_min, cfg.v_max, cfg.sigma_init
            )
        return ()

    def _init_buffer(self) -> None:
        """Initialize replay buffer based on configuration."""
        cfg = self.config

        if cfg.n_steps > 1:
            self.buffer = NStepReplayBuffer(
                cfg.buffer_size, cfg.n_steps, cfg.gamma
            )
        elif cfg.use_per:
            self.buffer = PrioritizedReplayBuffer(
                cfg.buffer_size, cfg.per_alpha, cfg.per_beta_start,
                cfg.per_beta_frames
            )
        else:
            self.buffer = ReplayBuffer(cfg.buffer_size)

    def _init_training_state(self) -> None:
        """Initialize training state variables."""
        self._epsilon = self.config.epsilon_start
        self._training_step = 0
        self._update_count = 0
        self._losses: List[float] = []
        self._q_values: List[float] = []

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self._epsilon

    @property
    def training_step(self) -> int:
        """Current training step count."""
        return self._training_step

    @property
    def losses(self) -> List[float]:
        """Training loss history."""
        return self._losses.copy()

    def select_action(self, state: FloatArray, training: bool = True) -> int:
        """
        Select action using exploration policy.

        For noisy variants: Uses network noise for exploration
        For others: Uses ε-greedy

        Parameters
        ----------
        state : FloatArray
            Current observation, shape (state_dim,)
        training : bool, default=True
            Whether in training mode (affects exploration)

        Returns
        -------
        int
            Selected action index

        Examples
        --------
        >>> action = agent.select_action(state, training=True)
        >>> 0 <= action < config.action_dim
        True
        """
        # Noisy networks: exploration is built-in
        if self.variant.uses_noisy_exploration:
            if training and hasattr(self.q_network, "reset_noise"):
                self.q_network.reset_noise()

            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                if self.variant.is_distributional:
                    q_values = self.q_network.get_q_values(state_tensor)
                else:
                    q_values = self.q_network(state_tensor)

            return int(q_values.argmax(dim=1).item())

        # ε-greedy exploration
        if training and random.random() < self._epsilon:
            return random.randint(0, self.config.action_dim - 1)

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if self.variant.is_distributional:
                q_values = self.q_network.get_q_values(state_tensor)
            else:
                q_values = self.q_network(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> None:
        """
        Store transition in replay buffer.

        Parameters
        ----------
        state : FloatArray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : FloatArray
            Next state
        done : bool
            Whether episode ended
        """
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """
        Perform one gradient update step.

        Returns
        -------
        Optional[float]
            Loss value, or None if buffer has insufficient samples

        Notes
        -----
        Automatically selects update method based on variant:
        - Value-based: Standard TD learning
        - Distributional: KL divergence on distributions
        """
        if not self.buffer.is_ready(self.config.min_buffer_size):
            return None

        if self.variant.is_distributional:
            return self._update_distributional()
        return self._update_value()

    def _update_value(self) -> float:
        """Update for value-based variants (DQN, Double, Dueling, Noisy)."""
        cfg = self.config

        # Sample batch
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            (
                states, actions, rewards, next_states, dones, indices, weights
            ) = self.buffer.sample(cfg.batch_size)
            weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        elif isinstance(self.buffer, NStepReplayBuffer):
            states, actions, rewards, next_states, dones, n_steps = (
                self.buffer.sample(cfg.batch_size)
            )
            weights_t = None
            gamma_n = cfg.gamma ** n_steps
        else:
            states, actions, rewards, next_states, dones = (
                self.buffer.sample(cfg.batch_size)
            )
            weights_t = None

        # Convert to tensors
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Reset noise for noisy variants
        if self.variant.uses_noisy_exploration:
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        # Current Q-values
        current_q_values = self.q_network(states_t)
        current_q = current_q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            if self.variant.uses_double:
                # Double DQN: select action with online, evaluate with target
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_network(next_states_t).max(dim=1)[0]

            # TD target
            if isinstance(self.buffer, NStepReplayBuffer):
                gamma_n_t = torch.as_tensor(
                    cfg.gamma ** n_steps, dtype=torch.float32, device=self.device
                )
                target_q = rewards_t + (gamma_n_t * next_q * (1.0 - dones_t))
            else:
                target_q = rewards_t + (cfg.gamma * next_q * (1.0 - dones_t))

        # Compute loss
        td_errors = current_q - target_q

        if weights_t is not None:
            # Weighted loss for PER
            loss = (weights_t * (td_errors ** 2)).mean()
            # Update priorities
            self.buffer.update_priorities(
                indices, td_errors.detach().cpu().numpy()
            )
        else:
            loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self._optimize(loss)

        loss_value = loss.item()
        self._losses.append(loss_value)
        self._q_values.append(current_q.mean().item())

        return loss_value

    def _update_distributional(self) -> float:
        """Update for distributional variants (Categorical, Rainbow)."""
        cfg = self.config

        # Sample batch
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            (
                states, actions, rewards, next_states, dones, indices, weights
            ) = self.buffer.sample(cfg.batch_size)
            weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        else:
            states, actions, rewards, next_states, dones = (
                self.buffer.sample(cfg.batch_size)
            )
            weights_t = None

        batch_size = len(states)

        # Convert to tensors
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Reset noise
        if self.variant.uses_noisy_exploration:
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        # Current distribution: log p(s, a)
        current_log_probs = self.q_network(states_t)
        current_log_probs = current_log_probs[
            torch.arange(batch_size, device=self.device), actions_t
        ]  # (batch, atoms)

        # Compute target distribution
        with torch.no_grad():
            # Select greedy action for next state
            next_q_values = self.target_network.get_q_values(next_states_t)

            if self.variant.uses_double:
                # Double: use online network for action selection
                online_next_q = self.q_network.get_q_values(next_states_t)
                next_actions = online_next_q.argmax(dim=1)
            else:
                next_actions = next_q_values.argmax(dim=1)

            # Get next state distribution
            next_log_probs = self.target_network(next_states_t)
            next_probs = next_log_probs.exp()
            next_probs = next_probs[
                torch.arange(batch_size, device=self.device), next_actions
            ]  # (batch, atoms)

            # Project distribution onto support
            target_probs = self._project_distribution(
                next_probs, rewards_t, dones_t
            )

        # KL divergence loss
        loss = -(target_probs * current_log_probs).sum(dim=-1)

        if weights_t is not None:
            loss = (weights_t * loss).mean()
            # Update priorities using expected Q-value error
            with torch.no_grad():
                current_q = (current_log_probs.exp() * self.q_network.support).sum(dim=-1)
                target_q = (target_probs * self.q_network.support).sum(dim=-1)
                td_errors = (current_q - target_q).cpu().numpy()
            self.buffer.update_priorities(indices, td_errors)
        else:
            loss = loss.mean()

        # Optimize
        self._optimize(loss)

        loss_value = loss.item()
        self._losses.append(loss_value)

        return loss_value

    def _project_distribution(
        self,
        next_probs: Tensor,
        rewards: Tensor,
        dones: Tensor,
    ) -> Tensor:
        """
        Project target distribution onto fixed support.

        Implements categorical projection algorithm from C51 paper:
        1. Compute projected atoms: Tz_j = r + γz_j (clipped to [V_min, V_max])
        2. Distribute probability mass to neighboring support atoms

        Parameters
        ----------
        next_probs : Tensor
            Next state distribution, shape (batch_size, num_atoms)
        rewards : Tensor
            Rewards, shape (batch_size,)
        dones : Tensor
            Done flags, shape (batch_size,)

        Returns
        -------
        Tensor
            Projected distribution, shape (batch_size, num_atoms)
        """
        cfg = self.config
        batch_size = next_probs.shape[0]

        support = self.q_network.support  # (atoms,)
        delta_z = self.q_network.delta_z

        # Compute projected support: Tz = r + γz (for non-terminal)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # For terminal states, only reward matters (no bootstrapping)
        tz = rewards + (1 - dones) * cfg.gamma * support
        tz = tz.clamp(cfg.v_min, cfg.v_max)

        # Compute indices and interpolation weights
        b = (tz - cfg.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Handle edge cases
        l = l.clamp(0, cfg.num_atoms - 1)
        u = u.clamp(0, cfg.num_atoms - 1)

        # Initialize target distribution
        target_probs = torch.zeros_like(next_probs)

        # Distribute probability mass
        offset = (
            torch.arange(batch_size, device=self.device).unsqueeze(1)
            * cfg.num_atoms
        )

        target_probs.view(-1).index_add_(
            0,
            (l + offset).view(-1),
            (next_probs * (u.float() - b)).view(-1),
        )
        target_probs.view(-1).index_add_(
            0,
            (u + offset).view(-1),
            (next_probs * (b - l.float())).view(-1),
        )

        return target_probs

    def _optimize(self, loss: Tensor) -> None:
        """Perform optimization step with gradient clipping."""
        self.optimizer.zero_grad()
        loss.backward()

        if self.config.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.q_network.parameters(),
                self.config.grad_clip,
            )

        self.optimizer.step()

        self._update_count += 1
        self._sync_target_network()

    def _sync_target_network(self) -> None:
        """Synchronize target network parameters."""
        cfg = self.config

        if cfg.soft_update_tau is not None:
            # Soft update: θ⁻ ← τθ + (1-τ)θ⁻
            tau = cfg.soft_update_tau
            for target_p, online_p in zip(
                self.target_network.parameters(),
                self.q_network.parameters(),
            ):
                target_p.data.copy_(
                    tau * online_p.data + (1 - tau) * target_p.data
                )
        elif self._update_count % cfg.target_update_freq == 0:
            # Hard update: θ⁻ ← θ
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate (for non-noisy variants)."""
        if self.variant.uses_noisy_exploration:
            return

        self._training_step += 1
        decay_progress = min(1.0, self._training_step / self.config.epsilon_decay_steps)
        self._epsilon = (
            self.config.epsilon_start
            + (self.config.epsilon_end - self.config.epsilon_start) * decay_progress
        )

    def train_step(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> Optional[float]:
        """
        Complete training step: store → update → decay epsilon.

        Parameters
        ----------
        state : FloatArray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : FloatArray
            Next state
        done : bool
            Whether episode ended

        Returns
        -------
        Optional[float]
            Loss value, or None if buffer insufficient

        Examples
        --------
        >>> loss = agent.train_step(state, action, reward, next_state, done)
        >>> if loss is not None:
        ...     print(f"Loss: {loss:.4f}")
        """
        self.store_transition(state, action, reward, next_state, done)
        loss = self.update()
        self.decay_epsilon()
        return loss

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.

        Parameters
        ----------
        path : Union[str, Path]
            Path to save checkpoint

        Examples
        --------
        >>> agent.save("checkpoint.pt")
        """
        checkpoint = {
            "variant": self.variant.value,
            "config": self.config.to_dict(),
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self._epsilon,
            "training_step": self._training_step,
            "update_count": self._update_count,
            "losses": self._losses[-1000:],
        }
        torch.save(checkpoint, path)

    def load(self, path: Union[str, Path]) -> None:
        """
        Load model checkpoint.

        Parameters
        ----------
        path : Union[str, Path]
            Path to checkpoint file

        Examples
        --------
        >>> agent.load("checkpoint.pt")
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._epsilon = checkpoint.get("epsilon", self.config.epsilon_end)
        self._training_step = checkpoint.get("training_step", 0)
        self._update_count = checkpoint.get("update_count", 0)
        self._losses = checkpoint.get("losses", [])

    def set_train_mode(self) -> None:
        """Set network to training mode."""
        self.q_network.train()

    def set_eval_mode(self) -> None:
        """Set network to evaluation mode."""
        self.q_network.eval()
