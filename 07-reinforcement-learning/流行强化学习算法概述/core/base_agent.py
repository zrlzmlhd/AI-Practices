"""
Base Agent Abstract Class
=========================

Core Idea
---------
Define a unified interface for all continuous control RL agents.
Enables polymorphic training loops and algorithm comparison.

Mathematical Theory
-------------------
All implemented agents follow the actor-critic paradigm:

**Actor** :math:`\\pi_\\theta(a|s)`: Policy that maps states to actions

**Critic** :math:`Q_\\phi(s, a)`: Value function estimating expected return

The general optimization objectives are:

**Critic Update** (minimize TD error):

.. math::

    L(\\phi) = \\mathbb{E}_{(s,a,r,s') \\sim \\mathcal{D}}
    [(Q_\\phi(s, a) - y)^2]

**Actor Update** (maximize expected return):

.. math::

    J(\\theta) = \\mathbb{E}_{s \\sim \\mathcal{D}}[Q_\\phi(s, \\pi_\\theta(s))]

Problem Statement
-----------------
Without a common interface:

1. Training loops must be algorithm-specific
2. No polymorphic algorithm comparison
3. Inconsistent method signatures cause bugs

BaseAgent ensures all algorithms implement the same interface.

Summary
-------
Abstract base class enforcing:

- select_action(): Policy inference
- store_transition(): Experience collection
- update(): Learning step
- save()/load(): Model persistence
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import torch


class BaseAgent(ABC):
    """
    Abstract Base Class for Continuous Control Agents
    ==================================================

    Core Idea
    ---------
    Unified interface for off-policy actor-critic algorithms (DDPG, TD3, SAC).
    Enables algorithm-agnostic training loops and fair comparisons.

    Mathematical Theory
    -------------------
    All agents share the actor-critic structure:

    1. **Experience Collection**: Store :math:`(s, a, r, s', d)` in replay buffer
    2. **Critic Learning**: Minimize Bellman error via gradient descent
    3. **Actor Learning**: Maximize Q-value (or entropy-regularized objective)

    The update frequency and specific objectives vary by algorithm:

    +----------+-------------------+---------------------------+
    | Algorithm| Critic Objective  | Actor Objective           |
    +==========+===================+===========================+
    | DDPG     | :math:`(Q - y)^2` | :math:`Q(s, \\mu(s))`     |
    | TD3      | :math:`(Q - y)^2` | :math:`Q(s, \\mu(s))`     |
    | SAC      | :math:`(Q - y)^2` | :math:`Q - \\alpha\\log\\pi`|
    +----------+-------------------+---------------------------+

    Problem Statement
    -----------------
    RL codebases often have algorithm-specific training code, causing:

    - Code duplication across algorithms
    - Difficult hyperparameter sweeps
    - Unfair comparisons due to implementation differences

    BaseAgent standardizes the interface for consistent experimentation.

    Interface Methods
    -----------------
    - select_action(state, deterministic): Return action for given state
    - store_transition(...): Add experience to replay buffer
    - update(): Perform one gradient update step
    - save(path): Serialize model to disk
    - load(path): Deserialize model from disk

    Attributes
    ----------
    device : torch.device
        Compute device (CPU/CUDA/MPS).
    total_updates : int
        Counter for gradient updates performed.
    training : bool
        Whether agent is in training mode.

    Example
    -------
    >>> # Algorithm-agnostic training loop
    >>> def train(agent: BaseAgent, env, timesteps: int):
    ...     state = env.reset()
    ...     for t in range(timesteps):
    ...         action = agent.select_action(state)
    ...         next_state, reward, done, _ = env.step(action)
    ...         agent.store_transition(state, action, reward, next_state, done)
    ...         agent.update()
    ...         state = next_state if not done else env.reset()
    """

    def __init__(self) -> None:
        """Initialize base agent attributes."""
        self.device: torch.device = torch.device("cpu")
        self.total_updates: int = 0
        self.training: bool = True

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action for given state.

        Parameters
        ----------
        state : np.ndarray
            Current observation, shape (state_dim,).
        deterministic : bool, default=False
            If True, use deterministic policy (no exploration noise).
            Used during evaluation.

        Returns
        -------
        np.ndarray
            Action to execute, shape (action_dim,).

        Notes
        -----
        - During training: Add exploration noise (algorithm-specific)
        - During evaluation: Use deterministic policy
        """
        raise NotImplementedError

    @abstractmethod
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
            Resulting observation.
        done : bool
            Whether episode terminated.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """
        Perform one gradient update step.

        Samples a batch from replay buffer and updates actor/critic networks.

        Returns
        -------
        Dict[str, float]
            Training metrics (losses, Q-values, etc.).
            Returns empty dict if buffer has insufficient samples.

        Notes
        -----
        Specific update logic varies by algorithm:

        - DDPG: Update critic, then actor, every step
        - TD3: Update critic every step, actor every `policy_delay` steps
        - SAC: Update critic, actor, and temperature every step
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent state to disk.

        Serializes all networks, optimizers, and configuration.

        Parameters
        ----------
        path : str
            Output file path (typically .pt or .pth).
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent state from disk.

        Restores networks, optimizers, and configuration.

        Parameters
        ----------
        path : str
            Input file path.
        """
        raise NotImplementedError

    def train_mode(self) -> None:
        """Set agent to training mode."""
        self.training = True

    def eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        self.training = False

    def get_config(self) -> Dict[str, Any]:
        """
        Get agent configuration.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary for logging/reproduction.
        """
        return {}

    def _soft_update(
        self,
        source: torch.nn.Module,
        target: torch.nn.Module,
        tau: float,
    ) -> None:
        """
        Soft update target network parameters.

        Implements Polyak averaging:

        .. math::

            \\theta_{target} \\leftarrow \\tau \\theta_{source}
            + (1 - \\tau) \\theta_{target}

        Parameters
        ----------
        source : nn.Module
            Source network (online).
        target : nn.Module
            Target network (slowly updated).
        tau : float
            Interpolation coefficient in (0, 1].
            Typical value: 0.005

        Notes
        -----
        This stabilizes training by providing slowly-changing targets
        for the Bellman backup, preventing oscillations and divergence.
        """
        for source_param, target_param in zip(
            source.parameters(), target.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )

    def _hard_update(
        self,
        source: torch.nn.Module,
        target: torch.nn.Module,
    ) -> None:
        """
        Hard update (direct copy) target network parameters.

        Parameters
        ----------
        source : nn.Module
            Source network.
        target : nn.Module
            Target network.
        """
        target.load_state_dict(source.state_dict())

    def _to_tensor(
        self,
        array: np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Convert numpy array to tensor on agent's device.

        Parameters
        ----------
        array : np.ndarray
            Input array.
        dtype : torch.dtype, default=torch.float32
            Target data type.

        Returns
        -------
        torch.Tensor
            Tensor on agent's device.
        """
        return torch.as_tensor(array, dtype=dtype, device=self.device)


if __name__ == "__main__":
    # BaseAgent is abstract, test via concrete implementation check
    print("Testing BaseAgent interface...")

    # Verify abstract methods cannot be instantiated
    try:
        agent = BaseAgent()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError as e:
        assert "abstract" in str(e).lower()
        print("  [PASS] Cannot instantiate abstract BaseAgent")

    # Verify soft update utility
    import torch.nn as nn

    class ConcreteAgent(BaseAgent):
        """Minimal concrete implementation for testing."""

        def select_action(self, state, deterministic=False):
            return np.zeros(2)

        def store_transition(self, state, action, reward, next_state, done):
            pass

        def update(self):
            return {}

        def save(self, path):
            pass

        def load(self, path):
            pass

    agent = ConcreteAgent()

    # Test soft update
    source = nn.Linear(4, 4)
    target = nn.Linear(4, 4)

    # Initialize differently
    nn.init.ones_(source.weight)
    nn.init.zeros_(target.weight)

    agent._soft_update(source, target, tau=0.1)

    expected = 0.1 * 1.0 + 0.9 * 0.0  # = 0.1
    actual = target.weight.data.mean().item()
    assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"
    print("  [PASS] Soft update utility")

    # Test hard update
    agent._hard_update(source, target)
    assert (source.weight.data == target.weight.data).all()
    print("  [PASS] Hard update utility")

    # Test tensor conversion
    arr = np.array([1.0, 2.0, 3.0])
    tensor = agent._to_tensor(arr)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == "cpu"
    print("  [PASS] Tensor conversion utility")

    print("\nAll BaseAgent tests passed!")
