"""
MDP Environment Module

Core Idea:
    Provides abstract and concrete implementations of Markov Decision Process
    environments, forming the foundation for dynamic programming algorithms.

Mathematical Theory:
    A Markov Decision Process (MDP) is defined by the tuple:

    .. math::
        \\mathcal{M} = \\langle \\mathcal{S}, \\mathcal{A}, P, R, \\gamma \\rangle

    where:
        - :math:`\\mathcal{S}`: State space (finite set of states)
        - :math:`\\mathcal{A}`: Action space (finite set of actions)
        - :math:`P(s'|s,a)`: Transition probability function
        - :math:`R(s,a,s')`: Reward function
        - :math:`\\gamma \\in [0,1]`: Discount factor

    The Markov property ensures:

    .. math::
        P(S_{t+1} | S_t, A_t, S_{t-1}, \\ldots) = P(S_{t+1} | S_t, A_t)

Problem Statement:
    Real-world sequential decision problems require a formal mathematical
    framework to model state transitions, rewards, and optimal decision-making.
    MDPs provide this framework, enabling principled algorithm development.

Comparison:
    MDPs vs Multi-Armed Bandits:
        - MDPs: Sequential states, transition dynamics, delayed rewards
        - Bandits: Single state, immediate rewards only

    MDPs vs POMDPs:
        - MDPs: Full state observability
        - POMDPs: Partial observability, requires belief states

Complexity:
    - State enumeration: O(|S|)
    - Transition query: O(1) amortized with precomputation
    - Memory: O(|S| × |A|) for transition storage

Summary:
    This module implements the GridWorld environment, a classic testbed for
    reinforcement learning algorithms. The agent navigates a grid to reach
    a goal while avoiding obstacles, receiving step penalties to encourage
    efficient paths.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Callable

# =============================================================================
# Type Definitions
# =============================================================================

State = Tuple[int, int]
"""State representation as (row, column) grid coordinates."""

Action = str
"""Action representation as string (e.g., '上', '下', '左', '右')."""

Policy = Dict[State, Dict[Action, float]]
"""Policy π(a|s) mapping states to action probability distributions."""

ValueFunction = Dict[State, float]
"""State value function V(s) mapping states to expected returns."""

QFunction = Dict[State, Dict[Action, float]]
"""Action value function Q(s,a) mapping state-action pairs to expected returns."""

TransitionResult = List[Tuple[State, float, float]]
"""Transition distribution: [(next_state, probability, reward), ...]"""


class ActionType(Enum):
    """
    Enumeration of available actions in GridWorld.

    Attributes:
        UP: Move one cell upward (row index decreases)
        DOWN: Move one cell downward (row index increases)
        LEFT: Move one cell leftward (column index decreases)
        RIGHT: Move one cell rightward (column index increases)
    """
    UP = '上'
    DOWN = '下'
    LEFT = '左'
    RIGHT = '右'


@dataclass(frozen=True)
class TransitionInfo:
    """
    Immutable container for state transition information.

    Attributes:
        next_state: The resulting state after transition
        probability: Probability of this transition occurring
        reward: Immediate reward received
    """
    next_state: State
    probability: float
    reward: float


# =============================================================================
# Abstract MDP Environment
# =============================================================================

class MDPEnvironment(ABC):
    """
    Abstract base class for Markov Decision Process environments.

    Core Idea:
        Defines the minimal interface that any MDP environment must implement,
        ensuring algorithmic code can work with arbitrary MDPs.

    Mathematical Theory:
        An MDP environment must provide:

        1. State space :math:`\\mathcal{S}`: All possible states
        2. Action space :math:`\\mathcal{A}`: All possible actions
        3. Transition function :math:`P(s'|s,a)`: State dynamics
        4. Reward function :math:`R(s,a,s')`: Feedback signal

    Summary:
        This abstract class enforces a contract for MDP environments,
        enabling polymorphic algorithm implementations.
    """

    @property
    @abstractmethod
    def states(self) -> List[State]:
        """
        Return the complete state space.

        Returns:
            List of all valid states in the MDP.
        """
        pass

    @property
    @abstractmethod
    def actions(self) -> List[Action]:
        """
        Return the action space.

        Returns:
            List of all available actions.
        """
        pass

    @property
    @abstractmethod
    def terminal_states(self) -> List[State]:
        """
        Return the set of terminal states.

        Terminal states have no outgoing transitions (or self-loops with
        zero reward). Episodes end upon reaching a terminal state.

        Returns:
            List of terminal states.
        """
        pass

    @abstractmethod
    def get_transitions(self, state: State, action: Action) -> TransitionResult:
        """
        Get the transition probability distribution for a state-action pair.

        Implements :math:`P(s'|s,a)` and :math:`R(s,a,s')`.

        Args:
            state: Current state :math:`s`
            action: Action to execute :math:`a`

        Returns:
            List of (next_state, probability, reward) tuples representing
            the complete transition distribution. Probabilities must sum to 1.
        """
        pass

    def is_terminal(self, state: State) -> bool:
        """
        Check if a state is terminal.

        Args:
            state: State to check

        Returns:
            True if state is terminal, False otherwise.
        """
        return state in self.terminal_states


# =============================================================================
# GridWorld Configuration
# =============================================================================

@dataclass
class GridWorldConfig:
    """
    Configuration parameters for GridWorld environment.

    Core Idea:
        Encapsulates all environment hyperparameters in a single immutable
        configuration object, enabling reproducible experiments.

    Attributes:
        size: Grid dimension (creates size × size grid)
        start: Starting state coordinates (row, column)
        goal: Goal state coordinates (terminal state)
        obstacles: List of obstacle positions (impassable cells)
        slip_probability: Probability of slipping perpendicular to intended
            direction (0.0 for deterministic, >0 for stochastic)
        step_reward: Reward for each step (typically negative to encourage
            efficient paths)
        goal_reward: Reward for reaching the goal state
        wall_penalty: Additional penalty for hitting walls (optional)

    Example:
        >>> config = GridWorldConfig(
        ...     size=4,
        ...     start=(0, 0),
        ...     goal=(3, 3),
        ...     obstacles=[(1, 1), (2, 2)],
        ...     slip_probability=0.1
        ... )
    """
    size: int = 4
    start: State = (0, 0)
    goal: State = (3, 3)
    obstacles: List[State] = field(default_factory=list)
    slip_probability: float = 0.0
    step_reward: float = -1.0
    goal_reward: float = 100.0
    wall_penalty: float = -1.0

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.size < 2:
            raise ValueError(f"Grid size must be >= 2, got: {self.size}")

        if self.slip_probability < 0.0 or self.slip_probability > 1.0:
            raise ValueError(
                f"Slip probability must be in [0, 1], got: {self.slip_probability}"
            )


# =============================================================================
# GridWorld Environment
# =============================================================================

class GridWorld(MDPEnvironment):
    """
    Classic GridWorld navigation environment.

    Core Idea:
        An agent navigates an N×N grid from a start position to a goal,
        receiving negative rewards for each step to encourage efficient paths.
        This environment serves as the canonical testbed for tabular RL algorithms.

    Mathematical Theory:
        State space: :math:`\\mathcal{S} = \\{(i,j) : 0 \\leq i,j < N\\} \\setminus \\text{obstacles}`

        Action space: :math:`\\mathcal{A} = \\{\\text{上}, \\text{下}, \\text{左}, \\text{右}\\}`

        Deterministic transitions:

        .. math::
            P(s'|s,a) = \\begin{cases}
                1 & \\text{if } s' = \\text{clip}(s + \\delta_a) \\\\
                0 & \\text{otherwise}
            \\end{cases}

        Stochastic transitions (with slip probability :math:`p`):

        .. math::
            P(s'|s,a) = (1-p) \\cdot \\mathbb{1}[s'=\\text{intended}] +
                        \\frac{p}{2} \\cdot \\mathbb{1}[s'=\\text{perpendicular}]

    Problem Statement:
        GridWorld addresses the fundamental challenge of sequential decision-making
        under known dynamics. It demonstrates core concepts like:
        - State-value and action-value functions
        - Optimal policy computation via dynamic programming
        - Trade-offs between exploration and exploitation

    Comparison:
        vs. Frozen Lake (OpenAI Gym):
            - GridWorld: Configurable obstacles, flexible rewards
            - Frozen Lake: Fixed layout, binary success/failure

        vs. Continuous Control:
            - GridWorld: Discrete state/action, exact solutions possible
            - Continuous: Requires function approximation

    Complexity:
        - State space: O(N²) where N is grid size
        - Action space: O(1) constant (4 actions)
        - Transition computation: O(1) per query
        - Memory: O(N²) for state enumeration

    Summary:
        GridWorld is the "hello world" of reinforcement learning. Its simplicity
        enables exact dynamic programming solutions while illustrating fundamental
        concepts that scale to complex domains.

    Visual Representation (4×4 grid):
        ┌────┬────┬────┬────┐
        │ S  │    │    │    │   S: Start (0,0)
        ├────┼────┼────┼────┤   G: Goal (3,3)
        │    │ X  │    │    │   X: Obstacle
        ├────┼────┼────┼────┤
        │    │    │    │    │
        ├────┼────┼────┼────┤
        │    │    │    │ G  │
        └────┴────┴────┴────┘
    """

    # Action to coordinate delta mapping
    ACTION_DELTAS: Dict[str, Tuple[int, int]] = {
        '上': (-1, 0),   # Up: decrease row
        '下': (1, 0),    # Down: increase row
        '左': (0, -1),   # Left: decrease column
        '右': (0, 1)     # Right: increase column
    }

    # Perpendicular actions for stochastic transitions
    PERPENDICULAR_ACTIONS: Dict[str, List[str]] = {
        '上': ['左', '右'],
        '下': ['左', '右'],
        '左': ['上', '下'],
        '右': ['上', '下']
    }

    def __init__(self, config: Optional[GridWorldConfig] = None):
        """
        Initialize GridWorld environment.

        Args:
            config: Environment configuration. Uses default if None.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        self.config = config or GridWorldConfig()
        self._validate_config()
        self._build_state_space()

    def _validate_config(self) -> None:
        """
        Validate configuration consistency.

        Raises:
            ValueError: If start/goal positions are invalid or overlap with obstacles.
        """
        cfg = self.config

        if not self._is_valid_position(cfg.start):
            raise ValueError(f"Start position out of bounds: {cfg.start}")

        if not self._is_valid_position(cfg.goal):
            raise ValueError(f"Goal position out of bounds: {cfg.goal}")

        if cfg.start in cfg.obstacles:
            raise ValueError(f"Start position cannot be an obstacle: {cfg.start}")

        if cfg.goal in cfg.obstacles:
            raise ValueError(f"Goal position cannot be an obstacle: {cfg.goal}")

    def _is_valid_position(self, pos: State) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= pos[0] < self.config.size and 0 <= pos[1] < self.config.size

    def _build_state_space(self) -> None:
        """Construct valid state space excluding obstacles."""
        self._states = [
            (i, j)
            for i in range(self.config.size)
            for j in range(self.config.size)
            if (i, j) not in self.config.obstacles
        ]

    @property
    def states(self) -> List[State]:
        """Return copy of state space to prevent external modification."""
        return self._states.copy()

    @property
    def actions(self) -> List[Action]:
        """Return available actions."""
        return list(self.ACTION_DELTAS.keys())

    @property
    def terminal_states(self) -> List[State]:
        """Return terminal states (goal only)."""
        return [self.config.goal]

    @property
    def num_states(self) -> int:
        """Return number of valid states."""
        return len(self._states)

    @property
    def num_actions(self) -> int:
        """Return number of actions."""
        return len(self.ACTION_DELTAS)

    def _execute_move(self, state: State, action: Action) -> State:
        """
        Execute movement action with boundary and obstacle collision handling.

        Args:
            state: Current position
            action: Direction to move

        Returns:
            New position after move (may be same as input if blocked).
        """
        di, dj = self.ACTION_DELTAS[action]
        next_i = max(0, min(self.config.size - 1, state[0] + di))
        next_j = max(0, min(self.config.size - 1, state[1] + dj))
        next_state = (next_i, next_j)

        # Collision with obstacle: stay in place
        if next_state in self.config.obstacles:
            return state

        return next_state

    def _compute_reward(self, current: State, next_state: State) -> float:
        """
        Compute immediate reward for state transition.

        Args:
            current: State before transition
            next_state: State after transition

        Returns:
            Immediate reward value.
        """
        if next_state == self.config.goal:
            return self.config.goal_reward
        return self.config.step_reward

    def get_transitions(self, state: State, action: Action) -> TransitionResult:
        """
        Get transition distribution for state-action pair.

        Implements both deterministic and stochastic (slippery) transitions.
        In stochastic mode, agent has `slip_probability` chance of moving
        perpendicular to intended direction.

        Args:
            state: Current state
            action: Action to execute

        Returns:
            List of (next_state, probability, reward) tuples.
            Probabilities are guaranteed to sum to 1.0.
        """
        # Terminal state: self-loop with zero reward
        if self.is_terminal(state):
            return [(state, 1.0, 0.0)]

        transitions: TransitionResult = []

        if self.config.slip_probability == 0.0:
            # Deterministic environment
            next_state = self._execute_move(state, action)
            reward = self._compute_reward(state, next_state)
            transitions.append((next_state, 1.0, reward))
        else:
            # Stochastic environment with slipping
            main_prob = 1.0 - self.config.slip_probability
            slip_prob = self.config.slip_probability / 2.0

            # Intended direction
            main_next = self._execute_move(state, action)
            main_reward = self._compute_reward(state, main_next)
            transitions.append((main_next, main_prob, main_reward))

            # Perpendicular directions (slipping)
            for perp_action in self.PERPENDICULAR_ACTIONS[action]:
                perp_next = self._execute_move(state, perp_action)
                perp_reward = self._compute_reward(state, perp_next)
                transitions.append((perp_next, slip_prob, perp_reward))

        return transitions

    def render_policy(
        self,
        policy: Policy,
        stream: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Render policy as ASCII visualization.

        Args:
            policy: Policy to visualize
            stream: Output function (default: print)

        Returns:
            Rendered string representation.
        """
        symbols = {'上': '↑', '下': '↓', '左': '←', '右': '→'}
        output = stream or print

        lines = ["\nPolicy Visualization:"]
        lines.append("┌" + "───┬" * (self.config.size - 1) + "───┐")

        for i in range(self.config.size):
            row = "│"
            for j in range(self.config.size):
                if (i, j) == self.config.goal:
                    row += " G │"
                elif (i, j) in self.config.obstacles:
                    row += " X │"
                elif (i, j) in policy:
                    best_action = max(policy[(i, j)], key=policy[(i, j)].get)
                    row += f" {symbols[best_action]} │"
                else:
                    row += "   │"
            lines.append(row)

            if i < self.config.size - 1:
                lines.append("├" + "───┼" * (self.config.size - 1) + "───┤")

        lines.append("└" + "───┴" * (self.config.size - 1) + "───┘")

        result = "\n".join(lines)
        output(result)
        return result

    def render_values(
        self,
        V: ValueFunction,
        stream: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Render value function as ASCII visualization.

        Args:
            V: Value function to visualize
            stream: Output function (default: print)

        Returns:
            Rendered string representation.
        """
        output = stream or print

        lines = ["\nState Value Function:"]
        lines.append("┌" + "──────┬" * (self.config.size - 1) + "──────┐")

        for i in range(self.config.size):
            row = "│"
            for j in range(self.config.size):
                if (i, j) == self.config.goal:
                    row += "   G  │"
                elif (i, j) in self.config.obstacles:
                    row += "   X  │"
                elif (i, j) in V:
                    row += f"{V[(i,j)]:6.1f}│"
                else:
                    row += "      │"
            lines.append(row)

            if i < self.config.size - 1:
                lines.append("├" + "──────┼" * (self.config.size - 1) + "──────┤")

        lines.append("└" + "──────┴" * (self.config.size - 1) + "──────┘")

        result = "\n".join(lines)
        output(result)
        return result

    def to_matrix_form(self) -> Tuple[Dict[State, int], Dict[int, State]]:
        """
        Create state-to-index mappings for matrix-based algorithms.

        Returns:
            Tuple of (state_to_idx, idx_to_state) dictionaries.
        """
        state_to_idx = {s: i for i, s in enumerate(self._states)}
        idx_to_state = {i: s for i, s in enumerate(self._states)}
        return state_to_idx, idx_to_state
