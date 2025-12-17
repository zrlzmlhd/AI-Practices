"""
Dynamic Programming Algorithms for MDP

Core Idea:
    Dynamic Programming (DP) methods solve MDPs by systematically exploiting
    the recursive structure of the Bellman equations. These algorithms require
    a complete model of the environment (transition probabilities and rewards).

Mathematical Theory:
    **Bellman Expectation Equation** (for policy π):

    .. math::
        V^\\pi(s) = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a) [R(s,a,s') + \\gamma V^\\pi(s')]

    **Bellman Optimality Equation**:

    .. math::
        V^*(s) = \\max_a \\sum_{s'} P(s'|s,a) [R(s,a,s') + \\gamma V^*(s')]

    Both equations express value functions recursively: the value of a state
    equals the expected immediate reward plus the discounted value of successor
    states.

Problem Statement:
    Given a fully-specified MDP (known P and R), compute the optimal policy
    π* that maximizes expected cumulative discounted reward from any state.
    This is the planning problem in reinforcement learning.

Comparison:
    Policy Iteration vs Value Iteration:
        - Policy Iteration: Few outer iterations, expensive inner evaluations
        - Value Iteration: Many iterations, cheap per-iteration cost
        - Asymptotic: Both converge to optimal; PI often faster for small MDPs

    DP vs Model-Free Methods:
        - DP: Requires model, exact solutions, polynomial complexity
        - Model-Free: No model needed, approximate solutions, sample complexity

Complexity:
    - Policy Evaluation: O(|S|²|A|) per iteration, linear convergence
    - Policy Iteration: O(|S|²|A| × k) where k is number of outer iterations
    - Value Iteration: O(|S|²|A| × log(1/ε)/(1-γ)) for ε-optimal solution

Summary:
    This module implements the classical DP algorithms that form the theoretical
    foundation of reinforcement learning. While limited to tabular MDPs with
    known models, these algorithms provide optimal solutions and serve as
    benchmarks for approximate methods.

References:
    [1] Bellman, R. (1957). Dynamic Programming. Princeton University Press.
    [2] Howard, R. (1960). Dynamic Programming and Markov Processes. MIT Press.
    [3] Puterman, M. (1994). Markov Decision Processes. Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from .environment import (
    MDPEnvironment,
    State,
    Action,
    Policy,
    ValueFunction,
    QFunction,
)


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class AlgorithmResult:
    """
    Container for algorithm execution results.

    Attributes:
        policy: Computed policy mapping states to action distributions
        value_function: Computed state value function V(s)
        iterations: Number of iterations until convergence
        converged: Whether algorithm converged within iteration limit
    """
    policy: Policy
    value_function: ValueFunction
    iterations: int
    converged: bool


# =============================================================================
# Dynamic Programming Solver
# =============================================================================

class DynamicProgrammingSolver:
    """
    Dynamic Programming solver for finite MDPs.

    Core Idea:
        Implements exact solution methods that iteratively apply Bellman
        operators until convergence. These are model-based methods requiring
        complete knowledge of transition probabilities and rewards.

    Mathematical Theory:
        **Bellman Operator for Policy π**:

        .. math::
            (T^\\pi V)(s) = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a)[R + \\gamma V(s')]

        This operator is a γ-contraction, guaranteeing convergence to V^π.

        **Bellman Optimality Operator**:

        .. math::
            (T^* V)(s) = \\max_a \\sum_{s'} P(s'|s,a)[R + \\gamma V(s')]

        Also a γ-contraction, converging to V* (Banach fixed-point theorem).

    Problem Statement:
        Dynamic programming addresses the planning problem: given complete
        knowledge of an MDP, find the optimal value function V* and policy π*.
        This contrasts with the learning problem where dynamics are unknown.

    Comparison:
        **Policy Iteration**:
            - Evaluates policy fully before improving
            - Fewer outer iterations (typically 3-10)
            - Higher per-iteration cost
            - Produces valid intermediate policies

        **Value Iteration**:
            - Truncates evaluation to single Bellman backup
            - More iterations required
            - Lower per-iteration cost
            - No valid policy until convergence

        **Unified View**: Value iteration is policy iteration with evaluation
        truncated to one step. Both are special cases of generalized policy
        iteration (GPI).

    Complexity:
        - Space: O(|S|) for value function storage
        - Time per iteration: O(|S|²|A|) for full state sweep
        - Convergence: Linear rate, factor γ per iteration

    Summary:
        This class provides production-ready implementations of classical DP
        algorithms. Use policy_iteration() for small MDPs requiring intermediate
        policies, and value_iteration() for larger MDPs where memory efficiency
        matters.

    Attributes:
        env: MDP environment instance
        gamma: Discount factor γ ∈ [0, 1]
        theta: Convergence threshold for value function updates
        verbose: Whether to print progress information

    Example:
        >>> env = GridWorld(GridWorldConfig(size=4))
        >>> solver = DynamicProgrammingSolver(env, gamma=0.99)
        >>> result = solver.value_iteration()
        >>> print(f"Converged in {result.iterations} iterations")
    """

    def __init__(
        self,
        env: MDPEnvironment,
        gamma: float = 0.99,
        theta: float = 1e-6,
        verbose: bool = True
    ):
        """
        Initialize the DP solver.

        Args:
            env: MDP environment providing states, actions, and transitions
            gamma: Discount factor controlling future reward weight.
                   γ=0 is myopic, γ=1 is undiscounted (may not converge).
            theta: Convergence threshold. Iteration stops when max value
                   change falls below theta.
            verbose: If True, print algorithm progress.

        Raises:
            ValueError: If gamma not in [0,1] or theta not positive.
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"Discount factor must be in [0,1], got: {gamma}")
        if theta <= 0:
            raise ValueError(f"Convergence threshold must be positive, got: {theta}")

        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Conditionally print message based on verbose setting."""
        if self.verbose:
            print(message)

    # =========================================================================
    # Policy Evaluation
    # =========================================================================

    def evaluate_policy(
        self,
        policy: Policy,
        max_iterations: int = 10000
    ) -> Tuple[ValueFunction, int]:
        """
        Policy Evaluation: Compute value function for a given policy.

        Core Idea:
            Iteratively apply the Bellman expectation operator until the value
            function converges. This answers: "How good is this policy?"

        Mathematical Theory:
            Iterative solution of the Bellman expectation equation:

            .. math::
                V_{k+1}(s) = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a)[R + \\gamma V_k(s')]

            The update rule substitutes current estimates V_k into the equation's
            right-hand side to produce improved estimates V_{k+1}.

            **Convergence Guarantee**: The Bellman operator T^π is a γ-contraction:

            .. math::
                \\|T^\\pi V - T^\\pi U\\|_\\infty \\leq \\gamma \\|V - U\\|_\\infty

            By Banach's fixed-point theorem, iteration converges to unique V^π.

        Problem Statement:
            Given policy π, compute V^π(s) = E_π[Σγ^t R | s_0=s]. This is a
            system of |S| linear equations that can be solved iteratively.

        Comparison:
            - Iterative: O(|S|²|A|) per iteration, linear convergence
            - Direct (matrix inversion): O(|S|³) one-time, numerically unstable
            - Iterative preferred for large state spaces

        Complexity:
            - Time: O(|S|²|A|) per iteration
            - Space: O(|S|) for value function
            - Iterations: O(log(1/θ)/(1-γ)) for θ-convergence

        Summary:
            Policy evaluation is the inner loop of policy iteration. It computes
            how much cumulative reward we expect under the given policy. This
            information is then used for policy improvement.

        Args:
            policy: Policy π(a|s) to evaluate, mapping states to action distributions
            max_iterations: Maximum number of Bellman backups

        Returns:
            Tuple of (value_function, iteration_count).
            value_function maps each state to its expected return under π.
        """
        # Initialize value function to zeros
        V: ValueFunction = {s: 0.0 for s in self.env.states}

        for iteration in range(1, max_iterations + 1):
            delta = 0.0

            for state in self.env.states:
                if self.env.is_terminal(state):
                    continue

                old_value = V[state]
                new_value = 0.0

                # Bellman expectation backup
                for action in self.env.actions:
                    action_prob = policy.get(state, {}).get(action, 0.0)
                    if action_prob > 0:
                        for next_state, trans_prob, reward in self.env.get_transitions(state, action):
                            new_value += action_prob * trans_prob * (
                                reward + self.gamma * V.get(next_state, 0.0)
                            )

                V[state] = new_value
                delta = max(delta, abs(old_value - new_value))

            # Check convergence
            if delta < self.theta:
                return V, iteration

        return V, max_iterations

    # =========================================================================
    # Policy Improvement
    # =========================================================================

    def improve_policy(self, V: ValueFunction) -> Policy:
        """
        Policy Improvement: Construct greedy policy from value function.

        Core Idea:
            For each state, select the action that maximizes expected one-step
            return plus discounted successor value. This greedy policy is
            guaranteed to be at least as good as the original.

        Mathematical Theory:
            **Greedy Policy Construction**:

            .. math::
                \\pi'(s) = \\arg\\max_a \\sum_{s'} P(s'|s,a)[R + \\gamma V(s')]

            This selects the action maximizing the action-value Q(s,a).

            **Policy Improvement Theorem**: For any policy π, if π' is greedy
            with respect to V^π, then:

            .. math::
                V^{\\pi'}(s) \\geq V^\\pi(s) \\quad \\forall s \\in \\mathcal{S}

            Equality holds iff π is already optimal.

        Problem Statement:
            Given value function V (from policy evaluation), construct an
            improved policy. This transforms value information into action
            selection.

        Comparison:
            - Deterministic greedy: Always select argmax (used here)
            - Soft greedy (Boltzmann): Probabilistic selection for exploration
            - ε-greedy: Mix of greedy and random for online learning

        Complexity:
            - Time: O(|S||A|) for single sweep
            - Space: O(|S||A|) for policy storage

        Summary:
            Policy improvement extracts a better policy from value estimates.
            Combined with policy evaluation, this creates a monotonically
            improving sequence that converges to optimality.

        Args:
            V: State value function to derive policy from

        Returns:
            Deterministic policy selecting value-maximizing actions.
        """
        policy: Policy = {}

        for state in self.env.states:
            if self.env.is_terminal(state):
                # Terminal: uniform random (doesn't matter)
                policy[state] = {a: 1.0 / len(self.env.actions) for a in self.env.actions}
                continue

            # Compute Q-values for each action
            q_values: Dict[Action, float] = {}
            for action in self.env.actions:
                q_val = 0.0
                for next_state, trans_prob, reward in self.env.get_transitions(state, action):
                    q_val += trans_prob * (reward + self.gamma * V.get(next_state, 0.0))
                q_values[action] = q_val

            # Select best action(s) - handle ties
            best_value = max(q_values.values())
            best_actions = [a for a, v in q_values.items() if abs(v - best_value) < 1e-9]

            # Deterministic policy: select first best action
            policy[state] = {
                a: 1.0 if a == best_actions[0] else 0.0
                for a in self.env.actions
            }

        return policy

    # =========================================================================
    # Policy Iteration
    # =========================================================================

    def policy_iteration(self, max_iterations: int = 100) -> AlgorithmResult:
        """
        Policy Iteration: Alternating evaluation and improvement.

        Core Idea:
            Alternate between fully evaluating the current policy and improving
            it greedily. This produces a sequence of monotonically improving
            policies that converge to optimal in finite steps.

        Mathematical Theory:
            **Algorithm Structure**:
                1. Initialize π₀ (e.g., uniform random)
                2. Policy Evaluation: Compute V^{πₖ}
                3. Policy Improvement: πₖ₊₁ = greedy(V^{πₖ})
                4. If πₖ₊₁ = πₖ, return; else k ← k+1, goto 2

            **Convergence Proof Sketch**:
                - Policy improvement theorem: V^{πₖ₊₁} ≥ V^{πₖ}
                - Strict improvement unless πₖ already optimal
                - Finite policy space → must converge in finite steps
                - Converges to global optimum (no local optima in MDPs)

        Problem Statement:
            Proposed by Ronald Howard (1960) for operations research problems.
            Policy iteration decomposes the difficult optimization problem into
            two tractable subproblems: evaluation (linear system) and
            improvement (greedy selection).

        Comparison:
            **Advantages over Value Iteration**:
                - Typically fewer outer iterations (3-10)
                - Each iteration produces a valid, executable policy
                - More interpretable intermediate results

            **Disadvantages**:
                - Each evaluation requires many inner iterations
                - Higher per-iteration memory (stores both policy and values)
                - Less amenable to asynchronous updates

        Complexity:
            - Outer iterations: O(|A|^|S|) worst case, O(|S||A|) typical
            - Per iteration: O(|S|²|A| × eval_iterations)
            - Total: Usually faster than VI for small MDPs

        Summary:
            Policy iteration is the classical algorithm for exact MDP solution.
            It exploits the structure of the problem by separating evaluation
            and improvement, often converging in very few iterations.

        Args:
            max_iterations: Maximum outer iterations (rarely needed >10)

        Returns:
            AlgorithmResult with optimal policy, values, and convergence info.
        """
        self._log("=" * 60)
        self._log("Policy Iteration")
        self._log("=" * 60)

        # Initialize uniform random policy
        num_actions = len(self.env.actions)
        policy: Policy = {
            s: {a: 1.0 / num_actions for a in self.env.actions}
            for s in self.env.states
        }

        V: ValueFunction = {}

        for iteration in range(1, max_iterations + 1):
            self._log(f"\n--- Outer Iteration {iteration} ---")

            # Policy Evaluation
            V, eval_iters = self.evaluate_policy(policy)
            self._log(f"Policy evaluation: {eval_iters} inner iterations")

            # Policy Improvement
            new_policy = self.improve_policy(V)

            # Check convergence (policy unchanged)
            if self._policies_equal(policy, new_policy):
                self._log(f"\nPolicy iteration converged in {iteration} iterations")
                return AlgorithmResult(
                    policy=new_policy,
                    value_function=V,
                    iterations=iteration,
                    converged=True
                )

            policy = new_policy

        self._log(f"\nReached maximum iterations: {max_iterations}")
        return AlgorithmResult(
            policy=policy,
            value_function=V,
            iterations=max_iterations,
            converged=False
        )

    # =========================================================================
    # Value Iteration
    # =========================================================================

    def value_iteration(self, max_iterations: int = 10000) -> AlgorithmResult:
        """
        Value Iteration: Direct Bellman optimality iteration.

        Core Idea:
            Iterate the Bellman optimality equation directly, without maintaining
            an explicit policy. This is equivalent to policy iteration with
            evaluation truncated to a single backup.

        Mathematical Theory:
            **Update Rule**:

            .. math::
                V_{k+1}(s) = \\max_a \\sum_{s'} P(s'|s,a)[R + \\gamma V_k(s')]

            **Contraction Property**: The Bellman optimality operator T* satisfies:

            .. math::
                \\|T^*V - T^*U\\|_\\infty \\leq \\gamma\\|V - U\\|_\\infty

            By Banach's theorem, unique fixed point V* exists and iteration
            converges at rate O(γ^k).

            **Convergence Rate**:

            .. math::
                \\|V_k - V^*\\|_\\infty \\leq \\gamma^k \\|V_0 - V^*\\|_\\infty

            For ε-convergence: k = O(log(1/ε)/(1-γ)) iterations.

        Problem Statement:
            Proposed by Bellman (1957) as a fundamental dynamic programming
            technique. Value iteration provides a simple, memory-efficient
            algorithm for computing optimal value functions.

        Comparison:
            **Advantages over Policy Iteration**:
                - Lower per-iteration cost (no inner loop)
                - Lower memory (no policy storage during iteration)
                - Better for large state spaces
                - More amenable to asynchronous/parallel updates

            **Disadvantages**:
                - More total iterations required
                - No valid policy until convergence
                - Convergence slower as γ → 1

        Complexity:
            - Time per iteration: O(|S||A|)
            - Total iterations: O(log(1/θ)/(1-γ))
            - Space: O(|S|) for value function only

        Summary:
            Value iteration is the workhorse algorithm for tabular MDP solution.
            Its simplicity and low per-iteration cost make it the default choice
            for medium-sized problems. The final policy is extracted by a single
            greedy improvement step after convergence.

        Args:
            max_iterations: Maximum Bellman backups

        Returns:
            AlgorithmResult with optimal policy, values, and convergence info.
        """
        self._log("=" * 60)
        self._log("Value Iteration")
        self._log("=" * 60)

        V: ValueFunction = {s: 0.0 for s in self.env.states}

        for iteration in range(1, max_iterations + 1):
            delta = 0.0

            for state in self.env.states:
                if self.env.is_terminal(state):
                    continue

                old_value = V[state]

                # Bellman optimality backup
                q_values = []
                for action in self.env.actions:
                    q_val = 0.0
                    for next_state, trans_prob, reward in self.env.get_transitions(state, action):
                        q_val += trans_prob * (reward + self.gamma * V.get(next_state, 0.0))
                    q_values.append(q_val)

                V[state] = max(q_values)
                delta = max(delta, abs(old_value - V[state]))

            if delta < self.theta:
                self._log(f"Value iteration converged in {iteration} iterations")
                policy = self.improve_policy(V)
                return AlgorithmResult(
                    policy=policy,
                    value_function=V,
                    iterations=iteration,
                    converged=True
                )

        self._log(f"Reached maximum iterations: {max_iterations}")
        policy = self.improve_policy(V)
        return AlgorithmResult(
            policy=policy,
            value_function=V,
            iterations=max_iterations,
            converged=False
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def compute_q_function(self, V: ValueFunction) -> QFunction:
        """
        Compute action-value function from state-value function.

        Mathematical Definition:
            .. math::
                Q(s,a) = \\sum_{s'} P(s'|s,a)[R(s,a,s') + \\gamma V(s')]

        Args:
            V: State value function

        Returns:
            Action-value function Q(s,a).
        """
        Q: QFunction = {}

        for state in self.env.states:
            Q[state] = {}
            for action in self.env.actions:
                q_val = 0.0
                for next_state, trans_prob, reward in self.env.get_transitions(state, action):
                    q_val += trans_prob * (reward + self.gamma * V.get(next_state, 0.0))
                Q[state][action] = q_val

        return Q

    @staticmethod
    def _policies_equal(p1: Policy, p2: Policy) -> bool:
        """
        Check if two policies are equal.

        Args:
            p1: First policy
            p2: Second policy

        Returns:
            True if policies assign same probabilities to all state-action pairs.
        """
        if set(p1.keys()) != set(p2.keys()):
            return False

        for state in p1:
            for action in p1[state]:
                if abs(p1[state].get(action, 0) - p2[state].get(action, 0)) > 1e-9:
                    return False
        return True
