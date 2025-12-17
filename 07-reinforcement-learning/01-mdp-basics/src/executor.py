"""
Policy Executor Module

Core Idea:
    Provides utilities for executing policies in MDP environments and
    collecting performance statistics. Essential for empirical validation
    of learned or computed policies.

Mathematical Theory:
    **Episode Return**:
    Given a policy π, an episode generates trajectory τ = (s₀, a₀, r₁, s₁, ...):

    .. math::
        G_0 = \\sum_{t=0}^{T-1} \\gamma^t r_{t+1}

    **Expected Return** (Monte Carlo estimate):

    .. math::
        \\hat{V}^\\pi(s) = \\frac{1}{N} \\sum_{i=1}^{N} G_0^{(i)}

    where N is number of episodes starting from state s.

Problem Statement:
    After computing a policy via dynamic programming, we need to:
    1. Verify the policy works correctly in the environment
    2. Estimate empirical performance (especially in stochastic environments)
    3. Visualize agent behavior for debugging and presentation

Comparison:
    - Deterministic execution: Single trajectory, reproducible results
    - Monte Carlo evaluation: Multiple episodes, statistical estimates
    - On-policy vs Off-policy: Executor follows given policy (on-policy)

Complexity:
    - Single episode: O(T × |A|) where T is episode length
    - Multi-episode evaluation: O(N × T × |A|)
    - Memory: O(T) for trajectory storage

Summary:
    The PolicyExecutor bridges the gap between theoretical policy computation
    and practical validation. It enables both qualitative (trajectory
    visualization) and quantitative (statistical) analysis of policy
    performance.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np

from .environment import (
    GridWorld,
    State,
    Policy,
)


class PolicyExecutor:
    """
    Execute and evaluate policies in GridWorld environments.

    Core Idea:
        Runs episodes by selecting actions according to a policy and observing
        environment transitions. Collects trajectories and reward statistics.

    Attributes:
        env: GridWorld environment instance
        rng: NumPy random number generator for reproducibility

    Example:
        >>> executor = PolicyExecutor(env, seed=42)
        >>> reward, steps, trajectory = executor.run_episode(policy)
        >>> stats = executor.evaluate_policy(policy, num_episodes=100)
    """

    def __init__(self, env: GridWorld, seed: Optional[int] = None):
        """
        Initialize policy executor.

        Args:
            env: GridWorld environment
            seed: Random seed for reproducibility. If None, uses system entropy.
        """
        self.env = env
        self.rng = np.random.default_rng(seed)

    def run_episode(
        self,
        policy: Policy,
        max_steps: int = 100,
        verbose: bool = True
    ) -> Tuple[float, int, List[State]]:
        """
        Execute a single episode following the given policy.

        Core Idea:
            Starting from the environment's start state, iteratively select
            actions according to policy probabilities and execute transitions
            until reaching a terminal state or step limit.

        Mathematical Theory:
            Episode generation process:
            1. Initialize s₀ = start_state
            2. For t = 0, 1, 2, ...:
               - Sample a_t ~ π(·|s_t)
               - Sample s_{t+1} ~ P(·|s_t, a_t)
               - Observe r_{t+1} = R(s_t, a_t, s_{t+1})
               - If s_{t+1} is terminal, stop
            3. Return cumulative reward G₀ = Σ r_t

        Args:
            policy: Policy to execute (state → action distribution)
            max_steps: Maximum steps before forced termination
            verbose: Whether to print step-by-step execution

        Returns:
            Tuple of:
                - total_reward: Cumulative undiscounted reward
                - steps: Number of steps taken
                - trajectory: List of visited states including start
        """
        state = self.env.config.start
        total_reward = 0.0
        trajectory = [state]

        if verbose:
            print(f"\nStarting state: {state}")

        for step in range(max_steps):
            if self.env.is_terminal(state):
                if verbose:
                    print(f"Goal reached! Steps: {step}, Total reward: {total_reward:.1f}")
                return total_reward, step, trajectory

            # Sample action from policy distribution
            action_probs = policy[state]
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            action = self.rng.choice(actions, p=probs)

            # Execute transition
            transitions = self.env.get_transitions(state, action)
            trans_probs = [t[1] for t in transitions]
            idx = self.rng.choice(len(transitions), p=trans_probs)
            next_state, _, reward = transitions[idx]

            if verbose:
                print(f"Step {step + 1}: {state} --[{action}]--> {next_state}, reward: {reward:.1f}")

            total_reward += reward
            state = next_state
            trajectory.append(state)

        if verbose:
            print(f"Reached step limit, total reward: {total_reward:.1f}")

        return total_reward, max_steps, trajectory

    def evaluate_policy(
        self,
        policy: Policy,
        num_episodes: int = 100,
        max_steps: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate policy performance over multiple episodes.

        Core Idea:
            Monte Carlo estimation of policy performance metrics. Runs many
            episodes and computes statistics over returns and success rates.

        Mathematical Theory:
            **Sample Mean Estimator**:

            .. math::
                \\hat{\\mu} = \\frac{1}{N} \\sum_{i=1}^{N} G_i

            **Sample Standard Deviation**:

            .. math::
                \\hat{\\sigma} = \\sqrt{\\frac{1}{N-1} \\sum_{i=1}^{N} (G_i - \\hat{\\mu})^2}

            **Success Rate**: Fraction of episodes reaching goal state.

        Args:
            policy: Policy to evaluate
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with statistics:
                - mean_reward: Average episode return
                - std_reward: Standard deviation of returns
                - mean_steps: Average episode length
                - success_rate: Fraction reaching goal (0 to 1)
        """
        rewards = []
        steps_list = []
        successes = 0

        for _ in range(num_episodes):
            reward, steps, trajectory = self.run_episode(
                policy,
                max_steps,
                verbose=False
            )
            rewards.append(reward)
            steps_list.append(steps)

            if trajectory[-1] == self.env.config.goal:
                successes += 1

        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_steps': float(np.mean(steps_list)),
            'success_rate': successes / num_episodes
        }

    def get_optimal_path(
        self,
        policy: Policy,
        start: Optional[State] = None
    ) -> List[State]:
        """
        Extract deterministic optimal path from policy.

        For deterministic policies, returns the unique path from start to goal.
        For stochastic policies, follows the most probable action at each step.

        Args:
            policy: Policy defining action selection
            start: Starting state (default: environment start state)

        Returns:
            List of states from start to goal (or step limit).
        """
        state = start or self.env.config.start
        path = [state]
        visited = {state}
        max_steps = self.env.num_states * 2

        for _ in range(max_steps):
            if self.env.is_terminal(state):
                break

            # Select most probable action
            action_probs = policy[state]
            best_action = max(action_probs, key=action_probs.get)

            # Get deterministic next state (most probable)
            transitions = self.env.get_transitions(state, best_action)
            next_state = max(transitions, key=lambda t: t[1])[0]

            # Cycle detection
            if next_state in visited:
                break

            visited.add(next_state)
            path.append(next_state)
            state = next_state

        return path
