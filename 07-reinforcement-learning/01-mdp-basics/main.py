#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDP Basics: Main Entry Point

This module provides a demonstration of the complete MDP solution pipeline:
    1. Environment creation and configuration
    2. Dynamic programming algorithms (Policy Iteration, Value Iteration)
    3. Policy visualization and evaluation
    4. Comparative analysis

Execute this file to run the full demonstration with tests.

Usage:
    python main.py              # Run demonstration
    python main.py --test-only  # Run tests only

References:
    [1] Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 4
    [2] Bellman, R. "Dynamic Programming", Princeton University Press, 1957
"""

from __future__ import annotations

import sys
import argparse

from src.environment import GridWorld, GridWorldConfig
from src.algorithms import DynamicProgrammingSolver
from src.executor import PolicyExecutor


def run_tests() -> bool:
    """
    Execute complete test suite.

    Returns:
        True if all tests pass, False otherwise.
    """
    from tests.test_mdp import run_all_tests
    return run_all_tests()


def run_demonstration() -> None:
    """
    Run complete MDP demonstration.

    Demonstrates:
        - Environment setup with configurable parameters
        - Policy Iteration algorithm
        - Value Iteration algorithm
        - Policy visualization and execution
        - Stochastic environment handling
    """
    print("\n" + "=" * 70)
    print("MDP Basics: GridWorld Dynamic Programming Demonstration")
    print("=" * 70)

    # =========================================================================
    # Environment Configuration
    # =========================================================================

    config = GridWorldConfig(
        size=4,
        start=(0, 0),
        goal=(3, 3),
        obstacles=[(1, 1), (2, 3)],
        slip_probability=0.0,
        step_reward=-1.0,
        goal_reward=100.0
    )

    env = GridWorld(config)

    print(f"\nEnvironment Configuration:")
    print(f"  Grid size: {config.size}×{config.size}")
    print(f"  State space: {env.num_states} states")
    print(f"  Action space: {env.num_actions} actions")
    print(f"  Obstacles: {config.obstacles}")
    print(f"  Discount factor: γ = 0.99")

    solver = DynamicProgrammingSolver(
        env,
        gamma=0.99,
        theta=1e-6,
        verbose=True
    )

    # =========================================================================
    # Policy Iteration
    # =========================================================================

    print("\n" + "=" * 70)
    result_pi = solver.policy_iteration()
    env.render_policy(result_pi.policy)
    env.render_values(result_pi.value_function)

    # =========================================================================
    # Value Iteration
    # =========================================================================

    print("\n" + "=" * 70)
    result_vi = solver.value_iteration()
    env.render_policy(result_vi.policy)
    env.render_values(result_vi.value_function)

    # =========================================================================
    # Algorithm Comparison
    # =========================================================================

    print("\n" + "=" * 70)
    print("Algorithm Comparison")
    print("=" * 70)
    print(f"{'Algorithm':<25} {'Iterations':>15} {'Converged':>12}")
    print("-" * 55)
    print(f"{'Policy Iteration':<25} {result_pi.iterations:>15} {'Yes' if result_pi.converged else 'No':>12}")
    print(f"{'Value Iteration':<25} {result_vi.iterations:>15} {'Yes' if result_vi.converged else 'No':>12}")

    # Verify value function agreement
    value_diff = sum(
        abs(result_pi.value_function[s] - result_vi.value_function[s])
        for s in env.states
    )
    print(f"\nValue function total difference: {value_diff:.2e}")

    # =========================================================================
    # Policy Execution
    # =========================================================================

    print("\n" + "=" * 70)
    print("Policy Execution Demo")
    print("=" * 70)

    executor = PolicyExecutor(env, seed=42)
    executor.run_episode(result_vi.policy, verbose=True)

    # Statistical evaluation
    stats = executor.evaluate_policy(result_vi.policy, num_episodes=100)
    print(f"\n100-Episode Statistics:")
    print(f"  Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean steps: {stats['mean_steps']:.2f}")
    print(f"  Success rate: {stats['success_rate']*100:.1f}%")

    # =========================================================================
    # Stochastic Environment
    # =========================================================================

    print("\n" + "=" * 70)
    print("Stochastic Environment (20% slip probability)")
    print("=" * 70)

    stochastic_config = GridWorldConfig(
        size=4,
        start=(0, 0),
        goal=(3, 3),
        obstacles=[(1, 1), (2, 3)],
        slip_probability=0.2
    )
    stochastic_env = GridWorld(stochastic_config)
    stochastic_solver = DynamicProgrammingSolver(
        stochastic_env,
        gamma=0.99,
        verbose=True
    )

    result_stoch = stochastic_solver.value_iteration()
    stochastic_env.render_policy(result_stoch.policy)
    stochastic_env.render_values(result_stoch.value_function)

    # Evaluate in stochastic environment
    stochastic_executor = PolicyExecutor(stochastic_env, seed=42)
    stochastic_stats = stochastic_executor.evaluate_policy(
        result_stoch.policy,
        num_episodes=100
    )
    print(f"\nStochastic Environment Statistics (100 episodes):")
    print(f"  Mean reward: {stochastic_stats['mean_reward']:.2f} ± {stochastic_stats['std_reward']:.2f}")
    print(f"  Mean steps: {stochastic_stats['mean_steps']:.2f}")
    print(f"  Success rate: {stochastic_stats['success_rate']*100:.1f}%")

    print("\n" + "=" * 70)
    print("Demonstration Complete")
    print("=" * 70)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="MDP Basics: Dynamic Programming for GridWorld",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Run tests + demonstration
    python main.py --test-only  # Run tests only
    python main.py --demo-only  # Run demonstration only
        """
    )

    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Run unit tests only'
    )
    parser.add_argument(
        '--demo-only',
        action='store_true',
        help='Run demonstration only (skip tests)'
    )

    args = parser.parse_args()

    if args.test_only:
        success = run_tests()
        sys.exit(0 if success else 1)

    if args.demo_only:
        run_demonstration()
        return

    # Default: run tests first, then demonstration
    print("Running unit tests first...\n")
    success = run_tests()

    if not success:
        print("\nUnit tests failed. Please fix issues before running demonstration.")
        sys.exit(1)

    run_demonstration()


if __name__ == "__main__":
    main()
