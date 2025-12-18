#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Gradient Methods - Main Entry Point.

This script provides a command-line interface for training and evaluating
policy gradient algorithms.

Usage:
    # Run unit tests
    python main.py --mode test

    # Train a single algorithm
    python main.py --mode train --algo a2c --env CartPole-v1 --episodes 500

    # Compare multiple algorithms
    python main.py --mode compare --env CartPole-v1 --episodes 300

    # Evaluate trained agent
    python main.py --mode eval --algo a2c --env CartPole-v1 --episodes 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Conditional imports
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None

# Local imports
from core.config import TrainingConfig
from algorithms.reinforce import REINFORCE
from algorithms.reinforce_baseline import REINFORCEBaseline
from algorithms.a2c import A2C
from utils.training import train_policy_gradient, evaluate_agent, compare_algorithms
from utils.visualization import plot_training_curves
from tests.test_policy_gradient import run_tests


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Policy Gradient Methods - Training and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode test
    python main.py --mode train --algo reinforce --episodes 300
    python main.py --mode compare --episodes 500
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "train", "compare", "eval"],
        help="Running mode: test, train, compare, or eval",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment name",
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="a2c",
        choices=["reinforce", "reinforce_baseline", "a2c"],
        help="Algorithm to train",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Number of training episodes",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation",
    )

    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save training curve plot",
    )

    return parser.parse_args()


def create_agent(algo: str, state_dim: int, action_dim: int, config: TrainingConfig, continuous: bool = False):
    """Create agent based on algorithm name."""
    if algo == "reinforce":
        return REINFORCE(state_dim, action_dim, config, continuous)
    elif algo == "reinforce_baseline":
        return REINFORCEBaseline(state_dim, action_dim, config, continuous)
    elif algo == "a2c":
        return A2C(state_dim, action_dim, config, continuous, use_gae=True)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def main():
    """Main entry point."""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "test":
        # Run unit tests
        success = run_tests()
        sys.exit(0 if success else 1)

    elif args.mode == "train":
        if not HAS_GYMNASIUM:
            print("Error: gymnasium is required for training.")
            print("Install with: pip install gymnasium")
            sys.exit(1)

        # Get environment info
        env = gym.make(args.env)
        state_dim = env.observation_space.shape[0]

        if isinstance(env.action_space, spaces.Discrete):
            action_dim = env.action_space.n
            continuous = False
        else:
            action_dim = env.action_space.shape[0]
            continuous = True
        env.close()

        # Create config and agent
        config = TrainingConfig(
            gamma=args.gamma,
            lr_actor=args.lr,
            lr_critic=args.lr,
            entropy_coef=0.01,
        )

        agent = create_agent(args.algo, state_dim, action_dim, config, continuous)

        # Train
        agent, rewards = train_policy_gradient(
            agent,
            args.env,
            num_episodes=args.episodes,
            log_interval=50,
            seed=args.seed,
        )

        # Plot results
        if args.save_plot:
            plot_training_curves(
                {args.algo.upper(): rewards},
                save_path=args.save_plot,
            )

    elif args.mode == "compare":
        if not HAS_GYMNASIUM:
            print("Error: gymnasium is required for training.")
            sys.exit(1)

        # Compare all algorithms
        results = compare_algorithms(
            args.env,
            args.episodes,
            args.seed,
        )

        # Plot comparison
        save_path = args.save_plot or "policy_gradient_comparison.png"
        plot_training_curves(results, save_path=save_path)

    elif args.mode == "eval":
        if not HAS_GYMNASIUM:
            print("Error: gymnasium is required for evaluation.")
            sys.exit(1)

        # Get environment info
        env = gym.make(args.env)
        state_dim = env.observation_space.shape[0]

        if isinstance(env.action_space, spaces.Discrete):
            action_dim = env.action_space.n
            continuous = False
        else:
            action_dim = env.action_space.shape[0]
            continuous = True
        env.close()

        config = TrainingConfig(gamma=args.gamma, lr_actor=args.lr)
        agent = create_agent(args.algo, state_dim, action_dim, config, continuous)

        # Quick training for evaluation demo
        print("Training agent for evaluation...")
        agent, _ = train_policy_gradient(
            agent,
            args.env,
            num_episodes=min(args.episodes, 100),
            verbose=False,
            seed=args.seed,
        )

        # Evaluate
        mean_reward, std_reward = evaluate_agent(
            agent,
            args.env,
            num_episodes=10,
            render=args.render,
            seed=args.seed,
        )

        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
