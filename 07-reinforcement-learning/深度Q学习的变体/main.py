"""
Main Entry Point for DQN Variants Module.

This script demonstrates the usage of the DQN variants module with
a simple training example on CartPole environment.

Usage:
    python main.py [--variant VARIANT] [--episodes NUM] [--eval]

Examples:
    # Train Double DQN for 200 episodes
    python main.py --variant double --episodes 200

    # Train Rainbow and evaluate
    python main.py --variant rainbow --episodes 500 --eval

    # Compare all variants
    python main.py --compare
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import DQNVariantConfig
from core.enums import DQNVariant
from agents.variant_agent import DQNVariantAgent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate DQN variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="double",
        choices=["vanilla", "double", "dueling", "noisy", "categorical", "rainbow"],
        help="DQN variant to use (default: double)",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment name (default: CartPole-v1)",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of training episodes (default: 200)",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate agent after training",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple variants",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    return parser.parse_args()


def get_variant(name: str) -> DQNVariant:
    """Convert string to DQNVariant enum."""
    mapping = {
        "vanilla": DQNVariant.VANILLA,
        "double": DQNVariant.DOUBLE,
        "dueling": DQNVariant.DUELING,
        "noisy": DQNVariant.NOISY,
        "categorical": DQNVariant.CATEGORICAL,
        "rainbow": DQNVariant.RAINBOW,
    }
    return mapping[name.lower()]


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        import gymnasium as gym
        from utils.training import train_agent, evaluate_agent
        from utils.comparison import compare_variants
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install gymnasium: pip install gymnasium")
        return

    # Get environment dimensions
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    if args.compare:
        print("=" * 60)
        print("DQN Variants Comparison")
        print("=" * 60)

        results = compare_variants(
            args.env,
            variants=[DQNVariant.DOUBLE, DQNVariant.DUELING, DQNVariant.RAINBOW],
            num_episodes=args.episodes,
            seed=args.seed,
        )
        return

    # Create configuration
    config = DQNVariantConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_size=10000,
        min_buffer_size=500,
        epsilon_decay_steps=5000,
        seed=args.seed,
    )

    # Create agent
    variant = get_variant(args.variant)
    print(f"\n{'=' * 60}")
    print(f"Training {variant} on {args.env}")
    print(f"{'=' * 60}\n")

    agent = DQNVariantAgent(config, variant)

    # Train
    agent, logger = train_agent(
        agent,
        args.env,
        num_episodes=args.episodes,
        seed=args.seed,
    )

    print(f"\nTraining complete!")
    print(f"Final mean reward (100 ep): {logger.mean_reward:.1f}")

    # Evaluate
    if args.eval:
        print(f"\nEvaluating on 100 episodes...")
        mean_reward, std_reward = evaluate_agent(
            agent, args.env, num_episodes=100, seed=args.seed
        )
        print(f"Evaluation: {mean_reward:.1f} ± {std_reward:.1f}")


if __name__ == "__main__":
    main()
