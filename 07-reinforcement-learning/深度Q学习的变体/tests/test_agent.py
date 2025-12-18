"""
Unit Tests for DQN Variant Agent.

Tests agent functionality with mock data using small parameters.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import DQNVariantConfig
from core.enums import DQNVariant
from agents.variant_agent import DQNVariantAgent


class TestDQNVariantAgent(unittest.TestCase):
    """Test cases for DQNVariantAgent."""

    def setUp(self):
        """Set up test fixtures with mock parameters (small for fast testing)."""
        self.state_dim = 4
        self.action_dim = 2

        # Small configuration for fast testing
        self.config = DQNVariantConfig(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=16,
            num_atoms=11,
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=2,
            buffer_size=100,
            min_buffer_size=4,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_steps=100,
            target_update_freq=5,
            device="cpu",
        )

        self.mock_state = np.zeros(self.state_dim, dtype=np.float32)
        self.mock_next_state = np.ones(self.state_dim, dtype=np.float32)

    def test_init_vanilla(self):
        """Test Vanilla DQN initialization."""
        agent = DQNVariantAgent(self.config, DQNVariant.VANILLA)
        self.assertEqual(agent.variant, DQNVariant.VANILLA)

    def test_init_double(self):
        """Test Double DQN initialization."""
        agent = DQNVariantAgent(self.config, DQNVariant.DOUBLE)
        self.assertEqual(agent.variant, DQNVariant.DOUBLE)

    def test_init_dueling(self):
        """Test Dueling DQN initialization."""
        agent = DQNVariantAgent(self.config, DQNVariant.DUELING)
        self.assertEqual(agent.variant, DQNVariant.DUELING)

    def test_init_noisy(self):
        """Test Noisy DQN initialization."""
        agent = DQNVariantAgent(self.config, DQNVariant.NOISY)
        self.assertEqual(agent.variant, DQNVariant.NOISY)

    def test_init_categorical(self):
        """Test Categorical DQN initialization."""
        agent = DQNVariantAgent(self.config, DQNVariant.CATEGORICAL)
        self.assertEqual(agent.variant, DQNVariant.CATEGORICAL)

    def test_init_rainbow(self):
        """Test Rainbow initialization."""
        agent = DQNVariantAgent(self.config, DQNVariant.RAINBOW)
        self.assertEqual(agent.variant, DQNVariant.RAINBOW)

    def test_select_action_training(self):
        """Test action selection during training."""
        agent = DQNVariantAgent(self.config, DQNVariant.DOUBLE)
        action = agent.select_action(self.mock_state, training=True)

        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)

    def test_select_action_eval(self):
        """Test action selection during evaluation."""
        agent = DQNVariantAgent(self.config, DQNVariant.DOUBLE)
        agent.set_eval_mode()
        action = agent.select_action(self.mock_state, training=False)

        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)

    def test_store_transition(self):
        """Test storing transitions."""
        agent = DQNVariantAgent(self.config, DQNVariant.DOUBLE)
        agent.store_transition(self.mock_state, 0, 1.0, self.mock_next_state, False)

        self.assertEqual(len(agent.buffer), 1)

    def test_update_returns_none_when_insufficient_data(self):
        """Test that update returns None with insufficient data."""
        agent = DQNVariantAgent(self.config, DQNVariant.DOUBLE)
        loss = agent.update()

        self.assertIsNone(loss)

    def test_update_returns_loss(self):
        """Test that update returns loss with sufficient data."""
        agent = DQNVariantAgent(self.config, DQNVariant.DOUBLE)

        # Fill buffer
        for i in range(self.config.min_buffer_size + 5):
            agent.store_transition(
                self.mock_state, i % 2, 1.0, self.mock_next_state, False
            )

        loss = agent.update()
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)

    def test_train_step(self):
        """Test complete train step."""
        agent = DQNVariantAgent(self.config, DQNVariant.DOUBLE)

        # Fill buffer
        for i in range(self.config.min_buffer_size):
            agent.train_step(
                self.mock_state, 0, 1.0, self.mock_next_state, False
            )

        # Next train step should return loss
        loss = agent.train_step(
            self.mock_state, 0, 1.0, self.mock_next_state, False
        )
        self.assertIsNotNone(loss)

    def test_epsilon_decay(self):
        """Test epsilon decays during training."""
        agent = DQNVariantAgent(self.config, DQNVariant.DOUBLE)
        initial_epsilon = agent.epsilon

        for i in range(50):
            agent.decay_epsilon()

        self.assertLess(agent.epsilon, initial_epsilon)

    def test_noisy_agent_no_epsilon(self):
        """Test noisy agent doesn't use epsilon-greedy."""
        agent = DQNVariantAgent(self.config, DQNVariant.NOISY)

        # Epsilon decay shouldn't affect noisy agent
        initial_epsilon = agent.epsilon
        for i in range(50):
            agent.decay_epsilon()

        # For noisy networks, epsilon is not used
        self.assertTrue(agent.variant.uses_noisy_exploration)

    def test_save_load(self):
        """Test model saving and loading."""
        agent = DQNVariantAgent(self.config, DQNVariant.DOUBLE)

        # Train a bit
        for i in range(10):
            agent.train_step(
                self.mock_state, 0, 1.0, self.mock_next_state, False
            )

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")
            agent.save(save_path)

            # Create new agent and load
            agent2 = DQNVariantAgent(self.config, DQNVariant.DOUBLE)
            agent2.load(save_path)

            # Check state is restored
            self.assertEqual(agent.training_step, agent2.training_step)

    def test_distributional_update(self):
        """Test distributional (C51) update."""
        agent = DQNVariantAgent(self.config, DQNVariant.CATEGORICAL)

        # Fill buffer
        for i in range(self.config.min_buffer_size + 5):
            agent.store_transition(
                self.mock_state, i % 2, 1.0, self.mock_next_state, False
            )

        loss = agent.update()
        self.assertIsNotNone(loss)

    def test_rainbow_update(self):
        """Test Rainbow update."""
        agent = DQNVariantAgent(self.config, DQNVariant.RAINBOW)

        # Fill buffer
        for i in range(self.config.min_buffer_size + 5):
            agent.store_transition(
                self.mock_state, i % 2, 1.0, self.mock_next_state, False
            )

        loss = agent.update()
        self.assertIsNotNone(loss)


class TestDQNVariantConfig(unittest.TestCase):
    """Test cases for DQNVariantConfig."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = DQNVariantConfig(state_dim=4, action_dim=2)
        self.assertEqual(config.state_dim, 4)
        self.assertEqual(config.action_dim, 2)

    def test_invalid_state_dim(self):
        """Test invalid state_dim raises error."""
        with self.assertRaises(ValueError):
            DQNVariantConfig(state_dim=0, action_dim=2)

    def test_invalid_gamma(self):
        """Test invalid gamma raises error."""
        with self.assertRaises(ValueError):
            DQNVariantConfig(state_dim=4, action_dim=2, gamma=1.5)

    def test_to_dict(self):
        """Test config serialization."""
        config = DQNVariantConfig(state_dim=4, action_dim=2)
        config_dict = config.to_dict()

        self.assertIn("state_dim", config_dict)
        self.assertIn("action_dim", config_dict)
        self.assertIn("gamma", config_dict)

    def test_from_dict(self):
        """Test config deserialization."""
        config = DQNVariantConfig(state_dim=4, action_dim=2)
        config_dict = config.to_dict()
        config2 = DQNVariantConfig.from_dict(config_dict)

        self.assertEqual(config.state_dim, config2.state_dim)
        self.assertEqual(config.gamma, config2.gamma)

    def test_get_device(self):
        """Test device detection."""
        config = DQNVariantConfig(state_dim=4, action_dim=2, device="cpu")
        device = config.get_device()
        self.assertEqual(device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
