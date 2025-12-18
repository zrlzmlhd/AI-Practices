"""
Unit Tests for Neural Networks.

Tests all network architectures with mock data using small parameters.
"""

from __future__ import annotations

import unittest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from networks.base import DQNNetwork, init_weights
from networks.dueling import DuelingNetwork
from networks.noisy import NoisyLinear, NoisyNetwork, NoisyDuelingNetwork
from networks.categorical import CategoricalNetwork, CategoricalDuelingNetwork
from networks.rainbow import RainbowNetwork


class TestDQNNetwork(unittest.TestCase):
    """Test cases for standard DQN network."""

    def setUp(self):
        """Set up test fixtures with small parameters."""
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dim = 16  # Small for testing
        self.batch_size = 2

        self.net = DQNNetwork(
            self.state_dim, self.action_dim, self.hidden_dim
        )
        self.sample_state = torch.randn(self.batch_size, self.state_dim)

    def test_output_shape(self):
        """Test output has correct shape."""
        q_values = self.net(self.sample_state)
        self.assertEqual(q_values.shape, (self.batch_size, self.action_dim))

    def test_forward_deterministic(self):
        """Test forward pass is deterministic."""
        q1 = self.net(self.sample_state)
        q2 = self.net(self.sample_state)
        self.assertTrue(torch.allclose(q1, q2))

    def test_gradient_flow(self):
        """Test gradients flow through network."""
        q_values = self.net(self.sample_state)
        loss = q_values.sum()
        loss.backward()

        for param in self.net.parameters():
            self.assertIsNotNone(param.grad)


class TestDuelingNetwork(unittest.TestCase):
    """Test cases for Dueling network."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dim = 16
        self.batch_size = 2

        self.net = DuelingNetwork(
            self.state_dim, self.action_dim, self.hidden_dim
        )
        self.sample_state = torch.randn(self.batch_size, self.state_dim)

    def test_output_shape(self):
        """Test output has correct shape."""
        q_values = self.net(self.sample_state)
        self.assertEqual(q_values.shape, (self.batch_size, self.action_dim))

    def test_value_advantage_decomposition(self):
        """Test V+A-mean(A) decomposition structure exists."""
        # Check that value and advantage streams exist
        self.assertTrue(hasattr(self.net, 'value_stream'))
        self.assertTrue(hasattr(self.net, 'advantage_stream'))


class TestNoisyLinear(unittest.TestCase):
    """Test cases for NoisyLinear layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.in_features = 16
        self.out_features = 8
        self.batch_size = 2

        self.layer = NoisyLinear(self.in_features, self.out_features)
        self.sample_input = torch.randn(self.batch_size, self.in_features)

    def test_output_shape(self):
        """Test output shape."""
        output = self.layer(self.sample_input)
        self.assertEqual(output.shape, (self.batch_size, self.out_features))

    def test_noise_reset_changes_output(self):
        """Test that resetting noise changes output."""
        self.layer.train()
        self.layer.reset_noise()
        out1 = self.layer(self.sample_input).clone()
        self.layer.reset_noise()
        out2 = self.layer(self.sample_input)

        # Outputs should differ due to different noise
        self.assertFalse(torch.allclose(out1, out2))

    def test_eval_mode_deterministic(self):
        """Test eval mode gives deterministic output."""
        self.layer.eval()
        out1 = self.layer(self.sample_input)
        self.layer.reset_noise()
        out2 = self.layer(self.sample_input)

        # In eval mode, noise is not used
        self.assertTrue(torch.allclose(out1, out2))

    def test_parameters(self):
        """Test learnable parameters exist."""
        param_names = [name for name, _ in self.layer.named_parameters()]
        self.assertIn('weight_mu', param_names)
        self.assertIn('weight_sigma', param_names)
        self.assertIn('bias_mu', param_names)
        self.assertIn('bias_sigma', param_names)


class TestNoisyNetwork(unittest.TestCase):
    """Test cases for NoisyNetwork."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dim = 16
        self.batch_size = 2

        self.net = NoisyNetwork(
            self.state_dim, self.action_dim, self.hidden_dim
        )
        self.sample_state = torch.randn(self.batch_size, self.state_dim)

    def test_output_shape(self):
        """Test output shape."""
        q_values = self.net(self.sample_state)
        self.assertEqual(q_values.shape, (self.batch_size, self.action_dim))

    def test_reset_noise(self):
        """Test reset_noise method."""
        self.net.train()
        self.net.reset_noise()
        out1 = self.net(self.sample_state).clone()
        self.net.reset_noise()
        out2 = self.net(self.sample_state)

        self.assertFalse(torch.allclose(out1, out2))


class TestCategoricalNetwork(unittest.TestCase):
    """Test cases for CategoricalNetwork (C51)."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dim = 16
        self.num_atoms = 11  # Small for testing
        self.v_min = -5.0
        self.v_max = 5.0
        self.batch_size = 2

        self.net = CategoricalNetwork(
            self.state_dim, self.action_dim, self.hidden_dim,
            self.num_atoms, self.v_min, self.v_max
        )
        self.sample_state = torch.randn(self.batch_size, self.state_dim)

    def test_log_probs_shape(self):
        """Test log probabilities shape."""
        log_probs = self.net(self.sample_state)
        self.assertEqual(
            log_probs.shape,
            (self.batch_size, self.action_dim, self.num_atoms)
        )

    def test_probs_sum_to_one(self):
        """Test probabilities sum to 1."""
        log_probs = self.net(self.sample_state)
        probs = log_probs.exp()
        prob_sums = probs.sum(dim=-1)

        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))

    def test_q_values_shape(self):
        """Test get_q_values method."""
        q_values = self.net.get_q_values(self.sample_state)
        self.assertEqual(q_values.shape, (self.batch_size, self.action_dim))

    def test_q_values_in_range(self):
        """Test Q-values are within support range."""
        q_values = self.net.get_q_values(self.sample_state)
        self.assertTrue(torch.all(q_values >= self.v_min))
        self.assertTrue(torch.all(q_values <= self.v_max))

    def test_support_shape(self):
        """Test support tensor."""
        self.assertEqual(self.net.support.shape, (self.num_atoms,))
        self.assertAlmostEqual(self.net.support[0].item(), self.v_min)
        self.assertAlmostEqual(self.net.support[-1].item(), self.v_max)


class TestRainbowNetwork(unittest.TestCase):
    """Test cases for RainbowNetwork."""

    def setUp(self):
        """Set up test fixtures with small parameters."""
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dim = 16
        self.num_atoms = 11
        self.v_min = -5.0
        self.v_max = 5.0
        self.batch_size = 2

        self.net = RainbowNetwork(
            self.state_dim, self.action_dim, self.hidden_dim,
            self.num_atoms, self.v_min, self.v_max
        )
        self.sample_state = torch.randn(self.batch_size, self.state_dim)

    def test_log_probs_shape(self):
        """Test log probabilities shape."""
        log_probs = self.net(self.sample_state)
        self.assertEqual(
            log_probs.shape,
            (self.batch_size, self.action_dim, self.num_atoms)
        )

    def test_q_values_shape(self):
        """Test Q-values shape."""
        q_values = self.net.get_q_values(self.sample_state)
        self.assertEqual(q_values.shape, (self.batch_size, self.action_dim))

    def test_reset_noise(self):
        """Test noise reset in Rainbow."""
        self.net.train()
        self.net.reset_noise()
        out1 = self.net(self.sample_state).clone()
        self.net.reset_noise()
        out2 = self.net(self.sample_state)

        self.assertFalse(torch.allclose(out1, out2))

    def test_noisy_layers_exist(self):
        """Test that noisy layers are present."""
        self.assertTrue(hasattr(self.net, 'value_noisy1'))
        self.assertTrue(hasattr(self.net, 'value_noisy2'))
        self.assertTrue(hasattr(self.net, 'adv_noisy1'))
        self.assertTrue(hasattr(self.net, 'adv_noisy2'))

    def test_gradient_flow(self):
        """Test gradients flow through all components."""
        log_probs = self.net(self.sample_state)
        loss = log_probs.sum()
        loss.backward()

        for name, param in self.net.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")


if __name__ == "__main__":
    unittest.main()
