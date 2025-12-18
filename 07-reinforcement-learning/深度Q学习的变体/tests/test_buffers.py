"""
Unit Tests for Replay Buffers.

Tests all buffer implementations with mock data using small parameters.
"""

from __future__ import annotations

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from buffers.base import ReplayBuffer
from buffers.sum_tree import SumTree
from buffers.prioritized import PrioritizedReplayBuffer
from buffers.n_step import NStepReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    """Test cases for uniform ReplayBuffer."""

    def setUp(self):
        """Set up test fixtures with mock parameters."""
        self.capacity = 10  # Small capacity for testing
        self.buffer = ReplayBuffer(self.capacity)
        self.state_dim = 4
        self.mock_state = np.zeros(self.state_dim, dtype=np.float32)
        self.mock_next_state = np.ones(self.state_dim, dtype=np.float32)

    def test_init(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.capacity, self.capacity)

    def test_invalid_capacity(self):
        """Test that invalid capacity raises error."""
        with self.assertRaises(ValueError):
            ReplayBuffer(0)
        with self.assertRaises(ValueError):
            ReplayBuffer(-1)

    def test_push_single(self):
        """Test pushing a single transition."""
        self.buffer.push(self.mock_state, 0, 1.0, self.mock_next_state, False)
        self.assertEqual(len(self.buffer), 1)

    def test_push_multiple(self):
        """Test pushing multiple transitions."""
        for i in range(5):
            self.buffer.push(self.mock_state, i % 2, float(i), self.mock_next_state, False)
        self.assertEqual(len(self.buffer), 5)

    def test_push_overflow(self):
        """Test FIFO eviction when buffer is full."""
        for i in range(self.capacity + 5):
            self.buffer.push(self.mock_state, 0, float(i), self.mock_next_state, False)
        self.assertEqual(len(self.buffer), self.capacity)

    def test_sample_shape(self):
        """Test that sample returns correct shapes."""
        batch_size = 2
        for i in range(5):
            self.buffer.push(self.mock_state, i % 2, float(i), self.mock_next_state, i == 4)

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        self.assertEqual(states.shape, (batch_size, self.state_dim))
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(next_states.shape, (batch_size, self.state_dim))
        self.assertEqual(dones.shape, (batch_size,))

    def test_sample_insufficient_data(self):
        """Test that sampling more than buffer size raises error."""
        self.buffer.push(self.mock_state, 0, 1.0, self.mock_next_state, False)
        with self.assertRaises(ValueError):
            self.buffer.sample(5)

    def test_is_ready(self):
        """Test is_ready method."""
        self.assertFalse(self.buffer.is_ready(3))
        for i in range(3):
            self.buffer.push(self.mock_state, 0, 1.0, self.mock_next_state, False)
        self.assertTrue(self.buffer.is_ready(3))

    def test_clear(self):
        """Test buffer clearing."""
        for i in range(5):
            self.buffer.push(self.mock_state, 0, 1.0, self.mock_next_state, False)
        self.buffer.clear()
        self.assertEqual(len(self.buffer), 0)


class TestSumTree(unittest.TestCase):
    """Test cases for SumTree data structure."""

    def setUp(self):
        """Set up test fixtures."""
        self.capacity = 8
        self.tree = SumTree(self.capacity)

    def test_init(self):
        """Test tree initialization."""
        self.assertEqual(len(self.tree), 0)
        self.assertEqual(self.tree.total_priority, 0.0)

    def test_add(self):
        """Test adding elements."""
        self.tree.add(1.0, "data_0")
        self.assertEqual(len(self.tree), 1)
        self.assertAlmostEqual(self.tree.total_priority, 1.0)

    def test_add_multiple(self):
        """Test adding multiple elements."""
        priorities = [1.0, 2.0, 3.0, 4.0]
        for i, p in enumerate(priorities):
            self.tree.add(p, f"data_{i}")
        self.assertEqual(len(self.tree), len(priorities))
        self.assertAlmostEqual(self.tree.total_priority, sum(priorities))

    def test_get(self):
        """Test proportional sampling."""
        self.tree.add(1.0, "first")
        self.tree.add(3.0, "second")

        # Sample with cumsum in first priority range
        idx1, priority1, data1 = self.tree.get(0.5)
        self.assertEqual(data1, "first")

        # Sample with cumsum in second priority range
        idx2, priority2, data2 = self.tree.get(2.0)
        self.assertEqual(data2, "second")

    def test_update_priority(self):
        """Test priority update."""
        self.tree.add(1.0, "data")
        idx, _, _ = self.tree.get(0.5)
        self.tree.update_priority(idx, 5.0)
        self.assertAlmostEqual(self.tree.total_priority, 5.0)

    def test_min_priority(self):
        """Test minimum priority retrieval."""
        self.tree.add(1.0, "a")
        self.tree.add(3.0, "b")
        self.tree.add(2.0, "c")
        self.assertAlmostEqual(self.tree.min_priority(), 1.0)


class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Test cases for PrioritizedReplayBuffer."""

    def setUp(self):
        """Set up test fixtures with mock parameters."""
        self.capacity = 10
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.capacity,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=100,
        )
        self.state_dim = 4
        self.mock_state = np.zeros(self.state_dim, dtype=np.float32)
        self.mock_next_state = np.ones(self.state_dim, dtype=np.float32)

    def test_init(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.capacity, self.capacity)
        self.assertAlmostEqual(self.buffer.beta, 0.4)

    def test_push(self):
        """Test pushing transitions."""
        self.buffer.push(self.mock_state, 0, 1.0, self.mock_next_state, False)
        self.assertEqual(len(self.buffer), 1)

    def test_sample_with_weights(self):
        """Test that sample returns importance sampling weights."""
        batch_size = 2
        for i in range(5):
            self.buffer.push(self.mock_state, 0, float(i), self.mock_next_state, False)

        states, actions, rewards, next_states, dones, indices, weights = (
            self.buffer.sample(batch_size)
        )

        self.assertEqual(states.shape, (batch_size, self.state_dim))
        self.assertEqual(indices.shape, (batch_size,))
        self.assertEqual(weights.shape, (batch_size,))
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all(weights <= 1))

    def test_update_priorities(self):
        """Test priority updates based on TD errors."""
        for i in range(5):
            self.buffer.push(self.mock_state, 0, float(i), self.mock_next_state, False)

        _, _, _, _, _, indices, _ = self.buffer.sample(2)
        td_errors = np.array([0.5, 1.0])
        self.buffer.update_priorities(indices, td_errors)

    def test_beta_annealing(self):
        """Test beta annealing over samples."""
        for i in range(5):
            self.buffer.push(self.mock_state, 0, 1.0, self.mock_next_state, False)

        initial_beta = self.buffer.beta
        for _ in range(10):
            self.buffer.sample(2)

        self.assertGreater(self.buffer.beta, initial_beta)


class TestNStepReplayBuffer(unittest.TestCase):
    """Test cases for NStepReplayBuffer."""

    def setUp(self):
        """Set up test fixtures with mock parameters."""
        self.capacity = 10
        self.n_steps = 3
        self.gamma = 0.99
        self.buffer = NStepReplayBuffer(
            capacity=self.capacity,
            n_steps=self.n_steps,
            gamma=self.gamma,
        )
        self.state_dim = 4
        self.mock_state = np.zeros(self.state_dim, dtype=np.float32)
        self.mock_next_state = np.ones(self.state_dim, dtype=np.float32)

    def test_init(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.n_steps, self.n_steps)
        self.assertEqual(self.buffer.gamma, self.gamma)

    def test_push_accumulates(self):
        """Test that push accumulates n-step transitions."""
        # First push doesn't create n-step transition
        result1 = self.buffer.push(self.mock_state, 0, 1.0, self.mock_next_state, False)
        self.assertIsNone(result1)
        self.assertEqual(len(self.buffer), 0)

        # Second push
        result2 = self.buffer.push(self.mock_state, 0, 1.0, self.mock_next_state, False)
        self.assertIsNone(result2)

        # Third push creates first n-step transition
        result3 = self.buffer.push(self.mock_state, 0, 1.0, self.mock_next_state, False)
        self.assertIsNotNone(result3)
        self.assertEqual(len(self.buffer), 1)

    def test_n_step_return_calculation(self):
        """Test n-step return is calculated correctly."""
        rewards = [1.0, 2.0, 3.0]
        for i, r in enumerate(rewards):
            result = self.buffer.push(self.mock_state, 0, r, self.mock_next_state, False)

        # Expected: 1.0 + 0.99*2.0 + 0.99^2*3.0
        expected_return = 1.0 + self.gamma * 2.0 + (self.gamma ** 2) * 3.0
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.n_step_return, expected_return, places=5)

    def test_episode_termination(self):
        """Test handling of episode termination."""
        # Push 2 transitions then terminal
        self.buffer.push(self.mock_state, 0, 1.0, self.mock_next_state, False)
        self.buffer.push(self.mock_state, 0, 2.0, self.mock_next_state, False)
        result = self.buffer.push(self.mock_state, 0, 3.0, self.mock_next_state, True)

        self.assertIsNotNone(result)
        self.assertTrue(result.done)

    def test_sample(self):
        """Test sampling from n-step buffer."""
        batch_size = 2
        for i in range(10):
            self.buffer.push(
                self.mock_state, 0, 1.0,
                self.mock_next_state, i == 9
            )

        states, actions, returns, next_states, dones, n_steps = (
            self.buffer.sample(batch_size)
        )

        self.assertEqual(states.shape, (batch_size, self.state_dim))
        self.assertEqual(returns.shape, (batch_size,))
        self.assertEqual(n_steps.shape, (batch_size,))


if __name__ == "__main__":
    unittest.main()
