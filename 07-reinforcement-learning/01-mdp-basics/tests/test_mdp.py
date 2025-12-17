"""
Unit Tests for MDP Basics Module

Comprehensive test suite validating:
    - GridWorld environment correctness
    - Dynamic programming algorithm convergence
    - Policy execution accuracy
    - Edge cases and boundary conditions
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld, GridWorldConfig
from src.algorithms import DynamicProgrammingSolver
from src.executor import PolicyExecutor


class TestGridWorldEnvironment(unittest.TestCase):
    """Test cases for GridWorld environment."""

    def setUp(self):
        """Initialize test fixtures."""
        self.config = GridWorldConfig(
            size=4,
            start=(0, 0),
            goal=(3, 3),
            obstacles=[(1, 1)],
            slip_probability=0.0
        )
        self.env = GridWorld(self.config)

    def test_state_space_size(self):
        """Verify state space excludes obstacles."""
        expected_states = 4 * 4 - 1  # 16 cells minus 1 obstacle
        self.assertEqual(len(self.env.states), expected_states)

    def test_obstacle_not_in_states(self):
        """Verify obstacles are excluded from state space."""
        self.assertNotIn((1, 1), self.env.states)

    def test_action_space(self):
        """Verify action space contains expected actions."""
        expected_actions = {'上', '下', '左', '右'}
        self.assertEqual(set(self.env.actions), expected_actions)

    def test_terminal_state_identification(self):
        """Verify terminal state detection."""
        self.assertTrue(self.env.is_terminal((3, 3)))
        self.assertFalse(self.env.is_terminal((0, 0)))
        self.assertFalse(self.env.is_terminal((2, 2)))

    def test_deterministic_transition(self):
        """Verify deterministic transition produces single outcome."""
        transitions = self.env.get_transitions((0, 0), '下')
        self.assertEqual(len(transitions), 1)
        next_state, prob, _ = transitions[0]
        self.assertEqual(next_state, (1, 0))
        self.assertEqual(prob, 1.0)

    def test_boundary_collision(self):
        """Verify wall collision keeps agent in place."""
        # Top boundary
        transitions = self.env.get_transitions((0, 0), '上')
        next_state, _, _ = transitions[0]
        self.assertEqual(next_state, (0, 0))

        # Left boundary
        transitions = self.env.get_transitions((0, 0), '左')
        next_state, _, _ = transitions[0]
        self.assertEqual(next_state, (0, 0))

    def test_obstacle_collision(self):
        """Verify obstacle collision keeps agent in place."""
        # Moving into obstacle at (1,1) from (0,1)
        transitions = self.env.get_transitions((0, 1), '下')
        next_state, _, _ = transitions[0]
        self.assertEqual(next_state, (0, 1))

    def test_goal_reward(self):
        """Verify reaching goal yields goal reward."""
        transitions = self.env.get_transitions((3, 2), '右')
        _, _, reward = transitions[0]
        self.assertEqual(reward, 100.0)

    def test_step_reward(self):
        """Verify regular step yields step penalty."""
        transitions = self.env.get_transitions((0, 0), '下')
        _, _, reward = transitions[0]
        self.assertEqual(reward, -1.0)

    def test_terminal_state_self_loop(self):
        """Verify terminal state transitions to itself with zero reward."""
        transitions = self.env.get_transitions((3, 3), '上')
        next_state, prob, reward = transitions[0]
        self.assertEqual(next_state, (3, 3))
        self.assertEqual(prob, 1.0)
        self.assertEqual(reward, 0.0)


class TestStochasticGridWorld(unittest.TestCase):
    """Test cases for stochastic GridWorld."""

    def setUp(self):
        """Initialize stochastic environment."""
        self.config = GridWorldConfig(
            size=4,
            slip_probability=0.2
        )
        self.env = GridWorld(self.config)

    def test_stochastic_transition_count(self):
        """Verify stochastic transitions have three outcomes."""
        transitions = self.env.get_transitions((1, 1), '右')
        self.assertEqual(len(transitions), 3)

    def test_stochastic_probability_sum(self):
        """Verify transition probabilities sum to 1."""
        transitions = self.env.get_transitions((1, 1), '右')
        total_prob = sum(t[1] for t in transitions)
        self.assertAlmostEqual(total_prob, 1.0)

    def test_main_direction_probability(self):
        """Verify main direction has correct probability."""
        transitions = self.env.get_transitions((1, 1), '右')
        main_prob = max(t[1] for t in transitions)
        self.assertAlmostEqual(main_prob, 0.8)


class TestDynamicProgramming(unittest.TestCase):
    """Test cases for DP algorithms."""

    def setUp(self):
        """Initialize solver with small grid for fast tests."""
        self.config = GridWorldConfig(
            size=3,
            start=(0, 0),
            goal=(2, 2),
            obstacles=[],
            slip_probability=0.0
        )
        self.env = GridWorld(self.config)
        self.solver = DynamicProgrammingSolver(
            self.env,
            gamma=0.99,
            theta=1e-8,
            verbose=False
        )

    def test_policy_evaluation_convergence(self):
        """Verify policy evaluation converges."""
        policy = {
            s: {a: 0.25 for a in self.env.actions}
            for s in self.env.states
        }
        V, iterations = self.solver.evaluate_policy(policy)

        # Should converge within reasonable iterations
        self.assertLess(iterations, 500)
        # Goal state should have zero value
        self.assertEqual(V[(2, 2)], 0.0)

    def test_policy_improvement_deterministic(self):
        """Verify policy improvement produces deterministic policy for non-terminal states."""
        V = {s: 0.0 for s in self.env.states}
        V[(2, 1)] = 50.0
        V[(1, 2)] = 50.0

        policy = self.solver.improve_policy(V)

        for state in self.env.states:
            probs = list(policy[state].values())
            self.assertAlmostEqual(sum(probs), 1.0)
            # Non-terminal states should be deterministic (one 1.0, rest 0.0)
            # Terminal states can have any valid distribution
            if not self.env.is_terminal(state):
                self.assertTrue(any(abs(p - 1.0) < 1e-9 for p in probs))

    def test_policy_iteration_convergence(self):
        """Verify policy iteration converges to optimal."""
        result = self.solver.policy_iteration()

        self.assertTrue(result.converged)
        self.assertIsNotNone(result.policy)
        self.assertIsNotNone(result.value_function)
        self.assertLess(result.iterations, 20)

    def test_value_iteration_convergence(self):
        """Verify value iteration converges to optimal."""
        result = self.solver.value_iteration()

        self.assertTrue(result.converged)
        self.assertIsNotNone(result.policy)
        self.assertIsNotNone(result.value_function)

    def test_algorithms_produce_same_values(self):
        """Verify both algorithms converge to same optimal values."""
        result_pi = self.solver.policy_iteration()
        result_vi = self.solver.value_iteration()

        for state in self.env.states:
            self.assertAlmostEqual(
                result_pi.value_function[state],
                result_vi.value_function[state],
                places=4
            )

    def test_optimal_policy_reaches_goal(self):
        """Verify optimal policy reaches goal from any state."""
        result = self.solver.value_iteration()
        executor = PolicyExecutor(self.env, seed=42)

        stats = executor.evaluate_policy(
            result.policy,
            num_episodes=50,
            max_steps=50
        )

        # Deterministic environment: 100% success rate
        self.assertEqual(stats['success_rate'], 1.0)

    def test_value_function_monotonicity(self):
        """Verify values increase as we approach goal."""
        result = self.solver.value_iteration()
        V = result.value_function

        # States adjacent to goal should have high values
        self.assertGreater(V[(2, 1)], V[(0, 0)])
        self.assertGreater(V[(1, 2)], V[(0, 0)])


class TestPolicyExecutor(unittest.TestCase):
    """Test cases for policy execution."""

    def setUp(self):
        """Initialize executor with optimal policy."""
        self.config = GridWorldConfig(size=3, goal=(2, 2))
        self.env = GridWorld(self.config)
        self.solver = DynamicProgrammingSolver(self.env, gamma=0.99, verbose=False)
        self.result = self.solver.value_iteration()
        self.executor = PolicyExecutor(self.env, seed=42)

    def test_episode_execution(self):
        """Verify episode runs correctly."""
        reward, steps, trajectory = self.executor.run_episode(
            self.result.policy,
            verbose=False
        )

        self.assertIsInstance(reward, float)
        self.assertGreater(len(trajectory), 0)
        self.assertEqual(trajectory[0], self.env.config.start)

    def test_trajectory_reaches_goal(self):
        """Verify trajectory ends at goal."""
        _, _, trajectory = self.executor.run_episode(
            self.result.policy,
            verbose=False
        )

        self.assertEqual(trajectory[-1], self.env.config.goal)

    def test_evaluation_statistics(self):
        """Verify evaluation returns valid statistics."""
        stats = self.executor.evaluate_policy(
            self.result.policy,
            num_episodes=10,
            max_steps=50
        )

        self.assertIn('mean_reward', stats)
        self.assertIn('std_reward', stats)
        self.assertIn('mean_steps', stats)
        self.assertIn('success_rate', stats)

        self.assertGreaterEqual(stats['success_rate'], 0.0)
        self.assertLessEqual(stats['success_rate'], 1.0)

    def test_optimal_path_extraction(self):
        """Verify optimal path extraction."""
        path = self.executor.get_optimal_path(self.result.policy)

        self.assertEqual(path[0], self.env.config.start)
        self.assertEqual(path[-1], self.env.config.goal)
        self.assertLessEqual(len(path), self.env.num_states)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation."""

    def test_invalid_grid_size(self):
        """Verify rejection of invalid grid size."""
        with self.assertRaises(ValueError):
            GridWorldConfig(size=1)

    def test_invalid_slip_probability(self):
        """Verify rejection of invalid slip probability."""
        with self.assertRaises(ValueError):
            GridWorldConfig(slip_probability=-0.1)

        with self.assertRaises(ValueError):
            GridWorldConfig(slip_probability=1.5)

    def test_start_on_obstacle(self):
        """Verify rejection of start position on obstacle."""
        with self.assertRaises(ValueError):
            config = GridWorldConfig(
                obstacles=[(0, 0)],
                start=(0, 0)
            )
            GridWorld(config)

    def test_goal_on_obstacle(self):
        """Verify rejection of goal position on obstacle."""
        with self.assertRaises(ValueError):
            config = GridWorldConfig(
                obstacles=[(3, 3)],
                goal=(3, 3)
            )
            GridWorld(config)

    def test_invalid_gamma(self):
        """Verify rejection of invalid discount factor."""
        env = GridWorld(GridWorldConfig())

        with self.assertRaises(ValueError):
            DynamicProgrammingSolver(env, gamma=-0.1)

        with self.assertRaises(ValueError):
            DynamicProgrammingSolver(env, gamma=1.5)

    def test_invalid_theta(self):
        """Verify rejection of invalid convergence threshold."""
        env = GridWorld(GridWorldConfig())

        with self.assertRaises(ValueError):
            DynamicProgrammingSolver(env, theta=0.0)

        with self.assertRaises(ValueError):
            DynamicProgrammingSolver(env, theta=-0.001)


def run_all_tests():
    """Run complete test suite with verbose output."""
    print("=" * 70)
    print("MDP Basics - Unit Test Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestGridWorldEnvironment,
        TestStochasticGridWorld,
        TestDynamicProgramming,
        TestPolicyExecutor,
        TestConfigurationValidation,
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
