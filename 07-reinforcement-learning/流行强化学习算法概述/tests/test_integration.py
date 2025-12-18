"""
Integration Tests
=================

End-to-end tests with mock environments.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.ddpg import DDPGConfig, DDPGAgent
from algorithms.td3 import TD3Config, TD3Agent
from algorithms.sac import SACConfig, SACAgent


class MockEnvironment:
    """
    Simple mock environment for testing.

    Simulates a continuous control environment without gym dependency.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        max_action: float = 1.0,
        max_steps: int = 100,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.max_steps = max_steps

        self._step = 0
        self._state = None

        # Create action_space mock
        class ActionSpace:
            def __init__(self, dim, high):
                self.shape = (dim,)
                self.high = np.array([high] * dim)
                self.low = -self.high

            def sample(self):
                return np.random.uniform(self.low, self.high)

        self.action_space = ActionSpace(action_dim, max_action)

    def reset(self):
        """Reset environment."""
        self._step = 0
        self._state = np.random.randn(self.state_dim).astype(np.float32)
        return self._state, {}

    def step(self, action):
        """Take action and return transition."""
        self._step += 1

        # Simple reward: negative distance from origin
        reward = -np.sum(np.abs(self._state)) + np.random.randn() * 0.1

        # Next state: some dynamics
        next_state = (
            self._state * 0.9 +
            action.sum() * 0.1 +
            np.random.randn(self.state_dim).astype(np.float32) * 0.1
        )

        done = self._step >= self.max_steps
        truncated = False

        self._state = next_state
        return next_state, reward, done, truncated, {}


class TestIntegration:
    """Integration tests with mock environment."""

    def test_ddpg_training_loop(self) -> None:
        """Test DDPG can run training loop."""
        env = MockEnvironment(state_dim=4, action_dim=2)

        config = DDPGConfig(
            state_dim=4,
            action_dim=2,
            max_action=1.0,
            hidden_dims=[32, 32],
            buffer_size=200,
            batch_size=16,
            start_timesteps=50,
        )
        agent = DDPGAgent(config)

        state, _ = env.reset()
        total_reward = 0

        for step in range(100):
            if step < config.start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            if step >= config.start_timesteps:
                agent.update()

            total_reward += reward
            state = next_state

            if done:
                state, _ = env.reset()

        assert len(agent.buffer) > 0, "Buffer should have samples"
        assert agent.total_updates > 0, "Should have done some updates"

    def test_td3_training_loop(self) -> None:
        """Test TD3 can run training loop."""
        env = MockEnvironment(state_dim=4, action_dim=2)

        config = TD3Config(
            state_dim=4,
            action_dim=2,
            max_action=1.0,
            hidden_dims=[32, 32],
            buffer_size=200,
            batch_size=16,
            start_timesteps=50,
            policy_delay=2,
        )
        agent = TD3Agent(config)

        state, _ = env.reset()

        for step in range(100):
            if step < config.start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            if step >= config.start_timesteps:
                agent.update()

            state = next_state
            if done:
                state, _ = env.reset()

        assert agent.total_updates > 0

    def test_sac_training_loop(self) -> None:
        """Test SAC can run training loop."""
        env = MockEnvironment(state_dim=4, action_dim=2)

        config = SACConfig(
            state_dim=4,
            action_dim=2,
            max_action=1.0,
            hidden_dims=[32, 32],
            buffer_size=200,
            batch_size=16,
            start_timesteps=50,
            auto_alpha=True,
        )
        agent = SACAgent(config)

        state, _ = env.reset()
        initial_alpha = agent.alpha

        for step in range(100):
            if step < config.start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            if step >= config.start_timesteps:
                agent.update()

            state = next_state
            if done:
                state, _ = env.reset()

        assert agent.total_updates > 0
        # Alpha should have changed (auto-tuning)
        assert agent.alpha != initial_alpha or agent.total_updates < 5

    def test_evaluation_mode(self) -> None:
        """Test evaluation mode uses deterministic actions."""
        env = MockEnvironment(state_dim=4, action_dim=2)

        agents = [
            DDPGAgent(DDPGConfig(state_dim=4, action_dim=2, buffer_size=500, batch_size=32)),
            TD3Agent(TD3Config(state_dim=4, action_dim=2, buffer_size=500, batch_size=32)),
            SACAgent(SACConfig(state_dim=4, action_dim=2, buffer_size=500, batch_size=32)),
        ]

        for agent in agents:
            agent.eval_mode()
            state, _ = env.reset()

            actions = []
            for _ in range(5):
                action = agent.select_action(state, deterministic=True)
                actions.append(action.copy())

            # All deterministic actions should be identical
            for a in actions[1:]:
                assert np.allclose(actions[0], a), \
                    f"{type(agent).__name__} deterministic actions should match"

    def test_gradient_not_nan(self) -> None:
        """Test gradients don't become NaN during training."""
        env = MockEnvironment(state_dim=4, action_dim=2)

        config = SACConfig(
            state_dim=4,
            action_dim=2,
            max_action=1.0,
            hidden_dims=[32, 32],
            buffer_size=200,
            batch_size=16,
        )
        agent = SACAgent(config)

        state, _ = env.reset()

        # Fill buffer with random data
        for _ in range(100):
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            if done:
                state, _ = env.reset()

        # Run many updates
        for _ in range(50):
            metrics = agent.update()

            # Check for NaN losses
            for key, value in metrics.items():
                assert not np.isnan(value), f"{key} is NaN"

        # Check network parameters
        for name, param in agent.actor.named_parameters():
            assert not torch.isnan(param).any(), f"Actor {name} has NaN"

        for name, param in agent.critic.named_parameters():
            assert not torch.isnan(param).any(), f"Critic {name} has NaN"


def run_all_tests() -> None:
    """Run all integration tests."""
    print("=" * 60)
    print("Running Integration Tests")
    print("=" * 60)

    test_class = TestIntegration()
    test_methods = [m for m in dir(test_class) if m.startswith("test_")]

    print(f"\nTestIntegration")
    print("-" * 40)

    for method_name in test_methods:
        try:
            getattr(test_class, method_name)()
            print(f"  [PASS] {method_name}")
        except AssertionError as e:
            print(f"  [FAIL] {method_name}: {e}")
        except Exception as e:
            print(f"  [ERROR] {method_name}: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Integration tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
