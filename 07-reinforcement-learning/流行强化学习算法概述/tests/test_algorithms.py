"""
Algorithm Tests
===============

Unit tests for DDPG, TD3, and SAC implementations.
"""

import sys
import os
import numpy as np
import torch
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.ddpg import DDPGConfig, DDPGAgent
from algorithms.td3 import TD3Config, TD3Agent
from algorithms.sac import SACConfig, SACAgent


class TestDDPG:
    """Tests for DDPG algorithm."""

    def setup(self) -> DDPGAgent:
        """Create DDPG agent for testing."""
        config = DDPGConfig(
            state_dim=4,
            action_dim=2,
            max_action=1.0,
            hidden_dims=[32, 32],
            buffer_size=500,
            batch_size=16,
            exploration_noise=0.1,
        )
        return DDPGAgent(config)

    def test_initialization(self) -> None:
        """Test agent initialization."""
        agent = self.setup()
        assert agent.actor is not None
        assert agent.critic is not None
        assert agent.actor_target is not None
        assert agent.critic_target is not None

    def test_select_action_shape(self) -> None:
        """Test action selection returns correct shape."""
        agent = self.setup()
        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state)
        assert action.shape == (2,)
        assert (np.abs(action) <= 1.0).all()

    def test_select_action_deterministic(self) -> None:
        """Test deterministic action selection."""
        agent = self.setup()
        state = np.random.randn(4).astype(np.float32)

        a1 = agent.select_action(state, deterministic=True)
        a2 = agent.select_action(state, deterministic=True)
        assert np.allclose(a1, a2)

    def test_store_transition(self) -> None:
        """Test transition storage."""
        agent = self.setup()

        for _ in range(50):
            agent.store_transition(
                np.random.randn(4).astype(np.float32),
                np.random.randn(2).astype(np.float32),
                1.0,
                np.random.randn(4).astype(np.float32),
                False,
            )

        assert len(agent.buffer) == 50

    def test_update(self) -> None:
        """Test update returns metrics."""
        agent = self.setup()

        # Fill buffer
        for _ in range(50):
            agent.store_transition(
                np.random.randn(4).astype(np.float32),
                np.random.randn(2).astype(np.float32),
                1.0,
                np.random.randn(4).astype(np.float32),
                False,
            )

        metrics = agent.update()
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "q_mean" in metrics

    def test_save_load(self) -> None:
        """Test checkpoint save/load."""
        agent = self.setup()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save(path)
            assert os.path.exists(path)

            agent2 = self.setup()
            agent2.load(path)

            # Verify parameters match
            for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)

    def test_config_validation(self) -> None:
        """Test config validation."""
        try:
            config = DDPGConfig(state_dim=-1, action_dim=2)
            config.validate()
            assert False, "Should raise ValueError"
        except ValueError:
            pass


class TestTD3:
    """Tests for TD3 algorithm."""

    def setup(self) -> TD3Agent:
        """Create TD3 agent for testing."""
        config = TD3Config(
            state_dim=4,
            action_dim=2,
            max_action=1.0,
            hidden_dims=[32, 32],
            buffer_size=500,
            batch_size=16,
            policy_delay=2,
            target_noise=0.2,
            noise_clip=0.5,
        )
        return TD3Agent(config)

    def test_initialization(self) -> None:
        """Test agent initialization."""
        agent = self.setup()
        assert agent.actor is not None
        assert agent.critic is not None
        assert hasattr(agent.critic, 'q1')
        assert hasattr(agent.critic, 'q2')

    def test_twin_q_networks(self) -> None:
        """Test twin Q-networks produce different values."""
        agent = self.setup()
        state = torch.randn(8, 4, device=agent.device)
        action = torch.randn(8, 2, device=agent.device)

        q1, q2 = agent.critic(state, action)
        assert not torch.allclose(q1, q2), "Twin Qs should differ"

    def test_delayed_policy_update(self) -> None:
        """Test policy is updated every policy_delay steps."""
        agent = self.setup()

        # Fill buffer
        for _ in range(50):
            agent.store_transition(
                np.random.randn(4).astype(np.float32),
                np.random.randn(2).astype(np.float32),
                1.0,
                np.random.randn(4).astype(np.float32),
                False,
            )

        # First update - no actor update
        metrics1 = agent.update()
        assert "critic_loss" in metrics1
        assert "actor_loss" not in metrics1

        # Second update - should include actor update
        metrics2 = agent.update()
        assert "actor_loss" in metrics2

    def test_select_action(self) -> None:
        """Test action selection."""
        agent = self.setup()
        state = np.random.randn(4).astype(np.float32)

        action = agent.select_action(state)
        assert action.shape == (2,)
        assert (np.abs(action) <= 1.0).all()

    def test_save_load(self) -> None:
        """Test checkpoint save/load."""
        agent = self.setup()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save(path)
            agent2 = self.setup()
            agent2.load(path)

            for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)


class TestSAC:
    """Tests for SAC algorithm."""

    def setup(self, auto_alpha: bool = True) -> SACAgent:
        """Create SAC agent for testing."""
        config = SACConfig(
            state_dim=4,
            action_dim=2,
            max_action=1.0,
            hidden_dims=[32, 32],
            buffer_size=500,
            batch_size=16,
            auto_alpha=auto_alpha,
            initial_alpha=0.2,
        )
        return SACAgent(config)

    def test_initialization(self) -> None:
        """Test agent initialization."""
        agent = self.setup()
        assert agent.actor is not None
        assert agent.critic is not None
        assert agent.alpha > 0

    def test_stochastic_action(self) -> None:
        """Test stochastic action selection."""
        agent = self.setup()
        state = np.random.randn(4).astype(np.float32)

        # Stochastic actions should differ
        actions = [agent.select_action(state, deterministic=False) for _ in range(5)]
        all_same = all(np.allclose(actions[0], a) for a in actions[1:])
        assert not all_same, "Stochastic actions should vary"

    def test_deterministic_action(self) -> None:
        """Test deterministic action selection."""
        agent = self.setup()
        state = np.random.randn(4).astype(np.float32)

        a1 = agent.select_action(state, deterministic=True)
        a2 = agent.select_action(state, deterministic=True)
        assert np.allclose(a1, a2)

    def test_action_bounds(self) -> None:
        """Test actions are within bounds."""
        agent = self.setup()

        for _ in range(100):
            state = np.random.randn(4).astype(np.float32)
            action = agent.select_action(state)
            assert (np.abs(action) <= 1.0).all()

    def test_update_with_auto_alpha(self) -> None:
        """Test update with automatic temperature."""
        agent = self.setup(auto_alpha=True)

        # Fill buffer
        for _ in range(50):
            agent.store_transition(
                np.random.randn(4).astype(np.float32),
                np.random.randn(2).astype(np.float32),
                1.0,
                np.random.randn(4).astype(np.float32),
                False,
            )

        metrics = agent.update()
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha" in metrics
        assert "alpha_loss" in metrics
        assert "entropy" in metrics

    def test_update_fixed_alpha(self) -> None:
        """Test update with fixed temperature."""
        agent = self.setup(auto_alpha=False)

        for _ in range(50):
            agent.store_transition(
                np.random.randn(4).astype(np.float32),
                np.random.randn(2).astype(np.float32),
                1.0,
                np.random.randn(4).astype(np.float32),
                False,
            )

        initial_alpha = agent.alpha
        metrics = agent.update()

        assert "alpha_loss" not in metrics
        assert abs(agent.alpha - initial_alpha) < 1e-6

    def test_save_load(self) -> None:
        """Test checkpoint save/load."""
        agent = self.setup()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save(path)
            agent2 = self.setup()
            agent2.load(path)

            for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)


class TestAlgorithmComparison:
    """Tests comparing all three algorithms."""

    def test_interface_consistency(self) -> None:
        """Test all agents have consistent interface."""
        agents = [
            DDPGAgent(DDPGConfig(state_dim=4, action_dim=2, buffer_size=100, batch_size=8)),
            TD3Agent(TD3Config(state_dim=4, action_dim=2, buffer_size=100, batch_size=8)),
            SACAgent(SACConfig(state_dim=4, action_dim=2, buffer_size=100, batch_size=8)),
        ]

        state = np.random.randn(4).astype(np.float32)

        for agent in agents:
            # All should have these methods
            action = agent.select_action(state)
            assert action.shape == (2,)

            agent.store_transition(state, action, 1.0, state, False)

            # Fill buffer minimally
            for _ in range(10):
                agent.store_transition(state, action, 1.0, state, False)

            metrics = agent.update()
            assert isinstance(metrics, dict)

    def test_action_bounds_all_agents(self) -> None:
        """Test all agents respect action bounds."""
        max_action = 2.0

        configs = [
            DDPGConfig(state_dim=4, action_dim=2, max_action=max_action, buffer_size=500, batch_size=32),
            TD3Config(state_dim=4, action_dim=2, max_action=max_action, buffer_size=500, batch_size=32),
            SACConfig(state_dim=4, action_dim=2, max_action=max_action, buffer_size=500, batch_size=32),
        ]

        for config_class, config in zip([DDPGAgent, TD3Agent, SACAgent], configs):
            agent = config_class(config)

            for _ in range(50):
                state = np.random.randn(4).astype(np.float32)
                action = agent.select_action(state)
                assert (np.abs(action) <= max_action).all(), \
                    f"{config_class.__name__} violated action bounds"


def run_all_tests() -> None:
    """Run all algorithm tests."""
    print("=" * 60)
    print("Running Algorithm Tests")
    print("=" * 60)

    test_classes = [TestDDPG, TestTD3, TestSAC, TestAlgorithmComparison]

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)

        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in test_methods:
            try:
                getattr(instance, method_name)()
                print(f"  [PASS] {method_name}")
            except AssertionError as e:
                print(f"  [FAIL] {method_name}: {e}")
            except Exception as e:
                print(f"  [ERROR] {method_name}: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Algorithm tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
