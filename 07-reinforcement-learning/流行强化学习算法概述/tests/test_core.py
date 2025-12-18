"""
Core Module Tests
=================

Unit tests for config, buffer, networks, and base agent.
"""

import sys
import os
import numpy as np
import torch
import tempfile

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import BaseConfig, DeviceMixin
from core.buffer import ReplayBuffer, Transition
from core.networks import (
    DeterministicActor,
    GaussianActor,
    QNetwork,
    TwinQNetwork,
    ValueNetwork,
    orthogonal_init,
    create_mlp,
)
from core.base_agent import BaseAgent


class TestConfig:
    """Tests for BaseConfig class."""

    def test_instantiation(self) -> None:
        """Test basic config creation."""
        config = BaseConfig(state_dim=4, action_dim=2)
        assert config.state_dim == 4
        assert config.action_dim == 2
        assert config.gamma == 0.99  # Default value

    def test_device_auto_selection(self) -> None:
        """Test automatic device selection."""
        config = BaseConfig(state_dim=4, action_dim=2, device="auto")
        device = config.get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]

    def test_validation_valid(self) -> None:
        """Test validation passes for valid config."""
        config = BaseConfig(
            state_dim=10,
            action_dim=3,
            max_action=2.0,
            hidden_dims=[256, 256],
            lr_actor=1e-4,
            gamma=0.99,
        )
        config.validate()  # Should not raise

    def test_validation_invalid_state_dim(self) -> None:
        """Test validation fails for invalid state_dim."""
        config = BaseConfig(state_dim=-1, action_dim=2)
        try:
            config.validate()
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "state_dim" in str(e)

    def test_validation_invalid_gamma(self) -> None:
        """Test validation fails for gamma > 1."""
        config = BaseConfig(state_dim=4, action_dim=2, gamma=1.5)
        try:
            config.validate()
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "gamma" in str(e)

    def test_serialization(self) -> None:
        """Test JSON serialization round-trip."""
        config = BaseConfig(state_dim=8, action_dim=4, gamma=0.95)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            config.to_json(path)
            loaded = BaseConfig.from_json(path)
            assert loaded.state_dim == 8
            assert loaded.action_dim == 4
            assert loaded.gamma == 0.95
        finally:
            os.unlink(path)


class TestReplayBuffer:
    """Tests for ReplayBuffer class."""

    def test_push_and_sample(self) -> None:
        """Test basic push and sample operations."""
        buffer = ReplayBuffer(capacity=100, state_dim=4, action_dim=2)

        for i in range(50):
            state = np.random.randn(4).astype(np.float32)
            action = np.random.randn(2).astype(np.float32)
            buffer.push(state, action, 1.0, state, False)

        assert len(buffer) == 50

        states, actions, rewards, next_states, dones = buffer.sample(16)
        assert states.shape == (16, 4)
        assert actions.shape == (16, 2)
        assert rewards.shape == (16, 1)

    def test_circular_buffer(self) -> None:
        """Test circular buffer behavior at capacity."""
        buffer = ReplayBuffer(capacity=10, state_dim=2, action_dim=1)

        for i in range(25):
            buffer.push(
                np.array([i, i], dtype=np.float32),
                np.array([i], dtype=np.float32),
                float(i),
                np.array([i, i], dtype=np.float32),
                False,
            )

        assert len(buffer) == 10  # Should not exceed capacity

    def test_is_ready(self) -> None:
        """Test is_ready check."""
        buffer = ReplayBuffer(capacity=100)

        for i in range(30):
            buffer.push(np.zeros(3), np.zeros(1), 0.0, np.zeros(3), False)

        assert buffer.is_ready(30)
        assert not buffer.is_ready(31)

    def test_clear(self) -> None:
        """Test buffer clear operation."""
        buffer = ReplayBuffer(capacity=50, state_dim=3, action_dim=1)

        for _ in range(30):
            buffer.push(np.zeros(3), np.zeros(1), 0.0, np.zeros(3), False)

        buffer.clear()
        assert len(buffer) == 0

    def test_device_placement(self) -> None:
        """Test tensor device placement."""
        buffer = ReplayBuffer(capacity=50, state_dim=3, action_dim=1)

        for _ in range(20):
            buffer.push(np.zeros(3), np.zeros(1), 0.0, np.zeros(3), False)

        states, _, _, _, _ = buffer.sample(8, device=torch.device("cpu"))
        assert states.device.type == "cpu"


class TestNetworks:
    """Tests for neural network architectures."""

    def test_orthogonal_init(self) -> None:
        """Test orthogonal initialization."""
        layer = torch.nn.Linear(64, 64)
        orthogonal_init(layer, gain=1.0)

        weight = layer.weight.data
        gram = weight @ weight.T
        identity = torch.eye(64)
        error = (gram - identity).abs().mean().item()
        assert error < 0.1, f"Orthogonality error: {error}"

    def test_create_mlp(self) -> None:
        """Test MLP factory function."""
        mlp = create_mlp(10, [64, 64], 3)
        x = torch.randn(32, 10)
        y = mlp(x)
        assert y.shape == (32, 3)

    def test_deterministic_actor(self) -> None:
        """Test DeterministicActor."""
        actor = DeterministicActor(state_dim=10, action_dim=3, max_action=2.0)
        state = torch.randn(32, 10)
        action = actor(state)

        assert action.shape == (32, 3)
        assert (action.abs() <= 2.0).all()

    def test_gaussian_actor_sample(self) -> None:
        """Test GaussianActor sampling."""
        actor = GaussianActor(state_dim=10, action_dim=3, max_action=1.0)
        state = torch.randn(32, 10)

        action, log_prob = actor.sample(state)
        assert action.shape == (32, 3)
        assert log_prob.shape == (32, 1)
        assert (action.abs() <= 1.0).all()

    def test_gaussian_actor_deterministic(self) -> None:
        """Test GaussianActor deterministic mode."""
        actor = GaussianActor(state_dim=10, action_dim=3, max_action=1.0)
        state = torch.randn(1, 10)

        action1, _ = actor.sample(state, deterministic=True)
        action2, _ = actor.sample(state, deterministic=True)
        assert torch.allclose(action1, action2)

    def test_q_network(self) -> None:
        """Test QNetwork."""
        q_net = QNetwork(state_dim=10, action_dim=3)
        state = torch.randn(32, 10)
        action = torch.randn(32, 3)

        q_value = q_net(state, action)
        assert q_value.shape == (32, 1)

    def test_twin_q_network(self) -> None:
        """Test TwinQNetwork."""
        twin_q = TwinQNetwork(state_dim=10, action_dim=3)
        state = torch.randn(32, 10)
        action = torch.randn(32, 3)

        q1, q2 = twin_q(state, action)
        assert q1.shape == (32, 1)
        assert q2.shape == (32, 1)

        # Q1 and Q2 should differ (independent networks)
        assert not torch.allclose(q1, q2)

        q_min = twin_q.min_q(state, action)
        assert (q_min == torch.min(q1, q2)).all()

    def test_value_network(self) -> None:
        """Test ValueNetwork."""
        v_net = ValueNetwork(state_dim=10)
        state = torch.randn(32, 10)

        value = v_net(state)
        assert value.shape == (32, 1)

    def test_gradient_flow(self) -> None:
        """Test gradients flow through networks."""
        actor = DeterministicActor(state_dim=5, action_dim=2, max_action=1.0)
        state = torch.randn(8, 5, requires_grad=True)

        action = actor(state)
        loss = action.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in actor.parameters()
        )
        assert has_grad, "Gradients should flow through actor"


def run_all_tests() -> None:
    """Run all core module tests."""
    print("=" * 60)
    print("Running Core Module Tests")
    print("=" * 60)

    test_classes = [TestConfig, TestReplayBuffer, TestNetworks]

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
    print("Core module tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
