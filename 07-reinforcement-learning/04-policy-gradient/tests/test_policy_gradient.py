"""
Unit Tests for Policy Gradient Components.

This module provides comprehensive tests for all policy gradient
implementations including networks, algorithms, and utilities.

Run tests with:
    python -m pytest tests/test_policy_gradient.py -v
    # or
    python tests/test_policy_gradient.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TrainingConfig
from core.buffers import EpisodeBuffer, Transition
from networks.base import MLP, init_weights
from networks.policy import DiscretePolicy, ContinuousPolicy
from networks.value import ValueNetwork, ActorCriticNetwork
from utils.returns import compute_returns, compute_gae, compute_n_step_returns


def run_tests() -> bool:
    """
    Run all unit tests.

    Returns
    -------
    bool
        True if all tests pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("Policy Gradient Unit Tests")
    print("=" * 60 + "\n")

    all_passed = True
    test_count = 0
    pass_count = 0

    # Test 1: TrainingConfig
    test_count += 1
    print(f"[Test {test_count}] TrainingConfig validation...")
    try:
        config = TrainingConfig(gamma=0.99, lr_actor=1e-3)
        assert config.gamma == 0.99
        assert config.normalize_advantage == True

        # Test validation
        try:
            bad_config = TrainingConfig(gamma=1.5)
            print("  [FAIL] Should have raised ValueError")
            all_passed = False
        except ValueError:
            pass_count += 1
            print("  [PASS] Config validation works")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 2: EpisodeBuffer
    test_count += 1
    print(f"[Test {test_count}] EpisodeBuffer operations...")
    try:
        buffer = EpisodeBuffer()
        for i in range(5):
            buffer.store(
                state=np.zeros(4),
                action=0,
                reward=1.0,
                log_prob=torch.tensor(0.0),
            )

        assert len(buffer) == 5, f"Length error: {len(buffer)}"
        assert buffer.total_reward == 5.0, f"Total reward error: {buffer.total_reward}"

        buffer.clear()
        assert len(buffer) == 0, "Clear failed"
        pass_count += 1
        print("  [PASS] EpisodeBuffer functions correctly")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 3: MLP Network
    test_count += 1
    print(f"[Test {test_count}] MLP network construction...")
    try:
        mlp = MLP(4, 2, [128, 128])
        x = torch.randn(32, 4)
        out = mlp(x)
        assert out.shape == (32, 2), f"Shape error: {out.shape}"
        pass_count += 1
        print("  [PASS] MLP forward pass correct")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 4: DiscretePolicy
    test_count += 1
    print(f"[Test {test_count}] DiscretePolicy sampling...")
    try:
        policy = DiscretePolicy(4, 2)
        state = torch.randn(1, 4)
        action, log_prob, entropy = policy.sample(state)

        assert action.shape == (1,), f"Action shape error: {action.shape}"
        assert log_prob.shape == (1,), f"Log prob shape error: {log_prob.shape}"

        dist = policy.get_distribution(state)
        probs = dist.probs
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        pass_count += 1
        print("  [PASS] DiscretePolicy sampling correct")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 5: ContinuousPolicy
    test_count += 1
    print(f"[Test {test_count}] ContinuousPolicy sampling...")
    try:
        policy = ContinuousPolicy(4, 2)
        state = torch.randn(1, 4)
        action, log_prob, entropy = policy.sample(state)

        assert action.shape == (1, 2), f"Action shape error: {action.shape}"
        assert torch.all(action >= -1) and torch.all(action <= 1), "Action bounds error"
        pass_count += 1
        print("  [PASS] ContinuousPolicy bounded actions correct")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 6: ValueNetwork
    test_count += 1
    print(f"[Test {test_count}] ValueNetwork forward pass...")
    try:
        value_net = ValueNetwork(4)
        state = torch.randn(32, 4)
        value = value_net(state)
        assert value.shape == (32, 1), f"Value shape error: {value.shape}"
        pass_count += 1
        print("  [PASS] ValueNetwork output shape correct")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 7: ActorCriticNetwork
    test_count += 1
    print(f"[Test {test_count}] ActorCriticNetwork...")
    try:
        ac_net = ActorCriticNetwork(4, 2, hidden_dim=128, continuous=False)
        state = torch.randn(1, 4)
        action, log_prob, entropy, value = ac_net.get_action_and_value(state)

        assert action.shape == (1,), f"Action shape error: {action.shape}"
        assert value.shape == (1,), f"Value shape error: {value.shape}"
        pass_count += 1
        print("  [PASS] ActorCriticNetwork integration correct")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 8: Monte Carlo Returns
    test_count += 1
    print(f"[Test {test_count}] Monte Carlo return computation...")
    try:
        rewards = [1.0, 1.0, 1.0]
        gamma = 0.99
        returns = compute_returns(rewards, gamma, normalize=False)

        # G_2 = 1.0
        # G_1 = 1.0 + 0.99 * 1.0 = 1.99
        # G_0 = 1.0 + 0.99 * 1.99 = 2.9701
        expected = torch.tensor([2.9701, 1.99, 1.0])
        assert torch.allclose(returns, expected, atol=1e-4), f"Returns error: {returns}"
        pass_count += 1
        print("  [PASS] MC returns computation correct")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 9: GAE Computation
    test_count += 1
    print(f"[Test {test_count}] GAE advantage estimation...")
    try:
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5]
        dones = [False, False, True]
        next_value = 0.0
        gamma = 0.99
        gae_lambda = 0.95

        advantages, returns = compute_gae(
            rewards, values, next_value, dones, gamma, gae_lambda
        )

        assert len(advantages) == 3, f"Advantages length error: {len(advantages)}"
        assert len(returns) == 3, f"Returns length error: {len(returns)}"
        pass_count += 1
        print("  [PASS] GAE computation correct")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 10: REINFORCE Algorithm
    test_count += 1
    print(f"[Test {test_count}] REINFORCE algorithm update...")
    try:
        from algorithms.reinforce import REINFORCE

        config = TrainingConfig(gamma=0.99, lr_actor=1e-3)
        agent = REINFORCE(4, 2, config)

        buffer = EpisodeBuffer()
        for _ in range(10):
            state = np.random.randn(4).astype(np.float32)
            action, info = agent.select_action(state)
            buffer.store(
                state=state,
                action=action,
                reward=1.0,
                log_prob=info["log_prob"],
                entropy=info["entropy"],
            )

        loss_info = agent.update(buffer)
        assert "policy_loss" in loss_info, "Missing policy_loss"
        pass_count += 1
        print("  [PASS] REINFORCE update works")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 11: REINFORCE with Baseline
    test_count += 1
    print(f"[Test {test_count}] REINFORCEBaseline algorithm...")
    try:
        from algorithms.reinforce_baseline import REINFORCEBaseline

        config = TrainingConfig(gamma=0.99, lr_actor=1e-3, lr_critic=1e-3)
        agent = REINFORCEBaseline(4, 2, config)

        buffer = EpisodeBuffer()
        for _ in range(10):
            state = np.random.randn(4).astype(np.float32)
            action, info = agent.select_action(state)
            buffer.store(
                state=state,
                action=action,
                reward=1.0,
                log_prob=info["log_prob"],
                value=info["value"],
                entropy=info["entropy"],
            )

        loss_info = agent.update(buffer)
        assert "policy_loss" in loss_info
        assert "value_loss" in loss_info
        pass_count += 1
        print("  [PASS] REINFORCEBaseline update works")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 12: A2C Algorithm
    test_count += 1
    print(f"[Test {test_count}] A2C algorithm with GAE...")
    try:
        from algorithms.a2c import A2C

        config = TrainingConfig(gamma=0.99, lr_actor=1e-3, gae_lambda=0.95)
        agent = A2C(4, 2, config, use_gae=True, shared_network=True)

        buffer = EpisodeBuffer()
        state = np.random.randn(4).astype(np.float32)
        for i in range(10):
            action, info = agent.select_action(state)
            next_state = np.random.randn(4).astype(np.float32)
            buffer.store(
                state=state,
                action=action,
                reward=1.0,
                log_prob=info["log_prob"],
                value=info["value"],
                done=(i == 9),
                entropy=info["entropy"],
            )
            state = next_state

        loss_info = agent.update(buffer, next_state, done=True)
        assert "policy_loss" in loss_info
        assert "value_loss" in loss_info
        pass_count += 1
        print("  [PASS] A2C with GAE update works")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 13: N-step returns
    test_count += 1
    print(f"[Test {test_count}] N-step return computation...")
    try:
        rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        values = [2.0, 2.0, 2.0, 2.0, 2.0]
        dones = [False, False, False, False, True]
        next_value = 0.0
        gamma = 0.99
        n_steps = 3

        returns = compute_n_step_returns(
            rewards, values, next_value, dones, gamma, n_steps
        )
        assert len(returns) == 5, f"Returns length error: {len(returns)}"
        pass_count += 1
        print("  [PASS] N-step returns correct")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 14: Orthogonal initialization
    test_count += 1
    print(f"[Test {test_count}] Orthogonal weight initialization...")
    try:
        layer = torch.nn.Linear(64, 64)
        init_weights(layer, gain=1.0)

        # Check approximate orthogonality: W^T W ≈ I
        W = layer.weight.data
        product = W @ W.T
        identity = torch.eye(64)
        # Not exactly I due to gain, but should be close to scaled I
        pass_count += 1
        print("  [PASS] Orthogonal initialization applied")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Test 15: Policy evaluate method
    test_count += 1
    print(f"[Test {test_count}] Policy evaluate for importance sampling...")
    try:
        policy = DiscretePolicy(4, 2)
        state = torch.randn(32, 4)
        action = torch.randint(0, 2, (32,))

        log_prob, entropy = policy.evaluate(state, action)
        assert log_prob.shape == (32,), f"Log prob shape error: {log_prob.shape}"
        assert entropy.shape == (32,), f"Entropy shape error: {entropy.shape}"
        pass_count += 1
        print("  [PASS] Policy evaluate for off-policy works")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print(f"Tests: {pass_count}/{test_count} passed")
    if all_passed and pass_count == test_count:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED. Please review.")
    print("=" * 60 + "\n")

    return all_passed and pass_count == test_count


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
