"""
Unit Tests for DQN Variants Module.

This module provides comprehensive tests for all components using mock data.
"""

from tests.test_buffers import TestReplayBuffer, TestPrioritizedReplayBuffer, TestNStepReplayBuffer
from tests.test_networks import TestDQNNetwork, TestDuelingNetwork, TestNoisyLinear, TestRainbowNetwork
from tests.test_agent import TestDQNVariantAgent

__all__ = [
    "TestReplayBuffer",
    "TestPrioritizedReplayBuffer",
    "TestNStepReplayBuffer",
    "TestDQNNetwork",
    "TestDuelingNetwork",
    "TestNoisyLinear",
    "TestRainbowNetwork",
    "TestDQNVariantAgent",
]
