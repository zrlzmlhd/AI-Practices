"""
DQN Agent Implementations Module.

This module provides the unified DQN variant agent:
    - DQNVariantAgent: Supports all DQN variants through configuration

Core Idea (核心思想)
====================
统一的Agent接口支持所有DQN变体，通过DQNVariant枚举选择算法：

- VANILLA: 原始DQN
- DOUBLE: Double DQN
- DUELING: Dueling DQN
- NOISY: Noisy Networks
- CATEGORICAL: C51
- DOUBLE_DUELING: Double + Dueling
- RAINBOW: 全部改进组合

Usage Pattern (使用模式)
========================
>>> from agents import DQNVariantAgent
>>> from core import DQNVariantConfig, DQNVariant
>>>
>>> config = DQNVariantConfig(state_dim=4, action_dim=2)
>>> agent = DQNVariantAgent(config, DQNVariant.RAINBOW)
>>> action = agent.select_action(state, training=True)
>>> agent.train_step(state, action, reward, next_state, done)

References:
    [1] Mnih et al. (2015). Human-level control through deep RL.
    [2] van Hasselt et al. (2016). Deep RL with Double Q-learning.
    [3] Wang et al. (2016). Dueling Network Architectures.
    [4] Fortunato et al. (2017). Noisy Networks for Exploration.
    [5] Bellemare et al. (2017). A Distributional Perspective on RL.
    [6] Hessel et al. (2018). Rainbow: Combining Improvements.
"""

from agents.variant_agent import DQNVariantAgent

__all__ = [
    "DQNVariantAgent",
]
