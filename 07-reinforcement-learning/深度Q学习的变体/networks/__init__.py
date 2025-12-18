"""
Neural Network Architectures Module.

This module provides network architectures for DQN variants:
    - DQNNetwork: Standard Q-network architecture
    - DuelingNetwork: Value-advantage decomposition
    - NoisyLinear: Parametric noise layer for exploration
    - NoisyNetwork: Q-network with noisy layers
    - CategoricalNetwork: Distributional value network (C51)
    - RainbowNetwork: Combined architecture for Rainbow DQN

Core Idea (核心思想)
====================
不同的网络架构针对DQN的不同失效模式：

- **DQNNetwork**: 标准全连接网络，映射 s → Q(s,·)
- **DuelingNetwork**: 分离V(s)和A(s,a)，提升泛化能力
- **NoisyNetwork**: 参数化噪声实现状态依赖探索
- **CategoricalNetwork**: 建模完整回报分布
- **RainbowNetwork**: 组合所有改进

Mathematical Foundation (数学基础)
==================================
Standard Q-network:
    Q: S → ℝ^|A|

Dueling decomposition:
    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)

Noisy layer:
    y = (μ^w + σ^w ⊙ ε^w) x + (μ^b + σ^b ⊙ ε^b)

Distributional Q-network:
    Z(s,a) ~ Categorical(z_1, ..., z_N; p_1, ..., p_N)

Example:
    >>> from networks import DQNNetwork, DuelingNetwork, RainbowNetwork
    >>> net = DuelingNetwork(state_dim=4, action_dim=2)
    >>> q_values = net(torch.randn(1, 4))

References:
    [1] Mnih et al. (2015). Human-level control through deep RL.
    [2] Wang et al. (2016). Dueling Network Architectures.
    [3] Fortunato et al. (2017). Noisy Networks for Exploration.
    [4] Bellemare et al. (2017). A Distributional Perspective on RL.
    [5] Hessel et al. (2018). Rainbow: Combining Improvements.
"""

from networks.base import DQNNetwork, init_weights
from networks.dueling import DuelingNetwork
from networks.noisy import NoisyLinear, NoisyNetwork, NoisyDuelingNetwork
from networks.categorical import CategoricalNetwork, CategoricalDuelingNetwork
from networks.rainbow import RainbowNetwork

__all__ = [
    "DQNNetwork",
    "init_weights",
    "DuelingNetwork",
    "NoisyLinear",
    "NoisyNetwork",
    "NoisyDuelingNetwork",
    "CategoricalNetwork",
    "CategoricalDuelingNetwork",
    "RainbowNetwork",
]
