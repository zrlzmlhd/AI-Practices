"""
DQN Variant Enumeration.

This module defines the available DQN algorithm variants.

Algorithm Overview (算法概述)
============================
Each variant addresses specific limitations of vanilla DQN:

+------------------+--------------------------------------------------+
| Variant          | Key Innovation                                   |
+==================+==================================================+
| VANILLA          | Original DQN (Mnih et al., 2015)                 |
+------------------+--------------------------------------------------+
| DOUBLE           | Decoupled action selection/evaluation            |
+------------------+--------------------------------------------------+
| DUELING          | Value-advantage decomposition                    |
+------------------+--------------------------------------------------+
| NOISY            | Parametric exploration via noisy layers          |
+------------------+--------------------------------------------------+
| CATEGORICAL      | Full return distribution modeling (C51)          |
+------------------+--------------------------------------------------+
| DOUBLE_DUELING   | Combines Double and Dueling                      |
+------------------+--------------------------------------------------+
| RAINBOW          | All improvements combined                        |
+------------------+--------------------------------------------------+
"""

from enum import Enum


class DQNVariant(Enum):
    """
    Available DQN algorithm variants.

    Core Idea (核心思想)
    --------------------
    每个变体针对原始DQN的特定失效模式进行优化：

    - **VANILLA**: 原始DQN，存在过估计偏差
    - **DOUBLE**: 解耦动作选择与评估，消除过估计
    - **DUELING**: 分离状态价值与动作优势，提升泛化
    - **NOISY**: 参数化噪声实现状态依赖探索
    - **CATEGORICAL**: 建模完整回报分布
    - **DOUBLE_DUELING**: 结合Double和Dueling的优势
    - **RAINBOW**: 集成所有改进，达到最优性能

    Performance on Atari (Atari性能)
    --------------------------------
    Median human-normalized score:

    - VANILLA: 79%
    - DOUBLE: 117%
    - DUELING: 151%
    - CATEGORICAL: 235%
    - RAINBOW: 441%

    Examples
    --------
    >>> variant = DQNVariant.RAINBOW
    >>> variant.value
    'rainbow'
    >>> DQNVariant('double')
    <DQNVariant.DOUBLE: 'double'>

    References
    ----------
    [1] Mnih et al. (2015). Human-level control through deep RL. Nature.
    [2] van Hasselt et al. (2016). Deep RL with Double Q-learning. AAAI.
    [3] Wang et al. (2016). Dueling Network Architectures. ICML.
    [4] Fortunato et al. (2017). Noisy Networks for Exploration. ICLR.
    [5] Bellemare et al. (2017). A Distributional Perspective on RL. ICML.
    [6] Hessel et al. (2018). Rainbow: Combining Improvements. AAAI.
    """

    VANILLA = "vanilla"
    """Original DQN with uniform replay and ε-greedy exploration."""

    DOUBLE = "double"
    """Double DQN: decoupled action selection and evaluation."""

    DUELING = "dueling"
    """Dueling DQN: value-advantage decomposition architecture."""

    NOISY = "noisy"
    """Noisy DQN: parametric exploration via noisy linear layers."""

    CATEGORICAL = "categorical"
    """Categorical DQN (C51): distributional value estimation."""

    DOUBLE_DUELING = "double_dueling"
    """Combination of Double DQN and Dueling architecture."""

    RAINBOW = "rainbow"
    """Rainbow: Double + Dueling + Noisy + Categorical + PER + N-step."""

    def __str__(self) -> str:
        """Return human-readable variant name."""
        return self.value.replace("_", " ").title()

    @property
    def uses_noisy_exploration(self) -> bool:
        """Check if variant uses noisy networks for exploration."""
        return self in (DQNVariant.NOISY, DQNVariant.RAINBOW)

    @property
    def is_distributional(self) -> bool:
        """Check if variant uses distributional RL."""
        return self in (DQNVariant.CATEGORICAL, DQNVariant.RAINBOW)

    @property
    def uses_double(self) -> bool:
        """Check if variant uses Double DQN."""
        return self in (
            DQNVariant.DOUBLE,
            DQNVariant.DOUBLE_DUELING,
            DQNVariant.RAINBOW,
        )

    @property
    def uses_dueling(self) -> bool:
        """Check if variant uses Dueling architecture."""
        return self in (
            DQNVariant.DUELING,
            DQNVariant.DOUBLE_DUELING,
            DQNVariant.RAINBOW,
        )
