#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gymnasium 环境包装器模块

本模块提供环境预处理和后处理的包装器工具:
- 统计工具 (statistics.py)
- 观测包装器 (observation.py)
- 动作包装器 (action.py)
- 奖励包装器 (reward.py)
- 工厂函数 (factory.py)
"""

from wrappers.statistics import (
    RunningStatistics,
    ExponentialMovingAverage,
)

from wrappers.observation import (
    NormalizeObservationWrapper,
    FrameStackWrapper,
    FlattenObservationWrapper,
)

from wrappers.action import (
    ClipActionWrapper,
    RescaleActionWrapper,
    StickyActionWrapper,
)

from wrappers.reward import (
    NormalizeRewardWrapper,
    ClipRewardWrapper,
    SignRewardWrapper,
)

from wrappers.factory import (
    make_wrapped_env,
    TimeLimitWrapper,
    EpisodeStatisticsWrapper,
    ActionRepeatWrapper,
)

__all__ = [
    # statistics
    'RunningStatistics',
    'ExponentialMovingAverage',
    # observation
    'NormalizeObservationWrapper',
    'FrameStackWrapper',
    'FlattenObservationWrapper',
    # action
    'ClipActionWrapper',
    'RescaleActionWrapper',
    'StickyActionWrapper',
    # reward
    'NormalizeRewardWrapper',
    'ClipRewardWrapper',
    'SignRewardWrapper',
    # factory
    'make_wrapped_env',
    'TimeLimitWrapper',
    'EpisodeStatisticsWrapper',
    'ActionRepeatWrapper',
]
