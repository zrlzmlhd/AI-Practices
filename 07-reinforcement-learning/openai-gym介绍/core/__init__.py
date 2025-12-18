#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gymnasium 核心接口模块

本模块提供 Gymnasium 环境交互的核心工具和抽象:
- 空间分析 (spaces.py)
- 环境规格 (env_spec.py)
- 回合执行 (episode.py)
- 策略评估 (evaluation.py)
"""

from core.spaces import (
    SpaceType,
    SpaceInfo,
    analyze_space,
    get_action_dim,
    get_obs_dim,
)

from core.env_spec import (
    EnvironmentSpec,
    get_env_spec,
)

from core.episode import (
    StepResult,
    EpisodeResult,
    run_episode,
)

from core.evaluation import (
    evaluate_policy,
    PolicyEvaluator,
)

__all__ = [
    # spaces
    'SpaceType',
    'SpaceInfo',
    'analyze_space',
    'get_action_dim',
    'get_obs_dim',
    # env_spec
    'EnvironmentSpec',
    'get_env_spec',
    # episode
    'StepResult',
    'EpisodeResult',
    'run_episode',
    # evaluation
    'evaluate_policy',
    'PolicyEvaluator',
]
