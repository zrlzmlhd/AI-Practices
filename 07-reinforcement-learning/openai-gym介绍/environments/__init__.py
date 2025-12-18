#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经典控制环境模块

本模块提供 Gymnasium 经典控制环境的详细描述和策略实现:
- 环境描述 (descriptions.py)
- 策略实现 (policies.py)
- 可视化工具 (visualization.py)
"""

from environments.descriptions import (
    EnvironmentDescription,
    CARTPOLE_DESCRIPTION,
    MOUNTAIN_CAR_DESCRIPTION,
    ACROBOT_DESCRIPTION,
    PENDULUM_DESCRIPTION,
    get_description,
    print_description,
)

from environments.policies import (
    BasePolicy,
    CartPolePolicy,
    MountainCarPolicy,
    PendulumPolicy,
    AcrobotPolicy,
)

__all__ = [
    # descriptions
    'EnvironmentDescription',
    'CARTPOLE_DESCRIPTION',
    'MOUNTAIN_CAR_DESCRIPTION',
    'ACROBOT_DESCRIPTION',
    'PENDULUM_DESCRIPTION',
    'get_description',
    'print_description',
    # policies
    'BasePolicy',
    'CartPolePolicy',
    'MountainCarPolicy',
    'PendulumPolicy',
    'AcrobotPolicy',
]
