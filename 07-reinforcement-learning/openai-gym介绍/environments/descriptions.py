#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经典控制环境描述

================================================================================
核心思想 (Core Idea)
================================================================================
经典控制 (Classic Control) 环境是一组基于控制论经典问题设计的低维环境。
它们具有简单的状态空间和明确的物理意义，是验证强化学习算法的理想测试平台。
优势: 训练速度快、可解释性强，适合算法原型开发和教学演示。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
**CartPole (倒立摆)**:
小车运动方程:
    $$\ddot{x} = \frac{F + m_p l (\dot{\theta}^2 \sin\theta - \ddot{\theta}\cos\theta)}{m_c + m_p}$$

摆杆运动方程:
    $$\ddot{\theta} = \frac{g\sin\theta + \cos\theta \cdot \frac{-F - m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p}}{l(\frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p})}$$

**MountainCar**:
位置更新: $x_{t+1} = x_t + v_{t+1}$
速度更新: $v_{t+1} = v_t + 0.001 \cdot a - 0.0025 \cdot \cos(3x_t)$

**Pendulum**:
    $$\ddot{\theta} = -\frac{3g}{2l}\sin(\theta + \pi) + \frac{3}{ml^2}u$$

================================================================================
问题背景 (Problem Statement)
================================================================================
1. **CartPole**: 欠驱动系统控制 - 自由度 > 控制输入数
2. **MountainCar**: 稀疏奖励问题 - 随机策略几乎无法成功
3. **Acrobot**: 双摆控制 - 利用摆动产生角动量
4. **Pendulum**: 连续控制 - 学习精确的扭矩输出

================================================================================
算法对比 (Comparison)
================================================================================
| 环境           | 状态维度 | 动作空间 | 难度   | 适合算法     |
|----------------|----------|----------|--------|--------------|
| CartPole-v1    | 4        | 离散(2)  | 简单   | DQN, A2C     |
| MountainCar-v0 | 2        | 离散(3)  | 中等   | 需要探索机制 |
| Acrobot-v1     | 6        | 离散(3)  | 中等   | DQN, PPO     |
| Pendulum-v1    | 3        | 连续(1)  | 中等   | DDPG, SAC    |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EnvironmentDescription:
    """
    环境描述数据类

    封装 Gymnasium 环境的完整描述信息，包括物理参数、状态空间、动作空间等。

    Attributes
    ----------
    name : str
        环境名称（中文）
    env_id : str
        Gymnasium 环境 ID
    state_space : dict
        状态空间描述，键为变量名，值为描述字符串
    action_space : dict
        动作空间描述，键为动作标识，值为描述字符串
    reward_description : str
        奖励函数描述
    termination_conditions : list of str
        终止条件列表
    success_threshold : float
        成功阈值（平均奖励）
    physics : dict
        物理参数字典
    dynamics_equations : list of str, optional
        动力学方程列表（LaTeX 格式）

    Example
    -------
    >>> desc = CARTPOLE_DESCRIPTION
    >>> print(f"环境: {desc.name} ({desc.env_id})")
    环境: CartPole (倒立摆) (CartPole-v1)
    """
    name: str
    env_id: str
    state_space: Dict[str, str]
    action_space: Dict[str, str]
    reward_description: str
    termination_conditions: List[str]
    success_threshold: float
    physics: Dict[str, float] = field(default_factory=dict)
    dynamics_equations: List[str] = field(default_factory=list)


# =============================================================================
#                           环境描述定义
# =============================================================================

CARTPOLE_DESCRIPTION = EnvironmentDescription(
    name="CartPole (倒立摆)",
    env_id="CartPole-v1",
    state_space={
        "cart_position": "小车位置 x ∈ [-4.8, 4.8] m",
        "cart_velocity": "小车速度 ẋ ∈ (-∞, +∞) m/s",
        "pole_angle": "摆杆角度 θ ∈ [-0.418, 0.418] rad (约 ±24°)",
        "pole_angular_velocity": "摆杆角速度 θ̇ ∈ (-∞, +∞) rad/s"
    },
    action_space={
        "0": "向左施加 -10N 的力",
        "1": "向右施加 +10N 的力"
    },
    reward_description="每存活一步 +1，目标是尽可能长时间保持摆杆直立",
    termination_conditions=[
        "摆杆角度超过 ±12° (约 0.2095 rad)",
        "小车位置超过 ±2.4 m",
        "回合长度达到 500 步 (截断)"
    ],
    success_threshold=475.0,
    physics={
        "gravity": 9.8,           # m/s²
        "cart_mass": 1.0,         # kg
        "pole_mass": 0.1,         # kg
        "pole_length": 0.5,       # m (半长)
        "force_magnitude": 10.0,  # N
        "dt": 0.02                # s (时间步长)
    },
    dynamics_equations=[
        r"\ddot{\theta} = \frac{g\sin\theta + \cos\theta \cdot \frac{-F - m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p}}{l(\frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p})}",
        r"\ddot{x} = \frac{F + m_p l (\dot{\theta}^2 \sin\theta - \ddot{\theta}\cos\theta)}{m_c + m_p}"
    ]
)


MOUNTAIN_CAR_DESCRIPTION = EnvironmentDescription(
    name="MountainCar (爬山车)",
    env_id="MountainCar-v0",
    state_space={
        "position": "位置 x ∈ [-1.2, 0.6]",
        "velocity": "速度 v ∈ [-0.07, 0.07]"
    },
    action_space={
        "0": "向左加速 (a = -1)",
        "1": "不加速 (a = 0)",
        "2": "向右加速 (a = +1)"
    },
    reward_description="每步 -1，到达山顶 (x >= 0.5) 回合结束",
    termination_conditions=[
        "到达山顶 (position >= 0.5)",
        "回合长度达到 200 步 (截断)"
    ],
    success_threshold=-110.0,  # 平均 110 步内到达
    physics={
        "min_position": -1.2,
        "max_position": 0.6,
        "max_speed": 0.07,
        "goal_position": 0.5,
        "goal_velocity": 0.0,
        "force": 0.001,
        "gravity": 0.0025
    },
    dynamics_equations=[
        r"v_{t+1} = v_t + 0.001 \cdot a - 0.0025 \cdot \cos(3x_t)",
        r"x_{t+1} = x_t + v_{t+1}"
    ]
)


ACROBOT_DESCRIPTION = EnvironmentDescription(
    name="Acrobot (双摆)",
    env_id="Acrobot-v1",
    state_space={
        "cos_theta1": "关节1角度的余弦 cos(θ₁)",
        "sin_theta1": "关节1角度的正弦 sin(θ₁)",
        "cos_theta2": "关节2角度的余弦 cos(θ₂)",
        "sin_theta2": "关节2角度的正弦 sin(θ₂)",
        "theta1_dot": "关节1角速度 θ̇₁ ∈ [-12.57, 12.57]",
        "theta2_dot": "关节2角速度 θ̇₂ ∈ [-28.27, 28.27]"
    },
    action_space={
        "0": "在关节2施加 -1 扭矩",
        "1": "在关节2施加 0 扭矩",
        "2": "在关节2施加 +1 扭矩"
    },
    reward_description="每步 -1，末端超过目标线回合结束",
    termination_conditions=[
        "末端高度超过目标线 (第一连杆顶端 + 第二连杆长度)",
        "回合长度达到 500 步 (截断)"
    ],
    success_threshold=-100.0,
    physics={
        "link_length_1": 1.0,
        "link_length_2": 1.0,
        "link_mass_1": 1.0,
        "link_mass_2": 1.0,
        "link_com_pos_1": 0.5,
        "link_com_pos_2": 0.5,
        "link_moi": 1.0  # 转动惯量
    }
)


PENDULUM_DESCRIPTION = EnvironmentDescription(
    name="Pendulum (单摆)",
    env_id="Pendulum-v1",
    state_space={
        "cos_theta": "角度的余弦 cos(θ) ∈ [-1, 1]",
        "sin_theta": "角度的正弦 sin(θ) ∈ [-1, 1]",
        "theta_dot": "角速度 θ̇ ∈ [-8, 8] rad/s"
    },
    action_space={
        "torque": "施加的扭矩 u ∈ [-2, 2] N·m (连续)"
    },
    reward_description="r = -(θ² + 0.1·θ̇² + 0.001·u²)，θ ∈ [-π, π]",
    termination_conditions=[
        "回合长度达到 200 步 (无提前终止)"
    ],
    success_threshold=-200.0,
    physics={
        "max_speed": 8.0,
        "max_torque": 2.0,
        "dt": 0.05,
        "gravity": 10.0,
        "mass": 1.0,
        "length": 1.0
    },
    dynamics_equations=[
        r"\ddot{\theta} = -\frac{3g}{2l}\sin(\theta + \pi) + \frac{3}{ml^2}u"
    ]
)


# 环境描述字典
_ENV_DESCRIPTIONS = {
    "cartpole": CARTPOLE_DESCRIPTION,
    "CartPole-v1": CARTPOLE_DESCRIPTION,
    "mountaincar": MOUNTAIN_CAR_DESCRIPTION,
    "MountainCar-v0": MOUNTAIN_CAR_DESCRIPTION,
    "acrobot": ACROBOT_DESCRIPTION,
    "Acrobot-v1": ACROBOT_DESCRIPTION,
    "pendulum": PENDULUM_DESCRIPTION,
    "Pendulum-v1": PENDULUM_DESCRIPTION,
}


def get_description(env_name: str) -> Optional[EnvironmentDescription]:
    """
    获取环境描述

    Parameters
    ----------
    env_name : str
        环境名称或 ID

    Returns
    -------
    EnvironmentDescription or None
        环境描述对象，未找到则返回 None
    """
    return _ENV_DESCRIPTIONS.get(env_name.lower()) or _ENV_DESCRIPTIONS.get(env_name)


def print_description(desc: EnvironmentDescription) -> None:
    """
    打印格式化的环境描述

    Parameters
    ----------
    desc : EnvironmentDescription
        环境描述对象
    """
    print(f"\n{'=' * 70}")
    print(f"{desc.name} ({desc.env_id})")
    print('=' * 70)

    print("\n【状态空间】")
    for key, value in desc.state_space.items():
        print(f"  • {key}: {value}")

    print("\n【动作空间】")
    for key, value in desc.action_space.items():
        print(f"  • {key}: {value}")

    print(f"\n【奖励设计】")
    print(f"  {desc.reward_description}")

    print("\n【终止条件】")
    for cond in desc.termination_conditions:
        print(f"  • {cond}")

    print(f"\n【成功阈值】")
    print(f"  平均奖励 >= {desc.success_threshold}")

    if desc.physics:
        print("\n【物理参数】")
        for key, value in desc.physics.items():
            print(f"  • {key}: {value}")

    if desc.dynamics_equations:
        print("\n【动力学方程】")
        for eq in desc.dynamics_equations:
            print(f"  ${eq}$")


def list_all_descriptions() -> List[EnvironmentDescription]:
    """返回所有环境描述列表"""
    return [
        CARTPOLE_DESCRIPTION,
        MOUNTAIN_CAR_DESCRIPTION,
        ACROBOT_DESCRIPTION,
        PENDULUM_DESCRIPTION
    ]


# =============================================================================
#                           单元测试
# =============================================================================

def _run_tests() -> bool:
    """运行单元测试"""
    print("\n" + "=" * 60)
    print("环境描述模块单元测试")
    print("=" * 60)

    all_passed = True

    # 测试 1: 描述数据完整性
    print("\n[测试 1] 描述数据完整性...")
    try:
        for desc in list_all_descriptions():
            assert desc.name, "名称不能为空"
            assert desc.env_id, "环境ID不能为空"
            assert desc.state_space, "状态空间不能为空"
            assert desc.action_space, "动作空间不能为空"
            assert desc.reward_description, "奖励描述不能为空"
            assert desc.termination_conditions, "终止条件不能为空"
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 2: get_description 函数
    print("\n[测试 2] get_description 函数...")
    try:
        # 通过环境 ID
        desc = get_description("CartPole-v1")
        assert desc is not None
        assert desc.env_id == "CartPole-v1"

        # 通过简短名称
        desc = get_description("pendulum")
        assert desc is not None
        assert desc.env_id == "Pendulum-v1"

        # 不存在的环境
        desc = get_description("NotExist-v0")
        assert desc is None

        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 3: 物理参数
    print("\n[测试 3] 物理参数验证...")
    try:
        cartpole = CARTPOLE_DESCRIPTION
        assert cartpole.physics["gravity"] == 9.8
        assert cartpole.physics["cart_mass"] == 1.0
        assert cartpole.physics["pole_mass"] == 0.1

        pendulum = PENDULUM_DESCRIPTION
        assert pendulum.physics["max_torque"] == 2.0
        assert pendulum.physics["gravity"] == 10.0

        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 4: 打印描述
    print("\n[测试 4] print_description 函数...")
    try:
        print_description(CARTPOLE_DESCRIPTION)
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    print("\n" + "=" * 60)
    print("测试结果:", "全部通过" if all_passed else "部分失败")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _run_tests()
