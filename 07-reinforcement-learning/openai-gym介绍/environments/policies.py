#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经典控制环境策略实现

================================================================================
核心思想 (Core Idea)
================================================================================
策略 (Policy) 是强化学习的核心概念，定义了智能体在给定状态下如何选择动作。
本模块为经典控制环境实现了多种基线策略，从简单的随机策略到经典的控制理论方法。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
**确定性策略**:
    $$a = \pi(s)$$

**随机策略**:
    $$a \sim \pi(a|s)$$

**PID 控制**:
比例-积分-微分控制是经典控制理论的核心方法:
    $$u(t) = K_p e(t) + K_i \int_0^t e(\tau)d\tau + K_d \frac{de(t)}{dt}$$

其中 $e(t)$ 是误差信号，$K_p, K_i, K_d$ 是控制增益。

对于 CartPole，简化为 PD 控制（忽略积分项）:
    $$u = K_{p,\theta} \cdot \theta + K_{d,\theta} \cdot \dot{\theta} + K_{p,x} \cdot x + K_{d,x} \cdot \dot{x}$$

**能量控制**:
通过调节系统能量实现摆起控制:
    $$E = \frac{1}{2}ml^2\dot{\theta}^2 - mgl\cos\theta$$

控制律: 当 $E < E_{target}$ 时增加能量，当 $E > E_{target}$ 时减少能量。

================================================================================
算法总结 (Summary)
================================================================================
本模块提供:
- BasePolicy: 策略基类定义
- CartPolePolicy: 倒立摆控制策略集合
- MountainCarPolicy: 爬山车策略集合
- PendulumPolicy: 单摆连续控制策略
- AcrobotPolicy: 双摆控制策略
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

try:
    import gymnasium as gym
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    warnings.warn("gymnasium 未安装", ImportWarning)


class BasePolicy(ABC):
    """
    策略基类

    定义策略的标准接口，所有具体策略都应继承此类。

    Methods
    -------
    __call__(observation)
        根据观测选择动作（主要接口）
    description()
        返回策略的描述字符串
    reset()
        重置策略内部状态（如果有）
    """

    @abstractmethod
    def __call__(self, observation: np.ndarray) -> Union[int, np.ndarray]:
        """
        根据观测选择动作

        Parameters
        ----------
        observation : np.ndarray
            环境观测

        Returns
        -------
        int or np.ndarray
            选择的动作（离散或连续）
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        返回策略描述

        Returns
        -------
        str
            策略的文字描述
        """
        pass

    def reset(self) -> None:
        """重置策略内部状态"""
        pass


class CartPolePolicy(BasePolicy):
    """
    CartPole 环境策略集合

    实现多种控制策略，从简单的规则策略到经典的 PID 控制。

    Parameters
    ----------
    method : str
        策略方法，可选:
        - "random": 随机策略
        - "angle": 基于角度的简单策略
        - "pid": PID 控制策略
        - "linear": 线性策略

    Attributes
    ----------
    method : str
        当前使用的策略方法
    kp_theta : float
        角度比例增益
    kd_theta : float
        角度微分增益
    kp_x : float
        位置比例增益
    kd_x : float
        位置微分增益

    Example
    -------
    >>> policy = CartPolePolicy(method="pid")
    >>> action = policy(observation)
    """

    def __init__(self, method: str = "angle"):
        """
        初始化策略

        Parameters
        ----------
        method : str
            策略方法: "random", "angle", "pid", "linear"
        """
        self.method = method

        # PID 控制器参数（经过调优）
        self.kp_theta = 50.0   # 角度比例增益
        self.kd_theta = 10.0   # 角度微分增益
        self.kp_x = 0.5        # 位置比例增益
        self.kd_x = 1.0        # 位置微分增益

        # 线性策略权重（可通过简单优化获得）
        self.linear_weights = np.array([0.1, 0.5, 1.0, 0.5])

    def __call__(self, obs: np.ndarray) -> int:
        """选择动作"""
        if self.method == "random":
            return np.random.randint(2)

        elif self.method == "angle":
            # 简单角度策略：杆往哪边倒就往哪边推
            pole_angle = obs[2]
            return 1 if pole_angle > 0 else 0

        elif self.method == "pid":
            # PD 控制（无积分项）
            x, x_dot, theta, theta_dot = obs

            # 角度 PD 控制
            u_theta = self.kp_theta * theta + self.kd_theta * theta_dot
            # 位置 PD 控制（辅助，防止小车跑出边界）
            u_x = self.kp_x * x + self.kd_x * x_dot

            # 综合控制信号
            u = u_theta + u_x
            return 1 if u > 0 else 0

        elif self.method == "linear":
            # 线性策略：状态的加权和
            u = np.dot(self.linear_weights, obs)
            return 1 if u > 0 else 0

        else:
            raise ValueError(f"未知策略方法: {self.method}")

    def description(self) -> str:
        descriptions = {
            "random": "随机策略: 等概率选择左/右推力",
            "angle": "角度策略: 根据摆杆倾斜方向推动",
            "pid": f"PID 控制: Kp_θ={self.kp_theta}, Kd_θ={self.kd_theta}",
            "linear": f"线性策略: w={self.linear_weights}"
        }
        return descriptions.get(self.method, "未知策略")


class MountainCarPolicy(BasePolicy):
    """
    MountainCar 环境策略

    实现基于物理直觉的动量策略和能量策略。

    Parameters
    ----------
    method : str
        策略方法: "random", "momentum", "energy"

    Notes
    -----
    MountainCar 是一个稀疏奖励环境，随机策略几乎无法成功。
    有效的策略需要利用动量或能量守恒的物理规律。
    """

    def __init__(self, method: str = "momentum"):
        self.method = method

    def __call__(self, obs: np.ndarray) -> int:
        """选择动作 (0=左, 1=不动, 2=右)"""
        position, velocity = obs

        if self.method == "random":
            return np.random.randint(3)

        elif self.method == "momentum":
            # 动量策略：跟随当前速度方向
            # 物理直觉：想要爬上山顶，需要在下坡时蓄力
            if velocity > 0:
                return 2  # 向右加速
            else:
                return 0  # 向左加速

        elif self.method == "energy":
            # 能量策略：考虑位置和速度的综合效果
            # 在左侧山坡蓄力，在右侧冲刺
            if position < -0.4:
                # 在低位置时，跟随速度摆动蓄能
                return 2 if velocity > 0 else 0
            elif position > 0.3:
                # 接近目标时，全力向右
                return 2
            else:
                # 中间区域，跟随速度
                return 2 if velocity > 0 else 0

        else:
            raise ValueError(f"未知策略: {self.method}")

    def description(self) -> str:
        descriptions = {
            "random": "随机策略: 等概率选择三个动作",
            "momentum": "动量策略: 跟随当前速度方向加速",
            "energy": "能量策略: 利用势能蓄力后冲刺"
        }
        return descriptions.get(self.method, "未知策略")


class PendulumPolicy(BasePolicy):
    """
    Pendulum 环境连续控制策略

    实现 PD 控制器和能量成形控制器。

    Parameters
    ----------
    method : str
        策略方法: "random", "pd", "energy"

    Attributes
    ----------
    kp : float
        比例增益
    kd : float
        微分增益

    Notes
    -----
    Pendulum 是连续动作空间环境，输出为 [-2, 2] 范围的扭矩。
    能量控制器结合能量泵浦（摆起）和 PD 稳定（保持直立）。
    """

    def __init__(self, method: str = "pd"):
        self.method = method
        self.kp = 10.0
        self.kd = 2.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """选择连续动作"""
        cos_theta, sin_theta, theta_dot = obs
        # 从三角函数恢复角度
        theta = np.arctan2(sin_theta, cos_theta)

        if self.method == "random":
            return np.array([np.random.uniform(-2, 2)])

        elif self.method == "pd":
            # PD 控制: u = -Kp*θ - Kd*θ̇
            # 目标：将摆杆稳定在直立位置 (θ=0)
            torque = -self.kp * theta - self.kd * theta_dot
            return np.clip([torque], -2.0, 2.0)

        elif self.method == "energy":
            # 能量成形控制
            # 思路：先通过能量泵浦将摆杆摆起，接近直立后切换到 PD 稳定

            # 物理参数
            g, l, m = 10.0, 1.0, 1.0

            # 当前系统能量 (动能 + 势能)
            kinetic = 0.5 * m * l**2 * theta_dot**2
            potential = -m * g * l * cos_theta  # 最低点为参考
            E = kinetic + potential

            # 目标能量 (直立位置)
            E_target = m * g * l

            # 控制策略
            if np.abs(theta) < 0.3:
                # 接近直立位置，使用 PD 控制稳定
                torque = -self.kp * theta - self.kd * theta_dot
            else:
                # 远离直立位置，使用能量泵浦
                # 控制律：u = -k * (E - E_target) * θ̇
                # 当能量不足时，与速度同向施力；能量过多时，与速度反向
                torque = -3.0 * (E - E_target) * theta_dot

            return np.clip([torque], -2.0, 2.0)

        else:
            raise ValueError(f"未知策略: {self.method}")

    def description(self) -> str:
        descriptions = {
            "random": "随机策略: 均匀采样扭矩",
            "pd": f"PD 控制: Kp={self.kp}, Kd={self.kd}",
            "energy": "能量控制: 能量泵浦 + PD 稳定"
        }
        return descriptions.get(self.method, "未知策略")


class AcrobotPolicy(BasePolicy):
    """
    Acrobot 环境策略

    实现能量泵浦策略。

    Parameters
    ----------
    method : str
        策略方法: "random", "energy"

    Notes
    -----
    Acrobot 是欠驱动双摆，只能在第二关节施加扭矩。
    需要通过摆动积累能量，类似于秋千。
    """

    def __init__(self, method: str = "energy"):
        self.method = method

    def __call__(self, obs: np.ndarray) -> int:
        """选择动作 (0=-1, 1=0, 2=+1)"""
        # 状态: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ̇1, θ̇2]
        cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1_dot, theta2_dot = obs

        if self.method == "random":
            return np.random.randint(3)

        elif self.method == "energy":
            # 能量泵浦策略：在第二连杆与第一连杆同向摆动时施加扭矩
            # 简化版：根据角速度符号选择扭矩方向
            theta2 = np.arctan2(sin_theta2, cos_theta2)

            # 当第二连杆向某方向摆动时，顺着它的方向施力
            if theta2_dot > 0.5:
                return 2  # +1 扭矩
            elif theta2_dot < -0.5:
                return 0  # -1 扭矩
            else:
                # 速度较小时，根据位置决定
                if theta2 > 0:
                    return 0
                else:
                    return 2

        else:
            raise ValueError(f"未知策略: {self.method}")

    def description(self) -> str:
        descriptions = {
            "random": "随机策略",
            "energy": "能量泵浦: 顺着角速度方向施加扭矩"
        }
        return descriptions.get(self.method, "未知策略")


# =============================================================================
#                           工厂函数
# =============================================================================

def get_policy(env_id: str, method: str = "default") -> BasePolicy:
    """
    获取指定环境的策略

    Parameters
    ----------
    env_id : str
        环境 ID
    method : str
        策略方法，"default" 表示使用默认方法

    Returns
    -------
    BasePolicy
        策略对象
    """
    env_policies = {
        "CartPole-v1": ("angle", CartPolePolicy),
        "MountainCar-v0": ("momentum", MountainCarPolicy),
        "Pendulum-v1": ("pd", PendulumPolicy),
        "Acrobot-v1": ("energy", AcrobotPolicy),
    }

    if env_id not in env_policies:
        raise ValueError(f"未知环境: {env_id}")

    default_method, policy_class = env_policies[env_id]
    actual_method = default_method if method == "default" else method

    return policy_class(method=actual_method)


# =============================================================================
#                           单元测试
# =============================================================================

def _run_tests() -> bool:
    """运行单元测试"""
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    print("\n" + "=" * 60)
    print("策略模块单元测试")
    print("=" * 60)

    all_passed = True

    # 测试 1: CartPole 策略
    print("\n[测试 1] CartPole 策略...")
    try:
        env = gym.make("CartPole-v1")

        for method in ["random", "angle", "pid", "linear"]:
            policy = CartPolePolicy(method=method)
            obs, _ = env.reset()

            for _ in range(10):
                action = policy(obs)
                assert action in [0, 1], f"无效动作: {action}"
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            assert policy.description(), "描述不能为空"

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 2: MountainCar 策略
    print("\n[测试 2] MountainCar 策略...")
    try:
        env = gym.make("MountainCar-v0")

        for method in ["random", "momentum", "energy"]:
            policy = MountainCarPolicy(method=method)
            obs, _ = env.reset()

            for _ in range(10):
                action = policy(obs)
                assert action in [0, 1, 2], f"无效动作: {action}"
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 3: Pendulum 策略
    print("\n[测试 3] Pendulum 策略...")
    try:
        env = gym.make("Pendulum-v1")

        for method in ["random", "pd", "energy"]:
            policy = PendulumPolicy(method=method)
            obs, _ = env.reset()

            for _ in range(10):
                action = policy(obs)
                assert action.shape == (1,), f"动作形状错误: {action.shape}"
                assert -2 <= action[0] <= 2, f"动作超出范围: {action}"
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 4: Acrobot 策略
    print("\n[测试 4] Acrobot 策略...")
    try:
        env = gym.make("Acrobot-v1")

        for method in ["random", "energy"]:
            policy = AcrobotPolicy(method=method)
            obs, _ = env.reset()

            for _ in range(10):
                action = policy(obs)
                assert action in [0, 1, 2], f"无效动作: {action}"
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        env.close()
        print("  [通过]")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 5: 工厂函数
    print("\n[测试 5] get_policy 工厂函数...")
    try:
        policy = get_policy("CartPole-v1", method="default")
        assert isinstance(policy, CartPolePolicy)

        policy = get_policy("Pendulum-v1", method="energy")
        assert isinstance(policy, PendulumPolicy)
        assert policy.method == "energy"

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
