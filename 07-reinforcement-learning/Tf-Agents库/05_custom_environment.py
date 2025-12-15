#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF-Agents 自定义环境开发指南

本模块演示如何使用 TF-Agents 创建自定义强化学习环境，
包括环境规范定义、状态转移逻辑、奖励设计等核心概念。

================================================================================
核心思想 (Core Idea)
================================================================================
TF-Agents 环境继承自 PyEnvironment 抽象类，需要实现以下核心方法：
1. observation_spec(): 定义观测空间
2. action_spec(): 定义动作空间
3. _reset(): 重置环境
4. _step(): 执行动作并返回下一状态

自定义环境使得算法可以应用于任何领域问题，是强化学习工程化的关键技能。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
环境定义了 MDP 的核心组件：

状态空间 $\mathcal{S}$:
- 由 observation_spec() 定义
- 可以是离散或连续空间
- TF-Agents 使用 ArraySpec/BoundedArraySpec 描述

动作空间 $\mathcal{A}$:
- 由 action_spec() 定义
- 离散: BoundedArraySpec(dtype=int32, minimum=0, maximum=n-1)
- 连续: BoundedArraySpec(dtype=float32, minimum=-1, maximum=1)

转移函数 $P(s'|s,a)$:
- 在 _step() 方法中实现
- 可以是确定性或随机性

奖励函数 $R(s,a,s')$:
- 在 _step() 方法中计算
- 设计良好的奖励函数是训练成功的关键

================================================================================
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Any
from abc import abstractmethod

# TF-Agents imports
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils as env_utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class GridWorldEnvironment(py_environment.PyEnvironment):
    """
    网格世界环境

    一个经典的强化学习测试环境，智能体在二维网格中移动，
    目标是到达指定位置同时避开障碍物。

    核心思想 (Core Idea):
        网格世界是最简单的空间导航任务，适合理解 MDP 各组件的作用。
        状态为智能体位置，动作为四个方向移动。

    数学原理 (Mathematical Theory):
        状态空间: $\mathcal{S} = \{(x, y) | 0 \leq x < W, 0 \leq y < H\}$
        动作空间: $\mathcal{A} = \{上, 下, 左, 右\} = \{0, 1, 2, 3\}$
        转移函数: 确定性，边界处保持不动
        奖励函数:
        $$R(s, a, s') = \begin{cases}
            +10 & \text{if } s' = \text{goal} \\
            -10 & \text{if } s' = \text{obstacle} \\
            -0.1 & \text{otherwise (step cost)}
        \end{cases}$$

    Attributes:
        _width: 网格宽度
        _height: 网格高度
        _goal_position: 目标位置
        _obstacles: 障碍物位置集合
        _agent_position: 智能体当前位置
        _episode_ended: 回合是否结束
        _max_steps: 最大步数限制
        _current_step: 当前步数
    """

    # 动作常量
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        goal_position: Tuple[int, int] = (4, 4),
        obstacles: Optional[list[Tuple[int, int]]] = None,
        start_position: Tuple[int, int] = (0, 0),
        max_steps: int = 100
    ):
        """
        初始化网格世界环境

        Args:
            width: 网格宽度
            height: 网格高度
            goal_position: 目标位置 (x, y)
            obstacles: 障碍物位置列表
            start_position: 起始位置
            max_steps: 每回合最大步数
        """
        super().__init__()

        self._width = width
        self._height = height
        self._goal_position = np.array(goal_position, dtype=np.int32)
        self._obstacles = set(obstacles) if obstacles else set()
        self._start_position = np.array(start_position, dtype=np.int32)
        self._max_steps = max_steps

        # 状态变量
        self._agent_position = self._start_position.copy()
        self._episode_ended = False
        self._current_step = 0

        # 动作映射
        self._action_to_direction = {
            self.ACTION_UP: np.array([0, 1]),
            self.ACTION_DOWN: np.array([0, -1]),
            self.ACTION_LEFT: np.array([-1, 0]),
            self.ACTION_RIGHT: np.array([1, 0])
        }

    def observation_spec(self) -> array_spec.BoundedArraySpec:
        """
        定义观测空间规范

        观测为智能体的 (x, y) 坐标，归一化到 [0, 1] 范围。

        Returns:
            观测空间规范
        """
        return array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
            name='observation'
        )

    def action_spec(self) -> array_spec.BoundedArraySpec:
        """
        定义动作空间规范

        离散动作空间：0=上, 1=下, 2=左, 3=右

        Returns:
            动作空间规范
        """
        return array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=3,
            name='action'
        )

    def _get_observation(self) -> np.ndarray:
        """
        获取当前观测（归一化坐标）

        Returns:
            归一化的 (x, y) 坐标
        """
        return np.array([
            self._agent_position[0] / (self._width - 1),
            self._agent_position[1] / (self._height - 1)
        ], dtype=np.float32)

    def _reset(self) -> ts.TimeStep:
        """
        重置环境到初始状态

        Returns:
            初始 TimeStep
        """
        self._agent_position = self._start_position.copy()
        self._episode_ended = False
        self._current_step = 0

        return ts.restart(self._get_observation())

    def _step(self, action: np.ndarray) -> ts.TimeStep:
        """
        执行动作并返回下一个 TimeStep

        核心思想 (Core Idea):
            实现状态转移和奖励计算逻辑。
            处理边界碰撞、障碍物碰撞、目标到达等情况。

        Args:
            action: 动作（0-3）

        Returns:
            下一个 TimeStep
        """
        if self._episode_ended:
            return self.reset()

        self._current_step += 1

        # 计算新位置
        direction = self._action_to_direction[int(action)]
        new_position = self._agent_position + direction

        # 边界检查
        new_position[0] = np.clip(new_position[0], 0, self._width - 1)
        new_position[1] = np.clip(new_position[1], 0, self._height - 1)

        # 障碍物检查
        if tuple(new_position) in self._obstacles:
            # 撞到障碍物，回合结束
            self._episode_ended = True
            return ts.termination(self._get_observation(), reward=-10.0)

        # 更新位置
        self._agent_position = new_position

        # 目标检查
        if np.array_equal(self._agent_position, self._goal_position):
            self._episode_ended = True
            return ts.termination(self._get_observation(), reward=10.0)

        # 超时检查
        if self._current_step >= self._max_steps:
            self._episode_ended = True
            return ts.termination(self._get_observation(), reward=-1.0)

        # 正常转移（带步长惩罚鼓励快速到达）
        return ts.transition(self._get_observation(), reward=-0.1, discount=1.0)

    def render(self, mode: str = 'human') -> Optional[str]:
        """
        渲染环境状态

        Args:
            mode: 渲染模式 ('human' 或 'ansi')

        Returns:
            如果 mode='ansi'，返回字符串表示
        """
        grid = [['.' for _ in range(self._width)] for _ in range(self._height)]

        # 标记障碍物
        for obs in self._obstacles:
            grid[self._height - 1 - obs[1]][obs[0]] = 'X'

        # 标记目标
        goal = self._goal_position
        grid[self._height - 1 - goal[1]][goal[0]] = 'G'

        # 标记智能体
        agent = self._agent_position
        grid[self._height - 1 - agent[1]][agent[0]] = 'A'

        # 构建字符串
        result = '\n'.join([''.join(row) for row in grid])

        if mode == 'human':
            print(result)
            return None
        else:
            return result


class BanditEnvironment(py_environment.PyEnvironment):
    """
    多臂老虎机环境

    经典的探索-利用权衡问题，智能体需要学习选择最优臂。

    核心思想 (Core Idea):
        多臂老虎机是最简单的强化学习问题，没有状态转移。
        每次选择一个臂，获得随机奖励，目标是最大化累积奖励。

    数学原理 (Mathematical Theory):
        问题定义:
        - K 个臂，每个臂有未知的奖励分布
        - 动作 $a \in \{0, 1, ..., K-1\}$
        - 奖励 $r \sim P_a$（依赖选择的臂）

        目标是最小化遗憾 (Regret)：
        $$\text{Regret}(T) = T \cdot \mu^* - \sum_{t=1}^{T} \mu_{a_t}$$

        其中 $\mu^* = \max_a \mu_a$ 是最优臂的期望奖励。

    Attributes:
        _num_arms: 臂的数量
        _arm_means: 每个臂的真实期望奖励
        _arm_stds: 每个臂的奖励标准差
    """

    def __init__(
        self,
        num_arms: int = 10,
        arm_means: Optional[np.ndarray] = None,
        arm_stds: Optional[np.ndarray] = None
    ):
        """
        初始化老虎机环境

        Args:
            num_arms: 臂的数量
            arm_means: 每个臂的期望奖励（None 则随机生成）
            arm_stds: 每个臂的奖励标准差（None 则统一为 1.0）
        """
        super().__init__()

        self._num_arms = num_arms

        if arm_means is None:
            # 随机生成期望奖励
            self._arm_means = np.random.randn(num_arms).astype(np.float32)
        else:
            self._arm_means = np.array(arm_means, dtype=np.float32)

        if arm_stds is None:
            self._arm_stds = np.ones(num_arms, dtype=np.float32)
        else:
            self._arm_stds = np.array(arm_stds, dtype=np.float32)

        # 最优臂
        self._optimal_arm = np.argmax(self._arm_means)
        self._optimal_mean = self._arm_means[self._optimal_arm]

    def observation_spec(self) -> array_spec.ArraySpec:
        """
        定义观测空间

        老虎机问题没有真正的状态，返回常量观测。

        Returns:
            观测空间规范
        """
        return array_spec.ArraySpec(
            shape=(1,),
            dtype=np.float32,
            name='observation'
        )

    def action_spec(self) -> array_spec.BoundedArraySpec:
        """
        定义动作空间

        离散动作，对应选择哪个臂。

        Returns:
            动作空间规范
        """
        return array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=self._num_arms - 1,
            name='action'
        )

    def _reset(self) -> ts.TimeStep:
        """
        重置环境

        Returns:
            初始 TimeStep
        """
        return ts.restart(np.array([0.0], dtype=np.float32))

    def _step(self, action: np.ndarray) -> ts.TimeStep:
        """
        执行动作（拉动臂）

        Args:
            action: 选择的臂索引

        Returns:
            包含奖励的 TimeStep（非终止）
        """
        arm = int(action)

        # 从对应臂的分布采样奖励
        reward = np.random.normal(
            self._arm_means[arm],
            self._arm_stds[arm]
        )

        # 老虎机问题永不终止（或每步都是新回合）
        return ts.transition(
            np.array([0.0], dtype=np.float32),
            reward=float(reward),
            discount=1.0
        )

    def get_optimal_arm(self) -> int:
        """获取最优臂索引"""
        return self._optimal_arm

    def get_arm_means(self) -> np.ndarray:
        """获取所有臂的期望奖励"""
        return self._arm_means.copy()


class ContinuousControlEnvironment(py_environment.PyEnvironment):
    """
    简单的连续控制环境

    演示如何定义连续动作空间环境。
    任务是控制一个点沿直线移动到目标位置。

    核心思想 (Core Idea):
        连续控制问题的动作是实数值，需要使用连续策略（如 Gaussian Policy）。
        本环境演示基本的连续控制任务定义。

    数学原理 (Mathematical Theory):
        状态: $s = [x, \dot{x}, x_{target}]$ (位置、速度、目标)
        动作: $a \in [-1, 1]$ (力/加速度)
        动力学: $\ddot{x} = a$
        奖励: $r = -|x - x_{target}|$ (距离惩罚)

    Attributes:
        _position: 当前位置
        _velocity: 当前速度
        _target: 目标位置
        _dt: 时间步长
        _max_steps: 最大步数
    """

    def __init__(
        self,
        target_range: Tuple[float, float] = (-1.0, 1.0),
        dt: float = 0.1,
        max_steps: int = 200
    ):
        """
        初始化连续控制环境

        Args:
            target_range: 目标位置范围
            dt: 时间步长
            max_steps: 最大步数
        """
        super().__init__()

        self._target_range = target_range
        self._dt = dt
        self._max_steps = max_steps

        # 状态变量
        self._position = 0.0
        self._velocity = 0.0
        self._target = 0.0
        self._current_step = 0
        self._episode_ended = False

    def observation_spec(self) -> array_spec.BoundedArraySpec:
        """
        定义观测空间

        观测包含：位置、速度、目标位置（相对）

        Returns:
            观测空间规范
        """
        return array_spec.BoundedArraySpec(
            shape=(3,),
            dtype=np.float32,
            minimum=np.array([-2.0, -2.0, -2.0]),
            maximum=np.array([2.0, 2.0, 2.0]),
            name='observation'
        )

    def action_spec(self) -> array_spec.BoundedArraySpec:
        """
        定义动作空间

        连续动作：控制力在 [-1, 1] 范围

        Returns:
            动作空间规范
        """
        return array_spec.BoundedArraySpec(
            shape=(1,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='action'
        )

    def _get_observation(self) -> np.ndarray:
        """获取当前观测"""
        return np.array([
            self._position,
            self._velocity,
            self._target - self._position  # 相对目标
        ], dtype=np.float32)

    def _reset(self) -> ts.TimeStep:
        """重置环境"""
        self._position = 0.0
        self._velocity = 0.0
        self._target = np.random.uniform(*self._target_range)
        self._current_step = 0
        self._episode_ended = False

        return ts.restart(self._get_observation())

    def _step(self, action: np.ndarray) -> ts.TimeStep:
        """
        执行动作

        使用简单的欧拉积分更新状态。

        Args:
            action: 控制力 [-1, 1]

        Returns:
            下一个 TimeStep
        """
        if self._episode_ended:
            return self.reset()

        self._current_step += 1

        # 应用动作（力/加速度）
        force = np.clip(float(action[0]), -1.0, 1.0)

        # 欧拉积分
        self._velocity += force * self._dt
        self._velocity = np.clip(self._velocity, -2.0, 2.0)  # 速度限制
        self._position += self._velocity * self._dt

        # 计算奖励（距离惩罚）
        distance = abs(self._position - self._target)
        reward = -distance

        # 到达目标判定
        if distance < 0.05:
            self._episode_ended = True
            return ts.termination(self._get_observation(), reward=10.0)

        # 超时或出界
        if self._current_step >= self._max_steps or abs(self._position) > 2.0:
            self._episode_ended = True
            return ts.termination(self._get_observation(), reward=-1.0)

        return ts.transition(self._get_observation(), reward=reward, discount=0.99)


def validate_environment(env: py_environment.PyEnvironment) -> bool:
    """
    验证自定义环境的正确性

    核心思想 (Core Idea):
        使用 TF-Agents 提供的验证工具检查环境实现是否符合接口规范。

    Args:
        env: 待验证的环境

    Returns:
        验证是否通过

    Raises:
        Exception: 如果环境验证失败
    """
    try:
        env_utils.validate_py_environment(env, episodes=3)
        print(f"环境 {env.__class__.__name__} 验证通过！")
        return True
    except Exception as e:
        print(f"环境验证失败: {e}")
        return False


def demonstrate_environments():
    """
    演示自定义环境的创建和使用
    """
    print("=" * 60)
    print("自定义环境演示")
    print("=" * 60)

    # 1. GridWorld 环境
    print("\n1. GridWorld 环境")
    grid_env = GridWorldEnvironment(
        width=5,
        height=5,
        goal_position=(4, 4),
        obstacles=[(2, 2), (2, 3), (3, 2)],
        start_position=(0, 0)
    )

    print(f"   观测空间: {grid_env.observation_spec()}")
    print(f"   动作空间: {grid_env.action_spec()}")

    # 验证环境
    validate_environment(grid_env)

    # 运行几步
    time_step = grid_env.reset()
    print(f"   初始观测: {time_step.observation}")
    grid_env.render()

    # 2. Bandit 环境
    print("\n2. 多臂老虎机环境")
    bandit_env = BanditEnvironment(num_arms=5)

    print(f"   臂数量: {bandit_env._num_arms}")
    print(f"   臂期望奖励: {bandit_env.get_arm_means()}")
    print(f"   最优臂: {bandit_env.get_optimal_arm()}")

    validate_environment(bandit_env)

    # 3. 连续控制环境
    print("\n3. 连续控制环境")
    continuous_env = ContinuousControlEnvironment()

    print(f"   观测空间: {continuous_env.observation_spec()}")
    print(f"   动作空间: {continuous_env.action_spec()}")

    validate_environment(continuous_env)

    # 4. 转换为 TensorFlow 环境
    print("\n4. TensorFlow 环境包装")
    tf_grid_env = tf_py_environment.TFPyEnvironment(grid_env)

    print(f"   TF 观测规范: {tf_grid_env.observation_spec()}")
    print(f"   批大小: {tf_grid_env.batch_size}")

    # 运行一步
    time_step = tf_grid_env.reset()
    print(f"   TF 观测: {time_step.observation}")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_environments()
