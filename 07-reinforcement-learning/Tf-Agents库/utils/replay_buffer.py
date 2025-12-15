#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经验回放缓冲区工具

提供标准和优先级经验回放实现，支持高效的样本存储和采样。

================================================================================
核心思想 (Core Idea)
================================================================================
经验回放是 off-policy 强化学习的核心组件，通过存储和重采样历史经验，
打破样本的时间相关性，提高训练稳定性和样本效率。

优先级经验回放 (PER) 根据 TD 误差对样本赋予不同的采样概率，
使得高信息量的样本（TD 误差大）被更频繁地学习。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
均匀采样经验回放：
$$P(i) = \frac{1}{N}, \quad \forall i \in \mathcal{D}$$

优先级采样经验回放：
$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

其中：
- $p_i = |\delta_i| + \epsilon$: 优先级（基于 TD 误差）
- $\alpha$: 优先级指数（0=均匀，1=完全优先）
- $\epsilon$: 小常数防止零优先级

重要性采样权重（纠正优先级采样偏差）：
$$w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^\beta$$

归一化权重：
$$\tilde{w}_i = \frac{w_i}{\max_j w_j}$$

================================================================================
Reference:
    Schaul, T. et al. (2016). Prioritized Experience Replay. ICLR.
================================================================================
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, NamedTuple, Any
from dataclasses import dataclass, field
import heapq


class Experience(NamedTuple):
    """
    单步经验数据结构

    Attributes:
        state: 当前状态
        action: 执行的动作
        reward: 获得的奖励
        next_state: 下一状态
        done: 是否终止
    """
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class SumTree:
    """
    求和树数据结构

    用于高效实现优先级采样，支持 O(log N) 的采样和更新操作。

    核心思想 (Core Idea):
        使用完全二叉树存储优先级，叶节点存储样本优先级，
        内部节点存储子树优先级之和。采样时根据优先级总和
        进行区间查找。

    数学原理 (Mathematical Theory):
        树结构：
        - 叶节点数 = N（缓冲区容量）
        - 总节点数 = 2N - 1
        - 内部节点 i 的子节点: 2i+1, 2i+2

        采样算法：
        1. 生成随机数 s ∈ [0, total_priority)
        2. 从根节点开始，根据左右子树优先级分配进入对应分支
        3. 到达叶节点即为采样结果

    Attributes:
        capacity: 缓冲区容量（叶节点数）
        tree: 存储优先级的数组
        data: 存储经验数据的数组
        write_index: 当前写入位置
        num_entries: 当前存储的经验数量
    """

    capacity: int
    tree: np.ndarray = field(init=False)
    data: np.ndarray = field(init=False)
    write_index: int = field(default=0, init=False)
    num_entries: int = field(default=0, init=False)

    def __post_init__(self):
        """初始化树结构"""
        # 树大小 = 2 * capacity - 1
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.empty(self.capacity, dtype=object)

    def _propagate(self, index: int, change: float) -> None:
        """
        向上传播优先级变化

        Args:
            index: 叶节点索引
            change: 优先级变化量
        """
        parent = (index - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, index: int, value: float) -> int:
        """
        根据采样值检索叶节点

        Args:
            index: 当前节点索引
            value: 采样值

        Returns:
            叶节点索引
        """
        left = 2 * index + 1
        right = left + 1

        # 到达叶节点
        if left >= len(self.tree):
            return index

        # 根据优先级分配进入左或右子树
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    @property
    def total_priority(self) -> float:
        """获取总优先级（根节点值）"""
        return self.tree[0]

    def add(self, priority: float, data: Any) -> None:
        """
        添加新经验

        Args:
            priority: 经验优先级
            data: 经验数据
        """
        # 计算叶节点索引
        tree_index = self.write_index + self.capacity - 1

        # 存储数据
        self.data[self.write_index] = data

        # 更新优先级
        self.update(tree_index, priority)

        # 更新写入位置
        self.write_index = (self.write_index + 1) % self.capacity
        self.num_entries = min(self.num_entries + 1, self.capacity)

    def update(self, tree_index: int, priority: float) -> None:
        """
        更新指定位置的优先级

        Args:
            tree_index: 树节点索引
            priority: 新优先级
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    def get(self, value: float) -> Tuple[int, float, Any]:
        """
        根据采样值获取经验

        Args:
            value: 采样值 ∈ [0, total_priority)

        Returns:
            (树索引, 优先级, 数据) 元组
        """
        tree_index = self._retrieve(0, value)
        data_index = tree_index - self.capacity + 1

        return tree_index, self.tree[tree_index], self.data[data_index]


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区

    核心思想 (Core Idea):
        根据 TD 误差对经验进行优先级排序，高误差样本更频繁被采样。
        使用求和树实现 O(log N) 的高效采样。

    数学原理 (Mathematical Theory):
        优先级定义：
        $$p_i = |\delta_i| + \epsilon$$

        采样概率：
        $$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

        重要性采样权重：
        $$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

        权重在训练过程中逐渐从 β_init 增加到 1，
        确保收敛时的无偏估计。

    算法对比 (Comparison):
        vs 均匀采样：
        - 优点：重要样本学习更多，提高样本效率
        - 缺点：计算开销增加，需要调参

    Attributes:
        capacity: 缓冲区容量
        alpha: 优先级指数
        beta_init: IS 权重初始值
        beta_increment: IS 权重增量
        epsilon: 优先级下限

    Example:
        >>> buffer = PrioritizedReplayBuffer(capacity=10000)
        >>> buffer.add(state, action, reward, next_state, done)
        >>> batch, indices, weights = buffer.sample(batch_size=32)
        >>> buffer.update_priorities(indices, td_errors)
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_init: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """
        初始化优先级回放缓冲区

        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数（0=均匀，1=完全优先）
            beta_init: IS 权重初始 β 值
            beta_increment: β 每次采样的增量
            epsilon: 防止零优先级的小常数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_init
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        添加新经验

        新经验默认使用最大优先级，确保至少被采样一次。

        Args:
            state: 当前状态
            action: 执行动作
            reward: 获得奖励
            next_state: 下一状态
            done: 是否终止
        """
        experience = Experience(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(
        self,
        batch_size: int
    ) -> Tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        优先级采样

        核心思想:
            将总优先级分成 batch_size 个区间，每个区间采样一个样本，
            确保采样覆盖各优先级级别。

        Args:
            batch_size: 批大小

        Returns:
            (经验批次, 树索引, IS权重) 元组
        """
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float32)

        # 分层采样
        segment = self.tree.total_priority / batch_size

        # 更新 beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            # 在每个区间内随机采样
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)

            tree_idx, priority, experience = self.tree.get(value)

            indices[i] = tree_idx
            priorities[i] = priority

            batch_states.append(experience.state)
            batch_actions.append(experience.action)
            batch_rewards.append(experience.reward)
            batch_next_states.append(experience.next_state)
            batch_dones.append(experience.done)

        # 计算重要性采样权重
        num_samples = self.tree.num_entries
        probabilities = priorities / self.tree.total_priority
        weights = (num_samples * probabilities) ** (-self.beta)
        weights = weights / weights.max()  # 归一化

        batch = {
            'states': np.array(batch_states),
            'actions': np.array(batch_actions),
            'rewards': np.array(batch_rewards),
            'next_states': np.array(batch_next_states),
            'dones': np.array(batch_dones)
        }

        return batch, indices, weights.astype(np.float32)

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray
    ) -> None:
        """
        更新样本优先级

        Args:
            indices: 样本的树索引
            td_errors: TD 误差（用于计算新优先级）
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        """返回当前存储的经验数量"""
        return self.tree.num_entries


class ExperienceCollector:
    """
    经验收集器

    封装环境交互和经验存储逻辑，支持多种策略。

    核心思想 (Core Idea):
        将环境交互、经验存储和数据收集解耦，
        支持灵活的收集策略（固定步数、固定回合等）。

    Attributes:
        env: 交互环境
        buffer: 经验回放缓冲区
        policy: 动作选择策略
    """

    def __init__(
        self,
        env,
        buffer,
        policy
    ):
        """
        初始化收集器

        Args:
            env: TF-Agents 环境
            buffer: 回放缓冲区
            policy: 策略对象
        """
        self.env = env
        self.buffer = buffer
        self.policy = policy

        self._last_observation = None

    def collect_steps(self, num_steps: int) -> dict:
        """
        收集固定步数的经验

        Args:
            num_steps: 收集步数

        Returns:
            收集统计信息
        """
        stats = {
            'total_reward': 0.0,
            'num_episodes': 0,
            'episode_lengths': []
        }

        if self._last_observation is None:
            time_step = self.env.reset()
            self._last_observation = time_step.observation
        else:
            time_step = self.env.current_time_step()

        current_episode_length = 0
        current_episode_reward = 0.0

        for _ in range(num_steps):
            # 选择动作
            action_step = self.policy.action(time_step)
            action = action_step.action

            # 执行动作
            next_time_step = self.env.step(action)

            # 存储经验（需要转换为 numpy）
            state = time_step.observation.numpy() if hasattr(time_step.observation, 'numpy') else time_step.observation
            next_state = next_time_step.observation.numpy() if hasattr(next_time_step.observation, 'numpy') else next_time_step.observation
            action_np = action.numpy() if hasattr(action, 'numpy') else action
            reward = float(next_time_step.reward.numpy()[0] if hasattr(next_time_step.reward, 'numpy') else next_time_step.reward)
            done = next_time_step.is_last()

            self.buffer.add(state[0], action_np[0], reward, next_state[0], done)

            current_episode_length += 1
            current_episode_reward += reward

            if done:
                stats['total_reward'] += current_episode_reward
                stats['num_episodes'] += 1
                stats['episode_lengths'].append(current_episode_length)

                time_step = self.env.reset()
                current_episode_length = 0
                current_episode_reward = 0.0
            else:
                time_step = next_time_step

            self._last_observation = time_step.observation

        return stats

    def collect_episodes(self, num_episodes: int) -> dict:
        """
        收集固定回合数的经验

        Args:
            num_episodes: 收集回合数

        Returns:
            收集统计信息
        """
        stats = {
            'total_reward': 0.0,
            'num_episodes': 0,
            'episode_lengths': [],
            'episode_rewards': []
        }

        for _ in range(num_episodes):
            time_step = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            while not time_step.is_last():
                action_step = self.policy.action(time_step)
                action = action_step.action

                next_time_step = self.env.step(action)

                state = time_step.observation.numpy()[0]
                next_state = next_time_step.observation.numpy()[0]
                action_np = action.numpy()[0]
                reward = float(next_time_step.reward.numpy()[0])
                done = next_time_step.is_last()

                self.buffer.add(state, action_np, reward, next_state, done)

                episode_reward += reward
                episode_length += 1
                time_step = next_time_step

            stats['total_reward'] += episode_reward
            stats['num_episodes'] += 1
            stats['episode_lengths'].append(episode_length)
            stats['episode_rewards'].append(episode_reward)

        return stats


if __name__ == "__main__":
    # 测试优先级回放缓冲区
    print("=" * 60)
    print("优先级经验回放缓冲区测试")
    print("=" * 60)

    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta_init=0.4)

    # 添加测试数据
    for i in range(50):
        state = np.random.randn(4).astype(np.float32)
        action = np.array([np.random.randint(2)])
        reward = np.random.randn()
        next_state = np.random.randn(4).astype(np.float32)
        done = np.random.random() < 0.1

        buffer.add(state, action, reward, next_state, done)

    print(f"缓冲区大小: {len(buffer)}")

    # 采样测试
    batch, indices, weights = buffer.sample(batch_size=8)
    print(f"采样批次形状: states={batch['states'].shape}")
    print(f"IS 权重: {weights}")

    # 更新优先级
    td_errors = np.random.randn(8)
    buffer.update_priorities(indices, td_errors)
    print("优先级已更新")

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
