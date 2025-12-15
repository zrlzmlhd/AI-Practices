"""
Q-Learning 算法完整实现

============================================================
核心思想 (Core Idea)
============================================================
Q-Learning 是一种无模型 (model-free)、离策略 (off-policy) 的时序差分 (TD)
控制算法。其核心直觉是：通过与环境交互采样，直接学习最优动作价值函数 Q*，
无需知道环境的状态转移概率和奖励函数。

============================================================
数学原理 (Mathematical Theory)
============================================================
Q-Learning 基于贝尔曼最优方程的随机近似：

1. 贝尔曼最优方程 (Bellman Optimality Equation):

   Q*(s, a) = E[R_{t+1} + γ max_{a'} Q*(S_{t+1}, a') | S_t = s, A_t = a]

2. Q-Learning 更新规则 (TD Control):

   Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
                                  \_________________________/
                                            TD 目标
                                  \________________________________________________/
                                                   TD 误差 δ_t

   其中：
   - α ∈ (0, 1]: 学习率，控制更新步长
   - γ ∈ [0, 1]: 折扣因子，权衡即时与未来奖励
   - R_{t+1}: 执行动作后获得的即时奖励
   - max_a Q(S_{t+1}, a): 下一状态的最优动作价值（离策略的关键）

3. 收敛条件 (Convergence Conditions):
   在满足以下条件时，Q-Learning 以概率 1 收敛到 Q*：
   - 所有状态-动作对被无限次访问
   - 学习率满足 Robbins-Monro 条件：Σα_t = ∞, Σα_t² < ∞

============================================================
问题背景 (Problem Statement)
============================================================
传统动态规划方法求解 MDP 需要完整的环境模型 P(s'|s,a) 和 R(s,a,s')，
在现实问题中通常不可行：
- 状态转移概率未知或难以精确建模
- 状态空间可能极其庞大（如围棋约 10^{170} 种状态）
- 需要遍历所有状态和动作，计算复杂度高

Q-Learning 通过采样交互解决这些问题，是强化学习从理论走向实践的关键算法。

============================================================
算法对比 (Comparison)
============================================================
与其他 TD 控制算法的对比：

| 特性          | Q-Learning      | SARSA          | Expected SARSA |
|---------------|-----------------|----------------|----------------|
| 策略类型      | Off-Policy      | On-Policy      | On-Policy      |
| TD 目标       | max_a Q(S',a)   | Q(S',A')       | E[Q(S',A')]    |
| 学习目标      | 最优策略 Q*     | 当前策略 Q^π   | 当前策略 Q^π   |
| 方差          | 低              | 高             | 低             |
| 探索风险考虑  | 不考虑          | 考虑           | 考虑           |
| 样本效率      | 高（可重用）    | 较低           | 中等           |
| 收敛速度      | 快              | 较慢           | 中等           |

优点：
- 离策略：可从任意策略的经验中学习，包括历史数据、人类演示
- 直接学习最优策略，不受探索策略影响
- 实现简单，只需维护一个 Q 表

缺点：
- 可能过估计 Q 值（max 操作的正偏差）
- 在随机环境中可能不稳定
- 表格型方法无法处理连续/高维状态空间

============================================================
复杂度 (Complexity)
============================================================
表格型 Q-Learning：
- 空间复杂度：O(|S| × |A|)，存储 Q 表
- 时间复杂度：每步更新 O(|A|)，需计算 max

总训练复杂度取决于：
- 状态空间大小 |S|
- 动作空间大小 |A|
- 收敛所需回合数（与探索策略和学习率相关）

============================================================
算法总结 (Summary)
============================================================
Q-Learning 通过"尝试-反馈-更新"的循环学习最优策略：
1. 智能体在当前状态选择动作（探索或利用）
2. 环境返回奖励和下一状态
3. 使用 TD 误差更新 Q 值：如果实际比预期好（TD误差>0），增大 Q 值
4. 重复直到 Q 函数收敛

关键洞见：
- max 操作使 Q-Learning 学习最优策略，而非当前探索策略
- 这是"离策略"的本质：行为策略可以大胆探索，学习目标始终指向最优
- 代价是可能过度乐观估计，需要 Double Q-Learning 等方法缓解

============================================================
参考文献 (References)
============================================================
[1] Watkins, C.J.C.H. (1989). Learning from Delayed Rewards. PhD Thesis,
    Cambridge University.
[2] Watkins, C.J.C.H. & Dayan, P. (1992). Q-Learning. Machine Learning,
    8(3-4):279-292.
[3] Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An Introduction,
    2nd ed. MIT Press. Chapter 6.
[4] Van Hasselt, H. (2010). Double Q-learning. NeurIPS.

============================================================
依赖 (Dependencies)
============================================================
    pip install numpy matplotlib gymnasium

Author: Ziming Ding
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

# ============================================================
# 类型定义 (Type Definitions)
# ============================================================

State = TypeVar("State")
Action = TypeVar("Action", bound=int)


class Environment(Protocol):
    """环境协议，定义强化学习环境的标准接口"""

    def reset(self) -> Any:
        """重置环境，返回初始状态"""
        ...

    def step(self, action: int) -> Tuple[Any, float, bool]:
        """执行动作，返回 (next_state, reward, done)"""
        ...


class ExplorationStrategy(Enum):
    """
    探索策略枚举

    探索与利用的平衡是强化学习的核心挑战之一。
    不同的探索策略适用于不同的场景：

    - EPSILON_GREEDY: 简单高效，广泛适用
    - SOFTMAX: 根据 Q 值差异分配概率，更精细
    - UCB: 有理论保证，适合多臂老虎机问题
    """

    EPSILON_GREEDY = "epsilon_greedy"
    SOFTMAX = "softmax"
    UCB = "ucb"


# ============================================================
# 配置数据类 (Configuration Data Classes)
# ============================================================


@dataclass
class AgentConfig:
    """
    智能体配置参数

    封装 Q-Learning 智能体的所有超参数，便于实验管理和参数调优。

    Attributes:
        n_actions: 动作空间大小 |A|
        learning_rate: 学习率 α ∈ (0, 1]，控制 Q 值更新步长
            - 较大值 (0.5~1.0): 快速学习，但可能震荡
            - 较小值 (0.01~0.1): 稳定收敛，但学习较慢
        discount_factor: 折扣因子 γ ∈ [0, 1]，权衡即时与未来奖励
            - γ = 0: 完全贪婪，只看即时奖励
            - γ = 1: 无折扣，同等重视所有未来奖励
            - 通常取 0.9~0.99
        epsilon: 初始探索率 ε，epsilon-greedy 策略的随机探索概率
        epsilon_decay: 探索率衰减系数，每回合乘以此系数
        epsilon_min: 最小探索率下限，保证持续探索
        exploration: 探索策略类型
        temperature: Softmax 策略的温度参数 τ
            - τ→0: 趋向贪心选择
            - τ→∞: 趋向均匀随机
        ucb_c: UCB 策略的探索系数 c，控制探索倾向
    """

    n_actions: int
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    exploration: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    temperature: float = 1.0
    ucb_c: float = 2.0

    def __post_init__(self) -> None:
        """参数验证"""
        if not 0 < self.learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        if not 0 <= self.discount_factor <= 1:
            raise ValueError(f"discount_factor must be in [0, 1], got {self.discount_factor}")
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
        if not 0 <= self.epsilon_min <= self.epsilon:
            raise ValueError(f"epsilon_min must be in [0, epsilon], got {self.epsilon_min}")
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {self.n_actions}")


@dataclass
class TrainingMetrics:
    """
    训练指标记录

    记录训练过程中的关键指标，用于监控学习进度和分析算法性能。

    Attributes:
        episode_rewards: 每回合累计奖励，反映策略质量
        episode_lengths: 每回合步数，反映任务完成效率
        epsilon_history: 探索率变化历史，监控探索衰减
        td_errors: TD 误差历史，反映价值估计的准确性
    """

    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)
    td_errors: List[float] = field(default_factory=list)

    def get_moving_average(self, window: int = 100) -> np.ndarray:
        """
        计算奖励的移动平均

        移动平均可以平滑噪声，更清晰地展示学习趋势。

        Args:
            window: 滑动窗口大小

        Returns:
            平滑后的奖励序列
        """
        if len(self.episode_rewards) < window:
            return np.array(self.episode_rewards)
        return np.convolve(
            self.episode_rewards, np.ones(window) / window, mode="valid"
        )

    def get_statistics(self, last_n: int = 100) -> Dict[str, float]:
        """
        获取最近 N 回合的统计信息

        Args:
            last_n: 统计最近的回合数

        Returns:
            包含均值、标准差等统计量的字典
        """
        rewards = self.episode_rewards[-last_n:] if self.episode_rewards else []
        steps = self.episode_lengths[-last_n:] if self.episode_lengths else []

        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "mean_steps": float(np.mean(steps)) if steps else 0.0,
        }


# ============================================================
# 探索策略实现 (Exploration Strategies)
# ============================================================


class ExplorationMixin:
    """
    探索策略混入类

    提供多种动作选择方法，解决强化学习中的探索-利用困境 (Exploration-Exploitation Dilemma)。

    探索-利用困境：
    - 利用 (Exploitation): 选择当前已知最优的动作，最大化即时收益
    - 探索 (Exploration): 尝试新动作，可能发现更好的策略

    过度利用 → 陷入局部最优，错过全局最优解
    过度探索 → 无法收敛，浪费计算资源
    """

    def _epsilon_greedy(self, q_values: np.ndarray, epsilon: float) -> int:
        """
        ε-Greedy 策略

        最简单且广泛使用的探索策略。以概率 ε 随机选择动作（探索），
        以概率 1-ε 选择 Q 值最大的动作（利用）。

        数学形式：
            π(a|s) = { 1 - ε + ε/|A|,  if a = argmax_a Q(s,a)
                     { ε/|A|,           otherwise

        Args:
            q_values: 当前状态的 Q 值数组，shape = (n_actions,)
            epsilon: 探索概率 ε ∈ [0, 1]

        Returns:
            选择的动作索引

        Note:
            当存在多个最大 Q 值时，随机选择其中之一，打破平局。
        """
        if np.random.random() < epsilon:
            return np.random.randint(len(q_values))

        max_q = np.max(q_values)
        max_actions = np.where(np.isclose(q_values, max_q))[0]
        return np.random.choice(max_actions)

    def _softmax(self, q_values: np.ndarray, temperature: float) -> int:
        """
        Softmax (Boltzmann) 策略

        根据 Q 值的 softmax 分布选择动作，Q 值越大的动作被选择的概率越高。

        数学形式：
            π(a|s) = exp(Q(s,a)/τ) / Σ_{a'} exp(Q(s,a')/τ)

        其中 τ 是温度参数 (temperature)：
            - τ → 0: 概率分布趋向确定性，接近贪心选择
            - τ → ∞: 概率分布趋向均匀，接近随机选择
            - τ = 1: 标准 softmax 分布

        与 ε-greedy 的区别：
            - ε-greedy: 随机选择是均匀的，不考虑 Q 值差异
            - Softmax: 根据 Q 值差异分配概率，更"有目的"地探索

        Args:
            q_values: 当前状态的 Q 值数组
            temperature: 温度参数 τ > 0

        Returns:
            选择的动作索引
        """
        # 数值稳定性：减去最大值防止 exp 溢出
        q_scaled = (q_values - np.max(q_values)) / max(temperature, 1e-8)
        exp_q = np.exp(q_scaled)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)

    def _ucb(
        self,
        q_values: np.ndarray,
        action_counts: np.ndarray,
        total_count: int,
        c: float,
    ) -> int:
        """
        Upper Confidence Bound (UCB) 策略

        选择置信上界最大的动作，显式平衡利用（Q 值）和探索（不确定性）。

        数学形式：
            A_t = argmax_a [Q(s,a) + c √(ln(t) / N(s,a))]
                           \_____/   \________________/
                           利用项         探索奖励

        其中：
            - Q(s,a): 当前 Q 值估计（利用）
            - N(s,a): 动作 a 在状态 s 的选择次数
            - t: 总步数
            - c: 探索系数，控制探索程度

        UCB 的优点：
            - 有理论保证：遗憾界 (regret bound) 为 O(ln(T))
            - 自动平衡：随着动作被选择，其不确定性减小，探索奖励降低

        Args:
            q_values: 当前状态的 Q 值数组
            action_counts: 每个动作的选择次数
            total_count: 总选择次数
            c: 探索系数

        Returns:
            选择的动作索引
        """
        # 未访问的动作优先选择（无限大的不确定性）
        if np.any(action_counts == 0):
            return np.random.choice(np.where(action_counts == 0)[0])

        ucb_values = q_values + c * np.sqrt(
            np.log(total_count + 1) / (action_counts + 1e-8)
        )
        return int(np.argmax(ucb_values))


# ============================================================
# 基础智能体抽象类 (Base Agent Abstract Class)
# ============================================================


class BaseAgent(ABC, ExplorationMixin):
    """
    表格型强化学习智能体基类

    提供 Q 表管理、探索策略、模型持久化等通用功能。
    子类需实现 update 方法定义具体的学习规则。

    设计原则：
    - 单一职责：基类只处理通用功能，具体算法逻辑由子类实现
    - 开闭原则：对扩展开放（易于添加新算法），对修改封闭
    - 依赖倒置：依赖抽象（AgentConfig），不依赖具体实现
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        初始化智能体

        Args:
            config: 智能体配置参数对象
        """
        self.config = config
        self.n_actions = config.n_actions
        self.lr = config.learning_rate
        self.gamma = config.discount_factor
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.exploration = config.exploration
        self.temperature = config.temperature
        self.ucb_c = config.ucb_c

        # Q 表：状态 → 动作 Q 值数组
        # 使用 defaultdict 自动初始化未访问状态的 Q 值为 0
        self.q_table: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

        # UCB 策略需要的计数器
        self.action_counts: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )
        self.total_steps = 0

        # 训练指标记录器
        self.metrics = TrainingMetrics()

    def get_action(self, state: Any, training: bool = True) -> int:
        """
        根据当前策略选择动作

        Args:
            state: 当前环境状态（可哈希的任意类型）
            training: 是否为训练模式
                - True: 使用探索策略
                - False: 使用纯贪心策略（测试/部署时）

        Returns:
            选择的动作索引 ∈ [0, n_actions)
        """
        q_values = self.q_table[state]

        if not training:
            # 测试模式：纯贪心，选择 Q 值最大的动作
            max_q = np.max(q_values)
            max_actions = np.where(np.isclose(q_values, max_q))[0]
            return int(np.random.choice(max_actions))

        # 训练模式：根据配置选择探索策略
        if self.exploration == ExplorationStrategy.EPSILON_GREEDY:
            action = self._epsilon_greedy(q_values, self.epsilon)
        elif self.exploration == ExplorationStrategy.SOFTMAX:
            action = self._softmax(q_values, self.temperature)
        elif self.exploration == ExplorationStrategy.UCB:
            action = self._ucb(
                q_values, self.action_counts[state], self.total_steps, self.ucb_c
            )
        else:
            action = self._epsilon_greedy(q_values, self.epsilon)

        # 更新 UCB 计数器
        self.action_counts[state][action] += 1
        self.total_steps += 1

        return int(action)

    @abstractmethod
    def update(self, *args, **kwargs) -> float:
        """
        更新 Q 值，由子类实现具体的学习规则

        Returns:
            TD 误差值，用于监控学习进度
        """
        pass

    def decay_epsilon(self) -> None:
        """
        衰减探索率

        使用指数衰减：ε_{t+1} = max(ε_min, ε_t × decay)

        随着训练进行，逐渐从探索转向利用。
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_greedy_policy(self) -> Dict[Any, int]:
        """
        提取当前贪心策略

        Returns:
            状态到最优动作的映射字典 {state: argmax_a Q(s,a)}
        """
        return {state: int(np.argmax(q_values)) for state, q_values in self.q_table.items()}

    def get_value_function(self) -> Dict[Any, float]:
        """
        提取状态价值函数

        V(s) = max_a Q(s,a)，即在状态 s 下采取最优动作的期望回报。

        Returns:
            状态到价值的映射字典 {state: V(s)}
        """
        return {
            state: float(np.max(q_values))
            for state, q_values in self.q_table.items()
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """
        保存模型到文件

        支持 JSON（人类可读）和 Pickle（高效二进制）两种格式。

        Args:
            filepath: 保存路径，根据后缀自动选择格式
                - .json: JSON 格式
                - 其他: Pickle 格式
        """
        filepath = Path(filepath)

        data = {
            "config": {
                "n_actions": self.n_actions,
                "learning_rate": self.lr,
                "discount_factor": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
            },
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "total_steps": self.total_steps,
        }

        if filepath.suffix == ".json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

    def load(self, filepath: Union[str, Path]) -> None:
        """
        从文件加载模型

        Args:
            filepath: 模型文件路径
        """
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        # 恢复 Q 表
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for k, v in data["q_table"].items():
            try:
                key = eval(k)
            except (SyntaxError, NameError):
                key = k
            self.q_table[key] = np.array(v)

        self.total_steps = data.get("total_steps", 0)

    def reset(self) -> None:
        """重置智能体状态，用于新的训练轮次"""
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.action_counts = defaultdict(lambda: np.zeros(self.n_actions))
        self.total_steps = 0
        self.epsilon = self.config.epsilon
        self.metrics = TrainingMetrics()


# ============================================================
# Q-Learning 智能体 (Q-Learning Agent)
# ============================================================


class QLearningAgent(BaseAgent):
    """
    Q-Learning 智能体 (Off-Policy TD Control)

    ============================================================
    核心思想
    ============================================================
    Q-Learning 是一种离策略 (off-policy) 时序差分控制算法，
    直接学习最优动作价值函数 Q*，与实际采取的行为策略无关。

    ============================================================
    数学原理
    ============================================================
    更新公式：
        Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
                                      \____________________________________/
                                                   TD 误差 δ_t

    其中：
        - α: 学习率
        - γ: 折扣因子
        - R_{t+1}: 即时奖励
        - max_a Q(S_{t+1}, a): 下一状态的最优动作价值

    ============================================================
    算法对比
    ============================================================
    相比 SARSA (On-Policy):
        优点：
        - 直接学习最优策略，不受探索影响
        - 样本效率高，可重用历史数据
        - 收敛更快

        缺点：
        - 可能过估计 Q 值（max 操作的正偏差）
        - 在高风险环境中可能学到危险策略

    ============================================================
    复杂度
    ============================================================
    - 时间复杂度：每步 O(|A|)，计算 max
    - 空间复杂度：O(|S| × |A|)，存储 Q 表

    Example:
        >>> config = AgentConfig(n_actions=4)
        >>> agent = QLearningAgent(config)
        >>> action = agent.get_action(state)
        >>> td_error = agent.update(state, action, reward, next_state, done)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        double_q: bool = False,
    ) -> None:
        """
        初始化 Q-Learning 智能体

        支持两种初始化方式：
        1. 传入 AgentConfig 对象（推荐）
        2. 传入单独的参数（便于快速实验）

        Args:
            config: 配置对象，若提供则忽略其他参数
            n_actions: 动作数量
            learning_rate: 学习率 α
            discount_factor: 折扣因子 γ
            epsilon: 初始探索率
            epsilon_decay: 探索率衰减系数
            epsilon_min: 最小探索率
            double_q: 是否使用 Double Q-Learning（减少过估计）
        """
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
            )
        super().__init__(config)

        self.double_q = double_q
        if double_q:
            # Double Q-Learning 需要两个独立的 Q 表
            self.q_table2: Dict[Any, np.ndarray] = defaultdict(
                lambda: np.zeros(self.n_actions)
            )

    def get_action(self, state: Any, training: bool = True) -> int:
        """
        选择动作

        Double Q-Learning 使用两个 Q 表的和来选择动作，
        减少因过估计导致的次优动作选择。
        """
        if self.double_q:
            q_values = self.q_table[state] + self.q_table2[state]
        else:
            q_values = self.q_table[state]

        if not training:
            max_q = np.max(q_values)
            max_actions = np.where(np.isclose(q_values, max_q))[0]
            return int(np.random.choice(max_actions))

        if self.exploration == ExplorationStrategy.EPSILON_GREEDY:
            action = self._epsilon_greedy(q_values, self.epsilon)
        elif self.exploration == ExplorationStrategy.SOFTMAX:
            action = self._softmax(q_values, self.temperature)
        else:
            action = self._ucb(
                q_values, self.action_counts[state], self.total_steps, self.ucb_c
            )

        self.action_counts[state][action] += 1
        self.total_steps += 1
        return int(action)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """
        Q-Learning 更新规则

        使用 TD 目标 R + γ max_a Q(S', a) 更新 Q 值。

        更新过程：
        1. 计算当前 Q 值估计
        2. 计算 TD 目标（即时奖励 + 折扣后的未来最优价值）
        3. 计算 TD 误差（目标与估计的差）
        4. 沿 TD 误差方向更新 Q 值

        Args:
            state: 当前状态 S_t
            action: 执行的动作 A_t
            reward: 获得的奖励 R_{t+1}
            next_state: 转移到的下一状态 S_{t+1}
            done: 是否为终止状态

        Returns:
            TD 误差 δ_t = target - Q(S_t, A_t)
        """
        if self.double_q:
            return self._double_q_update(state, action, reward, next_state, done)

        current_q = self.q_table[state][action]

        # 计算 TD 目标
        if done:
            # 终止状态没有后续奖励
            target = reward
        else:
            # Q-Learning 核心：使用 max 选择下一状态的最优动作价值
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD 误差：反映预测与实际的差距
        td_error = target - current_q

        # 更新 Q 值：沿着减小 TD 误差的方向调整
        self.q_table[state][action] += self.lr * td_error

        return float(td_error)

    def _double_q_update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """
        Double Q-Learning 更新

        ============================================================
        问题背景
        ============================================================
        标准 Q-Learning 使用 max 操作选择和评估动作，导致过估计问题：
        - 假设 Q 值估计有噪声：Q̂(s,a) = Q*(s,a) + ε_a
        - E[max_a Q̂(s,a)] ≥ max_a E[Q̂(s,a)] = max_a Q*(s,a)
        - 这种正偏差会通过 bootstrapping 累积

        ============================================================
        解决方案
        ============================================================
        解耦动作选择和价值评估：
        - 维护两个独立的 Q 表 Q1 和 Q2
        - 以 0.5 概率选择更新 Q1 或 Q2
        - 更新 Q1 时：用 Q1 选择动作，用 Q2 评估价值
        - 更新 Q2 时：用 Q2 选择动作，用 Q1 评估价值

        数学形式：
            a* = argmax_a Q1(S', a)
            Q1(S, A) ← Q1(S, A) + α [R + γ Q2(S', a*) - Q1(S, A)]
        """
        if np.random.random() < 0.5:
            # 更新 Q1
            current_q = self.q_table[state][action]
            if done:
                target = reward
            else:
                # Q1 选择动作，Q2 评估价值
                best_action = int(np.argmax(self.q_table[next_state]))
                target = reward + self.gamma * self.q_table2[next_state][best_action]
            td_error = target - current_q
            self.q_table[state][action] += self.lr * td_error
        else:
            # 更新 Q2
            current_q = self.q_table2[state][action]
            if done:
                target = reward
            else:
                # Q2 选择动作，Q1 评估价值
                best_action = int(np.argmax(self.q_table2[next_state]))
                target = reward + self.gamma * self.q_table[next_state][best_action]
            td_error = target - current_q
            self.q_table2[state][action] += self.lr * td_error

        return float(td_error)


# ============================================================
# SARSA 智能体 (SARSA Agent)
# ============================================================


class SARSAAgent(BaseAgent):
    """
    SARSA 智能体 (On-Policy TD Control)

    ============================================================
    核心思想
    ============================================================
    SARSA (State-Action-Reward-State-Action) 是一种在策略 (on-policy)
    时序差分控制算法，学习当前行为策略的价值函数 Q^π。

    ============================================================
    数学原理
    ============================================================
    更新公式：
        Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

    与 Q-Learning 的关键区别：
        - 使用实际采取的下一动作 A_{t+1}，而非 max
        - 学习的是当前 ε-greedy 策略的价值，而非最优策略

    ============================================================
    算法对比
    ============================================================
    与 Q-Learning 的行为差异（以悬崖行走为例）：
        - Q-Learning：学习沿悬崖边的最短路径（因为更新不考虑探索）
        - SARSA：学习远离悬崖的安全路径（考虑探索时掉下悬崖的可能）

    适用场景：
        - 需要安全探索的环境（如机器人控制）
        - 探索失败代价高昂的任务

    ============================================================
    复杂度
    ============================================================
    与 Q-Learning 相同：
    - 时间复杂度：每步 O(1)
    - 空间复杂度：O(|S| × |A|)

    Example:
        >>> agent = SARSAAgent(config)
        >>> action = agent.get_action(state)
        >>> next_action = agent.get_action(next_state)
        >>> td_error = agent.update(state, action, reward, next_state, next_action, done)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        """初始化 SARSA 智能体"""
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
            )
        super().__init__(config)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        next_action: int,
        done: bool,
    ) -> float:
        """
        SARSA 更新规则

        使用 TD 目标 R + γ Q(S', A') 更新 Q 值。

        注意：与 Q-Learning 不同，SARSA 需要提供下一步实际采取的动作。
        这使得 SARSA 学习的是当前策略（包含探索）的价值函数。

        Args:
            state: 当前状态 S_t
            action: 执行的动作 A_t
            reward: 获得的奖励 R_{t+1}
            next_state: 下一状态 S_{t+1}
            next_action: 下一状态实际采取的动作 A_{t+1}
            done: 是否终止

        Returns:
            TD 误差
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            # SARSA 核心：使用实际的 next_action
            target = reward + self.gamma * self.q_table[next_state][next_action]

        td_error = target - current_q
        self.q_table[state][action] += self.lr * td_error

        return float(td_error)


# ============================================================
# Expected SARSA 智能体 (Expected SARSA Agent)
# ============================================================


class ExpectedSARSAAgent(BaseAgent):
    """
    Expected SARSA 智能体

    ============================================================
    核心思想
    ============================================================
    Expected SARSA 使用下一状态 Q 值的期望，而非采样值。
    它结合了 Q-Learning 的低方差和 SARSA 的在策略特性。

    ============================================================
    数学原理
    ============================================================
    更新公式：
        Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ E_π[Q(S_{t+1}, A_{t+1})] - Q(S_t, A_t)]

    其中期望在当前策略 π 下计算：
        E_π[Q(S', A')] = Σ_a π(a|S') Q(S', a)

    对于 ε-greedy 策略：
        E_π[Q(S', A')] = (1-ε) max_a Q(S', a) + ε/|A| Σ_a Q(S', a)

    ============================================================
    算法对比
    ============================================================
    | 特性 | SARSA | Expected SARSA | Q-Learning |
    |------|-------|----------------|------------|
    | 方差 | 高    | 低             | 低         |
    | 偏差 | 低    | 低             | 高         |
    | 策略 | On    | On             | Off        |

    Expected SARSA 通过使用期望消除了 SARSA 中的采样方差，
    同时保持了在策略学习的特性。
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
            )
        super().__init__(config)

    def _get_expected_q(self, state: Any) -> float:
        """
        计算 ε-greedy 策略下的期望 Q 值

        E_π[Q(s, A)] = Σ_a π(a|s) Q(s, a)

        对于 ε-greedy：
            π(a|s) = ε/|A|              (非最优动作)
            π(a*|s) = 1 - ε + ε/|A|     (最优动作 a* = argmax_a Q(s,a))
        """
        q_values = self.q_table[state]

        # ε-greedy 策略下的动作概率
        n_actions = len(q_values)
        probs = np.ones(n_actions) * self.epsilon / n_actions
        best_action = int(np.argmax(q_values))
        probs[best_action] += 1 - self.epsilon

        return float(np.dot(probs, q_values))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """Expected SARSA 更新"""
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self._get_expected_q(next_state)

        td_error = target - current_q
        self.q_table[state][action] += self.lr * td_error

        return float(td_error)


# ============================================================
# 悬崖行走环境 (Cliff Walking Environment)
# ============================================================


class CliffWalkingEnv:
    """
    悬崖行走环境 (Cliff Walking)

    ============================================================
    核心思想
    ============================================================
    经典的强化学习测试环境，用于演示 Q-Learning 和 SARSA 的行为差异。
    智能体需要从起点到达目标，同时避免掉入悬崖。

    ============================================================
    环境布局
    ============================================================
    4×12 网格世界：

        ┌─────────────────────────────────────────────┐
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 0
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 1
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 2
        │ S  C  C  C  C  C  C  C  C  C  C  G  │  row 3
        └─────────────────────────────────────────────┘
          0  1  2  3  4  5  6  7  8  9 10 11    columns

    符号说明：
        S: 起点 (Start) - 坐标 (3, 0)
        G: 目标 (Goal) - 坐标 (3, 11)
        C: 悬崖 (Cliff) - 坐标 (3, 1) ~ (3, 10)
        .: 普通格子

    ============================================================
    动作空间
    ============================================================
    4 种动作：
        0: 上 (↑) - 行坐标 -1
        1: 右 (→) - 列坐标 +1
        2: 下 (↓) - 行坐标 +1
        3: 左 (←) - 列坐标 -1

    边界处理：撞墙则留在原地

    ============================================================
    奖励设计
    ============================================================
    - 每步：-1（鼓励快速到达目标）
    - 掉入悬崖：-100，并重置到起点（惩罚危险行为）
    - 到达目标：0，回合结束

    ============================================================
    算法行为差异
    ============================================================
    这个环境经典地展示了 Q-Learning 和 SARSA 的区别：

    Q-Learning（离策略）：
        - 学习最短路径（沿悬崖边走）
        - 原因：更新时使用 max，假设未来总是采取最优动作
        - 结果：学到的策略最优，但训练过程中经常掉入悬崖

    SARSA（在策略）：
        - 学习安全路径（远离悬崖）
        - 原因：更新时考虑实际会采取的动作，包括探索
        - 结果：学到的策略更保守，但训练过程更稳定
    """

    # 动作映射：动作索引 → (行偏移, 列偏移)
    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),  # 上
        1: (0, 1),   # 右
        2: (1, 0),   # 下
        3: (0, -1),  # 左
    }
    ACTION_NAMES: List[str] = ["上", "右", "下", "左"]

    def __init__(self, height: int = 4, width: int = 12) -> None:
        """
        初始化环境

        Args:
            height: 网格高度，默认 4
            width: 网格宽度，默认 12
        """
        self.height = height
        self.width = width

        # 特殊位置定义
        self.start: Tuple[int, int] = (height - 1, 0)
        self.goal: Tuple[int, int] = (height - 1, width - 1)
        self.cliff: List[Tuple[int, int]] = [
            (height - 1, j) for j in range(1, width - 1)
        ]

        # 当前状态
        self.state: Tuple[int, int] = self.start

        # 环境属性
        self.n_states = height * width
        self.n_actions = 4

    def reset(self) -> Tuple[int, int]:
        """
        重置环境到初始状态

        Returns:
            起始状态坐标 (row, col)
        """
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行动作，返回转移结果

        实现确定性状态转移：给定状态和动作，下一状态是确定的。

        Args:
            action: 动作索引 (0-3)

        Returns:
            (next_state, reward, done) 三元组：
            - next_state: 下一状态坐标
            - reward: 获得的奖励
            - done: 是否为终止状态
        """
        # 计算下一位置（边界裁剪，撞墙则留在原地）
        di, dj = self.ACTIONS[action]
        new_i = int(np.clip(self.state[0] + di, 0, self.height - 1))
        new_j = int(np.clip(self.state[1] + dj, 0, self.width - 1))
        next_state = (new_i, new_j)

        # 检查是否掉入悬崖
        if next_state in self.cliff:
            self.state = self.start  # 重置到起点
            return self.state, -100.0, False

        self.state = next_state

        # 检查是否到达目标
        if self.state == self.goal:
            return self.state, 0.0, True

        return self.state, -1.0, False

    def render(self, path: Optional[List[Tuple[int, int]]] = None) -> str:
        """
        渲染环境状态

        Args:
            path: 可选的路径点列表，用于可视化策略轨迹

        Returns:
            环境的字符串表示
        """
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        # 标记悬崖
        for pos in self.cliff:
            grid[pos[0]][pos[1]] = "C"

        # 标记起点和终点
        grid[self.start[0]][self.start[1]] = "S"
        grid[self.goal[0]][self.goal[1]] = "G"

        # 标记路径
        if path:
            for pos in path[1:-1]:
                if pos not in self.cliff and pos != self.start and pos != self.goal:
                    grid[pos[0]][pos[1]] = "*"

        # 标记当前位置
        if self.state != self.start and self.state != self.goal:
            if self.state not in self.cliff:
                grid[self.state[0]][self.state[1]] = "@"

        # 构建输出字符串
        border = "┌" + "─" * (self.width * 2 + 1) + "┐"
        lines = [border]
        for row in grid:
            lines.append("│ " + " ".join(row) + " │")
        lines.append("└" + "─" * (self.width * 2 + 1) + "┘")

        output = "\n".join(lines)
        print(output)
        return output

    def get_optimal_path(self) -> List[Tuple[int, int]]:
        """
        获取最短路径（不考虑悬崖风险）

        这是 Q-Learning 在训练收敛后会学到的路径：
        沿着悬崖边从起点直接走到终点。

        Returns:
            从起点到终点的最短路径
        """
        path = [self.start]
        for j in range(1, self.width):
            path.append((self.height - 1, j))
        return path

    def get_safe_path(self) -> List[Tuple[int, int]]:
        """
        获取安全路径（远离悬崖）

        这是 SARSA 在训练收敛后会学到的路径：
        先向上远离悬崖，然后向右，最后向下到达目标。

        Returns:
            避开悬崖的安全路径
        """
        path = [self.start]
        # 先向上
        for i in range(self.height - 2, -1, -1):
            path.append((i, 0))
        # 向右
        for j in range(1, self.width):
            path.append((0, j))
        # 向下到达目标
        for i in range(1, self.height):
            path.append((i, self.width - 1))
        return path


# ============================================================
# 训练工具函数 (Training Utilities)
# ============================================================


def train_q_learning(
    env: Environment,
    agent: QLearningAgent,
    episodes: int = 500,
    max_steps: int = 200,
    verbose: bool = True,
    log_interval: int = 100,
) -> TrainingMetrics:
    """
    训练 Q-Learning 智能体

    实现标准的 Q-Learning 训练循环，适用于任何实现了 reset() 和 step() 方法的环境。

    Args:
        env: 环境实例（需实现 reset() 和 step() 方法）
        agent: Q-Learning 智能体
        episodes: 训练回合数
        max_steps: 每回合最大步数（防止无限循环）
        verbose: 是否打印训练进度
        log_interval: 日志打印间隔

    Returns:
        训练指标记录
    """
    metrics = TrainingMetrics()

    for episode in range(episodes):
        # 环境重置，兼容 Gymnasium 和自定义环境
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result

        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            # 选择动作
            action = agent.get_action(state, training=True)

            # 执行动作，兼容不同的返回格式
            result = env.step(action)
            if len(result) == 3:
                next_state, reward, done = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            # 更新 Q 值
            td_error = agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # 衰减探索率
        agent.decay_epsilon()

        # 记录指标
        metrics.episode_rewards.append(total_reward)
        metrics.episode_lengths.append(steps)
        metrics.epsilon_history.append(agent.epsilon)

        # 打印进度
        if verbose and (episode + 1) % log_interval == 0:
            avg_reward = np.mean(metrics.episode_rewards[-log_interval:])
            avg_steps = np.mean(metrics.episode_lengths[-log_interval:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Steps: {avg_steps:6.1f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    agent.metrics = metrics
    return metrics


def train_sarsa(
    env: Environment,
    agent: SARSAAgent,
    episodes: int = 500,
    max_steps: int = 200,
    verbose: bool = True,
    log_interval: int = 100,
) -> TrainingMetrics:
    """
    训练 SARSA 智能体

    SARSA 的训练循环与 Q-Learning 略有不同：
    - 需要在循环开始前选择初始动作
    - 更新时需要传入下一个动作

    Args:
        env: 环境实例
        agent: SARSA 智能体
        episodes: 训练回合数
        max_steps: 每回合最大步数
        verbose: 是否打印训练进度
        log_interval: 日志打印间隔

    Returns:
        训练指标记录
    """
    metrics = TrainingMetrics()

    for episode in range(episodes):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result

        # SARSA 需要先选择初始动作
        action = agent.get_action(state, training=True)

        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            # 执行动作
            result = env.step(action)
            if len(result) == 3:
                next_state, reward, done = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            # 选择下一个动作
            next_action = agent.get_action(next_state, training=True)

            # SARSA 更新（需要 next_action）
            td_error = agent.update(state, action, reward, next_state, next_action, done)

            # 状态和动作传递
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        metrics.episode_rewards.append(total_reward)
        metrics.episode_lengths.append(steps)
        metrics.epsilon_history.append(agent.epsilon)

        if verbose and (episode + 1) % log_interval == 0:
            avg_reward = np.mean(metrics.episode_rewards[-log_interval:])
            avg_steps = np.mean(metrics.episode_lengths[-log_interval:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Steps: {avg_steps:6.1f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    agent.metrics = metrics
    return metrics


def extract_path(
    agent: BaseAgent,
    env: CliffWalkingEnv,
    max_steps: int = 50,
) -> List[Tuple[int, int]]:
    """
    从训练好的智能体提取贪心策略路径

    使用纯贪心策略（不探索）生成轨迹，展示学到的最优行为。

    Args:
        agent: 训练好的智能体
        env: 环境实例
        max_steps: 最大步数（防止无限循环）

    Returns:
        策略产生的状态序列
    """
    state = env.reset()
    path = [state]

    for _ in range(max_steps):
        action = agent.get_action(state, training=False)  # 不探索
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
        if done:
            break

    return path


# ============================================================
# 可视化工具 (Visualization Utilities)
# ============================================================


def plot_learning_curves(
    metrics_dict: Dict[str, TrainingMetrics],
    window: int = 10,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    绘制学习曲线对比图

    可视化多个算法的训练过程，便于对比分析。

    Args:
        metrics_dict: {算法名称: 训练指标} 字典
        window: 平滑窗口大小，用于移动平均
        figsize: 图形尺寸
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 奖励曲线
    ax1 = axes[0]
    for name, metrics in metrics_dict.items():
        smoothed = np.convolve(
            metrics.episode_rewards, np.ones(window) / window, mode="valid"
        )
        ax1.plot(smoothed, label=name, alpha=0.8)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Learning Curve: Episode Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 步数曲线
    ax2 = axes[1]
    for name, metrics in metrics_dict.items():
        smoothed = np.convolve(
            metrics.episode_lengths, np.ones(window) / window, mode="valid"
        )
        ax2.plot(smoothed, label=name, alpha=0.8)

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Learning Curve: Episode Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def visualize_q_table(
    agent: BaseAgent,
    env: CliffWalkingEnv,
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    可视化 Q 表和策略

    生成三个子图：
    1. 价值函数热力图：展示每个状态的最大 Q 值
    2. 策略箭头图：展示每个状态的贪心动作
    3. Q 值分布直方图：展示所有 Q 值的分布

    Args:
        agent: 训练好的智能体
        env: 环境实例
        figsize: 图形尺寸
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 准备数据
    v_table = np.zeros((env.height, env.width))
    policy_arrows = np.zeros((env.height, env.width), dtype=int)

    for i in range(env.height):
        for j in range(env.width):
            state = (i, j)
            if state in agent.q_table:
                v_table[i, j] = np.max(agent.q_table[state])
                policy_arrows[i, j] = np.argmax(agent.q_table[state])

    # 价值函数热力图
    ax1 = axes[0]
    im = ax1.imshow(v_table, cmap="RdYlGn")
    ax1.set_title("Value Function V(s)")
    plt.colorbar(im, ax=ax1)

    # 标记悬崖
    for pos in env.cliff:
        ax1.add_patch(
            plt.Rectangle(
                (pos[1] - 0.5, pos[0] - 0.5), 1, 1, fill=True, color="black", alpha=0.5
            )
        )

    # 策略箭头图
    ax2 = axes[1]
    arrow_map = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    for i in range(env.height):
        for j in range(env.width):
            if (i, j) in env.cliff:
                ax2.text(j, i, "X", ha="center", va="center", fontsize=12)
            elif (i, j) == env.goal:
                ax2.text(j, i, "G", ha="center", va="center", fontsize=12, color="green")
            elif (i, j) == env.start:
                ax2.text(j, i, "S", ha="center", va="center", fontsize=12, color="blue")
            else:
                ax2.text(
                    j, i, arrow_map[policy_arrows[i, j]], ha="center", va="center", fontsize=14
                )

    ax2.set_xlim(-0.5, env.width - 0.5)
    ax2.set_ylim(env.height - 0.5, -0.5)
    ax2.set_title("Greedy Policy")
    ax2.grid(True)

    # Q 值分布直方图
    ax3 = axes[2]
    q_max_values = [np.max(q) for q in agent.q_table.values() if np.any(q != 0)]
    if q_max_values:
        ax3.hist(q_max_values, bins=30, edgecolor="black", alpha=0.7)
    ax3.set_xlabel("Q Value")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Q Value Distribution")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================


def compare_cliff_walking(
    episodes: int = 500,
    learning_rate: float = 0.5,
    epsilon: float = 0.1,
    show_plots: bool = True,
) -> Tuple[QLearningAgent, SARSAAgent]:
    """
    在悬崖行走环境上对比 Q-Learning 和 SARSA

    这是展示两种算法行为差异的经典实验：
    - Q-Learning 学习最优但风险较高的路径
    - SARSA 学习安全但较长的路径

    Args:
        episodes: 训练回合数
        learning_rate: 学习率
        epsilon: 固定探索率（不衰减，便于观察差异）
        show_plots: 是否显示图表

    Returns:
        (q_agent, sarsa_agent) 元组
    """
    print("=" * 60)
    print("悬崖行走环境: Q-Learning vs SARSA 对比实验")
    print("=" * 60)

    env = CliffWalkingEnv()

    # 创建智能体（固定 epsilon，不衰减）
    q_agent = QLearningAgent(
        n_actions=4,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_decay=1.0,
        epsilon_min=epsilon,
    )

    sarsa_agent = SARSAAgent(
        n_actions=4,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_decay=1.0,
        epsilon_min=epsilon,
    )

    # 训练
    print("\n训练 Q-Learning...")
    q_metrics = train_q_learning(env, q_agent, episodes=episodes, verbose=True, log_interval=100)

    print("\n训练 SARSA...")
    sarsa_metrics = train_sarsa(env, sarsa_agent, episodes=episodes, verbose=True, log_interval=100)

    # 显示学到的路径
    print("\n" + "=" * 60)
    print("学习到的策略路径")
    print("=" * 60)

    print("\nQ-Learning (倾向最短路径):")
    q_path = extract_path(q_agent, env)
    env.render(q_path)

    env.reset()
    print("\nSARSA (倾向安全路径):")
    sarsa_path = extract_path(sarsa_agent, env)
    env.render(sarsa_path)

    # 统计
    print("\n" + "=" * 60)
    print("训练统计")
    print("=" * 60)
    print(f"Q-Learning 最后100回合平均奖励: {np.mean(q_metrics.episode_rewards[-100:]):.2f}")
    print(f"SARSA 最后100回合平均奖励: {np.mean(sarsa_metrics.episode_rewards[-100:]):.2f}")

    if show_plots:
        plot_learning_curves(
            {"Q-Learning": q_metrics, "SARSA": sarsa_metrics},
            window=10,
            save_path="cliff_walking_comparison.png",
        )

    return q_agent, sarsa_agent


def train_taxi(episodes: int = 2000, show_plots: bool = True) -> Optional[QLearningAgent]:
    """
    在 Taxi-v3 环境上训练 Q-Learning

    Taxi-v3 是 Gymnasium 内置的经典强化学习环境：
    - 状态空间：500 种状态
    - 动作空间：6 种动作
    - 目标：接送乘客，最大化奖励

    Args:
        episodes: 训练回合数
        show_plots: 是否显示图表

    Returns:
        训练好的智能体，如果 gymnasium 不可用则返回 None
    """
    try:
        import gymnasium as gym
    except ImportError:
        print("请安装 gymnasium: pip install gymnasium")
        return None

    print("\n" + "=" * 60)
    print("Taxi-v3 Q-Learning 训练")
    print("=" * 60)

    env = gym.make("Taxi-v3")

    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    metrics = train_q_learning(env, agent, episodes=episodes, verbose=True, log_interval=200)

    env.close()

    if show_plots:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        window = 50
        smoothed_rewards = np.convolve(
            metrics.episode_rewards, np.ones(window) / window, mode="valid"
        )
        axes[0].plot(smoothed_rewards)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total Reward")
        axes[0].set_title("Taxi-v3: Reward per Episode")
        axes[0].grid(True, alpha=0.3)

        smoothed_steps = np.convolve(
            metrics.episode_lengths, np.ones(window) / window, mode="valid"
        )
        axes[1].plot(smoothed_steps)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].set_title("Taxi-v3: Steps per Episode")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("taxi_training.png", dpi=150)
        plt.show()

    return agent


def main() -> None:
    """主程序入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Q-Learning 与 SARSA 实验")
    parser.add_argument(
        "--exp",
        type=str,
        default="cliff",
        choices=["cliff", "taxi", "all"],
        help="实验类型: cliff/taxi/all",
    )
    parser.add_argument("--episodes", type=int, default=500, help="训练回合数")
    parser.add_argument("--no-plot", action="store_true", help="不显示图表")

    args = parser.parse_args()

    if args.exp in ["cliff", "all"]:
        compare_cliff_walking(episodes=args.episodes, show_plots=not args.no_plot)

    if args.exp in ["taxi", "all"]:
        train_taxi(
            episodes=args.episodes * 4 if args.exp == "all" else args.episodes,
            show_plots=not args.no_plot,
        )


# ============================================================
# 单元测试 (Unit Tests)
# ============================================================


def _run_tests() -> bool:
    """
    运行单元测试

    验证核心组件的正确性：
    1. AgentConfig 参数验证
    2. Q-Learning 更新规则
    3. SARSA 更新规则
    4. Double Q-Learning 更新
    5. 环境基本功能
    6. 训练收敛性

    Returns:
        所有测试是否通过
    """
    print("运行单元测试...")
    print("=" * 50)

    passed = 0
    failed = 0

    # 测试 1: AgentConfig 参数验证
    try:
        config = AgentConfig(n_actions=4, learning_rate=0.1)
        assert config.n_actions == 4
        assert config.learning_rate == 0.1

        # 验证无效参数
        try:
            AgentConfig(n_actions=0)
            assert False, "应该抛出异常"
        except ValueError:
            pass

        print("测试 1 通过: AgentConfig 参数验证")
        passed += 1
    except AssertionError as e:
        print(f"测试 1 失败: {e}")
        failed += 1

    # 测试 2: Q-Learning 更新规则
    try:
        agent = QLearningAgent(n_actions=4, learning_rate=0.5, discount_factor=0.9)
        state = (0, 0)
        next_state = (0, 1)

        # 初始 Q 值应为 0
        assert agent.q_table[state][0] == 0.0, "初始 Q 值应为 0"

        # 执行更新
        td_error = agent.update(state, 0, -1.0, next_state, False)

        # 验证更新后的 Q 值
        # Q(s,a) = 0 + 0.5 * (-1 + 0.9 * 0 - 0) = -0.5
        expected_q = -0.5
        assert np.isclose(agent.q_table[state][0], expected_q), (
            f"Q 值更新错误: {agent.q_table[state][0]} != {expected_q}"
        )

        print("测试 2 通过: Q-Learning 更新规则")
        passed += 1
    except AssertionError as e:
        print(f"测试 2 失败: {e}")
        failed += 1

    # 测试 3: SARSA 更新规则
    try:
        agent = SARSAAgent(n_actions=4, learning_rate=0.5, discount_factor=0.9)
        state = (0, 0)
        next_state = (0, 1)

        # 设置 next_state 的 Q 值
        agent.q_table[next_state] = np.array([1.0, 2.0, 0.0, 0.0])

        # SARSA 更新：使用 next_action=1 (Q 值为 2.0)
        agent.update(state, 0, -1.0, next_state, 1, False)

        # Q(s,a) = 0 + 0.5 * (-1 + 0.9 * 2.0 - 0) = 0.5 * 0.8 = 0.4
        expected = 0.4
        assert np.isclose(agent.q_table[state][0], expected), (
            f"SARSA 更新错误: {agent.q_table[state][0]} != {expected}"
        )

        print("测试 3 通过: SARSA 更新规则")
        passed += 1
    except AssertionError as e:
        print(f"测试 3 失败: {e}")
        failed += 1

    # 测试 4: Double Q-Learning 更新
    try:
        agent = QLearningAgent(n_actions=4, learning_rate=0.5, double_q=True)
        state = (0, 0)
        next_state = (0, 1)

        # 多次更新，验证两个 Q 表都被更新
        np.random.seed(42)
        for _ in range(10):
            agent.update(state, 0, -1.0, next_state, False)

        # 验证两个 Q 表都有更新
        assert agent.q_table[state][0] != 0 or agent.q_table2[state][0] != 0

        print("测试 4 通过: Double Q-Learning 更新")
        passed += 1
    except AssertionError as e:
        print(f"测试 4 失败: {e}")
        failed += 1

    # 测试 5: 环境基本功能
    try:
        env = CliffWalkingEnv()
        state = env.reset()
        assert state == (3, 0), f"起始状态错误: {state}"

        # 向上移动
        next_state, reward, done = env.step(0)
        assert next_state == (2, 0), f"向上移动后状态错误: {next_state}"
        assert reward == -1.0, f"奖励错误: {reward}"
        assert not done, "不应该结束"

        # 悬崖惩罚
        env.reset()
        next_state, reward, done = env.step(1)  # 向右进入悬崖
        assert reward == -100.0, f"悬崖惩罚错误: {reward}"
        assert next_state == env.start, "掉入悬崖后应重置到起点"

        print("测试 5 通过: 环境基本功能")
        passed += 1
    except AssertionError as e:
        print(f"测试 5 失败: {e}")
        failed += 1

    # 测试 6: 训练收敛性
    try:
        env = CliffWalkingEnv()
        agent = QLearningAgent(
            n_actions=4,
            learning_rate=0.5,
            epsilon=0.1,
            epsilon_decay=1.0,
            epsilon_min=0.1,
        )

        metrics = train_q_learning(env, agent, episodes=200, verbose=False)

        # 最后 50 回合平均奖励应该大于 -100
        avg_reward = np.mean(metrics.episode_rewards[-50:])
        assert avg_reward > -100, f"训练未收敛: avg_reward = {avg_reward}"

        print(f"测试 6 通过: 训练收敛 (最后 50 回合平均奖励: {avg_reward:.2f})")
        passed += 1
    except AssertionError as e:
        print(f"测试 6 失败: {e}")
        failed += 1

    # 总结
    print("=" * 50)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    if failed == 0:
        print("所有测试通过！")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = _run_tests()
        sys.exit(0 if success else 1)
    else:
        main()
