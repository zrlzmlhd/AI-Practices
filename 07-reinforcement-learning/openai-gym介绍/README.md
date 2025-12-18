# OpenAI Gymnasium 强化学习环境接口指南

本模块提供 Gymnasium (原 OpenAI Gym) 强化学习环境接口的完整教程，包括环境使用、
包装器应用和经典控制任务详解。采用**模块化架构设计**，便于维护和扩展。

---

## 目录结构

```
openai-gym介绍/
├── README.md                    # 本文件 - 项目说明
├── 知识点.md                    # 核心知识点总结与复习指南
│
├── core/                        # 核心功能模块
│   ├── __init__.py             # 模块导出
│   ├── spaces.py               # 空间分析工具
│   ├── env_spec.py             # 环境规格提取
│   ├── episode.py              # 回合执行工具
│   └── evaluation.py           # 策略评估工具
│
├── environments/                # 环境描述与策略
│   ├── __init__.py             # 模块导出
│   ├── descriptions.py         # 环境物理描述
│   └── policies.py             # 控制策略实现
│
├── wrappers/                    # 环境包装器
│   ├── __init__.py             # 模块导出
│   ├── statistics.py           # 在线统计工具
│   ├── observation.py          # 观测包装器
│   ├── action.py               # 动作包装器
│   ├── reward.py               # 奖励包装器
│   └── factory.py              # 工厂函数
│
├── notebooks/                   # Jupyter 教程
│   ├── 01-gymnasium-quickstart.ipynb     # 快速入门
│   └── 02-classic-control-deep-dive.ipynb # 经典控制深度解析
│
└── tests/                       # 单元测试
    └── (每个模块内置 _run_tests 函数)
```

---

## 核心概念

### 马尔可夫决策过程 (MDP)

强化学习建模为**马尔可夫决策过程**：

$$MDP = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

| 符号 | 含义 |
|------|------|
| $\mathcal{S}$ | 状态空间 |
| $\mathcal{A}$ | 动作空间 |
| $P(s'\|s,a)$ | 转移概率 |
| $R(s,a,s')$ | 奖励函数 |
| $\gamma \in [0,1]$ | 折扣因子 |

### 期望累积折扣奖励

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

---

## 模块说明

### 1. core - 核心功能模块

提供 Gymnasium 环境交互的基础工具。

#### spaces.py - 空间分析

```python
from core.spaces import analyze_space, SpaceType, SpaceInfo

# 分析空间类型
info = analyze_space(env.observation_space)
print(f"类型: {info.space_type}, 维度: {info.flat_dim}")
```

#### env_spec.py - 环境规格

```python
from core.env_spec import get_env_spec

spec = get_env_spec("CartPole-v1")
print(f"观测维度: {spec.obs_info.flat_dim}")
print(f"最大步数: {spec.max_episode_steps}")
```

#### episode.py - 回合执行

```python
from core.episode import run_episode, EpisodeResult

result = run_episode(env, policy_fn, seed=42)
print(f"总奖励: {result.total_reward}")
print(f"折扣回报: {result.get_discounted_returns(gamma=0.99)[0]:.2f}")
```

#### evaluation.py - 策略评估

```python
from core.evaluation import evaluate_policy, PolicyEvaluator

# 单策略评估
result = evaluate_policy(env, policy, n_episodes=100)
print(f"平均奖励: {result.mean_reward:.1f} ± {result.std_reward:.1f}")

# 多策略比较
evaluator = PolicyEvaluator(env)
comparison = evaluator.compare({"Random": random_policy, "PID": pid_policy})
```

---

### 2. environments - 环境与策略

#### descriptions.py - 环境描述

```python
from environments.descriptions import CARTPOLE_DESCRIPTION

print(CARTPOLE_DESCRIPTION.dynamics)
print(CARTPOLE_DESCRIPTION.physics_params)
```

#### policies.py - 策略实现

```python
from environments.policies import CartPolePolicy, PendulumPolicy

# CartPole 策略
policy = CartPolePolicy(method="pid")
action = policy(observation)

# Pendulum 策略
pd_controller = PendulumPolicy(method="pd", Kp=10.0, Kd=2.0)
torque = pd_controller(observation)
```

**支持的策略方法：**

| 环境 | 方法 |
|------|------|
| CartPole | `random`, `angle`, `pid`, `linear` |
| MountainCar | `random`, `momentum`, `energy` |
| Pendulum | `random`, `pd`, `energy` |
| Acrobot | `random`, `energy` |

---

### 3. wrappers - 环境包装器

#### 观测包装器

```python
from wrappers import NormalizeObservationWrapper, FrameStackWrapper

env = NormalizeObservationWrapper(env, epsilon=1e-8)
env = FrameStackWrapper(env, n_frames=4)
```

#### 动作包装器

```python
from wrappers import ClipActionWrapper, RescaleActionWrapper

env = ClipActionWrapper(env)  # 裁剪到有效范围
env = RescaleActionWrapper(env)  # [-1, 1] → [low, high]
```

#### 奖励包装器

```python
from wrappers import NormalizeRewardWrapper, ClipRewardWrapper

env = NormalizeRewardWrapper(env, gamma=0.99)
env = ClipRewardWrapper(env, min_reward=-1, max_reward=1)
```

#### 工厂函数

```python
from wrappers import make_wrapped_env

env = make_wrapped_env(
    "Pendulum-v1",
    normalize_obs=True,
    normalize_reward=True,
    clip_action=True,
    frame_stack=0,
    time_limit=200,
    record_stats=True
)
```

---

### 4. notebooks - Jupyter 教程

| 文件 | 内容 |
|------|------|
| `01-gymnasium-quickstart.ipynb` | Gymnasium 核心 API、空间类型、基础交互 |
| `02-classic-control-deep-dive.ipynb` | CartPole/MountainCar/Pendulum 深度解析 |

---

## 快速开始

### 安装依赖

```bash
pip install gymnasium numpy matplotlib
pip install gymnasium[classic-control]
```

### 示例 1: 基础环境交互

```python
import gymnasium as gym
from core import run_episode, evaluate_policy

env = gym.make("CartPole-v1")

# 定义策略
def angle_policy(obs):
    return 1 if obs[2] > 0 else 0

# 评估
result = evaluate_policy(env, angle_policy, n_episodes=50)
print(f"平均奖励: {result.mean_reward:.1f}")
```

### 示例 2: 使用包装器

```python
from wrappers import make_wrapped_env

env = make_wrapped_env(
    "Pendulum-v1",
    normalize_obs=True,
    normalize_reward=True
)

obs, _ = env.reset()
# obs 已归一化
```

### 示例 3: 策略比较

```python
from core import PolicyEvaluator
from environments import CartPolePolicy
import gymnasium as gym

env = gym.make("CartPole-v1")
evaluator = PolicyEvaluator(env)

policies = {
    "Random": CartPolePolicy("random"),
    "Angle": CartPolePolicy("angle"),
    "PID": CartPolePolicy("pid")
}

comparison = evaluator.compare(policies, n_episodes=50)
evaluator.print_comparison(comparison)
```

---

## 运行测试

每个模块内置 `_run_tests()` 函数：

```bash
# 测试核心模块
python -c "from core.spaces import _run_tests; _run_tests()"
python -c "from core.evaluation import _run_tests; _run_tests()"

# 测试包装器
python -c "from wrappers.observation import _run_tests; _run_tests()"
python -c "from wrappers.factory import _run_tests; _run_tests()"

# 测试策略
python -c "from environments.policies import _run_tests; _run_tests()"
```

---

## 环境列表

### 经典控制 (Classic Control)

| 环境 ID | 状态维度 | 动作空间 | 挑战 |
|---------|----------|----------|------|
| CartPole-v1 | 4 | Discrete(2) | 欠驱动控制 |
| MountainCar-v0 | 2 | Discrete(3) | 稀疏奖励 |
| Pendulum-v1 | 3 | Box(1) | 连续控制 |
| Acrobot-v1 | 6 | Discrete(3) | 欠驱动双摆 |

---

## 设计原则

### 架构特点

1. **模块化设计**: 每个功能独立成模块，低耦合高内聚
2. **类型安全**: 使用 dataclass 和类型注解
3. **可测试性**: 每个模块内置单元测试
4. **渐进式导入**: 核心功能不强制依赖可选库

### 代码规范

- 遵循 Google Python Style Guide
- 完整的中英文双语文档字符串
- LaTeX 数学公式标注算法原理

---

## 知识点复习

详见 [`知识点.md`](知识点.md)，包含：

- MDP 数学定义
- Gymnasium API 详解
- 空间类型说明
- 经典控制环境物理学
- 包装器设计模式
- 控制理论基础 (PID, 能量控制)
- 在线统计算法 (Welford, EMA)
- 思考题与练习

---

## 参考文献

1. Brockman, G. et al. (2016). OpenAI Gym. arXiv:1606.01540
2. Towers, M. et al. (2023). Gymnasium: A Standard Interface for RL Environments
3. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
4. Welford, B. P. (1962). Note on a method for calculating corrected sums

---

## 许可证

MIT License

---

*最后更新: 2025-12*
