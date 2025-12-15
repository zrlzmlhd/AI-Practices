# TF-Agents 深度强化学习库学习模块

## 概述

TF-Agents 是 Google 开发的基于 TensorFlow 的强化学习库，提供了模块化、可扩展的强化学习算法实现。本模块系统性地介绍 TF-Agents 的核心概念、主流算法及其工程实践。

## 模块结构

```
Tf-Agents库/
├── README.md                           # 本文档
├── 01_tf_agents_fundamentals.py        # TF-Agents 核心组件与基础概念
├── 02_dqn_cartpole.py                  # DQN 算法实现（离散动作空间）
├── 03_sac_continuous_control.py        # SAC 算法实现（连续动作空间）
├── 04_ppo_agent.py                     # PPO 算法实现（策略梯度方法）
├── 05_custom_environment.py            # 自定义环境开发
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py                # 经验回放缓冲区工具
│   ├── metrics.py                      # 评估指标与可视化
│   └── networks.py                     # 神经网络架构
├── notebooks/
│   ├── 01_tf_agents_tutorial.ipynb     # TF-Agents 交互式教程
│   ├── 02_dqn_deep_dive.ipynb          # DQN 深度解析
│   ├── 03_actor_critic_methods.ipynb   # Actor-Critic 方法详解
│   └── 04_hyperparameter_tuning.ipynb  # 超参数调优实践
└── tests/
    └── test_agents.py                  # 单元测试
```

## TF-Agents 核心组件

### 1. Environment（环境）

环境是智能体交互的世界，遵循 OpenAI Gym 接口规范：

```python
# TF-Agents 环境接口
class PyEnvironment:
    def reset(self) -> TimeStep          # 重置环境
    def step(action) -> TimeStep         # 执行动作
    def observation_spec() -> ArraySpec  # 观测空间规范
    def action_spec() -> ArraySpec       # 动作空间规范
```

### 2. Agent（智能体）

智能体封装了策略、价值函数和学习算法：

- **DQN Agent**: 离散动作空间，基于 Q-Learning
- **SAC Agent**: 连续动作空间，最大熵强化学习
- **PPO Agent**: 策略梯度方法，近端策略优化
- **DDPG Agent**: 确定性策略梯度
- **TD3 Agent**: Twin Delayed DDPG

### 3. Policy（策略）

策略定义了智能体的行为方式：

```python
policy.action(time_step)           # 获取动作
policy.distribution(time_step)     # 获取动作分布
collect_policy                     # 数据收集策略（带探索）
greedy_policy                      # 贪婪策略（无探索）
```

### 4. Replay Buffer（经验回放）

存储和采样历史经验：

- **Uniform Replay Buffer**: 均匀采样
- **Prioritized Replay Buffer**: 优先级采样（TD-error）
- **Reverb**: 分布式高性能回放系统

### 5. Driver（驱动器）

自动化数据收集过程：

```python
DynamicStepDriver    # 固定步数收集
DynamicEpisodeDriver # 固定回合收集
```

## 算法对比

| 算法 | 动作空间 | 样本效率 | 稳定性 | 复杂度 | 适用场景 |
|------|---------|---------|--------|--------|---------|
| DQN  | 离散    | 中等    | 高     | 低     | Atari游戏、离散控制 |
| SAC  | 连续    | 高      | 高     | 中     | 机器人控制、连续优化 |
| PPO  | 两者    | 低      | 高     | 中     | 通用任务、多智能体 |
| DDPG | 连续    | 中等    | 中     | 中     | 连续控制（较老方法）|
| TD3  | 连续    | 中等    | 高     | 中     | 连续控制（DDPG改进）|

## 环境配置

```bash
# 创建虚拟环境
conda create -n tf-agents python=3.10
conda activate tf-agents

# 安装依赖
pip install tensorflow==2.15.0
pip install tf-agents==0.19.0
pip install gymnasium[classic_control]
pip install matplotlib pandas

# 可选：GPU 支持
pip install tensorflow[and-cuda]
```

## 快速开始

```python
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network

# 1. 创建环境
env = suite_gym.load('CartPole-v1')

# 2. 构建 Q 网络
q_net = q_network.QNetwork(
    env.observation_spec(),
    env.action_spec(),
    fc_layer_params=(100, 50)
)

# 3. 创建 DQN 智能体
agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=tf.keras.optimizers.Adam(1e-3)
)

# 4. 训练循环
# ... 详见具体实现文件
```

## 学习路径

1. **入门阶段** (1-2天)
   - 阅读 `01_tf_agents_fundamentals.py` 理解核心组件
   - 运行 `notebooks/01_tf_agents_tutorial.ipynb` 进行交互式学习

2. **进阶阶段** (3-5天)
   - 学习 DQN 实现：`02_dqn_cartpole.py`
   - 理解 Actor-Critic 架构：`03_sac_continuous_control.py`
   - 掌握 PPO：`04_ppo_agent.py`

3. **实践阶段** (1周+)
   - 自定义环境开发：`05_custom_environment.py`
   - 超参数调优：`notebooks/04_hyperparameter_tuning.ipynb`
   - 将算法应用到实际问题

## 参考文献

1. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
2. Haarnoja, T. et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL. *ICML*.
3. Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. *arXiv*.
4. TF-Agents 官方文档: https://www.tensorflow.org/agents

## 许可证

本模块代码遵循 Apache 2.0 许可证。
