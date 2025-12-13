# 07-Reinforcement Learning | 强化学习

> MDP、Q-Learning 与深度强化学习

---

## 目录结构

```
07-reinforcement-learning/
├── 01-mdp-basics/                      # MDP 基础：状态、动作、奖励、策略
│   ├── 马尔可夫决策过程.md
│   └── grid_world_dp.py
├── 02-q-learning/                      # Q-Learning：值函数、探索与利用
│   ├── Q-Learning详解.md
│   └── q_learning_sarsa.py
├── 03-deep-rl/                         # 深度 RL：DQN、A2C、PPO
│   ├── 深度强化学习.md
│   └── dqn_cartpole.py
├── 04-policy-gradient/                 # 策略梯度：REINFORCE、Actor-Critic
│   ├── 策略梯度方法.md
│   └── policy_gradient.py
├── 学习优化奖励/                        # 奖励函数设计与优化
├── 策略搜索/                            # 策略搜索算法
├── openai-gym介绍/                     # OpenAI Gym/Gymnasium 环境
├── 神经网络策略/                        # 神经网络作为策略函数
├── 评估动作-信用分配问题/               # Credit Assignment Problem
├── 策略梯度/                            # 策略梯度方法详解
├── 马尔科夫决策过程/                    # MDP 理论深入
├── 时序差分学习/                        # TD Learning
├── Q学习/                               # Q-Learning 及变体
├── 实现深度Q学习/                       # DQN 实现详解
├── 深度Q学习的变体/                     # Double DQN, Dueling DQN, Rainbow
├── Tf-Agents库/                        # TensorFlow Agents 库使用
└── 流行强化学习算法概述/                # PPO, SAC, TD3 等算法介绍
```

---

## 学习路线

```
┌─────────────────────────────────────────────────────────────────────┐
│                        强化学习学习路线                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  基础理论                                                           │
│  ├── 马尔可夫决策过程 (MDP)                                          │
│  ├── 贝尔曼方程                                                     │
│  └── 动态规划                                                       │
│         │                                                          │
│         ▼                                                          │
│  无模型方法                                                         │
│  ├── 时序差分学习 (TD)                                              │
│  ├── Q-Learning / SARSA                                            │
│  └── 探索与利用                                                     │
│         │                                                          │
│         ▼                                                          │
│  深度强化学习                                                       │
│  ├── DQN 及变体                                                     │
│  ├── 策略梯度 (REINFORCE)                                           │
│  └── Actor-Critic (A2C/A3C)                                        │
│         │                                                          │
│         ▼                                                          │
│  高级算法                                                           │
│  ├── PPO / TRPO                                                    │
│  ├── SAC / TD3                                                     │
│  └── 多智能体 RL                                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心内容

| 子模块 | 核心概念 | 实践重点 |
|--------|----------|----------|
| MDP 基础 | 马尔可夫决策过程、贝尔曼方程 | 网格世界、动态规划 |
| Q-Learning | 值迭代、ε-greedy、TD 误差 | 表格型 Q-Learning、SARSA |
| 深度 RL | 经验回放、目标网络 | DQN、CartPole/Atari |
| 策略梯度 | REINFORCE、优势函数 | Actor-Critic、连续控制 |
| OpenAI Gym | 环境接口、标准化 | Gymnasium 使用 |
| TF-Agents | 高级 RL 库 | 生产级实现 |

---

## 已完成内容

### 01-mdp-basics
- [x] 马尔可夫决策过程理论
- [x] 贝尔曼方程详解
- [x] 动态规划 (策略迭代、值迭代)
- [x] 网格世界代码实现

### 02-q-learning
- [x] Q-Learning 算法详解
- [x] SARSA 算法
- [x] 探索与利用策略
- [x] 悬崖行走实战

### 03-deep-rl
- [x] DQN 原理与实现
- [x] Double DQN、Dueling DQN
- [x] Actor-Critic 方法
- [x] CartPole 实战

### 04-policy-gradient
- [x] 策略梯度定理
- [x] REINFORCE 算法
- [x] A2C 实现
- [x] 方差减少技术

---

## 环境配置

```bash
# 基础依赖
pip install numpy matplotlib

# PyTorch (根据你的系统选择)
pip install torch

# Gymnasium (OpenAI Gym 的维护版本)
pip install gymnasium
pip install gymnasium[classic-control]  # CartPole 等经典环境
pip install gymnasium[atari]            # Atari 游戏

# TensorFlow Agents (可选)
pip install tf-agents
```

---

## 推荐资源

### 书籍
- Sutton & Barto, "Reinforcement Learning: An Introduction" (圣经)
- 《深度强化学习》- 王树森

### 课程
- David Silver, UCL RL Course
- OpenAI Spinning Up
- 李宏毅深度强化学习

### 工具库
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)

---

[返回主页](../README.md)
