# 07 - 强化学习

从马尔可夫决策过程到深度强化学习，掌握智能体决策技术。

## 模块概览

| 属性 | 值 |
|:-----|:---|
| **前置要求** | 概率论, 优化理论, 深度学习基础 |
| **学习时长** | 3-4 周 |
| **Notebooks** | 12+ |
| **难度** | ⭐⭐⭐⭐ 高级 |

## 学习目标

完成本模块后，你将能够：

- ✅ 理解马尔可夫决策过程 (MDP) 的数学框架
- ✅ 掌握值迭代、策略迭代等经典算法
- ✅ 实现 Q-Learning 和 DQN
- ✅ 理解策略梯度方法和 Actor-Critic 架构
- ✅ 应用 PPO 等现代算法解决实际问题

---

## 子模块详解

### 01. 马尔可夫决策过程 (MDP)

强化学习的数学基础。

**MDP 定义**：

一个 MDP 由五元组 $(S, A, P, R, \gamma)$ 定义：

| 符号 | 含义 | 说明 |
|:-----|:-----|:-----|
| $S$ | 状态空间 | 所有可能的状态集合 |
| $A$ | 动作空间 | 所有可能的动作集合 |
| $P$ | 状态转移概率 | $P(s'|s,a)$ |
| $R$ | 奖励函数 | $R(s, a, s')$ |
| $\gamma$ | 折扣因子 | $\gamma \in [0, 1)$ |

**核心概念**：

```
时刻 t:    状态 s_t ──► 动作 a_t ──► 奖励 r_t+1 ──► 状态 s_t+1
                │            │            │            │
                └────────────┴────────────┴────────────┘
                              交互循环
```

**回报与价值函数**：

**累积回报**：
$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

**状态价值函数**：
$$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]$$

**动作价值函数**：
$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$$

**贝尔曼方程**：

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a)[r + \gamma V^\pi(s')]$$

$$Q^\pi(s, a) = \sum_{s', r} p(s', r|s, a)[r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]$$

---

### 02. 动态规划

已知环境模型时的最优求解。

**策略评估 (Policy Evaluation)**：

```python
def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V
```

**值迭代 (Value Iteration)**：

$$V_{k+1}(s) = \max_a \sum_{s', r} p(s', r|s, a)[r + \gamma V_k(s')]$$

**策略迭代 (Policy Iteration)**：

```
初始化策略 π
循环:
    1. 策略评估: 计算 V^π
    2. 策略改进: π'(s) = argmax_a Q^π(s, a)
    3. 如果 π' = π，停止
```

---

### 03. 无模型方法：蒙特卡洛与 TD

不需要环境模型的学习方法。

**方法对比**：

| 方法 | 更新时机 | 偏差 | 方差 |
|:-----|:---------|:-----|:-----|
| **蒙特卡洛 (MC)** | 回合结束 | 无偏 | 高 |
| **时序差分 (TD)** | 每步 | 有偏 | 低 |
| **n-步 TD** | n 步后 | 折中 | 折中 |

**TD(0) 更新**：

$$V(s_t) \leftarrow V(s_t) + \alpha[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

**SARSA (On-Policy TD Control)**：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

```python
def sarsa(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for episode in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        while True:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            # SARSA 更新
            td_target = reward + gamma * Q[next_state][next_action]
            Q[state][action] += alpha * (td_target - Q[state][action])

            if done:
                break
            state, action = next_state, next_action

    return Q
```

**Q-Learning (Off-Policy TD Control)**：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)]$$

---

### 04. Deep Q-Network (DQN)

使用深度神经网络近似 Q 函数。

**DQN 核心创新**：

| 技术 | 作用 |
|:-----|:-----|
| **经验回放** | 打破数据相关性 |
| **目标网络** | 稳定训练 |
| **Clip Reward** | 归一化奖励 |

**DQN 算法**：

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def train_dqn(env, num_episodes):
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = ReplayBuffer(10000)

    for episode in range(num_episodes):
        state = env.reset()

        while True:
            # ε-greedy 选择动作
            action = select_action(state, policy_net, epsilon)
            next_state, reward, done, _ = env.step(action)

            # 存储经验
            memory.push(state, action, reward, next_state, done)

            # 训练
            if len(memory.buffer) > batch_size:
                batch = memory.sample(batch_size)
                loss = compute_loss(batch, policy_net, target_net)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 定期更新目标网络
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
            state = next_state
```

**DQN 改进变体**：

| 变体 | 创新点 |
|:-----|:-------|
| Double DQN | 解耦动作选择与评估 |
| Dueling DQN | 分离状态价值与优势函数 |
| Prioritized Experience Replay | 优先采样重要经验 |
| Rainbow | 集成多种改进 |

---

### 05. 策略梯度方法

直接优化策略参数。

**策略梯度定理**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

**REINFORCE 算法**：

```python
def reinforce(env, policy_net, optimizer, num_episodes):
    for episode in range(num_episodes):
        states, actions, rewards = [], [], []
        state = env.reset()

        # 生成一条轨迹
        while True:
            action = policy_net.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            if done:
                break
            state = next_state

        # 计算回报
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # 策略梯度更新
        loss = 0
        for state, action, G in zip(states, actions, returns):
            log_prob = policy_net.log_prob(state, action)
            loss += -log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

### 06. Actor-Critic 方法

结合策略梯度和价值函数。

**架构**：

```
状态 s ──┬──► Actor (策略网络) ──► 动作 a
         │
         └──► Critic (价值网络) ──► V(s) 或 Q(s,a)
```

**Advantage Actor-Critic (A2C)**：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)\right]$$

其中优势函数：
$$A(s, a) = Q(s, a) - V(s) \approx r + \gamma V(s') - V(s)$$

**PPO (Proximal Policy Optimization)**：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

**PPO 优势**：

| 特性 | 说明 |
|:-----|:-----|
| 稳定性 | 限制策略更新幅度 |
| 样本效率 | 可重复使用经验 |
| 易实现 | 无需 KL 散度约束 |
| SOTA | 大多数任务表现优异 |

---

### 07. 多智能体强化学习

多个智能体协同或竞争。

| 场景 | 特点 | 应用 |
|:-----|:-----|:-----|
| **合作** | 共同目标 | 机器人协作 |
| **竞争** | 零和博弈 | 游戏对战 |
| **混合** | 部分合作竞争 | 自动驾驶 |

**核心挑战**：

- 非平稳环境（其他智能体也在学习）
- 信用分配问题
- 通信与协调

---

## 实验列表

| 实验 | 内容 | 文件 |
|:-----|:-----|:-----|
| MDP 基础 | GridWorld 求解 | `01_mdp_gridworld.ipynb` |
| 值迭代 | 冰湖环境 | `02_value_iteration.ipynb` |
| Q-Learning | Taxi-v3 任务 | `03_q_learning.ipynb` |
| DQN | CartPole 平衡 | `04_dqn_cartpole.ipynb` |
| REINFORCE | LunarLander | `05_reinforce.ipynb` |
| A2C | Atari 游戏 | `06_a2c_atari.ipynb` |
| PPO | MuJoCo 连续控制 | `07_ppo_mujoco.ipynb` |
| Multi-Agent | 多智能体粒子环境 | `08_multiagent.ipynb` |

---

## 参考资源

### 教材
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.) - [在线阅读](http://incompleteideas.net/book/the-book-2nd.html)
- Szepesvári, C. (2010). *Algorithms for Reinforcement Learning*

### 论文
- Mnih et al. (2015). Human-level control through deep reinforcement learning (DQN)
- Schulman et al. (2017). Proximal Policy Optimization Algorithms (PPO)
- Silver et al. (2016). Mastering the game of Go with deep neural networks

### 课程
- [Stanford CS234](http://web.stanford.edu/class/cs234/) - Reinforcement Learning
- [DeepMind x UCL RL Course](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)
- [OpenAI Spinning Up](https://spinningup.openai.com/) - RL 实战教程

### 工具
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - PyTorch RL 算法库
- [RLlib](https://docs.ray.io/en/latest/rllib/) - 分布式 RL 框架
- [Gym](https://www.gymlibrary.dev/) - RL 环境标准接口
