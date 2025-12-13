# 02 - 神经网络

掌握深度学习核心技术与训练方法，从零构建神经网络。

## 模块概览

| 属性 | 值 |
|:-----|:---|
| **前置要求** | 01-机器学习基础, 线性代数, 微积分 |
| **学习时长** | 3-4 周 |
| **Notebooks** | 15+ |
| **难度** | ⭐⭐⭐ 中级 |

## 学习目标

完成本模块后，你将能够：

- ✅ 理解神经网络的数学原理和计算图
- ✅ 手写实现前向传播和反向传播算法
- ✅ 掌握各种激活函数的特性和选择策略
- ✅ 熟练使用 SGD、Adam 等优化器
- ✅ 应用 Dropout、BatchNorm 等正则化技术
- ✅ 使用 Keras/PyTorch 构建自定义模型

---

## 子模块详解

### 01. 神经网络基础

理解神经网络的基本构成和工作原理。

| 主题 | 内容 | 关键概念 |
|:-----|:-----|:---------|
| 感知器 | 单层神经元模型 | 线性分类器 |
| 多层感知器 | MLP 架构 | 通用近似定理 |
| 激活函数 | ReLU, Sigmoid, Tanh, GELU | 非线性变换 |
| 前向传播 | 逐层计算输出 | 计算图 |

**激活函数对比**：

| 函数 | 公式 | 优点 | 缺点 |
|:-----|:-----|:-----|:-----|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | 输出 (0,1) | 梯度消失 |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | 零中心化 | 梯度消失 |
| ReLU | $\max(0, x)$ | 计算简单，缓解梯度消失 | Dead ReLU |
| GELU | $x \cdot \Phi(x)$ | 平滑，Transformer 常用 | 计算稍复杂 |

---

### 02. 反向传播算法

深度学习的核心训练算法。

**链式法则**：

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**计算图示例**：

```
输入 x ──► 线性变换 z=Wx+b ──► 激活 a=σ(z) ──► 损失 L
                │                    │              │
                ▼                    ▼              ▼
           ∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
```

**代码实现**：

```python
class Layer:
    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, grad_output):
        # 激活函数梯度
        grad_z = grad_output * self.activation_derivative(self.z)
        # 权重梯度
        self.grad_W = np.dot(self.x.T, grad_z)
        self.grad_b = np.sum(grad_z, axis=0)
        # 传递给上一层
        grad_input = np.dot(grad_z, self.W.T)
        return grad_input
```

---

### 03. 优化器

选择合适的优化算法加速训练。

| 优化器 | 更新规则 | 特点 |
|:-------|:---------|:-----|
| SGD | $\theta = \theta - \eta \nabla L$ | 简单，需调学习率 |
| Momentum | $v = \gamma v + \eta \nabla L$ | 加速收敛 |
| Adam | 自适应学习率 + 动量 | 最常用，鲁棒 |
| AdamW | Adam + 权重衰减解耦 | Transformer 首选 |

**Adam 更新公式**：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

### 04. 正则化技术

防止过拟合，提升泛化能力。

| 技术 | 原理 | 使用场景 |
|:-----|:-----|:---------|
| **Dropout** | 随机丢弃神经元 | 全连接层 |
| **BatchNorm** | 标准化中间层输出 | CNN, MLP |
| **LayerNorm** | 对单样本标准化 | Transformer |
| **L2 正则化** | 权重衰减 | 所有模型 |

**Dropout 实现**：

```python
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    mask = np.random.binomial(1, 1-p, size=x.shape) / (1-p)
    return x * mask
```

**BatchNorm 公式**：

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

---

### 05. 权重初始化

正确的初始化对训练至关重要。

| 方法 | 公式 | 适用激活函数 |
|:-----|:-----|:-------------|
| Xavier | $W \sim \mathcal{N}(0, \frac{2}{n_{in}+n_{out}})$ | Sigmoid, Tanh |
| He | $W \sim \mathcal{N}(0, \frac{2}{n_{in}})$ | ReLU |
| Orthogonal | 正交矩阵 | RNN |

---

### 06. 自定义模型 (Keras/PyTorch)

使用框架构建灵活的模型架构。

::: code-group

```python [Keras]
import tensorflow as tf
from tensorflow import keras

class CustomModel(keras.Model):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.dense1 = keras.layers.Dense(hidden_dim, activation='relu')
        self.bn = keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        return self.dense2(x)
```

```python [PyTorch]
import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.dense1 = nn.Linear(784, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = self.bn(x)
        x = self.dropout(x)
        return self.dense2(x)
```

:::

---

## 实验列表

| 实验 | 内容 | 文件 |
|:-----|:-----|:-----|
| 感知器实现 | 从零实现感知器 | `01_perceptron.ipynb` |
| MLP 手写 | NumPy 实现多层感知器 | `02_mlp_from_scratch.ipynb` |
| 反向传播 | 手动计算梯度 | `03_backpropagation.ipynb` |
| 优化器对比 | SGD vs Adam vs AdamW | `04_optimizers.ipynb` |
| 正则化实验 | Dropout, BatchNorm 效果 | `05_regularization.ipynb` |
| Keras 入门 | Sequential 和 Functional API | `06_keras_intro.ipynb` |
| 自定义层 | 实现自定义 Layer | `07_custom_layers.ipynb` |
| 自定义训练 | 自定义训练循环 | `08_custom_training.ipynb` |

---

## 参考资源

### 教材
- Goodfellow et al. (2016). *Deep Learning* - [在线阅读](https://www.deeplearningbook.org/)
- Nielsen, M. (2015). *Neural Networks and Deep Learning* - [在线阅读](http://neuralnetworksanddeeplearning.com/)

### 论文
- Kingma & Ba (2014). Adam: A Method for Stochastic Optimization
- Ioffe & Szegedy (2015). Batch Normalization
- Srivastava et al. (2014). Dropout

### 视频课程
- [3Blue1Brown - 神经网络](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Stanford CS231n](http://cs231n.stanford.edu/)
