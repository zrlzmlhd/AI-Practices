# 激活函数与损失函数

深度学习核心组件的详细参考资料。

<div align="center">

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

</div>

---

## 🚀 快速开始

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation

# 激活函数使用
x = Dense(64, activation='relu')(inputs)          # ReLU
x = Dense(64, activation='leaky_relu')(x)         # Leaky ReLU
outputs = Dense(10, activation='softmax')(x)       # Softmax输出

# 损失函数使用
model.compile(
    loss='sparse_categorical_crossentropy',  # 多分类
    optimizer='adam',
    metrics=['accuracy']
)
```

## 📚 模块简介

本模块提供了深度学习中最重要的两个组件的详细说明：**激活函数**和**损失函数**。这些是构建神经网络的基础模块，理解它们对于设计和优化模型至关重要。

### 🎯 学习目标

- ✅ 理解各种激活函数的特点和适用场景
- ✅ 掌握不同损失函数的数学原理
- ✅ 学会为特定任务选择合适的函数
- ✅ 了解函数选择对模型性能的影响

## 📂 内容结构

```
激活函数与损失函数/
├── 常见激活函数及其图像/
│   └── 激活函数可视化.ipynb
└── 损失函数/
    └── 损失函数.md
```

## ⚡ 激活函数

激活函数为神经网络引入非线性，使其能够学习复杂的模式。

### 常见激活函数

#### 1. Sigmoid (σ)

**数学公式：**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**特点：**
- 输出范围：(0, 1)
- 平滑的S型曲线
- 可以解释为概率

**优点：**
- 输出有界
- 平滑可导

**缺点：**
- 梯度消失问题
- 输出不以零为中心
- 计算开销较大

**适用场景：**
- 二分类问题的输出层
- 需要概率输出的场景

---

#### 2. Tanh (双曲正切)

**数学公式：**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**特点：**
- 输出范围：(-1, 1)
- 零为中心
- 比Sigmoid更陡峭

**优点：**
- 零为中心，收敛更快
- 梯度更强

**缺点：**
- 仍有梯度消失问题
- 计算开销较大

**适用场景：**
- 隐藏层
- RNN/LSTM

---

#### 3. ReLU (Rectified Linear Unit)

**数学公式：**
$$\text{ReLU}(x) = \max(0, x)$$

**特点：**
- 最流行的激活函数
- 计算简单高效
- 不饱和

**优点：**
- 计算效率高
- 缓解梯度消失
- 加速收敛

**缺点：**
- 神经元"死亡"问题
- 输出不以零为中心

**适用场景：**
- CNN卷积层
- 全连接层
- 大多数深度网络

---

#### 4. Leaky ReLU

**数学公式：**
$$\text{Leaky ReLU}(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}$$

其中 α 通常是 0.01

**特点：**
- 解决ReLU的"死亡"问题
- 允许负值梯度

**优点：**
- 所有输入都有梯度
- 保留ReLU的优点

**适用场景：**
- 替代ReLU
- 深层网络

---

#### 5. ELU (Exponential Linear Unit)

**数学公式：**
$$\text{ELU}(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{if } x \leq 0
\end{cases}$$

**特点：**
- 负值输出
- 平滑性好

**优点：**
- 加速学习
- 输出接近零均值
- 避免神经元"死亡"

**缺点：**
- 计算开销较大

---

#### 6. Swish / SiLU

**数学公式：**
$$\text{Swish}(x) = x \cdot \sigma(x)$$

**特点：**
- Google提出的新激活函数
- 平滑非单调

**优点：**
- 某些任务上优于ReLU
- 自适应性强

---

#### 7. Softmax

**数学公式：**
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

**特点：**
- 多分类问题专用
- 输出和为1
- 可解释为概率分布

**适用场景：**
- 多分类问题的输出层

---

### 激活函数选择指南

| 层类型 | 推荐激活函数 | 原因 |
|-------|------------|------|
| 卷积层 | ReLU, Leaky ReLU | 计算效率高，性能好 |
| 全连接隐藏层 | ReLU, ELU | 训练快，效果好 |
| RNN/LSTM | Tanh, Sigmoid | RNN标准配置 |
| 二分类输出层 | Sigmoid | 输出概率 |
| 多分类输出层 | Softmax | 输出概率分布 |
| 回归输出层 | Linear (无激活) | 不限制输出范围 |
| GAN判别器 | Leaky ReLU | 避免梯度问题 |

---

## 📉 损失函数

损失函数衡量模型预测与真实值的差距，指导模型优化方向。

### 分类任务损失函数

#### 1. 二分类交叉熵 (Binary Cross-Entropy)

**数学公式：**
$$BCE = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**适用场景：**
- 二分类问题
- 多标签分类

**Keras实现：**
```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

**PyTorch实现：**
```python
criterion = nn.BCELoss()
# 或配合sigmoid
criterion = nn.BCEWithLogitsLoss()
```

---

#### 2. 多分类交叉熵 (Categorical Cross-Entropy)

**数学公式：**
$$CCE = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}y_{i,c}\log(\hat{y}_{i,c})$$

**适用场景：**
- 多分类问题（互斥类别）
- 类别需要one-hot编码

**Keras实现：**
```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

---

#### 3. 稀疏分类交叉熵 (Sparse Categorical Cross-Entropy)

**特点：**
- 与categorical_crossentropy相同
- 但标签是整数而非one-hot

**适用场景：**
- 多分类问题
- 标签是整数格式

**Keras实现：**
```python
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

---

### 回归任务损失函数

#### 1. 均方误差 (MSE - Mean Squared Error)

**数学公式：**
$$MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

**特点：**
- 最常用的回归损失
- 对异常值敏感
- 误差被平方放大

**适用场景：**
- 标准回归问题
- 预测连续值

**实现：**
```python
# Keras
model.compile(loss='mse', optimizer='adam')

# PyTorch
criterion = nn.MSELoss()
```

---

#### 2. 平均绝对误差 (MAE - Mean Absolute Error)

**数学公式：**
$$MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$$

**特点：**
- 对异常值更鲁棒
- 梯度恒定

**适用场景：**
- 有异常值的数据
- 需要鲁棒性

**实现：**
```python
# Keras
model.compile(loss='mae', optimizer='adam')

# PyTorch
criterion = nn.L1Loss()
```

---

#### 3. Huber损失

**数学公式：**
$$L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

**特点：**
- 结合MSE和MAE优点
- 小误差用MSE，大误差用MAE
- 对异常值鲁棒

**适用场景：**
- 有异常值的回归问题

**实现：**
```python
# Keras
model.compile(loss='huber', optimizer='adam')

# PyTorch
criterion = nn.SmoothL1Loss()
```

---

### 特殊任务损失函数

#### 1. Hinge Loss

**数学公式：**
$$L = \max(0, 1 - y \cdot \hat{y})$$

**适用场景：**
- SVM
- 二分类问题（标签为±1）

---

#### 2. KL散度 (Kullback-Leibler Divergence)

**数学公式：**
$$D_{KL}(P||Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)}$$

**适用场景：**
- VAE
- 分布匹配
- 知识蒸馏

---

#### 3. Focal Loss

**数学公式：**
$$FL = -\alpha_t(1-p_t)^\gamma \log(p_t)$$

**特点：**
- 解决类别不平衡
- 关注难分类样本

**适用场景：**
- 目标检测
- 类别不平衡问题

---

## 🎯 选择指南

### 根据任务类型选择

| 任务类型 | 输出激活函数 | 损失函数 |
|---------|------------|---------|
| 二分类 | Sigmoid | Binary Cross-Entropy |
| 多分类（互斥） | Softmax | Categorical Cross-Entropy |
| 多标签分类 | Sigmoid | Binary Cross-Entropy |
| 回归 | Linear | MSE / MAE |
| 序列生成 | Softmax | Sparse Categorical CE |

### 根据数据特点选择

| 数据特点 | 推荐损失函数 |
|---------|------------|
| 有异常值 | MAE, Huber |
| 类别不平衡 | Weighted CE, Focal Loss |
| 标准数据 | MSE, CE |

## 📊 可视化对比

查看notebook获取各激活函数的可视化对比：

- 函数曲线
- 导数曲线
- 输出分布
- 梯度流动

## 💡 最佳实践

### 激活函数

1. **默认选择ReLU**
   - 大多数情况下表现良好
   - 计算高效

2. **遇到问题时尝试变体**
   - 神经元"死亡" → Leaky ReLU, ELU
   - 需要负值 → Leaky ReLU, ELU
   - GAN → Leaky ReLU

3. **输出层要慎重**
   - 二分类 → Sigmoid
   - 多分类 → Softmax
   - 回归 → Linear

### 损失函数

1. **匹配任务类型**
   - 分类用交叉熵
   - 回归用MSE/MAE

2. **考虑数据特点**
   - 有异常值用MAE
   - 类别不平衡用加权损失

3. **可以组合使用**
   ```python
   total_loss = α * loss1 + β * loss2
   ```

## 🔍 调试技巧

### 激活函数问题

**症状：** 梯度消失
**解决：**
- 检查是否使用Sigmoid/Tanh在深层网络
- 尝试ReLU系列

**症状：** 神经元"死亡"
**解决：**
- 降低学习率
- 使用Leaky ReLU
- 检查初始化

### 损失函数问题

**症状：** 损失爆炸/NaN
**解决：**
- 检查数值稳定性
- 使用LogSoftmax + NLLLoss
- 梯度裁剪

**症状：** 训练不收敛
**解决：**
- 检查损失函数是否匹配任务
- 检查标签格式
- 调整学习率

## 📚 参考资料

### 优质学习资源

| 资源类型 | 名称 | 链接 |
|---------|------|------|
| GitHub | PyTorch激活函数实现 | [pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py) |
| GitHub | TensorFlow激活函数 | [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras/activations.py) |
| GitHub | 自定义损失函数示例 | [keras-team/keras-io](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/creating_tfrecords.py) |
| 官方文档 | TensorFlow激活函数 | [tensorflow.org/api_docs/python/tf/keras/activations](https://www.tensorflow.org/api_docs/python/tf/keras/activations) |
| 官方文档 | PyTorch损失函数 | [pytorch.org/docs/stable/nn](https://pytorch.org/docs/stable/nn.html#loss-functions) |

### 论文
- Glorot & Bengio (2010): Understanding the difficulty of training deep feedforward neural networks
- He et al. (2015): Delving Deep into Rectifiers (PReLU)
- Ramachandran et al. (2017): Searching for Activation Functions (Swish)

### 文档
- [TensorFlow激活函数](https://www.tensorflow.org/api_docs/python/tf/keras/activations)
- [PyTorch激活函数](https://pytorch.org/docs/stable/nn.html#non-linear-activations)
- [PyTorch损失函数](https://pytorch.org/docs/stable/nn.html#loss-functions)

### 博客
- [CS231n: Neural Networks Part 1](http://cs231n.github.io/neural-networks-1/)
- [Activation Functions Explained](https://mlfromscratch.com/activation-functions-explained/)

## 🔬 实验建议

1. **比较不同激活函数**
   - 在同一模型上测试
   - 记录收敛速度和最终性能

2. **可视化激活分布**
   - 使用TensorBoard
   - 检查是否有大量死亡神经元

3. **损失函数对比**
   - MSE vs MAE
   - 不同权重的组合

## 🤝 贡献

发现错误或想添加新内容？欢迎贡献！

查看[贡献指南](../CONTRIBUTING.md)

---

掌握激活函数和损失函数是深度学习的基础！

[返回主页](../README.md)
