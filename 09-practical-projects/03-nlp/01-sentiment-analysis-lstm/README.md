# LSTM情感分析 - 入门级项目

**难度**: ⭐⭐☆☆☆ (入门)

## 📋 项目简介

本项目使用LSTM（长短期记忆网络）对电影评论进行情感分析，判断评论是正面还是负面。这是学习LSTM的最佳入门项目，你将深入理解LSTM的工作原理和应用方法。

### 🎯 学习目标

- ✅ 深入理解LSTM的原理和门控机制
- ✅ 掌握文本预处理和词嵌入技术
- ✅ 学会构建和训练LSTM模型
- ✅ 理解为什么每一层这样设计
- ✅ 掌握模型评估和优化方法

## 🧠 为什么使用LSTM？

### 问题背景

情感分析需要理解句子的上下文关系，例如：
- "这部电影**不是**很好" - 负面（需要理解"不是"的否定作用）
- "虽然开头无聊，**但是**结局很精彩" - 正面（需要理解转折关系）

### 为什么不用简单RNN？

**简单RNN的问题**：
```
输入: "这部电影虽然开头很无聊但是结局非常精彩"
      ↓
简单RNN: 处理到"结局"时，已经忘记了"虽然"
      ↓
结果: 可能误判为负面
```

**LSTM的优势**：
```
输入: "这部电影虽然开头很无聊但是结局非常精彩"
      ↓
LSTM: 通过记忆单元保存"虽然...但是"的转折关系
      ↓
结果: 正确判断为正面
```

## 🏗️ LSTM原理详解

### LSTM的三个门

LSTM通过三个"门"来控制信息的流动：

#### 1. 遗忘门 (Forget Gate)
**作用**：决定丢弃哪些旧信息

```python
# 例子：处理 "这部电影很好，但是演员表演很差"
# 当读到"但是"时，遗忘门会：
# - 降低"很好"的权重（因为要转折了）
# - 为新的负面信息腾出空间
```

**公式**：`f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`
- 输出0-1之间的值
- 接近0：忘记这个信息
- 接近1：保留这个信息

#### 2. 输入门 (Input Gate)
**作用**：决定存储哪些新信息

```python
# 例子：处理 "这部电影很好，但是演员表演很差"
# 当读到"很差"时，输入门会：
# - 决定要存储这个负面评价
# - 更新记忆单元的状态
```

**公式**：
- `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)` - 决定更新多少
- `C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)` - 候选值

#### 3. 输出门 (Output Gate)
**作用**：决定输出哪些信息

```python
# 例子：在句子末尾
# 输出门会综合所有信息：
# - "很好"的正面信息（权重较低）
# - "但是"的转折信号
# - "很差"的负面信息（权重较高）
# 最终输出：负面情感
```

**公式**：`o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`

### LSTM vs 简单RNN对比

| 特性 | 简单RNN | LSTM |
|-----|---------|------|
| 记忆能力 | 短期（几个词） | 长期（整个句子） |
| 梯度消失 | 严重 | 有效缓解 |
| 训练难度 | 简单 | 较复杂 |
| 参数量 | 少 | 多（4倍） |
| 适用场景 | 简单序列 | 复杂上下文 |

## 📊 数据集

**IMDB电影评论数据集**
- 训练集：25,000条评论
- 测试集：25,000条评论
- 类别：正面(1) / 负面(0)
- 平均长度：约230个词

**数据示例**：
```
正面评论: "This movie is absolutely fantastic! The plot is engaging..."
负面评论: "Waste of time. The acting was terrible and the story made no sense..."
```

## 🏗️ 模型架构详解

### 整体架构

```
输入文本 → 词嵌入层 → LSTM层 → 全连接层 → Dropout → 输出层
  ↓           ↓          ↓         ↓          ↓        ↓
"好电影"   [0.2,...]  [0.5,...]  [0.8,...]  [0.7,..] 0.95(正面)
```

### 逐层详解

#### 第1层：Embedding（词嵌入层）
```python
layers.Embedding(max_words, 128, input_length=max_len)
```

**是什么**：将词转换为稠密向量
**做什么**：
```
"好" → [0.2, 0.5, -0.3, 0.8, ...]  (128维向量)
"电影" → [0.1, -0.2, 0.6, 0.4, ...]
```

**为什么这样设计**：
- `max_words=10000`：词汇表大小，覆盖最常用的10000个词
  - 太小（如1000）：很多词会被标记为"未知"
  - 太大（如50000）：稀有词太多，训练效果差
  - 10000：经验最佳值，平衡覆盖率和效果

- `embedding_dim=128`：每个词的向量维度
  - 太小（如32）：表达能力不足，无法捕获词的细微差别
  - 太大（如512）：参数过多，容易过拟合
  - 128：适中的维度，足够表达词义

- `input_length=200`：固定序列长度
  - 短于200的评论：填充0
  - 长于200的评论：截断
  - 200：覆盖大部分评论，同时控制计算量

#### 第2层：LSTM（长短期记忆层）
```python
layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)
```

**是什么**：处理序列数据的核心层
**做什么**：
```
输入序列: [词1, 词2, 词3, ..., 词200]
         ↓
LSTM处理: 每个词都会更新记忆单元
         ↓
输出: 一个128维的向量（整个句子的语义表示）
```

**为什么这样设计**：
- `units=128`：LSTM单元数量
  - 太小（如32）：记忆容量不足，无法捕获复杂模式
  - 太大（如512）：计算量大，容易过拟合
  - 128：平衡记忆能力和计算效率
  - 与Embedding维度相同：信息流动更顺畅

- `dropout=0.2`：输入dropout
  - 作用：随机丢弃20%的输入连接
  - 为什么：防止模型过度依赖某些特定词
  - 20%：经验值，不会太影响训练

- `recurrent_dropout=0.2`：循环dropout
  - 作用：在时间步之间应用dropout
  - 为什么：防止在序列处理中过拟合
  - 注意：不能太大，否则会破坏时序信息

- `return_sequences=False`：只返回最后一个时间步
  - False：返回最后的输出（用于分类）
  - True：返回所有时间步（用于序列标注）

#### 第3层：Dense（全连接层）
```python
layers.Dense(64, activation='relu')
```

**是什么**：特征组合层
**做什么**：
```
LSTM输出(128维) → 组合特征 → 压缩到64维
```

**为什么这样设计**：
- `units=64`：神经元数量
  - 作用：将LSTM的128维输出压缩到64维
  - 为什么压缩：
    1. 降维可以去除冗余信息
    2. 减少过拟合风险
    3. 为最终分类做准备
  - 64：LSTM输出的一半，适度压缩

- `activation='relu'`：ReLU激活函数
  - 作用：引入非线性，过滤负值
  - 为什么用ReLU：
    1. 计算简单：max(0, x)
    2. 缓解梯度消失
    3. 训练速度快
  - 替代方案：tanh（输出范围-1到1，但训练较慢）

#### 第4层：Dropout（正则化层）
```python
layers.Dropout(0.5)
```

**是什么**：随机丢弃神经元
**做什么**：
```
训练时: 随机关闭50%的神经元
测试时: 使用所有神经元（输出乘以0.5）
```

**为什么这样设计**：
- `rate=0.5`：丢弃率50%
  - 为什么这么高：
    1. 全连接层容易过拟合
    2. 强制网络学习鲁棒特征
    3. 相当于训练多个子网络的集成
  - 为什么不用在LSTM层：
    - LSTM层已经有dropout和recurrent_dropout
    - 过多dropout会破坏时序信息

#### 第5层：Output（输出层）
```python
layers.Dense(1, activation='sigmoid')
```

**是什么**：二分类输出层
**做什么**：
```
输入(64维) → 输出一个0-1之间的概率
0.8 → 正面情感（80%置信度）
0.2 → 负面情感（80%置信度）
```

**为什么这样设计**：
- `units=1`：单个输出神经元
  - 为什么是1：二分类问题，一个概率值就够了
  - 如果是多分类：需要units=类别数

- `activation='sigmoid'`：Sigmoid激活函数
  - 作用：将任意实数映射到0-1之间
  - 公式：σ(x) = 1 / (1 + e^(-x))
  - 为什么用sigmoid：
    1. 输出可以解释为概率
    2. 适合二分类问题
    3. 配合binary_crossentropy损失函数
  - 如果是多分类：用softmax

### 完整模型总结

```python
模型参数统计：
- Embedding层：10000 * 128 = 1,280,000 参数
- LSTM层：4 * (128 * 128 + 128 * 128 + 128) = 131,584 参数
  (4倍是因为：遗忘门 + 输入门 + 输出门 + 候选值)
- Dense层：128 * 64 + 64 = 8,256 参数
- Output层：64 * 1 + 1 = 65 参数
- 总计：约 1,420,000 参数

信息流动：
文本(200词) → Embedding(200×128) → LSTM(128) → Dense(64) → Dropout → Output(1)
```

## 📁 项目结构

```
01_情感分析_LSTM入门/
├── README.md                          # 本文件
├── requirements.txt                   # 项目依赖
│
├── notebooks/                         # Jupyter notebooks
│   ├── 00_LSTM原理详解.ipynb         # ⭐ LSTM原理和门控机制
│   ├── 01_数据探索.ipynb             # 数据加载和探索
│   ├── 02_数据预处理.ipynb           # 文本预处理和词嵌入
│   ├── 03_简单LSTM模型.ipynb         # 单层LSTM模型
│   ├── 04_改进LSTM模型.ipynb         # 双向LSTM和堆叠LSTM
│   ├── 05_模型评估.ipynb             # 详细的模型评估
│   └── 06_模型优化.ipynb             # 超参数调优
│
├── src/                               # 源代码
│   ├── __init__.py
│   ├── data.py                        # 数据加载和预处理
│   ├── model.py                       # 模型定义（带详细注释）
│   ├── train.py                       # 训练脚本
│   ├── evaluate.py                    # 评估脚本
│   └── utils.py                       # 工具函数
│
├── data/                              # 数据目录
│   ├── download_data.py               # 自动下载数据
│   └── README.md                      # 数据说明
│
├── models/                            # 保存的模型
│   └── .gitkeep
│
└── results/                           # 结果和图表
    └── .gitkeep
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd 实战项目/03_自然语言处理项目/01_情感分析_LSTM入门
pip install -r requirements.txt
```

### 2. 下载数据

```bash
cd data
python download_data.py
```

### 3. 运行notebooks

按顺序运行notebooks目录下的文件：
1. `00_LSTM原理详解.ipynb` - 理解LSTM原理
2. `01_数据探索.ipynb` - 了解数据
3. `02_数据预处理.ipynb` - 预处理数据
4. `03_简单LSTM模型.ipynb` - 构建第一个模型
5. `04_改进LSTM模型.ipynb` - 改进模型
6. `05_模型评估.ipynb` - 评估模型
7. `06_模型优化.ipynb` - 优化模型

### 4. 或者直接运行脚本

```bash
# 训练模型
python src/train.py --model_type lstm --epochs 10

# 评估模型
python src/evaluate.py --model_path models/lstm_best.h5
```

## 📈 预期结果

### 性能指标

| 模型 | 准确率 | 训练时间 | 参数量 |
|-----|--------|---------|--------|
| 简单LSTM | ~87% | 5分钟 | 1.4M |
| BiLSTM | ~89% | 8分钟 | 2.8M |
| 堆叠LSTM | ~88% | 10分钟 | 2.1M |

### 学习曲线

训练过程中你会看到：
- 前3个epoch：快速提升（70% → 85%）
- 3-7个epoch：缓慢提升（85% → 87%）
- 7个epoch后：趋于稳定

## 🎓 学习要点

### 1. LSTM的核心概念

- ✅ 理解三个门的作用（遗忘门、输入门、输出门）
- ✅ 理解记忆单元如何保存长期信息
- ✅ 理解LSTM如何解决梯度消失问题

### 2. 模型设计原则

- ✅ 为什么选择128维的Embedding
- ✅ 为什么LSTM单元数是128
- ✅ 为什么使用dropout和recurrent_dropout
- ✅ 为什么最后用sigmoid而不是softmax

### 3. 常见问题

**Q1: 为什么不使用更深的LSTM？**
A: 情感分析是相对简单的任务，单层LSTM足够。更深的网络：
- 优点：理论上能学习更复杂的模式
- 缺点：训练时间长，容易过拟合，梯度消失风险增加

**Q2: 为什么不使用预训练词向量（如Word2Vec、GloVe）？**
A: 本项目从零训练Embedding是为了学习基础。实际项目中：
- 数据少：用预训练词向量
- 数据多：从零训练或微调预训练向量

**Q3: BiLSTM比单向LSTM好在哪里？**
A: BiLSTM同时看前后文：
```
单向LSTM: "不" → "好" (只看到"不"，可能误判)
BiLSTM:   "不" ← "好" → "看" (同时看到"不好看"，判断更准确)
```

**Q4: 如何选择序列长度？**
A: 分析数据分布：
```python
# 查看评论长度分布
lengths = [len(review.split()) for review in reviews]
plt.hist(lengths, bins=50)
# 选择覆盖80-90%数据的长度
```

## 🔧 进阶优化

### 1. 使用预训练词向量

```python
# 加载GloVe词向量
embedding_matrix = load_glove_embeddings()
layers.Embedding(max_words, 128,
                weights=[embedding_matrix],
                trainable=False)  # 冻结预训练权重
```

### 2. 使用注意力机制

```python
# 添加注意力层
lstm_output = layers.LSTM(128, return_sequences=True)(embedded)
attention = layers.Attention()([lstm_output, lstm_output])
```

### 3. 使用双向LSTM

```python
# 双向LSTM
layers.Bidirectional(layers.LSTM(64))(embedded)
```

## 📚 参考资料

### 论文
- [LSTM原始论文](http://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter & Schmidhuber, 1997
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Christopher Olah

### 教程
- [Keras LSTM文档](https://keras.io/api/layers/recurrent_layers/lstm/)
- [TensorFlow文本分类教程](https://www.tensorflow.org/tutorials/text/text_classification_rnn)

### 数据集
- [IMDB数据集](https://ai.stanford.edu/~amaas/data/sentiment/)

## 🎯 下一步

完成本项目后，你可以：
1. 尝试中级项目：**温度预测**（多变量时间序列）
2. 学习更高级的模型：**Transformer**（注意力机制）
3. 尝试其他NLP任务：**命名实体识别**、**机器翻译**

---

**项目作者**: AI-Practices
**难度等级**: ⭐⭐☆☆☆ (入门)
**预计学习时间**: 1-2周
**前置知识**: Python基础、神经网络基础

祝学习愉快！🎉
