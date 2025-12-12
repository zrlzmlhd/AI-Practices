# Transformer文本分类 - 入门级项目

**难度**: ⭐⭐⭐☆☆ (中级偏入门)

## 📋 项目简介

本项目从零实现Transformer进行文本分类，深入理解注意力机制的工作原理。这是学习Transformer的最佳入门项目，你将理解为什么Transformer革命性地改变了NLP领域。

### 🎯 学习目标

- ✅ 深入理解Self-Attention（自注意力）机制
- ✅ 掌握Multi-Head Attention（多头注意力）
- ✅ 理解Positional Encoding（位置编码）
- ✅ 学习Transformer Encoder的完整实现
- ✅ 理解为什么Transformer比LSTM更强

## 🧠 为什么使用Transformer？

### LSTM的局限

```
LSTM处理: "这部电影虽然开头很无聊但是结局非常精彩"
         ↓
问题1: 串行处理，无法并行
      这 → 部 → 电影 → ... → 精彩 (慢)

问题2: 长距离依赖衰减
      处理到"精彩"时，"虽然"的信息已经很弱

问题3: 固定的记忆容量
      LSTM单元数固定，记忆有限
```

### Transformer的优势

```
Transformer处理: "这部电影虽然开头很无聊但是结局非常精彩"
               ↓
优势1: 并行处理
      所有词同时处理 (快10倍)

优势2: 直接建模长距离依赖
      "精彩"可以直接关注"虽然"
      不需要通过中间词传递

优势3: 动态的注意力权重
      自动学习哪些词重要
```

## 🏗️ Transformer核心原理详解

### 1. Self-Attention（自注意力）机制

**核心思想**：让每个词关注句子中的所有词

#### 计算过程

**步骤1：生成Q、K、V**
```python
# 输入: 词嵌入向量
Word = [0.2, 0.5, -0.3, 0.8, ...]  # 维度: d_model=512

# 通过三个权重矩阵变换
Query (Q) = Word × W_Q  # 查询：我要找什么
Key (K)   = Word × W_K  # 键：我是什么
Value (V) = Word × W_V  # 值：我的内容是什么
```

**为什么需要Q、K、V？**
```
类比图书馆:
- Query: 你要找的书（搜索词）
- Key: 书的标签（索引）
- Value: 书的内容

过程:
1. 用Query搜索所有Key（计算相似度）
2. 找到最相关的Key
3. 返回对应的Value
```

**步骤2：计算注意力分数**
```python
# 计算Query和所有Key的相似度
Score = Q · K^T / sqrt(d_k)

# 例子: "电影很好"
#       电影  很  好
# 电影  [1.0, 0.3, 0.5]  # 电影最关注自己
# 很    [0.2, 1.0, 0.8]  # 很最关注"好"
# 好    [0.3, 0.7, 1.0]  # 好关注"很"和自己
```

**为什么除以sqrt(d_k)？**
```
问题: d_k很大时，点积值很大
     ↓
结果: softmax后梯度很小（梯度消失）
     ↓
解决: 除以sqrt(d_k)进行缩放
```

**步骤3：Softmax归一化**
```python
Attention_Weights = softmax(Score)

# 例子: "电影很好"中"很"的注意力权重
Score:   [0.2, 1.0, 0.8]
        ↓ softmax
Weights: [0.09, 0.49, 0.42]  # 和为1
```

**步骤4：加权求和**
```python
Output = Attention_Weights · V

# 例子: "很"的输出
Output = 0.09*V_电影 + 0.49*V_很 + 0.42*V_好
       = 综合了三个词的信息，重点关注"很"和"好"
```

### 2. Multi-Head Attention（多头注意力）

**为什么需要多头？**

**单头注意力的局限**：
```
单头: 只能学习一种关系
     例如: 只关注语义相似的词
```

**多头注意力的优势**：
```
8个头，每个头学习不同的关系:
- Head 1: 关注语义相似（"好"关注"棒"）
- Head 2: 关注语法关系（动词关注主语）
- Head 3: 关注位置关系（关注相邻词）
- Head 4: 关注情感词（"好"关注"很"）
- ... 等8个不同的关注模式
```

**实现方式**：
```python
# 将d_model=512分成8个头，每个头64维
d_model = 512
num_heads = 8
d_k = d_model // num_heads = 64

# 每个头独立计算注意力
for i in range(num_heads):
    Q_i = X · W_Q_i  # (batch, seq_len, 64)
    K_i = X · W_K_i
    V_i = X · W_V_i

    head_i = Attention(Q_i, K_i, V_i)

# 拼接所有头
MultiHead = Concat(head_1, ..., head_8)  # (batch, seq_len, 512)
Output = MultiHead · W_O
```

### 3. Positional Encoding（位置编码）

**为什么需要位置编码？**

```
问题: Self-Attention是无序的
     "我爱你" 和 "你爱我" 的注意力计算结果相同
     ↓
解决: 添加位置信息
```

**位置编码公式**：
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# pos: 词的位置 (0, 1, 2, ...)
# i: 维度索引 (0, 1, 2, ..., d_model/2)
```

**为什么用sin/cos？**
```
优点1: 值域在[-1, 1]，不会太大
优点2: 不同位置的编码不同
优点3: 相对位置关系可以通过三角函数计算
      PE(pos+k) 可以表示为 PE(pos) 的线性组合
```

### 4. Transformer Encoder完整架构

```
输入文本 → Token Embedding → + Positional Encoding
                                    ↓
                          Multi-Head Attention
                                    ↓
                          Add & Norm (残差连接+层归一化)
                                    ↓
                          Feed Forward Network
                                    ↓
                          Add & Norm
                                    ↓
                          [重复N次]
                                    ↓
                          Global Average Pooling
                                    ↓
                          Dense → Output
```

#### 逐层详解

**第1层：Embedding + Positional Encoding**
```python
# Token Embedding
token_emb = Embedding(vocab_size, d_model)  # (batch, seq_len, 512)

# Positional Encoding
pos_enc = PositionalEncoding(d_model)  # (seq_len, 512)

# 相加
x = token_emb + pos_enc
```

**为什么是相加而不是拼接？**
- 相加：保持维度不变，位置信息融入词向量
- 拼接：维度翻倍，计算量大

**第2层：Multi-Head Attention**
```python
# 自注意力
attn_output = MultiHeadAttention(
    query=x,
    key=x,
    value=x,
    num_heads=8
)
```

**为什么Q、K、V都是x？**
- Self-Attention：自己关注自己
- 每个词既是查询者，也是被查询者

**第3层：Add & Norm（残差连接+层归一化）**
```python
# 残差连接
x = x + attn_output

# 层归一化
x = LayerNorm(x)
```

**为什么需要残差连接？**
```
问题: 深层网络梯度消失
     ↓
解决: 残差连接提供梯度直通路径
     ↓
效果: 可以堆叠更多层（BERT用12层）
```

**为什么用LayerNorm而不是BatchNorm？**
```
BatchNorm: 对batch维度归一化
          问题: 序列长度不同时效果差

LayerNorm: 对特征维度归一化
          优点: 不受序列长度影响
```

**第4层：Feed Forward Network**
```python
# 两层全连接网络
ffn = Dense(d_ff, activation='relu')(x)  # d_ff=2048
ffn = Dense(d_model)(ffn)  # 512

# 残差连接 + 层归一化
x = LayerNorm(x + ffn)
```

**为什么需要FFN？**
- Attention：建模词之间的关系
- FFN：对每个词独立进行非线性变换
- 作用：增强模型的表达能力

**为什么d_ff=2048（4倍d_model）？**
- 经验值：扩大再压缩
- 类似瓶颈结构：512 → 2048 → 512

## 📊 数据集

**IMDB电影评论数据集**：
- 训练集：25,000条评论
- 测试集：25,000条评论
- 类别：正面(1) / 负面(0)

## 📁 项目结构

```
02_Transformer文本分类_入门/
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 00_注意力机制原理.ipynb         # ⭐ Attention详解
│   ├── 01_Self_Attention实现.ipynb     # ⭐ 从零实现
│   ├── 02_Multi_Head_Attention.ipynb   # ⭐ 多头注意力
│   ├── 03_Positional_Encoding.ipynb    # ⭐ 位置编码
│   ├── 04_Transformer_Encoder.ipynb    # ⭐ 完整Encoder
│   ├── 05_模型训练.ipynb               # 训练和评估
│   ├── 06_注意力可视化.ipynb           # 可视化注意力权重
│   └── 07_BERT微调对比.ipynb           # 与预训练模型对比
│
├── src/
│   ├── __init__.py
│   ├── attention.py                     # ⭐ 注意力机制实现
│   ├── transformer.py                   # ⭐ Transformer实现
│   ├── model.py                         # 完整模型
│   └── train.py
│
├── data/
├── models/
└── results/
```

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行notebooks（按顺序学习）
jupyter notebook notebooks/

# 3. 训练模型
python src/train.py --d_model 512 --num_heads 8 --num_layers 6
```

## 📈 预期结果

| 模型 | 准确率 | 训练时间 | 参数量 |
|-----|--------|---------|--------|
| LSTM | 87% | 10分钟 | 1.4M |
| BiLSTM | 89% | 15分钟 | 2.8M |
| **Transformer (小)** | 88% | 8分钟 | 2.5M |
| **Transformer (大)** | **91%** | 20分钟 | 10M |
| **BERT (微调)** | **94%** | 5分钟 | 110M |

## 🎓 学习要点

### 1. Transformer vs LSTM

| 特性 | LSTM | Transformer |
|-----|------|-------------|
| **并行性** | 串行（慢） | 并行（快） |
| **长距离依赖** | 衰减 | 直接建模 |
| **计算复杂度** | O(n) | O(n²) |
| **内存占用** | 小 | 大 |
| **训练速度** | 慢 | 快 |
| **推理速度** | 快 | 慢（长序列） |

### 2. 注意力机制的关键点

**Q、K、V的作用**：
```
Query: 我要找什么信息
Key: 我有什么信息
Value: 信息的具体内容

类比搜索引擎:
Query: 用户搜索词
Key: 网页标题和摘要
Value: 网页完整内容
```

**Scaled Dot-Product的必要性**：
```
不缩放: 点积值很大 → softmax梯度小 → 训练困难
缩放: 除以sqrt(d_k) → 梯度正常 → 训练稳定
```

### 3. 常见问题

**Q1: 为什么Transformer比LSTM快？**
A:
```
LSTM: 必须串行处理
     词1 → 词2 → 词3 → ... → 词100
     无法并行

Transformer: 所有词同时处理
     词1, 词2, 词3, ..., 词100
     完全并行，利用GPU
```

**Q2: Transformer的计算复杂度是O(n²)，为什么还快？**
A:
```
短序列(n<512): O(n²)可接受，并行优势明显
长序列(n>1024): 确实慢，需要优化
              → Longformer, BigBird等改进
```

**Q3: 位置编码为什么不用学习的方式？**
A:
```
固定位置编码(sin/cos):
- 优点: 可以处理任意长度序列
- 缺点: 不够灵活

学习位置编码:
- 优点: 更灵活，效果可能更好
- 缺点: 只能处理训练时见过的长度
- BERT使用学习的位置编码
```

**Q4: 为什么需要多头注意力？**
A:
```
单头: 只能学习一种关系模式
多头: 学习多种关系模式

类比:
单头 = 只用一个角度看问题
多头 = 从多个角度看问题（更全面）
```

## 🔧 进阶优化

### 1. 预训练模型微调
```python
# 使用BERT预训练模型
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# 只需训练几个epoch，效果远超从零训练
```

### 2. 长序列优化
```python
# Longformer: 稀疏注意力
# 复杂度从O(n²)降到O(n)
from transformers import LongformerForSequenceClassification
```

### 3. 知识蒸馏
```python
# 用大模型(BERT)教小模型(Transformer)
# 保持性能，减少参数量
```

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化教程
- [Transformers from Scratch](https://peterbloem.nl/blog/transformers)

## 🎯 下一步

完成本项目后，可以尝试：
1. **中级项目**：命名实体识别（Token级别分类）
2. **高级项目**：机器翻译（Encoder-Decoder架构）
3. **预训练模型**：BERT、GPT微调

---

**难度等级**: ⭐⭐⭐☆☆ (中级偏入门)
**预计学习时间**: 2-3周
**前置知识**: 深度学习基础、注意力机制概念
**重要性**: ⭐⭐⭐⭐⭐ (现代NLP的基础)
