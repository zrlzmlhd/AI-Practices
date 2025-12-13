# 04 - 序列模型

从 RNN 到 Transformer，掌握序列建模的核心技术与 NLP 应用。

## 模块概览

| 属性 | 值 |
|:-----|:---|
| **前置要求** | 02-神经网络, 线性代数, 概率论 |
| **学习时长** | 3-4 周 |
| **Notebooks** | 18+ |
| **难度** | ⭐⭐⭐ 中级到高级 |

## 学习目标

完成本模块后，你将能够：

- ✅ 理解 RNN、LSTM、GRU 的工作原理和梯度问题
- ✅ 掌握注意力机制的数学原理和实现
- ✅ 深入理解 Transformer 架构的每个组件
- ✅ 了解 BERT、GPT 等预训练模型的设计思想
- ✅ 应用序列模型解决 NLP 实际问题

---

## 子模块详解

### 01. 循环神经网络 (RNN)

处理序列数据的基础架构。

**RNN 基本结构**：

```
x₁ ──► [h₁] ──► x₂ ──► [h₂] ──► x₃ ──► [h₃] ──► ... ──► yₜ
         │              │              │
         ▼              ▼              ▼
        h₁            h₂            h₃
```

**核心公式**：

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$

$$y_t = W_{hy}h_t + b_y$$

**梯度消失/爆炸问题**：

| 问题 | 原因 | 解决方案 |
|:-----|:-----|:---------|
| 梯度消失 | 连乘导致梯度趋近于0 | LSTM/GRU 门控机制 |
| 梯度爆炸 | 连乘导致梯度过大 | 梯度裁剪 (Gradient Clipping) |

---

### 02. LSTM 与 GRU

通过门控机制解决长期依赖问题。

**LSTM 架构**：

```
        ┌─────────────────────────────────────────┐
        │                Cell State               │
        │    ─────────────────────────────────    │
        │         ×           +           ×       │
        │         │           │           │       │
        │    ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ │
        │    │ 遗忘门 f │ │ 输入门 i │ │ 输出门 o │ │
        │    └────┬────┘ └────┬────┘ └────┬────┘ │
        │         │           │           │       │
        └─────────┴───────────┴───────────┴───────┘
                              │
                           [h_t]
```

**LSTM 公式**：

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$ (遗忘门)

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$ (输入门)

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$ (候选值)

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$ (细胞状态)

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$ (输出门)

$$h_t = o_t \odot \tanh(C_t)$$ (隐藏状态)

**GRU vs LSTM 对比**：

| 特性 | LSTM | GRU |
|:-----|:-----|:----|
| 门数量 | 3 (遗忘、输入、输出) | 2 (重置、更新) |
| 参数量 | 较多 | 较少 |
| 训练速度 | 较慢 | 较快 |
| 长序列性能 | 更好 | 稍弱 |

**PyTorch 实现**：

```python
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embed = self.embedding(x)
        output, (hidden, cell) = self.lstm(embed)
        # 取最后时刻的输出
        out = self.fc(output[:, -1, :])
        return out
```

---

### 03. 注意力机制

让模型关注输入序列的重要部分。

**注意力的直觉**：

```
Query: "它"指代什么？

Keys/Values: [小猫, 在, 沙发, 上, 睡觉, 它, 很, 可爱]
                ↑                              ↑
            高注意力                        当前位置

Attention Weights: [0.6, 0.05, 0.1, 0.05, 0.1, 0.0, 0.05, 0.05]
```

**Scaled Dot-Product Attention**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $\sqrt{d_k}$ 用于缩放，防止点积过大导致 softmax 梯度消失。

**注意力类型**：

| 类型 | 描述 | 应用 |
|:-----|:-----|:-----|
| Self-Attention | Q=K=V 来自同一序列 | Transformer |
| Cross-Attention | Q 来自解码器，K/V 来自编码器 | Seq2Seq |
| Multi-Head | 多组并行注意力 | Transformer |
| Masked | 遮蔽未来位置 | GPT 解码器 |

**代码实现**：

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

---

### 04. Transformer 架构

"Attention Is All You Need" - 革命性的序列建模架构。

**整体架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                        Transformer                          │
├────────────────────────┬────────────────────────────────────┤
│       Encoder          │           Decoder                  │
│                        │                                    │
│  ┌──────────────────┐  │  ┌──────────────────────────────┐ │
│  │ Multi-Head Attn  │  │  │ Masked Multi-Head Attention  │ │
│  │       ↓          │  │  │            ↓                 │ │
│  │ Add & Norm       │  │  │ Add & Norm                   │ │
│  │       ↓          │  │  │            ↓                 │ │
│  │ Feed Forward     │  │  │ Cross Multi-Head Attention   │ │
│  │       ↓          │  │  │            ↓                 │ │
│  │ Add & Norm       │  │  │ Add & Norm                   │ │
│  └──────────────────┘  │  │            ↓                 │ │
│         × N            │  │ Feed Forward                 │ │
│                        │  │            ↓                 │ │
│                        │  │ Add & Norm                   │ │
│                        │  └──────────────────────────────┘ │
│                        │            × N                    │
└────────────────────────┴────────────────────────────────────┘
```

**核心组件**：

| 组件 | 功能 | 公式/说明 |
|:-----|:-----|:---------|
| Multi-Head Attention | 多视角注意力 | $\text{Concat}(head_1, ..., head_h)W^O$ |
| Position Encoding | 位置信息注入 | $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$ |
| Feed Forward | 非线性变换 | $\text{FFN}(x) = \text{ReLU}(xW_1+b_1)W_2+b_2$ |
| Layer Norm | 层归一化 | 稳定训练 |
| Residual Connection | 残差连接 | 缓解梯度消失 |

**Multi-Head Attention**：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O$$

$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**PyTorch 实现**：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性变换并分头
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(attn_output)
```

---

### 05. 预训练语言模型

大规模预训练 + 下游任务微调范式。

**模型对比**：

| 模型 | 架构 | 预训练任务 | 特点 |
|:-----|:-----|:-----------|:-----|
| **BERT** | Encoder-only | MLM + NSP | 双向理解 |
| **GPT** | Decoder-only | CLM | 自回归生成 |
| **T5** | Encoder-Decoder | Span Corruption | 统一文本到文本 |
| **RoBERTa** | Encoder-only | MLM (优化) | 更大数据，更长训练 |
| **DeBERTa** | Encoder-only | MLM + RTD | 解耦注意力 |

**BERT 预训练任务**：

```
Masked Language Model (MLM):
输入: The [MASK] sat on the mat
输出: cat

Next Sentence Prediction (NSP):
句子A: The cat sat on the mat
句子B: It was a sunny day
标签: IsNext / NotNext
```

**使用 Transformers 库**：

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 编码输入
inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
```

---

### 06. Seq2Seq 与机器翻译

序列到序列的编码器-解码器架构。

**架构流程**：

```
源语言: "我 爱 你"
         │
         ▼
    ┌─────────┐
    │ Encoder │ ──► Context Vector
    └─────────┘
         │
         ▼
    ┌─────────┐
    │ Decoder │ ──► "I love you"
    └─────────┘
```

**训练技巧**：

| 技巧 | 描述 | 作用 |
|:-----|:-----|:-----|
| Teacher Forcing | 训练时用真实标签作为输入 | 加速收敛 |
| Scheduled Sampling | 逐渐减少 Teacher Forcing | 缓解曝光偏差 |
| Label Smoothing | 软化标签分布 | 提升泛化 |
| Beam Search | 解码时保留多个候选 | 提升生成质量 |

---

## 实验列表

| 实验 | 内容 | 文件 |
|:-----|:-----|:-----|
| RNN 基础 | 从零实现 RNN | `01_rnn_basics.ipynb` |
| LSTM 实现 | 手写 LSTM 单元 | `02_lstm_from_scratch.ipynb` |
| 文本分类 | LSTM 情感分析 | `03_lstm_sentiment.ipynb` |
| 注意力机制 | 可视化注意力权重 | `04_attention_visualization.ipynb` |
| Transformer | 从零实现 Transformer | `05_transformer_scratch.ipynb` |
| BERT 微调 | 文本分类任务 | `06_bert_finetuning.ipynb` |
| GPT 生成 | 文本生成实验 | `07_gpt_generation.ipynb` |
| 机器翻译 | Seq2Seq 翻译模型 | `08_seq2seq_translation.ipynb` |
| 命名实体识别 | BERT + CRF | `09_ner_bert_crf.ipynb` |

---

## 参考资源

### 教材
- Juerta & Martin (2023). *Speech and Language Processing* (3rd ed.) - [在线阅读](https://web.stanford.edu/~jurafsky/slp3/)
- Tunstall et al. (2022). *Natural Language Processing with Transformers*

### 论文
- Vaswani et al. (2017). Attention Is All You Need
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
- Radford et al. (2018). Improving Language Understanding by Generative Pre-Training (GPT)
- He et al. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention

### 视频课程
- [Stanford CS224N](http://web.stanford.edu/class/cs224n/) - Natural Language Processing with Deep Learning
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Course](https://huggingface.co/course)
