# Module 04: Sequence Models

**序列模型与 NLP** - 掌握序列数据建模技术，从传统 RNN 到现代 Transformer。

---

## Module Overview

| Attribute | Value |
|:----------|:------|
| **Prerequisites** | Module 02 |
| **Duration** | 3-4 weeks |
| **Notebooks** | 16+ |
| **Difficulty** | ⭐⭐⭐ Intermediate |

---

## Learning Objectives

- ✅ 理解循环神经网络的原理和梯度问题
- ✅ 掌握 LSTM、GRU 的门控机制
- ✅ 应用词嵌入和文本预处理技术
- ✅ 理解 Attention 机制和 Transformer 架构

---

## Submodules

### 01. RNN Basics

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| Recurrent Networks | Hidden State | `01_rnn_basics.ipynb` |
| BPTT | Backprop Through Time | `02_bptt.ipynb` |
| Vanishing Gradient | Long-term Dependencies | `03_vanishing_gradient.ipynb` |

**Key Equations:**
```
RNN:
hₜ = tanh(Wₓₕ xₜ + Wₕₕ hₜ₋₁ + bₕ)
yₜ = Wₕᵧ hₜ + bᵧ
```

### 02. LSTM & GRU

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| LSTM | Forget/Input/Output Gates | `01_lstm.ipynb` |
| GRU | Reset/Update Gates | `02_gru.ipynb` |
| Bidirectional | Forward + Backward | `03_bidirectional.ipynb` |
| Stacked RNN | Deep Sequence Models | `04_stacked.ipynb` |

**LSTM Equations:**
```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)     # Forget gate
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)     # Input gate
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)  # Candidate
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ         # Cell state
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)     # Output gate
hₜ = oₜ * tanh(Cₜ)               # Hidden state
```

### 03. Text Processing

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| Tokenization | Word, Subword, BPE | `01_tokenization.ipynb` |
| Word Embeddings | Word2Vec, GloVe | `02_embeddings.ipynb` |
| Pretrained Embeddings | FastText | `03_pretrained.ipynb` |
| Text Classification | Sentiment Analysis | `04_classification.ipynb` |

### 04. Attention & Transformer

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| Attention Mechanism | Query, Key, Value | `01_attention.ipynb` |
| Self-Attention | Scaled Dot-Product | `02_self_attention.ipynb` |
| Multi-Head Attention | Parallel Attention | `03_multi_head.ipynb` |
| Transformer | Encoder-Decoder | `04_transformer.ipynb` |
| BERT | Bidirectional Encoding | `05_bert.ipynb` |
| GPT | Autoregressive | `06_gpt.ipynb` |

**Attention Equations:**
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

Multi-Head:
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) Wᴼ
headᵢ = Attention(Q Wᵢᵠ, K Wᵢᴷ, V Wᵢⱽ)
```

---

## Model Comparison

| Model | Type | Params | Use Case |
|:------|:-----|:------:|:---------|
| LSTM | RNN | ~1M | Sequential data |
| GRU | RNN | ~0.75M | Efficient RNN |
| Transformer | Attention | ~65M | Parallel training |
| BERT-base | Encoder | 110M | Understanding |
| GPT-2 | Decoder | 117M-1.5B | Generation |
| T5 | Encoder-Decoder | 220M-11B | Seq2Seq |

---

<div align="center">

**[[← Module 03|Module-03:-Computer-Vision]]** | **[[Module 05 →|Module-05:-Advanced-Topics]]**

</div>
