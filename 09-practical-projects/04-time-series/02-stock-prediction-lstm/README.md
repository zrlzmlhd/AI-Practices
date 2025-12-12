# LSTM股票价格预测 - 高级项目

**难度**: ⭐⭐⭐⭐☆ (高级)

## 📋 项目简介

本项目使用高级LSTM技术进行股票价格预测，包括注意力机制、多任务学习、技术指标特征工程等。**注意：本项目仅供学习，不构成投资建议。**

### 🎯 学习目标

- ✅ 掌握金融时间序列的特征工程（技术指标）
- ✅ 学习LSTM + Attention注意力机制
- ✅ 掌握多任务学习（同时预测价格和趋势）
- ✅ 理解为什么使用注意力机制
- ✅ 学习模型集成和风险管理

## 🧠 为什么使用注意力机制？

### 问题背景

股票价格受多种因素影响：
- **历史价格**：过去的价格走势
- **技术指标**：MA、RSI、MACD等
- **关键事件**：财报发布、政策变化
- **市场情绪**：成交量、波动率

### 普通LSTM的局限

```
普通LSTM: 平等对待所有历史数据
         ↓
问题: 昨天的数据和30天前的数据权重相同
     ↓
结果: 无法突出关键时间点（如财报日）
```

### 注意力机制的优势

```
LSTM + Attention: 自动学习哪些时间点重要
                ↓
优势: 财报日权重高，普通日权重低
     ↓
结果: 预测更准确，可解释性更强
```

## 🏗️ 注意力机制原理详解

### 什么是注意力机制？

**核心思想**：让模型"关注"重要的信息

```python
# 例子：预测明天的股价
历史数据: [30天前, 29天前, ..., 昨天]
         ↓
注意力权重: [0.01, 0.01, ..., 0.3]  # 昨天最重要
         ↓
加权求和: 重点关注昨天和最近几天
         ↓
预测: 明天的股价
```

### 注意力机制的计算过程

#### 1. 计算注意力分数
```python
# Query: 当前要预测的时间点
# Key: 历史每个时间点
# 计算相似度
score = Query · Key^T / sqrt(d_k)
```

#### 2. 归一化为权重
```python
# Softmax归一化
attention_weights = softmax(score)
# 例如: [0.05, 0.08, 0.12, ..., 0.35]
```

#### 3. 加权求和
```python
# Value: 历史数据的表示
output = attention_weights · Value
# 重点关注权重高的时间点
```

### 为什么注意力机制有效？

**1. 自适应权重**
```
传统方法: 手动设定权重（如指数平滑）
注意力机制: 自动学习权重
```

**2. 可解释性**
```
可以可视化注意力权重，看模型关注什么
例如: 发现模型在财报日前后权重特别高
```

**3. 长距离依赖**
```
LSTM: 记忆会衰减
Attention: 可以直接关注很久以前的关键事件
```

## 🏗️ 模型架构详解

### 架构：LSTM + Attention + 多任务学习

```
输入特征 → LSTM → Attention → 分支1: 价格预测
                           → 分支2: 趋势分类
```

### 逐层详解

#### 第1层：特征输入层
```python
# 输入形状: (batch, 60, 20)
# 60: 过去60天
# 20: 20个特征（价格 + 技术指标）
```

**特征包括**：
```python
价格特征 (5个):
- Open, High, Low, Close, Volume

技术指标 (15个):
- MA5, MA10, MA20, MA60          # 移动平均
- RSI                             # 相对强弱指标
- MACD, MACD_signal, MACD_hist   # MACD指标
- Bollinger_upper, Bollinger_lower # 布林带
- ATR                             # 平均真实波幅
- OBV                             # 能量潮
- ... 等
```

**为什么使用技术指标？**
- 原始价格：只有趋势信息
- 技术指标：包含动量、波动、趋势等多维信息
- 例如RSI：告诉我们是否超买/超卖

#### 第2层：LSTM层
```python
layers.LSTM(128, return_sequences=True, dropout=0.2)
```

**为什么return_sequences=True？**
- 需要返回所有时间步的输出
- 给注意力机制使用
- 输出形状：(batch, 60, 128)

#### 第3层：注意力层（核心）
```python
class AttentionLayer(layers.Layer):
    def call(self, inputs):
        # inputs: (batch, 60, 128)

        # 1. 计算注意力分数
        score = tf.matmul(inputs, inputs, transpose_b=True)
        # shape: (batch, 60, 60)

        # 2. 归一化
        attention_weights = tf.nn.softmax(score, axis=-1)
        # shape: (batch, 60, 60)

        # 3. 加权求和
        context = tf.matmul(attention_weights, inputs)
        # shape: (batch, 60, 128)

        return context, attention_weights
```

**为什么这样设计？**
- **Self-Attention**：每个时间点关注所有时间点
- **学习重要性**：自动学习哪些历史数据重要
- **可视化**：attention_weights可以可视化

#### 第4层：多任务输出

**任务1：价格预测（回归）**
```python
# 预测明天的收盘价
price_output = layers.Dense(1, name='price_prediction')
```

**任务2：趋势分类（分类）**
```python
# 预测涨跌（3分类：跌、平、涨）
trend_output = layers.Dense(3, activation='softmax',
                           name='trend_classification')
```

**为什么多任务学习？**
```
单任务: 只预测价格
      ↓
问题: 价格预测误差大时，无法判断趋势

多任务: 同时预测价格和趋势
      ↓
优势: 1. 趋势分类更稳定
      2. 两个任务互相促进
      3. 可以根据趋势调整价格预测
```

### 损失函数设计

```python
# 总损失 = 价格损失 + 趋势损失
total_loss = alpha * price_loss + beta * trend_loss

# 价格损失: MSE
price_loss = mean_squared_error(y_true_price, y_pred_price)

# 趋势损失: 交叉熵
trend_loss = categorical_crossentropy(y_true_trend, y_pred_trend)

# 权重: alpha=0.7, beta=0.3
# 为什么: 价格预测是主要任务
```

## 📊 数据集

**数据来源**：
- Yahoo Finance API
- 时间跨度：2010-2023年
- 股票：S&P 500成分股

**数据预处理**：
```python
1. 缺失值处理：前向填充
2. 异常值处理：3σ原则
3. 归一化：MinMaxScaler (0-1)
4. 滑动窗口：60天预测1天
```

## 📁 项目结构

```
02_股票价格预测_LSTM高级/
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 00_金融时间序列基础.ipynb      # 金融数据特点
│   ├── 01_数据获取和探索.ipynb        # 下载和分析数据
│   ├── 02_技术指标特征工程.ipynb      # ⭐ MA, RSI, MACD等
│   ├── 03_基础LSTM模型.ipynb          # 不带注意力
│   ├── 04_LSTM_Attention模型.ipynb    # ⭐ 注意力机制详解
│   ├── 05_多任务学习.ipynb            # ⭐ 价格+趋势
│   ├── 06_模型集成.ipynb              # 集成多个模型
│   ├── 07_回测和风险管理.ipynb        # 交易策略
│   └── 08_模型可解释性.ipynb          # 注意力可视化
│
├── src/
│   ├── __init__.py
│   ├── data.py                         # 数据下载和预处理
│   ├── features.py                     # ⭐ 技术指标计算
│   ├── model.py                        # ⭐ 注意力LSTM模型
│   ├── train.py
│   ├── evaluate.py
│   └── backtest.py                     # 回测系统
│
├── data/
├── models/
└── results/
```

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据
python src/data.py --ticker AAPL --start 2010-01-01

# 3. 训练模型
python src/train.py --model attention_lstm --epochs 100

# 4. 回测
python src/backtest.py --model models/best_model.h5
```

## 📈 预期结果

| 模型 | MAE | RMSE | 方向准确率 | 夏普比率 |
|-----|-----|------|-----------|---------|
| 基础LSTM | $2.5 | $3.2 | 52% | 0.3 |
| LSTM+Attention | $1.8 | $2.4 | 58% | 0.8 |
| 多任务学习 | $1.6 | $2.1 | 62% | 1.2 |
| 模型集成 | $1.4 | $1.9 | 65% | 1.5 |

**注意**：
- 方向准确率 > 50%才有意义
- 夏普比率 > 1.0表示策略可行
- 实际交易需考虑交易成本

## 🎓 学习要点

### 1. 技术指标的作用

**移动平均线（MA）**：
```python
MA5 = Close.rolling(5).mean()   # 短期趋势
MA20 = Close.rolling(20).mean() # 中期趋势

# 金叉: MA5上穿MA20 → 买入信号
# 死叉: MA5下穿MA20 → 卖出信号
```

**RSI（相对强弱指标）**：
```python
RSI = 100 - 100/(1 + RS)
# RS = 平均涨幅 / 平均跌幅

# RSI > 70: 超买，可能下跌
# RSI < 30: 超卖，可能上涨
```

**MACD**：
```python
MACD = EMA12 - EMA26
Signal = EMA9(MACD)

# MACD上穿Signal: 买入信号
# MACD下穿Signal: 卖出信号
```

### 2. 注意力机制的关键点

- ✅ 使用Self-Attention（自注意力）
- ✅ 可视化注意力权重，理解模型决策
- ✅ 注意力权重可以发现关键事件
- ✅ 结合LSTM的记忆能力和Attention的选择能力

### 3. 多任务学习的优势

**为什么同时预测价格和趋势？**
```
场景1: 价格预测 $150.5，实际 $152
      → 误差$1.5，但趋势正确（都是涨）
      → 交易策略：可以盈利

场景2: 价格预测 $150.5，实际 $148
      → 误差$2.5，趋势错误（预测涨，实际跌）
      → 交易策略：会亏损

结论: 趋势比精确价格更重要！
```

### 4. 常见问题

**Q1: 为什么不能用未来数据？**
A: 数据泄露！
```python
# 错误: 使用未来数据计算MA
MA = data['Close'].rolling(20).mean()

# 正确: 只使用历史数据
MA = data['Close'].shift(1).rolling(20).mean()
```

**Q2: 为什么股票预测这么难？**
A:
- 市场是弱有效的（历史信息已反映在价格中）
- 受新闻、政策等外部因素影响大
- 机器学习只能捕获历史模式
- 黑天鹅事件无法预测

**Q3: 如何评估模型？**
A: 不能只看MAE/RMSE
```python
评估指标:
1. 方向准确率: 预测涨跌是否正确
2. 夏普比率: 风险调整后收益
3. 最大回撤: 最大亏损幅度
4. 胜率: 盈利交易占比
```

**Q4: 注意力机制的计算复杂度？**
A:
- 时间复杂度：O(n²)，n是序列长度
- 空间复杂度：O(n²)
- 60天序列：可接受
- 更长序列：考虑稀疏注意力

## 🔧 进阶优化

### 1. Transformer替代LSTM
```python
# 完全基于注意力机制
# 并行计算，训练更快
# 长距离依赖更好
```

### 2. 强化学习
```python
# 直接学习交易策略
# 考虑交易成本
# 最大化收益而非最小化误差
```

### 3. 情感分析
```python
# 结合新闻情感
# 社交媒体情绪
# 提高预测准确率
```

## ⚠️ 风险提示

1. **过拟合风险**：历史表现不代表未来
2. **交易成本**：手续费、滑点会大幅降低收益
3. **市场变化**：模型需要定期重新训练
4. **黑天鹅事件**：极端事件无法预测
5. **仅供学习**：不构成投资建议

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Multi-Task Learning](https://arxiv.org/abs/1706.05098)
- [Technical Analysis Library](https://github.com/bukosabino/ta)

## 🎯 下一步

完成本项目后，可以尝试：
1. **Transformer时间序列**：完全基于注意力
2. **强化学习交易**：DQN、PPO等
3. **高频交易**：分钟级、秒级预测

---

**难度等级**: ⭐⭐⭐⭐☆ (高级)
**预计学习时间**: 3-4周
**前置知识**: LSTM、注意力机制、金融基础
**风险提示**: 仅供学习，不构成投资建议
