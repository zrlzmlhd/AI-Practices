# LSTM温度预测 - 中级项目

**难度**: ⭐⭐⭐☆☆ (中级)

## 📋 项目简介

本项目使用LSTM进行多变量时间序列预测，基于历史气象数据预测未来温度。这是LSTM的经典应用场景，你将学习如何处理时间序列数据、构建多层LSTM模型，以及对比不同LSTM架构的效果。

### 🎯 学习目标

- ✅ 掌握时间序列数据的特征工程
- ✅ 理解滑动窗口和多变量预测
- ✅ 学习堆叠LSTM和双向LSTM的应用
- ✅ 掌握LSTM vs GRU的对比
- ✅ 理解为什么使用多层LSTM

## 🧠 为什么使用LSTM进行时间序列预测？

### 问题背景

温度预测需要考虑：
- **时间依赖性**：今天的温度受昨天、前天的影响
- **周期性模式**：每天、每周、每季节的温度变化规律
- **多变量关系**：温度、湿度、气压、风速等相互影响

### 为什么不用传统方法？

**传统方法（ARIMA）的局限**：
```
ARIMA: 只能处理单变量，假设线性关系
      ↓
问题: 无法捕获温度与湿度、气压的复杂非线性关系
```

**LSTM的优势**：
```
LSTM: 可以处理多变量，学习非线性关系
     ↓
优势: 同时考虑温度、湿度、气压等多个因素
     ↓
结果: 预测更准确
```

## 🏗️ 多层LSTM原理详解

### 为什么需要多层LSTM？

#### 单层LSTM的局限
```
输入: [温度, 湿度, 气压] 的时间序列
     ↓
单层LSTM: 只能学习一个层次的时间模式
     ↓
问题: 难以同时捕获短期波动和长期趋势
```

#### 多层LSTM的优势
```
输入: [温度, 湿度, 气压] 的时间序列
     ↓
第1层LSTM: 学习短期模式（小时级波动）
     ↓
第2层LSTM: 学习中期模式（日级变化）
     ↓
第3层LSTM: 学习长期模式（季节性趋势）
     ↓
输出: 综合多层次信息的预测
```

### 堆叠LSTM架构

```python
# 三层堆叠LSTM
输入 (batch, 168, 5)  # 168小时，5个特征
  ↓
LSTM层1 (128单元) → 学习低级时间特征
  ↓ return_sequences=True
LSTM层2 (64单元)  → 学习中级时间特征
  ↓ return_sequences=True
LSTM层3 (32单元)  → 学习高级时间特征
  ↓ return_sequences=False
Dense层 → 输出预测
```

**为什么逐层递减单元数？**
- 第1层(128)：需要大容量捕获原始数据的细节
- 第2层(64)：中级特征更抽象，需要的容量减半
- 第3层(32)：高级特征最抽象，容量最小

### 双向LSTM在时间序列中的应用

**注意**：时间序列预测通常**不使用**双向LSTM！

**为什么？**
```
训练时: 可以看到未来数据（数据泄露）
      ↓
问题: 模型学会"作弊"，利用未来信息
      ↓
预测时: 没有未来数据，性能大幅下降
```

**什么时候可以用双向LSTM？**
- 时间序列分类（判断整个序列的类别）
- 异常检测（分析历史数据）
- 不需要实时预测的场景

## 📊 数据集

**Jena气候数据集**
- 时间跨度：2009-2016年
- 采样频率：每10分钟一次
- 特征数量：14个气象指标
- 总样本数：420,551条记录

**特征说明**：
```
1. Temperature (°C)        - 温度（目标变量）
2. Pressure (mbar)         - 气压
3. Humidity (%)            - 湿度
4. Wind Speed (m/s)        - 风速
5. Wind Direction (°)      - 风向
... 等14个特征
```

## 🏗️ 模型架构详解

### 架构1：堆叠LSTM（推荐）

```
输入 → LSTM(128) → LSTM(64) → LSTM(32) → Dense(16) → Output
```

#### 逐层详解

**第1层：LSTM(128, return_sequences=True)**
```python
layers.LSTM(128, return_sequences=True, dropout=0.2)
```

**为什么这样设计**：
- `units=128`：大容量捕获原始时间序列的细节
  - 需要记住：每小时的温度变化、突然的天气变化
  - 128个单元足够记住168小时（7天）的复杂模式

- `return_sequences=True`：返回所有时间步
  - **关键**：必须返回序列给下一层LSTM
  - 输出形状：(batch, 168, 128)
  - 如果False：只返回最后一个输出，无法堆叠

- `dropout=0.2`：较小的dropout
  - 为什么只有20%：时间序列对dropout敏感
  - 太大会破坏时序信息
  - 第一层需要保留更多信息

**第2层：LSTM(64, return_sequences=True)**
```python
layers.LSTM(64, return_sequences=True, dropout=0.2)
```

**为什么这样设计**：
- `units=64`：容量减半
  - 第一层已经提取了低级特征
  - 第二层学习更抽象的模式（日级变化）
  - 减少参数，防止过拟合

- `return_sequences=True`：继续返回序列
  - 还有第三层LSTM需要接收

**第3层：LSTM(32, return_sequences=False)**
```python
layers.LSTM(32, return_sequences=False, dropout=0.2)
```

**为什么这样设计**：
- `units=32`：最小容量
  - 学习最高级的抽象特征（周级、季节性）
  - 只需要记住整体趋势

- `return_sequences=False`：只返回最后输出
  - 不再需要序列，准备输出预测
  - 输出形状：(batch, 32)

**第4层：Dense(16)**
```python
layers.Dense(16, activation='relu')
```

**为什么这样设计**：
- `units=16`：进一步压缩
  - 从32维压缩到16维
  - 去除冗余，提取最关键的预测特征

**第5层：Output**
```python
layers.Dense(1)  # 回归任务，不用激活函数
```

**为什么这样设计**：
- `units=1`：单值输出（温度）
- 无激活函数：回归任务，输出可以是任意实数

### 架构2：LSTM vs GRU对比

**GRU（门控循环单元）**：
- 只有2个门（更新门、重置门）
- 参数量是LSTM的75%
- 训练更快，但表达能力稍弱

**何时选择GRU？**
```
数据量小 → GRU（参数少，不易过拟合）
数据量大 → LSTM（表达能力强）
需要快速训练 → GRU
需要最高精度 → LSTM
```

## 📁 项目结构

```
01_温度预测_LSTM中级/
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 00_时间序列基础.ipynb           # 时间序列概念
│   ├── 01_数据探索.ipynb               # 数据可视化和分析
│   ├── 02_特征工程.ipynb               # 滑动窗口、归一化
│   ├── 03_单层LSTM.ipynb               # 基础模型
│   ├── 04_堆叠LSTM.ipynb               # ⭐ 多层LSTM详解
│   ├── 05_LSTM_vs_GRU.ipynb            # ⭐ 对比实验
│   ├── 06_模型评估.ipynb               # 详细评估
│   └── 07_超参数调优.ipynb             # 优化技巧
│
├── src/
│   ├── __init__.py
│   ├── data.py                          # 数据加载和预处理
│   ├── model.py                         # ⭐ 模型定义（详细注释）
│   ├── train.py
│   └── evaluate.py
│
├── data/
│   ├── download_data.py
│   └── README.md
│
├── models/
└── results/
```

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据
cd data && python download_data.py

# 3. 运行notebooks（按顺序）
jupyter notebook notebooks/

# 4. 或直接训练
python src/train.py --model_type stacked_lstm --epochs 50
```

## 📈 预期结果

| 模型 | MAE | RMSE | 训练时间 | 参数量 |
|-----|-----|------|---------|--------|
| 单层LSTM | 2.8°C | 3.5°C | 10分钟 | 150K |
| 堆叠LSTM | 2.3°C | 2.9°C | 25分钟 | 380K |
| GRU | 2.5°C | 3.1°C | 15分钟 | 280K |

## 🎓 学习要点

### 1. 时间序列特征工程

**滑动窗口**：
```python
# 使用过去7天（168小时）预测下一小时
window_size = 168
X = data[i:i+168]  # 输入
y = data[i+168]    # 目标
```

**多变量输入**：
```python
# 同时使用5个特征
features = ['temperature', 'pressure', 'humidity',
            'wind_speed', 'wind_direction']
X.shape = (batch, 168, 5)
```

### 2. 堆叠LSTM的关键点

- ✅ 第一层必须`return_sequences=True`
- ✅ 中间层也要`return_sequences=True`
- ✅ 最后一层`return_sequences=False`
- ✅ 逐层递减单元数（128→64→32）

### 3. 常见问题

**Q1: 为什么堆叠3层而不是更多？**
A:
- 3层已经能捕获多层次的时间模式
- 更多层：训练困难，梯度消失，过拟合
- 经验：2-3层最佳

**Q2: 如何选择窗口大小？**
A:
- 太小（24小时）：信息不足
- 太大（720小时）：计算量大，噪声多
- 经验：168小时（7天）平衡效果和效率

**Q3: 为什么不用双向LSTM？**
A:
- 预测任务不能看未来数据
- 双向LSTM会导致数据泄露
- 只在分类/异常检测中使用

## 🔧 进阶优化

### 1. 注意力机制
```python
# 让模型关注重要的时间步
attention = layers.Attention()([lstm_out, lstm_out])
```

### 2. 多步预测
```python
# 预测未来24小时
output = layers.Dense(24)  # 输出24个值
```

### 3. 集成学习
```python
# 结合LSTM和传统方法
final_pred = 0.7 * lstm_pred + 0.3 * arima_pred
```

## 📚 参考资料

- [Time Series Forecasting with LSTM](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)
- [Jena Climate Dataset](https://www.kaggle.com/datasets/mnassrib/jena-climate)

## 🎯 下一步

完成本项目后，可以尝试：
1. **高级项目**：股票价格预测（多任务学习）
2. **Transformer**：使用注意力机制进行时间序列预测
3. **实时预测**：部署模型到生产环境

---

**难度等级**: ⭐⭐⭐☆☆ (中级)
**预计学习时间**: 2-3周
**前置知识**: LSTM基础、时间序列基础
