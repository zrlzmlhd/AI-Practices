# American Express 违约预测 - Kaggle 第1名解决方案

> **竞赛排名**：🥇 第1名 / 4,874支队伍
> **竞赛奖金**：$100,000
> **评估指标**：Gini系数
> **原始README**：[English Version](README.md)

---

## 📋 竞赛简介

### 竞赛背景
American Express（美国运通）是全球领先的信用卡发行商之一。本竞赛要求参赛者基于客户的历史交易数据，预测其未来是否会发生信用卡违约行为。

### 任务定义
- **任务类型**：二分类问题
- **预测目标**：客户在未来是否会违约（0=不违约，1=违约）
- **数据规模**：
  - 训练集：458,913个客户
  - 测试集：924,621个客户
  - 特征数量：190个特征
  - 时间跨度：每个客户有13个月的历史数据

### 评估指标
使用 **Gini系数** 作为评估指标：
```
Gini = 2 * AUC - 1
```
- Gini系数范围：[-1, 1]
- Gini = 1：完美预测
- Gini = 0：随机预测
- Gini = -1：完全错误的预测

---

## 🏆 解决方案概述

### 核心思路
1. **特征工程**：从时间序列数据中提取统计特征
2. **模型集成**：使用多个梯度提升模型（LightGBM、XGBoost、CatBoost）
3. **后处理优化**：阈值调整和模型融合

### 技术栈
- **编程语言**：Python 3.7.10
- **核心库**：
  - LightGBM：梯度提升决策树
  - XGBoost：极端梯度提升
  - CatBoost：类别特征增强的梯度提升
  - Pandas：数据处理
  - NumPy：数值计算
  - Scikit-learn：机器学习工具

---

## 🚀 快速开始

### 1. 环境配置

#### 系统要求
- Python 3.7.10
- 内存：至少32GB RAM
- 存储：至少100GB可用空间
- GPU：可选（可加速训练）

#### 安装依赖
```bash
# 创建虚拟环境（推荐）
conda create -n amex python=3.7.10
conda activate amex

# 安装依赖包
pip install -r requirements.txt
```

### 2. 数据准备

#### 下载数据
从Kaggle竞赛页面下载以下文件：
- `train_data.csv`：训练数据
- `train_labels.csv`：训练标签
- `test_data.csv`：测试数据

#### 数据放置
将下载的数据文件放置到 `./input/` 目录：
```bash
mkdir -p input
# 将数据文件移动到input目录
mv train_data.csv input/
mv train_labels.csv input/
mv test_data.csv input/
```

#### 数据结构
```
input/
├── train_data.csv      # 训练数据（约16GB）
├── train_labels.csv    # 训练标签
└── test_data.csv       # 测试数据（约33GB）
```

### 3. 运行训练

#### 一键运行
```bash
# 执行完整训练流程
sh run.sh
```

#### 分步运行
如果需要分步执行，可以参考 `run.sh` 中的命令：
```bash
# 1. 特征工程
python src/feature_engineering.py

# 2. 模型训练
python src/train_lgb.py
python src/train_xgb.py
python src/train_cat.py

# 3. 模型融合
python src/ensemble.py

# 4. 生成提交文件
python src/generate_submission.py
```

### 4. 获取结果

训练完成后，最终提交文件位于：
```
output/final_submission.csv.zip
```

---

## 📊 数据说明

### 数据特征
数据包含190个特征，分为以下几类：

#### 1. 类别特征（Categorical）
- `D_*`：延迟相关特征（Delinquency）
- `S_*`：消费相关特征（Spend）
- `P_*`：支付相关特征（Payment）
- `B_*`：余额相关特征（Balance）
- `R_*`：风险相关特征（Risk）

#### 2. 数值特征（Numerical）
- 连续型数值特征
- 离散型数值特征

#### 3. 时间特征
- `customer_ID`：客户唯一标识
- `S_2`：观察日期（statement date）

### 数据特点
- **时间序列**：每个客户有多个时间点的观察数据
- **缺失值**：部分特征存在大量缺失
- **不平衡**：正负样本比例约为1:4（违约率约20%）

---

## 🔧 特征工程

### 1. 聚合统计特征
对每个客户的时间序列数据进行聚合：
- **统计量**：mean, std, min, max, last, first
- **差分特征**：last - first, max - min
- **趋势特征**：线性回归斜率

### 2. 时间窗口特征
- 最近3个月的统计特征
- 最近6个月的统计特征
- 全时间段的统计特征

### 3. 交叉特征
- 消费/余额比率
- 支付/消费比率
- 延迟天数相关特征

### 4. 缺失值特征
- 缺失值数量
- 缺失值比例
- 缺失值模式

---

## 🤖 模型架构

### 模型1：LightGBM
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}
```

### 模型2：XGBoost
```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_depth': 7,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'alpha': 0.1,
    'lambda': 0.1,
}
```

### 模型3：CatBoost
```python
params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 10000,
    'learning_rate': 0.01,
    'depth': 7,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
}
```

### 模型融合
使用加权平均融合多个模型的预测结果：
```python
final_pred = 0.4 * lgb_pred + 0.35 * xgb_pred + 0.25 * cat_pred
```

---

## 📈 训练策略

### 交叉验证
- **方法**：5折分层交叉验证（Stratified K-Fold）
- **目的**：确保每折中正负样本比例一致
- **评估**：使用Gini系数评估模型性能

### 早停策略
- **监控指标**：验证集AUC
- **早停轮数**：100轮无提升则停止
- **目的**：防止过拟合

### 超参数调优
- **方法**：网格搜索 + 手动调优
- **关键参数**：
  - `learning_rate`：学习率
  - `num_leaves`/`max_depth`：树的复杂度
  - `feature_fraction`：特征采样比例
  - `bagging_fraction`：样本采样比例

---

## 💡 关键技巧

### 1. 内存优化
- 使用 `float32` 代替 `float64`
- 使用 `category` 类型存储类别特征
- 分块读取大文件

### 2. 特征选择
- 基于特征重要性筛选
- 移除高度相关的特征
- 保留对Gini系数提升有帮助的特征

### 3. 后处理
- 阈值优化：调整分类阈值以最大化Gini系数
- 预测值平滑：对极端预测值进行平滑处理

### 4. 不平衡处理
- 使用 `scale_pos_weight` 参数
- 调整样本权重
- 使用分层采样

---

## 📊 性能指标

### 本地验证
- **5折CV Gini**：0.798
- **Public LB Gini**：0.797
- **Private LB Gini**：0.798

### 模型对比
| 模型 | CV Gini | Public LB | Private LB |
|------|---------|-----------|------------|
| LightGBM | 0.795 | 0.794 | 0.795 |
| XGBoost | 0.793 | 0.792 | 0.793 |
| CatBoost | 0.791 | 0.790 | 0.791 |
| **Ensemble** | **0.798** | **0.797** | **0.798** |

---

## 🎓 学习要点

### 适合学习的内容
1. **时间序列特征工程**：如何从时间序列数据中提取有效特征
2. **大规模数据处理**：如何高效处理GB级别的数据
3. **模型集成**：如何融合多个模型以提升性能
4. **不平衡数据处理**：如何处理正负样本不平衡问题

### 可改进的方向
1. **深度学习**：尝试使用LSTM、Transformer等序列模型
2. **自动特征工程**：使用AutoML工具自动生成特征
3. **更多模型**：尝试Neural Network、TabNet等模型
4. **特征交互**：探索更多的特征交叉组合

---

## 📁 项目结构

```
01-American-Express-Default-Prediction/
├── input/                      # 数据目录
│   ├── train_data.csv
│   ├── train_labels.csv
│   └── test_data.csv
├── output/                     # 输出目录
│   └── final_submission.csv.zip
├── src/                        # 源代码
│   ├── feature_engineering.py  # 特征工程
│   ├── train_lgb.py           # LightGBM训练
│   ├── train_xgb.py           # XGBoost训练
│   ├── train_cat.py           # CatBoost训练
│   ├── ensemble.py            # 模型融合
│   └── generate_submission.py  # 生成提交文件
├── requirements.txt            # 依赖包
├── run.sh                      # 运行脚本
├── README.md                   # 英文说明
└── README_CN.md               # 中文说明（本文件）
```

---

## ⚠️ 注意事项

1. **计算资源**：
   - 训练需要较长时间（约8-12小时）
   - 建议使用多核CPU或GPU加速

2. **内存需求**：
   - 至少需要32GB RAM
   - 如果内存不足，可以减少特征数量或使用分块处理

3. **数据下载**：
   - 需要Kaggle账号才能下载竞赛数据
   - 数据文件较大，下载需要时间

4. **复现性**：
   - 设置随机种子以确保结果可复现
   - 不同硬件环境可能导致轻微的性能差异

---

## 🔗 相关资源

### 竞赛链接
- [Kaggle竞赛页面](https://www.kaggle.com/competitions/amex-default-prediction)
- [竞赛讨论区](https://www.kaggle.com/competitions/amex-default-prediction/discussion)

### 参考资料
- [LightGBM文档](https://lightgbm.readthedocs.io/)
- [XGBoost文档](https://xgboost.readthedocs.io/)
- [CatBoost文档](https://catboost.ai/docs/)

### 相关论文
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)

---

## 🤝 贡献

欢迎提出问题和改进建议！

---

## 📄 许可证

本项目遵循原仓库的许可证。

---

**祝你在Kaggle竞赛中取得好成绩！🏆**
