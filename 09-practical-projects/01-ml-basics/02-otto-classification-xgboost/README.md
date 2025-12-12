# XGBoost Otto分类挑战 - 中级项目

**难度**: ⭐⭐⭐☆☆ (中级)

## 📋 项目简介

本项目使用XGBoost、LightGBM、CatBoost等梯度提升算法进行多分类，并通过Stacking集成提升性能。这是Kaggle经典竞赛，你将学习模型集成、超参数调优、以及如何达到竞赛级别的性能。

### 🎯 学习目标

- ✅ 掌握多分类问题的XGBoost应用
- ✅ 学习XGBoost、LightGBM、CatBoost的对比
- ✅ 掌握Stacking和Blending集成技术
- ✅ 学习Optuna自动超参数调优
- ✅ 理解为什么模型集成能提升性能

## 🧠 为什么使用模型集成？

### 单模型的局限

```
XGBoost单模型: 准确率 78%
              ↓
问题: 每个模型都有偏差
     XGBoost可能在某些样本上表现差
              ↓
解决: 集成多个不同的模型
```

### 集成学习的优势

```
XGBoost:    78% (擅长捕获非线性关系)
LightGBM:   77% (训练快，特征重要性不同)
CatBoost:   76% (处理类别特征好)
              ↓
Stacking集成: 80% (结合各模型优势)
              ↓
结果: 比任何单模型都好
```

## 🏗️ 模型集成原理详解

### 1. Voting（投票）

**硬投票**：
```python
# 3个模型的预测
模型1: 类别2
模型2: 类别2
模型3: 类别3
       ↓
最终: 类别2 (多数投票)
```

**软投票**：
```python
# 3个模型的概率预测
模型1: [0.1, 0.7, 0.2]  # 类别2概率最高
模型2: [0.2, 0.6, 0.2]  # 类别2概率最高
模型3: [0.3, 0.3, 0.4]  # 类别3概率最高
       ↓
平均: [0.2, 0.53, 0.27]
       ↓
最终: 类别2 (平均概率最高)
```

### 2. Stacking（堆叠）

**两层结构**：
```
第一层（基模型）:
XGBoost → 预测概率 [0.2, 0.5, 0.3]
LightGBM → 预测概率 [0.3, 0.4, 0.3]
CatBoost → 预测概率 [0.1, 0.6, 0.3]
              ↓
第二层（元模型）:
输入: 9个特征（3个模型×3个类别概率）
逻辑回归 → 最终预测
```

**为什么Stacking更强？**
- Voting：简单平均，没有学习
- Stacking：元模型学习如何组合基模型
- 例如：元模型可能学到"XGBoost在类别1上更准确"

### 3. Blending（混合）

**与Stacking的区别**：
```
Stacking:
- 使用交叉验证生成元特征
- 充分利用训练数据
- 训练时间长

Blending:
- 使用holdout验证集生成元特征
- 实现简单
- 训练时间短
```

## 📊 数据集

**Otto Group Product Classification**：
- 训练集：61,878个样本
- 测试集：144,368个样本
- 特征：93个匿名特征（都是数值型）
- 类别：9个产品类别（Class_1 到 Class_9）

**数据特点**：
```
1. 特征已脱敏（不知道具体含义）
2. 特征值都是整数（可能是计数）
3. 类别不平衡（某些类别样本少）
4. 评估指标：Multi-class Log Loss
```

## 🏗️ XGBoost多分类详解

### 多分类 vs 二分类

**二分类**：
```python
# 输出1个概率
output = sigmoid(score)  # 0-1之间
```

**多分类**：
```python
# 输出9个概率（Otto有9个类别）
output = softmax(scores)  # 和为1
# 例如: [0.05, 0.3, 0.1, 0.2, 0.05, 0.1, 0.15, 0.03, 0.02]
```

### XGBoost多分类参数

```python
XGBClassifier(
    objective='multi:softprob',  # 多分类 + 输出概率
    num_class=9,                 # 9个类别
    eval_metric='mlogloss',      # 多分类对数损失

    # 树的参数
    max_depth=8,                 # 比二分类稍深
    min_child_weight=1,

    # 提升参数
    learning_rate=0.05,
    n_estimators=500,

    # 正则化
    reg_alpha=0.1,
    reg_lambda=1,

    # 采样
    subsample=0.8,               # 行采样
    colsample_bytree=0.8,        # 列采样
)
```

**为什么max_depth=8？**
- 二分类：通常3-6
- 多分类：需要更深的树
- 原因：9个类别需要更复杂的决策边界

## 🏗️ 三大梯度提升算法对比

### XGBoost vs LightGBM vs CatBoost

| 特性 | XGBoost | LightGBM | CatBoost |
|-----|---------|----------|----------|
| **训练速度** | 中等 | 最快 | 较慢 |
| **内存占用** | 中等 | 最小 | 较大 |
| **准确率** | 高 | 高 | 最高 |
| **类别特征** | 需编码 | 需编码 | 自动处理 |
| **过拟合** | 中等 | 容易 | 不易 |
| **最佳场景** | 通用 | 大数据 | 类别特征多 |

### LightGBM的优势

**Leaf-wise生长**：
```
XGBoost (Level-wise):
    根
   /  \
  A    B    # 先分裂完这一层
 / \  / \
C  D E  F  # 再分裂下一层

LightGBM (Leaf-wise):
    根
   /  \
  A    B
 / \       # 只分裂增益最大的叶子
C  D
```

**为什么更快？**
- 只分裂增益最大的叶子
- 减少不必要的分裂
- 训练速度提升2-3倍

### CatBoost的优势

**自动处理类别特征**：
```python
# XGBoost/LightGBM: 需要手动编码
df['category'] = LabelEncoder().fit_transform(df['category'])

# CatBoost: 自动处理
model = CatBoostClassifier(cat_features=['category'])
model.fit(X, y)  # 直接使用原始类别特征
```

**Ordered Boosting**：
- 防止目标泄露
- 提高泛化能力

## 📁 项目结构

```
02_Otto分类挑战_XGBoost中级/
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 00_多分类问题基础.ipynb         # 多分类概念
│   ├── 01_数据探索.ipynb               # EDA
│   ├── 02_XGBoost基础模型.ipynb        # 单模型
│   ├── 03_LightGBM模型.ipynb           # LightGBM
│   ├── 04_CatBoost模型.ipynb           # CatBoost
│   ├── 05_模型对比分析.ipynb           # ⭐ 三大算法对比
│   ├── 06_Stacking集成.ipynb           # ⭐ 模型堆叠
│   ├── 07_Optuna调参.ipynb             # ⭐ 自动调参
│   └── 08_特征工程.ipynb               # 高级特征
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py                        # ⭐ 三大模型实现
│   ├── ensemble.py                      # ⭐ 集成方法
│   ├── tuning.py                        # ⭐ Optuna调参
│   └── evaluate.py
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
cd data && python download_data.py

# 3. 训练单模型
python src/train.py --model xgboost

# 4. 训练集成模型
python src/train.py --model stacking

# 5. 自动调参
python src/tuning.py --trials 100
```

## 📈 预期结果

| 模型 | Log Loss | 训练时间 | Kaggle排名 |
|-----|----------|---------|-----------|
| XGBoost | 0.45 | 5分钟 | Top 30% |
| LightGBM | 0.46 | 2分钟 | Top 35% |
| CatBoost | 0.44 | 10分钟 | Top 25% |
| **Voting** | 0.43 | 17分钟 | Top 20% |
| **Stacking** | **0.41** | 25分钟 | **Top 10%** |

## 🎓 学习要点

### 1. 多分类评估指标

**Log Loss（对数损失）**：
```python
# 公式
LogLoss = -1/N * Σ Σ y_ij * log(p_ij)

# 例子
真实类别: Class_2
预测概率: [0.1, 0.7, 0.2, ...]
         ↓
LogLoss = -log(0.7) = 0.36

# 越小越好
# 完美预测: LogLoss = 0
# 随机猜测: LogLoss = 2.2 (9个类别)
```

### 2. Stacking实现技巧

**交叉验证生成元特征**：
```python
# 5折交叉验证
for fold in range(5):
    # 训练基模型
    model.fit(X_train_fold, y_train_fold)

    # 预测验证集（作为元特征）
    meta_features[val_idx] = model.predict_proba(X_val_fold)

    # 预测测试集（取平均）
    test_meta += model.predict_proba(X_test) / 5

# 元模型训练
meta_model.fit(meta_features, y_train)
```

### 3. Optuna自动调参

**定义搜索空间**：
```python
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=5).mean()
    return score

# 运行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**为什么用Optuna？**
- GridSearch：穷举所有组合，太慢
- RandomSearch：随机搜索，效率低
- Optuna：贝叶斯优化，智能搜索

### 4. 常见问题

**Q1: 为什么集成能提升性能？**
A: 偏差-方差分解
```
单模型误差 = 偏差² + 方差 + 噪声

集成效果:
- 降低方差：多个模型平均，减少随机性
- 保持偏差：模型能力不变
- 结果：总误差降低
```

**Q2: 如何选择基模型？**
A: 多样性原则
```
好的组合:
- XGBoost + LightGBM + 逻辑回归
- 不同算法，预测差异大

坏的组合:
- XGBoost + XGBoost(不同参数)
- 同一算法，预测相似
```

**Q3: Stacking会过拟合吗？**
A: 会，需要注意
```
防止过拟合:
1. 使用交叉验证生成元特征
2. 元模型使用简单模型（逻辑回归）
3. 元模型加正则化
4. 不要堆叠太多层（1-2层足够）
```

**Q4: 如何处理类别不平衡？**
A:
```python
# 方法1: 调整样本权重
model = XGBClassifier(scale_pos_weight=10)

# 方法2: 使用分层采样
cv = StratifiedKFold(n_splits=5)

# 方法3: 过采样少数类
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
```

## 🔧 进阶优化

### 1. 伪标签（Pseudo Labeling）
```python
# 用训练好的模型预测测试集
test_pred = model.predict(X_test)

# 选择高置信度的预测作为伪标签
confident_idx = test_pred.max(axis=1) > 0.9
X_pseudo = X_test[confident_idx]
y_pseudo = test_pred[confident_idx].argmax(axis=1)

# 加入训练集重新训练
X_train_new = np.vstack([X_train, X_pseudo])
y_train_new = np.hstack([y_train, y_pseudo])
```

### 2. 特征工程
```python
# 统计特征
df['sum'] = df.iloc[:, :93].sum(axis=1)
df['mean'] = df.iloc[:, :93].mean(axis=1)
df['std'] = df.iloc[:, :93].std(axis=1)
df['max'] = df.iloc[:, :93].max(axis=1)
df['min'] = df.iloc[:, :93].min(axis=1)

# 特征交互
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)
```

### 3. 模型校准
```python
# 校准预测概率
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(model, cv=5)
calibrated.fit(X_train, y_train)
```

## 📚 参考资料

- [Otto Group Kaggle竞赛](https://www.kaggle.com/c/otto-group-product-classification-challenge)
- [Optuna文档](https://optuna.readthedocs.io/)
- [Stacking教程](https://mlwave.com/kaggle-ensembling-guide/)

## 🎯 下一步

完成本项目后，可以尝试：
1. **高级项目**：Kaggle竞赛级别（SHAP解释 + 深度特征工程）
2. **深度学习**：TabNet（深度学习处理表格数据）
3. **AutoML**：H2O AutoML、AutoGluon

---

**难度等级**: ⭐⭐⭐☆☆ (中级)
**预计学习时间**: 2-3周
**前置知识**: XGBoost基础、交叉验证
**Kaggle排名**: Top 10%（Stacking后）
