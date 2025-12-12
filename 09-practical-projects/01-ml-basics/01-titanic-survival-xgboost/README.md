# XGBoost Titanic生存预测 - 入门级项目

**难度**: ⭐⭐☆☆☆ (入门)

## 📋 项目简介

本项目使用XGBoost预测泰坦尼克号乘客的生存概率，这是学习XGBoost的最佳入门项目。你将深入理解XGBoost的原理、为什么它在Kaggle竞赛中表现优异，以及如何进行特征工程和超参数调优。

### 🎯 学习目标

- ✅ 深入理解XGBoost的原理和优势
- ✅ 掌握梯度提升树的工作机制
- ✅ 学习特征工程和特征重要性分析
- ✅ 掌握超参数调优技巧
- ✅ 理解为什么XGBoost比随机森林更强

## 🧠 为什么使用XGBoost？

### XGBoost的全称

**XGBoost** = **eXtreme Gradient Boosting**（极端梯度提升）

### 问题背景

Titanic生存预测是一个**表格数据分类**问题，特点：
- 特征多样（数值、类别、缺失值）
- 特征之间有复杂的交互关系
- 需要高准确率

### 为什么不用其他算法？

**逻辑回归的局限**：
```
逻辑回归: 假设特征之间是线性关系
        ↓
问题: 无法捕获复杂的特征交互
     例如: "年轻女性 + 头等舱" 的生存率特别高
        ↓
结果: 准确率有限（~78%）
```

**随机森林 vs XGBoost**：
```
随机森林: 并行训练多棵独立的树
        ↓
优点: 训练快，不易过拟合
缺点: 树之间不互相学习

XGBoost: 串行训练，每棵树学习前一棵的错误
       ↓
优点: 准确率更高（~82%）
缺点: 训练稍慢，需要调参
```

## 🏗️ XGBoost原理详解

### 1. 什么是梯度提升？

**核心思想**：每棵新树都在纠正之前所有树的错误

```python
# 第1棵树
预测: [0.6, 0.3, 0.8, ...]
真实: [1,   0,   1,   ...]
误差: [0.4, -0.3, 0.2, ...]  # 重点关注误差大的样本

# 第2棵树（专门学习第1棵的误差）
预测: [0.3, -0.2, 0.15, ...]  # 纠正第1棵的错误
总预测: 第1棵 + 第2棵 = [0.9, 0.1, 0.95, ...]

# 第3棵树（继续纠正）
预测: [0.08, -0.05, 0.03, ...]
总预测: 第1棵 + 第2棵 + 第3棵 = [0.98, 0.05, 0.98, ...]

# 最终: 100棵树的累加
```

### 2. XGBoost的三大优势

#### 优势1：正则化（防止过拟合）

**随机森林**：
```python
# 没有正则化，容易过拟合
树的复杂度: 不限制
```

**XGBoost**：
```python
# 内置正则化
损失函数 = 预测误差 + α*叶子数 + β*叶子权重²

# 效果: 自动控制树的复杂度
```

#### 优势2：处理缺失值

**传统方法**：
```python
# 需要手动填充缺失值
Age.fillna(Age.median())
```

**XGBoost**：
```python
# 自动学习缺失值的最佳方向
if Age is missing:
    try left branch → 计算增益
    try right branch → 计算增益
    选择增益大的方向
```

#### 优势3：并行化

**决策树训练**：
```python
# 串行: 必须一个节点一个节点分裂
for node in tree:
    find_best_split(node)  # 慢
```

**XGBoost**：
```python
# 并行: 同时计算所有特征的最佳分裂点
parallel_for feature in features:
    find_best_split(feature)  # 快
```

### 3. XGBoost的关键参数

#### 树的参数

**max_depth**（树的最大深度）：
```python
max_depth = 3  # 浅树，防止过拟合
max_depth = 10 # 深树，可能过拟合

# Titanic推荐: 3-6
# 为什么: 数据量小，浅树足够
```

**min_child_weight**（最小叶子权重）：
```python
min_child_weight = 1  # 允许小叶子
min_child_weight = 5  # 叶子至少5个样本

# Titanic推荐: 1-3
# 为什么: 防止过拟合
```

#### 提升参数

**learning_rate**（学习率）：
```python
learning_rate = 0.3  # 快速学习，可能不稳定
learning_rate = 0.01 # 慢速学习，更稳定

# Titanic推荐: 0.05-0.1
# 为什么: 平衡速度和稳定性
```

**n_estimators**（树的数量）：
```python
n_estimators = 100   # 少量树
n_estimators = 1000  # 大量树

# Titanic推荐: 100-500
# 为什么: 配合early_stopping
```

#### 正则化参数

**reg_alpha**（L1正则化）：
```python
reg_alpha = 0   # 无L1正则化
reg_alpha = 1   # 特征选择，稀疏解

# Titanic推荐: 0-0.1
```

**reg_lambda**（L2正则化）：
```python
reg_lambda = 1   # 默认值
reg_lambda = 10  # 强正则化

# Titanic推荐: 1-5
```

## 📊 数据集

**Titanic数据集**：
- 训练集：891名乘客
- 测试集：418名乘客
- 目标：预测生存（0=遇难，1=生还）

**特征说明**：
```
1. Pclass      - 船舱等级（1/2/3）
2. Sex         - 性别
3. Age         - 年龄
4. SibSp       - 兄弟姐妹/配偶数量
5. Parch       - 父母/子女数量
6. Fare        - 票价
7. Embarked    - 登船港口（C/Q/S）
8. Name        - 姓名（可提取头衔）
9. Ticket      - 票号
10. Cabin      - 船舱号
```

## 🏗️ 特征工程详解

### 1. 基础特征处理

**缺失值处理**：
```python
# Age: 用中位数填充
Age.fillna(Age.median())

# Embarked: 用众数填充
Embarked.fillna('S')

# Cabin: 创建新特征"是否有船舱号"
Has_Cabin = Cabin.notna().astype(int)
```

### 2. 特征创建

**家庭规模**：
```python
Family_Size = SibSp + Parch + 1

# 为什么有效:
# - 独自一人: 生存率低
# - 小家庭(2-4人): 生存率高
# - 大家庭(>4人): 生存率低（难以协调）
```

**头衔提取**：
```python
# 从姓名中提取头衔
Name = "Braund, Mr. Owen Harris"
Title = "Mr"

# 头衔分类:
# - Mr: 成年男性（生存率低）
# - Miss/Mrs: 女性（生存率高）
# - Master: 男孩（生存率中等）
# - Rare: 稀有头衔（生存率高，如Lady, Sir）
```

**年龄分组**：
```python
# 将连续年龄转为类别
Age_Group = pd.cut(Age, bins=[0, 12, 18, 35, 60, 100],
                   labels=['Child', 'Teen', 'Adult',
                          'Middle', 'Senior'])

# 为什么: 不同年龄段生存率差异大
```

### 3. 特征交互

**性别 × 船舱等级**：
```python
Sex_Pclass = Sex + '_' + Pclass.astype(str)

# 例如:
# - Female_1: 头等舱女性（生存率最高 ~97%）
# - Male_3: 三等舱男性（生存率最低 ~14%）
```

## 📁 项目结构

```
01_Titanic生存预测_XGBoost入门/
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 00_XGBoost原理详解.ipynb       # ⭐ 梯度提升原理
│   ├── 01_数据探索.ipynb              # EDA和可视化
│   ├── 02_特征工程.ipynb              # ⭐ 特征创建和选择
│   ├── 03_基础XGBoost.ipynb           # 默认参数模型
│   ├── 04_特征重要性分析.ipynb        # ⭐ 理解模型决策
│   ├── 05_超参数调优.ipynb            # ⭐ GridSearch/RandomSearch
│   ├── 06_模型对比.ipynb              # XGBoost vs 其他算法
│   └── 07_模型解释.ipynb              # SHAP值分析
│
├── src/
│   ├── __init__.py
│   ├── data.py                         # 数据加载
│   ├── features.py                     # ⭐ 特征工程
│   ├── model.py                        # ⭐ XGBoost模型（详细注释）
│   ├── train.py
│   └── evaluate.py
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── download_data.py
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
python src/train.py --n_estimators 500 --max_depth 4
```

## 📈 预期结果

| 模型 | 准确率 | 训练时间 | 参数量 |
|-----|--------|---------|--------|
| 逻辑回归 | 78% | 1秒 | 10 |
| 随机森林 | 80% | 5秒 | 100棵树 |
| **XGBoost（默认）** | 81% | 3秒 | 100棵树 |
| **XGBoost（调优）** | **83%** | 10秒 | 500棵树 |

## 🎓 学习要点

### 1. XGBoost的核心概念

**梯度提升 vs Bagging**：
```
Bagging（随机森林）:
树1, 树2, 树3 → 独立训练 → 投票
优点: 并行快，不易过拟合
缺点: 树之间不互相学习

Boosting（XGBoost）:
树1 → 树2（学习树1的错误）→ 树3（学习树1+树2的错误）
优点: 准确率高，互相学习
缺点: 串行慢，容易过拟合（需要正则化）
```

### 2. 特征重要性

**三种重要性指标**：
```python
# 1. Weight: 特征被使用的次数
# 2. Gain: 特征带来的平均增益
# 3. Cover: 特征影响的样本数

# Titanic最重要特征:
1. Sex (Gain: 0.35)      # 性别最重要
2. Title (Gain: 0.22)    # 头衔第二
3. Fare (Gain: 0.15)     # 票价第三
4. Age (Gain: 0.12)      # 年龄第四
```

### 3. 超参数调优策略

**步骤1：固定树的数量，调整树的结构**
```python
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 2, 3],
}
```

**步骤2：调整学习率和树的数量**
```python
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500],
}
```

**步骤3：调整正则化参数**
```python
param_grid = {
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 5, 10],
}
```

### 4. 常见问题

**Q1: XGBoost为什么比随机森林准确率高？**
A:
- 随机森林：树之间独立，不互相学习
- XGBoost：后面的树专门学习前面树的错误
- 结果：XGBoost能逐步提高准确率

**Q2: 如何防止XGBoost过拟合？**
A:
```python
# 方法1: 降低学习率，增加树的数量
learning_rate = 0.01
n_estimators = 1000

# 方法2: 增加正则化
reg_alpha = 0.1
reg_lambda = 5

# 方法3: 限制树的复杂度
max_depth = 3
min_child_weight = 3

# 方法4: 使用early_stopping
early_stopping_rounds = 50
```

**Q3: XGBoost如何处理类别特征？**
A:
```python
# 方法1: Label Encoding（推荐）
Sex: Male→0, Female→1

# 方法2: One-Hot Encoding
Embarked: C→[1,0,0], Q→[0,1,0], S→[0,0,1]

# XGBoost推荐Label Encoding:
# - 树模型能自动学习类别的顺序
# - 减少特征数量
```

**Q4: 如何选择n_estimators？**
A:
```python
# 使用early_stopping自动选择
model = XGBClassifier(
    n_estimators=1000,  # 设置一个大值
    early_stopping_rounds=50,  # 50轮不提升就停止
)
model.fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         verbose=False)

print(f"最佳迭代: {model.best_iteration}")
```

## 🔧 进阶优化

### 1. 使用LightGBM
```python
# LightGBM: 更快的梯度提升
# 优点: 训练速度快，内存占用小
# 缺点: 小数据集上可能不如XGBoost
```

### 2. 使用CatBoost
```python
# CatBoost: 自动处理类别特征
# 优点: 不需要编码类别特征
# 缺点: 训练稍慢
```

### 3. 模型集成
```python
# Stacking: 结合多个模型
final_pred = 0.5 * xgb_pred + 0.3 * rf_pred + 0.2 * lr_pred
```

## 📚 参考资料

- [XGBoost官方文档](https://xgboost.readthedocs.io/)
- [XGBoost论文](https://arxiv.org/abs/1603.02754)
- [Kaggle Titanic竞赛](https://www.kaggle.com/c/titanic)

## 🎯 下一步

完成本项目后，可以尝试：
1. **中级项目**：Otto分类挑战（多分类 + 模型集成）
2. **高级项目**：Kaggle竞赛级别（SHAP解释 + 自动调参）
3. **其他算法**：LightGBM、CatBoost对比

---

**难度等级**: ⭐⭐☆☆☆ (入门)
**预计学习时间**: 1-2周
**前置知识**: Python基础、决策树基础
**Kaggle排名**: Top 10%（调优后）
