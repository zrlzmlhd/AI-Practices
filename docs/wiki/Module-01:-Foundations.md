# Module 01: Foundations

**机器学习基础** - 建立坚实的机器学习理论基础，掌握经典算法的原理与实现。

---

## Module Overview

| Attribute | Value |
|:----------|:------|
| **Prerequisites** | Python, NumPy, Linear Algebra, Calculus |
| **Duration** | 4-6 weeks |
| **Notebooks** | 25+ |
| **Difficulty** | ⭐⭐ Beginner to Intermediate |

---

## Learning Objectives

完成本模块后，你将能够：

- ✅ 理解机器学习的核心概念和数学基础
- ✅ 实现并应用线性回归、逻辑回归、SVM 等经典算法
- ✅ 掌握模型评估、交叉验证和超参数调优技术
- ✅ 理解集成学习的原理并应用 XGBoost、LightGBM
- ✅ 应用降维和聚类等无监督学习技术
- ✅ 完成一个端到端的机器学习项目

---

## Submodules

### 01. Training Models

> 模型训练方法论，梯度下降与正则化

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| Linear Regression | OLS, Normal Equation | `01_linear_regression.ipynb` |
| Gradient Descent | Batch, Mini-batch, SGD | `02_gradient_descent.ipynb` |
| Polynomial Regression | Feature Engineering | `03_polynomial_regression.ipynb` |
| Learning Curves | Bias-Variance Tradeoff | `04_learning_curves.ipynb` |
| Regularization | L1 (Lasso), L2 (Ridge), ElasticNet | `05_regularization.ipynb` |

**Key Equations:**

```
Linear Regression:
ŷ = Xθ
θ = (X^T X)^(-1) X^T y  (Normal Equation)

Gradient Descent:
θ := θ - α ∇θ J(θ)

Ridge Regression:
J(θ) = MSE(θ) + α Σ θ²ᵢ

Lasso Regression:
J(θ) = MSE(θ) + α Σ |θᵢ|
```

### 02. Classification

> 分类算法与性能评估

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| Binary Classification | Decision Boundary | `01_binary_classification.ipynb` |
| Logistic Regression | Sigmoid, Cross-Entropy | `02_logistic_regression.ipynb` |
| Softmax Regression | Multiclass Classification | `03_softmax_regression.ipynb` |
| Performance Metrics | Precision, Recall, F1, ROC-AUC | `04_metrics.ipynb` |

**Key Equations:**

```
Logistic Regression:
σ(z) = 1 / (1 + e^(-z))
ŷ = σ(X θ)

Cross-Entropy Loss:
J(θ) = -1/m Σ [y log(ŷ) + (1-y) log(1-ŷ)]

Softmax:
P(y=k|x) = exp(θₖ^T x) / Σⱼ exp(θⱼ^T x)
```

### 03. Support Vector Machines

> 最大间隔分类与核技巧

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| Linear SVM | Maximum Margin | `01_linear_svm.ipynb` |
| Soft Margin | C Parameter | `02_soft_margin.ipynb` |
| Kernel SVM | RBF, Polynomial | `03_kernel_svm.ipynb` |
| SVM Regression | SVR, ε-insensitive | `04_svm_regression.ipynb` |

**Key Equations:**

```
SVM Optimization:
minimize: 1/2 ||w||² + C Σ ξᵢ
subject to: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ

Kernel Trick:
K(x, x') = φ(x)^T φ(x')

RBF Kernel:
K(x, x') = exp(-γ ||x - x'||²)
```

### 04. Decision Trees

> 决策树与树集成基础

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| CART Algorithm | Gini, Entropy | `01_cart.ipynb` |
| Tree Pruning | Pre/Post Pruning | `02_pruning.ipynb` |
| Feature Importance | Impurity-based | `03_feature_importance.ipynb` |
| Regression Trees | MSE Splitting | `04_regression_trees.ipynb` |

**Key Equations:**

```
Gini Impurity:
G = 1 - Σ pᵢ²

Entropy:
H = -Σ pᵢ log₂(pᵢ)

Information Gain:
IG = H(parent) - Σ (nⱼ/n) H(childⱼ)
```

### 05. Ensemble Learning

> 集成方法与 Boosting 算法

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| Voting & Bagging | Bootstrap, Aggregation | `01_voting_bagging.ipynb` |
| Random Forest | Feature Subsampling | `02_random_forest.ipynb` |
| AdaBoost | Sample Weighting | `03_adaboost.ipynb` |
| Gradient Boosting | Residual Learning | `04_gradient_boosting.ipynb` |
| XGBoost | Regularized Boosting | `05_xgboost.ipynb` |
| LightGBM | Histogram-based | `06_lightgbm.ipynb` |
| CatBoost | Categorical Features | `07_catboost.ipynb` |
| Stacking | Meta-learning | `08_stacking.ipynb` |

**Key Concepts:**

```
Bagging:
- Bootstrap sampling
- Parallel training
- Variance reduction

Boosting:
- Sequential training
- Error correction
- Bias reduction

XGBoost Objective:
L = Σ l(yᵢ, ŷᵢ) + Σ Ω(fₖ)
Ω(f) = γT + 1/2 λ||w||²
```

### 06. Dimensionality Reduction

> 降维技术与流形学习

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| PCA | Variance Maximization | `01_pca.ipynb` |
| Kernel PCA | Nonlinear PCA | `02_kernel_pca.ipynb` |
| LLE | Local Linear Embedding | `03_lle.ipynb` |
| t-SNE | Stochastic Neighbor | `04_tsne.ipynb` |
| UMAP | Uniform Manifold | `05_umap.ipynb` |

**Key Equations:**

```
PCA:
Maximize: Var(Xw) = w^T Σ w
Subject to: ||w|| = 1

t-SNE:
KL(P||Q) = Σᵢ Σⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)

pⱼ|ᵢ = exp(-||xᵢ-xⱼ||²/2σᵢ²) / Σₖ≠ᵢ exp(-||xᵢ-xₖ||²/2σᵢ²)
```

### 07. Unsupervised Learning

> 聚类与异常检测

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| K-Means | Lloyd's Algorithm | `01_kmeans.ipynb` |
| DBSCAN | Density-based | `02_dbscan.ipynb` |
| Gaussian Mixture | EM Algorithm | `03_gmm.ipynb` |
| Hierarchical Clustering | Agglomerative | `04_hierarchical.ipynb` |
| Anomaly Detection | Isolation Forest, LOF | `05_anomaly_detection.ipynb` |

**Key Equations:**

```
K-Means:
minimize: Σᵢ Σⱼ ||xⱼ - μᵢ||²

GMM:
p(x) = Σₖ πₖ N(x|μₖ, Σₖ)

EM Algorithm:
E-step: Compute responsibilities
M-step: Update parameters
```

### 08. End-to-End Project

> 完整机器学习项目流程

| Step | Activities | Deliverables |
|:-----|:-----------|:-------------|
| 1. Problem Definition | 问题定义、评估指标 | Problem Statement |
| 2. Data Collection | 数据获取、探索性分析 | EDA Report |
| 3. Data Preparation | 清洗、特征工程 | Feature Pipeline |
| 4. Modeling | 算法选择、训练 | Trained Models |
| 5. Evaluation | 验证、测试 | Performance Report |
| 6. Deployment | 模型部署 | API / Service |

---

## Recommended Learning Path

```
Week 1: Training Models
├── Linear Regression
├── Gradient Descent
└── Regularization

Week 2: Classification
├── Logistic Regression
├── Softmax Regression
└── Performance Metrics

Week 3: SVM & Decision Trees
├── SVM (Linear & Kernel)
└── Decision Trees (CART)

Week 4: Ensemble Learning
├── Random Forest
├── XGBoost / LightGBM
└── Stacking

Week 5: Unsupervised Learning
├── Dimensionality Reduction
└── Clustering

Week 6: End-to-End Project
└── Complete ML Pipeline
```

---

## Resources

### Textbooks
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.

### Papers
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*.

### Online Courses
- Stanford CS229: Machine Learning
- Coursera: Machine Learning Specialization

---

## Next Module

完成本模块后，请继续学习 [[Module 02: Neural Networks]]。

---

<div align="center">

**[[← Architecture|Architecture]]** | **[[Module 02 →|Module-02:-Neural-Networks]]**

</div>
