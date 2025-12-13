# 01 - 机器学习基础

建立坚实的机器学习理论基础，掌握经典算法的原理与实现。

## 模块概览

| 属性 | 值 |
|:-----|:---|
| **前置要求** | Python, NumPy, 线性代数, 微积分 |
| **学习时长** | 4-6 周 |
| **Notebooks** | 25+ |
| **难度** | ⭐⭐ 初级到中级 |

## 学习目标

完成本模块后，你将能够：

- ✅ 理解机器学习的核心概念和数学基础
- ✅ 从零实现线性回归、逻辑回归、SVM 等经典算法
- ✅ 掌握模型评估、交叉验证和超参数调优
- ✅ 理解集成学习并熟练应用 XGBoost、LightGBM
- ✅ 应用降维和聚类等无监督学习技术

## 子模块详解

### 01. Training Models | 模型训练

学习机器学习模型训练的核心方法。

| 主题 | 内容 | 复杂度 |
|:-----|:-----|:------:|
| Linear Regression | 最小二乘法、正规方程 | O(nd²) |
| Gradient Descent | 批量/随机/小批量梯度下降 | O(nd) |
| Regularization | L1 (Lasso) / L2 (Ridge) / ElasticNet | O(nd²) |

**核心公式 - 线性回归正规方程：**

$$\hat{\theta} = (X^TX)^{-1}X^Ty$$

**梯度下降更新规则：**

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)$$

### 02. Classification | 分类算法

掌握经典分类算法的原理与实现。

| 主题 | 内容 | 关键概念 |
|:-----|:-----|:---------|
| Logistic Regression | 二分类、多分类 | Sigmoid, Softmax |
| Performance Metrics | 精确率、召回率、F1、AUC | 混淆矩阵 |

**Sigmoid 函数：**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**交叉熵损失：**

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right]$$

### 03. Support Vector Machines | 支持向量机

理解最大间隔分类器的数学原理。

| 主题 | 内容 | 关键概念 |
|:-----|:-----|:---------|
| Linear SVM | 硬间隔、软间隔 | 最大间隔 |
| Kernel SVM | RBF、多项式核 | 核技巧 |

**SVM 优化目标：**

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{m}\xi_i$$

$$\text{s.t. } y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

### 04. Decision Trees | 决策树

学习基于树的模型构建方法。

| 主题 | 内容 | 关键概念 |
|:-----|:-----|:---------|
| CART Algorithm | 分类与回归树 | 基尼系数、信息增益 |
| Feature Importance | 特征重要性分析 | 排列重要性 |

**信息增益：**

$$IG(D, A) = H(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} H(D_v)$$

### 05. Ensemble Learning | 集成学习

掌握集成方法提升模型性能。

| 主题 | 内容 | 关键概念 |
|:-----|:-----|:---------|
| Bagging | Random Forest | Bootstrap 采样 |
| Boosting | AdaBoost, GBDT | 残差学习 |
| XGBoost / LightGBM | 工业级梯度提升 | 正则化、直方图 |
| Stacking | 模型堆叠 | 元学习器 |

**GBDT 残差学习：**

$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

其中 $h_m(x)$ 拟合负梯度（伪残差）：

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

### 06. Dimensionality Reduction | 降维

学习高维数据的降维技术。

| 主题 | 内容 | 复杂度 |
|:-----|:-----|:------:|
| PCA | 主成分分析 | O(min(n²d, nd²)) |
| t-SNE | 可视化降维 | O(n²) |
| UMAP | 流形学习 | O(n^1.14) |

**PCA 优化目标：**

$$\max_{\mathbf{u}} \mathbf{u}^T \Sigma \mathbf{u} \quad \text{s.t. } \|\mathbf{u}\| = 1$$

### 07. Unsupervised Learning | 无监督学习

掌握聚类和异常检测方法。

| 主题 | 内容 | 关键概念 |
|:-----|:-----|:---------|
| K-Means | 划分聚类 | 质心、惯性 |
| DBSCAN | 密度聚类 | 核心点、边界点 |
| GMM | 高斯混合模型 | EM 算法 |

## 代码示例

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# 岭回归示例
ridge = Ridge(alpha=1.0)
scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
print(f"CV R² Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

## 参考资料

### 教材
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.)
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*

### 论文
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.

### 在线资源
- [Scikit-Learn 官方文档](https://scikit-learn.org/stable/)
- [StatQuest with Josh Starmer](https://www.youtube.com/c/joshstarmer)
