# 08 - 理论基础

深入理解机器学习的数学基础与理论保证。

## 模块概览

| 属性 | 值 |
|:-----|:---|
| **前置要求** | 线性代数, 微积分, 概率论 |
| **学习时长** | 持续学习 |
| **Notebooks** | 10+ |
| **难度** | ⭐⭐⭐⭐⭐ 理论深度 |

## 学习目标

完成本模块后，你将能够：

- ✅ 掌握线性代数在机器学习中的应用
- ✅ 理解概率论与统计推断基础
- ✅ 学习凸优化理论与算法
- ✅ 理解信息论在深度学习中的作用
- ✅ 了解统计学习理论与泛化界

---

## 子模块详解

### 01. 线性代数

机器学习的基础语言。

**核心概念**：

| 概念 | 应用 | 公式/性质 |
|:-----|:-----|:----------|
| **向量空间** | 特征表示 | 线性组合、基 |
| **矩阵乘法** | 神经网络前向传播 | $Y = XW$ |
| **特征值分解** | PCA | $A = Q\Lambda Q^T$ |
| **奇异值分解 (SVD)** | 矩阵分解、推荐系统 | $A = U\Sigma V^T$ |
| **范数** | 正则化 | $\|x\|_1, \|x\|_2$ |
| **梯度与雅可比** | 反向传播 | $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ |

**重要定理**：

**SVD 定理**：任意 $m \times n$ 矩阵 $A$ 可分解为

$$A = U\Sigma V^T$$

其中 $U$ 和 $V$ 是正交矩阵，$\Sigma$ 是对角矩阵。

**应用示例 - PCA**：

```python
def pca(X, n_components):
    # 中心化
    X_centered = X - X.mean(axis=0)

    # SVD 分解
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 投影到主成分
    X_pca = U[:, :n_components] @ np.diag(S[:n_components])

    return X_pca, Vt[:n_components]
```

---

### 02. 概率论与统计

不确定性建模的基础。

**核心概念**：

| 概念 | 公式 | 应用 |
|:-----|:-----|:-----|
| **贝叶斯定理** | $P(A\|B) = \frac{P(B\|A)P(A)}{P(B)}$ | 贝叶斯推断 |
| **期望** | $\mathbb{E}[X] = \sum_x xp(x)$ | 损失函数 |
| **方差** | $\text{Var}(X) = \mathbb{E}[(X-\mu)^2]$ | 不确定性度量 |
| **协方差** | $\text{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$ | 特征相关性 |
| **大数定律** | $\bar{X}_n \xrightarrow{P} \mu$ | 样本均值收敛 |
| **中心极限定理** | $\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)$ | 分布近似 |

**常见分布**：

| 分布 | 概率质量/密度函数 | 应用 |
|:-----|:------------------|:-----|
| **伯努利** | $P(X=1) = p$ | 二分类 |
| **高斯** | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | VAE, 噪声建模 |
| **多项分布** | $\frac{n!}{x_1!...x_k!}p_1^{x_1}...p_k^{x_k}$ | 多分类 |
| **泊松** | $\frac{\lambda^k e^{-\lambda}}{k!}$ | 计数数据 |

**最大似然估计 (MLE)**：

$$\hat{\theta}_{MLE} = \arg\max_\theta \prod_{i=1}^n p(x_i|\theta) = \arg\max_\theta \sum_{i=1}^n \log p(x_i|\theta)$$

**最大后验估计 (MAP)**：

$$\hat{\theta}_{MAP} = \arg\max_\theta p(\theta|X) = \arg\max_\theta [p(X|\theta)p(\theta)]$$

---

### 03. 优化理论

机器学习训练的核心。

**优化问题标准形式**：

$$\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, h_j(x) = 0$$

**梯度下降**：

$$x_{t+1} = x_t - \eta \nabla f(x_t)$$

**凸优化条件**：

| 条件 | 数学表述 | 意义 |
|:-----|:---------|:-----|
| **凸函数** | $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ | 无局部极小值 |
| **强凸** | $f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2$ | 唯一最优解 |
| **Lipschitz 连续** | $\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$ | 梯度平滑 |

**收敛率对比**：

| 方法 | 收敛率 | 条件 |
|:-----|:------:|:-----|
| 梯度下降 (GD) | $O(1/k)$ | 凸 + Lipschitz |
| GD | $O(e^{-\mu k/L})$ | 强凸 + 平滑 |
| 牛顿法 | $O(e^{-2^k})$ | 强凸 + 二阶平滑 |
| Adam | 经验良好 | 实际深度学习 |

**拉格朗日对偶**：

$$L(x, \lambda, \nu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \nu_j h_j(x)$$

**KKT 条件**（最优性必要条件）：

1. 平稳性：$\nabla f(x^*) + \sum_i \lambda_i \nabla g_i(x^*) + \sum_j \nu_j \nabla h_j(x^*) = 0$
2. 原始可行：$g_i(x^*) \leq 0, h_j(x^*) = 0$
3. 对偶可行：$\lambda_i \geq 0$
4. 互补松弛：$\lambda_i g_i(x^*) = 0$

---

### 04. 信息论

量化信息与不确定性。

**核心概念**：

| 概念 | 公式 | 解释 |
|:-----|:-----|:-----|
| **熵 (Entropy)** | $H(X) = -\sum_x p(x)\log p(x)$ | 不确定性度量 |
| **交叉熵** | $H(p, q) = -\sum_x p(x)\log q(x)$ | 分类损失 |
| **KL 散度** | $D_{KL}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$ | 分布差异 |
| **互信息** | $I(X;Y) = H(X) - H(X\|Y)$ | 相关性度量 |

**性质**：

- $H(X) \geq 0$，等号成立当且仅当 $X$ 确定
- $D_{KL}(p\|q) \geq 0$，等号成立当且仅当 $p = q$
- $H(p, q) = H(p) + D_{KL}(p\|q)$

**应用**：

```python
# 交叉熵损失（分类）
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

# KL 散度（VAE 正则化）
def kl_divergence(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

---

### 05. 统计学习理论

泛化能力的理论保证。

**经验风险最小化 (ERM)**：

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{n}\sum_{i=1}^n L(f(x_i), y_i)$$

**泛化误差分解**：

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

| 项 | 含义 | 如何减小 |
|:---|:-----|:---------|
| 偏差 | 模型拟合能力不足 | 增加模型复杂度 |
| 方差 | 对训练数据过拟合 | 正则化、更多数据 |
| 噪声 | 数据固有随机性 | 无法消除 |

**VC 维理论**：

对于 VC 维为 $d$ 的假设空间，泛化误差界为：

$$R(h) - \hat{R}(h) \leq O\left(\sqrt{\frac{d\log(n/d) + \log(1/\delta)}{n}}\right)$$

**Rademacher 复杂度**：

衡量假设空间对随机噪声的拟合能力。

$$\mathcal{R}_n(\mathcal{F}) = \mathbb{E}_\sigma\left[\sup_{f \in \mathcal{F}} \frac{1}{n}\sum_{i=1}^n \sigma_i f(x_i)\right]$$

---

### 06. 正则化理论

控制模型复杂度的理论基础。

**常见正则化方法**：

| 方法 | 形式 | 效果 |
|:-----|:-----|:-----|
| **L2 (Ridge)** | $\lambda\|\mathbf{w}\|_2^2$ | 参数平滑 |
| **L1 (Lasso)** | $\lambda\|\mathbf{w}\|_1$ | 稀疏解 |
| **Elastic Net** | $\lambda_1\|\mathbf{w}\|_1 + \lambda_2\|\mathbf{w}\|_2^2$ | 结合两者 |
| **Dropout** | 随机失活神经元 | 集成效应 |
| **Early Stopping** | 提前停止训练 | 隐式正则化 |

**贝叶斯视角**：

L2 正则化 = 高斯先验

$$p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \sigma^2 I)$$

L1 正则化 = 拉普拉斯先验

$$p(\mathbf{w}) = \prod_i \frac{1}{2b}e^{-|w_i|/b}$$

---

## 实验列表

| 实验 | 内容 | 文件 |
|:-----|:-----|:-----|
| 线性代数 | SVD 与 PCA | `01_linear_algebra.ipynb` |
| 概率论 | 分布可视化 | `02_probability.ipynb` |
| 优化算法 | 梯度下降变体对比 | `03_optimization.ipynb` |
| 凸优化 | 约束优化问题 | `04_convex_optimization.ipynb` |
| 信息论 | 熵与互信息 | `05_information_theory.ipynb` |
| 泛化理论 | 偏差-方差权衡 | `06_bias_variance.ipynb` |
| 正则化 | L1/L2 效果对比 | `07_regularization.ipynb` |

---

## 参考资源

### 教材
- Boyd & Vandenberghe (2004). *Convex Optimization* - [在线阅读](https://web.stanford.edu/~boyd/cvxbook/)
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*
- Goodfellow et al. (2016). *Deep Learning* - [在线阅读](https://www.deeplearningbook.org/)

### 论文
- Vapnik, V. (1998). Statistical Learning Theory
- Shalev-Shwartz & Ben-David (2014). Understanding Machine Learning: From Theory to Algorithms

### 课程
- [Stanford CS229](http://cs229.stanford.edu/) - Machine Learning
- [MIT 18.065](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) - Matrix Methods in Data Analysis
- [Stanford EE364a](https://web.stanford.edu/class/ee364a/) - Convex Optimization

### 在线资源
- [Matrix Calculus](http://www.matrixcalculus.org/) - 矩阵求导计算器
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
