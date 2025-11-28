# 机器学习实战

欢迎来到机器学习实战模块！本模块涵盖从基础机器学习算法到深度学习框架的完整学习路径。

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

</div>

---

## 🚀 快速开始

```python
# 使用项目工具模块
from utils import set_seed, get_data_path, plot_training_history, plot_confusion_matrix

# 设置随机种子保证可重复性
set_seed(42)

# 使用跨平台路径加载数据
data_path = get_data_path('datasets')

# 训练后可视化
plot_training_history(history.history)
plot_confusion_matrix(y_true, y_pred, classes)
```

## 📚 模块简介

本模块分为两个主要部分：

1. **机器学习基础知识** - 经典机器学习算法
2. **神经网络和深度学习** - 深度学习框架和技术

### 🎯 学习目标

完成本模块后，你将：

- ✅ 掌握核心机器学习算法
- ✅ 理解模型训练和评估的完整流程
- ✅ 熟练使用Scikit-learn库
- ✅ 掌握TensorFlow和Keras框架
- ✅ 能够构建和训练深度神经网络
- ✅ 理解卷积神经网络和循环神经网络

## 📂 目录结构

```
机器学习实战/
├── 机器学习基础知识/          # 第一部分：经典机器学习
│   ├── 训练模型/              # 线性回归、逻辑回归等
│   ├── Decision Tree/        # 决策树
│   ├── Support Vector Machine/  # 支持向量机
│   ├── 降维/                  # PCA、t-SNE等
│   ├── 无监督学习/            # 聚类算法
│   ├── 集成学习和随机森林/    # 集成方法
│   ├── 分类/                  # MNIST分类
│   └── 端到端机器学习项目/    # 完整项目案例
│
└── 神经网络和深度学习/         # 第二部分：深度学习
    ├── Keras人工神经网络简介/
    ├── Tensorflow加载和预处理数据/
    ├── 使用Tensorflow自定义模型和训练/
    ├── 使用卷积神经网络的深度计算机视觉/
    ├── 使用RNN和CNN处理序列/
    ├── 使用RNN和注意力机制进行NLP/
    └── 训练深度学习网络/
```

## 📖 第一部分：机器学习基础知识

### 1. 训练模型

学习各种线性模型及其训练方法。

**内容：**
- ✓ 线性回归 (LinearRegression)
- ✓ 多项式回归 (Polynomial Regression)
- ✓ 岭回归 (Ridge Regression)
- ✓ Lasso回归
- ✓ ElasticNet
- ✓ 逻辑回归 (Logistic Regression)
- ✓ 学习曲线分析
- ✓ 梯度下降算法

**关键概念：**
- 正规方程法
- 批量梯度下降
- 随机梯度下降
- 小批量梯度下降
- 正则化 (L1/L2)

**实战项目：**
- 房价预测
- 分类问题

**优质学习资源：**

| 资源类型 | 名称 | 链接 |
|---------|------|------|
| GitHub | Hands-On ML第三版 | [ageron/handson-ml3](https://github.com/ageron/handson-ml3) |
| Kaggle | 房价预测比赛 | [kaggle.com/c/house-prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| GitHub | Scikit-learn官方示例 | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn/tree/main/examples) |

📄 [详细笔记](机器学习基础知识/训练模型/第四章笔记.md)

---

### 2. 决策树 (Decision Tree)

理解决策树算法及其应用。

**内容：**
- ✓ 决策树分类
- ✓ 决策树回归
- ✓ 树的可视化
- ✓ 特征重要性分析

**关键概念：**
- 基尼不纯度
- 信息增益
- 剪枝技术
- 过拟合控制

**实战案例：**
- 鸢尾花分类
- 回归任务

**优质学习资源：**

| 资源类型 | 名称 | 链接 |
|---------|------|------|
| GitHub | 决策树可视化工具 | [parrt/dtreeviz](https://github.com/parrt/dtreeviz) |
| Kaggle | Titanic经典决策树 | [kaggle.com/c/titanic](https://www.kaggle.com/c/titanic) |

📄 [详细笔记](机器学习基础知识/Decision%20Tree/Decision%20Tree%20笔记.md)

---

### 3. 支持向量机 (SVM)

掌握强大的SVM算法。

**内容：**
- ✓ 线性SVM分类
- ✓ 非线性SVM
- ✓ 多项式核
- ✓ RBF核
- ✓ SVM回归

**关键概念：**
- 支持向量
- 核技巧
- 软间隔
- 超参数调优

**实战案例：**
- 二分类问题
- 多分类问题
- 回归任务

**优质学习资源：**

| 资源类型 | 名称 | 链接 |
|---------|------|------|
| GitHub | SVM教程 | [jakevdp/PythonDataScienceHandbook](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.07-Support-Vector-Machines.ipynb) |
| 官方文档 | Scikit-learn SVM | [scikit-learn.org/stable/modules/svm](https://scikit-learn.org/stable/modules/svm.html) |

📄 [详细笔记](机器学习基础知识/Support%20Vector%20Machine/第五章笔记.md)

---

### 4. 降维

学习降维技术以处理高维数据。

**内容：**
- ✓ 主成分分析 (PCA)
- ✓ 增量PCA
- ✓ 核PCA
- ✓ 随机PCA
- ✓ 局部线性嵌入 (LLE)

**关键概念：**
- 方差保留
- 主成分
- 流形学习
- 维度诅咒

**应用场景：**
- 数据可视化
- 特征工程
- 噪声过滤

**优质学习资源：**

| 资源类型 | 名称 | 链接 |
|---------|------|------|
| GitHub | t-SNE可视化 | [oreillymedia/t-SNE-tutorial](https://github.com/oreillymedia/t-SNE-tutorial) |
| GitHub | UMAP降维工具 | [lmcinnes/umap](https://github.com/lmcinnes/umap) |

📄 [详细笔记](机器学习基础知识/降维/降维.md)

---

### 5. 无监督学习

探索无监督学习算法。

**内容：**
- ✓ K-Means聚类
- ✓ Mini-Batch K-Means
- ✓ DBSCAN密度聚类
- ✓ 层次聚类
- ✓ 图像分割应用

**关键概念：**
- 聚类中心
- 簇内方差
- 轮廓系数
- 密度可达

**实战项目：**
- 客户分群
- 图像分割
- 异常检测

**优质学习资源：**

| 资源类型 | 名称 | 链接 |
|---------|------|------|
| Kaggle | 客户分群数据集 | [kaggle.com/datasets/vjchoudhary7/customer-segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |
| GitHub | 聚类算法合集 | [scikit-learn-contrib/hdbscan](https://github.com/scikit-learn-contrib/hdbscan) |

📄 [详细笔记](机器学习基础知识/无监督学习/无监督学习.md)

---

### 6. 集成学习和随机森林

掌握集成学习方法。

**内容：**
- ✓ 投票分类器
- ✓ Bagging和Pasting
- ✓ 随机森林
- ✓ AdaBoost
- ✓ Gradient Boosting
- ✓ XGBoost

**关键概念：**
- Bootstrap
- 袋外评估
- 特征重要性
- 提升算法

**性能对比：**
- 各算法优缺点
- 适用场景分析
- 调参技巧

**优质学习资源：**

| 资源类型 | 名称 | 链接 |
|---------|------|------|
| GitHub | XGBoost官方仓库 | [dmlc/xgboost](https://github.com/dmlc/xgboost) |
| GitHub | LightGBM官方仓库 | [microsoft/LightGBM](https://github.com/microsoft/LightGBM) |
| GitHub | CatBoost官方仓库 | [catboost/catboost](https://github.com/catboost/catboost) |
| Kaggle | Stacking集成示例 | [kaggle.com/code/serigne/stacked-regressions](https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard) |

📄 [详细笔记](机器学习基础知识/集成学习和随机森林/第七章笔记.md)

---

### 7. 分类任务

实战MNIST手写数字识别。

**内容：**
- ✓ 数据加载和预处理
- ✓ 二分类器构建
- ✓ 多分类策略
- ✓ 性能评估

**评估指标：**
- 准确率
- 精确率和召回率
- F1分数
- 混淆矩阵

**优质学习资源：**

| 资源类型 | 名称 | 链接 |
|---------|------|------|
| Kaggle | Digit Recognizer比赛 | [kaggle.com/c/digit-recognizer](https://www.kaggle.com/c/digit-recognizer) |
| GitHub | MNIST多种实现 | [hsjeong5/MNIST-for-Numpy](https://github.com/hsjeong5/MNIST-for-Numpy) |

---

### 8. 端到端机器学习项目

完整的项目实战。

**项目：** 波士顿房价预测

**流程：**
1. 问题定义
2. 数据获取和探索
3. 数据预处理
4. 模型选择
5. 模型训练
6. 模型调优
7. 结果展示

---

## 🤖 第二部分：神经网络和深度学习

### 1. Keras人工神经网络简介

入门深度学习框架。

**内容：**
- ✓ Sequential API基础
- ✓ Functional API
- ✓ Subclassing API
- ✓ 模型保存和加载
- ✓ 回调函数
- ✓ TensorBoard可视化

**关键技术：**
- 层的定义
- 模型编译
- 模型训练
- 超参数调优

**优质学习资源：**

| 资源类型 | 名称 | 链接 |
|---------|------|------|
| GitHub | Keras官方示例 | [keras-team/keras-io](https://github.com/keras-team/keras-io/tree/master/examples) |
| 官方文档 | Keras指南 | [keras.io/guides](https://keras.io/guides/) |
| GitHub | Deep Learning with Python | [fchollet/deep-learning-with-python-notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks) |

📄 [详细笔记](神经网络和深度学习/Keras人工神经网络简介/Keras神经网络简介.md)

---

### 2. TensorFlow加载和预处理数据

掌握数据处理技术。

**内容：**
- ✓ TFRecord格式
- ✓ tf.data API
- ✓ 数据增强
- ✓ 预处理层
- ✓ CSV数据处理

**技术要点：**
- 数据管道优化
- 并行处理
- 缓存策略

📄 [详细笔记](神经网络和深度学习/Tensorflow加载和预处理数据/TensorFlow加载和预处理数据.md)

---

### 3. 使用TensorFlow自定义模型和训练

深入理解TensorFlow。

**内容：**
- ✓ 自定义层
- ✓ 自定义损失函数
- ✓ 自定义训练循环
- ✓ 自定义指标

**高级主题：**
- 梯度带 (GradientTape)
- 自动微分
- 图模式和急切模式

📄 [详细笔记](神经网络和深度学习/使用Tensorflow自定义模型和训练/Tensorflow高度自定义化.md)

---

### 4. 使用卷积神经网络的深度计算机视觉

学习CNN架构和应用。

**内容：**
- ✓ 卷积层原理
- ✓ 池化层
- ✓ CNN架构设计
- ✓ ResNet实现
- ✓ 迁移学习
- ✓ 花卉分类项目

**经典架构：**
- LeNet-5
- AlexNet
- VGG
- ResNet
- Inception

**实战项目：**
- 图像分类
- 迁移学习应用

**优质学习资源：**

| 资源类型 | 名称 | 链接 |
|---------|------|------|
| GitHub | PyTorch图像分类 | [bentrevett/pytorch-image-classification](https://github.com/bentrevett/pytorch-image-classification) |
| Kaggle | Dogs vs. Cats | [kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats) |
| 官方教程 | TensorFlow迁移学习 | [tensorflow.org/tutorials/images/transfer_learning](https://www.tensorflow.org/tutorials/images/transfer_learning) |

📄 [详细笔记](神经网络和深度学习/使用卷积神经网络的深度计算机视觉/计算机视觉.md)

---

### 5. 使用RNN和CNN处理序列

序列数据处理技术。

**内容：**
- ✓ SimpleRNN
- ✓ LSTM
- ✓ GRU
- ✓ 1D CNN
- ✓ WaveNet
- ✓ 时间序列预测

**应用场景：**
- 时间序列预测
- 序列分类
- 序列生成

---

### 6. 使用RNN和注意力机制进行NLP

自然语言处理入门。

**内容：**
- ✓ 文本预处理
- ✓ 词嵌入
- ✓ LSTM文本分类
- ✓ 注意力机制基础

---

### 7. 训练深度学习网络

优化训练过程。

**内容：**
- ✓ 批标准化 (Batch Normalization)
- ✓ Dropout正则化
- ✓ 学习率调度
- ✓ 优化器比较
- ✓ 初始化策略
- ✓ 迁移学习

**优化技巧：**
- 梯度裁剪
- 权重初始化
- 学习率衰减
- 早停法

📄 [详细笔记1](神经网络和深度学习/训练深度学习网络/深度学习网络.md)
📄 [详细笔记2](神经网络和深度学习/训练深度学习网络/优化器比较.md)

---

## 🗺️ 推荐学习路径

### 初级路径 (2-3周)

适合机器学习初学者：

```
训练模型 → Decision Tree → 无监督学习 → 分类任务
```

### 中级路径 (3-4周)

有一定ML基础的学习者：

```
SVM → 降维 → 集成学习 → 端到端项目 → Keras基础
```

### 高级路径 (4-6周)

进阶深度学习：

```
Keras → TensorFlow数据处理 → 自定义模型 → CNN → RNN → 训练优化
```

## 💡 学习建议

### 1. 循序渐进
- 按照推荐路径学习
- 不要跳过基础内容
- 确保理解后再继续

### 2. 动手实践
- 运行所有代码
- 修改参数观察变化
- 尝试不同数据集

### 3. 记录笔记
- 记录关键概念
- 总结算法优缺点
- 记录遇到的问题

### 4. 项目驱动
- 尝试实际项目
- 参加Kaggle竞赛
- 解决实际问题

## 📚 参考资料

### 推荐书籍
- 《机器学习实战》 - Peter Harrington
- 《深度学习》 - Ian Goodfellow
- 《Python机器学习》 - Sebastian Raschka

### 在线资源
- [Scikit-learn文档](https://scikit-learn.org/)
- [TensorFlow教程](https://www.tensorflow.org/tutorials)
- [Keras文档](https://keras.io/)

### 视频课程
- Andrew Ng的机器学习课程
- Fast.ai深度学习课程

## 🔧 环境要求

### 软件要求
- Python 3.10+
- TensorFlow 2.13+
- Scikit-learn 1.3+
- NumPy 1.24+
- Matplotlib 3.7+

### 硬件建议
- CPU: 4核以上
- RAM: 8GB以上
- GPU: 可选，但对深度学习有帮助

## 📝 练习题

每个章节都包含练习题，建议完成：

1. **基础练习**: 修改代码参数，观察结果
2. **进阶练习**: 应用算法到新数据集
3. **挑战练习**: 改进模型性能

## 🐛 常见问题

### Q: 应该先学机器学习还是深度学习？
A: 建议先学习机器学习基础，理解核心概念后再学习深度学习。

### Q: 需要数学基础吗？
A: 基础的线性代数、微积分和概率论会很有帮助，但不是必需的。

### Q: 应该使用哪个深度学习框架？
A: 本教程主要使用TensorFlow/Keras，它们易学且功能强大。

### Q: 如何调试模型？
A: 使用TensorBoard可视化训练过程，检查学习曲线，分析错误案例。

## 🤝 贡献

发现错误或有改进建议？请查看[贡献指南](../CONTRIBUTING.md)。

---

祝学习愉快！如有问题，欢迎提issue讨论。

[返回主页](../README.md)
