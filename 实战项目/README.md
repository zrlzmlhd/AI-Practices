# 实战项目

将所学知识应用到真实场景，通过完整项目巩固技能。

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)

</div>

---

## 🚀 快速开始

```python
# 使用项目工具模块
from utils import set_seed, get_data_path, plot_training_history, plot_confusion_matrix

# 设置随机种子保证可重复性
set_seed(42)

# 使用跨平台路径加载数据
data_path = get_data_path('project_data')

# 训练后可视化结果
plot_training_history(history.history)
plot_confusion_matrix(y_true, y_pred, classes)
```

## 📚 模块简介

本模块提供端到端的实战项目，帮助你将理论知识转化为实际能力。每个项目都包含完整的流程：问题定义、数据处理、模型构建、评估优化和结果展示。

### 🎯 学习目标

- ✅ 掌握机器学习项目的完整流程
- ✅ 学会处理真实世界的数据
- ✅ 积累项目经验，建立作品集
- ✅ 为面试和工作做好准备

---

## 📂 项目列表

按照AI教材标准学习顺序排列：**机器学习基础 → 深度学习 → 计算机视觉 → 自然语言处理 → 时间序列 → 推荐系统 → 生成式AI**

---

## 🔢 第一部分：机器学习基础项目

### 1. 分类项目：Titanic生存预测 (入门)
**难度**: ⭐☆☆☆☆

**项目描述**: 预测泰坦尼克号乘客的生存概率，是最经典的机器学习入门项目。

**涉及技术**:
- 特征工程
- 缺失值处理
- 逻辑回归、决策树、随机森林

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | Titanic官方比赛 | [kaggle.com/c/titanic](https://www.kaggle.com/c/titanic) |
| GitHub | Titanic最佳解决方案 | [ageron/handson-ml3](https://github.com/ageron/handson-ml3/blob/main/03_classification.ipynb) |
| GitHub | 详细EDA教程 | [minsuk-heo/kaggle-titanic](https://github.com/minsuk-heo/kaggle-titanic) |

---

### 2. 回归项目：房价预测 (初级)
**难度**: ⭐⭐☆☆☆

**项目描述**: 预测波士顿/Ames地区的房价，学习回归模型和特征工程。

**涉及技术**:
- 线性回归、Ridge、Lasso
- 特征缩放和选择
- 交叉验证

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | House Prices比赛 | [kaggle.com/c/house-prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| GitHub | 详细解决方案 | [Shreyas3108/house-price-prediction](https://github.com/Shreyas3108/house-price-prediction) |
| GitHub | Stacking集成方案 | [krishnaraj30/ensemble-stacked-regressions](https://www.kaggle.com/code/krishnaraj30/ensemble-stacked-regressions-xgboost-lightgbm) |

---

### 3. 聚类项目：客户分群分析 (初级)
**难度**: ⭐⭐☆☆☆

**项目描述**: 使用无监督学习对客户进行分群，了解不同客户群体的特征。

**涉及技术**:
- K-Means聚类
- DBSCAN
- PCA降维可视化

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | Mall Customer数据集 | [kaggle.com/datasets/vjchoudhary7/customer-segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |
| GitHub | 客户分群完整教程 | [hduan2/customer_segmentation](https://github.com/hduan2/customer_segmentation) |
| GitHub | RFM分析方法 | [joaolage/RFM-analysis](https://github.com/joaolage/RFM-analysis) |

---

### 4. 集成学习项目：Otto分类挑战 (中级)
**难度**: ⭐⭐⭐☆☆

**项目描述**: 使用XGBoost、LightGBM等集成方法进行多分类。

**涉及技术**:
- XGBoost / LightGBM / CatBoost
- 模型集成 (Stacking, Blending)
- 超参数调优

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | Otto Group比赛 | [kaggle.com/c/otto-group](https://www.kaggle.com/c/otto-group-product-classification-challenge) |
| GitHub | XGBoost官方示例 | [dmlc/xgboost](https://github.com/dmlc/xgboost/tree/master/demo) |
| GitHub | LightGBM教程 | [microsoft/LightGBM](https://github.com/microsoft/LightGBM/tree/master/examples) |
| GitHub | Optuna调参 | [optuna/optuna-examples](https://github.com/optuna/optuna-examples) |

---

## 🖼️ 第二部分：计算机视觉项目

### 5. 图像分类：MNIST手写数字识别 (入门)
**难度**: ⭐⭐☆☆☆

**项目描述**: 识别手写数字0-9，是深度学习入门的经典项目。

**涉及技术**:
- CNN卷积神经网络
- 数据归一化
- Dropout正则化

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | Digit Recognizer比赛 | [kaggle.com/c/digit-recognizer](https://www.kaggle.com/c/digit-recognizer) |
| GitHub | TensorFlow官方教程 | [tensorflow/tutorials/quickstart](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb) |
| GitHub | PyTorch MNIST教程 | [pytorch/examples/mnist](https://github.com/pytorch/examples/tree/main/mnist) |
| Keras | Keras CNN示例 | [keras.io/examples/vision/mnist_convnet](https://keras.io/examples/vision/mnist_convnet/) |

---

### 6. 图像分类：猫狗分类器 (中级)
**难度**: ⭐⭐⭐☆☆

**项目描述**: 使用深度学习区分猫和狗的图像，学习迁移学习技术。

**涉及技术**:
- VGG/ResNet预训练模型
- 迁移学习和微调
- 数据增强

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | Dogs vs. Cats比赛 | [kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats) |
| GitHub | PyTorch图像分类教程 | [bentrevett/pytorch-image-classification](https://github.com/bentrevett/pytorch-image-classification) |
| Keras | Keras迁移学习指南 | [keras.io/guides/transfer_learning](https://keras.io/guides/transfer_learning/) |
| TensorFlow | TensorFlow迁移学习 | [tensorflow.org/tutorials/images/transfer_learning](https://www.tensorflow.org/tutorials/images/transfer_learning) |

---

### 7. 图像分类：CIFAR-10分类器 (中级)
**难度**: ⭐⭐⭐☆☆

**项目描述**: 识别10类常见物体（飞机、汽车、鸟类等）。

**涉及技术**:
- 深度CNN架构
- 批标准化
- 学习率调度

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | CIFAR-10数据集 | [kaggle.com/c/cifar-10](https://www.kaggle.com/c/cifar-10) |
| GitHub | PyTorch CIFAR教程 | [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) |
| GitHub | CNN架构实现 | [weiaicunzai/awesome-image-classification](https://github.com/weiaicunzai/awesome-image-classification) |

---

### 8. 目标检测系统 (高级)
**难度**: ⭐⭐⭐⭐☆

**项目描述**: 实现目标检测，识别图像中的物体位置和类别。

**涉及技术**:
- YOLO / Faster R-CNN
- 边界框回归
- 非极大值抑制

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | COCO数据集 | [kaggle.com/datasets/awsaf49/coco-2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) |
| GitHub | YOLOv5官方仓库 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) |
| GitHub | YOLOv8最新版 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| GitHub | Detectron2 | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) |

---

## 📝 第三部分：自然语言处理项目

### 9. 文本分类：情感分析 (初级)
**难度**: ⭐⭐☆☆☆

**项目描述**: 分析电影评论的情感倾向（正面/负面）。

**涉及技术**:
- 文本预处理
- 词嵌入 (Word2Vec, GloVe)
- LSTM/GRU

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | IMDB情感分析数据集 | [kaggle.com/datasets/lakshmi25npathi/imdb-dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| GitHub | BERT情感分析 | [google-research/bert](https://github.com/google-research/bert) |
| GitHub | 简单LSTM情感分析 | [bentrevett/pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis) |
| Keras | Keras文本分类教程 | [keras.io/examples/nlp/text_classification](https://keras.io/examples/nlp/text_classification_from_scratch/) |

---

### 10. NLP入门：灾难推文分类 (中级)
**难度**: ⭐⭐⭐☆☆

**项目描述**: 判断推文是否在描述真实灾难事件。

**涉及技术**:
- TF-IDF / Word2Vec
- TextCNN
- BERT微调

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | NLP Getting Started比赛 | [kaggle.com/c/nlp-getting-started](https://www.kaggle.com/c/nlp-getting-started) |
| GitHub | Hugging Face Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) |
| GitHub | 比赛解决方案集合 | [abhishekkrthakur/bert-sentiment](https://github.com/abhishekkrthakur/bert-sentiment) |

---

### 11. 序列标注：命名实体识别 (中级)
**难度**: ⭐⭐⭐☆☆

**项目描述**: 识别文本中的人名、地名、组织名等实体。

**涉及技术**:
- BiLSTM-CRF
- BERT for Token Classification
- 序列标注

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | CoNLL-2003 NER | [kaggle.com/datasets/juliangarratt/conll2003-dataset](https://www.kaggle.com/datasets/juliangarratt/conll2003-dataset) |
| GitHub | BERT NER | [kamalkraj/BERT-NER](https://github.com/kamalkraj/BERT-NER) |
| GitHub | BiLSTM-CRF实现 | [jiesutd/NCRFpp](https://github.com/jiesutd/NCRFpp) |

---

### 12. 对话系统：聊天机器人 (高级)
**难度**: ⭐⭐⭐⭐☆

**项目描述**: 构建一个简单的问答对话系统。

**涉及技术**:
- Seq2Seq模型
- 注意力机制
- Transformer

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | Cornell Movie对话数据集 | [kaggle.com/datasets/Cornell-University/movie-dialog](https://www.kaggle.com/datasets/Cornell-University/movie-dialog-corpus) |
| GitHub | PyTorch Chatbot教程 | [pytorch/tutorials/chatbot](https://github.com/pytorch/tutorials/blob/main/beginner_source/chatbot_tutorial.py) |
| GitHub | Rasa开源聊天框架 | [RasaHQ/rasa](https://github.com/RasaHQ/rasa) |
| 官方教程 | PyTorch Seq2Seq | [pytorch.org/tutorials/beginner/chatbot_tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html) |

---

## 📈 第四部分：时间序列项目

### 13. 时间序列分类：温度预测 (中级)
**难度**: ⭐⭐⭐☆☆

**项目描述**: 基于历史气象数据预测未来温度。

**涉及技术**:
- LSTM时间序列
- 多变量时间序列
- 滑动窗口

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | 气候数据集 | [kaggle.com/datasets/muthuj7/weather-dataset](https://www.kaggle.com/datasets/muthuj7/weather-dataset) |
| GitHub | LSTM时间序列预测 | [jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction](https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction) |
| TensorFlow | TensorFlow时间序列教程 | [tensorflow.org/tutorials/structured_data/time_series](https://www.tensorflow.org/tutorials/structured_data/time_series) |

---

### 14. 销量预测 (中级)
**难度**: ⭐⭐⭐☆☆

**项目描述**: 预测商品的未来销量，学习业务时间序列分析。

**涉及技术**:
- 时间序列分解
- Prophet模型
- 多步预测

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | Store Sales比赛 | [kaggle.com/c/store-sales-time-series-forecasting](https://www.kaggle.com/c/store-sales-time-series-forecasting) |
| GitHub | Facebook Prophet | [facebook/prophet](https://github.com/facebook/prophet) |
| GitHub | 时间序列工具集 | [unit8co/darts](https://github.com/unit8co/darts) |
| Kaggle | Rossmann Store Sales | [kaggle.com/c/rossmann-store-sales](https://www.kaggle.com/c/rossmann-store-sales) |

---

### 15. 股票价格预测 (高级)
**难度**: ⭐⭐⭐⭐☆

**项目描述**: 预测股票价格走势（注：仅供学习，不构成投资建议）。

**涉及技术**:
- LSTM/GRU时间序列
- 技术指标特征
- 多任务学习

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | 股票市场数据 | [kaggle.com/datasets/borismarjanovic/price-volume-data](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) |
| GitHub | Stock Prediction LSTM | [huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models) |
| GitHub | ML金融分析 | [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) |

---

## 🎮 第五部分：推荐系统项目

### 16. 电影推荐系统 (中级)
**难度**: ⭐⭐⭐☆☆

**项目描述**: 基于用户行为推荐电影。

**涉及技术**:
- 协同过滤
- 矩阵分解 (SVD, NMF)
- 深度学习推荐 (NCF)

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | MovieLens数据集 | [kaggle.com/datasets/grouplens/movielens](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) |
| GitHub | Surprise推荐库 | [NicolasHug/Surprise](https://github.com/NicolasHug/Surprise) |
| GitHub | Neural CF实现 | [hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering) |
| GitHub | Microsoft推荐系统 | [microsoft/recommenders](https://github.com/microsoft/recommenders) |

---

## 🎨 第六部分：生成式AI项目

### 17. 图像生成：DCGAN (高级)
**难度**: ⭐⭐⭐⭐☆

**项目描述**: 使用GAN生成逼真图像。

**涉及技术**:
- DCGAN架构
- 生成器/判别器训练
- 模式崩溃处理

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | CelebA人脸数据集 | [kaggle.com/datasets/jessicali9530/celeba](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) |
| GitHub | PyTorch-GAN实现大全 | [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) |
| GitHub | Keras-GAN实现 | [eriklindernoren/Keras-GAN](https://github.com/eriklindernoren/Keras-GAN) |
| 官方教程 | TensorFlow DCGAN教程 | [tensorflow.org/tutorials/generative/dcgan](https://www.tensorflow.org/tutorials/generative/dcgan) |
| 官方教程 | PyTorch DCGAN教程 | [pytorch.org/tutorials/beginner/dcgan_faces_tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) |

---

### 18. 文本生成：字符级LSTM (高级)
**难度**: ⭐⭐⭐⭐☆

**项目描述**: 使用LSTM生成文本（诗歌、代码等）。

**涉及技术**:
- 字符级语言模型
- 温度采样
- 序列生成

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| Kaggle | 莎士比亚文本 | [kaggle.com/datasets/kingburrito666/shakespeare-plays](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| GitHub | Char-RNN TensorFlow | [sherjilozair/char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow) |
| Keras | Keras文本生成教程 | [keras.io/examples/generative/lstm_character](https://keras.io/examples/generative/lstm_character_level_text_generation/) |

---

### 19. 风格迁移：Neural Style Transfer (高级)
**难度**: ⭐⭐⭐⭐⭐

**项目描述**: 将一张图像的艺术风格迁移到另一张图像。

**涉及技术**:
- VGG特征提取
- 内容损失/风格损失
- 优化算法

**资源链接**:

| 类型 | 名称 | 链接 |
|-----|------|------|
| GitHub | Fast Neural Style | [jcjohnson/fast-neural-style](https://github.com/jcjohnson/fast-neural-style) |
| GitHub | PyTorch实现 | [leongatys/PytorchNeuralStyleTransfer](https://github.com/leongatys/PytorchNeuralStyleTransfer) |
| TensorFlow | TensorFlow风格迁移 | [tensorflow.org/tutorials/generative/style_transfer](https://www.tensorflow.org/tutorials/generative/style_transfer) |

---

## 🗺️ 项目学习路径

### 初学者路径 (4-6周)

```
Titanic生存预测 → 房价预测 → MNIST识别 → 情感分析
```

### 进阶路径 (6-8周)

```
客户分群 → 猫狗分类 → NLP灾难推文 → 销量预测 → 电影推荐
```

### 高级路径 (8-10周)

```
集成学习 → 目标检测 → 命名实体识别 → 股票预测 → GAN图像生成
```

---

## 📋 项目模板

每个项目应包含以下结构：

```
项目名称/
├── README.md           # 项目说明
├── data/              # 数据目录
│   ├── raw/           # 原始数据
│   └── processed/     # 处理后数据
├── notebooks/         # Jupyter notebooks
│   ├── 01_数据探索.ipynb
│   ├── 02_数据预处理.ipynb
│   ├── 03_模型训练.ipynb
│   └── 04_模型评估.ipynb
├── src/               # 源代码
│   ├── data.py        # 数据处理
│   ├── model.py       # 模型定义
│   ├── train.py       # 训练脚本
│   └── evaluate.py    # 评估脚本
├── models/            # 保存的模型
├── results/           # 结果和图表
└── requirements.txt   # 项目依赖
```

---

## 💡 项目实施建议

### 1. 理解问题

在开始编码前，确保你理解：
- 问题类型（分类/回归/聚类）
- 评估指标
- 业务约束

### 2. 探索数据

花足够时间了解数据：
- 数据分布
- 缺失值
- 异常值
- 特征相关性

### 3. 建立基线

先实现简单模型作为基线：
- 随机猜测
- 简单规则
- 基础算法

### 4. 迭代改进

逐步改进模型：
- 特征工程
- 算法选择
- 超参数调优
- 模型集成

### 5. 记录过程

详细记录实验：
- 尝试的方法
- 效果对比
- 失败原因
- 最佳配置

---

## 📚 推荐资源

### 数据集来源

| 平台 | 链接 | 说明 |
|-----|------|------|
| Kaggle Datasets | [kaggle.com/datasets](https://www.kaggle.com/datasets) | 最全面的数据集平台 |
| UCI ML Repository | [archive.ics.uci.edu/ml](https://archive.ics.uci.edu/ml/) | 经典学术数据集 |
| Google Dataset Search | [datasetsearch.research.google.com](https://datasetsearch.research.google.com/) | 谷歌数据集搜索 |
| 天池数据集 | [tianchi.aliyun.com/dataset](https://tianchi.aliyun.com/dataset) | 中文数据集平台 |
| Hugging Face | [huggingface.co/datasets](https://huggingface.co/datasets) | NLP数据集 |

### 竞赛平台

| 平台 | 链接 | 特点 |
|-----|------|------|
| Kaggle | [kaggle.com](https://www.kaggle.com/) | 全球最大ML竞赛平台 |
| 天池 | [tianchi.aliyun.com](https://tianchi.aliyun.com/) | 阿里巴巴竞赛平台 |
| DataFountain | [datafountain.cn](https://www.datafountain.cn/) | 中国数据竞赛平台 |

### 论文和代码

| 资源 | 链接 | 说明 |
|-----|------|------|
| Papers With Code | [paperswithcode.com](https://paperswithcode.com/) | 论文+代码实现 |
| GitHub Awesome Lists | [github.com/topics/awesome](https://github.com/topics/awesome) | 精选资源列表 |

---

## 🎓 技能检查清单

完成项目后，检查是否掌握：

- [ ] 数据收集和清洗
- [ ] 探索性数据分析 (EDA)
- [ ] 特征工程
- [ ] 模型选择和训练
- [ ] 超参数调优
- [ ] 模型评估和对比
- [ ] 结果可视化
- [ ] 代码组织和文档

---

## 🤝 贡献项目

欢迎贡献新的实战项目！请参考[贡献指南](../CONTRIBUTING.md)。

---

准备好开始你的第一个实战项目了吗？选择一个合适的项目，开始动手吧！

[返回主页](../README.md)
