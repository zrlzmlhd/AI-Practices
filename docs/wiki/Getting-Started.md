# Getting Started

本指南将帮助你快速上手 AI-Practices 项目，从环境配置到运行第一个实验。

---

## Prerequisites

### 系统要求

| Component | Minimum | Recommended |
|:----------|:--------|:------------|
| **OS** | Windows 10 / macOS 10.15 / Ubuntu 18.04 | Windows 11 / macOS 13+ / Ubuntu 22.04 |
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 32 GB |
| **GPU** | GTX 1060 (6GB) | RTX 3080+ (10GB+) |
| **Storage** | 50 GB | 200 GB SSD |
| **Python** | 3.10 | 3.10 ~ 3.11 |

### 必备软件

1. **Git** - 版本控制
   ```bash
   git --version  # >= 2.30
   ```

2. **Conda** - 环境管理 (推荐 Miniconda)
   ```bash
   conda --version  # >= 23.0
   ```

3. **CUDA** (可选，GPU 用户)
   ```bash
   nvcc --version  # >= 11.8
   nvidia-smi      # 验证 GPU
   ```

---

## Installation

### Step 1: Clone Repository

```bash
# HTTPS
git clone https://github.com/zimingttkx/AI-Practices.git

# SSH (推荐)
git clone git@github.com:zimingttkx/AI-Practices.git

cd AI-Practices
```

### Step 2: Create Environment

```bash
# 创建 Conda 环境
conda create -n ai-practices python=3.10 -y

# 激活环境
conda activate ai-practices
```

### Step 3: Install Dependencies

```bash
# 安装核心依赖
pip install -r requirements.txt

# GPU 支持 (可选)
# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorFlow with CUDA
pip install tensorflow[and-cuda]
```

### Step 4: Verify Installation

```bash
# 验证 TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import tensorflow as tf; print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')"

# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# 验证 Scikit-learn
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
```

---

## Project Structure

```
AI-Practices/
│
├── 01-foundations/                 # 机器学习基础
│   ├── 01-training-models/         # 模型训练方法
│   ├── 02-classification/          # 分类算法
│   ├── 03-support-vector-machines/ # SVM
│   ├── 04-decision-trees/          # 决策树
│   ├── 05-ensemble-learning/       # 集成学习
│   ├── 06-dimensionality-reduction/# 降维
│   ├── 07-unsupervised-learning/   # 无监督学习
│   └── 08-end-to-end-project/      # 端到端项目
│
├── 02-neural-networks/             # 神经网络
├── 03-computer-vision/             # 计算机视觉
├── 04-sequence-models/             # 序列模型
├── 05-advanced-topics/             # 高级专题
├── 06-generative-models/           # 生成模型
├── 07-reinforcement-learning/      # 强化学习
├── 08-theory-notes/                # 理论笔记
├── 09-practical-projects/          # 实战项目
│
├── utils/                          # 工具库
│   ├── common.py                   # 通用函数
│   ├── paths.py                    # 路径管理
│   ├── visualization.py            # 可视化
│   └── metrics/                    # 评估指标
│
├── requirements.txt                # Python 依赖
├── environment.yml                 # Conda 环境
└── README.md                       # 项目文档
```

---

## Running Your First Experiment

### Example 1: Jupyter Notebook

```bash
# 启动 Jupyter Lab
jupyter lab --port=8888

# 在浏览器中打开
# http://localhost:8888
```

然后导航到 `01-foundations/01-training-models/` 打开第一个 notebook。

### Example 2: MNIST CNN Project

```bash
# 进入项目目录
cd 09-practical-projects/02-computer-vision/01-mnist-cnn

# 训练模型
python src/train.py --epochs 20 --batch_size 64

# 评估模型
python src/evaluate.py --checkpoint best_model.pt

# 查看训练曲线
tensorboard --logdir=runs/logs
```

**预期输出**:
```
Epoch 20/20 - loss: 0.0234 - accuracy: 0.9921
Test Accuracy: 99.12%
```

### Example 3: XGBoost Classification

```bash
# 进入项目目录
cd 09-practical-projects/01-ml-basics/01-titanic-survival-xgboost

# 运行训练
python train.py

# 或打开 Jupyter Notebook
jupyter lab notebooks/titanic_xgboost.ipynb
```

---

## Learning Path

### 初学者路径 (8-12 周)

```
Week 1-2:  01-foundations (Part 1: Training Models, Classification)
Week 3-4:  01-foundations (Part 2: SVM, Decision Trees, Ensemble)
Week 5-6:  02-neural-networks (Keras Introduction, Training Techniques)
Week 7-8:  03-computer-vision (CNN Basics, Classic Architectures)
Week 9-10: 04-sequence-models (RNN/LSTM, Text Processing)
Week 11-12: 09-practical-projects (ML Basics, CV Projects)
```

### 进阶路径 (4-6 周)

```
Week 1-2: 05-advanced-topics (Functional API, Optimization)
Week 3-4: 06-generative-models (GAN, Text Generation)
Week 5-6: 07-reinforcement-learning (DQN, Policy Gradient)
```

### 竞赛路径 (持续)

```
09-practical-projects/05-kaggle-competitions/
├── Feature Engineering Techniques
├── Model Ensembling Strategies
├── Cross-Validation Best Practices
└── Gold Medal Solutions Analysis
```

---

## Troubleshooting

### 常见问题

#### 1. CUDA 不可用
```bash
# 检查 CUDA 版本
nvcc --version
nvidia-smi

# 重新安装匹配版本的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 2. 内存不足 (OOM)
```python
# TensorFlow: 设置内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PyTorch: 减小 batch size
train_loader = DataLoader(dataset, batch_size=16)  # 减小 batch_size
```

#### 3. 包版本冲突
```bash
# 重建环境
conda deactivate
conda env remove -n ai-practices
conda create -n ai-practices python=3.10 -y
conda activate ai-practices
pip install -r requirements.txt
```

更多问题请参考 [[FAQ]] 和 [[Troubleshooting]]。

---

## Next Steps

1. **[[Installation]]** - 详细安装指南
2. **[[Architecture]]** - 了解系统架构
3. **[[Module 01: Foundations]]** - 开始学习机器学习基础
4. **[[Best Practices]]** - 最佳实践指南

---

<div align="center">

**[[← Home|Home]]** | **[[Installation →|Installation]]**

</div>
