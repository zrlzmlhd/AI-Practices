# 09 - 实战项目

从理论到实践，通过真实项目检验和巩固所学知识。

## 模块概览

| 属性 | 值 |
|:-----|:---|
| **前置要求** | 完成 01-04 模块学习 |
| **项目数量** | 19+ 个完整项目 |
| **难度范围** | ⭐⭐ ~ ⭐⭐⭐⭐⭐ |
| **特色** | 含 Kaggle 金牌方案 |

## 项目分类

### 📊 01-ML 基础项目

入门级机器学习项目，适合初学者练手。

| 项目 | 描述 | 技术栈 | 难度 |
|:-----|:-----|:-------|:----:|
| **Titanic 生存预测** | 经典二分类问题，学习特征工程 | XGBoost, Pandas | ⭐⭐ |
| **Otto 产品分类** | 多分类问题，学习集成方法 | LightGBM, Stacking | ⭐⭐⭐ |
| **House Prices 预测** | 回归问题，学习数据预处理 | Ridge, Lasso, XGBoost | ⭐⭐ |

### 👁️ 02-计算机视觉项目

图像识别与处理相关项目。

| 项目 | 描述 | 技术栈 | 难度 |
|:-----|:-----|:-------|:----:|
| **MNIST 手写数字识别** | CNN 入门经典项目 | TensorFlow, Keras | ⭐⭐ |
| **CIFAR-10 图像分类** | 多类别图像分类 | ResNet, Data Augmentation | ⭐⭐⭐ |
| **图像风格迁移** | 神经风格迁移实现 | VGG19, PyTorch | ⭐⭐⭐ |

### 📝 03-NLP 项目

自然语言处理相关项目。

| 项目 | 描述 | 技术栈 | 难度 |
|:-----|:-----|:-------|:----:|
| **LSTM 情感分析** | 电影评论情感分类 | LSTM, Word2Vec | ⭐⭐⭐ |
| **Transformer 文本分类** | 基于注意力机制的分类 | Transformer, PyTorch | ⭐⭐⭐⭐ |
| **命名实体识别 (NER)** | 序列标注任务 | BiLSTM-CRF, BERT | ⭐⭐⭐⭐ |
| **文本摘要生成** | Seq2Seq 摘要模型 | T5, Transformers | ⭐⭐⭐⭐ |

### 📈 04-时序预测项目

时间序列分析与预测。

| 项目 | 描述 | 技术栈 | 难度 |
|:-----|:-----|:-------|:----:|
| **温度预测 LSTM** | 多变量时序预测 | LSTM, Keras | ⭐⭐⭐ |
| **股票价格预测** | 金融时序分析 | LSTM, Attention | ⭐⭐⭐⭐ |
| **能源消耗预测** | 多步预测问题 | Transformer, Prophet | ⭐⭐⭐⭐ |

### 🏆 05-Kaggle 竞赛项目

真实竞赛的完整解决方案。

#### 🥇 金牌方案

| 竞赛 | 排名 | 奖金池 | 核心技术 |
|:-----|:----:|:------:|:---------|
| **Feedback Prize - ELL** | Top 1% | $160,000 | DeBERTa, Multi-task Learning, Pseudo Labeling |
| **RSNA Abdominal Trauma** | Top 1% | $140,000 | EfficientNet, 3D CNN, Multi-label Classification |

#### 🥈 银牌方案

| 竞赛 | 排名 | 核心技术 |
|:-----|:----:|:---------|
| **American Express Default** | Top 5% | GBDT Ensemble, Feature Engineering, Time Series |

#### 🥉 铜牌方案

| 竞赛 | 排名 | 核心技术 |
|:-----|:----:|:---------|
| **RSNA Lumbar Spine** | Top 10% | 3D Medical Imaging, Segmentation, Classification |

---

## 项目详解

### Feedback Prize - ELL (🥇 金牌)

**竞赛背景**：评估英语学习者的写作水平，预测 6 个维度的分数。

**解决方案亮点**：

```
数据处理 ──► 模型架构 ──► 训练策略 ──► 后处理
    │            │            │           │
    ▼            ▼            ▼           ▼
 文本清洗    DeBERTa-v3   多任务学习    模型融合
 数据增强    Large       伪标签       阈值优化
 分层采样    + Pooling   对抗训练     TTA
```

**核心代码结构**：

```python
class FeedbackModel(nn.Module):
    def __init__(self, model_name, num_labels=6):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooler = MeanPooling()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask)
        pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        return self.classifier(pooled)
```

**关键技术**：
- **DeBERTa-v3-Large**: 最强预训练语言模型
- **Multi-task Learning**: 同时预测 6 个目标
- **Pseudo Labeling**: 利用无标签数据
- **Adversarial Training**: AWP 对抗训练提升鲁棒性

---

### RSNA Abdominal Trauma (🥇 金牌)

**竞赛背景**：从 CT 扫描图像中检测腹部创伤。

**解决方案亮点**：

| 阶段 | 方法 | 说明 |
|:-----|:-----|:-----|
| **预处理** | 3D Volume Processing | 处理 DICOM 格式医学影像 |
| **模型** | EfficientNet-B5 + 3D CNN | 2D 切片 + 3D 体积特征 |
| **训练** | Multi-label Classification | 多器官损伤检测 |
| **推理** | TTA + Ensemble | 测试时增强 + 模型融合 |

**数据处理流程**：

```python
def process_dicom(dicom_path):
    # 读取 DICOM 序列
    slices = load_dicom_series(dicom_path)

    # 窗宽窗位调整
    volume = apply_windowing(slices, window_center=40, window_width=400)

    # 重采样到统一尺寸
    volume = resample_volume(volume, target_spacing=[1.5, 1.5, 3.0])

    return volume
```

---

## 项目运行指南

### 环境准备

```bash
# 进入项目目录
cd 09-practical-projects

# 安装项目依赖
pip install -r requirements.txt

# 下载预训练模型（如需要）
python scripts/download_models.py
```

### 运行示例

::: code-group

```bash [Titanic]
cd 01-ml-basics/titanic-survival-xgboost
jupyter lab titanic_analysis.ipynb
```

```bash [Feedback Prize]
cd 05-kaggle-competitions/01-Feedback-ELL
python train.py --config configs/deberta_large.yaml
```

```bash [RSNA]
cd 05-kaggle-competitions/02-RSNA-Abdominal
python train.py --fold 0 --model efficientnet_b5
```

:::

---

## 学习建议

### 入门者路线

```
Titanic ──► MNIST ──► 情感分析 ──► 温度预测
   │          │          │           │
   ▼          ▼          ▼           ▼
 特征工程    CNN基础    RNN/LSTM    时序建模
```

### 进阶者路线

```
Otto 多分类 ──► Transformer 文本分类 ──► Kaggle 银牌方案
      │                │                      │
      ▼                ▼                      ▼
   集成学习      注意力机制           工业级方案
```

### 竞赛者路线

```
Kaggle 铜牌 ──► Kaggle 银牌 ──► Kaggle 金牌
      │              │              │
      ▼              ▼              ▼
   基础方案      优化技巧      顶级方案
```

---

## 参考资源

### Kaggle 相关
- [Kaggle 官方文档](https://www.kaggle.com/docs)
- [Kaggle 竞赛技巧汇总](https://www.kaggle.com/competitions)

### 论文
- Vaswani et al. (2017). Attention Is All You Need
- He et al. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention

### 工具
- [Weights & Biases](https://wandb.ai/) - 实验追踪
- [Optuna](https://optuna.org/) - 超参数优化
