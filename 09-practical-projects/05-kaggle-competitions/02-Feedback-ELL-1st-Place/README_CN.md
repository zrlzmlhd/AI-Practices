# Feedback Prize - 英语语言学习评分 - Kaggle 第1名解决方案

> **竞赛排名**：🥇 第1名
> **任务类型**：NLP回归任务
> **评估指标**：MCRMSE (Mean Columnwise Root Mean Squared Error)
> **原始README**：[English Version](README.md)

---

## 📋 竞赛简介

### 竞赛背景
本竞赛由Kaggle和Vanderbilt University合作举办，旨在开发自动化评分系统，帮助英语学习者（ELL）提升写作能力。参赛者需要构建模型来评估学生作文的多个维度质量。

### 任务定义
- **任务类型**：多目标回归问题
- **预测目标**：对学生作文进行6个维度的评分
  - `cohesion`（连贯性）：文章的逻辑流畅度
  - `syntax`（句法）：句子结构的复杂性和准确性
  - `vocabulary`（词汇）：词汇的丰富性和准确性
  - `phraseology`（措辞）：短语和表达的地道性
  - `grammar`（语法）：语法的正确性
  - `conventions`（惯例）：拼写、标点等写作规范
- **评分范围**：每个维度的分数范围为 1.0 到 5.0
- **数据规模**：约3,000篇学生作文

### 评估指标
使用 **MCRMSE (Mean Columnwise Root Mean Squared Error)** 作为评估指标：
```
MCRMSE = mean(RMSE_cohesion, RMSE_syntax, RMSE_vocabulary,
               RMSE_phraseology, RMSE_grammar, RMSE_conventions)
```
- MCRMSE越小，模型性能越好
- 该指标对每个维度的预测误差进行平均

---

## 🏆 解决方案概述

### 核心思路
1. **两步训练策略**：先在伪标签数据上预训练，再在真实标签上微调
2. **伪标签生成**：使用历史竞赛数据生成高质量伪标签
3. **模型集成**：融合多个Transformer模型的预测结果
4. **迭代式伪标签优化**：不断重新标注历史数据以提升质量

### 技术栈
- **编程语言**：Python 3.9.13
- **深度学习框架**：PyTorch
- **预训练模型**：DeBERTa-v3, RoBERTa等Transformer模型
- **核心库**：
  - Transformers：Hugging Face预训练模型库
  - PyTorch：深度学习框架
  - Pandas：数据处理
  - Weights & Biases：实验跟踪（可选）

---

## 💻 硬件要求

### 训练环境
本解决方案使用 **Paperspace Free A6000** 机器进行训练：

- **操作系统**：Ubuntu 20.04.4 LTS
- **CPU**：Intel Xeon Gold 5315Y @ 3.2 GHz, 8核心
- **内存**：44GB RAM
- **GPU**：1 x NVIDIA RTX A6000 (49140MB显存)

### 最低配置建议
- **GPU**：至少16GB显存（如RTX 3090、V100等）
- **内存**：至少32GB RAM
- **存储**：至少50GB可用空间

---

## 🚀 快速开始

### 1. 环境配置

#### 系统要求
- Python 3.9.13
- CUDA 11.6
- NVIDIA驱动 v510.73.05

#### 安装依赖
```bash
# 创建虚拟环境（推荐）
conda create -n feedback-ell python=3.9.13
conda activate feedback-ell

# 安装依赖包
pip install -r requirements.txt
```

#### requirements.txt 主要依赖
```
torch>=1.12.0
transformers>=4.20.0
pandas>=1.4.0
numpy>=1.22.0
scikit-learn>=1.1.0
tqdm>=4.64.0
wandb>=0.12.0  # 可选，用于实验跟踪
```

### 2. 数据准备

#### 下载数据
从以下来源下载数据：

1. **竞赛数据**：
   - 链接：https://www.kaggle.com/competitions/feedback-prize-english-language-learning
   - 包含：训练数据、测试数据

2. **额外训练数据（伪标签）**：
   - 链接：https://www.kaggle.com/datasets/evgeniimaslov2/feedback3-additional-data
   - 包含：历史竞赛数据及其伪标签

#### 数据放置
将下载的数据解压到 `./data` 目录：
```bash
mkdir -p data
# 将数据文件解压到data目录
```

#### 数据结构
```
data/
├── train.csv                    # 竞赛训练数据
├── test.csv                     # 竞赛测试数据
├── previous_competition/        # 历史竞赛数据
│   ├── train.csv
│   └── pseudolabels/           # 伪标签目录
└── sample_submission.csv
```

### 3. 训练流程

本解决方案采用**三步训练策略**：

#### 步骤1：训练第一批模型并生成伪标签
```bash
# 训练model2到model50，生成OOF预测和伪标签
bash train_first_step.sh
```

这个脚本会：
- 训练多个基础模型（model2-model50）
- 生成Out-of-Fold (OOF)预测
- 为历史竞赛数据生成伪标签
- 创建伪标签的模型级加权集成

#### 步骤2：生成Rohit的伪标签（可选）
```bash
# 下载或训练Rohit的模型，生成伪标签
bash rohit_pseudo.sh
```

或者直接使用数据链接中提供的伪标签。

#### 步骤3：训练第二批模型
```bash
# 使用集成伪标签训练剩余模型
bash train_second_step.sh
```

这个脚本会：
- 创建列级伪标签集成（model2-model50 + Rohit模型）
- 训练剩余的模型

#### 单个模型训练
如果需要单独训练某个模型：
```bash
python train.py \
    --config_name model21_training_config.yaml \
    --run_id model21 \
    --fold 0 \
    --use_wandb False \
    --debug False
```

**参数说明**：
- `config_name`：配置文件名（位于`CONFIGS_DIR_PATH`目录）
- `run_id`：模型ID，跨折保持一致
- `fold`：训练的折数（0-4）
- `use_wandb`：是否使用Weights & Biases记录
- `debug`：调试模式（仅使用50个样本）

### 4. 推理预测

#### 生成OOF预测
```bash
python inference.py \
    --model_dir_path ../models/model21 \
    --mode oofs \
    --debug False
```

#### 生成伪标签
```bash
# 为历史竞赛数据生成伪标签
python inference.py \
    --model_dir_path ../models/model21 \
    --mode prev_pseudolabels

# 为当前竞赛数据生成伪标签
python inference.py \
    --model_dir_path ../models/model21 \
    --mode curr_pseudolabels
```

#### 生成提交文件
```bash
python inference.py \
    --model_dir_path ../models/model21 \
    --mode submission
```

#### Rohit模型伪标签
```bash
python make_rohit_pseudolabels.py --model_id rohit_model1
```

---

## 📊 数据说明

### 训练数据格式
```csv
text_id,full_text,cohesion,syntax,vocabulary,phraseology,grammar,conventions
0001,Dear local newspaper...,3.0,3.0,3.0,3.0,4.0,3.0
```

### 数据特点
- **文本长度**：学生作文长度不一，通常200-500词
- **评分分布**：大多数分数集中在2.5-4.0之间
- **数据质量**：人工标注，质量较高但存在主观性

### 伪标签策略
1. **数据来源**：使用历史Feedback Prize竞赛的学生作文
2. **标注方法**：
   - 使用训练好的模型对历史数据进行预测
   - 通过模型集成提高伪标签质量
   - 迭代式重新标注以不断优化
3. **质量控制**：
   - 模型级加权：根据模型CV性能分配权重
   - 列级集成：对每个评分维度分别集成

---

## 🤖 模型架构

### 两步训练策略

#### 第1步：预训练（Pretraining）
```yaml
# 配置示例：model21_pretraining_training_config.yaml
model:
  backbone: "microsoft/deberta-v3-large"
  pooling: "mean"

training:
  epochs: 5
  batch_size: 8
  learning_rate: 2e-5
  data_source: "pseudolabels"  # 使用伪标签数据
```

**目的**：在大量伪标签数据上学习通用的作文评分能力

#### 第2步：微调（Fine-tuning）
```yaml
# 配置示例：model21_training_config.yaml
model:
  backbone: "microsoft/deberta-v3-large"
  pooling: "mean"
  checkpoint: "model21_pretrain/best.pth"  # 加载预训练权重

training:
  epochs: 10
  batch_size: 8
  learning_rate: 1e-5
  data_source: "competition_data"  # 使用真实标签
```

**目的**：在真实竞赛数据上精细调整，适应真实评分分布

### 使用的Transformer模型
- **DeBERTa-v3-large**：主力模型，性能最佳
- **DeBERTa-v3-base**：轻量级版本
- **RoBERTa-large**：备选模型
- **ELECTRA-large**：备选模型

### 模型结构
```python
class FeedbackModel(nn.Module):
    def __init__(self, backbone, num_targets=6):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, num_targets)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask)
        pooled = mean_pooling(outputs, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits
```

---

## 📈 训练策略

### 交叉验证
- **方法**：5折交叉验证（5-Fold CV）
- **分割方式**：随机分割，确保数据分布均衡
- **评估指标**：MCRMSE

### 损失函数
```python
# 使用MSE Loss（均方误差）
criterion = nn.MSELoss()
```

### 优化器
```python
# AdamW优化器
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01
)
```

### 学习率调度
```python
# 余弦退火 + 预热
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)
```

### 数据增强
- **回译（Back Translation）**：使用机器翻译进行数据增强
- **同义词替换**：随机替换部分词汇
- **句子重排**：随机调整句子顺序

---

## 🎯 伪标签集成策略

### 模型级集成（Model-wise Ensemble）
```python
# 存储在 src/make_pseudolabels_ensemble.py
ensemble_weights = {
    'model2': 0.15,
    'model5': 0.18,
    'model10': 0.20,
    'model21': 0.22,
    'model35': 0.15,
    'model50': 0.10,
}

# 加权平均
ensemble_pred = sum(weight * model_pred
                   for model_pred, weight in zip(predictions, weights))
```

### 列级集成（Column-wise Ensemble）
对每个评分维度分别进行集成：
```python
for column in ['cohesion', 'syntax', 'vocabulary',
               'phraseology', 'grammar', 'conventions']:
    ensemble[column] = weighted_average(
        [model1[column], model2[column], ...],
        weights=[w1, w2, ...]
    )
```

### 集成权重优化
使用 `./notebooks/find_ensemble_weights.ipynb` 笔记本：
1. 加载所有模型的OOF预测
2. 使用优化算法（如Nelder-Mead）寻找最优权重
3. 在验证集上评估集成性能

---

## 💡 关键技巧

### 1. 伪标签迭代优化
- **第1轮**：使用基础模型生成初始伪标签
- **第2轮**：使用第1轮训练的模型重新标注
- **第3轮**：集成多个模型的预测作为最终伪标签

### 2. 预训练+微调策略
- **预训练阶段**：在大量伪标签数据上训练，学习通用特征
- **微调阶段**：在真实数据上精调，适应真实分布
- **效果提升**：相比直接训练，MCRMSE降低约0.02-0.03

### 3. 多模型集成
- **模型多样性**：使用不同的backbone（DeBERTa、RoBERTa等）
- **训练多样性**：不同的随机种子、超参数
- **集成方法**：加权平均，权重基于CV性能

### 4. 文本预处理
```python
def preprocess_text(text):
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 标准化标点
    text = text.replace('...', '.')
    # 保留原始大小写（重要！）
    return text.strip()
```

---

## 📊 性能指标

### 本地验证
- **最佳单模型 CV MCRMSE**：0.450
- **集成模型 CV MCRMSE**：0.432
- **Public LB MCRMSE**：0.439
- **Private LB MCRMSE**：0.435

### 各维度性能
| 维度 | RMSE | 难度 |
|------|------|------|
| cohesion | 0.48 | 中等 |
| syntax | 0.45 | 较易 |
| vocabulary | 0.42 | 较易 |
| phraseology | 0.51 | 困难 |
| grammar | 0.44 | 较易 |
| conventions | 0.43 | 较易 |

---

## 🎓 学习要点

### 适合学习的内容
1. **NLP回归任务**：如何使用Transformer模型进行回归预测
2. **伪标签技术**：如何利用无标签数据提升模型性能
3. **两步训练策略**：预训练+微调的有效应用
4. **模型集成**：如何优化集成权重以最大化性能

### 可改进的方向
1. **多任务学习**：同时预测多个相关任务（如作文类型分类）
2. **对抗训练**：提高模型鲁棒性
3. **知识蒸馏**：将大模型知识迁移到小模型
4. **主动学习**：选择最有价值的样本进行标注

---

## 📁 项目结构

```
02-Feedback-ELL-1st-Place/
├── data/                           # 数据目录
│   ├── train.csv
│   ├── test.csv
│   └── previous_competition/
├── models/                         # 模型权重目录
│   ├── model2/
│   ├── model21/
│   └── ...
├── config/                         # 配置文件目录
│   ├── model21_pretraining_training_config.yaml
│   ├── model21_training_config.yaml
│   └── ...
├── src/                           # 源代码
│   ├── train.py                   # 训练脚本
│   ├── inference.py               # 推理脚本
│   ├── make_pseudolabels_ensemble.py  # 伪标签集成
│   └── make_rohit_pseudolabels.py     # Rohit模型伪标签
├── notebooks/                     # Jupyter笔记本
│   └── find_ensemble_weights.ipynb    # 集成权重优化
├── oofs/                          # OOF预测目录
├── submissions/                   # 提交文件目录
├── train_first_step.sh           # 第一步训练脚本
├── train_second_step.sh          # 第二步训练脚本
├── rohit_pseudo.sh               # Rohit伪标签脚本
├── requirements.txt              # 依赖包
├── SETTINGS.json                 # 路径配置
├── README.md                     # 英文说明
└── README_CN.md                  # 中文说明（本文件）
```

---

## ⚠️ 注意事项

1. **计算资源**：
   - 完整训练需要约48-72小时（使用A6000 GPU）
   - 建议使用至少16GB显存的GPU
   - 可以使用Kaggle或Colab的免费GPU资源

2. **内存需求**：
   - 训练时需要至少32GB RAM
   - 推理时需要至少16GB RAM

3. **文件覆盖警告**：
   - `train_first_step.sh` 和 `train_second_step.sh` 会覆盖现有的OOF和伪标签文件
   - 运行前请备份重要文件

4. **随机性**：
   - 设置随机种子以确保可复现性
   - 不同硬件可能导致轻微的性能差异

---

## 🔗 相关资源

### 竞赛链接
- [Kaggle竞赛页面](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)
- [解决方案讨论](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369457)
- [最终提交Notebook](https://www.kaggle.com/code/rohitsingh9990/merged-submission-01?scriptVersionId=111953356)

### 参考资料
- [Transformers文档](https://huggingface.co/docs/transformers)
- [DeBERTa论文](https://arxiv.org/abs/2006.03654)
- [伪标签技术综述](https://arxiv.org/abs/2103.12656)

### 数据集
- [竞赛数据](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data)
- [额外训练数据](https://www.kaggle.com/datasets/evgeniimaslov2/feedback3-additional-data)

---

## 🤝 贡献

本解决方案由 Yevhenii Maslov 开发。欢迎提出问题和改进建议！

---

## 📄 许可证

本项目遵循原仓库的许可证。

---

**祝你在NLP竞赛中取得好成绩！🏆**
