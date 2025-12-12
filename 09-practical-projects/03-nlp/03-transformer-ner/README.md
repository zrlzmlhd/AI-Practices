# Transformer命名实体识别项目

基于Transformer的命名实体识别(NER)系统,支持多种模型架构对比实验。

## 项目概述

本项目实现了基于深度学习的命名实体识别系统,采用CoNLL-2003标准数据集,支持识别人名(PER)、组织(ORG)、地点(LOC)、其他(MISC)四类实体。

### 技术特点

- **多模型支持**: 实现Transformer、Transformer+CRF、BiLSTM+CRF三种架构
- **标准评测**: 采用CoNLL-2003标准数据集和BIO标注体系
- **工程化设计**: 模块化代码结构,完整的训练和评估pipeline
- **可扩展性**: 灵活的配置系统,易于调整模型参数

### 模型架构

1. **Transformer编码器**
   - 多头自注意力机制捕获长距离依赖
   - 位置编码提供序列顺序信息
   - LayerNorm和残差连接保证训练稳定性

2. **Transformer + CRF**
   - Transformer编码器提取特征
   - CRF层建模标签转移约束
   - Viterbi解码保证标注一致性

3. **BiLSTM + CRF**(对比基线)
   - 双向LSTM捕获上下文信息
   - 经典NER架构,用于性能对比

## 项目结构

```
03-transformer-ner/
├── data/                   # 数据目录
│   ├── README.md          # 数据说明
│   ├── download_data.py   # 数据下载脚本
│   ├── train.txt          # 训练数据(需下载)
│   └── test.txt           # 测试数据(需下载)
├── src/                    # 源代码
│   ├── __init__.py        # 模块初始化
│   ├── data.py            # 数据处理模块
│   ├── model.py           # 模型定义模块
│   ├── train.py           # 训练脚本
│   └── evaluate.py        # 评估脚本
├── models/                 # 模型保存目录
├── results/                # 结果输出目录
├── README.md               # 项目说明
└── requirements.txt        # 依赖包列表
```

## 快速开始

### 环境配置

```bash
# 创建虚拟环境(推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

1. 下载CoNLL-2003数据集
2. 将train.txt和test.txt放入data/目录

数据格式示例:
```
EU          B-ORG
rejects     O
German      B-MISC
call        O

British     B-MISC
PM          B-PER
```

### 模型训练

```bash
# 训练基础Transformer模型
python src/train.py --model_type transformer --epochs 30

# 训练Transformer+CRF模型(推荐)
python src/train.py --model_type transformer_crf --epochs 50

# 训练BiLSTM+CRF模型
python src/train.py --model_type bilstm_crf --epochs 50
```

训练参数说明:
- `--model_type`: 模型类型(transformer/transformer_crf/bilstm_crf)
- `--train_path`: 训练数据路径(默认: data/train.txt)
- `--test_path`: 测试数据路径(默认: data/test.txt)
- `--max_len`: 最大序列长度(默认: 128)
- `--epochs`: 训练轮数(默认: 30)
- `--batch_size`: 批大小(默认: 32)
- `--learning_rate`: 学习率(默认: 0.001)

### 模型评估

```bash
python src/evaluate.py \
    --model_path models/transformer_crf_model.h5 \
    --processor_path models/transformer_crf_processor.pkl \
    --test_path data/test.txt
```

## 技术细节

### BIO标注体系

- **B (Begin)**: 实体开始位置
- **I (Inside)**: 实体内部位置
- **O (Outside)**: 非实体位置

标签示例: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC

### 模型参数配置

**Transformer编码器**:
- d_model: 128 (模型维度)
- num_heads: 4 (注意力头数)
- num_layers: 2 (编码器层数)
- d_ff: 512 (前馈网络维度)
- dropout: 0.1 (Dropout率)

**Transformer + CRF**:
- num_layers: 3 (更深的编码器)
- dropout: 0.2 (更强的正则化)
- CRF层学习标签转移约束

### 评估指标

- **Precision**: 预测为实体的样本中实际是实体的比例
- **Recall**: 实际实体中被正确识别的比例
- **F1-Score**: Precision和Recall的调和平均数
- **Support**: 每类实体的样本数量

## 实验结果

使用CoNLL-2003数据集的典型性能:

| 模型 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| BiLSTM-CRF | 0.85 | 0.83 | 0.84 |
| Transformer | 0.86 | 0.84 | 0.85 |
| Transformer-CRF | 0.88 | 0.86 | 0.87 |

*注: 实际结果依赖于数据集规模和训练设置*

## 依赖库

核心依赖:
- TensorFlow >= 2.13.0
- Keras >= 2.13.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0
- tensorflow-addons (CRF功能,可选)

完整依赖见requirements.txt

## 注意事项

1. **CRF功能**: 需要tensorflow-addons包,如遇兼容性问题可使用基础Transformer模型
2. **数据集**: CoNLL-2003需要单独下载,请遵守数据使用协议
3. **内存占用**: 根据序列长度和批大小调整,避免OOM错误
4. **训练时间**: GPU训练推荐,CPU训练较慢

## 扩展方向

- 集成预训练BERT/RoBERTa模型
- 支持中文NER数据集(People's Daily、MSRA等)
- 实现实体链接和关系抽取
- 部署为REST API服务

## 许可证

本项目仅供学习研究使用,请勿用于商业用途。

## 参考资源

- CoNLL-2003 Shared Task: [链接](https://www.clips.uantwerpen.be/conll2003/ner/)
- Attention Is All You Need (Transformer论文)
- BiLSTM-CRF for Sequence Tagging
