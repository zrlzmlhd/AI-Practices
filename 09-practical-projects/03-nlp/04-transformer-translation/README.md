# Transformer机器翻译

基于Transformer架构（"Attention is All You Need", Vaswani et al., 2017）的完整机器翻译实现。

## 项目特点

- 完整的Transformer Encoder-Decoder架构
- 支持自定义词汇表大小和模型规模
- 包含位置编码、多头注意力、残差连接等核心组件
- 提供训练、评估、推理完整流程
- BLEU评估指标和翻译样本展示

## 环境配置

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据

下载双语平行语料，推荐数据集：

- **WMT**（英中、英德等）: http://www.statmt.org/wmt/
- **IWSLT**（TED演讲）: https://wit3.fbk.eu/
- **Tatoeba**（多语言句对）: https://tatoeba.org/

数据格式要求：
```
data/
├── train.en   # 源语言训练集
├── train.zh   # 目标语言训练集
├── test.en    # 源语言测试集
└── test.zh    # 目标语言测试集
```

### 2. 训练模型

```bash
# 使用全部数据训练
python src/train.py --src_path data/train.en --tgt_path data/train.zh

# 快速实验（使用10000个样本）
python src/train.py --max_samples 10000 --epochs 30 --batch_size 64
```

### 3. 评估模型

```bash
python src/evaluate.py \
    --model_path models/transformer_translation_model.h5 \
    --processor_path models/translation_processor.pkl \
    --test_src data/test.en \
    --test_tgt data/test.zh
```

## 项目结构

```
04-transformer-translation/
├── data/                   # 数据目录
│   ├── README.md
│   └── download_data.py   # 数据下载工具
├── src/                    # 源代码
│   ├── __init__.py
│   ├── data.py            # 数据处理
│   ├── model.py           # Transformer模型
│   ├── train.py           # 训练脚本
│   └── evaluate.py        # 评估脚本
├── models/                 # 保存的模型
├── results/                # 训练结果
├── README.md
└── requirements.txt
```

## 模型架构

### Transformer组件

1. **Positional Encoding**：正弦位置编码
2. **Multi-Head Attention**：多头自注意力机制
3. **Feed-Forward Network**：位置感知前馈网络
4. **Layer Normalization**：层归一化
5. **Residual Connection**：残差连接

### 超参数配置

```python
num_layers = 4          # 编码器/解码器层数
d_model = 256           # 模型维度
num_heads = 8           # 注意力头数
d_ff = 1024            # 前馈网络维度
dropout_rate = 0.1      # Dropout比率
```

## 训练技巧

### 学习率调度

采用Transformer论文中的学习率调度策略：
- Warmup阶段：前4000步线性增长
- 衰减阶段：按平方根倒数衰减

### 正则化

- Dropout正则化（0.1）
- Label Smoothing（可选）
- 早停策略

### 数据增强

- 句子打乱（Sentence Shuffling）
- 回译（Back Translation，可选）

## 评估指标

- **BLEU**：标准机器翻译评估指标
- **词级准确率**：忽略填充位置的准确率
- **翻译样本**：定性评估翻译质量

## 优化建议

1. **增大模型规模**：提升`d_model`和`num_layers`
2. **增加训练数据**：使用更大的平行语料
3. **Beam Search**：使用集束搜索代替贪心解码
4. **模型集成**：训练多个模型进行集成
5. **BPE分词**：使用子词单元代替词级分词

## 参考文献

- Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
- Sutskever et al. (2014). Sequence to Sequence Learning with Neural Networks. NIPS.

## 许可证

MIT License
