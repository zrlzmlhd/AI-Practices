# 数据说明

## IMDB电影评论数据集

### 数据来源

- **官方网站**: https://ai.stanford.edu/~amaas/data/sentiment/
- **Keras内置**: `keras.datasets.imdb`
- **论文**: [Learning Word Vectors for Sentiment Analysis](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)

### 数据集描述

IMDB数据集包含50,000条电影评论，用于二分类情感分析任务。

**统计信息**:
- 训练集: 25,000条评论
- 测试集: 25,000条评论
- 类别: 正面(1) / 负面(0)
- 平衡性: 正负样本各占50%

**数据格式**:
```python
# 每条评论是一个整数序列（词的索引）
review = [1, 14, 22, 16, 43, 530, 973, ...]

# 标签是0或1
label = 1  # 正面
```

### 词汇表

- 总词汇量: ~88,000个词
- 常用词汇: 通常使用前10,000个最常见的词
- 特殊索引:
  - 0: 填充符 (padding)
  - 1: 序列开始符 (start)
  - 2: 未知词 (unknown/OOV)
  - 3+: 实际词汇

### 数据下载

#### 方法1: 使用下载脚本（推荐）

```bash
cd data
python download_data.py
```

#### 方法2: 在代码中自动下载

```python
from tensorflow import keras

# Keras会自动下载并缓存
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
```

数据会被缓存到: `~/.keras/datasets/imdb.npz`

#### 方法3: 手动下载

如果自动下载失败，可以手动下载：

1. 访问: https://ai.stanford.edu/~amaas/data/sentiment/
2. 下载: `aclImdb_v1.tar.gz`
3. 解压到当前目录

### 数据预处理

项目中的预处理步骤：

1. **加载数据**: 使用Keras API加载
2. **限制词汇表**: 只使用最常见的10,000个词
3. **序列填充**: 将所有序列填充/截断到200个词
4. **划分数据集**: 训练集、验证集、测试集

```python
from data import load_imdb_data

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_imdb_data(
    max_words=10000,  # 词汇表大小
    max_len=200,      # 序列长度
    test_size=0.2     # 验证集比例
)
```

### 数据示例

**正面评论示例**:
```
"This movie is absolutely fantastic! The plot is engaging,
the acting is superb, and the cinematography is breathtaking.
I highly recommend it to everyone."
```

**负面评论示例**:
```
"Waste of time. The acting was terrible and the story made
no sense. I couldn't wait for it to end. Don't bother watching."
```

### 数据统计

**序列长度分布**:
- 最小长度: 11个词
- 最大长度: 2,494个词
- 平均长度: 238个词
- 中位数长度: 178个词
- 90%分位数: 434个词

**选择max_len=200的原因**:
- 覆盖约70%的评论
- 平衡信息保留和计算效率
- 是常用的经验值

### 引用

如果使用此数据集，请引用：

```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```

### 许可证

数据集遵循原始论文的许可证，仅供学术研究使用。

---

**更新日期**: 2025-11-29
