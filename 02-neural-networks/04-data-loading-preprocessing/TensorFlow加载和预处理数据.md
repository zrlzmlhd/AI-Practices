# TensorFlow 数据加载与预处理指南

本文档系统讲解 TensorFlow 中数据加载、预处理和数据流水线构建的核心概念与最佳实践。

## 核心目标

TensorFlow 数据工具（`tf.data`、`TFRecord`、Keras 预处理层）的核心目标是**解决数据 I/O 瓶颈，构建可扩展、可移植、高性能的训练流程**。

当模型在 GPU/TPU 上几毫秒完成一次前向传播时，如果 CPU 还在读取、解析、转换下一个批次的数据，GPU 就会处于等待状态。本章的所有知识点都是为了让数据以"流水线"的方式源源不断地、高效地供给模型。

---

## 1. tf.data API

`tf.data` API 是 TensorFlow 中构建数据输入流水线的核心工具。

### 1.1 设计动机

1. **性能**: 原生 Python 迭代器受全局解释器锁（GIL）限制，速度较慢。`tf.data` 底层使用 C++ 实现，支持多线程、并行处理和异步加载。
2. **解耦**: 将数据的"提取（Extract）"、"转换（Transform）"和"加载（Load）"逻辑与模型训练逻辑完全分开。
3. **抽象**: 无论数据源是 Numpy 数组、CSV 文件、TFRecord 文件还是云存储上的图片，`tf.data` 都提供统一的 API 来处理。

---

### 1.2 处理小数据集（内存数据）

**适用场景**：
- 数据集较小（小于机器 RAM 的 1/4）
- 快速原型设计和实验

**使用方法**：

```python
import tensorflow as tf
import numpy as np

# 假设 X_train 是 (10000, 784) 的图像, y_train 是 (10000,) 的标签
X_train = np.random.rand(10000, 784)
y_train = np.random.randint(0, 10, size=(10000,))

# 1. 创建数据集对象
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# 2. 应用转换
dataset = dataset.shuffle(buffer_size=10000) \
                 .batch(batch_size=32) \
                 .prefetch(tf.data.AUTOTUNE)

# 3. 传入 model.fit
# model.fit(dataset, epochs=10)
```

**注意事项**：
- `from_tensor_slices` 会将 Numpy 数组或 TF 张量**复制**到 TensorFlow 的内存中。如果数据非常大会导致内存溢出。
- `shuffle(buffer_size)` 中的 `buffer_size` 应该足够大才能实现良好的随机性，对于内存数据集，可以直接设为 `len(X_train)`。

---

### 1.3 处理大数据集（流式处理）

**适用场景**：
- 数据集 > 机器 RAM
- 数据以大量文件的形式存在

**使用方法（以 CSV 为例）**：

```python
# 假设有多个 CSV 文件: train_01.csv, train_02.csv, ...

# 1. list_files: 找到所有匹配的文件名
file_pattern = "data/train_*.csv"
files_ds = tf.data.Dataset.list_files(file_pattern)

# 2. interleave: 并行读取多个文件
raw_dataset = files_ds.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=5,
    num_parallel_calls=tf.data.AUTOTUNE
)

# 定义解析函数
def parse_csv_line(line):
    defs = [tf.constant(0.0)] * 8 + [tf.constant(0)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    features = tf.stack(fields[:-1])
    label = fields[-1]
    return features, label

# 3. shuffle: 在样本层面进行打乱
dataset = raw_dataset.shuffle(buffer_size=1000)

# 4. map: 并行应用预处理函数
dataset = dataset.map(parse_csv_line, num_parallel_calls=tf.data.AUTOTUNE)

# 5. batch: 组合成批次
dataset = dataset.batch(batch_size=32)

# 6. prefetch: 预取
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

**注意事项**：
- **顺序很重要！** `shuffle` 一定要在 `batch` 之前。`prefetch` 一定要在最后。
- `num_parallel_calls=tf.data.AUTOTUNE`：在 `interleave` 和 `map` 中**一定要加**，这是榨干 CPU 性能的关键。
- `prefetch(tf.data.AUTOTUNE)`：在流水线的**末尾一定要加**。

---

## 2. TFRecord 格式

`TFRecord` 是一种用于存储大量数据的简单、高效的**二进制文件格式**。

### 2.1 设计动机

1. **I/O 性能**: 训练 ImageNet 时有 120 万个 JPEG 小文件。操作系统在处理"打开文件、读取文件、关闭文件"这个操作上会浪费大量时间（磁盘寻道时间）。TFRecord 把这 120 万个小文件合并成几百个大文件。**读取大文件的效率远高于读取小文件**。
2. **序列化**: CSV 是文本，解析慢。JPEG/PNG 需要解码，也慢。TFRecord 内部使用 **Protocol Buffers** 协议，这是一种高效的二进制序列化格式。
3. **标准化**: 将任意类型的数据（图像、文本、标签）打包成一种统一格式，方便存储和读取。

### 2.2 适用场景

- **训练瓶颈在 I/O 时**: 当 GPU 利用率很低，但 CPU 却在忙于读取和解码数据时
- **超大数据集**: 尤其是图像、音频、视频或序列数据
- **需要预处理一次，多次使用**: 可以花时间把所有原始数据转成 TFRecord，之后每次训练都直接读取

---

### 2.3 Protocol Buffers 和 tf.train.Example

`tf.train.Example` 是 TensorFlow 预先定义好的协议缓冲区，本质上是一个字典：`{ "feature_name": tf.train.Feature }`

`tf.train.Feature` 可以是三种类型之一：
1. `BytesList`：存储字符串或原始字节（如编码后的图像）
2. `FloatList`：存储浮点数
3. `Int64List`：存储整数或布尔值

**写入 TFRecord（离线步骤）**：

```python
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 创建 tf.train.Example
example_proto = tf.train.Example(features=tf.train.Features(feature={
    'image_raw': _bytes_feature(image_bytes),
    'label': _int64_feature(label)
}))

# 写入 TFRecord 文件
options = tf.io.TFRecordOptions(compression_type="ZLIB")
with tf.io.TFRecordWriter("data/my_images.tfrecord", options) as writer:
    writer.write(example_proto.SerializeToString())
```

---

### 2.4 加载和解析 Example

**读取 TFRecord（在线步骤）**：

```python
# 1. 定义解码器
feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

# 2. 编写解析函数
def _parse_tfrecord_fn(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.io.decode_jpeg(parsed_features['image_raw'], channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0

    label = parsed_features['label']
    return image, label

# 3. 构建 tf.data 流水线
tfrecord_files = ["data/my_images.tfrecord", "data/my_images_2.tfrecord"]
dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="ZLIB")

dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
```

**注意事项**：
- `feature_description` 必须与写入 `Example` 时的**键名**和**类型**完全匹配
- `FixedLenFeature` 用于固定长度的特征。如果特征是变长的（如一个句子），使用 `VarLenFeature`

---

### 2.5 SequenceExample（处理序列数据）

`tf.train.Example` 适合"一张图、一个标签"这种数据。如果数据本身是**序列**呢？比如一篇文章（词的序列）、一段音频（采样点的序列）或一个视频（帧的序列），并且序列长度**不固定**。

**适用场景**：NLP（自然语言处理）、时间序列、视频分析

`SequenceExample` 有两部分：
1. `context`：`tf.train.Features`。存储"全局"信息，如 `video_id`，`user_id`
2. `feature_lists`：`tf.train.FeatureLists`。存储"序列"信息，如视频的每一帧、文章的每一个词

在 `map` 函数中，使用 `tf.io.parse_single_sequence_example` 来代替 `parse_single_example`。

---

## 3. 预处理输入特征

这一部分讨论**如何将原始特征转换成模型能理解的数字**。

### 3.1 Keras 预处理层的设计动机

1. **一致性**: 最大的坑点在于**训练**和**推理**时预处理不一致。比如，训练时用均值 127.5 归一化，推理时忘了，模型就会彻底失效。
2. **可移植性**: Keras 预处理层**是模型的一部分**。当 `model.save()` 时，这些预处理逻辑会**一起被保存**。
3. **性能**: 如果把预处理层放在模型里，它们可以在 GPU/TPU 上运行，而不是在 CPU 上，处理速度更快。

---

### 3.2 独热向量 (One-Hot) 编码

**适用场景**：当分类特征的**类别数量很少**时（比如 < 50）

**示例**：
- 特征："天气"
- 类别："晴天", "阴天", "雨天"
- "晴天" -> `[1, 0, 0]`
- "阴天" -> `[0, 1, 0]`
- "雨天" -> `[0, 0, 1]`

**不适用场景**：当类别数量非常多时（称为"高基数"特征），比如 "用户 ID"（几百万个）。如果用 One-Hot，会得到一个几百万维的、极其稀疏的向量。

**使用方法 (Keras)**：

```python
vocab = ["晴天", "阴天", "雨天"]
lookup_layer = tf.keras.layers.StringLookup(vocabulary=vocab)

one_hot_layer = tf.keras.layers.CategoryEncoding(
    num_tokens=lookup_layer.vocabulary_size(),
    output_mode="one_hot"
)

# 在模型中构建
input_tensor = tf.keras.Input(shape=(1,), dtype=tf.string)
indices = lookup_layer(input_tensor)
one_hot_output = one_hot_layer(indices)
```

---

### 3.3 嵌入 (Embedding) 编码

**适用场景**：处理**高基数**分类特征的**标准且高效**方法

**设计动机**：
- One-Hot 向量是稀疏的、高维的、且无意义的
- **Embedding** 将一个高维索引映射到一个**低维、稠密、且有意义的**浮点数向量

**核心思想**：`Embedding` 层是一个**可训练的**查找表。在训练过程中，模型会**学习**到这些向量。具有相似行为的"用户"（或相似含义的"词语"）最终会得到相似的嵌入向量。

**使用方法**：

```python
VOCAB_SIZE = 10000
EMBEDDING_DIM = 16

user_id_input = tf.keras.Input(shape=(1,), dtype="int64")

embedding_layer = tf.keras.layers.Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBEDDING_DIM
)

embedded_output = embedding_layer(user_id_input)
flattened_output = tf.keras.layers.Flatten()(embedded_output)
```

**注意事项**：
- `input_dim` 是词汇表大小，**不是**批次大小或样本总数
- `output_dim` 是超参数。常见选择是 8, 16, 32, 64
- `Embedding` 层的输入**必须是整数索引**

---

## 4. 性能优化最佳实践

### 4.1 数据流水线优化清单

| 优化技术 | 作用 | 使用位置 |
|---------|------|----------|
| `shuffle()` | 随机打乱数据 | `batch()` 之前 |
| `map(num_parallel_calls=AUTOTUNE)` | 并行预处理 | 数据转换时 |
| `batch()` | 组合成批次 | `shuffle()` 之后 |
| `prefetch(AUTOTUNE)` | CPU/GPU 并行 | 流水线末尾 |
| `interleave()` | 并行读取多文件 | 读取分片数据时 |
| `cache()` | 缓存数据到内存/磁盘 | 小数据集或重复使用 |

### 4.2 推荐的操作顺序

```
list_files -> shuffle(文件级) -> interleave -> shuffle(样本级) -> map -> batch -> prefetch
```

### 4.3 TFRecord 文件分片策略

- 每个 TFRecord 文件建议 100-200 MB
- 使用多个分片文件便于并行读取
- 文件命名使用 `data_000.tfrecord`, `data_001.tfrecord` 格式

---

## 总结

本章涵盖了 TensorFlow 数据处理的核心知识点：

1. **tf.data API**: 构建高效数据流水线的统一接口
2. **TFRecord 格式**: 大规模数据集的标准存储格式
3. **Keras 预处理层**: 确保训练和推理预处理一致性的最佳方案
4. **性能优化**: 通过并行化和预取最大化 GPU 利用率

掌握这些工具是构建高效深度学习训练流程的必经之路。
