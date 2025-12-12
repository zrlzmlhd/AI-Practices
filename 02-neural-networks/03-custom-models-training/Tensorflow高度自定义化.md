# TensorFlow高度自定义化指南

本章是TensorFlow进阶学习的关键节点，涵盖了从基础张量操作到高级自定义组件的完整知识体系。掌握这些内容是从"调包侠"进阶到"算法工程师"的必经之路。

---

## 1. 何时需要自定义？

Keras的高级API（如`model.fit`）覆盖了大部分常见场景，但以下情况需要使用自定义功能：

### 1.1 适用场景

1. **研究新算法时**：论文中的损失函数（如Focal Loss、Triplet Loss）在Keras中没有内置实现
2. **模型结构特殊时**：如GAN需要两个模型和两个优化器交替训练
3. **需要精细控制梯度时**：如梯度裁剪、梯度噪声注入等

### 1.2 注意事项

遵循"不要过早优化"原则。如果Sequential或Functional API能解决问题，优先使用标准方法。

### 1.3 核心语法：tf.GradientTape

自定义训练的核心是`tf.GradientTape`，它记录所有涉及`tf.Variable`的操作以便自动计算梯度：

```python
# 假设已有 model, optimizer, loss_fn 和数据 (x_batch, y_batch)

# 1. 开启梯度记录
with tf.GradientTape() as tape:
    # 2. 正向传播
    y_pred = model(x_batch, training=True)
    # 3. 计算损失
    loss = loss_fn(y_batch, y_pred)

# 4. 计算梯度
gradients = tape.gradient(loss, model.trainable_variables)

# 5. 应用梯度更新权重
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

---

## 2. 像NumPy一样使用TensorFlow

### 2.1 设计理念

TensorFlow的核心数据结构`tf.Tensor`在操作上与NumPy的`ndarray`高度一致（如`+`、`-`、`*`、`@`、索引等），这降低了学习门槛。

### 2.2 使用场景

在自定义层、损失函数、训练循环中进行数学计算时使用。**注意：不能在模型中间混用NumPy**，因为NumPy无法在GPU上运行，也无法被`tf.GradientTape`追踪。

### 2.3 关键注意事项

1. **严格的类型检查**：`tf.constant(1.0) + tf.constant(1)` 会**报错**。TensorFlow不会自动转换类型，需要手动使用`tf.cast()`
2. **性能陷阱**：频繁在NumPy和TensorFlow之间切换会严重影响性能
3. **精度差异**：TensorFlow默认`float32`，NumPy默认`float64`

### 2.4 类型转换示例

```python
t1 = tf.constant(1.0, dtype=tf.float32)
t2 = tf.constant(1, dtype=tf.int32)

# 错误: TypeError: Incompatible types
# result = t1 + t2

# 正确: 显式转换
result = t1 + tf.cast(t2, dtype=tf.float32)
print(result)  # 输出: tf.Tensor(2.0, shape=(), dtype=float32)
```

---

## 3. TensorFlow核心数据结构

### 3.1 tf.constant（常量）

- **用途**：存放不会改变的数据，如超参数、固定配置
- **特点**：不可变（Immutable）

### 3.2 tf.Variable（变量）

- **用途**：存放需要被训练和改变的数据。**所有模型权重和偏置都是tf.Variable**
- **核心特性**：`tf.GradientTape`默认只监视`tf.Variable`
- **关键参数**：`initial_value`（初始值）、`trainable=True`（是否参与梯度更新）

```python
v = tf.Variable(initial_value=[[1., 2.], [3., 4.]], trainable=True)
```

### 3.3 tf.SparseTensor（稀疏张量）

- **用途**：处理大部分元素为0的数据（如One-hot编码、用户-物品矩阵）
- **优势**：只存储非零值的位置和数值，节省内存

### 3.4 tf.RaggedTensor（不规则张量）

- **用途**：处理变长数据
- **场景**：NLP中不同长度的句子、不同长度的时间序列

### 3.5 tf.data.Dataset（数据集）

- **用途**：官方推荐的数据加载和预处理方式
- **优势**：构建高效的数据流水线，支持异步预取和并行处理

```python
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.map(preprocess_function)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

model.fit(dataset, epochs=10)
```

---

## 4. tf.function与计算图

### 4.1 设计目的

1. **提升性能**：Python是解释型语言，执行速度较慢。`@tf.function`将Python函数转换为静态计算图
2. **图优化**：TensorFlow可以对计算图进行合并操作、剪枝等优化
3. **摆脱瓶颈**：计算图在C++后端高效执行，绕过Python性能限制

### 4.2 适用场景

所有性能敏感的计算都应使用`@tf.function`装饰：
- 自定义训练步骤（`train_step`）
- 自定义模型的`call`方法
- 自定义损失函数
- `tf.data.Dataset.map()`中的预处理函数

> 注意：Keras的`model.fit`和内置层已自动应用此优化

### 4.3 注意事项

1. **只在首次调用时建图**：`@tf.function`只在第一次调用时（或输入shape/dtype改变时）运行Python代码来建图
2. **Python副作用失效**：`print()`只在建图时打印一次，应使用`tf.print()`
3. **变量创建限制**：变量应在函数外部创建（如模型的`__init__`中）
4. **控制流处理**：依赖Tensor的条件语句需使用`tf.cond()`和`tf.while_loop()`

### 4.4 示例对比

```python
# Python模式（Eager Execution）
def python_add(a, b):
    print("--- 正在用Python执行 ---")  # 每次都打印
    return a + b

# Graph模式
@tf.function
def graph_add(a, b):
    print("--- 正在建图(Tracing) ---")  # 只打印一次
    return a + b

a = tf.constant(1)
b = tf.constant(2)

print(graph_add(a, b))  # 打印建图信息和结果
print(graph_add(a, b))  # 只打印结果，直接重用已编译的图
```

---

## 5. 自定义组件速查表

| 组件类型 | 继承基类 | 必须实现的方法 |
|---------|---------|---------------|
| 损失函数 | `keras.losses.Loss` | `call(y_true, y_pred)` |
| 评估指标 | `keras.metrics.Metric` | `update_state()`, `result()`, `reset_state()` |
| 层 | `keras.layers.Layer` | `build()`, `call()` |
| 模型 | `keras.Model` | `call()` |
| 正则化器 | `keras.regularizers.Regularizer` | `__call__(weights)` |

---

## 6. 知识体系总结

| 模块 | 解决的问题 |
|-----|-----------|
| `tf.data` | 数据IO性能优化 |
| TensorFlow张量操作 | 计算基础 |
| 自定义层/模型/损失 | 算法灵活性 |
| `tf.GradientTape` | 自动求导 |
| `@tf.function` | Python性能瓶颈 |

掌握这五个核心模块，即可应对TensorFlow中绝大多数自定义需求。
