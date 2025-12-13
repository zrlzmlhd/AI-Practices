# Module 02: Neural Networks

**神经网络与深度学习** - 掌握深度学习核心技术与训练方法。

---

## Module Overview

| Attribute | Value |
|:----------|:------|
| **Prerequisites** | Module 01, Linear Algebra, Calculus |
| **Duration** | 3-4 weeks |
| **Notebooks** | 20+ |
| **Difficulty** | ⭐⭐⭐ Intermediate |

---

## Learning Objectives

- ✅ 理解神经网络的数学原理和反向传播算法
- ✅ 掌握 TensorFlow/Keras 框架的使用
- ✅ 应用 BatchNorm、Dropout 等正则化技术
- ✅ 实现自定义层、损失函数和训练循环
- ✅ 构建高效的数据加载管道

---

## Submodules

### 01. Keras Introduction

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| Sequential API | Dense, Activation | `01_sequential_api.ipynb` |
| Functional API | Multi-input/output | `02_functional_api.ipynb` |
| Model Compilation | Optimizer, Loss, Metrics | `03_compilation.ipynb` |
| Training & Evaluation | fit(), evaluate() | `04_training.ipynb` |

**Example:**
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 02. Training Deep Networks

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| Weight Initialization | Xavier, He | `01_initialization.ipynb` |
| Batch Normalization | Internal Covariate Shift | `02_batch_norm.ipynb` |
| Dropout | Regularization | `03_dropout.ipynb` |
| Learning Rate Schedules | Decay, Warmup | `04_lr_schedules.ipynb` |
| Gradient Clipping | Exploding Gradients | `05_gradient_clipping.ipynb` |

**Key Equations:**

```
Xavier Initialization:
W ~ N(0, √(2/(nᵢₙ + nₒᵤₜ)))

He Initialization:
W ~ N(0, √(2/nᵢₙ))

Batch Normalization:
x̂ = (x - μ) / √(σ² + ε)
y = γx̂ + β
```

### 03. Custom Models & Training

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| Custom Layers | call(), build() | `01_custom_layers.ipynb` |
| Custom Loss Functions | __call__() | `02_custom_loss.ipynb` |
| Custom Metrics | update_state() | `03_custom_metrics.ipynb` |
| Custom Training Loop | GradientTape | `04_custom_training.ipynb` |

**Example:**
```python
class CustomLayer(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

### 04. Data Loading & Preprocessing

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| tf.data API | Dataset, Pipeline | `01_tfdata.ipynb` |
| Data Augmentation | Image Transforms | `02_augmentation.ipynb` |
| TFRecord | Efficient Storage | `03_tfrecord.ipynb` |
| Mixed Precision | FP16 Training | `04_mixed_precision.ipynb` |

---

## Key Architectures

### Multi-Layer Perceptron (MLP)

```
Input Layer → Hidden Layers → Output Layer
    ↓              ↓              ↓
  Features     Dense+ReLU       Softmax
   (784)      (512→256→128)      (10)
```

### Optimization Algorithms

| Algorithm | Update Rule | Use Case |
|:----------|:------------|:---------|
| SGD | θ -= α∇θJ | Baseline |
| Momentum | v = βv - α∇θJ; θ += v | Faster convergence |
| RMSprop | Adaptive learning rate per param | RNN |
| Adam | Momentum + RMSprop | Default choice |
| AdamW | Adam + Weight decay | Transformers |

---

## Resources

### Documentation
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Keras Documentation](https://keras.io/guides/)

### Papers
- Ioffe & Szegedy (2015). Batch Normalization
- Kingma & Ba (2015). Adam Optimizer
- Srivastava et al. (2014). Dropout

---

<div align="center">

**[[← Module 01|Module-01:-Foundations]]** | **[[Module 03 →|Module-03:-Computer-Vision]]**

</div>
