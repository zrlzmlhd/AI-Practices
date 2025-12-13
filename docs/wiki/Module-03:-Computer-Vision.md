# Module 03: Computer Vision

**计算机视觉** - 系统学习 CNN 架构演进与视觉任务。

---

## Module Overview

| Attribute | Value |
|:----------|:------|
| **Prerequisites** | Module 02 |
| **Duration** | 3-4 weeks |
| **Notebooks** | 18+ |
| **Difficulty** | ⭐⭐⭐ Intermediate |

---

## Learning Objectives

- ✅ 理解卷积神经网络的原理
- ✅ 掌握经典 CNN 架构 (LeNet → ViT)
- ✅ 应用迁移学习和微调技术
- ✅ 使用 Grad-CAM 等可视化技术解释模型

---

## Architecture Evolution

```
LeNet (1998)     →    AlexNet (2012)    →    VGG (2014)
    5 layers            8 layers            16-19 layers
    60K params          60M params          138M params

        ↓                   ↓                   ↓

GoogLeNet (2014) →    ResNet (2015)     →    DenseNet (2016)
   Inception              Skip connections      Dense connections
   22 layers              50-152 layers         121-264 layers

        ↓                   ↓                   ↓

EfficientNet (2019) → Vision Transformer (2020) → Swin (2021)
   Compound scaling       Self-attention           Shifted windows
   5.3M-66M params        86M-632M params          88M-197M params
```

---

## Submodules

### 01. CNN Basics

| Topic | Key Concepts | Notebooks |
|:------|:-------------|:----------|
| Convolution | Filters, Stride, Padding | `01_convolution.ipynb` |
| Pooling | Max, Average, Global | `02_pooling.ipynb` |
| Architecture Design | Feature Extraction + Classifier | `03_architecture.ipynb` |

**Key Equations:**
```
Convolution:
(I * K)ᵢⱼ = Σₘ Σₙ I(i+m, j+n) · K(m, n)

Output Size:
O = (I - K + 2P) / S + 1
where I=input, K=kernel, P=padding, S=stride
```

### 02. Classic Architectures

| Architecture | Key Innovation | Notebooks |
|:-------------|:---------------|:----------|
| LeNet-5 | First successful CNN | `01_lenet.ipynb` |
| AlexNet | ReLU, Dropout, GPU | `02_alexnet.ipynb` |
| VGG | Small filters (3×3) | `03_vgg.ipynb` |
| GoogLeNet | Inception modules | `04_googlenet.ipynb` |
| ResNet | Residual connections | `05_resnet.ipynb` |
| DenseNet | Dense connections | `06_densenet.ipynb` |
| EfficientNet | Compound scaling | `07_efficientnet.ipynb` |

### 03. Transfer Learning

| Technique | Description | Notebooks |
|:----------|:------------|:----------|
| Feature Extraction | Freeze backbone | `01_feature_extraction.ipynb` |
| Fine-tuning | Unfreeze layers | `02_fine_tuning.ipynb` |
| Progressive Resizing | Curriculum learning | `03_progressive.ipynb` |

**Example:**
```python
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])
```

### 04. Model Visualization

| Technique | Purpose | Notebooks |
|:----------|:--------|:----------|
| Grad-CAM | Class activation maps | `01_gradcam.ipynb` |
| Occlusion Sensitivity | Region importance | `02_occlusion.ipynb` |
| Filter Visualization | Feature detectors | `03_filters.ipynb` |

---

## Benchmark Results

| Model | ImageNet Top-1 | Params | FLOPs |
|:------|:--------------:|:------:|:-----:|
| ResNet-50 | 76.1% | 25.6M | 4.1G |
| ResNet-152 | 78.3% | 60.2M | 11.6G |
| EfficientNet-B0 | 77.1% | 5.3M | 0.4G |
| EfficientNet-B7 | 84.3% | 66M | 37G |
| ViT-B/16 | 77.9% | 86M | 17.6G |
| ViT-L/16 | 79.7% | 304M | 61.6G |

---

<div align="center">

**[[← Module 02|Module-02:-Neural-Networks]]** | **[[Module 04 →|Module-04:-Sequence-Models]]**

</div>
