# 03 - 计算机视觉

掌握卷积神经网络架构演进与图像处理核心技术。

## 模块概览

| 属性 | 值 |
|:-----|:---|
| **前置要求** | 02-神经网络, 线性代数, Python |
| **学习时长** | 3-4 周 |
| **Notebooks** | 20+ |
| **难度** | ⭐⭐⭐ 中级 |

## 学习目标

完成本模块后，你将能够：

- ✅ 理解卷积、池化等基本操作的数学原理
- ✅ 掌握 CNN 经典架构的设计思想和演进历程
- ✅ 熟练应用迁移学习和数据增强技术
- ✅ 实现图像分类、目标检测和语义分割任务
- ✅ 理解 Vision Transformer 的工作原理

---

## 子模块详解

### 01. CNN 基础

理解卷积神经网络的核心组件。

| 组件 | 功能 | 关键参数 |
|:-----|:-----|:---------|
| 卷积层 | 特征提取 | kernel_size, stride, padding |
| 池化层 | 下采样，增强平移不变性 | pool_size, stride |
| 激活函数 | 引入非线性 | ReLU, LeakyReLU, GELU |
| 全连接层 | 分类决策 | units, dropout |

**卷积操作公式**：

$$O_{i,j} = \sum_{m}\sum_{n} I_{i+m, j+n} \cdot K_{m,n} + b$$

**输出尺寸计算**：

$$O = \frac{W - K + 2P}{S} + 1$$

其中 $W$ 为输入尺寸，$K$ 为卷积核大小，$P$ 为填充，$S$ 为步长。

**代码实现**：

```python
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

---

### 02. 经典架构演进

CNN 架构的发展历程与核心创新。

**架构演进时间线**：

```
1998        2012        2014           2015        2017         2020
 │           │           │              │           │            │
 ▼           ▼           ▼              ▼           ▼            ▼
LeNet ──► AlexNet ──► VGG/GoogLeNet ──► ResNet ──► SENet ──► EfficientNet/ViT
 │           │           │              │           │            │
 ▼           ▼           ▼              ▼           ▼            ▼
卷积+池化   ReLU+Dropout  深度网络      残差连接    注意力机制    NAS/Transformer
```

| 架构 | 年份 | 层数 | Top-5 错误率 | 核心创新 |
|:-----|:----:|:----:|:------------:|:---------|
| LeNet | 1998 | 5 | - | 卷积神经网络开创 |
| AlexNet | 2012 | 8 | 16.4% | ReLU, Dropout, GPU训练 |
| VGG | 2014 | 16/19 | 7.3% | 小卷积核堆叠 (3×3) |
| GoogLeNet | 2014 | 22 | 6.7% | Inception 模块 |
| ResNet | 2015 | 152 | 3.6% | 残差连接 |
| DenseNet | 2017 | 121+ | 3.5% | 密集连接 |
| EfficientNet | 2019 | - | 2.9% | 复合缩放 |
| ViT | 2020 | - | - | 纯 Transformer |

**ResNet 残差块**：

$$y = F(x, \{W_i\}) + x$$

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        return F.relu(out)
```

---

### 03. 迁移学习

利用预训练模型加速训练和提升性能。

| 策略 | 方法 | 适用场景 |
|:-----|:-----|:---------|
| **特征提取** | 冻结预训练层，只训练分类头 | 数据量少，任务相似 |
| **微调** | 解冻部分/全部层，小学习率训练 | 数据量中等 |
| **从头训练** | 只用预训练权重初始化 | 数据量大，任务差异大 |

**微调策略**：

```python
import torchvision.models as models

# 加载预训练模型
model = models.resnet50(pretrained=True)

# 冻结所有层
for param in model.parameters():
    param.requires_grad = False

# 替换分类头
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

# 只训练分类头
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

---

### 04. 数据增强

提升模型泛化能力的关键技术。

| 增强方法 | 描述 | 适用场景 |
|:---------|:-----|:---------|
| 几何变换 | 翻转、旋转、缩放、裁剪 | 通用 |
| 颜色变换 | 亮度、对比度、饱和度调整 | 通用 |
| Cutout | 随机遮挡图像区域 | 分类任务 |
| Mixup | 混合两张图像及标签 | 分类任务 |
| CutMix | 剪切粘贴图像区域 | 分类任务 |
| AutoAugment | 自动搜索增强策略 | 大规模数据集 |

**PyTorch 数据增强**：

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

### 05. 目标检测

从图像中定位和识别多个目标。

**检测算法分类**：

| 类型 | 代表算法 | 特点 |
|:-----|:---------|:-----|
| **两阶段** | R-CNN, Fast R-CNN, Faster R-CNN | 精度高，速度慢 |
| **单阶段** | YOLO, SSD, RetinaNet | 速度快，端到端 |
| **Anchor-Free** | CenterNet, FCOS | 无需预设锚框 |
| **Transformer** | DETR, Deformable DETR | 端到端，无NMS |

**YOLO 核心思想**：

```
输入图像 ──► 划分网格 ──► 每个网格预测 ──► NMS 后处理
                              │
                              ▼
                    (x, y, w, h, confidence, class_probs)
```

**IoU 计算**：

$$IoU = \frac{Area_{intersection}}{Area_{union}}$$

---

### 06. 语义分割

对图像进行像素级分类。

| 架构 | 特点 | 应用场景 |
|:-----|:-----|:---------|
| FCN | 全卷积，端到端 | 通用分割 |
| U-Net | 编码器-解码器，跳跃连接 | 医学图像 |
| DeepLab | 空洞卷积，ASPP | 高精度分割 |
| Mask R-CNN | 实例分割 | 目标检测+分割 |

**U-Net 架构**：

```
编码器 (下采样)          解码器 (上采样)
    │                        │
    ▼                        ▼
[64] ──────────────────► [64] ──► 输出
    │                        ▲
[128] ─────────────────► [128]
    │                        ▲
[256] ─────────────────► [256]
    │                        ▲
[512] ─────────────────► [512]
    │                        ▲
    └──────► [1024] ─────────┘
              瓶颈层
```

---

### 07. Vision Transformer (ViT)

将 Transformer 应用于计算机视觉。

**ViT 工作流程**：

```
输入图像 ──► 分割为 Patches ──► 线性嵌入 ──► 加位置编码 ──► Transformer Encoder ──► 分类
 (224×224)    (16×16 patches)    (768-dim)
```

**核心公式**：

$$z_0 = [x_{class}; x_p^1E; x_p^2E; ...; x_p^NE] + E_{pos}$$

$$z_l = MSA(LN(z_{l-1})) + z_{l-1}$$

$$z_l = MLP(LN(z_l)) + z_l$$

**ViT vs CNN 对比**：

| 特性 | CNN | ViT |
|:-----|:----|:----|
| 归纳偏置 | 局部性、平移等变性 | 较少 |
| 数据需求 | 中等 | 大量数据 |
| 计算复杂度 | O(n) | O(n²) |
| 全局建模 | 需要深层网络 | 天然支持 |

---

## 实验列表

| 实验 | 内容 | 文件 |
|:-----|:-----|:-----|
| CNN 基础 | 从零实现卷积层 | `01_cnn_basics.ipynb` |
| LeNet 实现 | MNIST 手写数字识别 | `02_lenet_mnist.ipynb` |
| VGG 训练 | CIFAR-10 分类 | `03_vgg_cifar10.ipynb` |
| ResNet 实现 | 残差网络从零实现 | `04_resnet_scratch.ipynb` |
| 迁移学习 | ImageNet 预训练微调 | `05_transfer_learning.ipynb` |
| 数据增强 | 增强策略对比实验 | `06_data_augmentation.ipynb` |
| 目标检测 | YOLO 实现与训练 | `07_yolo_detection.ipynb` |
| 语义分割 | U-Net 医学图像分割 | `08_unet_segmentation.ipynb` |
| ViT 实现 | Vision Transformer | `09_vit_implementation.ipynb` |

---

## 参考资源

### 教材
- Goodfellow et al. (2016). *Deep Learning* - Chapter 9: Convolutional Networks
- Prince, S. (2023). *Understanding Deep Learning* - [在线阅读](https://udlbook.github.io/udlbook/)

### 论文
- He et al. (2016). Deep Residual Learning for Image Recognition
- Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition
- Redmon et al. (2016). You Only Look Once: Unified, Real-Time Object Detection
- Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation

### 视频课程
- [Stanford CS231n](http://cs231n.stanford.edu/) - Convolutional Neural Networks for Visual Recognition
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
