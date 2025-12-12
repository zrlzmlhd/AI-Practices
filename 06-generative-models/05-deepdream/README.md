# DeepDream 实现指南

## 文件结构

```
05-deepdream/
├── README.md                           # 本文件
├── What is DeepDream                   # 技术原理详解
└── 用Keras实现DeepDream/
    └── 基于Keras 实现DeepDream.ipynb  # 完整实现代码
```

## 快速开始

### 1. 环境要求

- Python 3.8+
- TensorFlow 2.x
- Keras 3.x
- NumPy
- SciPy
- Matplotlib
- Pillow

### 2. 安装依赖

```bash
pip install tensorflow keras numpy scipy matplotlib pillow
```

### 3. 运行步骤

#### 方式一：使用 Jupyter Notebook（推荐）

```bash
cd 用Keras实现DeepDream
jupyter notebook "基于Keras 实现DeepDream.ipynb"
```

按顺序执行所有单元格即可。

#### 方式二：命令行运行

```bash
jupyter nbconvert --to script "基于Keras 实现DeepDream.ipynb"
python "基于Keras 实现DeepDream.py"
```

### 4. 模型权重下载

首次运行需要下载 Inception-v3 权重文件（约87MB）：

- **自动下载**：代码会自动尝试下载
- **手动下载**：如果网络受限，可从以下地址下载：
  - https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
  - 将文件放置到：`~/.keras/models/` 目录

### 5. 使用自己的图像

在notebook的测试部分，替换以下代码中的路径：

```python
result = deepdream('/path/to/your/image.jpg', output_path='deepdream_output.jpg')
```

## 参数配置

### 测试配置（快速验证）

当前notebook使用的是测试配置，适合快速验证代码：

```python
settings = {
    'step_size': 0.01,
    'num_iterations': 20,      # 较少迭代
    'octave_scale': 1.4,
    'num_octaves': 3,           # 较少octave
}
```

### 生产配置（高质量输出）

获得最佳效果，取消注释notebook最后的恢复函数：

```python
reset_to_production_settings()
```

生产参数：
```python
settings = {
    'step_size': 0.01,
    'num_iterations': 50,      # 更多迭代
    'octave_scale': 1.4,
    'num_octaves': 5,           # 更多octave层次
}
```

## 效果调整

### 增强效果强度

- 增大 `step_size`（如 0.02）
- 增加 `num_iterations`（如 100）

### 改变视觉风格

修改 `layer_contributions` 权重：

**精细纹理风格**：
```python
layer_contributions = {
    'mixed2': 5.0,   # 强调低层特征
    'mixed3': 1.0,
    'mixed4': 0.5,
    'mixed5': 0.2,
}
```

**抽象形状风格**：
```python
layer_contributions = {
    'mixed2': 0.1,
    'mixed3': 0.5,
    'mixed4': 2.0,
    'mixed5': 5.0,   # 强调高层特征
}
```

### 更多细节层次

- 增加 `num_octaves`（如 7）
- 调整 `octave_scale`（如 1.5）

## 性能优化

### GPU 加速

代码自动使用可用的GPU。检查GPU状态：

```python
import tensorflow as tf
print("GPU设备:", tf.config.list_physical_devices('GPU'))
```

### 处理速度

- **测试参数**：小图像（300x300）约 10-30秒
- **生产参数**：中等图像（600x600）约 1-3分钟
- 时间取决于：图像尺寸、迭代次数、octave数量、硬件性能

## 常见问题

### 1. 内存不足

- 减小输入图像尺寸
- 减少 `num_octaves`
- 使用较小的 `num_iterations`

### 2. 效果不明显

- 增大 `step_size`
- 增加 `num_iterations`
- 调整 `layer_contributions` 权重

### 3. 出现伪影

- 减小 `step_size`
- 确保图像预处理正确
- 检查图像质量（避免过度压缩的JPEG）

### 4. 模型加载失败

- 检查网络连接
- 手动下载权重文件
- 确认 `~/.keras/models/` 目录权限

## 技术细节

详细的技术原理和数学推导请参阅：

- `What is DeepDream` 文档
- Notebook中的详细注释

## 扩展应用

### 批量处理

```python
import glob

for img_path in glob.glob('images/*.jpg'):
    output_path = img_path.replace('.jpg', '_dream.jpg')
    deepdream(img_path, output_path=output_path, visualize=False)
```

### 视频处理

逐帧处理视频：

```python
import cv2

cap = cv2.VideoCapture('input.mp4')
# 读取帧 -> 处理 -> 写入输出视频
```

### 不同模型

尝试其他预训练模型：

```python
from tensorflow.keras.applications import VGG16, ResNet50

base_model = VGG16(weights='imagenet', include_top=False)
# 或
base_model = ResNet50(weights='imagenet', include_top=False)
```

## 参考资源

- 原始论文：Mordvintsev et al. (2015)
- TensorFlow 官方文档
- Keras Applications API

## 许可说明

本实现仅供学习和研究使用。
