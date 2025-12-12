# RSNA 2024 腰椎退行性分类 - Kaggle 第7名解决方案

> **竞赛排名**：🏅 第7名
> **任务类型**：医学影像多类别分类
> **评估指标**：Sample-weighted Multi-class Log Loss
> **原始README**：[English Version](README.md)

---

## 📋 竞赛简介

### 竞赛背景
本竞赛由北美放射学会（RSNA）主办，旨在开发AI系统自动分类腰椎MRI扫描中的退行性疾病。腰椎退行性疾病是导致下背痛和神经症状的主要原因，准确的诊断对于制定治疗方案至关重要。

### 任务定义
- **任务类型**：多类别分类（3类）
- **预测目标**：评估腰椎的3种退行性病变严重程度
  - **Spinal Canal Stenosis（椎管狭窄）**：正常/轻度/中度/重度
  - **Neural Foraminal Narrowing（神经孔狭窄）**：正常/轻度/中度/重度
  - **Subarticular Stenosis（关节下狭窄）**：正常/轻度/中度/重度
- **评估位置**：
  - 5个椎间盘水平：L1/L2, L2/L3, L3/L4, L4/L5, L5/S1
  - 神经孔和关节下狭窄需要评估左右两侧
- **数据类型**：腰椎MRI扫描（DICOM格式）
  - Sagittal T1（矢状位T1加权）
  - Sagittal T2（矢状位T2加权）
  - Axial T2（轴位T2加权）

### 评估指标
使用 **Sample-weighted Multi-class Log Loss**：
```
Loss = -1/N * Σ(w_i * Σ(y_ic * log(p_ic)))
```
- 对不同严重程度赋予不同权重
- 重度病变权重最高
- 损失越小，模型性能越好

---

## 🏆 解决方案概述

### 核心思路
本解决方案采用**单阶段多视图学习**方法：

1. **多视图输入**：
   - Sagittal T1：提供整体脊柱结构信息
   - Sagittal T2：显示椎间盘和神经根
   - Axial T2：提供横断面细节

2. **形状对齐**：
   - 使用关键点检测对齐MRI图像
   - 标准化不同患者的脊柱位置
   - 提高模型的泛化能力

3. **2D+3D混合建模**：
   - 2D CNN提取单切片特征
   - 3D解码器建模空间关系
   - 结合两者优势

### 技术栈
- **编程语言**：Python 3.10+
- **深度学习框架**：PyTorch
- **预训练模型**：PVT-v2, ConvNeXt, EfficientNet
- **核心库**：
  - timm：预训练模型库
  - albumentations：数据增强
  - pydicom：DICOM文件处理

---

## 💻 硬件要求

### 训练环境
本解决方案使用 **HP Z8 Fury-G5 工作站**：

- **操作系统**：Ubuntu 22.04.4 LTS
- **CPU**：Intel Xeon w7-3455 @ 2.5GHz, 24核心, 48线程
- **内存**：256GB RAM
- **GPU**：2 x NVIDIA Ada A6000（每个48GB显存）

### 最低配置建议
- **GPU**：至少1个24GB显存的GPU（如RTX 3090、RTX 4090）
- **内存**：至少64GB RAM
- **存储**：至少200GB可用空间

### Kaggle环境
推理脚本可在Kaggle Notebook上运行（GPU P100/T4）

---

## 🚀 快速开始

### 1. 环境配置

#### 系统要求
- Python >= 3.10.9
- CUDA 11.8+
- Ubuntu 22.04 或类似Linux系统

#### 安装依赖
```bash
# 创建虚拟环境
conda create -n rsna2024 python=3.10
conda activate rsna2024

# 安装依赖包
pip install -r requirements.txt
```

#### requirements.txt 主要依赖
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
albumentations>=1.3.0
pydicom>=2.3.0
opencv-python>=4.7.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
tqdm>=4.65.0
```

### 2. 目录结构设置

按照以下结构组织项目：
```
04-RSNA-2024-Lumbar-Spine/
├── <DATA_KAGGLE_DIR>              # Kaggle原始数据
│   └── rsna-2024-lumbar-spine-degenerative-classification/
│       ├── test_images/
│       ├── train_images/
│       ├── train.csv
│       ├── train_label_coordinates.csv
│       ├── train_series_descriptions.csv
│       └── ...
├── <DATA_PROCESSED_DIR>           # 预处理数据
│   ├── train_label_coordinates.fix01b.csv
│   ├── nfn_sag_t1_mean_shape.512.npy
│   ├── scs_sag_t2_mean.512.npy
│   └── ...（运行脚本后生成）
├── <RESULT_DIR>                   # 训练输出
│   ├── one-stage-nfn-fixed/
│   ├── one-stage-nfn-bugged/
│   └── one-stage-scs/
├── src/                           # 源代码
├── LICENSE
├── README.md
├── README_CN.md                   # 本文件
└── requirements.txt
```

#### 配置路径
编辑 `/src/third_party/_dir_setting_.py`：
```python
# 使用完整路径
DATA_KAGGLE_DIR = '/path/to/kaggle/data'
DATA_PROCESSED_DIR = '/path/to/processed/data'
RESULT_DIR = '/path/to/results'
```

### 3. 数据准备

#### 下载竞赛数据
从Kaggle下载数据：
- 链接：https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data
- 解压到 `<DATA_KAGGLE_DIR>/rsna-2024-lumbar-spine-degenerative-classification/`

#### 准备预处理数据
`<DATA_PROCESSED_DIR>` 包含3个手动创建的文件：

1. **train_label_coordinates.fix01b.csv**
   - 修正后的椎管狭窄标注点坐标
   - 位于本仓库的 `DATA_PROCESSED_DIR/` 文件夹

2. **nfn_sag_t1_mean_shape.512.npy**
   - 神经孔狭窄的平均参考形状
   - 从 https://www.kaggle.com/code/hengck23/shape-alignment 生成

3. **scs_sag_t2_mean.512.npy**
   - 椎管狭窄的平均参考形状
   - 从 https://www.kaggle.com/code/hengck23/shape-alignment 生成

#### 生成其他预处理数据
```bash
python src/process-data-01/run_make_data.py
```

#### 下载预处理数据（可选）
所有预处理数据的备份可从Google Drive下载：
- 链接：https://drive.google.com/drive/folders/1jPPxAP6DHGQMHJPUGjPO7_Q5Asrj_LL3?usp=sharing

### 4. 训练模型

#### 神经孔狭窄（NFN）模型

**注意**：提交的模型存在翻转增强的bug（左右关键点未重新排序）

**训练有bug的模型（复现提交结果）**：
```bash
cd src/nfn_trainer_bugged
python run_train_nfn_pvtv2_b4_bugged.py
```
输出：`<RESULT_DIR>/one-stage-nfn-bugged/pvt_v2_b4-decoder3d-01/`

**训练修复后的模型**：
```bash
cd src/nfn_trainer
python run_train_nfn_pvtv2_b4_fixed.py
```
输出：`<RESULT_DIR>/one-stage-nfn-fixed/pvt_v2_b4-decoder3d-01/`

**训练其他backbone（可选）**：
```bash
cd src/nfn_trainer

# ConvNeXt Small
python run_train_nfn_covnext_small.py

# EfficientNet B5
python run_train_nfn_effnet_b5.py
```

输出：
- `<RESULT_DIR>/one-stage-nfn-fixed/convnext_small-decoder3d-01/`
- `<RESULT_DIR>/one-stage-nfn-fixed/effnet_b5-decoder3d-01/`

**本地验证和集成**：
```bash
# 有bug版本的集成
cd src/nfn_trainer_bugged
python run_ensemble_and_local_validation.py

# 修复版本的集成
cd src/nfn_trainer
python run_ensemble_and_local_validation.py
```

#### 椎管狭窄（SCS）模型

**注意**：单阶段SCS模型未用于最终提交（未提升Public LB分数）

**训练SCS模型（可选）**：
```bash
cd src/scs_trainer

# PVT-v2 B4
python run_train_scs_pvtv2_b4_fixed.py

# ConvNeXt Base
python run_train_scs_covnext_base.py

# EfficientNet B3
python run_train_scs_effnet_b3.py
```

输出：
- `<RESULT_DIR>/one-stage-scs/pvt_v2_b4-decoder2d-01/`
- `<RESULT_DIR>/one-stage-scs/convnext_base-decoder2d-01/`
- `<RESULT_DIR>/one-stage-scs/effnet_b4-decoder2d-01/`

**本地验证和集成**：
```bash
cd src/scs_trainer
python run_ensemble_and_local_validation.py
```

### 5. 推理预测

#### 团队提交Notebook
- **提交版本**：https://www.kaggle.com/code/hengck23/lhw-v24-ensemble-add-heng
- **后提交版本**：https://www.kaggle.com/code/hengck23/post-lhw-v24-ensemble-add-heng

#### 单独推理Demo
- **Heng部分**：https://www.kaggle.com/code/hengck23/clean-final-submit02-scs-nfn-ensemble

---

## 📊 数据说明

### MRI序列类型

#### 1. Sagittal T1（矢状位T1加权）
- **用途**：评估神经孔狭窄（NFN）
- **特点**：
  - 显示骨骼结构（高信号）
  - 脂肪组织呈高信号
  - 提供整体脊柱形态

#### 2. Sagittal T2（矢状位T2加权）
- **用途**：评估椎管狭窄（SCS）
- **特点**：
  - 显示椎间盘和脊髓（高信号）
  - 水分呈高信号
  - 椎间盘退变清晰可见

#### 3. Axial T2（轴位T2加权）
- **用途**：评估所有3种病变
- **特点**：
  - 横断面视图
  - 显示神经根和椎管细节
  - 每个椎间盘水平多个切片

### 标注数据

#### 标签格式
```csv
study_id,condition,level,severity
12345,spinal_canal_stenosis,l1_l2,Normal/Mild
12345,spinal_canal_stenosis,l2_l3,Moderate
12345,left_neural_foraminal_narrowing,l3_l4,Severe
...
```

#### 严重程度分类
- **Normal/Mild**：正常或轻度（0级）
- **Moderate**：中度（1级）
- **Severe**：重度（2级）

#### 评估位置
- **椎管狭窄**：5个水平（L1/L2到L5/S1）
- **神经孔狭窄**：5个水平 × 2侧（左/右）= 10个位置
- **关节下狭窄**：5个水平 × 2侧（左/右）= 10个位置
- **总计**：25个预测目标

### 关键点标注
```csv
study_id,series_id,instance_number,condition,level,x,y
12345,67890,15,spinal_canal_stenosis,l3_l4,256,384
```
- 用于定位病变位置
- 辅助形状对齐

---

## 🤖 模型架构详解

### 整体架构

```
输入: Sagittal MRI (512x512)
  ↓
形状对齐（基于关键点）
  ↓
2D CNN Encoder (PVT-v2/ConvNeXt/EfficientNet)
  ↓
特征图: (H/32, W/32, C)
  ↓
3D Decoder（建模空间关系）
  ↓
分类头（每个水平/侧别）
  ↓
输出: 25个位置的3类概率
```

### 神经孔狭窄（NFN）模型

**输入**：Sagittal T1序列
```python
class NFNModel(nn.Module):
    def __init__(self, backbone='pvt_v2_b4'):
        super().__init__()
        # 2D编码器
        self.encoder = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True
        )

        # 3D解码器
        self.decoder3d = Decoder3D(
            in_channels=[64, 128, 320, 512],
            out_channels=256
        )

        # 分类头（10个位置：5个水平×2侧）
        self.classifier = nn.ModuleList([
            nn.Linear(256, 3)  # 3类：Normal/Mild, Moderate, Severe
            for _ in range(10)
        ])

    def forward(self, x):
        # x: (B, C, H, W)
        features = self.encoder(x)
        decoded = self.decoder3d(features)
        outputs = [head(decoded) for head in self.classifier]
        return outputs
```

### 椎管狭窄（SCS）模型

**输入**：Sagittal T2序列
```python
class SCSModel(nn.Module):
    def __init__(self, backbone='pvt_v2_b4'):
        super().__init__()
        # 2D编码器
        self.encoder = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True
        )

        # 2D解码器（SCS使用2D）
        self.decoder2d = Decoder2D(
            in_channels=[64, 128, 320, 512],
            out_channels=256
        )

        # 分类头（5个水平）
        self.classifier = nn.ModuleList([
            nn.Linear(256, 3)
            for _ in range(5)
        ])

    def forward(self, x):
        features = self.encoder(x)
        decoded = self.decoder2d(features)
        outputs = [head(decoded) for head in self.classifier]
        return outputs
```

### 3D解码器

```python
class Decoder3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
            for in_ch in in_channels
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # features: list of (B, C_i, H_i, W_i)
        decoded = [block(feat) for block, feat in zip(self.blocks, features)]
        fused = torch.cat(decoded, dim=1)
        output = self.fusion(fused)
        return output
```

---

## 📈 训练策略

### 形状对齐

**目的**：标准化不同患者的脊柱位置和方向

**方法**：
1. 检测关键点（椎体中心）
2. 计算仿射变换矩阵
3. 将图像对齐到参考形状

**代码示例**：
```python
def align_shape(image, keypoints, reference_shape):
    # 计算仿射变换
    transform = cv2.estimateAffinePartial2D(
        keypoints,
        reference_shape
    )[0]

    # 应用变换
    aligned = cv2.warpAffine(
        image,
        transform,
        (512, 512)
    )
    return aligned
```

### 交叉验证
- **方法**：5折交叉验证
- **分割方式**：按study_id分组（患者级）
- **评估指标**：Multi-class Log Loss

### 数据增强
```python
import albumentations as A

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.5
    ),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
])
```

### 训练配置

**通用设置**：
```python
# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

# 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)

# 损失函数
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 2.0, 4.0])  # 重度病变权重更高
)
```

**训练循环**：
```python
for epoch in range(epochs):
    for batch in train_loader:
        images, labels = batch
        outputs = model(images)

        # 计算多个位置的损失
        loss = sum(
            criterion(out, label)
            for out, label in zip(outputs, labels)
        ) / len(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
```

---

## 💡 关键技巧

### 1. 形状对齐
- **问题**：不同患者的脊柱位置、角度差异大
- **解决**：基于关键点的仿射变换对齐
- **效果**：显著提升模型泛化能力

### 2. 多视图学习
- **Sagittal T1**：神经孔狭窄
- **Sagittal T2**：椎管狭窄
- **Axial T2**：所有病变的细节
- **融合**：不同视图的模型集成

### 3. 2D+3D混合
- **2D编码器**：高效提取单切片特征
- **3D解码器**：建模相邻切片的空间关系
- **优势**：平衡性能和计算效率

### 4. 类别权重
```python
# 对重度病变赋予更高权重
class_weights = {
    'Normal/Mild': 1.0,
    'Moderate': 2.0,
    'Severe': 4.0,
}
```

### 5. Bug修复的影响
- **Bug**：翻转增强时左右关键点未重新排序
- **影响**：导致左右侧预测混淆
- **修复**：在截止日期前仅重训了fold 2和3
- **教训**：数据增强需要仔细验证

---

## 📊 性能指标

### 本地验证
- **NFN模型（PVT-v2-B4）**：
  - 有bug版本 5折CV：0.45x
  - 修复版本 5折CV：0.43x（提升约0.02）
- **SCS模型（PVT-v2-B4）**：
  - 5折CV：0.48x

### 竞赛排名
- **Public LB**：第7名
- **Private LB**：第7名
- **最终得分**：0.4x（具体分数见竞赛页面）

### 各病变性能
| 病变类型 | CV Score | 难度 |
|---------|----------|------|
| Neural Foraminal Narrowing | 0.43 | 中等 |
| Spinal Canal Stenosis | 0.48 | 困难 |
| Subarticular Stenosis | 0.50 | 最困难 |

---

## 🎓 学习要点

### 适合学习的内容
1. **医学影像分析**：MRI图像的预处理和理解
2. **形状对齐技术**：基于关键点的图像配准
3. **多视图学习**：融合不同成像序列的信息
4. **2D+3D混合建模**：平衡性能和效率
5. **类别不平衡处理**：使用类别权重和采样策略

### 可改进的方向
1. **注意力机制**：引入空间注意力定位病变区域
2. **多任务学习**：同时预测多种病变
3. **3D模型**：使用纯3D CNN或3D Transformer
4. **轴位图像利用**：更好地利用Axial T2序列
5. **关键点检测**：端到端学习关键点和分类

---

## 📁 项目结构

```
04-RSNA-2024-Lumbar-Spine/
├── DATA_KAGGLE_DIR/                      # Kaggle原始数据
│   └── rsna-2024-lumbar-spine-degenerative-classification/
├── DATA_PROCESSED_DIR/                   # 预处理数据
│   ├── train_label_coordinates.fix01b.csv
│   ├── nfn_sag_t1_mean_shape.512.npy
│   └── scs_sag_t2_mean.512.npy
├── RESULT_DIR/                           # 训练输出
│   ├── one-stage-nfn-bugged/
│   ├── one-stage-nfn-fixed/
│   └── one-stage-scs/
├── src/                                  # 源代码
│   ├── process-data-01/
│   │   └── run_make_data.py
│   ├── nfn_trainer_bugged/
│   │   ├── run_train_nfn_pvtv2_b4_bugged.py
│   │   └── run_ensemble_and_local_validation.py
│   ├── nfn_trainer/
│   │   ├── run_train_nfn_pvtv2_b4_fixed.py
│   │   ├── run_train_nfn_covnext_small.py
│   │   ├── run_train_nfn_effnet_b5.py
│   │   └── run_ensemble_and_local_validation.py
│   ├── scs_trainer/
│   │   ├── run_train_scs_pvtv2_b4_fixed.py
│   │   ├── run_train_scs_covnext_base.py
│   │   ├── run_train_scs_effnet_b3.py
│   │   └── run_ensemble_and_local_validation.py
│   └── third_party/
│       └── _dir_setting_.py
├── LICENSE
├── README.md                             # 英文说明
├── README_CN.md                          # 中文说明（本文件）
└── requirements.txt                      # 依赖包
```

---

## ⚠️ 注意事项

### 1. 计算资源
- **训练时间**：每个模型约12-24小时（使用2个A6000）
- **GPU需求**：建议至少24GB显存
- **内存需求**：至少64GB RAM

### 2. Bug警告
- 提交的NFN模型存在翻转增强bug
- 使用 `nfn_trainer_bugged` 可复现提交结果
- 使用 `nfn_trainer` 训练修复后的模型

### 3. 数据预处理
- 形状对齐需要关键点标注
- 预处理数据较大（数十GB）
- 建议使用SSD加快数据加载

### 4. 医学影像知识
- 理解不同MRI序列的特点
- 了解腰椎解剖结构
- 熟悉DICOM格式

---

## 🔗 相关资源

### 竞赛链接
- [Kaggle竞赛页面](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)
- [解决方案讨论](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539439)
- [团队提交Notebook](https://www.kaggle.com/code/hengck23/lhw-v24-ensemble-add-heng)
- [后提交Notebook](https://www.kaggle.com/code/hengck23/post-lhw-v24-ensemble-add-heng)
- [Demo Notebook](https://www.kaggle.com/code/hengck23/clean-final-submit02-scs-nfn-ensemble)

### 参考资料
- [PVT-v2论文](https://arxiv.org/abs/2106.13797)
- [ConvNeXt论文](https://arxiv.org/abs/2201.03545)
- [形状对齐代码](https://www.kaggle.com/code/hengck23/shape-alignment)

### 数据集
- [竞赛数据](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data)
- [预处理数据备份](https://drive.google.com/drive/folders/1jPPxAP6DHGQMHJPUGjPO7_Q5Asrj_LL3?usp=sharing)

### 相关竞赛
- [RSNA 2022 Cervical Spine Fracture Detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection)
- [RSNA 2023 Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)

---

## 🤝 贡献

本解决方案由 [@hengck23](https://www.kaggle.com/hengck23) 开发。

特别感谢HP提供Z8 Fury-G5数据科学工作站，强大的计算能力和大显存GPU使我们能够快速设计和实验模型。

---

## 📄 许可证

本项目遵循MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

---

**祝你在医学影像AI竞赛中取得好成绩！🏆**
