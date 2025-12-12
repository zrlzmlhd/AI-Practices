# RSNA 2023 腹部创伤检测 - Kaggle 第1名解决方案

> **竞赛排名**：🥇 第1名
> **任务类型**：医学影像多标签分类 + 分割
> **评估指标**：Sample-weighted Multi-label Log Loss
> **原始README**：[English Version](README.md)

---

## 📋 竞赛简介

### 竞赛背景
本竞赛由北美放射学会（RSNA）主办，旨在开发AI系统自动检测腹部CT扫描中的创伤性损伤。快速准确的创伤检测对于急诊医疗至关重要，可以帮助医生优先处理危重患者。

### 任务定义
- **任务类型**：多标签分类 + 图像分割
- **预测目标**：检测5个器官的损伤情况
  - **Liver（肝脏）**：健康、低级损伤、高级损伤
  - **Spleen（脾脏）**：健康、低级损伤、高级损伤
  - **Kidney（肾脏）**：健康、低级损伤、高级损伤
  - **Bowel（肠道）**：健康、损伤
  - **Extravasation（活动性出血）**：无、有
- **数据类型**：腹部CT扫描（DICOM格式）
- **数据规模**：约4,000个患者的CT扫描

### 评估指标
使用 **Sample-weighted Multi-label Log Loss**：
```
Loss = -1/N * Σ(w_i * Σ(y_ij * log(p_ij) + (1-y_ij) * log(1-p_ij)))
```
- 对不同器官损伤赋予不同权重
- 活动性出血（Extravasation）权重最高
- 损失越小，模型性能越好

---

## 🏆 解决方案概述

### 核心思路
本解决方案采用**三阶段流水线**：

1. **Stage 1：3D分割模型**
   - 使用3D U-Net分割器官（肝、脾、肾）
   - 生成器官掩码和边界框
   - 用于后续模型的ROI裁剪

2. **Stage 2：2.5D CNN+RNN（器官损伤检测）**
   - 检测肝、脾、肾、肠道损伤
   - 使用2D CNN提取特征 + GRU建模序列
   - 辅助分割损失提升性能

3. **Stage 3：2.5D CNN+RNN（出血检测）**
   - 专门检测肠道损伤和活动性出血
   - 针对性优化以提高敏感度

### 技术栈
- **深度学习框架**：PyTorch 2.0.1
- **预训练模型**：CoaT (Co-Scale Conv-Attentional Image Transformers)
- **核心库**：
  - segmentation_models_pytorch：分割模型
  - timm：预训练模型库
  - albumentations：数据增强
  - dicomsdl：DICOM文件读取

---

## 💻 硬件要求

### 训练环境
- **GPU**：3 x NVIDIA RTX A6000（每个48GB显存）或 3 x RTX 3090
- **CPU**：多核处理器
- **内存**：至少64GB RAM
- **存储**：至少500GB可用空间（用于存储CT扫描数据）

### 推理环境
- **GPU**：至少1个16GB显存的GPU
- **内存**：至少32GB RAM

### Kaggle环境
所有推理脚本可在Kaggle Notebook上运行（GPU P100/T4）

---

## 🚀 快速开始

### 1. 环境配置

#### 安装依赖
```bash
# 创建虚拟环境
conda create -n rsna2023 python=3.10
conda activate rsna2023

# 安装PyTorch（CUDA 11.8）
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install segmentation-models-pytorch==0.3.3
pip install pretrainedmodels==0.7.4
pip install efficientnet-pytorch==0.7.1
pip install albumentations
pip install timm==0.9.7
pip install transformers==4.31.0
pip install dicomsdl==0.109.2
pip install pytorch-toolbelt
```

### 2. 数据准备

#### 下载竞赛数据
从Kaggle下载数据：
- 链接：https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/data

#### 数据预处理

**步骤1：生成分割数据**
```bash
python Datasets/make_segmentation_data1.py
```

**步骤2：训练3D分割模型**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_segmentation_model.py
```

**步骤3：生成信息数据**
```bash
python Datasets/make_info_data.py
```

**步骤4：生成Theo预处理数据**
```bash
python Datasets/make_theo_data_volumes.py
```

**步骤5：生成自定义预处理数据**
```bash
python Datasets/make_our_data_volumes.py
```

#### 数据目录结构
按照 `paths.py` 中的路径配置放置数据：
```
data/
├── train_images/              # 原始训练CT扫描
├── test_images/               # 原始测试CT扫描
├── segmentation_masks/        # 3D分割掩码
├── theo_preprocessed/         # Theo预处理数据
├── our_preprocessed/          # 自定义预处理数据
└── train.csv                  # 训练标签
```

### 3. 训练模型

本解决方案包含多个模型，每个模型使用不同的配置和种子。

#### 训练器官损伤检测模型

**CoaT Medium模型（完整数据）**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_coatmed384fullseed.py --seed 969696
```

**CoaT Medium模型（新分割+自定义数据）**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_coat_med_newseg_ourdata_4f.py --fold 1
```

**CoaT Medium模型（自定义数据+多种子）**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_coatmed384ourdataseed.py --seed 100

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_coatmed384ourdataseed.py --seed 6969
```

**EfficientNetV2-S模型（多种子）**
```bash
for seed in 3407 123 123123 123123123; do
    CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
        --nproc_per_node=3 \
        TRAIN/train_v2s_try5_v10_fulldata.py --seed $seed
done
```

**CoaT Lite Medium模型（不同学习率和种子）**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_coat_lite_medium_bs2_lr_seed.py --seed 7 --lr 9e-5

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_coat_lite_medium_bs2_lr_seed.py --seed 7777 --lr 10e-5

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_coat_lite_medium_bs2_lr_seed.py --seed 7777777 --lr 11e-5
```

#### 训练出血检测模型

**CoaT Small模型（出血检测+U-Net）**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_coatsmall384extravast4funet.py --fold 1

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_coatsmall384extravast4funet.py --fold 3
```

**CoaT Small模型（完整出血数据）**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_fullextracoatsmall384.py --seed 2024

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    TRAIN/train_fullextracoatsmall384.py --seed 2717
```

**EfficientNetV2-S模型（出血检测）**
```bash
for fold in 1 2 3; do
    CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
        --nproc_per_node=3 \
        TRAIN/train_try11_v8_extrav.py --fold $fold
done
```

### 4. 推理预测

#### 最终提交Notebook
- **提交版本**：https://www.kaggle.com/nischaydnk/rsna-super-mega-lb-ensemble
- 包含所有模型的集成推理代码

#### 本地推理
```bash
# 使用训练好的模型进行推理
python inference.py --model_dir models/ --output_dir submissions/
```

---

## 📊 数据预处理详解

### 1. CT扫描预处理

#### 窗宽窗位调整
使用**软组织窗**（Soft-tissue Window）：
```python
def apply_window(image, window_center=40, window_width=400):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    image = np.clip(image, img_min, img_max)
    image = (image - img_min) / (img_max - img_min)
    return image
```

#### 图像尺寸
所有模型使用 **384 x 384** 像素的图像

### 2. 3D分割与裁剪

**目的**：定位器官位置，减少背景干扰

**流程**：
1. 使用3D U-Net对整个CT扫描进行分割
2. 为每个切片生成肝、脾、肾的掩码
3. 基于器官边界进行研究级裁剪
4. 保留器官可见的切片

### 3. 体积数据生成

**2.5D表示**：
- 从每个患者的CT扫描中提取96个等距切片
- 重塑为 `(32, 3, 384, 384)` 的形状
- 3个通道由相邻切片组成（类似RGB）

**示例**：
```python
# 切片序列：[slice_0, slice_1, ..., slice_95]
# 重塑为32个3通道图像：
# Image_0: [slice_0, slice_1, slice_2]
# Image_1: [slice_3, slice_4, slice_5]
# ...
# Image_31: [slice_93, slice_94, slice_95]
```

### 4. 软标签生成

**目的**：为每个切片生成细粒度标签

**方法**：
1. 计算每个切片中器官的可见度（基于分割掩码）
2. 归一化可见度到 [0, 1]
3. 将患者级标签乘以可见度得到切片级标签

**示例**：
```python
# 患者肝损伤标签 = 1（有损伤）
# 肝可见度序列 = [0., 0., 0.01, 0.05, 0.1, ..., 1.0, ..., 0.1, 0., 0.]
# 切片级标签 = 患者标签 * 可见度
# 结果 = [0., 0., 0.01, 0.05, 0.1, ..., 1.0, ..., 0.1, 0., 0.]
```

---

## 🤖 模型架构详解

### Stage 1：3D分割模型

**架构**：3D U-Net
```python
model = UNet3D(
    in_channels=1,
    out_channels=4,  # 背景 + 肝 + 脾 + 肾
    num_levels=4,
    f_maps=32
)
```

**训练配置**：
- **损失函数**：Dice Loss + BCE Loss
- **优化器**：AdamW
- **学习率**：1e-4
- **批大小**：2（每个GPU）

### Stage 2：2.5D CNN + RNN（器官损伤）

**架构概览**：
```
输入: (2, 32, 3, 384, 384)
  ↓
2D CNN Encoder (CoaT/EfficientNet)
  ↓
特征图: (2, 32, hidden_dim)
  ↓
GRU层
  ↓
分类头 + 分割头
  ↓
输出: (2, 32, n_classes)
```

**详细结构**：
```python
class OrganInjuryModel(nn.Module):
    def __init__(self, backbone='coat_lite_medium'):
        super().__init__()
        # 2D CNN编码器
        self.encoder = timm.create_model(backbone, pretrained=True)

        # GRU层（建模序列依赖）
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # 分类头
        self.classifier = nn.Linear(512, n_classes)

        # 分割头（辅助任务）
        self.seg_head_3 = self.get_mask_head(feat_dim_3)
        self.seg_head_4 = self.get_mask_head(feat_dim_4)

    def get_mask_head(self, nb_ft):
        return nn.Sequential(
            nn.Conv2d(nb_ft, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, kernel_size=1, padding=0),
        )
```

### 辅助分割损失

**关键创新**：使用分割任务作为辅助损失

**优势**：
- 提升训练稳定性
- 强制模型关注器官区域
- CV提升约 +0.01 到 +0.03

**实现**：
```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(
            smp.losses.MULTILABEL_MODE,
            from_logits=True
        )

    def forward(self, outputs, targets,
                masks_outputs, masks_outputs2, masks_targets):
        # 分类损失
        loss1 = self.bce(outputs, targets.float())

        # 分割损失（从两个特征层）
        masks_targets = masks_targets.float().flatten(0, 1)
        loss2 = (self.dice(masks_outputs, masks_targets) +
                 self.dice(masks_outputs2, masks_targets))

        # 组合损失
        loss = loss1 + (loss2 * 0.125)
        return loss
```

### 使用的Backbone

**最终集成中的模型**：
1. **CoaT Lite Medium + GRU**
   - 来源：https://github.com/mlpc-ucsd/CoaT
   - 特点：结合卷积和注意力机制

2. **CoaT Lite Small + GRU**
   - 轻量级版本

3. **EfficientNetV2-S + GRU**
   - 来源：timm库
   - 特点：高效的卷积网络

---

## 📈 训练策略

### 交叉验证
- **方法**：4折GroupKFold（患者级分组）
- **目的**：确保同一患者的数据不会同时出现在训练集和验证集

### 数据增强
```python
import albumentations as A

augmentations = A.Compose([
    A.Perspective(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(p=0.5, limit=(-25, 25)),
])
```

### 训练配置

**通用设置**：
- **学习率**：1e-4 到 4e-4
- **优化器**：AdamW
- **调度器**：Cosine Annealing with Warmup
- **损失函数**：
  - 分类：BCE Loss
  - 分割：Dice Loss

**示例配置**：
```python
optimizer = AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01
)

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6
)
```

---

## 🎯 模型集成策略

### 切片级集成
对于同一模型的不同折：
```python
# 在切片级别进行集成
slice_preds = []
for fold in range(4):
    pred = model_fold[fold].predict(slices)
    slice_preds.append(pred)

ensemble_slice = np.mean(slice_preds, axis=0)
```

### 最大值聚合
从切片级预测到患者级预测：
```python
# 对32个切片取最大值
patient_pred = np.max(slice_predictions, axis=0)
```

### 跨模型集成
不同架构和数据的模型在最大值聚合后集成：
```python
# 模型权重（基于CV性能）
weights = {
    'coat_medium_theo': 0.25,
    'coat_medium_ours': 0.25,
    'coat_small': 0.20,
    'efficientnet_v2s': 0.30,
}

final_pred = sum(w * preds[model]
                 for model, w in weights.items())
```

### 后处理
```python
# 缩放因子调整（基于CV优化）
scaling_factors = {
    'liver_injury': 1.0,
    'spleen_injury': 1.0,
    'kidney_injury': 1.0,
    'bowel_injury': 1.2,
    'extravasation': 1.5,  # 提高敏感度
}

for organ, factor in scaling_factors.items():
    final_pred[organ] *= factor
```

---

## 📊 性能指标

### 本地验证
- **最佳单模型 4折OOF CV**：0.326（CoaT Lite Medium）
- **最佳集成 OOF CV**：0.31x
- **Public LB**：0.30x
- **Private LB**：0.29x

### 各器官性能
单模型（CoaT Lite Medium）的器官级OOF：

| 器官 | CV Score | 难度 |
|------|----------|------|
| Liver | 0.32 | 中等 |
| Spleen | 0.31 | 中等 |
| Kidney | 0.33 | 中等 |
| Bowel | 0.35 | 困难 |
| Extravasation | 0.38 | 最困难 |

---

## 💡 关键技巧

### 1. 辅助分割损失
- 使用器官分割作为辅助任务
- 从编码器的最后两层提取特征
- 显著提升模型性能和稳定性

### 2. 软标签策略
- 基于器官可见度生成切片级标签
- 帮助模型学习器官位置和损伤关系
- 提供更细粒度的监督信号

### 3. 多数据源训练
- Theo预处理数据：标准化的窗宽窗位
- 自定义预处理数据：优化的软组织窗
- 两种数据源的模型集成提升鲁棒性

### 4. 2.5D表示
- 平衡2D和3D方法的优势
- 保留空间上下文信息
- 降低计算复杂度

---

## 🎓 学习要点

### 适合学习的内容
1. **医学影像处理**：CT扫描的预处理和窗宽窗位调整
2. **3D分割技术**：使用3D U-Net进行器官分割
3. **2.5D建模**：结合2D和3D的优势
4. **辅助任务学习**：使用分割任务提升分类性能
5. **多阶段流水线**：分割→检测的级联系统

### 可改进的方向
1. **端到端训练**：联合训练分割和分类模型
2. **注意力机制**：引入空间注意力定位损伤区域
3. **3D模型**：使用纯3D CNN或3D Transformer
4. **多模态融合**：结合不同窗宽窗位的图像
5. **弱监督学习**：利用患者级标签进行切片级定位

---

## 📁 项目结构

```
03-RSNA-2023-1st-Place/
├── Datasets/                          # 数据预处理脚本
│   ├── make_segmentation_data1.py
│   ├── make_info_data.py
│   ├── make_theo_data_volumes.py
│   └── make_our_data_volumes.py
├── TRAIN/                             # 训练脚本
│   ├── train_segmentation_model.py
│   ├── train_coatmed384fullseed.py
│   ├── train_coat_med_newseg_ourdata_4f.py
│   ├── train_v2s_try5_v10_fulldata.py
│   ├── train_coat_lite_medium_bs2_lr_seed.py
│   ├── train_coatsmall384extravast4funet.py
│   └── ...
├── models/                            # 模型权重
├── data/                              # 数据目录
├── paths.py                           # 路径配置
├── README.md                          # 英文说明
└── README_CN.md                       # 中文说明（本文件）
```

---

## ⚠️ 注意事项

1. **计算资源**：
   - 训练需要多GPU环境（建议3个A6000或3090）
   - 完整训练需要数天时间
   - 推理可在单GPU上进行

2. **内存需求**：
   - 训练时需要至少64GB RAM
   - 处理CT扫描数据需要大量内存

3. **数据存储**：
   - CT扫描数据非常大（数百GB）
   - 预处理数据也需要大量存储空间
   - 建议使用SSD以加快数据加载

4. **医学影像知识**：
   - 理解CT扫描的窗宽窗位概念
   - 了解腹部解剖结构
   - 熟悉DICOM格式

---

## 🔗 相关资源

### 竞赛链接
- [Kaggle竞赛页面](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
- [解决方案讨论](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion)
- [最终提交Notebook](https://www.kaggle.com/nischaydnk/rsna-super-mega-lb-ensemble)
- [3D分割代码](https://www.kaggle.com/code/haqishen/rsna-2023-1st-place-solution-train-3d-seg/notebook)

### 参考资料
- [CoaT论文](https://arxiv.org/abs/2104.06399)
- [3D U-Net论文](https://arxiv.org/abs/1606.06650)
- [医学影像分割综述](https://arxiv.org/abs/2004.10322)

### 相关竞赛
- [RSNA 2022 Cervical Spine Fracture Detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection)
- [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)

---

## 🤝 贡献

本解决方案由团队协作完成。感谢所有贡献者！

---

## 📄 许可证

本项目遵循原仓库的许可证。

---

**祝你在医学影像AI竞赛中取得好成绩！🏆**
