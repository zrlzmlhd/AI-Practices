# 05 - 高级专题

掌握深度学习工程化与优化技术，从研究到生产的完整链路。

## 模块概览

| 属性 | 值 |
|:-----|:---|
| **前置要求** | 01-04 模块, 熟悉 PyTorch/TensorFlow |
| **学习时长** | 2-3 周 |
| **Notebooks** | 12+ |
| **难度** | ⭐⭐⭐⭐ 高级 |

## 学习目标

完成本模块后，你将能够：

- ✅ 使用 Optuna/Ray Tune 进行自动超参数优化
- ✅ 掌握分布式训练技术 (DDP, FSDP)
- ✅ 应用模型压缩技术 (量化、剪枝、蒸馏)
- ✅ 使用 ONNX/TensorRT 部署模型到生产环境
- ✅ 理解混合精度训练和梯度累积

---

## 子模块详解

### 01. 超参数优化

自动化搜索最优超参数配置。

| 方法 | 原理 | 工具 |
|:-----|:-----|:-----|
| **网格搜索** | 穷举所有组合 | sklearn GridSearchCV |
| **随机搜索** | 随机采样 | sklearn RandomizedSearchCV |
| **贝叶斯优化** | 基于先验的智能搜索 | Optuna, Hyperopt |
| **进化算法** | 遗传算法优化 | Ray Tune |
| **早停策略** | 提前终止差的试验 | ASHA, Hyperband |

**Optuna 示例**：

```python
import optuna

def objective(trial):
    # 定义搜索空间
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])

    # 构建模型
    model = build_model(hidden_dim, dropout)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # 训练并返回验证指标
    val_loss = train_and_evaluate(model, optimizer)
    return val_loss

# 创建研究并优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=3600)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

**搜索空间设计建议**：

| 超参数 | 推荐范围 | 搜索方式 |
|:-------|:---------|:---------|
| 学习率 | 1e-5 ~ 1e-2 | 对数均匀 |
| Batch Size | 16, 32, 64, 128 | 离散 |
| Hidden Dim | 64 ~ 1024 | 整数 |
| Dropout | 0.1 ~ 0.5 | 均匀 |
| Weight Decay | 1e-6 ~ 1e-2 | 对数均匀 |

---

### 02. 分布式训练

多 GPU/多节点训练加速。

**训练策略对比**：

| 策略 | 原理 | 适用场景 |
|:-----|:-----|:---------|
| **DataParallel (DP)** | 单进程多线程 | 单机多卡，简单场景 |
| **DistributedDataParallel (DDP)** | 多进程，梯度同步 | 单机/多机多卡 |
| **Fully Sharded DDP (FSDP)** | 参数分片 | 超大模型 |
| **DeepSpeed ZeRO** | 优化器状态分片 | 超大模型 |
| **Pipeline Parallelism** | 模型层分布 | 超深模型 |

**DDP 示例**：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)

    # 模型包装
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # 分布式数据加载
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 确保每个 epoch 数据打乱不同
        for batch in dataloader:
            # 训练步骤
            loss = model(batch)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()

# 启动多进程
torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

**通信原语**：

```
AllReduce: 所有进程求和/平均梯度
┌─────┐    ┌─────┐    ┌─────┐
│ GPU0│◄──►│ GPU1│◄──►│ GPU2│
│ g0  │    │ g1  │    │ g2  │
└──┬──┘    └──┬──┘    └──┬──┘
   │          │          │
   └──────────┼──────────┘
              ▼
         g0+g1+g2 (广播到所有GPU)
```

---

### 03. 混合精度训练

使用 FP16/BF16 加速训练并减少显存。

**精度对比**：

| 精度 | 位数 | 范围 | 用途 |
|:-----|:----:|:-----|:-----|
| FP32 | 32 | ±3.4e38 | 默认精度 |
| FP16 | 16 | ±65504 | 混合精度训练 |
| BF16 | 16 | ±3.4e38 | 更大范围，Ampere+ |
| INT8 | 8 | -128~127 | 推理量化 |

**PyTorch AMP 示例**：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # 自动混合精度前向传播
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)

    # 缩放梯度并反向传播
    scaler.scale(loss).backward()

    # 梯度裁剪（可选）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 更新参数
    scaler.step(optimizer)
    scaler.update()
```

---

### 04. 模型压缩

减小模型体积，加速推理。

| 技术 | 原理 | 压缩率 | 精度损失 |
|:-----|:-----|:------:|:--------:|
| **量化** | 降低数值精度 | 2-4x | 低 |
| **剪枝** | 移除冗余参数 | 2-10x | 中 |
| **知识蒸馏** | 小模型学习大模型 | 可变 | 低 |
| **低秩分解** | 矩阵分解 | 2-5x | 中 |

**量化示例 (PyTorch)**：

```python
import torch.quantization as quant

# 动态量化（推理时量化）
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},
    dtype=torch.qint8
)

# 静态量化（需要校准数据）
model.qconfig = quant.get_default_qconfig('fbgemm')
model_prepared = quant.prepare(model)

# 校准
with torch.no_grad():
    for batch in calibration_loader:
        model_prepared(batch)

# 转换
quantized_model = quant.convert(model_prepared)
```

**知识蒸馏**：

```python
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    # 软标签损失
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)

    # 硬标签损失
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

---

### 05. 模型部署

将模型部署到生产环境。

**部署流程**：

```
训练模型 ──► 导出 ONNX ──► 优化 (TensorRT) ──► 部署服务
   │              │              │                │
PyTorch      标准格式       图优化/量化      REST API
TensorFlow                  算子融合        gRPC
```

**ONNX 导出**：

```python
import torch.onnx

# 导出模型
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=13
)

# 验证导出
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

**TensorRT 优化**：

```python
import tensorrt as trt

# 构建引擎
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# 解析 ONNX
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# 配置优化
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # 启用 FP16
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# 构建引擎
engine = builder.build_serialized_network(network, config)
```

**部署框架对比**：

| 框架 | 特点 | 适用场景 |
|:-----|:-----|:---------|
| TorchServe | PyTorch 官方 | PyTorch 模型 |
| TensorFlow Serving | TF 官方，gRPC | TensorFlow 模型 |
| Triton | NVIDIA，多框架 | GPU 推理 |
| ONNX Runtime | 跨平台 | 通用部署 |
| FastAPI + Uvicorn | 轻量级 | 快速原型 |

---

### 06. 实验管理

跟踪实验、管理模型版本。

| 工具 | 特点 | 适用场景 |
|:-----|:-----|:---------|
| **Weights & Biases** | 云端，可视化强 | 团队协作 |
| **MLflow** | 开源，全流程 | 企业部署 |
| **TensorBoard** | TF 官方，轻量 | 快速实验 |
| **Neptune** | 云端，元数据管理 | 大规模实验 |

**W&B 示例**：

```python
import wandb

# 初始化
wandb.init(project="my-project", config={
    "learning_rate": 1e-4,
    "epochs": 100,
    "batch_size": 32
})

# 训练循环中记录
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss, val_acc = evaluate()

    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "epoch": epoch
    })

# 保存模型
wandb.save("model.pt")
wandb.finish()
```

---

## 实验列表

| 实验 | 内容 | 文件 |
|:-----|:-----|:-----|
| Optuna 调参 | 自动超参数搜索 | `01_optuna_tuning.ipynb` |
| DDP 训练 | 分布式数据并行 | `02_ddp_training.ipynb` |
| 混合精度 | AMP 训练加速 | `03_mixed_precision.ipynb` |
| 模型量化 | INT8 量化部署 | `04_quantization.ipynb` |
| 知识蒸馏 | 模型压缩 | `05_knowledge_distillation.ipynb` |
| ONNX 导出 | 模型格式转换 | `06_onnx_export.ipynb` |
| TensorRT | GPU 推理优化 | `07_tensorrt_inference.ipynb` |
| W&B 实验 | 实验追踪管理 | `08_wandb_tracking.ipynb` |

---

## 参考资源

### 文档
- [PyTorch Distributed](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Weights & Biases Docs](https://docs.wandb.ai/)

### 论文
- Hinton et al. (2015). Distilling the Knowledge in a Neural Network
- Micikevicius et al. (2018). Mixed Precision Training
- Li et al. (2020). PyTorch Distributed: Experiences on Accelerating Data Parallel Training

### 工具
- [DeepSpeed](https://www.deepspeed.ai/) - 微软大模型训练框架
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/) - 简化分布式训练
