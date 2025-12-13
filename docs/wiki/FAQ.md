# FAQ

常见问题解答。

---

## Environment & Installation

### Q: 如何安装项目所需的全部依赖？

推荐使用 Conda 创建隔离环境：

```bash
conda create -n ai-practices python=3.10 -y
conda activate ai-practices
pip install -r requirements.txt
```

### Q: TensorFlow 安装报错怎么办？

1. 确保 Python 版本为 3.10.x
2. 升级 pip: `pip install --upgrade pip`
3. Mac M1/M2/M3 使用: `pip install tensorflow-macos tensorflow-metal`

### Q: GPU 不可用怎么办？

```python
# 检查 TensorFlow GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# 检查 PyTorch GPU
import torch
print(torch.cuda.is_available())
```

如果不可用，检查 CUDA 和 cuDNN 版本是否匹配。

---

## Jupyter Notebook

### Q: Notebook 中无法导入已安装的包？

为 Jupyter 注册环境：

```bash
python -m ipykernel install --user --name=ai-practices
```

然后在 Notebook 中选择正确的 Kernel。

### Q: Notebook 运行很慢或卡死？

1. 清理输出: Cell → All Output → Clear
2. 重启 Kernel: Kernel → Restart
3. 减小数据量进行测试

---

## Training

### Q: 损失值为 NaN 或 Inf？

1. 降低学习率
2. 检查数据中是否有 NaN/Inf 值
3. 添加梯度裁剪: `optimizer = Adam(clipnorm=1.0)`

### Q: 过拟合怎么办？

1. 添加正则化 (L1/L2)
2. 使用 Dropout
3. 数据增强
4. 早停 (Early Stopping)
5. 减小模型容量

### Q: GPU 内存不足 (OOM)?

```python
# TensorFlow: 设置内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 减小 batch_size
# 使用混合精度训练
```

---

## Learning Path

### Q: 应该先学机器学习还是深度学习？

推荐先学机器学习基础 (Module 01)：
- 理解基本概念（损失函数、优化、过拟合等）
- 掌握数据预处理和特征工程
- 深度学习是机器学习的子集

### Q: 应该学 TensorFlow 还是 PyTorch？

- **初学者**: 从 Keras (TensorFlow) 开始
- **研究者**: PyTorch 更灵活
- **工程师**: TensorFlow 部署方便
- **建议**: 两者都学，本项目两者都覆盖

---

## More Help

没找到答案？请在 [GitHub Issues](https://github.com/zimingttkx/AI-Practices/issues) 中搜索或提问。

---

<div align="center">

**[[← Home|Home]]**

</div>
