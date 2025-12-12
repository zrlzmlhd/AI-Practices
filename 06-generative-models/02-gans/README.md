# 深度卷积生成对抗网络（DCGAN）

本项目实现了基于CIFAR-10数据集的深度卷积生成对抗网络（DCGAN），用于图像生成任务。

## 项目结构

```
02-gans/
├── GAN网络实现.ipynb          # 完整的DCGAN实现（包含训练和可视化）
└── 关于GAN网络的说明.md       # GAN理论、架构和训练技巧详解
```

## 实现特点

### 网络架构
- **生成器**：使用转置卷积上采样，从32维噪声生成32×32×3彩色图像
- **判别器**：使用步进卷积下采样，输出真假概率
- **激活函数**：生成器使用tanh，判别器使用LeakyReLU
- **正则化**：判别器使用Dropout防止过拟合

### 训练策略
- 交替训练生成器和判别器
- 标签平滑技术提高训练稳定性
- 梯度裁剪防止梯度爆炸
- 学习率平衡（生成器学习率为判别器的一半）

### 功能模块
1. **模型构建**：生成器、判别器和GAN的完整实现
2. **训练循环**：包含损失记录和状态监控
3. **可视化**：训练过程中定期生成图像
4. **潜在空间探索**：线性插值和批量生成

## 使用方法

### 环境要求
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

### 安装依赖
```bash
pip install tensorflow numpy matplotlib
```

### 运行训练
打开Jupyter Notebook并按顺序执行所有单元格：
```bash
jupyter notebook "GAN网络实现.ipynb"
```

### 训练参数
- **迭代次数**：10000（可根据需要调整）
- **批次大小**：20
- **潜在维度**：32
- **学习率**：判别器0.0008，生成器0.0004

## 输出结果

训练完成后会生成以下文件：
- `gan_output/generated_epoch_*.png`：每100步生成的图像
- `gan_output/gan_weights_step_*.h5`：保存的模型权重
- `gan_output/training_history.png`：训练损失和准确率曲线
- `gan_output/latent_interpolation.png`：潜在空间插值结果
- `gan_output/generated_grid.png`：批量生成的图像网格

## 理论背景

### GAN目标函数
$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]$$

### DCGAN设计原则
1. 使用步进卷积替代池化层
2. 生成器使用Batch Normalization
3. 使用LeakyReLU激活函数
4. 生成器输出层使用tanh

详细理论说明请参考 `关于GAN网络的说明.md`。

## 训练监控

训练过程中需要关注以下指标：
- **判别器损失**：应保持在合理范围，不应趋近于0
- **生成器损失**：应逐渐下降
- **判别器准确率**：理想范围0.6-0.8
  - < 0.5：判别器过弱
  - > 0.95：判别器过拟合

## 常见问题

### 模式崩溃
生成器只生成少数几种图像。
**解决方案**：调整学习率、增加正则化、使用WGAN损失

### 训练不稳定
损失剧烈波动。
**解决方案**：降低学习率、增加标签平滑、增加Dropout

### 判别器过拟合
准确率接近100%。
**解决方案**：增加Dropout、减小判别器容量、添加噪声

## 扩展方向

1. **条件GAN（CGAN）**：添加类别条件生成特定类别图像
2. **Wasserstein GAN（WGAN）**：使用Wasserstein距离提高训练稳定性
3. **Progressive GAN**：逐步增加分辨率生成高清图像
4. **StyleGAN**：添加风格控制机制

## 参考文献

1. Goodfellow, I., et al. (2014). "Generative Adversarial Networks." NIPS.
2. Radford, A., et al. (2015). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." ICLR.
3. Arjovsky, M., et al. (2017). "Wasserstein GAN." ICML.

## 许可证

本项目仅供学习和研究使用。
