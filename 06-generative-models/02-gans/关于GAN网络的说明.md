# 生成对抗网络（GAN）技术文档

## 1. 概述

生成对抗网络（Generative Adversarial Networks, GAN）是由Ian Goodfellow等人于2014年提出的一种生成模型。GAN通过两个神经网络的对抗过程来学习数据分布，在图像生成、图像转换、超分辨率等领域取得了突破性进展。

## 2. 核心架构

### 2.1 组成部分

GAN由两个相互对抗的神经网络组成：

**生成器（Generator, G）**
- 功能：从低维潜在空间$z \sim p_z(z)$映射到高维数据空间$x \sim p_{data}(x)$
- 输入：随机噪声向量$z$（通常服从高斯分布或均匀分布）
- 输出：生成的样本$G(z)$
- 目标：生成与真实数据分布相似的样本，欺骗判别器

**判别器（Discriminator, D）**
- 功能：区分真实样本和生成样本
- 输入：样本$x$（可能来自真实数据集或生成器）
- 输出：概率值$D(x) \in [0,1]$，表示样本为真实数据的概率
- 目标：正确分类真实样本和生成样本

### 2.2 数学原理

#### 2.2.1 目标函数

GAN的训练目标是一个极小极大博弈（minimax game）：

$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1-D(G(z)))]
$$

其中：
- 第一项：判别器正确识别真实样本的能力
- 第二项：判别器正确识别生成样本的能力
- 判别器$D$试图最大化$V(D,G)$
- 生成器$G$试图最小化$V(D,G)$

#### 2.2.2 纳什均衡

理论上，当且仅当$p_g = p_{data}$时，GAN达到全局最优，此时：
- $D(x) = 1/2$对所有$x$成立
- 生成器完美复制了数据分布

#### 2.2.3 损失函数分解

**判别器损失：**
$$
L_D = -\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]
$$

判别器的目标是最小化该损失，即：
- 对真实样本输出高概率（接近1）
- 对生成样本输出低概率（接近0）

**生成器损失：**
$$
L_G = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]
$$

生成器的目标是最小化该损失，即让判别器对生成样本输出高概率。

实践中，通常使用非饱和损失（Non-saturating loss）：
$$
L_G = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]
$$

替代原始的：
$$
L_G = \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]
$$

因为后者在训练初期梯度较小，导致学习缓慢。

## 3. DCGAN架构

深度卷积生成对抗网络（Deep Convolutional GAN, DCGAN）是GAN在卷积神经网络上的成功应用，由Radford等人于2015年提出。

### 3.1 架构设计原则

1. **取消池化层**
   - 判别器：使用步进卷积（strided convolution）进行下采样
   - 生成器：使用转置卷积（transposed convolution）进行上采样

2. **使用Batch Normalization**
   - 作用：稳定训练，加速收敛
   - 例外：生成器输出层和判别器输入层不使用BN

3. **移除全连接层**
   - 使用全局池化或全卷积架构
   - 减少参数量，提高泛化能力

4. **激活函数**
   - 生成器：隐藏层使用ReLU，输出层使用tanh
   - 判别器：使用LeakyReLU（slope=0.2）

### 3.2 生成器架构

典型的DCGAN生成器架构：

```
输入: z ∈ R^latent_dim (潜在空间噪声)
    ↓
Dense: latent_dim → H/4 × W/4 × 512
    ↓
Reshape: → (H/4, W/4, 512)
    ↓
ConvTranspose2D(256, 4×4, strides=2) + BN + ReLU  # (H/2, W/2, 256)
    ↓
ConvTranspose2D(128, 4×4, strides=2) + BN + ReLU  # (H, W, 128)
    ↓
Conv2D(channels, 3×3, strides=1) + tanh           # (H, W, C)
    ↓
输出: x ∈ R^(H×W×C)
```

### 3.3 判别器架构

典型的DCGAN判别器架构：

```
输入: x ∈ R^(H×W×C)
    ↓
Conv2D(64, 4×4, strides=2) + LeakyReLU             # (H/2, W/2, 64)
    ↓
Conv2D(128, 4×4, strides=2) + BN + LeakyReLU      # (H/4, W/4, 128)
    ↓
Conv2D(256, 4×4, strides=2) + BN + LeakyReLU      # (H/8, W/8, 256)
    ↓
Conv2D(512, 4×4, strides=2) + BN + LeakyReLU      # (H/16, W/16, 512)
    ↓
Flatten + Dense(1) + Sigmoid
    ↓
输出: D(x) ∈ [0,1]
```

## 4. 训练策略

### 4.1 交替训练

GAN的训练采用交替优化策略：

1. **固定生成器，训练判别器$k$步**
   - 采样真实数据$x \sim p_{data}$
   - 采样噪声$z \sim p_z$，生成假样本$\tilde{x} = G(z)$
   - 更新判别器参数$\theta_D$以最大化$V(D,G)$

2. **固定判别器，训练生成器1步**
   - 采样噪声$z \sim p_z$
   - 更新生成器参数$\theta_G$以最小化$V(D,G)$

通常设置$k=1$，但在训练初期可以设置$k>1$以防止生成器过早崩溃。

### 4.2 训练技巧

#### 4.2.1 标签平滑（Label Smoothing）

不使用硬标签0和1，而是使用软标签：
- 真实样本：$y \in [0.9, 1.0]$
- 生成样本：$y \in [0.0, 0.1]$

作用：防止判别器过于自信，提高训练稳定性。

#### 4.2.2 单边标签平滑（One-sided Label Smoothing）

仅对真实样本进行标签平滑，生成样本保持标签0：
- 真实样本：$y = 0.9$
- 生成样本：$y = 0$

研究表明单边平滑效果更好。

#### 4.2.3 噪声注入

在判别器输入中添加噪声：
- 实例噪声（Instance Noise）：向输入图像添加高斯噪声
- 特征噪声（Feature Noise）：在中间特征层添加噪声

噪声强度随训练进行逐渐衰减。

#### 4.2.4 梯度裁剪

限制梯度范数，防止梯度爆炸：
```python
optimizer = RMSprop(clipvalue=1.0)  # 裁剪梯度值到[-1, 1]
```

#### 4.2.5 学习率平衡

- 判别器学习率通常高于生成器（如2:1）
- 使用学习率衰减策略
- 推荐优化器：RMSprop或Adam（β1=0.5）

#### 4.2.6 使用合适的激活函数

- tanh输出：将数据归一化到[-1, 1]
- 高斯分布采样：使用$\mathcal{N}(0,1)$而非均匀分布
- LeakyReLU：避免ReLU的"dying"问题

#### 4.2.7 批量大小

- 较大的批量大小有助于稳定训练（如64-256）
- Batch Normalization依赖批量统计，批量太小会不稳定

## 5. 常见问题与解决方案

### 5.1 模式崩溃（Mode Collapse）

**现象：** 生成器只生成少数几种样本，缺乏多样性。

**原因：** 生成器找到了能够欺骗判别器的"捷径"。

**解决方案：**
- 使用Unrolled GAN
- 添加多样性正则项
- 使用Minibatch Discrimination
- 尝试Wasserstein GAN (WGAN)

### 5.2 训练不稳定

**现象：** 损失剧烈波动，生成质量不稳定。

**解决方案：**
- 降低学习率
- 使用梯度惩罚（Gradient Penalty）
- 增加判别器训练频率
- 使用Spectral Normalization

### 5.3 梯度消失

**现象：** 判别器过强，生成器梯度接近0。

**解决方案：**
- 降低判别器学习率
- 增加判别器Dropout
- 使用WGAN或LSGAN损失
- 减少判别器层数

### 5.4 判别器过拟合

**现象：** 判别器准确率接近100%，生成器无法学习。

**判断标准：**
- 判别器损失趋近0
- 判别器准确率 > 0.95
- 生成器损失不断增大

**解决方案：**
- 增加Dropout（如0.3-0.5）
- 添加噪声到判别器输入
- 减小判别器容量（层数/通道数）
- 增加标签平滑强度

## 6. 评估指标

### 6.1 Inception Score (IS)

$$
IS = \exp(\mathbb{E}_x[D_{KL}(p(y|x) || p(y))])
$$

- 范围：$[1, +\infty)$，越高越好
- 衡量生成样本的质量和多样性
- 缺点：不考虑真实数据分布

### 6.2 Fréchet Inception Distance (FID)

$$
FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})
$$

- 范围：$[0, +\infty)$，越低越好
- 比较真实数据和生成数据在特征空间的分布距离
- 更符合人类感知

### 6.3 Precision & Recall

- **Precision**：生成样本中有多少比例是真实的
- **Recall**：真实样本中有多少比例被生成器覆盖
- 用于诊断模式崩溃和生成质量

## 7. GAN变体

### 7.1 条件GAN (CGAN)

在生成器和判别器中加入条件信息$y$：
$$
\min_G \max_D V(D,G) = \mathbb{E}_{x}[\log D(x|y)] + \mathbb{E}_{z}[\log(1-D(G(z|y)))]
$$

应用：类别条件生成、图像到图像翻译

### 7.2 Wasserstein GAN (WGAN)

使用Wasserstein距离替代JS散度：
$$
W(p_r, p_g) = \inf_{\gamma \sim \Pi(p_r,p_g)} \mathbb{E}_{(x,y)\sim\gamma}[||x-y||]
$$

优点：
- 训练更稳定
- 损失函数与生成质量相关
- 缓解模式崩溃

### 7.3 Progressive GAN

逐步增加生成器和判别器的分辨率：
- 从4×4开始，逐步增长到1024×1024
- 训练更稳定，生成高分辨率图像

### 7.4 StyleGAN

引入风格控制机制：
- 映射网络：$z \to w$
- 自适应实例归一化（AdaIN）
- 渐进式生成

应用：高质量人脸生成、风格迁移

## 8. 应用场景

### 8.1 图像生成
- 人脸生成（StyleGAN）
- 艺术创作（ArtGAN）
- 数据增强

### 8.2 图像转换
- Pix2Pix：成对图像翻译
- CycleGAN：非成对图像翻译
- StarGAN：多域图像翻译

### 8.3 超分辨率
- SRGAN：照片级超分辨率
- ESRGAN：增强型SRGAN

### 8.4 视频生成
- VideoGAN
- MoCoGAN：运动和内容分解

### 8.5 文本生成
- SeqGAN：序列生成
- TextGAN：文本生成

## 9. 实现注意事项

### 9.1 数据预处理
- 图像归一化：使用tanh输出时归一化到[-1, 1]
- 数据增强：提高泛化能力
- 批量归一化：稳定训练

### 9.2 超参数调优
- 学习率：生成器0.0001-0.0004，判别器0.0004-0.0008
- 批量大小：32-128
- 潜在维度：64-128（根据任务复杂度）
- 训练步数：根据数据集规模，通常50k-200k

### 9.3 监控指标
- 判别器准确率：理想范围0.6-0.8
- 损失比率：$L_G / L_D$接近1
- 生成样本质量：定期可视化
- FID分数：定期评估

### 9.4 调试技巧
- 先用小数据集验证
- 使用TensorBoard监控
- 保存中间权重
- 记录超参数配置

## 10. 参考文献

1. Goodfellow, I., et al. (2014). "Generative Adversarial Networks." NIPS.
2. Radford, A., et al. (2015). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." ICLR.
3. Arjovsky, M., et al. (2017). "Wasserstein GAN." ICML.
4. Karras, T., et al. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks." CVPR.
5. Salimans, T., et al. (2016). "Improved Techniques for Training GANs." NIPS.

## 11. 相关资源

- PyTorch GAN Zoo: https://github.com/facebookresearch/pytorch_GAN_zoo
- TensorFlow GAN: https://github.com/tensorflow/gan
- GAN Lab: https://poloclub.github.io/ganlab/
- Papers with Code: https://paperswithcode.com/method/gan
