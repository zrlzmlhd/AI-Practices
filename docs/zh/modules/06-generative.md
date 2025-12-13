# 06 - 生成模型

掌握生成式 AI 核心技术，从 VAE 到 Diffusion Models。

## 模块概览

| 属性 | 值 |
|:-----|:---|
| **前置要求** | 02-神经网络, 概率论, 信息论基础 |
| **学习时长** | 3-4 周 |
| **Notebooks** | 15+ |
| **难度** | ⭐⭐⭐⭐ 高级 |

## 学习目标

完成本模块后，你将能够：

- ✅ 理解生成模型的理论基础和分类
- ✅ 掌握 VAE 的变分推断原理和实现
- ✅ 深入理解 GAN 的对抗训练机制
- ✅ 学习 Diffusion Models 的去噪扩散原理
- ✅ 应用生成模型进行图像生成、风格迁移等任务

---

## 子模块详解

### 01. 生成模型概述

生成模型的分类与对比。

**生成模型分类**：

```
生成模型
├── 显式密度模型
│   ├── 精确推断: 自回归模型 (PixelCNN, GPT)
│   └── 近似推断: VAE, Flow-based
└── 隐式密度模型
    └── GAN
```

| 模型类型 | 代表 | 优点 | 缺点 |
|:---------|:-----|:-----|:-----|
| **VAE** | VAE, β-VAE | 稳定训练，潜在空间有意义 | 生成模糊 |
| **GAN** | DCGAN, StyleGAN | 生成质量高 | 训练不稳定，模式崩塌 |
| **Flow** | RealNVP, Glow | 精确似然，可逆 | 计算量大 |
| **Diffusion** | DDPM, Stable Diffusion | 质量最高，稳定 | 采样慢 |
| **自回归** | PixelCNN, GPT | 精确似然 | 生成慢 |

---

### 02. 变分自编码器 (VAE)

通过变分推断学习数据的潜在表示。

**VAE 架构**：

```
输入 x ──► Encoder ──► μ, σ ──► 重参数化 ──► z ──► Decoder ──► x̂
                         │           │
                         └─── KL散度 ─┘
```

**核心公式**：

**ELBO (Evidence Lower Bound)**：

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

- 第一项：重构损失
- 第二项：KL 散度正则化

**重参数化技巧**：

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**PyTorch 实现**：

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    # 重构损失
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL 散度
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss
```

---

### 03. 生成对抗网络 (GAN)

通过对抗训练生成逼真样本。

**GAN 博弈**：

```
随机噪声 z ──► Generator G ──► 假样本 G(z)
                                    │
                                    ▼
真实样本 x ──────────────────► Discriminator D ──► 真/假判断
```

**目标函数**：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**训练技巧**：

| 问题 | 解决方案 |
|:-----|:---------|
| 模式崩塌 | Mini-batch discrimination, Unrolled GAN |
| 训练不稳定 | Spectral Normalization, WGAN-GP |
| 梯度消失 | 使用 LSGAN 或 WGAN 损失 |
| 评估困难 | FID, IS 指标 |

**DCGAN 实现**：

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 4x4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16
            nn.ConvTranspose2d(128, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 32x32
        )

    def forward(self, z):
        return self.main(z.view(-1, z.size(1), 1, 1))

class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.main(img).view(-1)
```

**GAN 变体演进**：

| 模型 | 年份 | 创新点 |
|:-----|:----:|:-------|
| GAN | 2014 | 对抗训练框架 |
| DCGAN | 2015 | 卷积架构 |
| WGAN | 2017 | Wasserstein 距离 |
| Progressive GAN | 2018 | 渐进式训练 |
| StyleGAN | 2019 | 风格控制 |
| StyleGAN2 | 2020 | 改进架构 |
| StyleGAN3 | 2021 | 消除伪影 |

---

### 04. 扩散模型 (Diffusion Models)

通过逐步去噪生成高质量样本。

**扩散过程**：

```
x₀ ──► x₁ ──► x₂ ──► ... ──► xₜ ──► ... ──► x_T (纯噪声)
 │      │      │              │              │
 └──────┴──────┴──────────────┴──────────────┘
              前向过程 (加噪声)

x_T ──► x_{T-1} ──► ... ──► x₁ ──► x₀ (生成样本)
 │          │                │      │
 └──────────┴────────────────┴──────┘
              反向过程 (去噪声)
```

**核心公式**：

**前向过程**：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

**反向过程**：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**训练目标**：

$$L = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

**简化实现**：

```python
class DiffusionModel(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model  # UNet
        self.timesteps = timesteps

        # 定义 beta schedule
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x0, t):
        """前向加噪"""
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return xt, noise

    def training_step(self, x0):
        """训练步骤"""
        t = torch.randint(0, self.timesteps, (x0.size(0),))
        xt, noise = self.forward_diffusion(x0, t)
        predicted_noise = self.model(xt, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, shape):
        """采样生成"""
        x = torch.randn(shape)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t)
            predicted_noise = self.model(x, t_batch)
            # 去噪步骤
            x = self.denoise_step(x, predicted_noise, t)
        return x
```

**Diffusion 模型演进**：

| 模型 | 特点 |
|:-----|:-----|
| DDPM | 基础扩散模型 |
| DDIM | 加速采样 |
| Latent Diffusion | 潜在空间扩散 |
| Stable Diffusion | 文本到图像 |
| DALL-E 2 | 多模态生成 |
| Imagen | 高分辨率生成 |

---

### 05. 条件生成

根据条件信息控制生成内容。

| 条件类型 | 方法 | 应用 |
|:---------|:-----|:-----|
| 类别标签 | Conditional GAN, cVAE | 指定类别生成 |
| 文本描述 | CLIP 引导, Cross-Attention | 文本到图像 |
| 图像 | Image-to-Image | 风格迁移, 超分辨率 |
| 语义图 | SPADE, ControlNet | 布局控制 |

**Classifier-Free Guidance**：

$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

其中 $w$ 是引导强度，$c$ 是条件，$\emptyset$ 是无条件。

---

## 实验列表

| 实验 | 内容 | 文件 |
|:-----|:-----|:-----|
| VAE 基础 | MNIST VAE 实现 | `01_vae_mnist.ipynb` |
| β-VAE | 解耦表示学习 | `02_beta_vae.ipynb` |
| GAN 入门 | DCGAN 实现 | `03_dcgan.ipynb` |
| WGAN-GP | 稳定 GAN 训练 | `04_wgan_gp.ipynb` |
| 条件 GAN | cGAN 实现 | `05_conditional_gan.ipynb` |
| DDPM | 扩散模型基础 | `06_ddpm.ipynb` |
| 图像生成 | Stable Diffusion 使用 | `07_stable_diffusion.ipynb` |
| 风格迁移 | 神经风格迁移 | `08_style_transfer.ipynb` |

---

## 参考资源

### 论文
- Kingma & Welling (2014). Auto-Encoding Variational Bayes
- Goodfellow et al. (2014). Generative Adversarial Networks
- Ho et al. (2020). Denoising Diffusion Probabilistic Models
- Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models

### 教程
- [Lil'Log - What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

### 工具
- [Diffusers](https://huggingface.co/docs/diffusers/) - Hugging Face 扩散模型库
- [StyleGAN3](https://github.com/NVlabs/stylegan3) - NVIDIA 官方实现
