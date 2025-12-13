# Installation

本页面提供详细的安装指南，涵盖多种安装方式和平台配置。

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Method 1: Conda (Recommended)](#method-1-conda-recommended)
  - [Method 2: Docker](#method-2-docker)
  - [Method 3: Manual Installation](#method-3-manual-installation)
- [GPU Configuration](#gpu-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting-1)

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|:----------|:--------|:------------|:--------|
| **CPU** | 4 cores @ 2.5GHz | 8 cores @ 3.0GHz | 16+ cores @ 3.5GHz |
| **RAM** | 8 GB | 32 GB | 64 GB |
| **GPU** | GTX 1060 (6GB) | RTX 3080 (10GB) | RTX 4090 (24GB) / A100 |
| **Storage** | 50 GB HDD | 200 GB SSD | 500 GB NVMe SSD |
| **Network** | 10 Mbps | 100 Mbps | 1 Gbps |

### Software Requirements

| Software | Version | Purpose |
|:---------|:--------|:--------|
| **Python** | 3.10.x ~ 3.11.x | 主要编程语言 |
| **Git** | >= 2.30 | 版本控制 |
| **Conda** | >= 23.0 | 环境管理 |
| **CUDA** | >= 11.8 | GPU 计算 (可选) |
| **cuDNN** | >= 8.6 | 深度学习加速 (可选) |

### Supported Operating Systems

| OS | Version | Status |
|:---|:--------|:------:|
| **Ubuntu** | 20.04 LTS, 22.04 LTS | ✅ Fully Supported |
| **macOS** | 12.0+ (Intel/Apple Silicon) | ✅ Fully Supported |
| **Windows** | 10/11 (64-bit) | ✅ Fully Supported |
| **CentOS** | 7, 8 | ⚠️ Limited Support |
| **WSL2** | Ubuntu 20.04/22.04 | ✅ Fully Supported |

---

## Installation Methods

### Method 1: Conda (Recommended)

这是最推荐的安装方式，提供最佳的环境隔离和依赖管理。

#### Step 1: Install Miniconda

```bash
# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/.bashrc

# macOS (Intel)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh -b
source ~/.zshrc

# macOS (Apple Silicon)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh -b
source ~/.zshrc

# Windows (PowerShell)
# Download from https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
# Run installer and follow prompts
```

#### Step 2: Clone Repository

```bash
git clone git@github.com:zimingttkx/AI-Practices.git
cd AI-Practices
```

#### Step 3: Create Environment

**Option A: Using requirements.txt**
```bash
conda create -n ai-practices python=3.10 -y
conda activate ai-practices
pip install -r requirements.txt
```

**Option B: Using environment.yml**
```bash
conda env create -f environment.yml
conda activate ai-practices
```

#### Step 4: Install GPU Support (Optional)

```bash
# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorFlow with CUDA
pip install tensorflow[and-cuda]

# JAX with CUDA (Optional)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

### Method 2: Docker

使用 Docker 可以获得完全隔离的、可复现的环境。

#### Prerequisites

```bash
# Install Docker
# Linux
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# macOS / Windows
# Download Docker Desktop from https://www.docker.com/products/docker-desktop

# Install NVIDIA Container Toolkit (Linux GPU only)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Build and Run

```bash
# Clone repository
git clone git@github.com:zimingttkx/AI-Practices.git
cd AI-Practices

# Build image
docker build -t ai-practices:latest .

# Run container (CPU)
docker run -it --rm \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    ai-practices:latest

# Run container (GPU)
docker run -it --rm --gpus all \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    ai-practices:latest

# Run with docker-compose
docker-compose up -d
```

#### Dockerfile Reference

```dockerfile
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

---

### Method 3: Manual Installation

适用于需要精细控制安装过程的高级用户。

#### Core Dependencies

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core packages
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install scipy>=1.10.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install scikit-learn>=1.3.0
pip install jupyter>=1.0.0
pip install jupyterlab>=4.0.0
```

#### Deep Learning Frameworks

```bash
# TensorFlow
pip install tensorflow>=2.13.0

# PyTorch (CPU)
pip install torch torchvision torchaudio

# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hugging Face
pip install transformers>=4.30.0
pip install datasets>=2.14.0
pip install tokenizers>=0.13.0
```

#### Machine Learning Libraries

```bash
pip install xgboost>=2.0.0
pip install lightgbm>=4.0.0
pip install catboost>=1.2.0
pip install optuna>=3.3.0
```

#### Computer Vision

```bash
pip install opencv-python>=4.8.0
pip install pillow>=10.0.0
pip install albumentations>=1.3.0
pip install timm>=0.9.0
```

#### NLP

```bash
pip install nltk>=3.8.0
pip install spacy>=3.6.0
pip install gensim>=4.3.0
python -m spacy download en_core_web_sm
```

---

## GPU Configuration

### NVIDIA GPU Setup

#### 1. Install NVIDIA Driver

```bash
# Ubuntu
sudo apt update
sudo apt install nvidia-driver-535  # or latest version

# Verify
nvidia-smi
```

#### 2. Install CUDA Toolkit

```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1
```

#### 3. Install cuDNN

```bash
# Download from NVIDIA Developer (requires account)
# https://developer.nvidia.com/cudnn

sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn8 libcudnn8-dev
```

#### 4. Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Apple Silicon (M1/M2/M3) Setup

```bash
# TensorFlow for Apple Silicon
pip install tensorflow-macos
pip install tensorflow-metal

# PyTorch for Apple Silicon
pip install torch torchvision torchaudio

# Verify MPS (Metal Performance Shaders)
python -c "import torch; print(torch.backends.mps.is_available())"
```

---

## Verification

### Complete Verification Script

```python
#!/usr/bin/env python3
"""AI-Practices Installation Verification Script"""

import sys
import importlib

def check_package(name, min_version=None):
    try:
        module = importlib.import_module(name)
        version = getattr(module, '__version__', 'unknown')
        status = '✅'
        if min_version and version < min_version:
            status = '⚠️'
        print(f"{status} {name}: {version}")
        return True
    except ImportError:
        print(f"❌ {name}: Not installed")
        return False

def check_gpu():
    print("\n=== GPU Status ===")

    # TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow GPU: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu}")
        else:
            print("⚠️ TensorFlow GPU: No GPU detected")
    except Exception as e:
        print(f"❌ TensorFlow GPU: {e}")

    # PyTorch GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ PyTorch MPS: Apple Silicon GPU")
        else:
            print("⚠️ PyTorch GPU: No GPU detected")
    except Exception as e:
        print(f"❌ PyTorch GPU: {e}")

def main():
    print("=== AI-Practices Installation Verification ===\n")
    print(f"Python: {sys.version}\n")

    print("=== Core Packages ===")
    packages = [
        ('numpy', '1.24.0'),
        ('pandas', '2.0.0'),
        ('scipy', '1.10.0'),
        ('matplotlib', '3.7.0'),
        ('seaborn', '0.12.0'),
    ]

    for pkg, ver in packages:
        check_package(pkg, ver)

    print("\n=== Machine Learning ===")
    ml_packages = [
        ('sklearn', '1.3.0'),
        ('xgboost', '2.0.0'),
        ('lightgbm', '4.0.0'),
    ]

    for pkg, ver in ml_packages:
        check_package(pkg, ver)

    print("\n=== Deep Learning ===")
    dl_packages = [
        ('tensorflow', '2.13.0'),
        ('torch', '2.0.0'),
        ('transformers', '4.30.0'),
    ]

    for pkg, ver in dl_packages:
        check_package(pkg, ver)

    check_gpu()

    print("\n=== Verification Complete ===")

if __name__ == '__main__':
    main()
```

Save as `verify_installation.py` and run:

```bash
python verify_installation.py
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Conda Environment Activation Fails

```bash
# Solution: Initialize conda for your shell
conda init bash  # or zsh
source ~/.bashrc  # or ~/.zshrc
```

#### Issue 2: TensorFlow GPU Not Detected

```bash
# Check CUDA version compatibility
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"

# Reinstall with correct CUDA version
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

#### Issue 3: PyTorch CUDA Version Mismatch

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch version
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Issue 4: Memory Errors During Training

```python
# TensorFlow: Enable memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PyTorch: Use gradient checkpointing
from torch.utils.checkpoint import checkpoint
```

#### Issue 5: Package Version Conflicts

```bash
# Create fresh environment
conda deactivate
conda env remove -n ai-practices
conda create -n ai-practices python=3.10 -y
conda activate ai-practices
pip install -r requirements.txt
```

更多问题请参考 [[Troubleshooting]] 页面。

---

## Next Steps

- [[Getting Started]] - 快速入门
- [[Architecture]] - 系统架构
- [[Module 01: Foundations]] - 开始学习

---

<div align="center">

**[[← Getting Started|Getting-Started]]** | **[[Architecture →|Architecture]]**

</div>
