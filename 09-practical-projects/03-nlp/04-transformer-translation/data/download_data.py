"""
数据集下载脚本

使用方法:
    cd data
    python download_data.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def download_data():
    """
    下载数据集
    """
    print("="*60)
    print("下载数据集")
    print("="*60)

    print("\n请根据项目需求实现数据下载逻辑")
    print("\n提示:")
    print("  1. 使用Kaggle API下载")
    print("  2. 使用requests下载")
    print("  3. 使用tensorflow/keras内置数据集")

    # TODO: 实现数据下载逻辑


if __name__ == '__main__':
    download_data()
