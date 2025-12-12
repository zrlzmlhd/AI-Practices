"""
IMDB数据集下载脚本

功能：
    自动下载Stanford IMDB电影评论数据集

数据集信息：
    - 来源：Stanford AI Lab
    - 样本数：50,000条电影评论（训练集25,000，测试集25,000）
    - 任务：二分类（正面/负面情感）
    - 格式：文本文件

使用方法:
    cd data
    python download_data.py
"""

import os
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url, filepath):
    """
    下载文件并显示进度条

    Args:
        url: 下载链接
        filepath: 保存路径
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filepath, 'wb') as f, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_imdb_dataset():
    """
    下载并解压IMDB数据集
    """
    print("="*60)
    print("IMDB数据集下载")
    print("="*60)

    # 数据集URL
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    # 当前目录
    data_dir = Path(__file__).parent
    tar_path = data_dir / "aclImdb_v1.tar.gz"
    extract_dir = data_dir / "aclImdb"

    # 检查数据集是否已存在
    if extract_dir.exists():
        print(f"\n✓ 数据集已存在: {extract_dir}")
        print("\n数据集统计:")

        train_pos = len(list((extract_dir / "train" / "pos").glob("*.txt")))
        train_neg = len(list((extract_dir / "train" / "neg").glob("*.txt")))
        test_pos = len(list((extract_dir / "test" / "pos").glob("*.txt")))
        test_neg = len(list((extract_dir / "test" / "neg").glob("*.txt")))

        print(f"  训练集: {train_pos + train_neg} (正面: {train_pos}, 负面: {train_neg})")
        print(f"  测试集: {test_pos + test_neg} (正面: {test_pos}, 负面: {test_neg})")
        return

    # 下载数据集
    print(f"\n下载数据集...")
    print(f"  URL: {url}")
    print(f"  保存到: {tar_path}")

    try:
        download_file(url, tar_path)
        print(f"✓ 下载完成")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n备用方案:")
        print("  1. 手动下载:")
        print(f"     {url}")
        print(f"  2. 保存到: {data_dir}")
        print(f"  3. 重新运行此脚本解压")
        return

    # 解压数据集
    print(f"\n解压数据集...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print(f"✓ 解压完成: {extract_dir}")
    except Exception as e:
        print(f"✗ 解压失败: {e}")
        return

    # 删除压缩包
    print(f"\n清理临时文件...")
    tar_path.unlink()
    print(f"✓ 已删除: {tar_path}")

    # 统计信息
    print(f"\n数据集统计:")
    train_pos = len(list((extract_dir / "train" / "pos").glob("*.txt")))
    train_neg = len(list((extract_dir / "train" / "neg").glob("*.txt")))
    test_pos = len(list((extract_dir / "test" / "pos").glob("*.txt")))
    test_neg = len(list((extract_dir / "test" / "neg").glob("*.txt")))

    print(f"  训练集: {train_pos + train_neg} (正面: {train_pos}, 负面: {train_neg})")
    print(f"  测试集: {test_pos + test_neg} (正面: {test_pos}, 负面: {test_neg})")

    print("\n✓✓ 数据集准备完成！")
    print("\n下一步:")
    print("  python src/train.py --model_type simple --epochs 2 --max_samples 1000")


if __name__ == '__main__':
    download_imdb_dataset()
