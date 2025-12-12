"""
Titanic数据集下载脚本

支持多个数据源，确保下载成功
"""
import requests
from pathlib import Path
import time


def download_titanic_data():
    """
    下载Titanic数据集

    尝试多个数据源以确保下载成功：
    1. GitHub数据科学练习数据集
    2. 备用源
    """
    print("="*60)
    print("下载Titanic数据集")
    print("="*60)

    # 多个数据源（标准Kaggle格式）
    urls = [
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv",
    ]

    data_dir = Path(__file__).parent
    train_path = data_dir / 'train.csv'

    # 尝试每个URL
    for i, url in enumerate(urls, 1):
        try:
            print(f"\n尝试数据源 {i}/{len(urls)}...")
            print(f"URL: {url}")

            # 增加超时时间
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # 保存数据
            with open(train_path, 'w', encoding='utf-8') as f:
                f.write(response.text)

            # 验证文件
            if train_path.exists() and train_path.stat().st_size > 1000:
                print(f"\n✓ 下载成功！")
                print(f"  保存路径: {train_path}")
                print(f"  文件大小: {train_path.stat().st_size} 字节")
                return True

        except Exception as e:
            print(f"✗ 数据源 {i} 失败: {e}")
            if i < len(urls):
                print("尝试下一个数据源...")
                time.sleep(1)

    # 所有源都失败
    print(f"\n✗ 所有数据源下载失败")
    print(f"\n手动下载方式:")
    print(f"  1. 访问: https://www.kaggle.com/c/titanic/data")
    print(f"  2. 下载 train.csv")
    print(f"  3. 保存到: {train_path}")
    return False


if __name__ == '__main__':
    download_titanic_data()
