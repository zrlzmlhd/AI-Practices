"""
数据下载脚本

从Kaggle下载Titanic数据集
"""

import os
import sys
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.paths import ensure_dir


def download_from_kaggle():
    """
    使用Kaggle API下载数据

    需要先配置Kaggle API:
    1. 在 https://www.kaggle.com/account 创建API token
    2. 下载 kaggle.json 到 ~/.kaggle/
    3. chmod 600 ~/.kaggle/kaggle.json
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        print("正在使用Kaggle API下载数据...")
        api = KaggleApi()
        api.authenticate()

        # 下载数据集
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        ensure_dir(data_dir)

        api.competition_download_files(
            'titanic',
            path=str(data_dir),
            quiet=False
        )

        # 解压文件
        zip_file = data_dir / 'titanic.zip'
        if zip_file.exists():
            print("正在解压文件...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            zip_file.unlink()  # 删除zip文件

        print(f"✓ 数据已下载到: {data_dir}")
        return True

    except Exception as e:
        print(f"Kaggle API下载失败: {e}")
        return False


def download_from_url():
    """
    从备用URL下载数据（如果Kaggle API不可用）
    """
    print("正在从备用源下载数据...")

    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    ensure_dir(data_dir)

    # 备用下载链接
    urls = {
        'train.csv': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
    }

    for filename, url in urls.items():
        filepath = data_dir / filename

        if filepath.exists():
            print(f"✓ {filename} 已存在")
            continue

        try:
            print(f"正在下载 {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            print(f"✓ {filename} 下载完成")

        except Exception as e:
            print(f"✗ {filename} 下载失败: {e}")
            return False

    return True


def create_sample_data():
    """
    创建示例数据（用于演示）
    """
    import pandas as pd
    import numpy as np

    print("正在创建示例数据...")

    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    ensure_dir(data_dir)

    # 创建示例训练数据
    np.random.seed(42)
    n_samples = 891

    train_data = pd.DataFrame({
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.normal(30, 14, n_samples).clip(0.42, 80),
        'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.68, 0.23, 0.05, 0.02, 0.01, 0.01]),
        'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.01, 0.01, 0.005, 0.005]),
        'Ticket': [f'TICKET_{i}' for i in range(1, n_samples + 1)],
        'Fare': np.random.lognormal(3, 1, n_samples).clip(0, 512),
        'Cabin': [f'C{i}' if np.random.random() > 0.77 else '' for i in range(1, n_samples + 1)],
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
    })

    # 创建示例测试数据
    n_test = 418
    test_data = pd.DataFrame({
        'PassengerId': range(n_samples + 1, n_samples + n_test + 1),
        'Pclass': np.random.choice([1, 2, 3], n_test, p=[0.24, 0.21, 0.55]),
        'Name': [f'Passenger_{i}' for i in range(n_samples + 1, n_samples + n_test + 1)],
        'Sex': np.random.choice(['male', 'female'], n_test, p=[0.65, 0.35]),
        'Age': np.random.normal(30, 14, n_test).clip(0.42, 80),
        'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n_test, p=[0.68, 0.23, 0.05, 0.02, 0.01, 0.01]),
        'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_test, p=[0.76, 0.13, 0.08, 0.01, 0.01, 0.005, 0.005]),
        'Ticket': [f'TICKET_{i}' for i in range(n_samples + 1, n_samples + n_test + 1)],
        'Fare': np.random.lognormal(3, 1, n_test).clip(0, 512),
        'Cabin': [f'C{i}' if np.random.random() > 0.77 else '' for i in range(n_samples + 1, n_samples + n_test + 1)],
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_test, p=[0.19, 0.09, 0.72])
    })

    # 保存数据
    train_data.to_csv(data_dir / 'train.csv', index=False)
    test_data.to_csv(data_dir / 'test.csv', index=False)

    print(f"✓ 示例数据已创建: {data_dir}")
    print(f"  - train.csv: {len(train_data)} 条记录")
    print(f"  - test.csv: {len(test_data)} 条记录")

    return True


def main():
    """主函数"""
    print("=" * 60)
    print("Titanic数据集下载工具")
    print("=" * 60)

    # 尝试从Kaggle下载
    if download_from_kaggle():
        print("\n✓ 数据下载成功！")
        return

    # 尝试从备用URL下载
    print("\n尝试从备用源下载...")
    if download_from_url():
        print("\n✓ 数据下载成功！")
        return

    # 创建示例数据
    print("\n创建示例数据用于演示...")
    if create_sample_data():
        print("\n✓ 示例数据创建成功！")
        print("\n注意: 这是示例数据，不是真实的Titanic数据集")
        print("要获取真实数据，请配置Kaggle API或手动下载")
        return

    print("\n✗ 数据准备失败")
    print("\n手动下载步骤:")
    print("1. 访问 https://www.kaggle.com/c/titanic/data")
    print("2. 下载 train.csv 和 test.csv")
    print("3. 将文件放到 data/raw/ 目录")


if __name__ == '__main__':
    main()
