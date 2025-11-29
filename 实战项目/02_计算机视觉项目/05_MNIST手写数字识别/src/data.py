"""
MNIST数据加载和预处理
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed


def load_mnist_data(normalize=True):
    """
    加载MNIST数据集

    Args:
        normalize: 是否归一化

    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    print("正在加载MNIST数据集...")

    # 从Keras加载数据
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    print(f"✓ 训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"✓ 测试集: {X_test.shape}, 标签: {y_test.shape}")

    # 添加通道维度
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # 归一化到[0, 1]
    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        print("✓ 数据已归一化到[0, 1]")

    return (X_train, y_train), (X_test, y_test)


def prepare_data(test_size=0.1, random_state=42):
    """
    准备训练、验证和测试数据

    Args:
        test_size: 验证集比例
        random_state: 随机种子

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    set_seed(random_state)

    # 加载数据
    (X_train, y_train), (X_test, y_test) = load_mnist_data()

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train
    )

    print(f"\n数据划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_data_augmentation():
    """
    创建数据增强层

    Returns:
        数据增强Sequential模型
    """
    data_augmentation = keras.Sequential([
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomZoom(0.1),
    ], name='data_augmentation')

    return data_augmentation


def get_class_distribution(y):
    """
    获取类别分布

    Args:
        y: 标签数组

    Returns:
        dict: 类别分布
    """
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))

    print("\n类别分布:")
    total = len(y)
    for label, count in sorted(distribution.items()):
        percentage = count / total * 100
        print(f"  {label}: {count:5d} ({percentage:5.2f}%)")

    return distribution


def print_data_info(X, y, name='数据集'):
    """
    打印数据集信息

    Args:
        X: 特征数据
        y: 标签数据
        name: 数据集名称
    """
    print(f"\n{'=' * 60}")
    print(f"{name}信息")
    print(f"{'=' * 60}")
    print(f"形状: {X.shape}")
    print(f"数据类型: {X.dtype}")
    print(f"值范围: [{X.min():.3f}, {X.max():.3f}]")
    print(f"标签形状: {y.shape}")
    print(f"类别数: {len(np.unique(y))}")

    get_class_distribution(y)


if __name__ == '__main__':
    print("=" * 60)
    print("MNIST数据加载测试")
    print("=" * 60)

    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

    # 打印信息
    print_data_info(X_train, y_train, '训练集')
    print_data_info(X_val, y_val, '验证集')
    print_data_info(X_test, y_test, '测试集')

    print("\n✓ 数据加载测试完成！")
