"""
通用工具模块 - 提供常用的辅助函数

使用方法:
    from utils.common import set_seed, get_device, print_model_summary
"""

import os
import random
import numpy as np
from typing import Optional, Tuple


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保结果可重复

    Args:
        seed: 随机种子值

    Example:
        >>> set_seed(42)
        >>> # 现在所有随机操作都是可重复的
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 设置 TensorFlow 随机种子
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # 设置确定性操作
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except ImportError:
        pass

    # 设置 PyTorch 随机种子
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"Random seed set to {seed}")


def get_device() -> str:
    """
    获取可用的计算设备

    Returns:
        str: 'cuda', 'mps' 或 'cpu'
    """
    # PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass

    # TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return 'gpu'
    except ImportError:
        pass

    return 'cpu'


def print_separator(title: str = '', char: str = '=', length: int = 60) -> None:
    """
    打印分隔符

    Args:
        title: 标题文本
        char: 分隔字符
        length: 分隔符长度
    """
    if title:
        padding = (length - len(title) - 2) // 2
        print(f"{char * padding} {title} {char * padding}")
    else:
        print(char * length)


def format_number(num: float, precision: int = 4) -> str:
    """
    格式化数字显示

    Args:
        num: 要格式化的数字
        precision: 小数位数

    Returns:
        str: 格式化后的字符串
    """
    if abs(num) >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def train_test_val_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将数据集划分为训练集、验证集和测试集

    Args:
        X: 特征数据
        y: 标签数据
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练集）
        random_state: 随机种子

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split

    # 首先分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 然后从剩余数据中分出验证集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    print(f"Dataset split:")
    print(f"  Training:   {len(X_train):>6} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):>6} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test):>6} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


class Timer:
    """
    计时器上下文管理器

    Example:
        >>> with Timer("Training"):
        ...     model.fit(X, y)
        Training completed in 12.34 seconds
    """

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        import time
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.name} completed in {elapsed:.2f} seconds")

    @property
    def elapsed(self) -> float:
        """返回已用时间（秒）"""
        import time
        if self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0
