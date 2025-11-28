"""
路径工具模块 - 解决跨平台路径兼容性问题

使用方法:
    from utils.paths import get_project_root, get_data_path

    # 获取项目根目录
    root = get_project_root()

    # 获取数据目录路径
    data_dir = get_data_path('猫狗数据集/dataset/train')
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """
    获取项目根目录路径

    Returns:
        Path: 项目根目录的Path对象
    """
    # 从当前文件向上查找，直到找到包含 requirements.txt 的目录
    current = Path(__file__).resolve()

    for parent in [current] + list(current.parents):
        if (parent / 'requirements.txt').exists():
            return parent

    # 如果找不到，返回当前工作目录
    return Path.cwd()


def get_data_path(*paths: str) -> Path:
    """
    获取数据目录路径

    Args:
        *paths: 相对于项目根目录的路径片段

    Returns:
        Path: 完整的数据路径

    Example:
        >>> get_data_path('猫狗数据集', 'dataset', 'train')
        PosixPath('/path/to/project/猫狗数据集/dataset/train')
    """
    root = get_project_root()
    return root.joinpath(*paths)


def get_model_save_path(model_name: str, create_dir: bool = True) -> Path:
    """
    获取模型保存路径

    Args:
        model_name: 模型文件名
        create_dir: 是否自动创建目录

    Returns:
        Path: 模型保存的完整路径
    """
    models_dir = get_project_root() / 'saved_models'

    if create_dir and not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)

    return models_dir / model_name


def ensure_dir(path: Path) -> Path:
    """
    确保目录存在，如果不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path: 同一路径对象
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


# 常用目录常量
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'saved_models'
LOGS_DIR = PROJECT_ROOT / 'logs'
