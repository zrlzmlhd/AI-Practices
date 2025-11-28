"""
AI-Practices 通用工具模块
解决跨平台路径问题和提供常用功能

使用示例:
    from utils import set_seed, get_project_root, plot_training_history

    # 设置随机种子
    set_seed(42)

    # 获取项目根目录
    root = get_project_root()

    # 绘制训练历史
    plot_training_history(history.history)
"""

from utils.paths import (
    get_project_root,
    get_data_path,
    get_model_save_path,
    ensure_dir,
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    LOGS_DIR
)

from utils.common import (
    set_seed,
    get_device,
    print_separator,
    format_number,
    train_test_val_split,
    Timer
)

from utils.visualization import (
    setup_chinese_font,
    plot_training_history,
    plot_confusion_matrix,
    plot_images_grid
)

__all__ = [
    # paths
    'get_project_root',
    'get_data_path',
    'get_model_save_path',
    'ensure_dir',
    'PROJECT_ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
    # common
    'set_seed',
    'get_device',
    'print_separator',
    'format_number',
    'train_test_val_split',
    'Timer',
    # visualization
    'setup_chinese_font',
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_images_grid',
]
