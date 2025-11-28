"""
可视化工具模块 - 提供常用的绘图函数

使用方法:
    from utils.visualization import plot_training_history, setup_chinese_font
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Any


def setup_chinese_font():
    """
    配置matplotlib支持中文显示
    """
    import platform
    system = platform.system()

    if system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']

    plt.rcParams['axes.unicode_minus'] = False


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    figsize: tuple = (12, 4),
    title_prefix: str = ''
) -> None:
    """
    绘制训练历史曲线

    Args:
        history: 训练历史字典，通常来自 model.fit().history
        metrics: 要绘制的指标列表，默认自动检测
        figsize: 图形大小
        title_prefix: 标题前缀

    Example:
        >>> history = model.fit(X, y, validation_data=(X_val, y_val))
        >>> plot_training_history(history.history)
    """
    setup_chinese_font()

    # 如果传入的是 History 对象，获取其 history 属性
    if hasattr(history, 'history'):
        history = history.history

    # 自动检测指标
    if metrics is None:
        metrics = [k for k in history.keys() if not k.startswith('val_')]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    epochs = range(1, len(history[metrics[0]]) + 1)

    for ax, metric in zip(axes, metrics):
        # 训练指标
        ax.plot(epochs, history[metric], 'b-o', label=f'Training {metric}',
                markersize=4, linewidth=1.5)

        # 验证指标
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(epochs, history[val_metric], 'r-s', label=f'Validation {metric}',
                    markersize=4, linewidth=1.5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{title_prefix}{metric.capitalize()}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
    normalize: bool = False,
    figsize: tuple = (8, 6),
    cmap: str = 'Blues'
) -> None:
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称列表
        normalize: 是否归一化
        figsize: 图形大小
        cmap: 颜色映射
    """
    from sklearn.metrics import confusion_matrix

    setup_chinese_font()

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if classes is None:
        classes = [str(i) for i in range(len(cm))]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.show()


def plot_images_grid(
    images: np.ndarray,
    labels: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None,
    n_cols: int = 5,
    figsize: tuple = (15, 6),
    class_names: Optional[List[str]] = None
) -> None:
    """
    以网格形式显示多张图像

    Args:
        images: 图像数组 (N, H, W) 或 (N, H, W, C)
        labels: 真实标签
        predictions: 预测标签
        n_cols: 每行显示的图像数
        figsize: 图形大小
        class_names: 类别名称列表
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n_images:
            img = images[i]

            # 处理灰度图
            if len(img.shape) == 2 or img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)

            # 设置标题
            title_parts = []
            if labels is not None:
                label = class_names[labels[i]] if class_names else labels[i]
                title_parts.append(f'True: {label}')
            if predictions is not None:
                pred = class_names[predictions[i]] if class_names else predictions[i]
                title_parts.append(f'Pred: {pred}')

            if title_parts:
                ax.set_title('\n'.join(title_parts), fontsize=8)

        ax.axis('off')

    plt.tight_layout()
    plt.show()
