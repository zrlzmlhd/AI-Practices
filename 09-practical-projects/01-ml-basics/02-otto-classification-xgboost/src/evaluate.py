"""
模型评估脚本

使用方法:
    python src/evaluate.py --model_path models/xgboost_tuned_model.pkl --processor_path models/xgboost_tuned_processor.pkl
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import prepare_otto_data


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估Otto分类模型')

    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--processor_path', type=str, required=True,
                       help='数据处理器文件路径')
    parser.add_argument('--data_path', type=str, default='data/train.csv',
                       help='数据文件路径')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {save_path}")

    return cm


def plot_class_performance(y_true, y_pred, class_names, save_path=None):
    """绘制各类别性能"""
    from sklearn.metrics import precision_score, recall_score, f1_score

    precisions = []
    recalls = []
    f1s = []

    for i in range(len(class_names)):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)

        precisions.append(precision_score(y_true_binary, y_pred_binary, zero_division=0))
        recalls.append(recall_score(y_true_binary, y_pred_binary, zero_division=0))
        f1s.append(f1_score(y_true_binary, y_pred_binary, zero_division=0))

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance by Class', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 类别性能图已保存: {save_path}")


def plot_probability_distribution(y_proba, y_true, save_path=None):
    """绘制预测概率分布"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]

        # 正确预测的概率
        correct_mask = (y_true == i)
        correct_probs = y_proba[correct_mask, i]

        # 错误预测的概率
        incorrect_mask = (y_true != i)
        incorrect_probs = y_proba[incorrect_mask, i]

        ax.hist(correct_probs, bins=30, alpha=0.6, label='Correct', color='green')
        ax.hist(incorrect_probs, bins=30, alpha=0.6, label='Incorrect', color='red')

        ax.set_xlabel('Probability', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'Class {i+1}', fontsize=11)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 概率分布图已保存: {save_path}")


def analyze_errors(X_test, y_test, y_pred, y_proba, processor, num_examples=10):
    """分析错误样本"""
    print("\n" + "="*60)
    print("错误样本分析")
    print("="*60)

    # 找出错误样本
    error_indices = np.where(y_pred != y_test)[0]
    print(f"\n错误样本数量: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")

    if len(error_indices) == 0:
        print("没有错误样本！")
        return

    # 按置信度排序错误样本
    error_confidence = np.max(y_proba[error_indices], axis=1)
    sorted_indices = error_indices[np.argsort(-error_confidence)]

    # 显示最自信的错误样本
    print(f"\n最自信的 {min(num_examples, len(sorted_indices))} 个错误样本:")
    for i, idx in enumerate(sorted_indices[:num_examples]):
        true_label = processor.inverse_transform_labels([y_test[idx]])[0]
        pred_label = processor.inverse_transform_labels([y_pred[idx]])[0]
        confidence = y_proba[idx].max()
        pred_class = y_proba[idx].argmax()

        print(f"\n错误样本 {i+1}:")
        print(f"  真实标签: {true_label}")
        print(f"  预测标签: {pred_label}")
        print(f"  预测置信度: {confidence:.4f}")
        print(f"  预测概率分布: {y_proba[idx]}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("="*60)
    print("Otto分类 - 模型评估")
    print("="*60)
    print(f"\n模型路径: {args.model_path}")
    print(f"数据处理器路径: {args.processor_path}")

    # 创建结果目录
    project_dir = Path(__file__).parent.parent
    result_dir = project_dir / args.result_dir
    result_dir.mkdir(exist_ok=True)

    # ============================================
    # 步骤1: 加载数据处理器
    # ============================================
    print("\n" + "="*60)
    print("步骤1: 加载数据处理器")
    print("="*60)

    with open(args.processor_path, 'rb') as f:
        processor_data = pickle.load(f)

    print(f"✓ 数据处理器已加载")

    # ============================================
    # 步骤2: 准备数据
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 准备数据")
    print("="*60)

    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), processor = prepare_otto_data(
            data_path=args.data_path
        )
    except FileNotFoundError as e:
        print(f"\n✗ 数据文件不存在: {e}")
        return

    # ============================================
    # 步骤3: 加载模型
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 加载模型")
    print("="*60)

    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ 模型已加载")

    # ============================================
    # 步骤4: 预测
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 模型预测")
    print("="*60)

    print("\n预测测试集...")

    # 根据模型类型选择预测方法
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
    else:
        # LightGBM模型
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

    # ============================================
    # 步骤5: 计算评估指标
    # ============================================
    print("\n" + "="*60)
    print("步骤5: 计算评估指标")
    print("="*60)

    from sklearn.metrics import accuracy_score, log_loss

    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"\n测试集性能:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Log Loss: {logloss:.4f}")

    # 详细分类报告
    class_names = processor.label_encoder.classes_
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred,
                               target_names=class_names,
                               digits=4))

    # ============================================
    # 步骤6: 绘制混淆矩阵
    # ============================================
    print("\n" + "="*60)
    print("步骤6: 绘制混淆矩阵")
    print("="*60)

    model_name = Path(args.model_path).stem
    cm_path = result_dir / f'{model_name}_confusion_matrix.png'
    cm = plot_confusion_matrix(y_test, y_pred, class_names, cm_path)

    # ============================================
    # 步骤7: 绘制类别性能
    # ============================================
    print("\n" + "="*60)
    print("步骤7: 绘制类别性能")
    print("="*60)

    perf_path = result_dir / f'{model_name}_class_performance.png'
    plot_class_performance(y_test, y_pred, class_names, perf_path)

    # ============================================
    # 步骤8: 绘制概率分布
    # ============================================
    print("\n" + "="*60)
    print("步骤8: 绘制概率分布")
    print("="*60)

    prob_path = result_dir / f'{model_name}_probability_distribution.png'
    plot_probability_distribution(y_pred_proba, y_test, prob_path)

    # ============================================
    # 步骤9: 错误分析
    # ============================================
    analyze_errors(X_test, y_test, y_pred, y_pred_proba, processor, num_examples=10)

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)

    print(f"\n生成的文件:")
    print(f"  1. 混淆矩阵: {cm_path}")
    print(f"  2. 类别性能图: {perf_path}")
    print(f"  3. 概率分布图: {prob_path}")

    print(f"\n模型性能总结:")
    print(f"  准确率: {accuracy:.2%}")
    print(f"  Log Loss: {logloss:.4f}")

    if logloss < 0.5:
        print(f"\n  ✓✓ 模型性能优秀！")
    elif logloss < 0.6:
        print(f"\n  ✓ 模型性能良好")
    else:
        print(f"\n  ⚠ 模型性能有待提升")


if __name__ == '__main__':
    main()
