"""
模型评估脚本
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.visualization import setup_chinese_font, plot_confusion_matrix
from data import prepare_data
from model import TitanicPredictor


def evaluate_model(model_path, X_val, y_val, save_dir=None):
    """
    评估模型

    Args:
        model_path: 模型路径
        X_val: 验证特征
        y_val: 验证标签
        save_dir: 结果保存目录
    """
    print("=" * 60)
    print("模型评估")
    print("=" * 60)

    # 加载模型
    predictor = TitanicPredictor()
    predictor.load_model(model_path)

    # 预测
    y_pred = predictor.predict(X_val)
    y_pred_proba = predictor.predict_proba(X_val)[:, 1]

    # 计算评估指标
    metrics = {
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1-Score': f1_score(y_val, y_pred)
    }

    print("\n评估指标:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric:12s}: {value:.4f}")

    # 分类报告
    print("\n分类报告:")
    print("-" * 40)
    print(classification_report(
        y_val, y_pred,
        target_names=['未幸存', '幸存']
    ))

    # 混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    print("\n混淆矩阵:")
    print("-" * 40)
    print(f"              预测未幸存  预测幸存")
    print(f"实际未幸存      {cm[0, 0]:6d}    {cm[0, 1]:6d}")
    print(f"实际幸存        {cm[1, 0]:6d}    {cm[1, 1]:6d}")

    # 保存结果
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存评估指标
        metrics_df = pd.DataFrame([metrics])
        metrics_path = save_dir / 'evaluation_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n✓ 评估指标已保存: {metrics_path}")

        # 绘制并保存混淆矩阵
        plot_confusion_matrix_custom(cm, save_dir / 'confusion_matrix.png')

        # 绘制ROC曲线
        plot_roc_curve(y_val, y_pred_proba, save_dir / 'roc_curve.png')

        # 绘制PR曲线
        plot_pr_curve(y_val, y_pred_proba, save_dir / 'pr_curve.png')

        # 特征重要性
        importance = predictor.get_feature_importance(top_n=15)
        if importance is not None:
            plot_feature_importance(importance, save_dir / 'feature_importance.png')

    return metrics, y_pred, y_pred_proba


def plot_confusion_matrix_custom(cm, save_path):
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        save_path: 保存路径
    """
    setup_chinese_font()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['未幸存', '幸存'],
        yticklabels=['未幸存', '幸存']
    )
    plt.title('混淆矩阵', fontsize=16, pad=20)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 混淆矩阵已保存: {save_path}")


def plot_roc_curve(y_true, y_pred_proba, save_path):
    """
    绘制ROC曲线

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        save_path: 保存路径
    """
    setup_chinese_font()

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title('ROC曲线', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ ROC曲线已保存: {save_path}")


def plot_pr_curve(y_true, y_pred_proba, save_path):
    """
    绘制Precision-Recall曲线

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        save_path: 保存路径
    """
    setup_chinese_font()

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR曲线 (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('Precision-Recall曲线', fontsize=16, pad=20)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ PR曲线已保存: {save_path}")


def plot_feature_importance(importance_df, save_path):
    """
    绘制特征重要性

    Args:
        importance_df: 特征重要性DataFrame
        save_path: 保存路径
    """
    setup_chinese_font()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('重要性', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.title('特征重要性', fontsize=16, pad=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 特征重要性图已保存: {save_path}")


def analyze_predictions(y_true, y_pred, y_pred_proba, X_val):
    """
    分析预测结果

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率
        X_val: 验证特征
    """
    print("\n" + "=" * 60)
    print("预测分析")
    print("=" * 60)

    # 正确和错误预测
    correct = (y_true == y_pred)
    incorrect = ~correct

    print(f"\n正确预测: {correct.sum()} ({correct.sum() / len(y_true):.2%})")
    print(f"错误预测: {incorrect.sum()} ({incorrect.sum() / len(y_true):.2%})")

    # 假阳性和假阴性
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    print(f"\n假阳性 (FP): {fp}")
    print(f"假阴性 (FN): {fn}")

    # 预测概率分布
    print(f"\n预测概率统计:")
    print(f"  最小值: {y_pred_proba.min():.4f}")
    print(f"  最大值: {y_pred_proba.max():.4f}")
    print(f"  平均值: {y_pred_proba.mean():.4f}")
    print(f"  中位数: {np.median(y_pred_proba):.4f}")

    # 高置信度预测
    high_conf = (y_pred_proba > 0.9) | (y_pred_proba < 0.1)
    print(f"\n高置信度预测 (>0.9 或 <0.1): {high_conf.sum()} ({high_conf.sum() / len(y_pred_proba):.2%})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Titanic生存预测模型评估')

    parser.add_argument(
        '--model',
        type=str,
        default='best_model.pkl',
        help='模型文件名'
    )

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='进行详细的预测分析'
    )

    args = parser.parse_args()

    # 准备数据
    _, X_val, _, y_val, _ = prepare_data()

    # 模型路径
    model_dir = Path(__file__).parent.parent / 'models'
    model_path = model_dir / args.model

    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行 python src/train.py 训练模型")
        return

    # 结果保存目录
    results_dir = Path(__file__).parent.parent / 'results'

    # 评估模型
    metrics, y_pred, y_pred_proba = evaluate_model(
        model_path, X_val, y_val, save_dir=results_dir
    )

    # 详细分析
    if args.analyze:
        analyze_predictions(y_val, y_pred, y_pred_proba, X_val)

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    print(f"结果已保存到: {results_dir}")


if __name__ == '__main__':
    main()
