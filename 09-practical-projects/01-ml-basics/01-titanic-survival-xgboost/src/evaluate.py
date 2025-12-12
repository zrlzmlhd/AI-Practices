"""
模型评估脚本

使用方法:
    python src/evaluate.py --model_path models/tuned_model.pkl
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from data import load_titanic_data, get_feature_descriptions
from model import TitanicXGBoostClassifier


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估XGBoost Titanic模型')

    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['遇难', '生还'],
                yticklabels=['遇难', '生还'])
    plt.title('混淆矩阵', fontsize=14, pad=15)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {save_path}")

    return cm


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, pad=15)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC曲线已保存: {save_path}")


def plot_feature_importance(importance_df, save_path=None):
    """绘制特征重要性"""
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Feature Importance', fontsize=14, pad=15)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 特征重要性图已保存: {save_path}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("="*60)
    print("XGBoost Titanic生存预测 - 模型评估")
    print("="*60)
    print(f"\n模型路径: {args.model_path}")

    # 创建结果目录
    project_dir = Path(__file__).parent.parent
    result_dir = project_dir / args.result_dir
    result_dir.mkdir(exist_ok=True)

    # 加载数据
    print("\n" + "="*60)
    print("步骤1: 加载数据")
    print("="*60)

    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = load_titanic_data()
    except FileNotFoundError:
        print("\n✗ 数据文件不存在！")
        return

    # 加载模型
    print("\n" + "="*60)
    print("步骤2: 加载模型")
    print("="*60)

    classifier = TitanicXGBoostClassifier()
    classifier.load_model(args.model_path)

    # 预测
    print("\n" + "="*60)
    print("步骤3: 模型预测")
    print("="*60)

    print("\n预测测试集...")
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    # 评估指标
    print("\n" + "="*60)
    print("步骤4: 计算评估指标")
    print("="*60)

    metrics = classifier.evaluate(X_test, y_test)
    print(f"\n测试集性能:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")

    # 详细分类报告
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred,
                               target_names=['遇难', '生还'],
                               digits=4))

    # 混淆矩阵
    print("\n" + "="*60)
    print("步骤5: 绘制混淆矩阵")
    print("="*60)

    model_name = Path(args.model_path).stem
    cm_path = result_dir / f'{model_name}_confusion_matrix.png'
    cm = plot_confusion_matrix(y_test, y_pred, cm_path)

    print(f"\n混淆矩阵:")
    print(f"  真负例(TN): {cm[0, 0]} (正确预测遇难)")
    print(f"  假正例(FP): {cm[0, 1]} (错误预测生还)")
    print(f"  假负例(FN): {cm[1, 0]} (错误预测遇难)")
    print(f"  真正例(TP): {cm[1, 1]} (正确预测生还)")

    # ROC曲线
    print("\n" + "="*60)
    print("步骤6: 绘制ROC曲线")
    print("="*60)

    roc_path = result_dir / f'{model_name}_roc_curve.png'
    plot_roc_curve(y_test, y_pred_proba, roc_path)

    # 特征重要性
    print("\n" + "="*60)
    print("步骤7: 特征重要性分析")
    print("="*60)

    importance_df = classifier.get_feature_importance()
    print(f"\n特征重要性 (Top 10):")
    print(importance_df.head(10).to_string(index=False))

    # 绘制特征重要性
    importance_plot_path = result_dir / f'{model_name}_feature_importance.png'
    plot_feature_importance(importance_df.head(10), importance_plot_path)

    # 特征说明
    print(f"\n特征说明:")
    descriptions = get_feature_descriptions()
    for _, row in importance_df.head(5).iterrows():
        feature = row['feature']
        if feature in descriptions:
            print(f"  {feature}: {descriptions[feature]}")

    # 错误分析
    print("\n" + "="*60)
    print("步骤8: 错误分析")
    print("="*60)

    error_indices = np.where(y_pred != y_test)[0]
    print(f"\n错误样本数量: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")

    if len(error_indices) > 0:
        print(f"\n错误类型分布:")
        false_positives = np.sum((y_pred == 1) & (y_test == 0))
        false_negatives = np.sum((y_pred == 0) & (y_test == 1))
        print(f"  假正例 (预测生还，实际遇难): {false_positives}")
        print(f"  假负例 (预测遇难，实际生还): {false_negatives}")

    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)


if __name__ == '__main__':
    main()
