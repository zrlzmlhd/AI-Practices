"""
XGBoost高级技巧评估脚本

使用方法:
    python src/evaluate.py --model_path models/xgboost_basic_model.json --features_path models/selected_features.pkl
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix,
    classification_report
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_california_housing_data, prepare_advanced_features
from src.model import AdvancedXGBoostClassifier


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估XGBoost模型')

    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--features_path', type=str, required=True,
                       help='特征名称文件路径')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def plot_roc_and_pr_curves(y_true, y_proba, save_path):
    """
    绘制ROC和PR曲线

    【ROC曲线】：真阳性率 vs 假阳性率
    【PR曲线】：精确率 vs 召回率（适合不平衡数据）
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    axes[0].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='随机猜测')
    axes[0].set_xlabel('假阳性率 (FPR)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('真阳性率 (TPR)', fontsize=12, fontweight='bold')
    axes[0].set_title('ROC曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    axes[1].plot(recall, precision, linewidth=2, label='PR曲线')
    axes[1].set_xlabel('召回率 (Recall)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('精确率 (Precision)', fontsize=12, fontweight='bold')
    axes[1].set_title('Precision-Recall曲线', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC和PR曲线已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 原始计数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['低房价', '高房价'],
                yticklabels=['低房价', '高房价'])
    axes[0].set_xlabel('预测类别', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('真实类别', fontsize=12, fontweight='bold')
    axes[0].set_title('混淆矩阵（计数）', fontsize=14, fontweight='bold')

    # 归一化
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=['低房价', '高房价'],
                yticklabels=['低房价', '高房价'])
    axes[1].set_xlabel('预测类别', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('真实类别', fontsize=12, fontweight='bold')
    axes[1].set_title('混淆矩阵（归一化）', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵已保存: {save_path}")
    plt.close()


def plot_prediction_distribution(y_true, y_proba, save_path):
    """
    绘制预测概率分布

    【是什么】：展示模型对不同类别的预测置信度
    【如何解读】：
        - 两个类别的分布分离越好，模型越好
        - 重叠区域是模型不确定的样本
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 按真实类别分组
    class_0_proba = y_proba[y_true == 0]
    class_1_proba = y_proba[y_true == 1]

    # 直方图
    axes[0].hist(class_0_proba, bins=50, alpha=0.7, label='低房价（真实）', color='blue', edgecolor='black')
    axes[0].hist(class_1_proba, bins=50, alpha=0.7, label='高房价（真实）', color='red', edgecolor='black')
    axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='决策阈值')
    axes[0].set_xlabel('预测概率', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('频数', fontsize=12, fontweight='bold')
    axes[0].set_title('预测概率分布', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 箱线图
    data_to_plot = [class_0_proba, class_1_proba]
    axes[1].boxplot(data_to_plot, labels=['低房价（真实）', '高房价（真实）'])
    axes[1].axhline(0.5, color='black', linestyle='--', linewidth=2, label='决策阈值')
    axes[1].set_ylabel('预测概率', fontsize=12, fontweight='bold')
    axes[1].set_title('预测概率箱线图', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 预测分布图已保存: {save_path}")
    plt.close()


def analyze_threshold_impact(y_true, y_proba, save_path):
    """
    分析不同阈值的影响

    【是什么】：展示阈值对精确率、召回率、F1的影响
    【为什么】：帮助选择最优阈值
    """
    thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        # 计算指标
        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)

    # 绘制
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(thresholds, precisions, label='精确率', linewidth=2)
    ax.plot(thresholds, recalls, label='召回率', linewidth=2)
    ax.plot(thresholds, f1_scores, label='F1分数', linewidth=2)
    ax.plot(thresholds, accuracies, label='准确率', linewidth=2)

    # 标记最佳F1阈值
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    ax.axvline(best_threshold, color='red', linestyle='--', linewidth=2,
               label=f'最佳F1阈值 = {best_threshold:.2f}')

    ax.set_xlabel('阈值', fontsize=12, fontweight='bold')
    ax.set_ylabel('分数', fontsize=12, fontweight='bold')
    ax.set_title('阈值对性能指标的影响', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 阈值分析图已保存: {save_path}")
    plt.close()

    return best_threshold, f1_scores[best_f1_idx]


def main():
    """主评估流程"""
    args = parse_args()

    print("="*60)
    print("XGBoost高级技巧 - 模型评估")
    print("="*60)

    # 创建结果目录
    project_dir = Path(__file__).parent.parent
    result_dir = project_dir / args.result_dir
    result_dir.mkdir(exist_ok=True)

    # ============================================
    # 1. 加载模型和特征
    # ============================================
    print("\n" + "="*60)
    print("步骤1: 加载模型和特征")
    print("="*60)

    # 加载模型
    model = AdvancedXGBoostClassifier()
    model.load_model(args.model_path)

    # 加载特征名称
    with open(args.features_path, 'rb') as f:
        selected_features = pickle.load(f)
    print(f"✓ 特征名称已加载: {len(selected_features)}个特征")

    # ============================================
    # 2. 准备测试数据
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 准备测试数据")
    print("="*60)

    # 加载数据
    X, y, _ = load_california_housing_data()
    y_binary = (y > y.median()).astype(int)

    # 特征工程（使用相同的配置）
    (X_train, y_train), (X_test, y_test), _ = \
        prepare_advanced_features(
            X, y_binary,
            task_type='classification',
            create_interactions=True,
            create_polynomials=True,
            create_statistical=True,
            feature_selection=True,
            top_k_features=len(selected_features)
        )

    print(f"✓ 测试数据已准备")
    print(f"  测试集大小: {X_test.shape}")

    # ============================================
    # 3. 模型预测
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 模型预测")
    print("="*60)

    y_pred = model.predict(X_test.values)
    y_proba = model.predict_proba(X_test.values)[:, 1]

    print(f"✓ 预测完成")

    # ============================================
    # 4. 性能评估
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 性能评估")
    print("="*60)

    # 基础指标
    accuracy = accuracy_score(y_test.values, y_pred)
    auc = roc_auc_score(y_test.values, y_proba)

    print(f"\n测试集性能:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")

    # 分类报告
    report = classification_report(
        y_test.values, y_pred,
        target_names=['低房价', '高房价'],
        output_dict=True
    )

    print(f"\n分类报告:")
    print(f"  低房价:")
    print(f"    Precision: {report['低房价']['precision']:.4f}")
    print(f"    Recall: {report['低房价']['recall']:.4f}")
    print(f"    F1-score: {report['低房价']['f1-score']:.4f}")
    print(f"  高房价:")
    print(f"    Precision: {report['高房价']['precision']:.4f}")
    print(f"    Recall: {report['高房价']['recall']:.4f}")
    print(f"    F1-score: {report['高房价']['f1-score']:.4f}")

    # ============================================
    # 5. 可视化分析
    # ============================================
    print("\n" + "="*60)
    print("步骤5: 可视化分析")
    print("="*60)

    # ROC和PR曲线
    roc_pr_path = result_dir / 'roc_pr_curves.png'
    plot_roc_and_pr_curves(y_test.values, y_proba, roc_pr_path)

    # 混淆矩阵
    cm_path = result_dir / 'confusion_matrix.png'
    plot_confusion_matrix(y_test.values, y_pred, cm_path)

    # 预测分布
    dist_path = result_dir / 'prediction_distribution.png'
    plot_prediction_distribution(y_test.values, y_proba, dist_path)

    # 阈值分析
    threshold_path = result_dir / 'threshold_analysis.png'
    best_threshold, best_f1 = analyze_threshold_impact(y_test.values, y_proba, threshold_path)

    print(f"\n最佳阈值分析:")
    print(f"  最佳阈值: {best_threshold:.2f}")
    print(f"  对应F1分数: {best_f1:.4f}")

    # ============================================
    # 6. 保存详细结果
    # ============================================
    print("\n" + "="*60)
    print("步骤6: 保存详细结果")
    print("="*60)

    detailed_results_path = result_dir / 'detailed_evaluation_results.txt'
    with open(detailed_results_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("XGBoost高级技巧 - 详细评估结果\n")
        f.write("="*60 + "\n\n")

        f.write("测试集性能:\n")
        f.write(f"  准确率: {accuracy:.4f}\n")
        f.write(f"  AUC: {auc:.4f}\n\n")

        f.write("分类报告:\n")
        f.write("="*60 + "\n")
        f.write(f"低房价:\n")
        f.write(f"  Precision: {report['低房价']['precision']:.4f}\n")
        f.write(f"  Recall: {report['低房价']['recall']:.4f}\n")
        f.write(f"  F1-score: {report['低房价']['f1-score']:.4f}\n")
        f.write(f"  Support: {report['低房价']['support']}\n\n")

        f.write(f"高房价:\n")
        f.write(f"  Precision: {report['高房价']['precision']:.4f}\n")
        f.write(f"  Recall: {report['高房价']['recall']:.4f}\n")
        f.write(f"  F1-score: {report['高房价']['f1-score']:.4f}\n")
        f.write(f"  Support: {report['高房价']['support']}\n\n")

        f.write("最佳阈值分析:\n")
        f.write(f"  最佳阈值: {best_threshold:.2f}\n")
        f.write(f"  对应F1分数: {best_f1:.4f}\n")

    print(f"✓ 详细结果已保存: {detailed_results_path}")

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("评估总结")
    print("="*60)
    print(f"✓ 测试集准确率: {accuracy:.4f}")
    print(f"✓ 测试集AUC: {auc:.4f}")
    print(f"✓ 详细结果已保存: {detailed_results_path}")
    print(f"✓ 可视化结果已保存在: {result_dir}")


if __name__ == '__main__':
    main()
