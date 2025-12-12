"""
SVM文本分类评估脚本

使用方法:
    python src/evaluate.py --model_path models/svm_linear_model.pkl --extractor_path models/tfidf_extractor.pkl
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_20newsgroups_data, TextPreprocessor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估SVM文本分类模型')

    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--extractor_path', type=str, required=True,
                       help='特征提取器路径')
    parser.add_argument('--labels_path', type=str, default='models/label_names.pkl',
                       help='标签名称路径')
    parser.add_argument('--test_data', type=str, default='20newsgroups',
                       help='测试数据源')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                       help='类别列表')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def plot_roc_curves(y_test, y_proba, label_names, save_path):
    """
    绘制ROC曲线

    【是什么】：展示不同阈值下的真阳性率和假阳性率
    【如何解读】：
        - 曲线越靠近左上角，模型越好
        - AUC（曲线下面积）越大越好
    """
    n_classes = len(label_names)

    # 二值化标签
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # 计算每个类别的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

    for i, color in zip(range(min(n_classes, 10)), colors[:10]):  # 最多显示10个类别
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{label_names[i]} (AUC = {roc_auc[i]:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率 (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('真阳性率 (TPR)', fontsize=12, fontweight='bold')
    ax.set_title('ROC曲线', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC曲线已保存: {save_path}")
    plt.close()

    return roc_auc


def analyze_misclassifications(X_test_text, y_test, y_pred, label_names, save_path, top_n=10):
    """
    分析错误分类样本

    【是什么】：找出最容易混淆的类别对
    【为什么】：帮助理解模型的弱点
    """
    # 统计混淆对
    confusion_pairs = {}

    for true_label, pred_label, text in zip(y_test, y_pred, X_test_text):
        if true_label != pred_label:
            pair = (label_names[true_label], label_names[pred_label])
            if pair not in confusion_pairs:
                confusion_pairs[pair] = []
            confusion_pairs[pair].append(text)

    # 排序
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: len(x[1]), reverse=True)

    # 保存分析结果
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("错误分类分析\n")
        f.write("="*60 + "\n\n")

        f.write(f"总错误数: {len(y_test) - np.sum(y_test == y_pred)}\n")
        f.write(f"错误率: {1 - np.mean(y_test == y_pred):.4f}\n\n")

        f.write("最常见的混淆对:\n")
        f.write("="*60 + "\n")

        for i, ((true_label, pred_label), examples) in enumerate(sorted_pairs[:top_n]):
            f.write(f"\n{i+1}. {true_label} → {pred_label} ({len(examples)}个样本)\n")
            f.write("-"*60 + "\n")

            # 显示前3个样本
            for j, text in enumerate(examples[:3]):
                f.write(f"\n样本 {j+1}:\n")
                f.write(f"{text[:200]}...\n")

    print(f"✓ 错误分类分析已保存: {save_path}")


def plot_prediction_confidence(y_proba, y_test, y_pred, save_path):
    """
    绘制预测置信度分布

    【是什么】：展示模型预测的置信度
    【如何解读】：
        - 正确预测应该有高置信度
        - 错误预测的置信度分布反映模型的不确定性
    """
    # 获取最大概率（置信度）
    confidence = np.max(y_proba, axis=1)

    # 区分正确和错误预测
    correct_mask = (y_test == y_pred)
    correct_confidence = confidence[correct_mask]
    incorrect_confidence = confidence[~correct_mask]

    # 绘制
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 直方图
    axes[0].hist(correct_confidence, bins=30, alpha=0.7, label='正确预测', color='green', edgecolor='black')
    axes[0].hist(incorrect_confidence, bins=30, alpha=0.7, label='错误预测', color='red', edgecolor='black')
    axes[0].set_xlabel('预测置信度', fontsize=12)
    axes[0].set_ylabel('频数', fontsize=12)
    axes[0].set_title('预测置信度分布', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 箱线图
    data_to_plot = [correct_confidence, incorrect_confidence]
    axes[1].boxplot(data_to_plot, labels=['正确预测', '错误预测'])
    axes[1].set_ylabel('预测置信度', fontsize=12)
    axes[1].set_title('预测置信度箱线图', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 置信度分析已保存: {save_path}")
    plt.close()

    return {
        'correct_mean': np.mean(correct_confidence),
        'correct_std': np.std(correct_confidence),
        'incorrect_mean': np.mean(incorrect_confidence),
        'incorrect_std': np.std(incorrect_confidence)
    }


def main():
    """主评估流程"""
    args = parse_args()

    print("="*60)
    print("SVM文本分类 - 模型评估")
    print("="*60)

    # 创建结果目录
    project_dir = Path(__file__).parent.parent
    result_dir = project_dir / args.result_dir
    result_dir.mkdir(exist_ok=True)

    # ============================================
    # 1. 加载模型和特征提取器
    # ============================================
    print("\n" + "="*60)
    print("步骤1: 加载模型和特征提取器")
    print("="*60)

    # 加载模型
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ 模型已加载: {args.model_path}")

    # 加载特征提取器
    with open(args.extractor_path, 'rb') as f:
        feature_extractor = pickle.load(f)
    print(f"✓ 特征提取器已加载: {args.extractor_path}")

    # 加载标签名称
    labels_path = project_dir / args.labels_path
    with open(labels_path, 'rb') as f:
        label_names = pickle.load(f)
    print(f"✓ 标签名称已加载: {labels_path}")
    print(f"  类别数: {len(label_names)}")

    # ============================================
    # 2. 加载测试数据
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 加载测试数据")
    print("="*60)

    if args.test_data == '20newsgroups':
        texts_test, y_test, _ = load_20newsgroups_data(
            subset='test',
            categories=args.categories
        )

        # 预处理
        preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        texts_test_processed = preprocessor.preprocess_batch(texts_test, verbose=True)

        # 提取特征
        X_test = feature_extractor.transform(texts_test_processed)

        print(f"✓ 测试数据已加载")
        print(f"  测试样本数: {len(texts_test)}")

    else:
        print(f"✗ 不支持的测试数据源: {args.test_data}")
        return

    # ============================================
    # 3. 预测
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 模型预测")
    print("="*60)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"✓ 预测完成")

    # ============================================
    # 4. 评估
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 模型评估")
    print("="*60)

    # 准确率
    accuracy = np.mean(y_test == y_pred)
    print(f"\n测试集准确率: {accuracy:.4f}")

    # 分类报告
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)

    print("\n分类报告:")
    print("="*60)
    for label in label_names:
        if label in report:
            print(f"\n{label}:")
            print(f"  Precision: {report[label]['precision']:.4f}")
            print(f"  Recall: {report[label]['recall']:.4f}")
            print(f"  F1-score: {report[label]['f1-score']:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # ============================================
    # 5. 可视化分析
    # ============================================
    print("\n" + "="*60)
    print("步骤5: 可视化分析")
    print("="*60)

    # ROC曲线
    roc_path = result_dir / 'roc_curves.png'
    roc_auc = plot_roc_curves(y_test, y_proba, label_names, roc_path)

    # 置信度分析
    confidence_path = result_dir / 'prediction_confidence.png'
    confidence_stats = plot_prediction_confidence(y_proba, y_test, y_pred, confidence_path)

    # 错误分类分析
    misclass_path = result_dir / 'misclassification_analysis.txt'
    analyze_misclassifications(texts_test, y_test, y_pred, label_names, misclass_path)

    # ============================================
    # 6. 保存详细结果
    # ============================================
    print("\n" + "="*60)
    print("步骤6: 保存详细结果")
    print("="*60)

    detailed_results_path = result_dir / 'detailed_evaluation_results.txt'
    with open(detailed_results_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("SVM文本分类 - 详细评估结果\n")
        f.write("="*60 + "\n\n")

        f.write(f"测试集准确率: {accuracy:.4f}\n\n")

        f.write("分类报告:\n")
        f.write("="*60 + "\n")
        for label in label_names:
            if label in report:
                f.write(f"\n{label}:\n")
                f.write(f"  Precision: {report[label]['precision']:.4f}\n")
                f.write(f"  Recall: {report[label]['recall']:.4f}\n")
                f.write(f"  F1-score: {report[label]['f1-score']:.4f}\n")
                f.write(f"  Support: {report[label]['support']}\n")

        f.write(f"\n宏平均:\n")
        f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"  Recall: {report['macro avg']['recall']:.4f}\n")
        f.write(f"  F1-score: {report['macro avg']['f1-score']:.4f}\n")

        f.write("\n\nROC AUC分数:\n")
        f.write("="*60 + "\n")
        for i, label in enumerate(label_names[:10]):  # 最多显示10个
            if i in roc_auc:
                f.write(f"  {label}: {roc_auc[i]:.4f}\n")

        f.write("\n\n置信度统计:\n")
        f.write("="*60 + "\n")
        f.write(f"正确预测平均置信度: {confidence_stats['correct_mean']:.4f} ± {confidence_stats['correct_std']:.4f}\n")
        f.write(f"错误预测平均置信度: {confidence_stats['incorrect_mean']:.4f} ± {confidence_stats['incorrect_std']:.4f}\n")

    print(f"✓ 详细结果已保存: {detailed_results_path}")

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("评估总结")
    print("="*60)
    print(f"✓ 测试集准确率: {accuracy:.4f}")
    print(f"✓ 宏平均F1: {report['macro avg']['f1-score']:.4f}")
    print(f"✓ 详细结果已保存: {detailed_results_path}")
    print(f"✓ ROC曲线已保存: {roc_path}")
    print(f"✓ 置信度分析已保存: {confidence_path}")
    print(f"✓ 错误分类分析已保存: {misclass_path}")


if __name__ == '__main__':
    main()
