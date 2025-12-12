"""
SVM文本分类训练脚本

使用方法:
    python src/train.py --kernel linear --feature_method tfidf
    python src/train.py --kernel rbf --tune_hyperparams
    python src/train.py --ensemble --categories alt.atheism comp.graphics

【训练模式】:
- 单模型训练
- 超参数调优
- 集成模型训练
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import prepare_text_classification_data
from src.model import SVMTextClassifier, SVMHyperparameterTuner, SVMEnsembleClassifier


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练SVM文本分类模型')

    # 数据参数
    parser.add_argument('--data_source', type=str, default='20newsgroups',
                       help='数据源')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                       help='类别列表（None表示全部）')
    parser.add_argument('--feature_method', type=str, default='tfidf',
                       choices=['tfidf', 'count', 'word2vec'],
                       help='特征提取方法')
    parser.add_argument('--max_features', type=int, default=5000,
                       help='最大特征数')
    parser.add_argument('--preprocess', action='store_true', default=True,
                       help='是否预处理文本')

    # 模型参数
    parser.add_argument('--kernel', type=str, default='linear',
                       choices=['linear', 'rbf', 'poly'],
                       help='SVM核函数')
    parser.add_argument('--C', type=float, default=1.0,
                       help='正则化参数')
    parser.add_argument('--gamma', type=str, default='scale',
                       help='RBF核参数')

    # 训练模式
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='是否进行超参数调优')
    parser.add_argument('--search_method', type=str, default='grid',
                       choices=['grid', 'random'],
                       help='超参数搜索方法')
    parser.add_argument('--ensemble', action='store_true',
                       help='是否使用集成模型')

    # 保存路径
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def plot_confusion_matrix(cm, label_names, save_path):
    """
    绘制混淆矩阵

    【是什么】：展示分类结果的混淆情况
    【如何解读】：
        - 对角线：正确分类的样本
        - 非对角线：错误分类的样本
    """
    plt.figure(figsize=(12, 10))

    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 绘制热力图
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': '比例'}
    )

    plt.xlabel('预测类别', fontsize=12, fontweight='bold')
    plt.ylabel('真实类别', fontsize=12, fontweight='bold')
    plt.title('混淆矩阵（归一化）', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵已保存: {save_path}")
    plt.close()


def plot_feature_importance(importance_dict, label_names, save_path, top_n=15):
    """
    绘制特征重要性

    【是什么】：展示对分类最重要的词
    【解释】：权重绝对值越大，词越重要
    """
    n_classes = len(importance_dict)
    fig, axes = plt.subplots(1, min(n_classes, 3), figsize=(18, 6))

    if n_classes == 1:
        axes = [axes]
    elif n_classes == 2:
        axes = axes

    for idx, (class_key, features) in enumerate(list(importance_dict.items())[:3]):
        if idx >= len(axes):
            break

        # 提取特征和权重
        words = [f[0] for f in features[:top_n]]
        weights = [f[1] for f in features[:top_n]]

        # 颜色（正权重蓝色，负权重红色）
        colors = ['blue' if w > 0 else 'red' for w in weights]

        # 绘制条形图
        axes[idx].barh(range(len(words)), weights, color=colors, alpha=0.7)
        axes[idx].set_yticks(range(len(words)))
        axes[idx].set_yticklabels(words)
        axes[idx].set_xlabel('权重', fontsize=10)
        axes[idx].set_title(f'{label_names[idx] if idx < len(label_names) else class_key}\n重要特征',
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')
        axes[idx].invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 特征重要性已保存: {save_path}")
    plt.close()


def plot_classification_report(report, label_names, save_path):
    """
    绘制分类报告

    【指标说明】：
    - Precision（精确率）：预测为正的样本中真正为正的比例
    - Recall（召回率）：真正为正的样本中被预测为正的比例
    - F1-score：精确率和召回率的调和平均
    """
    # 提取每个类别的指标
    metrics = ['precision', 'recall', 'f1-score']
    data = []

    for label in label_names:
        if label in report:
            data.append([
                report[label]['precision'],
                report[label]['recall'],
                report[label]['f1-score']
            ])

    data = np.array(data)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, max(6, len(label_names) * 0.4)))

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # 设置刻度
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(label_names)

    # 添加数值标签
    for i in range(len(label_names)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_title('分类报告', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='分数')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 分类报告已保存: {save_path}")
    plt.close()


def main():
    """主训练流程"""
    args = parse_args()

    print("="*60)
    print("SVM文本分类 - 模型训练")
    print("="*60)
    print(f"\n训练配置:")
    print(f"  数据源: {args.data_source}")
    print(f"  类别: {args.categories if args.categories else '全部'}")
    print(f"  特征方法: {args.feature_method}")
    print(f"  最大特征数: {args.max_features}")
    print(f"  核函数: {args.kernel}")
    print(f"  超参数调优: {args.tune_hyperparams}")
    print(f"  集成模型: {args.ensemble}")

    # 创建保存目录
    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / args.model_dir
    result_dir = project_dir / args.result_dir
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    # ============================================
    # 1. 准备数据
    # ============================================
    print("\n" + "="*60)
    print("步骤1: 数据准备")
    print("="*60)

    try:
        (X_train, y_train), (X_test, y_test), feature_extractor, label_names = \
            prepare_text_classification_data(
                data_source=args.data_source,
                feature_method=args.feature_method,
                max_features=args.max_features,
                categories=args.categories,
                preprocess=args.preprocess
            )
    except Exception as e:
        print(f"\n✗ 数据准备失败: {e}")
        return

    # 保存特征提取器
    extractor_path = model_dir / f'{args.feature_method}_extractor.pkl'
    with open(extractor_path, 'wb') as f:
        pickle.dump(feature_extractor, f)
    print(f"\n✓ 特征提取器已保存: {extractor_path}")

    # 保存标签名称
    labels_path = model_dir / 'label_names.pkl'
    with open(labels_path, 'wb') as f:
        pickle.dump(label_names, f)

    # ============================================
    # 2. 训练模型
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 训练模型")
    print("="*60)

    if args.ensemble:
        # ============================================
        # 集成模型
        # ============================================
        print("\n训练集成模型...")
        model = SVMEnsembleClassifier(ensemble_method='voting')
        model.fit(X_train, y_train)

        model_path = model_dir / 'svm_ensemble_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    elif args.tune_hyperparams:
        # ============================================
        # 超参数调优
        # ============================================
        tuner = SVMHyperparameterTuner(
            kernel=args.kernel,
            search_method=args.search_method,
            cv=5
        )

        best_model, best_params = tuner.tune(X_train, y_train)

        # 保存最佳模型
        model_path = model_dir / f'svm_{args.kernel}_tuned_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        # 保存调优结果
        tuning_results_path = result_dir / 'hyperparameter_tuning_results.csv'
        tuner.search_results.to_csv(tuning_results_path, index=False)
        print(f"✓ 调优结果已保存: {tuning_results_path}")

        # 包装为SVMTextClassifier以便评估
        classifier = SVMTextClassifier(kernel=args.kernel)
        classifier.model = best_model
        model = classifier

    else:
        # ============================================
        # 单模型训练
        # ============================================
        model = SVMTextClassifier(
            kernel=args.kernel,
            C=args.C,
            gamma=args.gamma
        )
        model.fit(X_train, y_train)

        model_path = model_dir / f'svm_{args.kernel}_model.pkl'
        model.save_model(model_path)

    # ============================================
    # 3. 评估模型
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 评估模型")
    print("="*60)

    results = model.evaluate(X_test, y_test, label_names)

    print(f"\n测试集准确率: {results['accuracy']:.4f}")
    print("\n分类报告:")
    print("="*60)

    report = results['classification_report']
    for label in label_names:
        if label in report:
            print(f"\n{label}:")
            print(f"  Precision: {report[label]['precision']:.4f}")
            print(f"  Recall: {report[label]['recall']:.4f}")
            print(f"  F1-score: {report[label]['f1-score']:.4f}")

    print(f"\n宏平均:")
    print(f"  Precision: {report['macro avg']['precision']:.4f}")
    print(f"  Recall: {report['macro avg']['recall']:.4f}")
    print(f"  F1-score: {report['macro avg']['f1-score']:.4f}")

    # ============================================
    # 4. 可视化结果
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 可视化结果")
    print("="*60)

    # 混淆矩阵
    cm_path = result_dir / 'confusion_matrix.png'
    plot_confusion_matrix(results['confusion_matrix'], label_names, cm_path)

    # 分类报告
    report_path = result_dir / 'classification_report.png'
    plot_classification_report(report, label_names, report_path)

    # 特征重要性（仅线性核）
    if args.kernel == 'linear' and not args.ensemble:
        feature_names = feature_extractor.get_feature_names()
        if feature_names is not None:
            importance = model.get_feature_importance(feature_names, top_n=20)
            if importance:
                importance_path = result_dir / 'feature_importance.png'
                plot_feature_importance(importance, label_names, importance_path)

    # ============================================
    # 5. 保存结果
    # ============================================
    print("\n" + "="*60)
    print("步骤5: 保存结果")
    print("="*60)

    # 保存评估结果
    results_path = result_dir / 'evaluation_results.txt'
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("SVM文本分类 - 评估结果\n")
        f.write("="*60 + "\n\n")

        f.write(f"测试集准确率: {results['accuracy']:.4f}\n\n")

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

    print(f"✓ 评估结果已保存: {results_path}")

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("训练总结")
    print("="*60)
    print(f"✓ 模型已保存: {model_path}")
    print(f"✓ 特征提取器已保存: {extractor_path}")
    print(f"✓ 测试集准确率: {results['accuracy']:.4f}")
    print(f"\n使用以下命令进行评估:")
    print(f"  python src/evaluate.py --model_path {model_path} --extractor_path {extractor_path}")


if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)

    main()
