"""
模型评估脚本

使用方法:
    python src/evaluate.py --model_path models/simple_model.h5 --vocab_path models/simple_vocab.pkl
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_imdb_data, TextPreprocessor, Vocabulary
from tensorflow.keras.preprocessing.sequence import pad_sequences


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估Transformer文本分类模型')

    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='词汇表文件路径')
    parser.add_argument('--data_dir', type=str, default='data/aclImdb',
                       help='数据目录')
    parser.add_argument('--max_len', type=int, default=256,
                       help='最大序列长度')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数（用于快速测试）')

    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {save_path}")

    return cm


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    绘制ROC曲线

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        save_path: 保存路径
    """
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

    return roc_auc


def plot_prediction_distribution(y_true, y_pred_proba, save_path=None):
    """
    绘制预测概率分布

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))

    # 负面样本的预测概率分布
    neg_proba = y_pred_proba[y_true == 0]
    plt.hist(neg_proba, bins=50, alpha=0.5, label='Negative', color='red')

    # 正面样本的预测概率分布
    pos_proba = y_pred_proba[y_true == 1]
    plt.hist(pos_proba, bins=50, alpha=0.5, label='Positive', color='green')

    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Prediction Probability Distribution', fontsize=14, pad=15)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 预测分布图已保存: {save_path}")


def analyze_errors(X_test, y_test, y_pred, y_pred_proba, vocab, num_examples=10):
    """
    分析错误样本

    Args:
        X_test: 测试数据
        y_test: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率
        vocab: 词汇表
        num_examples: 显示样本数
    """
    print("\n" + "="*60)
    print("错误样本分析")
    print("="*60)

    # 找出错误样本
    error_indices = np.where(y_pred != y_test)[0]
    print(f"\n错误样本数量: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")

    if len(error_indices) == 0:
        print("没有错误样本！")
        return

    # 错误类型统计
    false_positives = np.where((y_pred == 1) & (y_test == 0))[0]
    false_negatives = np.where((y_pred == 0) & (y_test == 1))[0]

    print(f"\n错误类型分布:")
    print(f"  假正例 (预测正面，实际负面): {len(false_positives)} ({len(false_positives)/len(error_indices)*100:.1f}%)")
    print(f"  假负例 (预测负面，实际正面): {len(false_negatives)} ({len(false_negatives)/len(error_indices)*100:.1f}%)")

    # 按置信度排序错误样本
    error_confidence = np.abs(y_pred_proba[error_indices] - 0.5)
    sorted_indices = error_indices[np.argsort(-error_confidence)]

    # 显示最自信的错误样本
    print(f"\n最自信的 {min(num_examples, len(sorted_indices))} 个错误样本:")
    for i, idx in enumerate(sorted_indices[:num_examples]):
        text = vocab.decode(X_test[idx])
        text = text.replace(vocab.PAD_TOKEN, '').strip()
        if len(text) > 150:
            text = text[:150] + '...'

        print(f"\n错误样本 {i+1}:")
        print(f"  文本: {text}")
        print(f"  真实标签: {'正面' if y_test[idx] == 1 else '负面'}")
        print(f"  预测标签: {'正面' if y_pred[idx] == 1 else '负面'}")
        print(f"  预测概率: {y_pred_proba[idx]:.4f}")
        print(f"  置信度: {abs(y_pred_proba[idx] - 0.5):.4f}")


def analyze_correct_predictions(X_test, y_test, y_pred, y_pred_proba, vocab, num_examples=5):
    """
    分析正确预测样本

    Args:
        X_test: 测试数据
        y_test: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率
        vocab: 词汇表
        num_examples: 显示样本数
    """
    print("\n" + "="*60)
    print("正确预测样本分析")
    print("="*60)

    # 找出正确样本
    correct_indices = np.where(y_pred == y_test)[0]

    # 按置信度排序
    correct_confidence = np.abs(y_pred_proba[correct_indices] - 0.5)
    sorted_indices = correct_indices[np.argsort(-correct_confidence)]

    # 显示最自信的正确样本
    print(f"\n最自信的 {num_examples} 个正确样本:")
    for i, idx in enumerate(sorted_indices[:num_examples]):
        text = vocab.decode(X_test[idx])
        text = text.replace(vocab.PAD_TOKEN, '').strip()
        if len(text) > 150:
            text = text[:150] + '...'

        print(f"\n正确样本 {i+1}:")
        print(f"  文本: {text}")
        print(f"  真实标签: {'正面' if y_test[idx] == 1 else '负面'}")
        print(f"  预测概率: {y_pred_proba[idx]:.4f}")
        print(f"  置信度: {abs(y_pred_proba[idx] - 0.5):.4f}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("="*60)
    print("Transformer文本分类 - 模型评估")
    print("="*60)
    print(f"\n模型路径: {args.model_path}")
    print(f"词汇表路径: {args.vocab_path}")

    # 创建结果目录
    project_dir = Path(__file__).parent.parent
    result_dir = project_dir / args.result_dir
    result_dir.mkdir(exist_ok=True)

    # ============================================
    # 步骤1: 加载词汇表
    # ============================================
    print("\n" + "="*60)
    print("步骤1: 加载词汇表")
    print("="*60)

    vocab = Vocabulary()
    vocab.load(args.vocab_path)
    print(f"词汇表大小: {len(vocab)}")

    # ============================================
    # 步骤2: 加载数据
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 加载数据")
    print("="*60)

    try:
        test_dir = Path(args.data_dir) / 'test'
        test_texts, test_labels = load_imdb_data(test_dir, args.max_samples)
    except FileNotFoundError as e:
        print(f"\n✗ 数据文件不存在: {e}")
        return

    # 预处理
    preprocessor = TextPreprocessor()
    test_texts = [preprocessor(text) for text in test_texts]

    # 编码
    test_sequences = [vocab.encode(text) for text in test_texts]

    # 填充
    X_test = pad_sequences(
        test_sequences,
        maxlen=args.max_len,
        padding='post',
        truncating='post',
        value=vocab.word2idx[vocab.PAD_TOKEN]
    )
    y_test = np.array(test_labels)

    print(f"测试集形状: {X_test.shape}")
    print(f"正负样本比: {y_test.sum()}/{len(y_test)-y_test.sum()}")

    # ============================================
    # 步骤3: 加载模型
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 加载模型")
    print("="*60)

    model = keras.models.load_model(args.model_path)
    print(f"✓ 模型已加载")

    # 打印模型摘要
    print(f"\n模型结构:")
    model.summary()

    # ============================================
    # 步骤4: 预测
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 模型预测")
    print("="*60)

    print("\n预测测试集...")
    y_pred_proba = model.predict(X_test, verbose=1).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # ============================================
    # 步骤5: 计算评估指标
    # ============================================
    print("\n" + "="*60)
    print("步骤5: 计算评估指标")
    print("="*60)

    # 基本指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = auc(*roc_curve(y_test, y_pred_proba)[:2])

    print(f"\n测试集性能:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {roc_auc:.4f}")

    # 详细分类报告
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred,
                               target_names=['Negative', 'Positive'],
                               digits=4))

    # ============================================
    # 步骤6: 绘制混淆矩阵
    # ============================================
    print("\n" + "="*60)
    print("步骤6: 绘制混淆矩阵")
    print("="*60)

    model_name = Path(args.model_path).stem
    cm_path = result_dir / f'{model_name}_confusion_matrix.png'
    cm = plot_confusion_matrix(y_test, y_pred, cm_path)

    print(f"\n混淆矩阵:")
    print(f"  真负例(TN): {cm[0, 0]} (正确预测负面)")
    print(f"  假正例(FP): {cm[0, 1]} (错误预测正面)")
    print(f"  假负例(FN): {cm[1, 0]} (错误预测负面)")
    print(f"  真正例(TP): {cm[1, 1]} (正确预测正面)")

    # ============================================
    # 步骤7: 绘制ROC曲线
    # ============================================
    print("\n" + "="*60)
    print("步骤7: 绘制ROC曲线")
    print("="*60)

    roc_path = result_dir / f'{model_name}_roc_curve.png'
    plot_roc_curve(y_test, y_pred_proba, roc_path)

    # ============================================
    # 步骤8: 绘制预测分布
    # ============================================
    print("\n" + "="*60)
    print("步骤8: 绘制预测分布")
    print("="*60)

    dist_path = result_dir / f'{model_name}_prediction_distribution.png'
    plot_prediction_distribution(y_test, y_pred_proba, dist_path)

    # ============================================
    # 步骤9: 错误分析
    # ============================================
    analyze_errors(X_test, y_test, y_pred, y_pred_proba, vocab, num_examples=10)

    # ============================================
    # 步骤10: 正确预测分析
    # ============================================
    analyze_correct_predictions(X_test, y_test, y_pred, y_pred_proba, vocab, num_examples=5)

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)

    print(f"\n生成的文件:")
    print(f"  1. 混淆矩阵: {cm_path}")
    print(f"  2. ROC曲线: {roc_path}")
    print(f"  3. 预测分布: {dist_path}")

    print(f"\n模型性能总结:")
    print(f"  准确率: {accuracy:.2%}")
    print(f"  AUC: {roc_auc:.4f}")

    if accuracy >= 0.85:
        print(f"\n  ✓✓ 模型性能优秀！")
    elif accuracy >= 0.75:
        print(f"\n  ✓ 模型性能良好")
    else:
        print(f"\n  ⚠ 模型性能有待提升")


if __name__ == '__main__':
    main()
