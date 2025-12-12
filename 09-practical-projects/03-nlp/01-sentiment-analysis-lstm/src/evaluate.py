"""
模型评估脚本

使用方法:
    python src/evaluate.py --model_path models/simple_lstm_best.h5
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 添加src目录到路径
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from data import load_imdb_data, decode_review, get_word_index
from model import LSTMSentimentAnalyzer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Evaluate LSTM sentiment analysis model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Model file path')
    parser.add_argument('--max_words', type=int, default=10000,
                       help='Maximum vocabulary size')
    parser.add_argument('--max_len', type=int, default=200,
                       help='Maximum sequence length')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='Results save directory')

    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵"""
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
        plt.close()
        print(f"✓ Confusion matrix saved: {save_path}")

    return cm


def plot_prediction_distribution(y_true, y_pred_proba, save_path=None):
    """绘制预测概率分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 负面样本的预测概率分布
    neg_probs = y_pred_proba[y_true == 0]
    axes[0].hist(neg_probs, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[0].set_xlabel('Prediction Probability', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Negative Samples - Prediction Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 正面样本的预测概率分布
    pos_probs = y_pred_proba[y_true == 1]
    axes[1].hist(pos_probs, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[1].set_xlabel('Prediction Probability', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Positive Samples - Prediction Distribution', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Prediction distribution saved: {save_path}")


def show_prediction_examples(X, y_true, y_pred, y_pred_proba, word_index, n=10):
    """显示预测示例"""
    print("\n" + "="*60)
    print("Prediction Examples")
    print("="*60)

    # 随机选择样本
    indices = np.random.choice(len(X), n, replace=False)

    for i, idx in enumerate(indices):
        print(f"\nSample {i+1}:")
        print(f"True label:       {'Positive' if y_true[idx] == 1 else 'Negative'}")
        print(f"Predicted label:  {'Positive' if y_pred[idx] == 1 else 'Negative'}")
        print(f"Prediction prob:  {y_pred_proba[idx]:.4f}")
        print(f"Correct:          {'✓' if y_pred[idx] == y_true[idx] else '✗'}")

        # 解码文本
        text = decode_review(X[idx], word_index)
        print(f"Review text:      {text[:200]}...")


def main():
    """主函数"""
    args = parse_args()

    print("="*60)
    print("LSTM Sentiment Analysis - Evaluation")
    print("="*60)
    print(f"\nModel path: {args.model_path}")

    # 创建结果目录
    project_dir = Path(__file__).parent.parent
    result_dir = project_dir / args.result_dir
    result_dir.mkdir(exist_ok=True)

    # 加载数据
    print("\n" + "="*60)
    print("Step 1: Load Data")
    print("="*60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_imdb_data(
        max_words=args.max_words,
        max_len=args.max_len
    )

    # 加载模型
    print("\n" + "="*60)
    print("Step 2: Load Model")
    print("="*60)

    analyzer = LSTMSentimentAnalyzer(
        max_words=args.max_words,
        max_len=args.max_len
    )
    analyzer.load_model(args.model_path)

    # 预测
    print("\n" + "="*60)
    print("Step 3: Model Prediction")
    print("="*60)

    print("\nPredicting on test set...")
    y_pred_proba = analyzer.predict_proba(X_test)
    y_pred = analyzer.predict(X_test)

    # 评估指标
    print("\n" + "="*60)
    print("Step 4: Calculate Metrics")
    print("="*60)

    # 基本指标
    metrics = analyzer.evaluate(X_test, y_test)
    print(f"\nTest set performance:")
    print(f"  Loss:     {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

    # 详细分类报告
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                               target_names=['Negative', 'Positive'],
                               digits=4))

    # 混淆矩阵
    print("\n" + "="*60)
    print("Step 5: Plot Confusion Matrix")
    print("="*60)

    model_name = Path(args.model_path).stem
    cm_path = result_dir / f'{model_name}_confusion_matrix.png'
    cm = plot_confusion_matrix(y_test, y_pred, cm_path)

    print(f"\nConfusion Matrix:")
    print(f"  True Negative (TN):  {cm[0, 0]}")
    print(f"  False Positive (FP): {cm[0, 1]}")
    print(f"  False Negative (FN): {cm[1, 0]}")
    print(f"  True Positive (TP):  {cm[1, 1]}")

    # 预测概率分布
    print("\n" + "="*60)
    print("Step 6: Plot Prediction Distribution")
    print("="*60)

    dist_path = result_dir / f'{model_name}_prediction_distribution.png'
    plot_prediction_distribution(y_test, y_pred_proba, dist_path)

    # 显示预测示例
    print("\n" + "="*60)
    print("Step 7: Show Prediction Examples")
    print("="*60)

    word_index = get_word_index()
    show_prediction_examples(X_test, y_test, y_pred, y_pred_proba, word_index, n=5)

    # 错误分析
    print("\n" + "="*60)
    print("Step 8: Error Analysis")
    print("="*60)

    # 找出预测错误的样本
    error_indices = np.where(y_pred != y_test)[0]
    print(f"\nError samples: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")

    # 显示一些错误样本
    if len(error_indices) > 0:
        print(f"\nShowing 5 error samples:")
        error_sample_indices = np.random.choice(error_indices, min(5, len(error_indices)), replace=False)

        for i, idx in enumerate(error_sample_indices):
            print(f"\nError sample {i+1}:")
            print(f"True label:       {'Positive' if y_test[idx] == 1 else 'Negative'}")
            print(f"Predicted label:  {'Positive' if y_pred[idx] == 1 else 'Negative'}")
            print(f"Prediction prob:  {y_pred_proba[idx]:.4f}")

            text = decode_review(X_test[idx], word_index)
            print(f"Review text:      {text[:200]}...")

    print("\n" + "="*60)
    print("Evaluation Completed!")
    print("="*60)


if __name__ == '__main__':
    main()
