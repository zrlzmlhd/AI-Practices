"""
模型训练脚本

使用方法:
    python src/train.py --model_type simple_lstm --epochs 10
    python src/train.py --model_type bilstm --epochs 20
    python src/train.py --model_type stacked_lstm --epochs 15
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 添加src目录到路径
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from data import load_imdb_data
from model import LSTMSentimentAnalyzer, get_callbacks
from utils.visualization import plot_training_history


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train LSTM sentiment analysis model')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='simple_lstm',
                       choices=['simple_lstm', 'bilstm', 'stacked_lstm'],
                       help='Model architecture')
    parser.add_argument('--max_words', type=int, default=10000,
                       help='Maximum vocabulary size')
    parser.add_argument('--max_len', type=int, default=200,
                       help='Maximum sequence length')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')

    # 其他参数
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')

    # 保存路径
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Model save directory')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='Results save directory')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("="*60)
    print("LSTM Sentiment Analysis - Training")
    print("="*60)
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # 创建保存目录
    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / args.model_dir
    result_dir = project_dir / args.result_dir
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    # 加载数据
    print("\n" + "="*60)
    print("Step 1: Load Data")
    print("="*60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_imdb_data(
        max_words=args.max_words,
        max_len=args.max_len,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # 创建模型
    print("\n" + "="*60)
    print("Step 2: Create Model")
    print("="*60)

    analyzer = LSTMSentimentAnalyzer(
        model_type=args.model_type,
        max_words=args.max_words,
        max_len=args.max_len,
        random_state=args.random_state
    )

    # 设置回调函数
    model_path = model_dir / f'{args.model_type}_best.h5'
    callbacks = get_callbacks(model_path, patience=args.patience)

    # 训练模型
    print("\n" + "="*60)
    print("Step 3: Train Model")
    print("="*60)

    history = analyzer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    # 评估模型
    print("\n" + "="*60)
    print("Step 4: Evaluate Model")
    print("="*60)

    # 训练集评估
    train_metrics = analyzer.evaluate(X_train, y_train)
    print(f"\nTraining set performance:")
    print(f"  Loss:     {train_metrics['loss']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")

    # 验证集评估
    val_metrics = analyzer.evaluate(X_val, y_val)
    print(f"\nValidation set performance:")
    print(f"  Loss:     {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

    # 测试集评估
    test_metrics = analyzer.evaluate(X_test, y_test)
    print(f"\nTest set performance:")
    print(f"  Loss:     {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    # 保存模型
    print("\n" + "="*60)
    print("Step 5: Save Model")
    print("="*60)

    final_model_path = model_dir / f'{args.model_type}_final.h5'
    analyzer.save_model(final_model_path)

    # 绘制训练曲线
    print("\n" + "="*60)
    print("Step 6: Plot Training History")
    print("="*60)

    fig = plot_training_history(history.history)
    fig_path = result_dir / f'{args.model_type}_training_history.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Training curve saved: {fig_path}")

    # 保存训练历史
    history_path = result_dir / f'{args.model_type}_history.npz'
    np.savez(history_path, **history.history)
    print(f"✓ Training history saved: {history_path}")

    # 保存评估结果
    results = {
        'model_type': args.model_type,
        'train_loss': train_metrics['loss'],
        'train_accuracy': train_metrics['accuracy'],
        'val_loss': val_metrics['loss'],
        'val_accuracy': val_metrics['accuracy'],
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
    }

    results_path = result_dir / f'{args.model_type}_results.txt'
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Evaluation results saved: {results_path}")

    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"\nBest model: {model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Training curve: {fig_path}")
    print(f"\nTest accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == '__main__':
    main()
