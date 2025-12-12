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
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from data import load_imdb_data
from model import LSTMSentimentAnalyzer, get_callbacks
from utils.visualization import plot_training_history


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练LSTM情感分析模型')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='simple_lstm',
                       choices=['simple_lstm', 'bilstm', 'stacked_lstm'],
                       help='模型类型')
    parser.add_argument('--max_words', type=int, default=10000,
                       help='词汇表大小')
    parser.add_argument('--max_len', type=int, default=200,
                       help='序列最大长度')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')

    # 其他参数
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--patience', type=int, default=5,
                       help='早停耐心值')

    # 保存路径
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("="*60)
    print("LSTM情感分析 - 模型训练")
    print("="*60)
    print(f"\n配置:")
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
    print("步骤1: 加载数据")
    print("="*60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_imdb_data(
        max_words=args.max_words,
        max_len=args.max_len,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # 创建模型
    print("\n" + "="*60)
    print("步骤2: 创建模型")
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
    print("步骤3: 训练模型")
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
    print("步骤4: 评估模型")
    print("="*60)

    # 训练集评估
    train_metrics = analyzer.evaluate(X_train, y_train)
    print(f"\n训练集性能:")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")

    # 验证集评估
    val_metrics = analyzer.evaluate(X_val, y_val)
    print(f"\n验证集性能:")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

    # 测试集评估
    test_metrics = analyzer.evaluate(X_test, y_test)
    print(f"\n测试集性能:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    # 保存模型
    print("\n" + "="*60)
    print("步骤5: 保存模型")
    print("="*60)

    final_model_path = model_dir / f'{args.model_type}_final.h5'
    analyzer.save_model(final_model_path)

    # 绘制训练曲线
    print("\n" + "="*60)
    print("步骤6: 绘制训练曲线")
    print("="*60)

    fig = plot_training_history(history.history)
    fig_path = result_dir / f'{args.model_type}_training_history.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {fig_path}")

    # 保存训练历史
    history_path = result_dir / f'{args.model_type}_history.npz'
    np.savez(history_path, **history.history)
    print(f"✓ 训练历史已保存: {history_path}")

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
    print(f"✓ 评估结果已保存: {results_path}")

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"\n最佳模型: {model_path}")
    print(f"最终模型: {final_model_path}")
    print(f"训练曲线: {fig_path}")
    print(f"\n测试集准确率: {test_metrics['accuracy']:.4f}")


if __name__ == '__main__':
    main()
