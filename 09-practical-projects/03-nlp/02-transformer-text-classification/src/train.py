"""
Transformer文本分类模型训练脚本

使用方法:
    python src/train.py --model_type simple --epochs 10
    python src/train.py --model_type improved --epochs 20
    python src/train.py --model_type advanced --epochs 30
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import prepare_data
from src.model import TransformerTextClassifier


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Transformer文本分类模型')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='simple',
                       choices=['simple', 'improved', 'advanced'],
                       help='模型类型')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/aclImdb',
                       help='数据目录')
    parser.add_argument('--max_vocab_size', type=int, default=10000,
                       help='最大词汇表大小')
    parser.add_argument('--max_len', type=int, default=256,
                       help='最大序列长度')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数（用于快速测试）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='早停耐心值')

    # 其他参数
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def create_callbacks(model_path, patience=3):
    """
    创建训练回调函数

    Args:
        model_path: 模型保存路径
        patience: 早停耐心值

    Returns:
        回调函数列表
    """
    callbacks = []

    # ============================================
    # ModelCheckpoint: 保存最佳模型
    # ============================================
    # 【是什么】：在验证集上表现最好时保存模型
    # 【为什么】：
    #   - 防止过拟合
    #   - 保存最佳性能的模型
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)

    # ============================================
    # EarlyStopping: 早停
    # ============================================
    # 【是什么】：验证集性能不再提升时停止训练
    # 【为什么】：
    #   - 防止过拟合
    #   - 节省训练时间
    #   - 自动找到最佳训练轮数
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    callbacks.append(early_stopping)

    # ============================================
    # ReduceLROnPlateau: 学习率衰减
    # ============================================
    # 【是什么】：验证集性能停滞时降低学习率
    # 【为什么】：
    #   - 帮助模型跳出局部最优
    #   - 在训练后期进行精细调整
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        mode='min',
        verbose=1
    )
    callbacks.append(reduce_lr)

    return callbacks


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("="*60)
    print("Transformer文本分类 - 模型训练")
    print("="*60)
    print(f"\n配置:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # 设置随机种子
    # 【为什么】：保证结果可复现
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    # 创建保存目录
    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / args.model_dir
    result_dir = project_dir / args.result_dir
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    # ============================================
    # 步骤1: 准备数据
    # ============================================
    print("\n" + "="*60)
    print("步骤1: 准备数据")
    print("="*60)

    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab = prepare_data(
            data_dir=args.data_dir,
            max_vocab_size=args.max_vocab_size,
            max_len=args.max_len,
            random_state=args.random_state,
            max_samples=args.max_samples
        )
    except FileNotFoundError as e:
        print(f"\n✗ 数据文件不存在: {e}")
        print("\n请先下载数据:")
        print("  cd data")
        print("  python download_data.py")
        return

    # 保存词汇表
    vocab_path = model_dir / f'{args.model_type}_vocab.pkl'
    vocab.save(vocab_path)

    # ============================================
    # 步骤2: 创建模型
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 创建模型")
    print("="*60)

    classifier = TransformerTextClassifier(
        vocab_size=len(vocab),
        max_len=args.max_len,
        num_classes=2,
        model_type=args.model_type
    )

    # 打印模型摘要
    print(f"\n模型结构:")
    classifier.summary()

    # 计算参数量
    total_params = classifier.model.count_params()
    print(f"\n总参数量: {total_params:,}")

    # ============================================
    # 步骤3: 训练模型
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 训练模型")
    print("="*60)

    # 创建回调函数
    model_path = model_dir / f'{args.model_type}_model.h5'
    callbacks = create_callbacks(
        model_path=model_path,
        patience=args.early_stopping_patience
    )

    # 训练
    print(f"\n开始训练...")
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        callbacks=callbacks,
        verbose=1
    )

    # ============================================
    # 步骤4: 评估模型
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 评估模型")
    print("="*60)

    # 训练集评估
    train_metrics = classifier.evaluate(X_train, y_train)
    print(f"\n训练集性能:")
    for name, value in train_metrics.items():
        print(f"  {name}: {value:.4f}")

    # 验证集评估
    val_metrics = classifier.evaluate(X_val, y_val)
    print(f"\n验证集性能:")
    for name, value in val_metrics.items():
        print(f"  {name}: {value:.4f}")

    # 测试集评估
    test_metrics = classifier.evaluate(X_test, y_test)
    print(f"\n测试集性能:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    # ============================================
    # 步骤5: 保存结果
    # ============================================
    print("\n" + "="*60)
    print("步骤5: 保存结果")
    print("="*60)

    # 保存训练历史
    history_path = result_dir / f'{args.model_type}_history.npz'
    np.savez(
        history_path,
        **history.history
    )
    print(f"✓ 训练历史已保存: {history_path}")

    # 保存评估结果
    results = {
        'model_type': args.model_type,
        'total_params': total_params,
        'train_loss': train_metrics['loss'],
        'train_accuracy': train_metrics['accuracy'],
        'train_auc': train_metrics.get('auc', 0),
        'val_loss': val_metrics['loss'],
        'val_accuracy': val_metrics['accuracy'],
        'val_auc': val_metrics.get('auc', 0),
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_auc': test_metrics.get('auc', 0),
    }

    results_path = result_dir / f'{args.model_type}_results.txt'
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ 评估结果已保存: {results_path}")

    # ============================================
    # 步骤6: 示例预测
    # ============================================
    print("\n" + "="*60)
    print("步骤6: 示例预测")
    print("="*60)

    # 从测试集中随机选择几个样本
    num_examples = 5
    indices = np.random.choice(len(X_test), num_examples, replace=False)

    print(f"\n随机选择 {num_examples} 个测试样本:")
    for i, idx in enumerate(indices):
        # 预测
        x = X_test[idx:idx+1]
        y_true = y_test[idx]
        y_pred = classifier.predict(x)[0]
        y_proba = classifier.predict_proba(x)[0][0]

        # 解码文本
        text = vocab.decode(X_test[idx])
        # 去除padding
        text = text.replace(vocab.PAD_TOKEN, '').strip()
        # 截断显示
        if len(text) > 100:
            text = text[:100] + '...'

        print(f"\n样本 {i+1}:")
        print(f"  文本: {text}")
        print(f"  真实标签: {'正面' if y_true == 1 else '负面'}")
        print(f"  预测标签: {'正面' if y_pred == 1 else '负面'}")
        print(f"  预测概率: {y_proba:.4f}")
        print(f"  预测{'正确' if y_pred == y_true else '错误'}")

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"\n模型保存路径: {model_path}")
    print(f"词汇表保存路径: {vocab_path}")
    print(f"\n测试集性能:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  AUC: {test_metrics.get('auc', 0):.4f}")

    # 给出建议
    print(f"\n下一步:")
    print(f"  1. 查看训练历史: {history_path}")
    print(f"  2. 评估模型: python src/evaluate.py --model_path {model_path} --vocab_path {vocab_path}")
    print(f"  3. 尝试其他模型类型:")
    print(f"     python src/train.py --model_type improved")
    print(f"     python src/train.py --model_type advanced")

    # 性能分析
    print(f"\n性能分析:")
    if test_metrics['accuracy'] < 0.7:
        print("  ⚠ 准确率较低，建议:")
        print("    - 增加训练轮数 (--epochs)")
        print("    - 使用更复杂的模型 (--model_type improved/advanced)")
        print("    - 增加词汇表大小 (--max_vocab_size)")
    elif test_metrics['accuracy'] < 0.85:
        print("  ✓ 准确率中等，可以尝试:")
        print("    - 使用更复杂的模型")
        print("    - 调整学习率")
        print("    - 增加训练数据")
    else:
        print("  ✓✓ 准确率优秀！")

    # 过拟合检查
    if train_metrics['accuracy'] - test_metrics['accuracy'] > 0.1:
        print("\n  ⚠ 检测到过拟合，建议:")
        print("    - 增加Dropout")
        print("    - 减少模型复杂度")
        print("    - 增加训练数据")
        print("    - 使用数据增强")


if __name__ == '__main__':
    main()
