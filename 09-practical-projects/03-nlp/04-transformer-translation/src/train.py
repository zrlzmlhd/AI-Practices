"""
Transformer机器翻译训练脚本

使用方法:
    python src/train.py --src_path data/train.en --tgt_path data/train.zh --epochs 50
    python src/train.py --max_samples 10000 --epochs 30 --batch_size 64

【数据集建议】:
- WMT系列数据集（英中、英德等）
- IWSLT数据集（TED演讲翻译）
- Tatoeba数据集（多语言句对）
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import prepare_translation_data
from src.model import TransformerTranslationModel


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Transformer机器翻译模型')

    # 数据参数
    parser.add_argument('--src_path', type=str, default='data/train.en',
                       help='源语言训练文件路径')
    parser.add_argument('--tgt_path', type=str, default='data/train.zh',
                       help='目标语言训练文件路径')
    parser.add_argument('--max_len', type=int, default=50,
                       help='最大序列长度')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数（用于快速实验）')
    parser.add_argument('--max_vocab_size', type=int, default=10000,
                       help='最大词汇表大小')

    # 模型参数
    parser.add_argument('--num_layers', type=int, default=4,
                       help='编码器/解码器层数')
    parser.add_argument('--d_model', type=int, default=256,
                       help='模型维度')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=1024,
                       help='前馈网络维度')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout比率')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率（None表示使用自定义调度）')

    # 保存路径
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def plot_training_history(history, save_path):
    """
    绘制训练历史

    【可视化内容】:
    - 训练/验证损失曲线
    - 训练/验证准确率曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    axes[0].plot(history.history['loss'], label='训练损失', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练/验证损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    if 'masked_accuracy' in history.history:
        axes[1].plot(history.history['masked_accuracy'], label='训练准确率', linewidth=2)
        axes[1].plot(history.history['val_masked_accuracy'], label='验证准确率', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('训练/验证准确率曲线', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练历史已保存: {save_path}")
    plt.close()


def translate_samples(model, processor, src_sequences, num_samples=5):
    """
    翻译样本句子

    【是什么】：展示模型的翻译效果
    【为什么】：直观评估模型质量
    """
    print("\n" + "="*60)
    print("翻译样本展示")
    print("="*60)

    for i in range(min(num_samples, len(src_sequences))):
        # 源句子
        src_seq = src_sequences[i]
        src_words = [processor.src_idx2word.get(idx, '<UNK>')
                     for idx in src_seq if idx != 0]

        # 翻译
        translation = model.translate(
            src_seq,
            processor.tgt_word2idx,
            processor.tgt_idx2word,
            max_len=processor.max_len
        )

        print(f"\n样本 {i+1}:")
        print(f"  源句子: {' '.join(src_words)}")
        print(f"  翻译结果: {translation}")


def main():
    """主训练流程"""
    args = parse_args()

    print("="*60)
    print("Transformer机器翻译 - 模型训练")
    print("="*60)
    print(f"\n训练配置:")
    print(f"  源语言文件: {args.src_path}")
    print(f"  目标语言文件: {args.tgt_path}")
    print(f"  最大序列长度: {args.max_len}")
    print(f"  最大样本数: {args.max_samples}")
    print(f"  模型层数: {args.num_layers}")
    print(f"  模型维度: {args.d_model}")
    print(f"  注意力头数: {args.num_heads}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批大小: {args.batch_size}")

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
        (src_train, tgt_train), (src_val, tgt_val), processor = prepare_translation_data(
            args.src_path,
            args.tgt_path,
            max_len=args.max_len,
            max_samples=args.max_samples
        )
    except FileNotFoundError as e:
        print(f"\n✗ 数据文件不存在: {e}")
        print("\n【数据集获取建议】:")
        print("1. WMT数据集: http://www.statmt.org/wmt19/translation-task.html")
        print("2. IWSLT数据集: https://wit3.fbk.eu/")
        print("3. Tatoeba数据集: https://tatoeba.org/en/downloads")
        print("\n请下载数据集并放置在data/目录下")
        return

    # 保存处理器
    processor_path = model_dir / 'translation_processor.pkl'
    processor.save_processor(processor_path)

    print(f"\n数据统计:")
    print(f"  训练集大小: {len(src_train)}")
    print(f"  验证集大小: {len(src_val)}")
    print(f"  源语言词汇表: {len(processor.src_word2idx)}")
    print(f"  目标语言词汇表: {len(processor.tgt_word2idx)}")

    # ============================================
    # 2. 创建模型
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 创建模型")
    print("="*60)

    translator = TransformerTranslationModel(
        src_vocab_size=len(processor.src_word2idx),
        tgt_vocab_size=len(processor.tgt_word2idx),
        max_len=args.max_len,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout_rate=args.dropout_rate
    )

    print(f"\n模型结构:")
    translator.summary()

    # 计算参数量
    total_params = translator.model.count_params()
    print(f"\n总参数量: {total_params:,}")

    # ============================================
    # 3. 训练模型
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 训练模型")
    print("="*60)

    # 回调函数
    model_path = model_dir / 'transformer_translation_model.h5'
    callbacks = [
        # 【ModelCheckpoint】：保存最佳模型
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # 【EarlyStopping】：早停
        # 【为什么】：防止过拟合，节省训练时间
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        # 【ReduceLROnPlateau】：学习率衰减
        # 【为什么】：当验证损失不再下降时降低学习率
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),

        # 【TensorBoard】：可视化训练过程
        keras.callbacks.TensorBoard(
            log_dir=result_dir / 'logs',
            histogram_freq=1
        )
    ]

    # 训练
    print(f"\n开始训练...")
    history = translator.train(
        src_train, tgt_train,
        src_val, tgt_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n✓ 训练完成！")
    print(f"  最佳验证损失: {min(history.history['val_loss']):.4f}")
    if 'val_masked_accuracy' in history.history:
        print(f"  最佳验证准确率: {max(history.history['val_masked_accuracy']):.4f}")

    # ============================================
    # 4. 保存结果
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 保存结果")
    print("="*60)

    # 绘制训练历史
    plot_path = result_dir / 'training_history.png'
    plot_training_history(history, plot_path)

    # 保存训练历史
    history_path = result_dir / 'training_history.npy'
    np.save(history_path, history.history)
    print(f"✓ 训练历史已保存: {history_path}")

    # ============================================
    # 5. 翻译样本
    # ============================================
    print("\n" + "="*60)
    print("步骤5: 翻译样本")
    print("="*60)

    translate_samples(translator, processor, src_val, num_samples=5)

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("训练总结")
    print("="*60)
    print(f"✓ 模型已保存: {model_path}")
    print(f"✓ 处理器已保存: {processor_path}")
    print(f"✓ 训练历史已保存: {plot_path}")
    print(f"\n使用以下命令进行评估:")
    print(f"  python src/evaluate.py --model_path {model_path} --processor_path {processor_path}")
    print("\n使用TensorBoard查看训练过程:")
    print(f"  tensorboard --logdir {result_dir / 'logs'}")


if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)

    main()
