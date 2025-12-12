"""
Transformer机器翻译训练脚本

支持完整的训练流程：
- 数据加载与预处理
- 模型构建与编译
- 训练监控与可视化
- 模型检查点保存

使用方法：
    python src/train.py --src_path data/train.en --tgt_path data/train.zh
    python src/train.py --max_samples 10000 --epochs 30 --batch_size 64

推荐数据集：
- WMT系列（英中、英德等）: http://www.statmt.org/wmt/
- IWSLT（TED演讲翻译）: https://wit3.fbk.eu/
- Tatoeba（多语言句对）: https://tatoeba.org/
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import prepare_translation_data
from src.model import TransformerTranslationModel


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='训练Transformer机器翻译模型',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据参数
    data_group = parser.add_argument_group('数据参数')
    data_group.add_argument('--src_path', type=str, default='data/train.en',
                           help='源语言训练文件路径')
    data_group.add_argument('--tgt_path', type=str, default='data/train.zh',
                           help='目标语言训练文件路径')
    data_group.add_argument('--max_len', type=int, default=50,
                           help='最大序列长度')
    data_group.add_argument('--max_samples', type=int, default=None,
                           help='最大样本数（用于快速实验）')
    data_group.add_argument('--max_vocab_size', type=int, default=10000,
                           help='最大词汇表大小')

    # 模型参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--num_layers', type=int, default=4,
                            help='编码器/解码器层数')
    model_group.add_argument('--d_model', type=int, default=256,
                            help='模型维度')
    model_group.add_argument('--num_heads', type=int, default=8,
                            help='注意力头数')
    model_group.add_argument('--d_ff', type=int, default=1024,
                            help='前馈网络维度')
    model_group.add_argument('--dropout_rate', type=float, default=0.1,
                            help='Dropout比率')

    # 训练参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--epochs', type=int, default=50,
                            help='训练轮数')
    train_group.add_argument('--batch_size', type=int, default=64,
                            help='批大小')
    train_group.add_argument('--learning_rate', type=float, default=None,
                            help='学习率（None表示使用自定义调度）')

    # 路径参数
    path_group = parser.add_argument_group('路径参数')
    path_group.add_argument('--model_dir', type=str, default='models',
                           help='模型保存目录')
    path_group.add_argument('--result_dir', type=str, default='results',
                           help='结果保存目录')

    return parser.parse_args()


def plot_training_history(history: keras.callbacks.History, save_path: Path):
    """
    绘制训练历史曲线

    可视化内容：
    - 训练/验证损失曲线
    - 训练/验证准确率曲线

    Args:
        history: Keras训练历史对象
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    axes[0].plot(history.history['loss'], label='训练损失', linewidth=2, color='#2C3E50')
    axes[0].plot(history.history['val_loss'], label='验证损失', linewidth=2, color='#E74C3C')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('训练/验证损失', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # 准确率曲线
    if 'masked_accuracy' in history.history:
        axes[1].plot(history.history['masked_accuracy'],
                    label='训练准确率', linewidth=2, color='#2C3E50')
        axes[1].plot(history.history['val_masked_accuracy'],
                    label='验证准确率', linewidth=2, color='#27AE60')
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('训练/验证准确率', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图已保存: {save_path}")
    plt.close()


def translate_samples(
    model: TransformerTranslationModel,
    processor,
    src_sequences: np.ndarray,
    num_samples: int = 5
):
    """
    翻译样本句子

    展示模型的翻译效果，用于直观评估模型质量

    Args:
        model: 翻译模型实例
        processor: 数据处理器实例
        src_sequences: 源语言序列
        num_samples: 展示样本数
    """
    print(f"\n{'='*60}")
    print("翻译样本展示")
    print(f"{'='*60}")

    for i in range(min(num_samples, len(src_sequences))):
        # 解码源句子
        src_seq = src_sequences[i]
        src_words = [
            processor.src_idx2word.get(idx, '<UNK>')
            for idx in src_seq if idx != 0
        ]

        # 生成翻译
        translation = model.translate(
            src_seq,
            processor.tgt_word2idx,
            processor.tgt_idx2word,
            max_len=processor.max_len
        )

        print(f"\n样本 {i+1}:")
        print(f"  源句子: {' '.join(src_words)}")
        print(f"  翻译结果: {translation}")


def setup_callbacks(
    model_path: Path,
    result_dir: Path
) -> List[keras.callbacks.Callback]:
    """
    配置训练回调函数

    包含的回调：
    - ModelCheckpoint: 保存最佳模型
    - EarlyStopping: 防止过拟合
    - ReduceLROnPlateau: 自适应学习率调整
    - TensorBoard: 训练过程可视化

    Args:
        model_path: 模型保存路径
        result_dir: 结果保存目录

    Returns:
        回调函数列表
    """
    callbacks = [
        # 保存最佳模型
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),

        # 早停策略
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        # 学习率衰减
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),

        # TensorBoard可视化
        keras.callbacks.TensorBoard(
            log_dir=result_dir / 'logs',
            histogram_freq=1,
            write_graph=True
        )
    ]

    return callbacks


def main():
    """
    主训练流程

    流程步骤：
    1. 解析命令行参数
    2. 加载并预处理数据
    3. 构建Transformer模型
    4. 配置训练回调
    5. 执行训练
    6. 保存模型和结果
    7. 展示翻译样本
    """
    args = parse_args()

    # 打印配置信息
    print(f"{'='*60}")
    print("Transformer机器翻译 - 模型训练")
    print(f"{'='*60}")
    print(f"\n训练配置:")
    print(f"  源语言: {args.src_path}")
    print(f"  目标语言: {args.tgt_path}")
    print(f"  序列长度: {args.max_len}")
    print(f"  样本数: {args.max_samples or '全部'}")
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
    # 步骤1: 数据准备
    # ============================================
    print(f"\n{'='*60}")
    print("步骤1: 数据准备")
    print(f"{'='*60}")

    try:
        (src_train, tgt_train), (src_val, tgt_val), processor = prepare_translation_data(
            args.src_path,
            args.tgt_path,
            max_len=args.max_len,
            max_samples=args.max_samples
        )
    except FileNotFoundError as e:
        print(f"\n错误: 数据文件不存在")
        print(f"\n数据集获取建议:")
        print("  1. WMT数据集: http://www.statmt.org/wmt/")
        print("  2. IWSLT数据集: https://wit3.fbk.eu/")
        print("  3. Tatoeba数据集: https://tatoeba.org/downloads")
        print(f"\n请下载数据集并放置在data/目录下")
        return

    # 保存数据处理器
    processor_path = model_dir / 'translation_processor.pkl'
    processor.save_processor(processor_path)

    print(f"\n数据统计:")
    print(f"  训练集: {len(src_train)} 样本")
    print(f"  验证集: {len(src_val)} 样本")
    print(f"  源语言词汇表: {len(processor.src_word2idx)} 词")
    print(f"  目标语言词汇表: {len(processor.tgt_word2idx)} 词")

    # ============================================
    # 步骤2: 创建模型
    # ============================================
    print(f"\n{'='*60}")
    print("步骤2: 构建模型")
    print(f"{'='*60}")

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

    # 统计参数量
    total_params = translator.model.count_params()
    print(f"\n模型参数量: {total_params:,}")

    # ============================================
    # 步骤3: 训练模型
    # ============================================
    print(f"\n{'='*60}")
    print("步骤3: 训练模型")
    print(f"{'='*60}")

    # 配置回调
    model_path = model_dir / 'transformer_translation_model.h5'
    callbacks = setup_callbacks(model_path, result_dir)

    # 开始训练
    print(f"\n开始训练...")
    history = translator.train(
        src_train, tgt_train,
        src_val, tgt_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 打印训练结果
    print(f"\n训练完成！")
    print(f"  最佳验证损失: {min(history.history['val_loss']):.4f}")
    if 'val_masked_accuracy' in history.history:
        print(f"  最佳验证准确率: {max(history.history['val_masked_accuracy']):.4f}")

    # ============================================
    # 步骤4: 保存结果
    # ============================================
    print(f"\n{'='*60}")
    print("步骤4: 保存结果")
    print(f"{'='*60}")

    # 绘制训练历史
    plot_path = result_dir / 'training_history.png'
    plot_training_history(history, plot_path)

    # 保存训练历史数据
    history_path = result_dir / 'training_history.npy'
    np.save(history_path, history.history)
    print(f"训练历史数据已保存: {history_path}")

    # ============================================
    # 步骤5: 翻译样本
    # ============================================
    print(f"\n{'='*60}")
    print("步骤5: 翻译样本")
    print(f"{'='*60}")

    translate_samples(translator, processor, src_val, num_samples=5)

    # ============================================
    # 总结
    # ============================================
    print(f"\n{'='*60}")
    print("训练总结")
    print(f"{'='*60}")
    print(f"模型已保存: {model_path}")
    print(f"处理器已保存: {processor_path}")
    print(f"训练历史已保存: {plot_path}")
    print(f"\n使用以下命令进行评估:")
    print(f"  python src/evaluate.py --model_path {model_path} --processor_path {processor_path}")
    print(f"\n使用TensorBoard查看训练过程:")
    print(f"  tensorboard --logdir {result_dir / 'logs'}")


if __name__ == '__main__':
    # 设置随机种子，确保可复现性
    np.random.seed(42)
    tf.random.set_seed(42)

    # 运行主函数
    main()
