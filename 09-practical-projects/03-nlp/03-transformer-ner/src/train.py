"""
Transformer NER模型训练脚本

使用方法:
    python src/train.py --model_type transformer --epochs 30
    python src/train.py --model_type transformer_crf --epochs 50
    python src/train.py --model_type bilstm_crf --epochs 50
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import prepare_ner_data
from src.model import TransformerNERModel


def parse_args():
    parser = argparse.ArgumentParser(description='训练Transformer NER模型')
    parser.add_argument('--model_type', type=str, default='transformer_crf',
                       choices=['transformer', 'transformer_crf', 'bilstm_crf'])
    parser.add_argument('--train_path', type=str, default='data/train.txt')
    parser.add_argument('--test_path', type=str, default='data/test.txt')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--result_dir', type=str, default='results')
    return parser.parse_args()


def main():
    args = parse_args()
    print("="*60)
    print("Transformer NER - 模型训练")
    print("="*60)

    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / args.model_dir
    result_dir = project_dir / args.result_dir
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    # 准备数据
    try:
        (X_train, y_train, mask_train), \
        (X_val, y_val, mask_val), \
        (X_test, y_test, mask_test), \
        processor = prepare_ner_data(
            args.train_path,
            args.test_path,
            max_len=args.max_len
        )
    except FileNotFoundError as e:
        print(f"\n✗ 数据文件不存在: {e}")
        print("\n请先下载CoNLL-2003数据集")
        return

    # 保存处理器
    processor_path = model_dir / f'{args.model_type}_processor.pkl'
    processor.save_processor(processor_path)

    # 创建模型
    ner_model = TransformerNERModel(
        vocab_size=len(processor.word2idx),
        num_tags=len(processor.tag2idx),
        max_len=args.max_len,
        model_type=args.model_type
    )

    print(f"\n模型结构:")
    ner_model.summary()

    # 训练
    model_path = model_dir / f'{args.model_type}_model.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = ner_model.train(
        X_train, y_train, mask_train,
        X_val, y_val, mask_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    # 评估
    if X_test is not None:
        test_metrics = ner_model.evaluate(X_test, y_test, mask_test)
        print(f"\n测试集性能:")
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")

        # F1分数
        y_pred = ner_model.predict(X_test, mask_test)
        f1_scores = ner_model.calculate_f1_score(
            y_test, y_pred, mask_test, processor.idx2tag
        )
        print(f"\nF1分数:")
        for name, value in f1_scores.items():
            print(f"  {name}: {value:.4f}")

    print(f"\n✓ 训练完成！模型已保存: {model_path}")


if __name__ == '__main__':
    main()
