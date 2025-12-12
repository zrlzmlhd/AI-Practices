"""
命名实体识别模型评估脚本

提供模型评估功能,包括:
- 加载训练好的模型和数据处理器
- 在测试集上进行预测
- 计算详细的分类报告(precision, recall, F1)

使用方法:
    python src/evaluate.py --model_path models/transformer_crf_model.h5 \
                          --processor_path models/transformer_crf_processor.pkl
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from tensorflow import keras
try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except (ImportError, ModuleNotFoundError):
    HAS_TFA = False
    print("警告: tensorflow_addons未安装或不兼容,将无法加载CRF模型")
from sklearn.metrics import classification_report

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import prepare_ner_data


def parse_args():
    parser = argparse.ArgumentParser(description='评估NER模型性能')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型文件路径')
    parser.add_argument('--processor_path', type=str, required=True,
                       help='数据处理器文件路径')
    parser.add_argument('--test_path', type=str, default='data/test.txt',
                       help='测试数据文件路径')
    return parser.parse_args()


def main():
    args = parse_args()
    print("="*60)
    print("Transformer NER - 模型评估")
    print("="*60)

    # 加载数据处理器
    import pickle
    from src.data import NERDataProcessor

    processor = NERDataProcessor()
    try:
        processor.load_processor(args.processor_path)
    except FileNotFoundError:
        print(f"✗ 处理器文件不存在: {args.processor_path}")
        return

    # 加载测试数据
    test_sentences, test_tags = processor.load_conll_data(args.test_path)
    if not test_sentences:
        print("✗ 无法加载测试数据")
        return

    test_encoded_sentences, test_encoded_tags = processor.encode_sentences(
        test_sentences, test_tags
    )
    X_test, y_test, mask_test = processor.pad_sequences(
        test_encoded_sentences, test_encoded_tags
    )

    # 加载模型
    try:
        if HAS_TFA:
            model = keras.models.load_model(
                args.model_path,
                custom_objects={'CRF': tfa.layers.CRF}
            )
        else:
            model = keras.models.load_model(args.model_path)
        print(f"✓ 模型已加载: {args.model_path}")
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        return

    # 预测
    print("\n开始预测...")
    predictions = model.predict([X_test, mask_test])
    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=-1)

    # 计算评估指标
    y_true_flat = []
    y_pred_flat = []

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if mask_test[i][j] == 1:
                y_true_flat.append(processor.idx2tag[y_test[i][j]])
                y_pred_flat.append(processor.idx2tag[predictions[i][j]])

    print("\n" + "="*60)
    print("分类报告")
    print("="*60)
    print(classification_report(y_true_flat, y_pred_flat, digits=4))

    print("\n✓ 评估完成！")


if __name__ == '__main__':
    main()
