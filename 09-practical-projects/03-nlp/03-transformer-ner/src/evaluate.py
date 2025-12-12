"""
NER模型评估脚本

使用方法:
    python src/evaluate.py --model_path models/transformer_crf_model.h5 --processor_path models/transformer_crf_processor.pkl
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.metrics import classification_report

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import prepare_ner_data


def parse_args():
    parser = argparse.ArgumentParser(description='评估NER模型')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--processor_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, default='data/test.txt')
    return parser.parse_args()


def main():
    args = parse_args()
    print("="*60)
    print("Transformer NER - 模型评估")
    print("="*60)

    # 加载处理器
    import pickle
    with open(args.processor_path, 'rb') as f:
        processor_data = pickle.load(f)
    print(f"✓ 处理器已加载")

    # 加载数据
    from src.data import NERDataProcessor
    processor = NERDataProcessor()
    processor.load_processor(args.processor_path)

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
    model = keras.models.load_model(
        args.model_path,
        custom_objects={'CRF': tfa.layers.CRF}
    )
    print(f"✓ 模型已加载")

    # 预测
    predictions = model.predict([X_test, mask_test])
    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=-1)

    # 评估
    y_true_flat = []
    y_pred_flat = []

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if mask_test[i][j] == 1:
                y_true_flat.append(processor.idx2tag[y_test[i][j]])
                y_pred_flat.append(processor.idx2tag[predictions[i][j]])

    print("\n分类报告:")
    print(classification_report(y_true_flat, y_pred_flat, digits=4))

    print("\n✓ 评估完成！")


if __name__ == '__main__':
    main()
