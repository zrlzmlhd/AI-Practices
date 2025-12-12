"""
数据加载和预处理模块

负责IMDB数据集的加载、预处理和序列填充
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed


def load_imdb_data(max_words=10000, max_len=200, test_size=0.2, random_state=42):
    """
    加载并预处理IMDB数据集

    Args:
        max_words: 词汇表最大大小
        max_len: 序列最大长度
        test_size: 验证集比例
        random_state: 随机种子

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    print("="*60)
    print("Loading IMDB Dataset")
    print("="*60)

    set_seed(random_state)

    # 加载数据
    print("\n[1/4] Loading data from Keras...")
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.imdb.load_data(
        num_words=max_words
    )

    print(f"  Training samples: {len(X_train_full)}")
    print(f"  Test samples: {len(X_test)}")

    # 填充序列
    print(f"\n[2/4] Padding sequences to length {max_len}...")
    X_train_full = pad_sequences(X_train_full, maxlen=max_len, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')

    # 划分训练集和验证集
    print(f"\n[3/4] Splitting train/validation set (test_size={test_size})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_full
    )

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # 数据统计
    print(f"\n[4/4] Data statistics:")
    print(f"  Sequence shape: {X_train.shape}")
    print(f"  Positive samples: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)")
    print(f"  Negative samples: {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)")

    print("\n✓ Data loading completed")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_imdb_from_file(data_path, max_words=10000, max_len=200, test_size=0.2, random_state=42):
    """
    从本地文件加载IMDB数据集

    Args:
        data_path: 数据文件路径
        max_words: 词汇表最大大小
        max_len: 序列最大长度
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), tokenizer
    """
    print("="*60)
    print("Loading IMDB Dataset from File")
    print("="*60)

    set_seed(random_state)

    # 读取数据
    print(f"\n[1/4] Reading data file: {data_path}")
    df = pd.read_csv(data_path)

    print(f"  Data size: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")

    # 假设数据格式: review, sentiment
    texts = df['review'].values
    labels = df['sentiment'].map({'positive': 1, 'negative': 0}).values

    # 文本预处理
    print(f"\n[2/4] Text preprocessing...")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y = labels

    print(f"  Vocabulary size: {len(tokenizer.word_index)}")
    print(f"  Using top {max_words} words")

    # 划分数据集
    print(f"\n[3/4] Splitting dataset...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_full
    )

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    print("\n✓ Data loading completed")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), tokenizer


def get_word_index():
    """获取IMDB数据集的词汇索引"""
    return keras.datasets.imdb.get_word_index()


def decode_review(encoded_review, word_index=None):
    """
    将编码的评论解码为文本

    Args:
        encoded_review: 编码的评论（整数序列）
        word_index: 词汇索引字典

    Returns:
        decoded_text: 解码后的文本
    """
    if word_index is None:
        word_index = get_word_index()

    # 反转词汇索引（注意：索引偏移3，因为0,1,2是保留索引）
    reverse_word_index = {v: k for k, v in word_index.items()}
    decoded_text = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i > 0])

    return decoded_text


def analyze_sequence_lengths(sequences):
    """
    分析序列长度分布

    Args:
        sequences: 序列列表

    Returns:
        stats: 统计信息字典
    """
    lengths = [len(seq) for seq in sequences]

    stats = {
        'min': np.min(lengths),
        'max': np.max(lengths),
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'percentile_25': np.percentile(lengths, 25),
        'percentile_50': np.percentile(lengths, 50),
        'percentile_75': np.percentile(lengths, 75),
        'percentile_90': np.percentile(lengths, 90),
        'percentile_95': np.percentile(lengths, 95),
    }

    return stats


if __name__ == '__main__':
    """测试数据加载功能"""
    print("="*60)
    print("Testing Data Loading Module")
    print("="*60)

    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_imdb_data(
        max_words=10000,
        max_len=200,
        test_size=0.2,
        random_state=42
    )

    # 显示样本
    print("\n" + "="*60)
    print("Sample Examples")
    print("="*60)

    word_index = get_word_index()

    for i in range(3):
        print(f"\nSample {i+1}:")
        print(f"Label: {'Positive' if y_train[i] == 1 else 'Negative'}")
        print(f"Encoded sequence: {X_train[i][:20]}...")
        print(f"Decoded text: {decode_review(X_train[i], word_index)[:200]}...")

    print("\n✓ Data loading module test completed")
