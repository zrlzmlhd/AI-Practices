"""
数据加载和预处理模块

本模块负责：
1. 加载IMDB数据集
2. 文本预处理（分词、去除停用词等）
3. 构建词汇表
4. 序列填充和截断
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
    加载IMDB数据集（使用Keras内置数据集）

    Args:
        max_words: 词汇表最大大小
        max_len: 序列最大长度
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    print("="*60)
    print("加载IMDB数据集")
    print("="*60)

    set_seed(random_state)

    # 加载数据
    print("\n1. 从Keras加载数据...")
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.imdb.load_data(
        num_words=max_words
    )

    print(f"   训练集大小: {len(X_train_full)}")
    print(f"   测试集大小: {len(X_test)}")

    # 填充序列
    print(f"\n2. 填充序列到固定长度 {max_len}...")
    X_train_full = pad_sequences(X_train_full, maxlen=max_len, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')

    # 划分训练集和验证集
    print(f"\n3. 划分训练集和验证集 (test_size={test_size})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_full
    )

    print(f"   训练集: {len(X_train)} 样本")
    print(f"   验证集: {len(X_val)} 样本")
    print(f"   测试集: {len(X_test)} 样本")

    # 数据统计
    print(f"\n4. 数据统计:")
    print(f"   序列形状: {X_train.shape}")
    print(f"   正面样本: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)")
    print(f"   负面样本: {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)")

    print("\n✓ 数据加载完成！")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_imdb_from_file(data_path, max_words=10000, max_len=200, test_size=0.2, random_state=42):
    """
    从文件加载IMDB数据集（如果有本地文件）

    Args:
        data_path: 数据文件路径
        max_words: 词汇表最大大小
        max_len: 序列最大长度
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    print("="*60)
    print("从文件加载IMDB数据集")
    print("="*60)

    set_seed(random_state)

    # 读取数据
    print(f"\n1. 读取数据文件: {data_path}")
    df = pd.read_csv(data_path)

    print(f"   数据大小: {len(df)}")
    print(f"   列名: {df.columns.tolist()}")

    # 假设数据格式: review, sentiment
    texts = df['review'].values
    labels = df['sentiment'].map({'positive': 1, 'negative': 0}).values

    # 文本预处理
    print(f"\n2. 文本预处理...")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y = labels

    print(f"   词汇表大小: {len(tokenizer.word_index)}")
    print(f"   使用词汇数: {max_words}")

    # 划分数据集
    print(f"\n3. 划分数据集...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_full
    )

    print(f"   训练集: {len(X_train)} 样本")
    print(f"   验证集: {len(X_val)} 样本")
    print(f"   测试集: {len(X_test)} 样本")

    print("\n✓ 数据加载完成！")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), tokenizer


def get_word_index():
    """
    获取IMDB数据集的词汇索引

    Returns:
        word_index: 词汇到索引的映射字典
    """
    word_index = keras.datasets.imdb.get_word_index()
    return word_index


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

    # 反转词汇索引
    reverse_word_index = {v: k for k, v in word_index.items()}

    # 解码（注意：索引偏移3，因为0,1,2是保留索引）
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
    """
    测试数据加载
    """
    print("="*60)
    print("数据加载模块测试")
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
    print("样本示例")
    print("="*60)

    word_index = get_word_index()

    for i in range(3):
        print(f"\n样本 {i+1}:")
        print(f"标签: {'正面' if y_train[i] == 1 else '负面'}")
        print(f"编码序列: {X_train[i][:20]}...")
        print(f"解码文本: {decode_review(X_train[i], word_index)[:200]}...")

    print("\n✓ 数据加载模块测试完成！")
