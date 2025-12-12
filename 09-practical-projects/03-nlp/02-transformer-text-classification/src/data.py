"""
数据加载和预处理模块

本模块负责：
1. 加载IMDB数据集
2. 文本预处理
3. 构建词汇表
4. 序列填充
5. 数据集划分

每个步骤都有详细的注释说明。
"""

import os
import re
import numpy as np
from pathlib import Path
from collections import Counter
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class TextPreprocessor:
    """
    文本预处理器

    【是什么】：文本清洗和标准化工具
    【做什么】：将原始文本转换为干净的文本
    【为什么】：
        - 去除噪声（HTML标签、特殊字符）
        - 统一格式（小写、空格）
        - 提高模型性能
    """

    def __init__(self, lower=True, remove_html=True, remove_special_chars=True):
        """
        初始化预处理器

        Args:
            lower: 是否转换为小写
            remove_html: 是否去除HTML标签
            remove_special_chars: 是否去除特殊字符
        """
        self.lower = lower
        self.remove_html = remove_html
        self.remove_special_chars = remove_special_chars

    def clean_text(self, text):
        """
        清洗文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        # ============================================
        # 步骤1: 去除HTML标签
        # ============================================
        # 【是什么】：<br />, <p>, </div> 等
        # 【为什么】：IMDB数据包含HTML标签
        if self.remove_html:
            text = re.sub(r'<[^>]+>', ' ', text)

        # ============================================
        # 步骤2: 去除特殊字符
        # ============================================
        # 【是什么】：保留字母、数字、基本标点
        # 【为什么】：
        #   - 减少词汇表大小
        #   - 去除噪声
        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\'\-]', ' ', text)

        # ============================================
        # 步骤3: 处理多余空格
        # ============================================
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # ============================================
        # 步骤4: 转换为小写
        # ============================================
        # 【为什么】：
        #   - "Good"和"good"应该是同一个词
        #   - 减少词汇表大小
        if self.lower:
            text = text.lower()

        return text

    def __call__(self, text):
        """使预处理器可调用"""
        return self.clean_text(text)


class Vocabulary:
    """
    词汇表

    【是什么】：词和ID的映射
    【做什么】：
        - 词 -> ID（编码）
        - ID -> 词（解码）
    【为什么】：
        - 模型只能处理数字
        - 需要将词转换为ID
    """

    def __init__(self, max_vocab_size=10000, min_freq=2):
        """
        初始化词汇表

        Args:
            max_vocab_size: 最大词汇表大小
            min_freq: 最小词频（低于此频率的词被忽略）
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        # 特殊token
        # 【是什么】：特殊用途的token
        # 【为什么】：
        #   - <PAD>: 填充短序列
        #   - <UNK>: 未知词（不在词汇表中的词）
        #   - <CLS>: 分类token（可选，用于BERT风格）
        #   - <SEP>: 分隔token（可选）
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.CLS_TOKEN = '<CLS>'
        self.SEP_TOKEN = '<SEP>'

        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.CLS_TOKEN,
            self.SEP_TOKEN
        ]

        # 词汇表
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

    def build_vocab(self, texts):
        """
        构建词汇表

        Args:
            texts: 文本列表
        """
        print("\n构建词汇表...")

        # ============================================
        # 步骤1: 统计词频
        # ============================================
        # 【做什么】：计算每个词出现的次数
        # 【为什么】：
        #   - 保留高频词
        #   - 过滤低频词（可能是噪声）
        for text in texts:
            words = text.split()
            self.word_freq.update(words)

        print(f"  总词数: {len(self.word_freq)}")

        # ============================================
        # 步骤2: 过滤低频词
        # ============================================
        # 【是什么】：去除出现次数 < min_freq 的词
        # 【为什么】：
        #   - 低频词可能是拼写错误
        #   - 减少词汇表大小
        #   - 提高泛化能力
        filtered_words = [
            word for word, freq in self.word_freq.items()
            if freq >= self.min_freq
        ]
        print(f"  过滤后: {len(filtered_words)} (min_freq={self.min_freq})")

        # ============================================
        # 步骤3: 选择最常见的词
        # ============================================
        # 【是什么】：按频率排序，取前max_vocab_size个
        # 【为什么】：
        #   - 限制词汇表大小
        #   - 高频词包含更多信息
        most_common = self.word_freq.most_common(self.max_vocab_size - len(self.special_tokens))
        vocab_words = [word for word, _ in most_common]

        # ============================================
        # 步骤4: 构建映射
        # ============================================
        # 【做什么】：创建词<->ID的双向映射

        # 添加特殊token
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

        # 添加普通词
        for idx, word in enumerate(vocab_words, start=len(self.special_tokens)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"  最终词汇表大小: {len(self.word2idx)}")
        print(f"  覆盖率: {self._calculate_coverage(texts):.2%}")

    def _calculate_coverage(self, texts):
        """
        计算词汇表覆盖率

        Args:
            texts: 文本列表

        Returns:
            覆盖率（0-1之间）
        """
        total_words = 0
        covered_words = 0

        for text in texts:
            words = text.split()
            total_words += len(words)
            covered_words += sum(1 for word in words if word in self.word2idx)

        return covered_words / total_words if total_words > 0 else 0

    def encode(self, text, add_cls=False, add_sep=False):
        """
        将文本编码为ID序列

        Args:
            text: 文本字符串
            add_cls: 是否添加[CLS] token
            add_sep: 是否添加[SEP] token

        Returns:
            ID列表
        """
        words = text.split()

        # 转换为ID
        # 【是什么】：查表操作
        # 【为什么】：不在词汇表中的词用<UNK>代替
        ids = [
            self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])
            for word in words
        ]

        # 添加特殊token
        if add_cls:
            ids = [self.word2idx[self.CLS_TOKEN]] + ids
        if add_sep:
            ids = ids + [self.word2idx[self.SEP_TOKEN]]

        return ids

    def decode(self, ids):
        """
        将ID序列解码为文本

        Args:
            ids: ID列表

        Returns:
            文本字符串
        """
        words = [
            self.idx2word.get(idx, self.UNK_TOKEN)
            for idx in ids
        ]
        return ' '.join(words)

    def save(self, filepath):
        """保存词汇表"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'max_vocab_size': self.max_vocab_size,
                'min_freq': self.min_freq
            }, f)
        print(f"✓ 词汇表已保存: {filepath}")

    def load(self, filepath):
        """加载词汇表"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_freq = data['word_freq']
            self.max_vocab_size = data['max_vocab_size']
            self.min_freq = data['min_freq']
        print(f"✓ 词汇表已加载: {filepath}")

    def __len__(self):
        """返回词汇表大小"""
        return len(self.word2idx)


def load_imdb_data(data_dir, max_samples=None):
    """
    加载IMDB数据集

    Args:
        data_dir: 数据目录
        max_samples: 最大样本数（用于快速测试）

    Returns:
        (texts, labels) 元组
    """
    print("\n加载IMDB数据集...")

    data_dir = Path(data_dir)
    texts = []
    labels = []

    # ============================================
    # 加载正面评论
    # ============================================
    pos_dir = data_dir / 'pos'
    if pos_dir.exists():
        pos_files = list(pos_dir.glob('*.txt'))
        if max_samples:
            pos_files = pos_files[:max_samples // 2]

        for file_path in pos_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1)  # 正面=1

    # ============================================
    # 加载负面评论
    # ============================================
    neg_dir = data_dir / 'neg'
    if neg_dir.exists():
        neg_files = list(neg_dir.glob('*.txt'))
        if max_samples:
            neg_files = neg_files[:max_samples // 2]

        for file_path in neg_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(0)  # 负面=0

    print(f"  加载样本数: {len(texts)}")
    print(f"  正面样本: {sum(labels)}")
    print(f"  负面样本: {len(labels) - sum(labels)}")

    return texts, labels


def prepare_data(data_dir='data/aclImdb',
                 max_vocab_size=10000,
                 max_len=256,
                 test_size=0.2,
                 val_size=0.1,
                 random_state=42,
                 max_samples=None):
    """
    准备数据

    Args:
        data_dir: 数据目录
        max_vocab_size: 最大词汇表大小
        max_len: 最大序列长度
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        max_samples: 最大样本数

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab
    """
    print("="*60)
    print("数据准备")
    print("="*60)

    # ============================================
    # 步骤1: 加载数据
    # ============================================
    train_dir = Path(data_dir) / 'train'
    test_dir = Path(data_dir) / 'test'

    # 加载训练集
    train_texts, train_labels = load_imdb_data(train_dir, max_samples)

    # 加载测试集
    test_texts, test_labels = load_imdb_data(test_dir, max_samples)

    # ============================================
    # 步骤2: 文本预处理
    # ============================================
    print("\n文本预处理...")
    preprocessor = TextPreprocessor()

    train_texts = [preprocessor(text) for text in train_texts]
    test_texts = [preprocessor(text) for text in test_texts]

    print(f"  训练集样本: {len(train_texts)}")
    print(f"  测试集样本: {len(test_texts)}")

    # ============================================
    # 步骤3: 构建词汇表
    # ============================================
    vocab = Vocabulary(max_vocab_size=max_vocab_size)
    vocab.build_vocab(train_texts)

    # ============================================
    # 步骤4: 文本编码
    # ============================================
    print("\n文本编码...")

    # 【是什么】：将文本转换为ID序列
    # 【为什么】：模型只能处理数字
    train_sequences = [vocab.encode(text) for text in train_texts]
    test_sequences = [vocab.encode(text) for text in test_texts]

    # ============================================
    # 步骤5: 序列填充
    # ============================================
    print("\n序列填充...")

    # 【是什么】：将序列填充到相同长度
    # 【为什么】：
    #   - 批处理需要相同长度
    #   - 短序列用<PAD>填充
    #   - 长序列被截断

    # 【padding='post'】：在后面填充
    # 【truncating='post'】：从后面截断
    X_train = pad_sequences(
        train_sequences,
        maxlen=max_len,
        padding='post',
        truncating='post',
        value=vocab.word2idx[vocab.PAD_TOKEN]
    )

    X_test = pad_sequences(
        test_sequences,
        maxlen=max_len,
        padding='post',
        truncating='post',
        value=vocab.word2idx[vocab.PAD_TOKEN]
    )

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    print(f"  训练集形状: {X_train.shape}")
    print(f"  测试集形状: {X_test.shape}")

    # ============================================
    # 步骤6: 划分验证集
    # ============================================
    print("\n划分验证集...")

    # 【是什么】：从训练集中分出验证集
    # 【为什么】：
    #   - 用于调参和早停
    #   - 避免在测试集上调参
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train  # 保持类别比例
    )

    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")

    # ============================================
    # 数据统计
    # ============================================
    print("\n数据统计:")
    print(f"  词汇表大小: {len(vocab)}")
    print(f"  最大序列长度: {max_len}")
    print(f"  训练集正负比: {y_train.sum()}/{len(y_train)-y_train.sum()}")
    print(f"  验证集正负比: {y_val.sum()}/{len(y_val)-y_val.sum()}")
    print(f"  测试集正负比: {y_test.sum()}/{len(y_test)-y_test.sum()}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab


if __name__ == '__main__':
    """
    测试数据处理
    """
    print("="*60)
    print("数据处理模块测试")
    print("="*60)

    # 测试文本预处理
    print("\n测试文本预处理...")
    preprocessor = TextPreprocessor()
    text = "This is a <b>GREAT</b> movie!!! I really enjoyed it."
    cleaned = preprocessor(text)
    print(f"原始文本: {text}")
    print(f"清洗后: {cleaned}")

    # 测试词汇表
    print("\n测试词汇表...")
    texts = [
        "this is a good movie",
        "this is a bad movie",
        "i love this movie",
        "i hate this movie"
    ]
    vocab = Vocabulary(max_vocab_size=20, min_freq=1)
    vocab.build_vocab(texts)

    print(f"\n词汇表大小: {len(vocab)}")
    print(f"词汇表: {list(vocab.word2idx.keys())[:10]}")

    # 测试编码
    text = "this is a good movie"
    ids = vocab.encode(text)
    decoded = vocab.decode(ids)
    print(f"\n原始文本: {text}")
    print(f"编码: {ids}")
    print(f"解码: {decoded}")

    print("\n✓ 所有测试通过！")
