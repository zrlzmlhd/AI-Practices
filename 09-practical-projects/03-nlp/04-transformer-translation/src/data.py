"""
机器翻译数据处理模块

本模块负责：
1. 加载翻译数据集（使用公开数据集）
2. 构建源语言和目标语言词汇表
3. 序列编码和填充
4. 创建训练数据

使用数据集：可以使用WMT、IWSLT等公开数据集
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TranslationDataProcessor:
    """
    翻译数据处理器

    【是什么】：处理机器翻译数据的工具类
    【做什么】：
        - 加载平行语料
        - 构建双语词汇表
        - 序列编码和填充
    """

    def __init__(self, max_len=50, max_vocab_size=10000):
        """
        初始化数据处理器

        Args:
            max_len: 最大序列长度
            max_vocab_size: 最大词汇表大小
        """
        self.max_len = max_len
        self.max_vocab_size = max_vocab_size

        # 特殊token
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'  # Start of Sequence
        self.EOS_TOKEN = '<EOS>'  # End of Sequence

        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN
        ]

        # 词汇表
        self.src_word2idx = {}
        self.src_idx2word = {}
        self.tgt_word2idx = {}
        self.tgt_idx2word =

    def load_parallel_data(self, src_path, tgt_path, max_samples=None):
        """
        加载平行语料

        【格式】：
        源语言文件：每行一个句子
        目标语言文件：每行一个句子（对应翻译）

        Args:
            src_path: 源语言文件路径
            tgt_path: 目标语言文件路径
            max_samples: 最大样本数

        Returns:
            src_sentences: 源语言句子列表
            tgt_sentences: 目标语言句子列表
        """
        print(f"\n加载平行语料...")
        print(f"  源语言: {src_path}")
        print(f"  目标语言: {tgt_path}")

        src_sentences = []
        tgt_sentences = []

        try:
            with open(src_path, 'r', encoding='utf-8') as f_src, \
                 open(tgt_path, 'r', encoding='utf-8') as f_tgt:

                for src_line, tgt_line in zip(f_src, f_tgt):
                    src_line = src_line.strip()
                    tgt_line = tgt_line.strip()

                    if src_line and tgt_line:
                        # 简单的分词（按空格）
                        src_words = src_line.lower().split()
                        tgt_words = tgt_line.lower().split()

                        # 过滤过长的句子
                        if len(src_words) <= self.max_len and len(tgt_words) <= self.max_len:
                            src_sentences.append(src_words)
                            tgt_sentences.append(tgt_words)

                    if max_samples and len(src_sentences) >= max_samples:
                        break

        except FileNotFoundError as e:
            print(f"  ✗ 文件不存在: {e}")
            return [], []

        print(f"  加载句对数: {len(src_sentences)}")
        if src_sentences:
            print(f"  源语言平均长度: {np.mean([len(s) for s in src_sentences]):.1f}")
            print(f"  目标语言平均长度: {np.mean([len(s) for s in tgt_sentences]):.1f}")

        return src_sentences, tgt_sentences

    def build_vocab(self, sentences, vocab_type='src'):
        """
        构建词汇表

        Args:
            sentences: 句子列表
            vocab_type: 'src' 或 'tgt'
        """
        print(f"\n构建{vocab_type}词汇表...")

        # 统计词频
        word_freq = Counter()
        for sentence in sentences:
            word_freq.update(sentence)

        print(f"  总词数: {len(word_freq)}")

        # 选择最常见的词
        most_common = word_freq.most_common(self.max_vocab_size - len(self.special_tokens))

        # 构建映射
        word2idx = {}
        idx2word = {}

        for idx, token in enumerate(self.special_tokens):
            word2idx[token] = idx
            idx2word[idx] = token

        for idx, (word, _) in enumerate(most_common, start=len(self.special_tokens)):
            word2idx[word] = idx
            idx2word[idx] = word

        if vocab_type == 'src':
            self.src_word2idx = word2idx
            self.src_idx2word = idx2word
        else:
            self.tgt_word2idx = word2idx
            self.tgt_idx2word = idx2word

        print(f"  词汇表大小: {len(word2idx)}")

    def encode_sentences(self, sentences, vocab_type='src', add_sos_eos=False):
        """
        编码句子

        Args:
            sentences: 句子列表
            vocab_type: 'src' 或 'tgt'
            add_sos_eos: 是否添加SOS/EOS标记

        Returns:
            编码后的句子
        """
        word2idx = self.src_word2idx if vocab_type == 'src' else self.tgt_word2idx
        unk_idx = word2idx[self.UNK_TOKEN]
        sos_idx = word2idx[self.SOS_TOKEN]
        eos_idx = word2idx[self.EOS_TOKEN]

        encoded_sentences = []

        for sentence in sentences:
            encoded = [word2idx.get(word, unk_idx) for word in sentence]

            if add_sos_eos:
                encoded = [sos_idx] + encoded + [eos_idx]

            encoded_sentences.append(encoded)

        return encoded_sentences

    def pad_sequences(self, sequences):
        """填充序列"""
        padded = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding='post',
            truncating='post',
            value=self.src_word2idx[self.PAD_TOKEN]
        )
        return padded

    def save_processor(self, filepath):
        """保存数据处理器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'src_word2idx': self.src_word2idx,
                'src_idx2word': self.src_idx2word,
                'tgt_word2idx': self.tgt_word2idx,
                'tgt_idx2word': self.tgt_idx2word,
                'max_len': self.max_len
            }, f)
        print(f"✓ 数据处理器已保存: {filepath}")

    def load_processor(self, filepath):
        """加载数据处理器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.src_word2idx = data['src_word2idx']
            self.src_idx2word = data['src_idx2word']
            self.tgt_word2idx = data['tgt_word2idx']
            self.tgt_idx2word = data['tgt_idx2word']
            self.max_len = data['max_len']
        print(f"✓ 数据处理器已加载: {filepath}")


def prepare_translation_data(src_path, tgt_path, max_len=50, max_samples=None):
    """
    准备翻译数据

    Args:
        src_path: 源语言文件路径
        tgt_path: 目标语言文件路径
        max_len: 最大序列长度
        max_samples: 最大样本数

    Returns:
        训练集、验证集、处理器
    """
    print("="*60)
    print("机器翻译数据准备")
    print("="*60)

    processor = TranslationDataProcessor(max_len=max_len)

    # 加载数据
    src_sentences, tgt_sentences = processor.load_parallel_data(
        src_path, tgt_path, max_samples
    )

    if not src_sentences:
        raise FileNotFoundError("无法加载数据")

    # 构建词汇表
    processor.build_vocab(src_sentences, 'src')
    processor.build_vocab(tgt_sentences, 'tgt')

    # 编码
    src_encoded = processor.encode_sentences(src_sentences, 'src')
    tgt_encoded = processor.encode_sentences(tgt_sentences, 'tgt', add_sos_eos=True)

    # 填充
    src_padded = processor.pad_sequences(src_encoded)
    tgt_padded = processor.pad_sequences(tgt_encoded)

    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    src_train, src_val, tgt_train, tgt_val = train_test_split(
        src_padded, tgt_padded, test_size=0.1, random_state=42
    )

    print(f"\n训练集: {src_train.shape}")
    print(f"验证集: {src_val.shape}")

    return (src_train, tgt_train), (src_val, tgt_val), processor


if __name__ == '__main__':
    print("="*60)
    print("翻译数据处理模块测试")
    print("="*60)

    # 创建模拟数据
    src_data = "hello world\nhow are you\ngood morning\n"
    tgt_data = "你好 世界\n你 好 吗\n早上 好\n"

    with open('temp_src.txt', 'w') as f:
        f.write(src_data)
    with open('temp_tgt.txt', 'w') as f:
        f.write(tgt_data)

    try:
        (src_train, tgt_train), (src_val, tgt_val), processor = prepare_translation_data(
            'temp_src.txt', 'temp_tgt.txt', max_len=20
        )
        print("\n✓ 数据处理测试通过！")
    finally:
        import os
        os.remove('temp_src.txt')
        os.remove('temp_tgt.txt')
