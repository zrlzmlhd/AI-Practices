"""
NER数据处理模块

本模块负责：
1. 加载CoNLL-2003数据集
2. 构建词汇表和标签映射
3. 序列填充和编码
4. 数据集划分

使用CoNLL-2003数据集（公开标准NER数据集）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


class NERDataProcessor:
    """
    NER数据处理器

    【是什么】：处理命名实体识别数据的工具类
    【做什么】：
        - 加载CoNLL格式数据
        - 构建词汇表和标签映射
        - 序列编码和填充
    【为什么】：
        - NER是序列标注任务
        - 需要对齐词和标签
        - 需要特殊的数据处理
    """

    def __init__(self, max_len=128, max_vocab_size=10000):
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
        self.CLS_TOKEN = '<CLS>'
        self.SEP_TOKEN = '<SEP>'

        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.CLS_TOKEN,
            self.SEP_TOKEN
        ]

        # 词汇表和标签映射
        self.word2idx = {}
        self.idx2word = {}
        self.tag2idx = {}
        self.idx2tag = {}

        # CoNLL-2003标签体系
        # 【是什么】：BIO标注格式
        # 【B】：Begin（实体开始）
        # 【I】：Inside（实体内部）
        # 【O】：Outside（非实体）
        # 【实体类型】：PER（人名）、ORG（组织）、LOC（地点）、MISC（其他）
        self.default_tags = [
            'O',           # 非实体
            'B-PER',       # 人名-开始
            'I-PER',       # 人名-内部
            'B-ORG',       # 组织-开始
            'I-ORG',       # 组织-内部
            'B-LOC',       # 地点-开始
            'I-LOC',       # 地点-内部
            'B-MISC',      # 其他-开始
            'I-MISC'       # 其他-内部
        ]

    def load_conll_data(self, file_path):
        """
        加载CoNLL格式数据

        【CoNLL格式】：
        每行一个词和标签，用空格或制表符分隔
        句子之间用空行分隔

        示例：
        EU    B-ORG
        rejects    O
        German    B-MISC
        call    O

        British    B-MISC
        ...

        Args:
            file_path: 数据文件路径

        Returns:
            sentences: 句子列表
            tags: 标签列表
        """
        print(f"\n加载CoNLL数据: {file_path}")

        sentences = []
        tags = []
        current_sentence = []
        current_tags = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    # 空行表示句子结束
                    if not line:
                        if current_sentence:
                            sentences.append(current_sentence)
                            tags.append(current_tags)
                            current_sentence = []
                            current_tags = []
                        continue

                    # 跳过注释行
                    if line.startswith('-DOCSTART-'):
                        continue

                    # 解析词和标签
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        tag = parts[-1]  # 最后一列是NER标签

                        current_sentence.append(word)
                        current_tags.append(tag)

                # 处理最后一个句子
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)

        except FileNotFoundError:
            print(f"  ✗ 文件不存在: {file_path}")
            return [], []

        print(f"  加载句子数: {len(sentences)}")
        if sentences:
            print(f"  平均句子长度: {np.mean([len(s) for s in sentences]):.1f}")
            print(f"  最长句子: {max([len(s) for s in sentences])}")

        return sentences, tags

    def build_vocab(self, sentences):
        """
        构建词汇表

        Args:
            sentences: 句子列表
        """
        print("\n构建词汇表...")

        # 统计词频
        word_freq = Counter()
        for sentence in sentences:
            word_freq.update(sentence)

        print(f"  总词数: {len(word_freq)}")

        # 选择最常见的词
        most_common = word_freq.most_common(self.max_vocab_size - len(self.special_tokens))

        # 构建映射
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

        for idx, (word, _) in enumerate(most_common, start=len(self.special_tokens)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"  词汇表大小: {len(self.word2idx)}")

        # 计算覆盖率
        total_words = sum(word_freq.values())
        covered_words = sum(freq for word, freq in word_freq.items() if word in self.word2idx)
        coverage = covered_words / total_words
        print(f"  覆盖率: {coverage:.2%}")

    def build_tag_mapping(self, tags_list):
        """
        构建标签映射

        Args:
            tags_list: 标签列表
        """
        print("\n构建标签映射...")

        # 收集所有标签
        all_tags = set()
        for tags in tags_list:
            all_tags.update(tags)

        # 使用默认标签顺序（如果存在）
        unique_tags = []
        for tag in self.default_tags:
            if tag in all_tags:
                unique_tags.append(tag)

        # 添加其他标签
        for tag in sorted(all_tags):
            if tag not in unique_tags:
                unique_tags.append(tag)

        # 构建映射
        for idx, tag in enumerate(unique_tags):
            self.tag2idx[tag] = idx
            self.idx2tag[idx] = tag

        print(f"  标签数量: {len(self.tag2idx)}")
        print(f"  标签列表: {list(self.tag2idx.keys())}")

    def encode_sentences(self, sentences, tags_list=None):
        """
        编码句子和标签

        Args:
            sentences: 句子列表
            tags_list: 标签列表（可选）

        Returns:
            encoded_sentences: 编码后的句子
            encoded_tags: 编码后的标签（如果提供）
        """
        print("\n编码句子...")

        encoded_sentences = []
        encoded_tags = [] if tags_list is not None else None

        for i, sentence in enumerate(sentences):
            # 编码词
            encoded_sentence = [
                self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])
                for word in sentence
            ]
            encoded_sentences.append(encoded_sentence)

            # 编码标签
            if tags_list is not None:
                tags = tags_list[i]
                encoded_tag = [
                    self.tag2idx.get(tag, 0)  # 默认为'O'
                    for tag in tags
                ]
                encoded_tags.append(encoded_tag)

        print(f"  编码句子数: {len(encoded_sentences)}")

        return encoded_sentences, encoded_tags

    def pad_sequences(self, sequences, tags=None):
        """
        填充序列

        【重要】：标签也需要填充，且填充值通常是0（对应'O'标签）

        Args:
            sequences: 序列列表
            tags: 标签列表（可选）

        Returns:
            padded_sequences: 填充后的序列
            padded_tags: 填充后的标签
            mask: 填充掩码
        """
        print("\n填充序列...")

        # 填充词序列
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding='post',
            truncating='post',
            value=self.word2idx[self.PAD_TOKEN]
        )

        # 创建掩码（1表示真实词，0表示填充）
        mask = (padded_sequences != self.word2idx[self.PAD_TOKEN]).astype(np.float32)

        # 填充标签
        padded_tags = None
        if tags is not None:
            padded_tags = pad_sequences(
                tags,
                maxlen=self.max_len,
                padding='post',
                truncating='post',
                value=0  # 填充标签为'O'
            )

        print(f"  填充后形状: {padded_sequences.shape}")

        return padded_sequences, padded_tags, mask

    def decode_predictions(self, predictions, mask=None):
        """
        解码预测结果

        Args:
            predictions: 预测的标签ID
            mask: 填充掩码

        Returns:
            decoded_tags: 解码后的标签
        """
        decoded_tags = []

        for i, pred_seq in enumerate(predictions):
            decoded_seq = []
            for j, tag_id in enumerate(pred_seq):
                # 跳过填充位置
                if mask is not None and mask[i][j] == 0:
                    continue

                tag = self.idx2tag.get(tag_id, 'O')
                decoded_seq.append(tag)

            decoded_tags.append(decoded_seq)

        return decoded_tags

    def save_processor(self, filepath):
        """保存数据处理器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'tag2idx': self.tag2idx,
                'idx2tag': self.idx2tag,
                'max_len': self.max_len
            }, f)
        print(f"✓ 数据处理器已保存: {filepath}")

    def load_processor(self, filepath):
        """加载数据处理器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.tag2idx = data['tag2idx']
            self.idx2tag = data['idx2tag']
            self.max_len = data['max_len']
        print(f"✓ 数据处理器已加载: {filepath}")


def prepare_ner_data(train_path, test_path=None, max_len=128, max_vocab_size=10000):
    """
    准备NER数据

    Args:
        train_path: 训练数据路径
        test_path: 测试数据路径
        max_len: 最大序列长度
        max_vocab_size: 最大词汇表大小

    Returns:
        训练集、验证集、测试集、处理器
    """
    print("="*60)
    print("NER数据准备")
    print("="*60)

    # 创建处理器
    processor = NERDataProcessor(max_len=max_len, max_vocab_size=max_vocab_size)

    # 加载训练数据
    train_sentences, train_tags = processor.load_conll_data(train_path)

    if not train_sentences:
        raise FileNotFoundError(f"无法加载训练数据: {train_path}")

    # 构建词汇表和标签映射
    processor.build_vocab(train_sentences)
    processor.build_tag_mapping(train_tags)

    # 编码训练数据
    train_encoded_sentences, train_encoded_tags = processor.encode_sentences(
        train_sentences, train_tags
    )

    # 填充训练数据
    X_train, y_train, mask_train = processor.pad_sequences(
        train_encoded_sentences, train_encoded_tags
    )

    # 划分验证集（从训练集中分出10%）
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val, mask_train, mask_val = train_test_split(
        X_train, y_train, mask_train,
        test_size=0.1,
        random_state=42
    )

    print(f"\n训练集: {X_train.shape}")
    print(f"验证集: {X_val.shape}")

    # 加载测试数据（如果提供）
    X_test, y_test, mask_test = None, None, None
    if test_path:
        test_sentences, test_tags = processor.load_conll_data(test_path)
        if test_sentences:
            test_encoded_sentences, test_encoded_tags = processor.encode_sentences(
                test_sentences, test_tags
            )
            X_test, y_test, mask_test = processor.pad_sequences(
                test_encoded_sentences, test_encoded_tags
            )
            print(f"测试集: {X_test.shape}")

    print("\n数据准备完成！")

    return (X_train, y_train, mask_train), \
           (X_val, y_val, mask_val), \
           (X_test, y_test, mask_test), \
           processor


if __name__ == '__main__':
    """测试数据处理"""
    print("="*60)
    print("NER数据处理模块测试")
    print("="*60)

    # 创建模拟CoNLL数据
    print("\n创建模拟数据...")
    mock_data = """EU B-ORG
rejects O
German B-MISC
call O
to O
boycott O
British B-MISC
lamb O
.  O

Peter B-PER
Blackburn I-PER
"""

    # 保存临时文件
    temp_path = 'temp_conll_data.txt'
    with open(temp_path, 'w') as f:
        f.write(mock_data)

    try:
        # 测试数据处理
        (X_train, y_train, mask_train), \
        (X_val, y_val, mask_val), \
        (X_test, y_test, mask_test), \
        processor = prepare_ner_data(temp_path, max_len=20)

        print("\n✓ 数据处理测试通过！")

        # 测试解码
        print("\n测试解码...")
        decoded = processor.decode_predictions(y_val[:1], mask_val[:1])
        print(f"  解码标签: {decoded[0][:10]}")

    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n✓ 所有测试通过！")
