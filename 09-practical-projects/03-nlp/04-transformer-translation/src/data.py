"""
机器翻译数据处理模块

功能：
1. 加载双语平行语料
2. 构建源语言和目标语言词汇表
3. 文本序列编码与填充
4. 数据集划分与预处理

支持数据集：WMT、IWSLT、Tatoeba等公开翻译数据集
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional, Union
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TranslationDataProcessor:
    """
    双语翻译数据处理器

    负责机器翻译任务的数据预处理流程，包括：
    - 平行语料的加载与清洗
    - 双语词汇表的构建与维护
    - 文本序列的编码、填充和批处理

    特殊标记说明：
        <PAD>: 填充标记，用于对齐序列长度
        <UNK>: 未知词标记，处理词汇表外的词
        <SOS>: 序列开始标记，标识翻译起点
        <EOS>: 序列结束标记，标识翻译终点
    """

    def __init__(self, max_len: int = 50, max_vocab_size: int = 10000):
        """
        初始化数据处理器

        Args:
            max_len: 序列最大长度，超过该长度的句子将被截断
            max_vocab_size: 词汇表最大容量，保留最高频词汇
        """
        self.max_len = max_len
        self.max_vocab_size = max_vocab_size

        # 定义特殊标记
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'

        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN
        ]

        # 初始化词汇表
        self.src_word2idx: Dict[str, int] = {}
        self.src_idx2word: Dict[int, str] = {}
        self.tgt_word2idx: Dict[str, int] = {}
        self.tgt_idx2word: Dict[int, str] = {}

    def load_parallel_data(
        self,
        src_path: Union[str, Path],
        tgt_path: Union[str, Path],
        max_samples: Optional[int] = None
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        加载双语平行语料

        平行语料格式要求：
        - 源语言文件和目标语言文件行数相同
        - 每行一个句子，对应行互为翻译
        - UTF-8编码

        数据清洗策略：
        - 过滤空行
        - 移除过长句子（超过max_len）
        - 转换为小写（可选）
        - 基于空格的简单分词

        Args:
            src_path: 源语言文件路径
            tgt_path: 目标语言文件路径
            max_samples: 限制加载的样本数，用于快速实验

        Returns:
            (源语言句子列表, 目标语言句子列表)
            每个句子表示为词的列表

        Raises:
            FileNotFoundError: 当文件不存在时抛出
        """
        print(f"\n{'='*60}")
        print("加载平行语料")
        print(f"{'='*60}")
        print(f"源语言: {src_path}")
        print(f"目标语言: {tgt_path}")

        src_sentences = []
        tgt_sentences = []

        try:
            with open(src_path, 'r', encoding='utf-8') as f_src, \
                 open(tgt_path, 'r', encoding='utf-8') as f_tgt:

                for src_line, tgt_line in zip(f_src, f_tgt):
                    src_line = src_line.strip()
                    tgt_line = tgt_line.strip()

                    if not src_line or not tgt_line:
                        continue

                    # 简单分词（基于空格）
                    # 注意：对于中文等语言，建议使用专门的分词工具
                    src_words = src_line.lower().split()
                    tgt_words = tgt_line.lower().split()

                    # 过滤过长句子
                    if len(src_words) <= self.max_len and len(tgt_words) <= self.max_len:
                        src_sentences.append(src_words)
                        tgt_sentences.append(tgt_words)

                    if max_samples and len(src_sentences) >= max_samples:
                        break

        except FileNotFoundError as e:
            print(f"错误: 文件不存在 - {e}")
            raise

        # 打印统计信息
        print(f"\n加载完成:")
        print(f"  句对数量: {len(src_sentences)}")
        if src_sentences:
            src_avg_len = np.mean([len(s) for s in src_sentences])
            tgt_avg_len = np.mean([len(s) for s in tgt_sentences])
            print(f"  源语言平均长度: {src_avg_len:.1f} 词")
            print(f"  目标语言平均长度: {tgt_avg_len:.1f} 词")

        return src_sentences, tgt_sentences

    def build_vocab(self, sentences: List[List[str]], vocab_type: str = 'src'):
        """
        构建词汇表

        采用词频统计策略，保留高频词汇：
        1. 统计所有词的出现频率
        2. 按频率降序排列
        3. 选取前max_vocab_size个词
        4. 添加特殊标记

        Args:
            sentences: 句子列表，每个句子是词的列表
            vocab_type: 词汇表类型，'src'表示源语言，'tgt'表示目标语言
        """
        print(f"\n构建{vocab_type}词汇表...")

        # 统计词频
        word_freq = Counter()
        for sentence in sentences:
            word_freq.update(sentence)

        print(f"  总词数: {len(word_freq)}")

        # 选择最高频的词
        vocab_size = self.max_vocab_size - len(self.special_tokens)
        most_common = word_freq.most_common(vocab_size)

        # 构建双向映射
        word2idx = {}
        idx2word = {}

        # 首先添加特殊标记
        for idx, token in enumerate(self.special_tokens):
            word2idx[token] = idx
            idx2word[idx] = token

        # 添加高频词
        offset = len(self.special_tokens)
        for idx, (word, freq) in enumerate(most_common, start=offset):
            word2idx[word] = idx
            idx2word[idx] = word

        # 保存词汇表
        if vocab_type == 'src':
            self.src_word2idx = word2idx
            self.src_idx2word = idx2word
        else:
            self.tgt_word2idx = word2idx
            self.tgt_idx2word = idx2word

        print(f"  词汇表大小: {len(word2idx)}")
        print(f"  覆盖率: {sum([freq for _, freq in most_common]) / sum(word_freq.values()) * 100:.2f}%")

    def encode_sentences(
        self,
        sentences: List[List[str]],
        vocab_type: str = 'src',
        add_sos_eos: bool = False
    ) -> List[List[int]]:
        """
        将句子编码为整数序列

        编码规则：
        - 词汇表内的词映射到对应ID
        - 词汇表外的词映射到<UNK>的ID
        - 可选地添加<SOS>和<EOS>标记

        Args:
            sentences: 待编码的句子列表
            vocab_type: 使用的词汇表类型
            add_sos_eos: 是否添加序列起止标记

        Returns:
            编码后的整数序列列表
        """
        word2idx = self.src_word2idx if vocab_type == 'src' else self.tgt_word2idx
        unk_idx = word2idx[self.UNK_TOKEN]
        sos_idx = word2idx[self.SOS_TOKEN]
        eos_idx = word2idx[self.EOS_TOKEN]

        encoded_sentences = []

        for sentence in sentences:
            # 词到ID的转换
            encoded = [word2idx.get(word, unk_idx) for word in sentence]

            # 添加起止标记（通常用于目标语言）
            if add_sos_eos:
                encoded = [sos_idx] + encoded + [eos_idx]

            encoded_sentences.append(encoded)

        return encoded_sentences

    def pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        """
        填充序列到统一长度

        填充策略：
        - 序列长度统一为max_len
        - 在序列末尾填充<PAD>标记
        - 超长序列从末尾截断

        Args:
            sequences: 整数序列列表

        Returns:
            填充后的numpy数组，形状为(num_sequences, max_len)
        """
        padded = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding='post',
            truncating='post',
            value=self.src_word2idx[self.PAD_TOKEN]
        )
        return padded

    def save_processor(self, filepath: Union[str, Path]):
        """
        保存数据处理器状态

        保存内容包括：
        - 双语词汇表
        - 配置参数

        Args:
            filepath: 保存路径
        """
        save_dict = {
            'src_word2idx': self.src_word2idx,
            'src_idx2word': self.src_idx2word,
            'tgt_word2idx': self.tgt_word2idx,
            'tgt_idx2word': self.tgt_idx2word,
            'max_len': self.max_len,
            'max_vocab_size': self.max_vocab_size
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"数据处理器已保存: {filepath}")

    def load_processor(self, filepath: Union[str, Path]):
        """
        加载数据处理器状态

        Args:
            filepath: 保存路径
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.src_word2idx = data['src_word2idx']
        self.src_idx2word = data['src_idx2word']
        self.tgt_word2idx = data['tgt_word2idx']
        self.tgt_idx2word = data['tgt_idx2word']
        self.max_len = data['max_len']
        self.max_vocab_size = data.get('max_vocab_size', 10000)

        print(f"数据处理器已加载: {filepath}")


def prepare_translation_data(
    src_path: Union[str, Path],
    tgt_path: Union[str, Path],
    max_len: int = 50,
    max_samples: Optional[int] = None,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray],
           TranslationDataProcessor]:
    """
    完整的翻译数据准备流程

    流程步骤：
    1. 加载双语平行语料
    2. 构建源语言和目标语言词汇表
    3. 编码文本序列
    4. 填充序列到统一长度
    5. 划分训练集和验证集

    Args:
        src_path: 源语言文件路径
        tgt_path: 目标语言文件路径
        max_len: 最大序列长度
        max_samples: 限制样本数
        test_size: 验证集比例
        random_state: 随机种子

    Returns:
        ((训练集源语言, 训练集目标语言),
         (验证集源语言, 验证集目标语言),
         数据处理器实例)

    Raises:
        FileNotFoundError: 当数据文件不存在时抛出
    """
    print(f"\n{'='*60}")
    print("机器翻译数据准备")
    print(f"{'='*60}")

    # 初始化处理器
    processor = TranslationDataProcessor(max_len=max_len)

    # 加载数据
    src_sentences, tgt_sentences = processor.load_parallel_data(
        src_path, tgt_path, max_samples
    )

    if not src_sentences:
        raise FileNotFoundError("无法加载数据文件")

    # 构建词汇表
    processor.build_vocab(src_sentences, 'src')
    processor.build_vocab(tgt_sentences, 'tgt')

    # 编码句子
    # 注意：目标语言需要添加<SOS>和<EOS>标记用于训练
    src_encoded = processor.encode_sentences(src_sentences, 'src', add_sos_eos=False)
    tgt_encoded = processor.encode_sentences(tgt_sentences, 'tgt', add_sos_eos=True)

    # 填充序列
    src_padded = processor.pad_sequences(src_encoded)
    tgt_padded = processor.pad_sequences(tgt_encoded)

    # 划分训练集和验证集
    src_train, src_val, tgt_train, tgt_val = train_test_split(
        src_padded, tgt_padded,
        test_size=test_size,
        random_state=random_state
    )

    print(f"\n数据划分完成:")
    print(f"  训练集: {src_train.shape}")
    print(f"  验证集: {src_val.shape}")

    return (src_train, tgt_train), (src_val, tgt_val), processor


if __name__ == '__main__':
    """
    模块测试代码
    创建临时测试数据，验证数据处理流程
    """
    print(f"\n{'='*60}")
    print("翻译数据处理模块测试")
    print(f"{'='*60}")

    # 创建模拟数据
    test_src_data = [
        "hello world",
        "how are you",
        "good morning",
        "thank you very much"
    ]

    test_tgt_data = [
        "你好 世界",
        "你 好 吗",
        "早上 好",
        "非常 感谢 你"
    ]

    # 写入临时文件
    temp_src_path = 'temp_src.txt'
    temp_tgt_path = 'temp_tgt.txt'

    with open(temp_src_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_src_data))

    with open(temp_tgt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_tgt_data))

    try:
        # 测试数据准备流程
        (src_train, tgt_train), (src_val, tgt_val), processor = prepare_translation_data(
            temp_src_path,
            temp_tgt_path,
            max_len=20
        )

        print(f"\n{'='*60}")
        print("测试结果")
        print(f"{'='*60}")
        print(f"训练集源语言形状: {src_train.shape}")
        print(f"训练集目标语言形状: {tgt_train.shape}")
        print(f"验证集源语言形状: {src_val.shape}")
        print(f"验证集目标语言形状: {tgt_val.shape}")
        print(f"源语言词汇表大小: {len(processor.src_word2idx)}")
        print(f"目标语言词汇表大小: {len(processor.tgt_word2idx)}")

        print(f"\n测试通过！")

    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_src_path):
            os.remove(temp_src_path)
        if os.path.exists(temp_tgt_path):
            os.remove(temp_tgt_path)
