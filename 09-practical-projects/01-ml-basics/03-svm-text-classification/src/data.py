"""
SVM文本分类数据处理模块

本模块负责：
1. 加载文本分类数据集（20 Newsgroups、IMDB等）
2. 文本预处理（清洗、分词、去停用词）
3. 特征提取（TF-IDF、Word2Vec、Doc2Vec）
4. 数据划分和保存

【推荐数据集】：
- 20 Newsgroups（新闻分类）
- IMDB电影评论（情感分类）
- AG News（新闻分类）
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re
import pickle
from collections import Counter

# 文本处理
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 特征提取
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

# Word2Vec
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("警告: gensim未安装，Word2Vec功能不可用")


class TextPreprocessor:
    """
    文本预处理器

    【是什么】：文本清洗和标准化工具
    【做什么】：
        - 去除HTML标签、特殊字符
        - 转小写
        - 分词
        - 去停用词
        - 词形还原
    """

    def __init__(self, remove_stopwords=True, lemmatize=True):
        """
        初始化预处理器

        Args:
            remove_stopwords: 是否去除停用词
            lemmatize: 是否进行词形还原
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

        # 下载必要的NLTK数据
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("下载停用词列表...")
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

        try:
            word_tokenize("test")
        except LookupError:
            print("下载分词器...")
            nltk.download('punkt', quiet=True)

        if lemmatize:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.lemmatizer.lemmatize("test")
            except LookupError:
                print("下载词形还原数据...")
                nltk.download('wordnet', quiet=True)
                self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        清洗文本

        【步骤】：
        1. 去除HTML标签
        2. 去除URL
        3. 去除邮箱
        4. 去除特殊字符（保留字母和空格）
        5. 转小写
        """
        if not isinstance(text, str):
            return ""

        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)

        # 去除URL
        text = re.sub(r'http\S+|www\S+', '', text)

        # 去除邮箱
        text = re.sub(r'\S+@\S+', '', text)

        # 去除特殊字符（保留字母和空格）
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # 转小写
        text = text.lower()

        # 去除多余空格
        text = ' '.join(text.split())

        return text

    def tokenize(self, text):
        """分词"""
        return word_tokenize(text)

    def remove_stopwords_func(self, tokens):
        """去除停用词"""
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize_tokens(self, tokens):
        """词形还原"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text):
        """
        完整的预处理流程

        Args:
            text: 原始文本

        Returns:
            处理后的文本（字符串）
        """
        # 清洗
        text = self.clean_text(text)

        # 分词
        tokens = self.tokenize(text)

        # 去停用词
        if self.remove_stopwords:
            tokens = self.remove_stopwords_func(tokens)

        # 词形还原
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)

        # 重新组合
        return ' '.join(tokens)

    def preprocess_batch(self, texts, verbose=True):
        """
        批量预处理

        Args:
            texts: 文本列表
            verbose: 是否显示进度

        Returns:
            处理后的文本列表
        """
        processed = []
        total = len(texts)

        for i, text in enumerate(texts):
            if verbose and (i + 1) % 1000 == 0:
                print(f"  已处理: {i+1}/{total}")

            processed.append(self.preprocess(text))

        return processed


class TextFeatureExtractor:
    """
    文本特征提取器

    【是什么】：将文本转换为数值特征
    【支持的方法】：
        - TF-IDF（词频-逆文档频率）
        - Count Vectorizer（词频统计）
        - Word2Vec（词向量平均）
    """

    def __init__(self, method='tfidf', max_features=5000, **kwargs):
        """
        初始化特征提取器

        Args:
            method: 特征提取方法 ('tfidf', 'count', 'word2vec')
            max_features: 最大特征数
            **kwargs: 其他参数
        """
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
        self.word2vec_model = None

        if method == 'tfidf':
            # 【TF-IDF】：
            # - TF（词频）：词在文档中出现的频率
            # - IDF（逆文档频率）：log(总文档数/包含该词的文档数)
            # - TF-IDF = TF * IDF
            # 【为什么】：降低常见词的权重，提高区分性词的权重
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=kwargs.get('ngram_range', (1, 2)),  # 1-gram和2-gram
                min_df=kwargs.get('min_df', 2),  # 最小文档频率
                max_df=kwargs.get('max_df', 0.95),  # 最大文档频率
                sublinear_tf=True  # 使用对数TF
            )

        elif method == 'count':
            # 【Count Vectorizer】：简单的词频统计
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=kwargs.get('ngram_range', (1, 2)),
                min_df=kwargs.get('min_df', 2),
                max_df=kwargs.get('max_df', 0.95)
            )

        elif method == 'word2vec':
            if not GENSIM_AVAILABLE:
                raise ImportError("Word2Vec需要安装gensim: pip install gensim")
            # Word2Vec参数将在fit时设置
            self.vector_size = kwargs.get('vector_size', 100)
            self.window = kwargs.get('window', 5)
            self.min_count = kwargs.get('min_count', 2)

    def fit(self, texts):
        """
        训练特征提取器

        Args:
            texts: 文本列表
        """
        print(f"\n训练{self.method}特征提取器...")

        if self.method in ['tfidf', 'count']:
            self.vectorizer.fit(texts)
            print(f"  特征维度: {len(self.vectorizer.get_feature_names_out())}")

        elif self.method == 'word2vec':
            # 分词
            tokenized_texts = [text.split() for text in texts]

            # 训练Word2Vec
            self.word2vec_model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=4,
                sg=1  # Skip-gram
            )
            print(f"  词汇表大小: {len(self.word2vec_model.wv)}")
            print(f"  向量维度: {self.vector_size}")

    def transform(self, texts):
        """
        转换文本为特征

        Args:
            texts: 文本列表

        Returns:
            特征矩阵
        """
        if self.method in ['tfidf', 'count']:
            return self.vectorizer.transform(texts)

        elif self.method == 'word2vec':
            # 【Word2Vec特征】：对句子中所有词向量求平均
            features = []
            for text in texts:
                words = text.split()
                word_vectors = []

                for word in words:
                    if word in self.word2vec_model.wv:
                        word_vectors.append(self.word2vec_model.wv[word])

                if word_vectors:
                    # 平均词向量
                    features.append(np.mean(word_vectors, axis=0))
                else:
                    # 零向量
                    features.append(np.zeros(self.vector_size))

            return np.array(features)

    def fit_transform(self, texts):
        """训练并转换"""
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self):
        """获取特征名称"""
        if self.method in ['tfidf', 'count']:
            return self.vectorizer.get_feature_names_out()
        else:
            return None


def load_20newsgroups_data(subset='train', categories=None, remove=('headers', 'footers', 'quotes')):
    """
    加载20 Newsgroups数据集

    【是什么】：经典的文本分类数据集
    【包含】：20个新闻组的文章

    Args:
        subset: 'train', 'test', 或 'all'
        categories: 类别列表（None表示全部）
        remove: 要移除的部分

    Returns:
        texts, labels, label_names
    """
    from sklearn.datasets import fetch_20newsgroups

    print(f"\n加载20 Newsgroups数据集...")
    print(f"  子集: {subset}")

    data = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=remove,
        shuffle=True,
        random_state=42
    )

    texts = data.data
    labels = data.target
    label_names = data.target_names

    print(f"  样本数: {len(texts)}")
    print(f"  类别数: {len(label_names)}")
    print(f"  类别: {label_names}")

    return texts, labels, label_names


def prepare_text_classification_data(
    data_source='20newsgroups',
    feature_method='tfidf',
    max_features=5000,
    test_size=0.2,
    categories=None,
    preprocess=True
):
    """
    准备文本分类数据

    【完整流程】：
    1. 加载数据
    2. 文本预处理
    3. 特征提取
    4. 数据划分

    Args:
        data_source: 数据源 ('20newsgroups')
        feature_method: 特征提取方法 ('tfidf', 'count', 'word2vec')
        max_features: 最大特征数
        test_size: 测试集比例
        categories: 类别列表
        preprocess: 是否预处理

    Returns:
        (X_train, y_train), (X_test, y_test), feature_extractor, label_names
    """
    print("="*60)
    print("文本分类数据准备")
    print("="*60)

    # ============================================
    # 1. 加载数据
    # ============================================
    if data_source == '20newsgroups':
        texts_train, labels_train, label_names = load_20newsgroups_data(
            subset='train',
            categories=categories
        )
        texts_test, labels_test, _ = load_20newsgroups_data(
            subset='test',
            categories=categories
        )

        texts = texts_train + texts_test
        labels = np.concatenate([labels_train, labels_test])

    else:
        raise ValueError(f"不支持的数据源: {data_source}")

    # ============================================
    # 2. 文本预处理
    # ============================================
    if preprocess:
        print("\n" + "="*60)
        print("文本预处理")
        print("="*60)

        preprocessor = TextPreprocessor(
            remove_stopwords=True,
            lemmatize=True
        )

        texts = preprocessor.preprocess_batch(texts, verbose=True)
        print(f"  预处理完成")

    # ============================================
    # 3. 特征提取
    # ============================================
    print("\n" + "="*60)
    print("特征提取")
    print("="*60)

    feature_extractor = TextFeatureExtractor(
        method=feature_method,
        max_features=max_features
    )

    # 划分训练集和测试集
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    # 提取特征
    X_train = feature_extractor.fit_transform(X_train_text)
    X_test = feature_extractor.transform(X_test_text)

    print(f"\n数据集统计:")
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  类别数: {len(label_names)}")

    return (X_train, y_train), (X_test, y_test), feature_extractor, label_names


if __name__ == '__main__':
    """测试数据处理模块"""
    print("="*60)
    print("文本分类数据处理模块测试")
    print("="*60)

    # 测试预处理
    print("\n测试文本预处理...")
    preprocessor = TextPreprocessor()

    test_text = "This is a TEST! <html>Remove HTML</html> http://example.com"
    processed = preprocessor.preprocess(test_text)
    print(f"  原文: {test_text}")
    print(f"  处理后: {processed}")

    # 测试数据加载（使用少量类别）
    print("\n测试数据加载...")
    try:
        (X_train, y_train), (X_test, y_test), extractor, labels = \
            prepare_text_classification_data(
                categories=['alt.atheism', 'talk.religion.misc'],
                max_features=1000,
                preprocess=True
            )
        print("\n✓ 数据处理测试通过！")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
