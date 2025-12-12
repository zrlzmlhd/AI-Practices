"""
Otto分类数据处理模块

本模块负责：
1. 加载Otto数据集
2. 数据预处理和特征工程
3. 数据集划分
4. 特征统计分析

每个步骤都有详细的注释说明。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle


class OttoDataProcessor:
    """
    Otto数据处理器

    【是什么】：处理Otto多分类数据的工具类
    【做什么】：
        - 加载和清洗数据
        - 特征工程
        - 数据划分
        - 标签编码
    【为什么】：
        - Otto数据需要特殊处理
        - 多分类问题需要标签编码
        - 特征工程提升性能
    """

    def __init__(self, data_path):
        """
        初始化数据处理器

        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # 数据
        self.df = None
        self.feature_names = None
        self.target_name = 'target'

    def load_data(self):
        """
        加载数据

        Returns:
            DataFrame
        """
        print("\n加载Otto数据集...")

        # ============================================
        # 步骤1: 读取CSV文件
        # ============================================
        # 【是什么】：Otto Group Product Classification数据
        # 【格式】：CSV文件，包含id、93个特征、1个目标列
        self.df = pd.read_csv(self.data_path)

        print(f"  原始数据形状: {self.df.shape}")
        print(f"  特征数量: {self.df.shape[1] - 2}")  # 减去id和target

        # ============================================
        # 步骤2: 检查数据
        # ============================================
        print(f"\n数据概览:")
        print(f"  列名: {self.df.columns.tolist()[:5]}...")
        print(f"  前几行:\n{self.df.head(3)}")

        # 检查缺失值
        missing_count = self.df.isnull().sum().sum()
        print(f"  缺失值: {missing_count}")

        # 检查目标分布
        if self.target_name in self.df.columns:
            print(f"\n目标分布:")
            target_counts = self.df[self.target_name].value_counts().sort_index()
            for target, count in target_counts.items():
                percentage = count / len(self.df) * 100
                print(f"    {target}: {count} ({percentage:.1f}%)")

        return self.df

    def preprocess_data(self):
        """
        预处理数据

        Returns:
            X, y: 特征和标签
        """
        print("\n预处理数据...")

        # ============================================
        # 步骤1: 分离特征和标签
        # ============================================
        # 【是什么】：X是特征，y是目标

        # 移除id列（如果存在）
        if 'id' in self.df.columns:
            self.df = self.df.drop('id', axis=1)

        # 分离特征和标签
        X = self.df.drop(self.target_name, axis=1)
        y = self.df[self.target_name]

        self.feature_names = X.columns.tolist()

        print(f"  特征形状: {X.shape}")
        print(f"  标签形状: {y.shape}")

        # ============================================
        # 步骤2: 标签编码
        # ============================================
        # 【是什么】：将Class_1, Class_2, ... 转换为 0, 1, ...
        # 【为什么】：
        #   - XGBoost需要数值标签
        #   - 从0开始的连续整数
        #   - 便于计算和评估

        # 原始标签：['Class_1', 'Class_2', ..., 'Class_9']
        # 编码后：[0, 1, 2, ..., 8]
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"\n标签编码:")
        print(f"  原始标签: {self.label_encoder.classes_}")
        print(f"  编码范围: 0 到 {len(self.label_encoder.classes_) - 1}")

        # ============================================
        # 步骤3: 特征统计
        # ============================================
        print(f"\n特征统计:")
        print(f"  特征数量: {X.shape[1]}")
        print(f"  特征范围: {X.min().min():.2f} 到 {X.max().max():.2f}")
        print(f"  特征均值: {X.mean().mean():.2f}")
        print(f"  特征标准差: {X.std().mean():.2f}")

        return X.values, y_encoded

    def create_features(self, X):
        """
        创建工程特征

        【是什么】：基于原始特征创建新特征
        【为什么】：
            - 提升模型性能
            - 捕获特征间的关系
            - 增加模型的表达能力

        Args:
            X: 原始特征

        Returns:
            增强后的特征
        """
        print("\n创建工程特征...")

        X_df = pd.DataFrame(X, columns=self.feature_names)

        # ============================================
        # 特征1: 统计特征
        # ============================================
        # 【是什么】：每行的统计量
        # 【为什么】：
        #   - 捕获整体模式
        #   - 不同产品可能有不同的统计特性

        # 行求和
        X_df['feat_sum'] = X_df.sum(axis=1)
        # 【含义】：所有特征的总和
        # 【用途】：某些产品可能总体数值更高

        # 行均值
        X_df['feat_mean'] = X_df.mean(axis=1)
        # 【含义】：平均特征值
        # 【用途】：标准化的总体水平

        # 行标准差
        X_df['feat_std'] = X_df.std(axis=1)
        # 【含义】：特征的变化程度
        # 【用途】：某些产品特征更分散

        # 行最大值
        X_df['feat_max'] = X_df.max(axis=1)
        # 【含义】：最显著的特征
        # 【用途】：某些产品有突出特征

        # 行最小值
        X_df['feat_min'] = X_df.min(axis=1)
        # 【含义】：最弱的特征
        # 【用途】：特征的下界

        # ============================================
        # 特征2: 非零特征数
        # ============================================
        # 【是什么】：有多少个特征不为0
        # 【为什么】：
        #   - Otto特征是计数型
        #   - 非零特征数可能有区分度
        X_df['feat_nonzero'] = (X_df[self.feature_names] != 0).sum(axis=1)

        # ============================================
        # 特征3: 特征比率
        # ============================================
        # 【是什么】：最大值/均值
        # 【为什么】：
        #   - 衡量特征的集中度
        #   - 某些产品可能特征更集中
        X_df['feat_max_mean_ratio'] = X_df['feat_max'] / (X_df['feat_mean'] + 1e-5)

        print(f"  原始特征数: {len(self.feature_names)}")
        print(f"  新增特征数: {X_df.shape[1] - len(self.feature_names)}")
        print(f"  总特征数: {X_df.shape[1]}")

        return X_df.values

    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        划分数据集

        【重要】：多分类问题使用分层划分
        【为什么】：
            - 保持各类别的比例
            - 避免某些类别在验证集/测试集中缺失
            - 更准确的性能评估

        Args:
            X: 特征
            y: 标签
            test_size: 测试集比例
            val_size: 验证集比例（从训练集中划分）
            random_state: 随机种子

        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        print("\n划分数据集...")

        # ============================================
        # 步骤1: 划分训练集和测试集
        # ============================================
        # 【stratify=y】：分层划分，保持类别比例
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # 关键：保持类别比例
        )

        # ============================================
        # 步骤2: 从训练集中划分验证集
        # ============================================
        val_size_adjusted = val_size / (1 - test_size)  # 调整比例

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_train_val
        )

        print(f"  训练集: {X_train.shape}")
        print(f"  验证集: {X_val.shape}")
        print(f"  测试集: {X_test.shape}")

        # 检查类别分布
        print(f"\n类别分布检查:")
        for split_name, y_split in [('训练集', y_train), ('验证集', y_val), ('测试集', y_test)]:
            unique, counts = np.unique(y_split, return_counts=True)
            print(f"  {split_name}: {len(unique)}个类别")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_cv_folds(self, X, y, n_splits=5, random_state=42):
        """
        获取交叉验证折

        【是什么】：StratifiedKFold交叉验证
        【为什么】：
            - 用于Stacking集成
            - 充分利用训练数据
            - 保持类别比例

        Args:
            X: 特征
            y: 标签
            n_splits: 折数
            random_state: 随机种子

        Returns:
            StratifiedKFold对象
        """
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

        print(f"\n创建{n_splits}折交叉验证")
        print(f"  每折训练集: ~{len(X) * (n_splits-1) / n_splits:.0f}样本")
        print(f"  每折验证集: ~{len(X) / n_splits:.0f}样本")

        return skf

    def inverse_transform_labels(self, y_encoded):
        """
        反编码标签

        【是什么】：将数值标签转换回原始标签
        【为什么】：
            - 便于理解预测结果
            - 提交Kaggle需要原始标签

        Args:
            y_encoded: 编码后的标签

        Returns:
            原始标签
        """
        return self.label_encoder.inverse_transform(y_encoded)

    def save_processor(self, filepath):
        """保存数据处理器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        print(f"✓ 数据处理器已保存: {filepath}")

    def load_processor(self, filepath):
        """加载数据处理器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.label_encoder = data['label_encoder']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
        print(f"✓ 数据处理器已加载: {filepath}")


def prepare_otto_data(data_path='data/train.csv',
                      test_size=0.2,
                      val_size=0.1,
                      random_state=42,
                      create_features=True):
    """
    准备Otto分类数据

    Args:
        data_path: 数据文件路径
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        create_features: 是否创建工程特征

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), processor
    """
    print("="*60)
    print("Otto分类数据准备")
    print("="*60)

    # ============================================
    # 步骤1: 创建数据处理器
    # ============================================
    processor = OttoDataProcessor(data_path=data_path)

    # ============================================
    # 步骤2: 加载数据
    # ============================================
    processor.load_data()

    # ============================================
    # 步骤3: 预处理
    # ============================================
    X, y = processor.preprocess_data()

    # ============================================
    # 步骤4: 特征工程
    # ============================================
    if create_features:
        X = processor.create_features(X)

    # ============================================
    # 步骤5: 划分数据集
    # ============================================
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(
        X, y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    # ============================================
    # 数据统计
    # ============================================
    print("\n数据统计:")
    print(f"  特征数: {X_train.shape[1]}")
    print(f"  类别数: {len(np.unique(y))}")
    print(f"  训练样本数: {len(X_train)}")
    print(f"  验证样本数: {len(X_val)}")
    print(f"  测试样本数: {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), processor


if __name__ == '__main__':
    """
    测试数据处理
    """
    print("="*60)
    print("数据处理模块测试")
    print("="*60)

    # 创建模拟数据
    print("\n创建模拟数据...")
    n_samples = 1000
    n_features = 93
    n_classes = 9

    # 模拟Otto数据
    np.random.seed(42)
    X_mock = np.random.randint(0, 100, (n_samples, n_features))
    y_mock = np.random.randint(0, n_classes, n_samples)
    y_mock_labels = [f'Class_{i+1}' for i in y_mock]

    # 创建DataFrame
    df = pd.DataFrame(X_mock, columns=[f'feat_{i}' for i in range(n_features)])
    df['id'] = range(n_samples)
    df['target'] = y_mock_labels

    # 保存临时文件
    temp_path = 'temp_otto_data.csv'
    df.to_csv(temp_path, index=False)

    # 测试数据处理
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), processor = prepare_otto_data(
            data_path=temp_path,
            test_size=0.2,
            val_size=0.1,
            create_features=True
        )

        print("\n✓ 数据处理测试通过！")

        # 测试标签反编码
        print("\n测试标签反编码...")
        y_original = processor.inverse_transform_labels(y_test[:5])
        print(f"  编码标签: {y_test[:5]}")
        print(f"  原始标签: {y_original}")

        # 测试交叉验证
        print("\n测试交叉验证...")
        skf = processor.get_cv_folds(X_train, y_train, n_splits=5)
        print(f"  交叉验证折数: {skf.get_n_splits()}")

    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n✓ 所有测试通过！")
