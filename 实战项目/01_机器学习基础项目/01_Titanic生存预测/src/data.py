"""
数据加载和预处理模块
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed


class TitanicDataLoader:
    """Titanic数据加载器"""

    def __init__(self, data_dir=None):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录路径
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        self.data_dir = Path(data_dir)

    def load_train_data(self):
        """
        加载训练数据

        Returns:
            pd.DataFrame: 训练数据
        """
        train_path = self.data_dir / 'train.csv'
        if not train_path.exists():
            raise FileNotFoundError(
                f"训练数据不存在: {train_path}\n"
                "请先运行 python src/download_data.py 下载数据"
            )

        df = pd.read_csv(train_path)
        print(f"✓ 加载训练数据: {len(df)} 条记录, {len(df.columns)} 个特征")
        return df

    def load_test_data(self):
        """
        加载测试数据

        Returns:
            pd.DataFrame: 测试数据
        """
        test_path = self.data_dir / 'test.csv'
        if not test_path.exists():
            raise FileNotFoundError(
                f"测试数据不存在: {test_path}\n"
                "请先运行 python src/download_data.py 下载数据"
            )

        df = pd.read_csv(test_path)
        print(f"✓ 加载测试数据: {len(df)} 条记录, {len(df.columns)} 个特征")
        return df

    def get_data_info(self, df):
        """
        获取数据集信息

        Args:
            df: 数据框

        Returns:
            dict: 数据信息
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing': df.isnull().sum().to_dict(),
            'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        }

        # 数值特征统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info['numeric_stats'] = df[numeric_cols].describe().to_dict()

        # 类别特征统计
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            info['categorical_stats'] = {
                col: df[col].value_counts().to_dict()
                for col in categorical_cols
            }

        return info

    def print_data_summary(self, df, name='数据集'):
        """
        打印数据摘要

        Args:
            df: 数据框
            name: 数据集名称
        """
        print(f"\n{'=' * 60}")
        print(f"{name}摘要")
        print(f"{'=' * 60}")
        print(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"\n列信息:")
        print(df.info())

        print(f"\n缺失值统计:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            '缺失数量': missing,
            '缺失比例(%)': missing_pct
        })
        missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values(
            '缺失数量', ascending=False
        )
        if len(missing_df) > 0:
            print(missing_df)
        else:
            print("无缺失值")

        print(f"\n数值特征统计:")
        print(df.describe())

        if 'Survived' in df.columns:
            print(f"\n目标变量分布:")
            print(df['Survived'].value_counts())
            print(f"生存率: {df['Survived'].mean():.2%}")


class TitanicPreprocessor:
    """Titanic数据预处理器"""

    def __init__(self):
        """初始化预处理器"""
        self.fill_values = {}
        self.label_encoders = {}

    def handle_missing_values(self, df, fit=True):
        """
        处理缺失值

        Args:
            df: 数据框
            fit: 是否拟合填充值

        Returns:
            pd.DataFrame: 处理后的数据框
        """
        df = df.copy()

        # Age: 用中位数填充
        if 'Age' in df.columns:
            if fit:
                self.fill_values['Age'] = df['Age'].median()
            df['Age'].fillna(self.fill_values['Age'], inplace=True)

        # Embarked: 用众数填充
        if 'Embarked' in df.columns:
            if fit:
                self.fill_values['Embarked'] = df['Embarked'].mode()[0]
            df['Embarked'].fillna(self.fill_values['Embarked'], inplace=True)

        # Fare: 用中位数填充
        if 'Fare' in df.columns:
            if fit:
                self.fill_values['Fare'] = df['Fare'].median()
            df['Fare'].fillna(self.fill_values['Fare'], inplace=True)

        # Cabin: 填充为'Unknown'
        if 'Cabin' in df.columns:
            df['Cabin'].fillna('Unknown', inplace=True)

        return df

    def encode_categorical(self, df):
        """
        编码类别特征

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 编码后的数据框
        """
        df = df.copy()

        # Sex: male=1, female=0
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

        # Embarked: C=0, Q=1, S=2
        if 'Embarked' in df.columns:
            df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

        return df

    def drop_unnecessary_columns(self, df):
        """
        删除不必要的列

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 处理后的数据框
        """
        df = df.copy()

        # 删除不需要的列
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df

    def preprocess(self, df, fit=True):
        """
        完整的预处理流程

        Args:
            df: 数据框
            fit: 是否拟合预处理参数

        Returns:
            pd.DataFrame: 预处理后的数据框
        """
        print(f"\n开始预处理数据...")

        # 1. 处理缺失值
        df = self.handle_missing_values(df, fit=fit)
        print("✓ 缺失值处理完成")

        # 2. 编码类别特征
        df = self.encode_categorical(df)
        print("✓ 类别特征编码完成")

        # 3. 删除不必要的列
        df = self.drop_unnecessary_columns(df)
        print("✓ 删除不必要的列完成")

        print(f"✓ 预处理完成: {df.shape[0]} 行 × {df.shape[1]} 列")

        return df


def prepare_data(test_size=0.2, random_state=42):
    """
    准备训练和验证数据

    Args:
        test_size: 验证集比例
        random_state: 随机种子

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    set_seed(random_state)

    # 加载数据
    loader = TitanicDataLoader()
    df_train = loader.load_train_data()

    # 打印数据摘要
    loader.print_data_summary(df_train, '训练集')

    # 预处理
    preprocessor = TitanicPreprocessor()
    df_processed = preprocessor.preprocess(df_train, fit=True)

    # 分离特征和目标
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 条")
    print(f"  验证集: {len(X_val)} 条")
    print(f"  特征数: {X_train.shape[1]}")

    return X_train, X_val, y_train, y_val, preprocessor


def save_processed_data(output_dir=None):
    """
    保存预处理后的数据

    Args:
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'data' / 'processed'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载和预处理训练数据
    loader = TitanicDataLoader()
    df_train = loader.load_train_data()

    preprocessor = TitanicPreprocessor()
    df_train_processed = preprocessor.preprocess(df_train, fit=True)

    # 保存训练数据
    train_path = output_dir / 'train_processed.csv'
    df_train_processed.to_csv(train_path, index=False)
    print(f"✓ 训练数据已保存: {train_path}")

    # 加载和预处理测试数据
    try:
        df_test = loader.load_test_data()
        df_test_processed = preprocessor.preprocess(df_test, fit=False)

        # 保存测试数据
        test_path = output_dir / 'test_processed.csv'
        df_test_processed.to_csv(test_path, index=False)
        print(f"✓ 测试数据已保存: {test_path}")
    except FileNotFoundError:
        print("⚠ 测试数据不存在，跳过")


if __name__ == '__main__':
    # 测试数据加载和预处理
    print("=" * 60)
    print("Titanic数据预处理")
    print("=" * 60)

    # 准备数据
    X_train, X_val, y_train, y_val, preprocessor = prepare_data()

    # 保存预处理后的数据
    save_processed_data()

    print("\n✓ 数据准备完成！")
