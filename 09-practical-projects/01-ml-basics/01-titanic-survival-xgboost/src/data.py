"""
Titanic数据加载和预处理模块

本模块负责：
1. 加载Titanic数据集
2. 数据清洗和缺失值处理
3. 特征工程
4. 数据划分
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed


def load_titanic_data(data_path=None, test_size=0.2, random_state=42):
    """
    加载Titanic数据集

    Args:
        data_path: 数据文件路径（如果为None，使用项目data目录）
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names
    """
    print("="*60)
    print("加载Titanic数据集")
    print("="*60)

    set_seed(random_state)

    # 确定数据路径
    if data_path is None:
        project_dir = Path(__file__).parent.parent
        data_path = project_dir / 'data' / 'train.csv'

    # 加载数据
    print(f"\n1. 读取数据文件: {data_path}")
    df = pd.read_csv(data_path)

    print(f"   数据大小: {df.shape}")
    print(f"   列名: {df.columns.tolist()}")

    # 数据预处理
    print(f"\n2. 数据预处理...")
    df_processed = preprocess_data(df)

    # 分离特征和标签
    print(f"\n3. 分离特征和标签...")
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']

    feature_names = X.columns.tolist()

    print(f"   特征数量: {X.shape[1]}")
    print(f"   样本数量: {X.shape[0]}")
    print(f"   生存率: {y.mean():.2%}")

    # 划分数据集
    print(f"\n4. 划分数据集 (test_size={test_size})...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
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

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names


def preprocess_data(df):
    """
    数据预处理

    包括：
    1. 缺失值处理
    2. 特征工程
    3. 类别编码

    Args:
        df: 原始数据DataFrame

    Returns:
        处理后的DataFrame
    """
    df = df.copy()

    # ============================================
    # 1. 缺失值处理
    # ============================================

    # Age: 用中位数填充
    # 为什么用中位数：年龄分布可能有偏，中位数比均值更稳健
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Embarked: 用众数填充
    # 为什么用众数：登船港口是类别变量，用最常见的值填充
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Cabin: 创建新特征"是否有船舱号"
    # 为什么：船舱号缺失率很高，但"是否有船舱"本身是有用信息
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)

    # Fare: 用中位数填充（如果有缺失）
    if df['Fare'].isna().any():
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # ============================================
    # 2. 特征工程
    # ============================================

    # 2.1 家庭规模
    # 为什么有效：独自一人生存率低，小家庭生存率高，大家庭生存率低
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

    # 2.2 是否独自一人
    df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)

    # 2.3 从姓名中提取头衔
    # 为什么有效：头衔反映了社会地位和性别，与生存率高度相关
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # 合并稀有头衔
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Mme': 'Mrs',
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'].fillna('Rare', inplace=True)

    # 2.4 年龄分组
    # 为什么：不同年龄段的生存率差异大（儿童优先）
    df['Age_Group'] = pd.cut(df['Age'],
                             bins=[0, 12, 18, 35, 60, 100],
                             labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

    # 2.5 票价分组
    # 为什么：票价反映了社会地位，与生存率相关
    df['Fare_Group'] = pd.qcut(df['Fare'],
                               q=4,
                               labels=['Low', 'Medium', 'High', 'Very_High'],
                               duplicates='drop')

    # ============================================
    # 3. 类别编码
    # ============================================

    # 3.1 性别编码
    # 为什么用Label Encoding：XGBoost可以自动处理有序关系
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

    # 3.2 登船港口编码
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

    # 3.3 头衔编码
    df['Title'] = LabelEncoder().fit_transform(df['Title'])

    # 3.4 年龄分组编码
    df['Age_Group'] = LabelEncoder().fit_transform(df['Age_Group'])

    # 3.5 票价分组编码
    df['Fare_Group'] = LabelEncoder().fit_transform(df['Fare_Group'])

    # ============================================
    # 4. 选择最终特征
    # ============================================

    # 选择用于建模的特征
    features = [
        'Pclass',           # 船舱等级
        'Sex',              # 性别
        'Age',              # 年龄
        'SibSp',            # 兄弟姐妹/配偶数量
        'Parch',            # 父母/子女数量
        'Fare',             # 票价
        'Embarked',         # 登船港口
        'Family_Size',      # 家庭规模
        'Is_Alone',         # 是否独自一人
        'Has_Cabin',        # 是否有船舱号
        'Title',            # 头衔
        'Age_Group',        # 年龄分组
        'Fare_Group',       # 票价分组
    ]

    # 如果有Survived列，保留它
    if 'Survived' in df.columns:
        features.append('Survived')

    df_final = df[features].copy()

    return df_final


def get_feature_descriptions():
    """
    获取特征说明

    Returns:
        特征说明字典
    """
    descriptions = {
        'Pclass': '船舱等级（1=头等舱，2=二等舱，3=三等舱）',
        'Sex': '性别（0=female，1=male）',
        'Age': '年龄',
        'SibSp': '船上兄弟姐妹/配偶的数量',
        'Parch': '船上父母/子女的数量',
        'Fare': '票价',
        'Embarked': '登船港口（0=C，1=Q，2=S）',
        'Family_Size': '家庭规模（SibSp + Parch + 1）',
        'Is_Alone': '是否独自一人（1=是，0=否）',
        'Has_Cabin': '是否有船舱号（1=有，0=无）',
        'Title': '头衔（从姓名中提取）',
        'Age_Group': '年龄分组',
        'Fare_Group': '票价分组',
    }
    return descriptions


def analyze_data(df):
    """
    数据分析

    Args:
        df: 数据DataFrame

    Returns:
        分析结果字典
    """
    analysis = {}

    # 基本统计
    analysis['shape'] = df.shape
    analysis['missing_values'] = df.isnull().sum()
    analysis['dtypes'] = df.dtypes

    # 生存率统计
    if 'Survived' in df.columns:
        analysis['survival_rate'] = df['Survived'].mean()
        analysis['survival_by_sex'] = df.groupby('Sex')['Survived'].mean()
        analysis['survival_by_pclass'] = df.groupby('Pclass')['Survived'].mean()

    return analysis


if __name__ == '__main__':
    """
    测试数据加载
    """
    print("="*60)
    print("数据加载模块测试")
    print("="*60)

    # 加载数据
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = load_titanic_data()

        # 显示特征说明
        print("\n" + "="*60)
        print("特征说明")
        print("="*60)

        descriptions = get_feature_descriptions()
        for feature in feature_names:
            if feature in descriptions:
                print(f"{feature:15s}: {descriptions[feature]}")

        # 显示数据样本
        print("\n" + "="*60)
        print("数据样本")
        print("="*60)
        print(f"\n训练集前5行:")
        print(pd.DataFrame(X_train[:5], columns=feature_names))
        print(f"\n对应标签: {y_train[:5].tolist()}")

        print("\n✓ 数据加载模块测试完成！")

    except FileNotFoundError:
        print("\n⚠ 数据文件不存在")
        print("请先运行: cd data && python download_data.py")
