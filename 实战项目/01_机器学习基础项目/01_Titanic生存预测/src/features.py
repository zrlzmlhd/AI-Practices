"""
特征工程模块
"""

import pandas as pd
import numpy as np
import re


class TitanicFeatureEngineer:
    """Titanic特征工程器"""

    def __init__(self):
        """初始化特征工程器"""
        self.title_mapping = {}

    def extract_title(self, df):
        """
        从姓名中提取称谓

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 添加Title特征的数据框
        """
        df = df.copy()

        if 'Name' not in df.columns:
            return df

        # 提取称谓
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

        # 称谓映射
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

        # 编码Title
        title_encode = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
        df['Title'] = df['Title'].map(title_encode)

        return df

    def create_family_features(self, df):
        """
        创建家庭相关特征

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 添加家庭特征的数据框
        """
        df = df.copy()

        # 家庭规模
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

            # 是否独自一人
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

            # 家庭规模分类
            df['FamilySize_Cat'] = pd.cut(
                df['FamilySize'],
                bins=[0, 1, 4, 11],
                labels=[0, 1, 2]
            ).astype(int)

        return df

    def create_age_features(self, df):
        """
        创建年龄相关特征

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 添加年龄特征的数据框
        """
        df = df.copy()

        if 'Age' not in df.columns:
            return df

        # 年龄分组
        df['Age_Cat'] = pd.cut(
            df['Age'],
            bins=[0, 12, 18, 35, 60, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        # 年龄与船舱等级的交互特征
        if 'Pclass' in df.columns:
            df['Age_Class'] = df['Age'] * df['Pclass']

        return df

    def create_fare_features(self, df):
        """
        创建票价相关特征

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 添加票价特征的数据框
        """
        df = df.copy()

        if 'Fare' not in df.columns:
            return df

        # 票价分组
        df['Fare_Cat'] = pd.qcut(
            df['Fare'],
            q=4,
            labels=[0, 1, 2, 3],
            duplicates='drop'
        ).astype(int)

        # 人均票价
        if 'FamilySize' in df.columns:
            df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']

        return df

    def create_cabin_features(self, df):
        """
        创建客舱相关特征

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 添加客舱特征的数据框
        """
        df = df.copy()

        if 'Cabin' not in df.columns:
            return df

        # 是否有客舱信息
        df['Has_Cabin'] = (df['Cabin'] != 'Unknown').astype(int)

        # 提取甲板信息
        df['Deck'] = df['Cabin'].str[0]
        df['Deck'] = df['Deck'].map({
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5,
            'F': 6, 'G': 7, 'T': 8, 'U': 0
        })
        df['Deck'].fillna(0, inplace=True)
        df['Deck'] = df['Deck'].astype(int)

        return df

    def create_ticket_features(self, df):
        """
        创建船票相关特征

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 添加船票特征的数据框
        """
        df = df.copy()

        if 'Ticket' not in df.columns:
            return df

        # 提取票号前缀
        df['Ticket_Prefix'] = df['Ticket'].str.extract('([A-Za-z]+)', expand=False)
        df['Ticket_Prefix'].fillna('None', inplace=True)

        # 票号前缀频率
        ticket_freq = df['Ticket_Prefix'].value_counts()
        df['Ticket_Freq'] = df['Ticket_Prefix'].map(ticket_freq)

        return df

    def create_interaction_features(self, df):
        """
        创建交互特征

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 添加交互特征的数据框
        """
        df = df.copy()

        # Sex * Pclass
        if 'Sex' in df.columns and 'Pclass' in df.columns:
            df['Sex_Pclass'] = df['Sex'] * df['Pclass']

        # Age * Sex
        if 'Age' in df.columns and 'Sex' in df.columns:
            df['Age_Sex'] = df['Age'] * df['Sex']

        return df

    def engineer_features(self, df):
        """
        完整的特征工程流程

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 特征工程后的数据框
        """
        print("\n开始特征工程...")

        # 1. 提取称谓
        df = self.extract_title(df)
        print("✓ 提取称谓特征")

        # 2. 创建家庭特征
        df = self.create_family_features(df)
        print("✓ 创建家庭特征")

        # 3. 创建年龄特征
        df = self.create_age_features(df)
        print("✓ 创建年龄特征")

        # 4. 创建票价特征
        df = self.create_fare_features(df)
        print("✓ 创建票价特征")

        # 5. 创建客舱特征
        df = self.create_cabin_features(df)
        print("✓ 创建客舱特征")

        # 6. 创建船票特征
        df = self.create_ticket_features(df)
        print("✓ 创建船票特征")

        # 7. 创建交互特征
        df = self.create_interaction_features(df)
        print("✓ 创建交互特征")

        print(f"✓ 特征工程完成: {df.shape[1]} 个特征")

        return df


def get_feature_importance(model, feature_names, top_n=10):
    """
    获取特征重要性

    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        top_n: 返回前N个重要特征

    Returns:
        pd.DataFrame: 特征重要性
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return feature_importance.head(top_n)


if __name__ == '__main__':
    # 测试特征工程
    from data import TitanicDataLoader, TitanicPreprocessor

    print("=" * 60)
    print("Titanic特征工程测试")
    print("=" * 60)

    # 加载数据
    loader = TitanicDataLoader()
    df = loader.load_train_data()

    print(f"\n原始特征数: {df.shape[1]}")

    # 预处理（但保留Name, Cabin, Ticket用于特征工程）
    preprocessor = TitanicPreprocessor()
    df = preprocessor.handle_missing_values(df, fit=True)

    # 特征工程
    engineer = TitanicFeatureEngineer()
    df_engineered = engineer.engineer_features(df)

    print(f"\n特征工程后特征数: {df_engineered.shape[1]}")
    print(f"\n新增特征:")
    new_features = set(df_engineered.columns) - set(df.columns)
    for feat in sorted(new_features):
        print(f"  - {feat}")

    print("\n✓ 特征工程测试完成！")
