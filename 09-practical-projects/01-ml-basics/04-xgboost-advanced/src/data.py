"""
XGBoost高级技巧 - 数据处理模块

本模块展示竞赛级别的数据处理技巧：
1. 高级特征工程（统计特征、交互特征、目标编码）
2. 特征选择（重要性、相关性、递归消除）
3. 数据增强和采样
4. 特征缩放和归一化

【推荐数据集】：
- Kaggle竞赛数据集
- UCI机器学习库
- 本项目使用：California Housing（回归）或 Credit Card Fraud（分类）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_california_housing, make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineering:
    """
    高级特征工程

    【是什么】：竞赛级别的特征构造技巧
    【包含】：
        - 统计特征（均值、方差、分位数）
        - 交互特征（乘法、除法、多项式）
        - 聚合特征（分组统计）
        - 目标编码（Target Encoding）
    """

    def __init__(self):
        self.target_encoders = {}
        self.feature_names = []

    def create_statistical_features(self, df, numeric_cols):
        """
        创建统计特征

        【是什么】：基于现有特征的统计量
        【为什么】：捕获数据的分布特性
        """
        print("\n创建统计特征...")
        new_features = pd.DataFrame()

        # 【行统计】：每个样本的统计量
        new_features['row_mean'] = df[numeric_cols].mean(axis=1)
        new_features['row_std'] = df[numeric_cols].std(axis=1)
        new_features['row_min'] = df[numeric_cols].min(axis=1)
        new_features['row_max'] = df[numeric_cols].max(axis=1)
        new_features['row_median'] = df[numeric_cols].median(axis=1)

        # 【范围特征】
        new_features['row_range'] = new_features['row_max'] - new_features['row_min']

        # 【偏度和峰度】：分布形状
        new_features['row_skew'] = df[numeric_cols].skew(axis=1)
        new_features['row_kurt'] = df[numeric_cols].kurtosis(axis=1)

        print(f"  创建了 {len(new_features.columns)} 个统计特征")
        return new_features

    def create_interaction_features(self, df, numeric_cols, max_interactions=20):
        """
        创建交互特征

        【是什么】：特征之间的组合
        【为什么】：捕获特征间的非线性关系
        【技巧】：
            - 乘法：f1 * f2（联合效应）
            - 除法：f1 / f2（比率）
            - 差值：f1 - f2（对比）
        """
        print("\n创建交互特征...")
        new_features = pd.DataFrame()

        # 选择最重要的特征进行交互（避免特征爆炸）
        selected_cols = numeric_cols[:min(len(numeric_cols), 5)]

        count = 0
        for i, col1 in enumerate(selected_cols):
            for col2 in selected_cols[i+1:]:
                if count >= max_interactions:
                    break

                # 乘法
                new_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]

                # 除法（避免除零）
                new_features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)

                # 差值
                new_features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]

                count += 1

        print(f"  创建了 {len(new_features.columns)} 个交互特征")
        return new_features

    def create_polynomial_features(self, df, numeric_cols, degree=2, max_features=10):
        """
        创建多项式特征

        【是什么】：特征的幂次方
        【为什么】：捕获非线性关系
        """
        print("\n创建多项式特征...")
        new_features = pd.DataFrame()

        selected_cols = numeric_cols[:min(len(numeric_cols), max_features)]

        for col in selected_cols:
            for d in range(2, degree + 1):
                new_features[f'{col}_pow{d}'] = df[col] ** d

        print(f"  创建了 {len(new_features.columns)} 个多项式特征")
        return new_features

    def create_binning_features(self, df, numeric_cols, n_bins=5):
        """
        创建分箱特征

        【是什么】：将连续特征离散化
        【为什么】：
            - 减少异常值影响
            - 捕获非线性关系
            - 提高模型鲁棒性
        """
        print("\n创建分箱特征...")
        new_features = pd.DataFrame()

        for col in numeric_cols[:5]:  # 只对前5个特征分箱
            new_features[f'{col}_bin'] = pd.qcut(
                df[col],
                q=n_bins,
                labels=False,
                duplicates='drop'
            )

        print(f"  创建了 {len(new_features.columns)} 个分箱特征")
        return new_features

    def create_target_encoding(self, df, categorical_cols, target, smoothing=10):
        """
        目标编码（Target Encoding）

        【是什么】：用目标变量的统计量编码类别特征
        【为什么】：
            - 比One-Hot编码更紧凑
            - 包含目标信息
            - 适合高基数类别特征

        【技巧】：平滑处理防止过拟合
        公式：encoded = (n * mean + smoothing * global_mean) / (n + smoothing)
        """
        print("\n创建目标编码...")
        new_features = pd.DataFrame()

        global_mean = target.mean()

        for col in categorical_cols:
            # 计算每个类别的目标均值和计数
            agg = df.groupby(col)[target.name].agg(['mean', 'count'])

            # 平滑处理
            smoothed_mean = (
                agg['count'] * agg['mean'] + smoothing * global_mean
            ) / (agg['count'] + smoothing)

            # 映射
            new_features[f'{col}_target_enc'] = df[col].map(smoothed_mean)

            # 保存编码器（用于测试集）
            self.target_encoders[col] = smoothed_mean

        print(f"  创建了 {len(new_features.columns)} 个目标编码特征")
        return new_features

    def apply_target_encoding(self, df, categorical_cols, global_mean):
        """应用已训练的目标编码"""
        new_features = pd.DataFrame()

        for col in categorical_cols:
            if col in self.target_encoders:
                new_features[f'{col}_target_enc'] = df[col].map(
                    self.target_encoders[col]
                ).fillna(global_mean)

        return new_features


class AdvancedFeatureSelector:
    """
    高级特征选择

    【是什么】：选择最有价值的特征
    【方法】：
        - 基于重要性（XGBoost特征重要性）
        - 基于相关性（去除冗余特征）
        - 基于统计检验（卡方、互信息）
    """

    def __init__(self):
        self.selected_features = None

    def select_by_importance(self, X, y, feature_names, top_k=50, method='xgboost'):
        """
        基于重要性选择特征

        【原理】：训练模型，根据特征重要性排序
        """
        print(f"\n基于{method}重要性选择特征...")

        if method == 'xgboost':
            import xgboost as xgb

            # 训练XGBoost
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)

            # 获取重要性
            importance = model.feature_importances_

        else:  # mutual_info
            importance = mutual_info_classif(X, y, random_state=42)

        # 排序
        indices = np.argsort(importance)[::-1][:top_k]
        self.selected_features = [feature_names[i] for i in indices]

        print(f"  选择了 {len(self.selected_features)} 个特征")
        return self.selected_features, importance[indices]

    def remove_correlated_features(self, X, feature_names, threshold=0.95):
        """
        去除高度相关的特征

        【为什么】：
            - 减少冗余
            - 降低过拟合风险
            - 提高训练速度
        """
        print(f"\n去除相关性 > {threshold} 的特征...")

        # 计算相关矩阵
        corr_matrix = pd.DataFrame(X, columns=feature_names).corr().abs()

        # 找到高度相关的特征对
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # 删除相关性高的特征
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        remaining_features = [f for f in feature_names if f not in to_drop]

        print(f"  删除了 {len(to_drop)} 个高度相关的特征")
        print(f"  保留了 {len(remaining_features)} 个特征")

        return remaining_features


def load_california_housing_data():
    """
    加载California Housing数据集

    【是什么】：加州房价预测数据集
    【任务】：回归任务
    """
    print("\n加载California Housing数据集...")

    data = fetch_california_housing(as_frame=True)
    df = data.frame

    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    print(f"  样本数: {len(df)}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  特征: {list(X.columns)}")

    return X, y, list(X.columns)


def prepare_advanced_features(
    X, y,
    task_type='classification',
    create_interactions=True,
    create_polynomials=True,
    create_statistical=True,
    target_encoding=False,
    feature_selection=True,
    top_k_features=100
):
    """
    准备高级特征

    【完整流程】：
    1. 特征工程（统计、交互、多项式）
    2. 特征选择
    3. 数据划分

    Args:
        X: 原始特征
        y: 目标变量
        task_type: 任务类型 ('classification' 或 'regression')
        create_interactions: 是否创建交互特征
        create_polynomials: 是否创建多项式特征
        create_statistical: 是否创建统计特征
        target_encoding: 是否使用目标编码
        feature_selection: 是否进行特征选择
        top_k_features: 保留的特征数

    Returns:
        (X_train, y_train), (X_test, y_test), feature_names
    """
    print("="*60)
    print("高级特征工程")
    print("="*60)

    # 转换为DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

    if not isinstance(y, pd.Series):
        y = pd.Series(y, name='target')

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # ============================================
    # 1. 特征工程
    # ============================================
    feature_engineer = AdvancedFeatureEngineering()
    all_features = [X.copy()]

    if create_statistical and len(numeric_cols) > 1:
        stat_features = feature_engineer.create_statistical_features(X, numeric_cols)
        all_features.append(stat_features)

    if create_interactions and len(numeric_cols) > 1:
        interaction_features = feature_engineer.create_interaction_features(
            X, numeric_cols, max_interactions=20
        )
        all_features.append(interaction_features)

    if create_polynomials and len(numeric_cols) > 0:
        poly_features = feature_engineer.create_polynomial_features(
            X, numeric_cols, degree=2, max_features=5
        )
        all_features.append(poly_features)

    # 合并所有特征
    X_engineered = pd.concat(all_features, axis=1)

    print(f"\n特征工程后:")
    print(f"  原始特征数: {X.shape[1]}")
    print(f"  工程后特征数: {X_engineered.shape[1]}")

    # ============================================
    # 2. 特征选择
    # ============================================
    if feature_selection and X_engineered.shape[1] > top_k_features:
        selector = AdvancedFeatureSelector()

        # 转换为numpy数组
        X_array = X_engineered.values
        y_array = y.values

        # 选择特征
        selected_features, importance = selector.select_by_importance(
            X_array, y_array,
            X_engineered.columns.tolist(),
            top_k=top_k_features,
            method='xgboost' if task_type == 'classification' else 'mutual_info'
        )

        X_selected = X_engineered[selected_features]

        print(f"\n特征选择后:")
        print(f"  保留特征数: {len(selected_features)}")

    else:
        X_selected = X_engineered
        selected_features = X_engineered.columns.tolist()

    # ============================================
    # 3. 数据划分
    # ============================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=0.2,
        random_state=42,
        stratify=y if task_type == 'classification' else None
    )

    print(f"\n数据划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")

    return (X_train, y_train), (X_test, y_test), selected_features


if __name__ == '__main__':
    """测试数据处理模块"""
    print("="*60)
    print("XGBoost高级技巧 - 数据处理测试")
    print("="*60)

    # 加载数据
    X, y, feature_names = load_california_housing_data()

    # 转换为分类任务（简化测试）
    y_binary = (y > y.median()).astype(int)

    # 准备高级特征
    (X_train, y_train), (X_test, y_test), selected_features = \
        prepare_advanced_features(
            X, y_binary,
            task_type='classification',
            create_interactions=True,
            create_polynomials=True,
            create_statistical=True,
            feature_selection=True,
            top_k_features=50
        )

    print("\n✓ 数据处理测试通过！")
