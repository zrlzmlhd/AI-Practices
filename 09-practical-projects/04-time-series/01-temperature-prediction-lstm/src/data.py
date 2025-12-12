"""
时间序列数据处理模块

本模块负责：
1. 加载Jena气候数据集
2. 数据预处理和归一化
3. 创建滑动窗口序列
4. 数据集划分

每个步骤都有详细的注释说明。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle


class TimeSeriesDataProcessor:
    """
    时间序列数据处理器

    【是什么】：处理时间序列数据的工具类
    【做什么】：
        - 加载和清洗数据
        - 特征工程
        - 创建滑动窗口
        - 数据归一化
    【为什么】：
        - 时间序列需要特殊的处理方式
        - 滑动窗口是时间序列预测的核心
        - 归一化提高模型性能
    """

    def __init__(self, data_path, target_column='T (degC)',
                 selected_features=None, sampling_rate=6):
        """
        初始化数据处理器

        Args:
            data_path: 数据文件路径
            target_column: 目标列名（要预测的变量）
            selected_features: 选择的特征列（None表示使用所有特征）
            sampling_rate: 采样率（每隔多少条记录取一条）
        """
        self.data_path = data_path
        self.target_column = target_column
        self.selected_features = selected_features
        self.sampling_rate = sampling_rate

        # 数据归一化器
        self.scaler = StandardScaler()

        # 数据
        self.df = None
        self.feature_names = None

    def load_data(self):
        """
        加载数据

        Returns:
            DataFrame
        """
        print("\n加载数据...")

        # ============================================
        # 步骤1: 读取CSV文件
        # ============================================
        # 【是什么】：Jena气候数据集
        # 【格式】：CSV文件，包含日期时间和14个气象特征
        self.df = pd.read_csv(self.data_path)

        print(f"  原始数据形状: {self.df.shape}")
        print(f"  时间跨度: {self.df['Date Time'].iloc[0]} 到 {self.df['Date Time'].iloc[-1]}")

        # ============================================
        # 步骤2: 处理日期时间
        # ============================================
        # 【是什么】：将字符串转换为datetime对象
        # 【为什么】：
        #   - 方便时间相关的操作
        #   - 可以提取时间特征（小时、星期等）
        if 'Date Time' in self.df.columns:
            self.df['Date Time'] = pd.to_datetime(self.df['Date Time'])
            self.df.set_index('Date Time', inplace=True)

        # ============================================
        # 步骤3: 降采样
        # ============================================
        # 【是什么】：每隔sampling_rate条记录取一条
        # 【为什么】：
        #   - 原始数据每10分钟一条，数据量太大
        #   - 降采样到每小时一条（sampling_rate=6）
        #   - 减少计算量，保留主要模式
        if self.sampling_rate > 1:
            self.df = self.df[::self.sampling_rate]
            print(f"  降采样后形状: {self.df.shape}")

        # ============================================
        # 步骤4: 选择特征
        # ============================================
        if self.selected_features is not None:
            # 使用指定的特征
            self.df = self.df[self.selected_features]
            self.feature_names = self.selected_features
        else:
            # 使用所有数值特征
            self.feature_names = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.df = self.df[self.feature_names]

        print(f"  使用特征数: {len(self.feature_names)}")
        print(f"  特征列表: {self.feature_names}")

        # ============================================
        # 步骤5: 处理缺失值
        # ============================================
        # 【是什么】：检查并处理NaN值
        # 【为什么】：
        #   - 缺失值会导致模型训练失败
        #   - 时间序列常用前向填充
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            print(f"  发现缺失值: {missing_count}")
            # 前向填充：用前一个值填充
            self.df.fillna(method='ffill', inplace=True)
            # 后向填充：处理开头的缺失值
            self.df.fillna(method='bfill', inplace=True)
            print(f"  缺失值已处理")

        return self.df

    def normalize_data(self, train_split=0.7):
        """
        归一化数据

        Args:
            train_split: 训练集比例（用于fit scaler）

        Returns:
            归一化后的数据
        """
        print("\n归一化数据...")

        # ============================================
        # 重要：只在训练集上fit scaler
        # ============================================
        # 【是什么】：StandardScaler标准化
        # 【为什么只用训练集】：
        #   - 防止数据泄露
        #   - 测试集应该用训练集的统计量
        #   - 模拟真实预测场景

        train_size = int(len(self.df) * train_split)
        train_data = self.df.iloc[:train_size].values

        # 在训练集上fit
        self.scaler.fit(train_data)

        # 转换所有数据
        normalized_data = self.scaler.transform(self.df.values)

        print(f"  归一化完成")
        print(f"  均值: {self.scaler.mean_[:3]}")  # 显示前3个特征的均值
        print(f"  标准差: {self.scaler.scale_[:3]}")  # 显示前3个特征的标准差

        return normalized_data

    def create_sequences(self, data, lookback=168, forecast_horizon=24, step=1):
        """
        创建滑动窗口序列

        【是什么】：将时间序列转换为监督学习问题
        【做什么】：
            - 输入：过去lookback个时间步的数据
            - 输出：未来forecast_horizon个时间步的目标值
        【为什么】：
            - LSTM需要固定长度的输入
            - 滑动窗口捕获时间依赖关系

        Args:
            data: 归一化后的数据
            lookback: 回看窗口大小（输入序列长度）
            forecast_horizon: 预测时间范围（输出序列长度）
            step: 滑动步长

        Returns:
            X, y: 输入序列和目标值

        示例：
            lookback=168 (7天 * 24小时)
            forecast_horizon=24 (预测未来24小时)

            输入: 过去7天的[温度, 湿度, 气压, ...]
            输出: 未来24小时的温度
        """
        print("\n创建滑动窗口序列...")
        print(f"  回看窗口: {lookback} 小时")
        print(f"  预测范围: {forecast_horizon} 小时")
        print(f"  滑动步长: {step}")

        X, y = [], []

        # ============================================
        # 滑动窗口算法
        # ============================================
        # 【原理】：
        #   时刻t: 使用 [t-lookback, t] 预测 [t+1, t+forecast_horizon]
        #   时刻t+step: 使用 [t+step-lookback, t+step] 预测 [t+step+1, t+step+forecast_horizon]

        # 找到目标列的索引
        target_idx = self.feature_names.index(self.target_column)

        for i in range(0, len(data) - lookback - forecast_horizon + 1, step):
            # 输入序列：过去lookback个时间步的所有特征
            # 【形状】：(lookback, num_features)
            X.append(data[i:i+lookback])

            # 输出序列：未来forecast_horizon个时间步的目标值
            # 【形状】：(forecast_horizon,)
            # 【注意】：只预测目标列（温度）
            y.append(data[i+lookback:i+lookback+forecast_horizon, target_idx])

        X = np.array(X)
        y = np.array(y)

        print(f"  生成序列数: {len(X)}")
        print(f"  输入形状: {X.shape}")  # (samples, lookback, features)
        print(f"  输出形状: {y.shape}")  # (samples, forecast_horizon)

        return X, y

    def split_data(self, X, y, train_split=0.7, val_split=0.15):
        """
        划分数据集

        【重要】：时间序列不能随机划分！
        【为什么】：
            - 必须保持时间顺序
            - 训练集在前，验证集在中，测试集在后
            - 模拟真实预测场景

        Args:
            X: 输入序列
            y: 目标值
            train_split: 训练集比例
            val_split: 验证集比例

        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        print("\n划分数据集...")

        n_samples = len(X)

        # ============================================
        # 按时间顺序划分
        # ============================================
        # 【示例】：
        #   总样本: 10000
        #   训练集: 0-7000 (70%)
        #   验证集: 7000-8500 (15%)
        #   测试集: 8500-10000 (15%)

        train_size = int(n_samples * train_split)
        val_size = int(n_samples * val_split)

        # 训练集
        X_train = X[:train_size]
        y_train = y[:train_size]

        # 验证集
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]

        # 测试集
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]

        print(f"  训练集: {X_train.shape}")
        print(f"  验证集: {X_val.shape}")
        print(f"  测试集: {X_test.shape}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def inverse_transform_target(self, y):
        """
        反归一化目标值

        【是什么】：将归一化的预测值转换回原始尺度
        【为什么】：
            - 评估时需要原始尺度的值
            - 便于理解预测结果（如温度20°C）

        Args:
            y: 归一化的目标值

        Returns:
            原始尺度的目标值
        """
        # 找到目标列的索引
        target_idx = self.feature_names.index(self.target_column)

        # 获取目标列的均值和标准差
        mean = self.scaler.mean_[target_idx]
        scale = self.scaler.scale_[target_idx]

        # 反归一化：y_original = y_normalized * scale + mean
        return y * scale + mean

    def save_scaler(self, filepath):
        """保存归一化器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'target_column': self.target_column
            }, f)
        print(f"✓ 归一化器已保存: {filepath}")

    def load_scaler(self, filepath):
        """加载归一化器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.target_column = data['target_column']
        print(f"✓ 归一化器已加载: {filepath}")


def prepare_temperature_data(data_path='data/jena_climate_2009_2016.csv',
                             lookback=168,
                             forecast_horizon=24,
                             train_split=0.7,
                             val_split=0.15,
                             sampling_rate=6,
                             selected_features=None):
    """
    准备温度预测数据

    Args:
        data_path: 数据文件路径
        lookback: 回看窗口大小（小时）
        forecast_horizon: 预测范围（小时）
        train_split: 训练集比例
        val_split: 验证集比例
        sampling_rate: 采样率
        selected_features: 选择的特征

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), processor
    """
    print("="*60)
    print("温度预测数据准备")
    print("="*60)

    # ============================================
    # 步骤1: 创建数据处理器
    # ============================================
    processor = TimeSeriesDataProcessor(
        data_path=data_path,
        target_column='T (degC)',
        selected_features=selected_features,
        sampling_rate=sampling_rate
    )

    # ============================================
    # 步骤2: 加载数据
    # ============================================
    processor.load_data()

    # ============================================
    # 步骤3: 归一化
    # ============================================
    normalized_data = processor.normalize_data(train_split=train_split)

    # ============================================
    # 步骤4: 创建序列
    # ============================================
    X, y = processor.create_sequences(
        normalized_data,
        lookback=lookback,
        forecast_horizon=forecast_horizon
    )

    # ============================================
    # 步骤5: 划分数据集
    # ============================================
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(
        X, y,
        train_split=train_split,
        val_split=val_split
    )

    # ============================================
    # 数据统计
    # ============================================
    print("\n数据统计:")
    print(f"  特征数: {X_train.shape[2]}")
    print(f"  输入序列长度: {X_train.shape[1]} 小时")
    print(f"  预测范围: {y_train.shape[1]} 小时")
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

    # 注意：需要先下载数据
    # 这里使用模拟数据进行测试

    # 创建模拟数据
    print("\n创建模拟数据...")
    n_samples = 1000
    n_features = 5

    # 模拟时间序列数据（带有趋势和周期性）
    t = np.arange(n_samples)
    data = np.zeros((n_samples, n_features))

    for i in range(n_features):
        # 趋势 + 周期 + 噪声
        trend = 0.01 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 24)  # 24小时周期
        noise = np.random.randn(n_samples)
        data[:, i] = trend + seasonal + noise

    # 创建DataFrame
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    df.columns = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)', 'wd (deg)']

    # 保存临时文件
    temp_path = 'temp_test_data.csv'
    df.to_csv(temp_path, index=False)

    # 测试数据处理
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), processor = prepare_temperature_data(
            data_path=temp_path,
            lookback=24,
            forecast_horizon=6,
            sampling_rate=1
        )

        print("\n✓ 数据处理测试通过！")

        # 测试反归一化
        print("\n测试反归一化...")
        y_original = processor.inverse_transform_target(y_test[0])
        print(f"  归一化值: {y_test[0][:5]}")
        print(f"  原始值: {y_original[:5]}")

    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n✓ 所有测试通过！")
