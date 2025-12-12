"""
股票数据处理模块

本模块负责：
1. 加载股票数据
2. 计算技术指标
3. 创建滑动窗口序列
4. 数据归一化

每个步骤都有详细的注释说明。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle


class TechnicalIndicators:
    """
    技术指标计算器

    【是什么】：计算常用的股票技术指标
    【为什么】：
        - 原始价格信息有限
        - 技术指标包含趋势、动量、波动等信息
        - 提升模型预测能力
    """

    @staticmethod
    def calculate_ma(df, periods=[5, 10, 20, 60]):
        """
        计算移动平均线（Moving Average）

        【是什么】：过去N天的平均价格
        【为什么】：
            - 平滑价格波动
            - 识别趋势方向
            - MA5 > MA20: 上升趋势

        Args:
            df: DataFrame
            periods: 周期列表

        Returns:
            DataFrame with MA columns
        """
        for period in periods:
            df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
        return df

    @staticmethod
    def calculate_rsi(df, period=14):
        """
        计算相对强弱指标（RSI）

        【是什么】：衡量价格变动的速度和幅度
        【公式】：RSI = 100 - (100 / (1 + RS))
                 RS = 平均涨幅 / 平均跌幅
        【为什么】：
            - RSI > 70: 超买（可能下跌）
            - RSI < 30: 超卖（可能上涨）
            - 识别反转信号

        Args:
            df: DataFrame
            period: 周期

        Returns:
            DataFrame with RSI column
        """
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9):
        """
        计算MACD指标

        【是什么】：Moving Average Convergence Divergence
        【公式】：
            - MACD = EMA(12) - EMA(26)
            - Signal = EMA(MACD, 9)
            - Histogram = MACD - Signal
        【为什么】：
            - MACD > Signal: 买入信号
            - MACD < Signal: 卖出信号
            - 识别趋势变化

        Args:
            df: DataFrame
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期

        Returns:
            DataFrame with MACD columns
        """
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()

        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        return df

    @staticmethod
    def calculate_bollinger_bands(df, period=20, std_dev=2):
        """
        计算布林带（Bollinger Bands）

        【是什么】：价格的波动区间
        【公式】：
            - 中轨 = MA(20)
            - 上轨 = 中轨 + 2*标准差
            - 下轨 = 中轨 - 2*标准差
        【为什么】：
            - 价格触及上轨: 可能回调
            - 价格触及下轨: 可能反弹
            - 衡量波动性

        Args:
            df: DataFrame
            period: 周期
            std_dev: 标准差倍数

        Returns:
            DataFrame with Bollinger columns
        """
        df['BB_middle'] = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()

        df['BB_upper'] = df['BB_middle'] + (std * std_dev)
        df['BB_lower'] = df['BB_middle'] - (std * std_dev)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        return df

    @staticmethod
    def calculate_atr(df, period=14):
        """
        计算平均真实波幅（ATR）

        【是什么】：衡量价格波动性
        【公式】：TR = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
                 ATR = MA(TR, period)
        【为什么】：
            - ATR高: 波动大，风险高
            - ATR低: 波动小，风险低
            - 用于止损设置

        Args:
            df: DataFrame
            period: 周期

        Returns:
            DataFrame with ATR column
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=period).mean()
        return df

    @staticmethod
    def calculate_obv(df):
        """
        计算能量潮（OBV）

        【是什么】：On-Balance Volume，累积成交量
        【公式】：
            - 收盘价上涨: OBV += Volume
            - 收盘价下跌: OBV -= Volume
        【为什么】：
            - OBV上升: 买盘强劲
            - OBV下降: 卖盘强劲
            - 确认价格趋势

        Args:
            df: DataFrame

        Returns:
            DataFrame with OBV column
        """
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])

        df['OBV'] = obv
        return df

    @staticmethod
    def calculate_all_indicators(df):
        """
        计算所有技术指标

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicators
        """
        print("\n计算技术指标...")

        # 移动平均
        df = TechnicalIndicators.calculate_ma(df)
        print("  ✓ 移动平均线 (MA)")

        # RSI
        df = TechnicalIndicators.calculate_rsi(df)
        print("  ✓ 相对强弱指标 (RSI)")

        # MACD
        df = TechnicalIndicators.calculate_macd(df)
        print("  ✓ MACD指标")

        # 布林带
        df = TechnicalIndicators.calculate_bollinger_bands(df)
        print("  ✓ 布林带 (Bollinger Bands)")

        # ATR
        df = TechnicalIndicators.calculate_atr(df)
        print("  ✓ 平均真实波幅 (ATR)")

        # OBV
        df = TechnicalIndicators.calculate_obv(df)
        print("  ✓ 能量潮 (OBV)")

        return df


class StockDataProcessor:
    """
    股票数据处理器

    【是什么】：处理股票时间序列数据的工具类
    【做什么】：
        - 加载股票数据
        - 计算技术指标
        - 创建滑动窗口
        - 数据归一化
    """

    def __init__(self, data_path, target_column='Close'):
        """
        初始化数据处理器

        Args:
            data_path: 数据文件路径
            target_column: 目标列名
        """
        self.data_path = data_path
        self.target_column = target_column

        # 归一化器
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        # 数据
        self.df = None
        self.feature_names = None

    def load_data(self):
        """
        加载股票数据

        Returns:
            DataFrame
        """
        print("\n加载股票数据...")

        # 读取CSV
        self.df = pd.read_csv(self.data_path)

        # 确保有Date列
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)

        # 按日期排序
        self.df.sort_index(inplace=True)

        print(f"  数据形状: {self.df.shape}")
        print(f"  时间范围: {self.df.index[0]} 到 {self.df.index[-1]}")
        print(f"  列名: {self.df.columns.tolist()}")

        return self.df

    def create_features(self):
        """
        创建特征（包括技术指标）

        Returns:
            DataFrame with features
        """
        print("\n创建特征...")

        # 计算技术指标
        self.df = TechnicalIndicators.calculate_all_indicators(self.df)

        # 删除NaN值（技术指标计算会产生NaN）
        print(f"\n删除NaN值...")
        print(f"  删除前: {len(self.df)}")
        self.df = self.df.dropna()
        print(f"  删除后: {len(self.df)}")

        # 选择特征列
        # 基础特征：OHLCV
        base_features = ['Open', 'High', 'Low', 'Close', 'Volume']

        # 技术指标特征
        indicator_features = [col for col in self.df.columns
                            if col not in base_features and col != self.target_column]

        self.feature_names = base_features + indicator_features

        print(f"\n特征统计:")
        print(f"  基础特征: {len(base_features)}")
        print(f"  技术指标: {len(indicator_features)}")
        print(f"  总特征数: {len(self.feature_names)}")

        return self.df[self.feature_names]

    def normalize_data(self, X, y, train_split=0.7):
        """
        归一化数据

        【重要】：只在训练集上fit scaler

        Args:
            X: 特征
            y: 目标
            train_split: 训练集比例

        Returns:
            归一化后的X, y
        """
        print("\n归一化数据...")

        train_size = int(len(X) * train_split)

        # 在训练集上fit
        self.feature_scaler.fit(X[:train_size])
        self.target_scaler.fit(y[:train_size].reshape(-1, 1))

        # 转换所有数据
        X_normalized = self.feature_scaler.transform(X)
        y_normalized = self.target_scaler.transform(y.reshape(-1, 1)).flatten()

        print(f"  特征范围: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
        print(f"  目标范围: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")

        return X_normalized, y_normalized

    def create_sequences(self, X, y, lookback=60, forecast_horizon=1):
        """
        创建滑动窗口序列

        Args:
            X: 特征
            y: 目标
            lookback: 回看窗口
            forecast_horizon: 预测范围

        Returns:
            X_seq, y_seq, y_trend
        """
        print("\n创建滑动窗口序列...")
        print(f"  回看窗口: {lookback} 天")
        print(f"  预测范围: {forecast_horizon} 天")

        X_seq, y_seq, y_trend = [], [], []

        for i in range(lookback, len(X) - forecast_horizon + 1):
            # 输入序列
            X_seq.append(X[i-lookback:i])

            # 目标价格
            y_seq.append(y[i+forecast_horizon-1])

            # 趋势标签（多任务学习）
            # 【是什么】：涨(1)或跌(0)
            # 【为什么】：同时预测价格和趋势
            current_price = y[i-1]
            future_price = y[i+forecast_horizon-1]
            trend = 1 if future_price > current_price else 0
            y_trend.append(trend)

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        y_trend = np.array(y_trend)

        print(f"  生成序列数: {len(X_seq)}")
        print(f"  输入形状: {X_seq.shape}")
        print(f"  价格目标形状: {y_seq.shape}")
        print(f"  趋势目标形状: {y_trend.shape}")
        print(f"  趋势分布: 上涨={y_trend.sum()}, 下跌={len(y_trend)-y_trend.sum()}")

        return X_seq, y_seq, y_trend

    def split_data(self, X, y_price, y_trend, train_split=0.7, val_split=0.15):
        """
        划分数据集（保持时间顺序）

        Args:
            X: 输入序列
            y_price: 价格目标
            y_trend: 趋势目标
            train_split: 训练集比例
            val_split: 验证集比例

        Returns:
            训练集、验证集、测试集
        """
        print("\n划分数据集...")

        n_samples = len(X)
        train_size = int(n_samples * train_split)
        val_size = int(n_samples * val_split)

        # 训练集
        X_train = X[:train_size]
        y_price_train = y_price[:train_size]
        y_trend_train = y_trend[:train_size]

        # 验证集
        X_val = X[train_size:train_size+val_size]
        y_price_val = y_price[train_size:train_size+val_size]
        y_trend_val = y_trend[train_size:train_size+val_size]

        # 测试集
        X_test = X[train_size+val_size:]
        y_price_test = y_price[train_size+val_size:]
        y_trend_test = y_trend[train_size+val_size:]

        print(f"  训练集: {X_train.shape}")
        print(f"  验证集: {X_val.shape}")
        print(f"  测试集: {X_test.shape}")

        return (X_train, y_price_train, y_trend_train), \
               (X_val, y_price_val, y_trend_val), \
               (X_test, y_price_test, y_trend_test)

    def inverse_transform_price(self, y):
        """反归一化价格"""
        return self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    def save_processor(self, filepath):
        """保存数据处理器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'feature_names': self.feature_names
            }, f)
        print(f"✓ 数据处理器已保存: {filepath}")

    def load_processor(self, filepath):
        """加载数据处理器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.feature_scaler = data['feature_scaler']
            self.target_scaler = data['target_scaler']
            self.feature_names = data['feature_names']
        print(f"✓ 数据处理器已加载: {filepath}")


def prepare_stock_data(data_path,
                       lookback=60,
                       forecast_horizon=1,
                       train_split=0.7,
                       val_split=0.15):
    """
    准备股票数据

    Args:
        data_path: 数据文件路径
        lookback: 回看窗口
        forecast_horizon: 预测范围
        train_split: 训练集比例
        val_split: 验证集比例

    Returns:
        训练集、验证集、测试集、处理器
    """
    print("="*60)
    print("股票数据准备")
    print("="*60)

    # 创建处理器
    processor = StockDataProcessor(data_path)

    # 加载数据
    processor.load_data()

    # 创建特征
    X = processor.create_features()
    y = processor.df[processor.target_column].values

    # 归一化
    X_normalized, y_normalized = processor.normalize_data(X.values, y, train_split)

    # 创建序列
    X_seq, y_price, y_trend = processor.create_sequences(
        X_normalized, y_normalized,
        lookback, forecast_horizon
    )

    # 划分数据集
    train_data, val_data, test_data = processor.split_data(
        X_seq, y_price, y_trend,
        train_split, val_split
    )

    print("\n数据准备完成！")

    return train_data, val_data, test_data, processor


if __name__ == '__main__':
    """测试数据处理"""
    print("="*60)
    print("数据处理模块测试")
    print("="*60)

    # 创建模拟股票数据
    print("\n创建模拟数据...")
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)

    # 模拟价格走势
    price = 100
    prices = [price]
    for _ in range(499):
        change = np.random.randn() * 2
        price = max(price + change, 50)  # 价格不低于50
        prices.append(price)

    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 500)
    })

    # 保存临时文件
    temp_path = 'temp_stock_data.csv'
    df.to_csv(temp_path, index=False)

    try:
        # 测试数据处理
        train_data, val_data, test_data, processor = prepare_stock_data(
            temp_path,
            lookback=30,
            forecast_horizon=1
        )

        print("\n✓ 数据处理测试通过！")

    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n✓ 所有测试通过！")
