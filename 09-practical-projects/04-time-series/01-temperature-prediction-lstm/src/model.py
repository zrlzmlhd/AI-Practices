"""
LSTM温度预测模型

本模块实现三种LSTM架构：
1. 简单LSTM（单层）
2. 堆叠LSTM（多层）
3. GRU模型（对比）

每个模型都有详细的注释说明设计思路和参数选择。
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TemperatureLSTMPredictor:
    """
    LSTM温度预测器

    【是什么】：基于LSTM的时间序列预测模型
    【做什么】：预测未来24小时的温度
    【为什么】：
        - LSTM能捕获长期依赖关系
        - 适合处理时间序列数据
        - 可以同时考虑多个气象特征
    """

    def __init__(self,
                 input_shape,
                 forecast_horizon=24,
                 model_type='stacked',
                 **kwargs):
        """
        初始化预测器

        Args:
            input_shape: 输入形状 (lookback, num_features)
            forecast_horizon: 预测范围（小时）
            model_type: 模型类型 ('simple', 'stacked', 'gru')
            **kwargs: 其他参数
        """
        self.input_shape = input_shape
        self.forecast_horizon = forecast_horizon
        self.model_type = model_type

        # 根据模型类型设置参数
        self.config = self._get_model_config(model_type)
        self.config.update(kwargs)

        # 创建模型
        self.model = self._build_model()

    def _get_model_config(self, model_type):
        """
        获取模型配置

        Args:
            model_type: 模型类型

        Returns:
            配置字典
        """
        configs = {
            'simple': {
                # ============================================
                # 简单LSTM配置（单层）
                # ============================================
                # 【适用场景】：
                #   - 快速实验
                #   - 简单的时间序列模式
                #   - 计算资源有限

                'lstm_units': [64],     # 单层LSTM
                # 【为什么=64】：
                #   - 单层需要足够的容量
                #   - 64个单元是常见的起点
                #   - 平衡性能和速度

                'dropout': 0.2,         # Dropout比率
                # 【为什么=0.2】：
                #   - 轻度正则化
                #   - 防止过拟合

                'dense_units': [32],    # 全连接层
                # 【为什么需要】：
                #   - LSTM输出需要映射到预测范围
                #   - 增加非线性能力
            },

            'stacked': {
                # ============================================
                # 堆叠LSTM配置（多层）
                # ============================================
                # 【适用场景】：
                #   - 复杂的时间序列模式
                #   - 需要捕获多层次特征
                #   - 追求更好性能

                'lstm_units': [128, 64, 32],  # 三层LSTM
                # 【为什么逐层递减】：
                #   - 第1层(128)：学习低级时间特征
                #     * 捕获小时级波动
                #     * 需要大容量记住细节
                #   - 第2层(64)：学习中级时间特征
                #     * 捕获日级变化
                #     * 特征更抽象，容量减半
                #   - 第3层(32)：学习高级时间特征
                #     * 捕获周级趋势
                #     * 最抽象，容量最小

                'dropout': 0.3,         # Dropout比率
                # 【为什么=0.3】：
                #   - 深层网络需要更强正则化
                #   - 防止过拟合

                'dense_units': [64, 32],  # 两层全连接
                # 【为什么需要两层】：
                #   - 第1层：整合LSTM特征
                #   - 第2层：映射到预测范围
            },

            'gru': {
                # ============================================
                # GRU配置（对比LSTM）
                # ============================================
                # 【GRU vs LSTM】：
                #   - GRU：2个门（更新门、重置门）
                #   - LSTM：3个门（遗忘门、输入门、输出门）
                #   - GRU参数更少，训练更快
                #   - LSTM表达能力更强

                'gru_units': [128, 64],  # 两层GRU
                # 【为什么用GRU】：
                #   - 参数量少30%
                #   - 训练速度快
                #   - 在某些任务上效果相当

                'dropout': 0.3,
                'dense_units': [32],
            }
        }

        return configs.get(model_type, configs['stacked'])

    def _build_model(self):
        """
        构建模型

        Returns:
            Keras模型
        """
        # ============================================
        # 输入层
        # ============================================
        # 【形状】：(batch, lookback, num_features)
        # 例如：(32, 168, 5) = 32个样本，168小时，5个特征
        inputs = layers.Input(shape=self.input_shape, name='input')

        x = inputs

        # ============================================
        # LSTM/GRU层
        # ============================================
        if self.model_type == 'gru':
            # GRU模型
            gru_units = self.config['gru_units']

            for i, units in enumerate(gru_units):
                # 【return_sequences】：
                #   - True: 返回所有时间步（用于堆叠）
                #   - False: 只返回最后一个时间步
                return_sequences = (i < len(gru_units) - 1)

                x = layers.GRU(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config['dropout'],
                    recurrent_dropout=0.0,  # 循环dropout（可选）
                    name=f'gru_{i+1}'
                )(x)

                # 【为什么不用recurrent_dropout】：
                #   - 会显著降低训练速度
                #   - 普通dropout通常就够了

        else:
            # LSTM模型（simple或stacked）
            lstm_units = self.config['lstm_units']

            for i, units in enumerate(lstm_units):
                # ============================================
                # LSTM层详解
                # ============================================
                # 【return_sequences】：
                #   - 最后一层：False（只返回最后的输出）
                #   - 其他层：True（返回序列给下一层）

                return_sequences = (i < len(lstm_units) - 1)

                x = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config['dropout'],
                    recurrent_dropout=0.0,
                    name=f'lstm_{i+1}'
                )(x)

                # ============================================
                # Dropout层
                # ============================================
                # 【是什么】：随机丢弃神经元
                # 【为什么】：
                #   - 防止过拟合
                #   - 增加泛化能力
                #   - 类似于集成学习
                if i < len(lstm_units) - 1:
                    # 只在中间层添加额外的Dropout
                    x = layers.Dropout(self.config['dropout'], name=f'dropout_{i+1}')(x)

        # ============================================
        # 全连接层
        # ============================================
        # 【是什么】：Dense层
        # 【做什么】：
        #   - 整合LSTM/GRU的输出
        #   - 映射到预测范围
        # 【为什么】：
        #   - LSTM输出是固定维度（如32）
        #   - 需要映射到forecast_horizon（如24）

        dense_units = self.config['dense_units']

        for i, units in enumerate(dense_units):
            x = layers.Dense(
                units,
                activation='relu',
                name=f'dense_{i+1}'
            )(x)

            # Dropout
            x = layers.Dropout(self.config['dropout'], name=f'dense_dropout_{i+1}')(x)

        # ============================================
        # 输出层
        # ============================================
        # 【是什么】：线性层（无激活函数）
        # 【为什么无激活】：
        #   - 回归任务，输出可以是任意实数
        #   - 温度可以是负数，不能用ReLU
        #   - 不需要限制范围，不用sigmoid/tanh

        outputs = layers.Dense(
            self.forecast_horizon,
            activation='linear',
            name='output'
        )(x)

        # 创建模型
        model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=f'temperature_lstm_{self.model_type}'
        )

        return model

    def compile_model(self, learning_rate=0.001, loss='mse'):
        """
        编译模型

        Args:
            learning_rate: 学习率
            loss: 损失函数
        """
        # ============================================
        # 优化器
        # ============================================
        # 【是什么】：Adam优化器
        # 【为什么】：
        #   - 自适应学习率
        #   - 对超参数不敏感
        #   - 时间序列预测的标准选择
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # ============================================
        # 损失函数
        # ============================================
        # 【MSE vs MAE】：
        #   - MSE (Mean Squared Error)：
        #     * 对大误差惩罚更重
        #     * 适合关注极端值
        #   - MAE (Mean Absolute Error)：
        #     * 对所有误差一视同仁
        #     * 对异常值更鲁棒

        # ============================================
        # 评估指标
        # ============================================
        metrics = [
            'mae',  # 平均绝对误差
            'mse',  # 均方误差
            keras.metrics.RootMeanSquaredError(name='rmse')  # 均方根误差
        ]

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train(self, X_train, y_train, X_val, y_val,
              epochs=50, batch_size=32, learning_rate=0.001,
              callbacks=None, verbose=1):
        """
        训练模型

        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            callbacks: 回调函数列表
            verbose: 详细程度

        Returns:
            训练历史
        """
        # 编译模型
        self.compile_model(learning_rate=learning_rate)

        # 训练
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, X):
        """
        预测

        Args:
            X: 输入数据

        Returns:
            预测结果
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        评估模型

        Args:
            X: 测试数据
            y: 测试标签

        Returns:
            评估指标字典
        """
        results = self.model.evaluate(X, y, verbose=0)

        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value

        return metrics

    def calculate_metrics(self, y_true, y_pred):
        """
        计算详细评估指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            指标字典
        """
        # 展平数组（如果是多步预测）
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        metrics = {
            'mae': mean_absolute_error(y_true_flat, y_pred_flat),
            'mse': mean_squared_error(y_true_flat, y_pred_flat),
            'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
        }

        # 计算MAPE（平均绝对百分比误差）
        # 【注意】：避免除以0
        mask = y_true_flat != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
            metrics['mape'] = mape

        return metrics

    def save_model(self, filepath):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        self.model.save(filepath)
        print(f"✓ 模型已保存: {filepath}")

    def load_model(self, filepath):
        """
        加载模型

        Args:
            filepath: 模型路径
        """
        self.model = keras.models.load_model(filepath)
        print(f"✓ 模型已加载: {filepath}")

    def summary(self):
        """打印模型摘要"""
        self.model.summary()


if __name__ == '__main__':
    """
    测试模型
    """
    print("="*60)
    print("LSTM温度预测模型测试")
    print("="*60)

    # 测试参数
    lookback = 168  # 7天
    num_features = 5
    forecast_horizon = 24  # 预测24小时
    batch_size = 32

    # 输入形状
    input_shape = (lookback, num_features)

    # 创建随机数据
    X_train = np.random.randn(1000, lookback, num_features)
    y_train = np.random.randn(1000, forecast_horizon)
    X_val = np.random.randn(200, lookback, num_features)
    y_val = np.random.randn(200, forecast_horizon)

    # 测试三种模型
    for model_type in ['simple', 'stacked', 'gru']:
        print(f"\n{'='*60}")
        print(f"测试 {model_type} 模型")
        print(f"{'='*60}")

        # 创建模型
        predictor = TemperatureLSTMPredictor(
            input_shape=input_shape,
            forecast_horizon=forecast_horizon,
            model_type=model_type
        )

        # 打印摘要
        print(f"\n模型结构:")
        predictor.summary()

        # 训练
        print(f"\n训练模型...")
        history = predictor.train(
            X_train, y_train,
            X_val, y_val,
            epochs=2,
            batch_size=batch_size,
            verbose=0
        )

        # 评估
        metrics = predictor.evaluate(X_val, y_val)
        print(f"\n验证集性能:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # 预测
        predictions = predictor.predict(X_val[:5])
        print(f"\n预测形状: {predictions.shape}")
        print(f"预测示例: {predictions[0][:5]}")

    print("\n✓ 所有测试通过！")
