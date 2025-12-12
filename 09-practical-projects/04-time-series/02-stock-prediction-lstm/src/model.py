"""
LSTM股票预测模型

本模块实现：
1. 基础LSTM模型
2. LSTM + Attention模型
3. 多任务学习模型（价格+趋势）
4. 注意力机制可视化

每个模型都有详细的注释说明设计思路和参数选择。
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score


class AttentionLayer(layers.Layer):
    """
    注意力层

    【是什么】：自注意力机制
    【做什么】：学习每个时间步的重要性权重
    【为什么】：
        - 突出关键时间点（如财报日）
        - 提升预测准确性
        - 增加可解释性
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        构建层参数

        Args:
            input_shape: (batch, time_steps, features)
        """
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        """
        前向传播

        【步骤】：
        1. 计算注意力分数
        2. Softmax归一化
        3. 加权求和

        Args:
            x: 输入张量 (batch, time_steps, features)

        Returns:
            context: 上下文向量 (batch, features)
            attention_weights: 注意力权重 (batch, time_steps)
        """
        # ============================================
        # 步骤1: 计算注意力分数
        # ============================================
        # 【是什么】：每个时间步的重要性分数
        # 【公式】：score = tanh(x · W + b)
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        # 形状: (batch, time_steps, 1)

        # ============================================
        # 步骤2: Softmax归一化
        # ============================================
        # 【是什么】：将分数转换为概率分布
        # 【为什么】：权重和为1，便于解释
        attention_weights = tf.nn.softmax(e, axis=1)
        # 形状: (batch, time_steps, 1)

        # ============================================
        # 步骤3: 加权求和
        # ============================================
        # 【是什么】：用注意力权重对输入加权求和
        # 【为什么】：重点关注重要的时间步
        context = x * attention_weights
        context = tf.reduce_sum(context, axis=1)
        # 形状: (batch, features)

        return context, tf.squeeze(attention_weights, -1)


class StockLSTMPredictor:
    """
    LSTM股票预测器

    【是什么】：基于LSTM的股票价格预测模型
    【做什么】：预测未来股价和趋势
    【为什么】：
        - LSTM捕获时间依赖
        - 注意力机制突出关键时刻
        - 多任务学习提升泛化
    """

    def __init__(self,
                 input_shape,
                 model_type='lstm_attention',
                 **kwargs):
        """
        初始化预测器

        Args:
            input_shape: 输入形状 (time_steps, features)
            model_type: 模型类型
                - 'lstm_basic': 基础LSTM
                - 'lstm_attention': LSTM + Attention
                - 'multitask': 多任务学习
            **kwargs: 其他参数
        """
        self.input_shape = input_shape
        self.model_type = model_type

        # 配置
        self.config = self._get_model_config(model_type)
        self.config.update(kwargs)

        # 模型
        self.model = self._build_model()

    def _get_model_config(self, model_type):
        """获取模型配置"""
        configs = {
            'lstm_basic': {
                # ============================================
                # 基础LSTM配置
                # ============================================
                'lstm_units': [128, 64],
                # 【为什么两层】：
                #   - 第1层：学习低级时间模式
                #   - 第2层：学习高级时间模式

                'dropout': 0.2,
                # 【为什么=0.2】：
                #   - 轻度正则化
                #   - 防止过拟合

                'dense_units': [32],
                # 【为什么需要】：
                #   - 整合LSTM特征
                #   - 映射到输出
            },

            'lstm_attention': {
                # ============================================
                # LSTM + Attention配置
                # ============================================
                'lstm_units': [128, 64],
                'dropout': 0.3,
                # 【为什么=0.3】：
                #   - 注意力机制增加了模型复杂度
                #   - 需要更强的正则化

                'use_attention': True,
                # 【为什么使用注意力】：
                #   - 自动学习重要时间点
                #   - 提升预测准确性
                #   - 增加可解释性

                'dense_units': [64, 32],
            },

            'multitask': {
                # ============================================
                # 多任务学习配置
                # ============================================
                'lstm_units': [128, 64],
                'dropout': 0.3,
                'use_attention': True,

                'multitask': True,
                # 【是什么】：同时预测价格和趋势
                # 【为什么】：
                #   - 价格预测：回归任务
                #   - 趋势预测：分类任务
                #   - 共享特征提取，互相促进

                'dense_units': [64, 32],
            }
        }

        return configs.get(model_type, configs['lstm_attention'])

    def _build_model(self):
        """构建模型"""
        # ============================================
        # 输入层
        # ============================================
        inputs = layers.Input(shape=self.input_shape, name='input')

        x = inputs

        # ============================================
        # LSTM层
        # ============================================
        lstm_units = self.config['lstm_units']

        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1) or self.config.get('use_attention', False)

            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config['dropout'],
                name=f'lstm_{i+1}'
            )(x)

            if i < len(lstm_units) - 1:
                x = layers.Dropout(self.config['dropout'], name=f'dropout_{i+1}')(x)

        # ============================================
        # 注意力层（可选）
        # ============================================
        if self.config.get('use_attention', False):
            # 【是什么】：自注意力机制
            # 【做什么】：学习每个时间步的重要性
            # 【为什么】：
            #   - 突出关键时间点
            #   - 提升预测准确性
            context, attention_weights = AttentionLayer(name='attention')(x)
            x = context

            # 保存注意力权重（用于可视化）
            self.attention_model = keras.Model(inputs=inputs, outputs=attention_weights)

        # ============================================
        # 全连接层
        # ============================================
        dense_units = self.config['dense_units']

        for i, units in enumerate(dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.Dropout(self.config['dropout'], name=f'dense_dropout_{i+1}')(x)

        # ============================================
        # 输出层
        # ============================================
        if self.config.get('multitask', False):
            # ============================================
            # 多任务学习：价格 + 趋势
            # ============================================

            # 分支1：价格预测（回归）
            price_output = layers.Dense(1, activation='linear', name='price')(x)

            # 分支2：趋势预测（分类）
            trend_output = layers.Dense(1, activation='sigmoid', name='trend')(x)

            # 创建多输出模型
            model = keras.Model(
                inputs=inputs,
                outputs=[price_output, trend_output],
                name='stock_lstm_multitask'
            )

        else:
            # ============================================
            # 单任务：价格预测
            # ============================================
            output = layers.Dense(1, activation='linear', name='price')(x)

            model = keras.Model(
                inputs=inputs,
                outputs=output,
                name=f'stock_lstm_{self.model_type}'
            )

        return model

    def compile_model(self, learning_rate=0.001):
        """编译模型"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        if self.config.get('multitask', False):
            # 多任务学习
            self.model.compile(
                optimizer=optimizer,
                loss={
                    'price': 'mse',  # 价格：均方误差
                    'trend': 'binary_crossentropy'  # 趋势：二分类交叉熵
                },
                loss_weights={
                    'price': 1.0,  # 价格损失权重
                    'trend': 0.5   # 趋势损失权重（辅助任务）
                },
                metrics={
                    'price': ['mae', keras.metrics.RootMeanSquaredError(name='rmse')],
                    'trend': ['accuracy']
                }
            )
        else:
            # 单任务
            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
            )

    def train(self, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=32, learning_rate=0.001,
              callbacks=None, verbose=1):
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签（单任务）或 (y_price, y_trend)（多任务）
            X_val: 验证特征
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            callbacks: 回调函数
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
        """预测"""
        predictions = self.model.predict(X)

        if self.config.get('multitask', False):
            # 多任务：返回价格和趋势
            price_pred, trend_pred = predictions
            return price_pred.flatten(), (trend_pred > 0.5).astype(int).flatten()
        else:
            # 单任务：只返回价格
            return predictions.flatten()

    def predict_with_attention(self, X):
        """
        预测并返回注意力权重

        Args:
            X: 输入特征

        Returns:
            predictions: 预测结果
            attention_weights: 注意力权重
        """
        if not self.config.get('use_attention', False):
            raise ValueError("模型未使用注意力机制")

        predictions = self.model.predict(X)
        attention_weights = self.attention_model.predict(X)

        return predictions, attention_weights

    def evaluate(self, X, y):
        """评估模型"""
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
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        }

        # MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['mape'] = mape

        # 方向准确率
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            direction_accuracy = accuracy_score(true_direction, pred_direction)
            metrics['direction_accuracy'] = direction_accuracy

        return metrics

    def save_model(self, filepath):
        """保存模型"""
        self.model.save(filepath)
        print(f"✓ 模型已保存: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        self.model = keras.models.load_model(filepath, custom_objects={'AttentionLayer': AttentionLayer})
        print(f"✓ 模型已加载: {filepath}")

    def summary(self):
        """打印模型摘要"""
        self.model.summary()


if __name__ == '__main__':
    """测试模型"""
    print("="*60)
    print("LSTM股票预测模型测试")
    print("="*60)

    # 测试参数
    time_steps = 60
    n_features = 20
    batch_size = 32

    input_shape = (time_steps, n_features)

    # 创建随机数据
    X_train = np.random.randn(1000, time_steps, n_features)
    y_price_train = np.random.randn(1000)
    y_trend_train = np.random.randint(0, 2, 1000)

    X_val = np.random.randn(200, time_steps, n_features)
    y_price_val = np.random.randn(200)
    y_trend_val = np.random.randint(0, 2, 200)

    # 测试三种模型
    for model_type in ['lstm_basic', 'lstm_attention', 'multitask']:
        print(f"\n{'='*60}")
        print(f"测试 {model_type} 模型")
        print(f"{'='*60}")

        # 创建模型
        predictor = StockLSTMPredictor(
            input_shape=input_shape,
            model_type=model_type
        )

        # 打印摘要
        print(f"\n模型结构:")
        predictor.summary()

        # 准备训练数据
        if model_type == 'multitask':
            y_train = {'price': y_price_train, 'trend': y_trend_train}
            y_val = {'price': y_price_val, 'trend': y_trend_val}
        else:
            y_train = y_price_train
            y_val = y_price_val

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
        if model_type == 'multitask':
            price_pred, trend_pred = predictor.predict(X_val[:5])
            print(f"\n预测示例:")
            print(f"  价格: {price_pred}")
            print(f"  趋势: {trend_pred}")
        else:
            predictions = predictor.predict(X_val[:5])
            print(f"\n预测示例: {predictions}")

        # 测试注意力权重
        if model_type in ['lstm_attention', 'multitask']:
            print(f"\n测试注意力权重...")
            _, attention_weights = predictor.predict_with_attention(X_val[:1])
            print(f"  注意力权重形状: {attention_weights.shape}")
            print(f"  注意力权重示例: {attention_weights[0][:10]}")

    print("\n✓ 所有测试通过！")
