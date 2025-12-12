"""
LSTM情感分析模型定义

实现三种LSTM架构：
1. simple_lstm: 单向LSTM
2. bilstm: 双向LSTM
3. stacked_lstm: 堆叠LSTM
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed


class LSTMSentimentAnalyzer:
    """
    LSTM情感分析器

    支持三种模型架构：
    - simple_lstm: 单层LSTM（入门级，平衡效果和效率）
    - bilstm: 双向LSTM（中级，理解完整上下文）
    - stacked_lstm: 堆叠LSTM（高级，学习深层语义）
    """

    def __init__(self, model_type='simple_lstm', max_words=10000, max_len=200, random_state=42):
        """
        初始化情感分析器

        Args:
            model_type: 模型类型 ['simple_lstm', 'bilstm', 'stacked_lstm']
            max_words: 词汇表大小
            max_len: 序列最大长度
            random_state: 随机种子
        """
        self.model_type = model_type
        self.max_words = max_words
        self.max_len = max_len
        self.random_state = random_state
        self.model = None
        self.history = None

        set_seed(random_state)

    def create_simple_lstm(self, embedding_dim=128):
        """
        创建单向LSTM模型

        架构: Embedding → LSTM → Dense → Dropout → Output

        适用场景:
        - 学习LSTM基础
        - 数据量较小
        - 快速原型验证

        参数说明:
        - Embedding: max_words个词，每个词用128维向量表示
        - LSTM: 128个单元，dropout=0.2防止过拟合
        - Dense: 64个神经元，ReLU激活
        - Dropout: 0.5正则化，防止全连接层过拟合
        - Output: sigmoid激活，输出0-1概率

        Args:
            embedding_dim: 词嵌入维度

        Returns:
            Keras模型
        """
        model = models.Sequential(name='simple_lstm')

        # 词嵌入层: 将词索引转换为稠密向量
        model.add(layers.Embedding(
            input_dim=self.max_words,
            output_dim=embedding_dim,
            input_length=self.max_len,
            name='embedding'
        ))

        # LSTM层: 处理序列，记住长期依赖
        # dropout: 输入dropout，防止过度依赖某些词
        # recurrent_dropout: 时序dropout，防止时序过拟合
        model.add(layers.LSTM(
            units=128,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=False,
            name='lstm'
        ))

        # 全连接层: 特征组合与降维
        model.add(layers.Dense(
            units=64,
            activation='relu',
            name='dense'
        ))

        # Dropout层: 50%丢弃率，强力正则化
        model.add(layers.Dropout(
            rate=0.5,
            name='dropout'
        ))

        # 输出层: 二分类sigmoid激活
        model.add(layers.Dense(
            units=1,
            activation='sigmoid',
            name='output'
        ))

        return model

    def create_bilstm(self, embedding_dim=128):
        """
        创建双向LSTM模型

        架构: Embedding → BiLSTM → Dense → Dropout → Output

        优势:
        - 同时看前后文，理解更全面
        - 例如"不好看"，双向能同时看到"不"和"看"

        适用场景:
        - 需要理解完整上下文
        - 数据量充足
        - 对准确率要求较高

        注意:
        - 双向LSTM参数量是单向的2倍
        - 训练时间约为单向的1.5倍

        Args:
            embedding_dim: 词嵌入维度

        Returns:
            Keras模型
        """
        model = models.Sequential(name='bilstm')

        # 词嵌入层
        model.add(layers.Embedding(
            input_dim=self.max_words,
            output_dim=embedding_dim,
            input_length=self.max_len,
            name='embedding'
        ))

        # 双向LSTM层
        # 前向处理: "这" → "部" → "电影" → "不" → "好"
        # 后向处理: "好" → "不" → "电影" → "部" → "这"
        # 输出: 拼接前向和后向结果 (64+64=128维)
        model.add(layers.Bidirectional(
            layers.LSTM(
                units=64,  # 每个方向64单元，拼接后128维
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=False
            ),
            name='bilstm'
        ))

        # 全连接层、Dropout、输出层
        model.add(layers.Dense(64, activation='relu', name='dense'))
        model.add(layers.Dropout(0.5, name='dropout'))
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        return model

    def create_stacked_lstm(self, embedding_dim=128):
        """
        创建堆叠LSTM模型

        架构: Embedding → LSTM → LSTM → Dense → Dropout → Output

        优势:
        - 多层LSTM学习更复杂的模式
        - 第一层: 学习低级特征（词组）
        - 第二层: 学习高级特征（句子结构）

        适用场景:
        - 复杂的文本理解任务
        - 数据量大
        - 需要捕获深层语义

        注意:
        - 第一层return_sequences=True，返回完整序列
        - 第二层return_sequences=False，只返回最后输出

        Args:
            embedding_dim: 词嵌入维度

        Returns:
            Keras模型
        """
        model = models.Sequential(name='stacked_lstm')

        # 词嵌入层
        model.add(layers.Embedding(
            input_dim=self.max_words,
            output_dim=embedding_dim,
            input_length=self.max_len,
            name='embedding'
        ))

        # 第一层LSTM: 学习词组级别特征
        # return_sequences=True: 返回所有时间步，传递给下一层
        model.add(layers.LSTM(
            units=128,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=True,  # 返回完整序列
            name='lstm_1'
        ))

        # 第二层LSTM: 学习句子级别特征
        # return_sequences=False: 只返回最后输出，用于分类
        model.add(layers.LSTM(
            units=64,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=False,  # 只返回最后一个输出
            name='lstm_2'
        ))

        # 全连接层、Dropout、输出层
        model.add(layers.Dense(64, activation='relu', name='dense'))
        model.add(layers.Dropout(0.5, name='dropout'))
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        return model

    def create_model(self):
        """
        根据model_type创建对应的模型

        Returns:
            编译好的Keras模型
        """
        print(f"\n{'='*60}")
        print(f"Creating model: {self.model_type}")
        print(f"{'='*60}")

        if self.model_type == 'simple_lstm':
            model = self.create_simple_lstm()
        elif self.model_type == 'bilstm':
            model = self.create_bilstm()
        elif self.model_type == 'stacked_lstm':
            model = self.create_stacked_lstm()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # 编译模型
        # - 优化器: Adam，自适应学习率，训练稳定
        # - 损失函数: binary_crossentropy，二分类标准损失
        # - 评估指标: accuracy，直观易懂
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=10, batch_size=32, callbacks=None):
        """
        训练模型

        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            callbacks: 回调函数列表

        Returns:
            训练历史
        """
        print(f"\nStarting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Sequence length: {self.max_len}")
        print(f"Vocabulary size: {self.max_words}")

        # 创建模型
        self.model = self.create_model()

        # 打印模型结构
        print(f"\nModel architecture:")
        self.model.summary()

        # 准备验证数据
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            print(f"Validation samples: {len(X_val)}")

        # 训练模型
        print(f"\nTraining (epochs={epochs}, batch_size={batch_size})...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✓ Training completed")

        return self.history

    def predict(self, X):
        """
        预测类别

        Args:
            X: 输入数据

        Returns:
            预测类别（0或1）
        """
        if self.model is None:
            raise ValueError("Model not trained. Please call train() first")

        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """
        预测概率

        Args:
            X: 输入数据

        Returns:
            预测概率（0-1之间）
        """
        if self.model is None:
            raise ValueError("Model not trained. Please call train() first")

        return self.model.predict(X).flatten()

    def evaluate(self, X, y):
        """
        评估模型

        Args:
            X: 测试数据
            y: 测试标签

        Returns:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("Model not trained. Please call train() first")

        loss, accuracy = self.model.evaluate(X, y, verbose=0)

        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def save_model(self, filepath):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("Model not trained. Cannot save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(filepath)
        print(f"✓ Model saved: {filepath}")

    def load_model(self, filepath):
        """
        加载模型

        Args:
            filepath: 模型路径
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded: {filepath}")


def get_callbacks(model_path, patience=5):
    """
    获取训练回调函数

    Args:
        model_path: 模型保存路径
        patience: 早停耐心值

    Returns:
        回调函数列表
    """
    callbacks = [
        # 早停：验证集loss不再下降时停止训练
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),

        # 保存最佳模型
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        # 学习率调度：验证集loss不下降时降低学习率
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    return callbacks


if __name__ == '__main__':
    """测试模型创建"""
    print("="*60)
    print("Testing LSTM Sentiment Analysis Models")
    print("="*60)

    # 测试三种模型
    for model_type in ['simple_lstm', 'bilstm', 'stacked_lstm']:
        print(f"\n\n{'='*60}")
        print(f"Testing model: {model_type}")
        print(f"{'='*60}")

        analyzer = LSTMSentimentAnalyzer(model_type=model_type)
        model = analyzer.create_model()

        # 构建模型以便计算参数
        model.build(input_shape=(None, analyzer.max_len))

        print(f"\n✓ {model_type} model created successfully")
        print(f"Total parameters: {model.count_params():,}")

    print("\n\n✓ All models tested successfully")
