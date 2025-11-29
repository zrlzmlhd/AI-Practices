"""
情感分析模型定义
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed


class SentimentPredictor:
    """情感分析预测器"""

    def __init__(self, model_type='lstm', max_words=10000, max_len=200, random_state=42):
        """
        初始化预测器

        Args:
            model_type: 模型类型
            max_words: 最大词汇量
            max_len: 最大序列长度
            random_state: 随机种子
        """
        self.model_type = model_type
        self.max_words = max_words
        self.max_len = max_len
        self.random_state = random_state
        self.model = None
        self.tokenizer = None

        set_seed(random_state)

    def create_lstm_model(self, embedding_dim=128):
        """创建LSTM模型"""
        model = models.Sequential([
            layers.Embedding(self.max_words, embedding_dim, input_length=self.max_len),
            layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ], name='lstm_sentiment')

        return model

    def create_bilstm_model(self, embedding_dim=128):
        """创建BiLSTM模型"""
        model = models.Sequential([
            layers.Embedding(self.max_words, embedding_dim, input_length=self.max_len),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ], name='bilstm_sentiment')

        return model

    def create_cnn_model(self, embedding_dim=128):
        """创建CNN模型"""
        model = models.Sequential([
            layers.Embedding(self.max_words, embedding_dim, input_length=self.max_len),
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ], name='cnn_sentiment')

        return model

    def create_model(self):
        """创建模型"""
        if self.model_type == 'lstm':
            model = self.create_lstm_model()
        elif self.model_type == 'bilstm':
            model = self.create_bilstm_model()
        elif self.model_type == 'cnn':
            model = self.create_cnn_model()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=10, batch_size=32, callbacks=None):
        """训练模型"""
        print(f"\n开始训练模型: {self.model_type}")

        self.model = self.create_model()
        self.model.summary()

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✓ 模型训练完成")
        return history

    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")

        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")

        return self.model.predict(X).flatten()

    def evaluate(self, X, y):
        """评估模型"""
        if self.model is None:
            raise ValueError("模型未训练")

        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        return {'loss': loss, 'accuracy': accuracy}

    def save_model(self, filepath):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"✓ 模型已保存: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        self.model = keras.models.load_model(filepath)
        print(f"✓ 模型已加载: {filepath}")


if __name__ == '__main__':
    print("=" * 60)
    print("情感分析模型测试")
    print("=" * 60)

    # 创建模型
    predictor = SentimentPredictor(model_type='lstm')
    model = predictor.create_model()

    print("\n✓ 模型创建成功！")
