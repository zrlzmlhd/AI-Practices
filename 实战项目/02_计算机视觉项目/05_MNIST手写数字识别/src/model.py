"""
MNIST模型定义
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


class MNISTPredictor:
    """MNIST手写数字识别预测器"""

    def __init__(self, model_type='simple_cnn', random_state=42):
        """
        初始化预测器

        Args:
            model_type: 模型类型
            random_state: 随机种子
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.history = None

        set_seed(random_state)

    def create_simple_cnn(self, input_shape=(28, 28, 1), num_classes=10):
        """
        创建简单的CNN模型

        Args:
            input_shape: 输入形状
            num_classes: 类别数

        Returns:
            Keras模型
        """
        model = models.Sequential([
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),

            # 第二个卷积块
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # 全连接层
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ], name='simple_cnn')

        return model

    def create_improved_cnn(self, input_shape=(28, 28, 1), num_classes=10):
        """
        创建改进的CNN模型（使用批标准化）

        Args:
            input_shape: 输入形状
            num_classes: 类别数

        Returns:
            Keras模型
        """
        model = models.Sequential([
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # 第二个卷积块
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # 全连接层
            layers.Flatten(),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ], name='improved_cnn')

        return model

    def create_deep_cnn(self, input_shape=(28, 28, 1), num_classes=10):
        """
        创建深度CNN模型

        Args:
            input_shape: 输入形状
            num_classes: 类别数

        Returns:
            Keras模型
        """
        model = models.Sequential([
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),

            # 第二个卷积块
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),

            # 第三个卷积块
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            # 全局平均池化
            layers.GlobalAveragePooling2D(),

            # 全连接层
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ], name='deep_cnn')

        return model

    def create_model(self, input_shape=(28, 28, 1), num_classes=10):
        """
        创建模型

        Args:
            input_shape: 输入形状
            num_classes: 类别数

        Returns:
            Keras模型
        """
        if self.model_type == 'simple_cnn':
            model = self.create_simple_cnn(input_shape, num_classes)
        elif self.model_type == 'improved_cnn':
            model = self.create_improved_cnn(input_shape, num_classes)
        elif self.model_type == 'deep_cnn':
            model = self.create_deep_cnn(input_shape, num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        return model

    def compile_model(self, model, learning_rate=0.001):
        """
        编译模型

        Args:
            model: Keras模型
            learning_rate: 学习率
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=20, batch_size=128, callbacks=None):
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
        print(f"\n开始训练模型: {self.model_type}")
        print(f"训练样本数: {len(X_train)}")
        print(f"输入形状: {X_train.shape[1:]}")

        # 创建模型
        self.model = self.create_model(input_shape=X_train.shape[1:])
        self.model = self.compile_model(self.model)

        # 打印模型结构
        print("\n模型结构:")
        self.model.summary()

        # 验证数据
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✓ 模型训练完成")

        return self.history

    def predict(self, X):
        """
        预测

        Args:
            X: 输入数据

        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        # 确保输入形状正确
        if len(X.shape) == 2:
            X = X.reshape(-1, 28, 28, 1)
        elif len(X.shape) == 3:
            X = X.reshape(-1, 28, 28, 1)

        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        """
        预测概率

        Args:
            X: 输入数据

        Returns:
            预测概率
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        # 确保输入形状正确
        if len(X.shape) == 2:
            X = X.reshape(-1, 28, 28, 1)
        elif len(X.shape) == 3:
            X = X.reshape(-1, 28, 28, 1)

        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        评估模型

        Args:
            X: 测试数据
            y: 测试标签

        Returns:
            dict: 评估指标
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

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
            raise ValueError("模型未训练，无法保存")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(filepath)
        print(f"✓ 模型已保存: {filepath}")

    def load_model(self, filepath):
        """
        加载模型

        Args:
            filepath: 模型路径
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        self.model = keras.models.load_model(filepath)
        print(f"✓ 模型已加载: {filepath}")


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
        # 早停
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

        # 学习率调度
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
    from data import load_mnist_data

    print("=" * 60)
    print("MNIST模型测试")
    print("=" * 60)

    # 加载数据
    (X_train, y_train), (X_test, y_test) = load_mnist_data()

    # 使用小样本测试
    X_train_small = X_train[:1000]
    y_train_small = y_train[:1000]
    X_test_small = X_test[:200]
    y_test_small = y_test[:200]

    # 创建并训练模型
    predictor = MNISTPredictor(model_type='simple_cnn')
    predictor.train(
        X_train_small, y_train_small,
        X_test_small, y_test_small,
        epochs=3,
        batch_size=32
    )

    # 评估
    metrics = predictor.evaluate(X_test_small, y_test_small)
    print(f"\n测试集性能:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\n✓ 模型测试完成！")
