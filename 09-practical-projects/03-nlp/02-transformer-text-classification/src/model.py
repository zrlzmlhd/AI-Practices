"""
Transformer文本分类模型

本模块实现基于Transformer的文本分类器，包括：
1. 简单Transformer分类器（入门级）
2. 改进的Transformer分类器（中级）
3. 高级Transformer分类器（高级）

每个模型都有详细的注释说明设计思路和参数选择。
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from transformer import TransformerEncoder
from attention import create_padding_mask


class TransformerTextClassifier:
    """
    Transformer文本分类器

    【是什么】：基于Transformer编码器的文本分类模型
    【做什么】：将文本序列分类为不同类别（如情感分析）
    【为什么】：
        - Transformer能捕获长距离依赖
        - 并行处理，训练速度快
        - 在文本分类任务上效果优秀

    模型结构：
        Input (词ID序列)
          ↓
        Transformer Encoder
          ↓
        Global Average Pooling / [CLS] Token
          ↓
        Dense Layer(s)
          ↓
        Output (分类概率)
    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 num_classes=2,
                 model_type='simple',
                 **kwargs):
        """
        初始化分类器

        Args:
            vocab_size: 词汇表大小
            max_len: 最大序列长度
            num_classes: 分类类别数
            model_type: 模型类型 ('simple', 'improved', 'advanced')
            **kwargs: 其他参数
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_classes = num_classes
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
                # 简单配置（入门级）
                # ============================================
                # 【适用场景】：
                #   - 小数据集（<10k样本）
                #   - 快速实验
                #   - 学习Transformer基础

                'num_layers': 2,        # 编码器层数
                # 【为什么=2】：
                #   - 2层足够学习基本模式
                #   - 训练速度快
                #   - 不容易过拟合

                'd_model': 128,         # 模型维度
                # 【为什么=128】：
                #   - 较小的维度，参数量少
                #   - 适合小数据集
                #   - 训练速度快

                'num_heads': 4,         # 注意力头数
                # 【为什么=4】：
                #   - d_model=128，每个头32维
                #   - 4个头学习不同模式
                #   - 计算量适中

                'd_ff': 512,            # 前馈网络维度
                # 【为什么=512】：
                #   - 通常是d_model的4倍
                #   - 提供足够的非线性能力

                'dropout_rate': 0.1,    # Dropout比率
                # 【为什么=0.1】：
                #   - 轻度正则化
                #   - 防止过拟合

                'pooling': 'avg',       # 池化方式
                # 【为什么='avg'】：
                #   - 全局平均池化
                #   - 简单有效
                #   - 考虑所有位置
            },

            'improved': {
                # ============================================
                # 改进配置（中级）
                # ============================================
                # 【适用场景】：
                #   - 中等数据集（10k-100k样本）
                #   - 追求更好性能
                #   - 有一定计算资源

                'num_layers': 4,        # 编码器层数
                # 【为什么=4】：
                #   - 更深的网络学习更复杂模式
                #   - 4层是性能和速度的平衡点

                'd_model': 256,         # 模型维度
                # 【为什么=256】：
                #   - 更大的表示能力
                #   - 适合中等数据集

                'num_heads': 8,         # 注意力头数
                # 【为什么=8】：
                #   - d_model=256，每个头32维
                #   - 8个头学习更多样的模式

                'd_ff': 1024,           # 前馈网络维度
                # 【为什么=1024】：
                #   - d_model的4倍
                #   - 更强的非线性能力

                'dropout_rate': 0.2,    # Dropout比率
                # 【为什么=0.2】：
                #   - 中等强度正则化
                #   - 防止过拟合

                'pooling': 'cls',       # 池化方式
                # 【为什么='cls'】：
                #   - 使用[CLS] token
                #   - 类似BERT的做法
                #   - 学习全局表示
            },

            'advanced': {
                # ============================================
                # 高级配置（高级）
                # ============================================
                # 【适用场景】：
                #   - 大数据集（>100k样本）
                #   - 追求最佳性能
                #   - 有充足计算资源

                'num_layers': 6,        # 编码器层数
                # 【为什么=6】：
                #   - BERT-base的层数
                #   - 学习深层语义

                'd_model': 512,         # 模型维度
                # 【为什么=512】：
                #   - 标准Transformer维度
                #   - 强大的表示能力

                'num_heads': 8,         # 注意力头数
                # 【为什么=8】：
                #   - d_model=512，每个头64维
                #   - 标准配置

                'd_ff': 2048,           # 前馈网络维度
                # 【为什么=2048】：
                #   - d_model的4倍
                #   - 标准Transformer配置

                'dropout_rate': 0.3,    # Dropout比率
                # 【为什么=0.3】：
                #   - 较强正则化
                #   - 大模型需要更强的正则化

                'pooling': 'attention', # 池化方式
                # 【为什么='attention'】：
                #   - 注意力池化
                #   - 自动学习重要位置
                #   - 最灵活的方式
            }
        }

        return configs.get(model_type, configs['simple'])

    def _build_model(self):
        """
        构建模型

        Returns:
            Keras模型
        """
        # ============================================
        # 输入层
        # ============================================
        # 【是什么】：词ID序列
        # 【形状】：(batch, seq_len)
        inputs = layers.Input(shape=(self.max_len,), dtype=tf.int32, name='input_ids')

        # ============================================
        # Transformer编码器
        # ============================================
        # 【是什么】：多层Transformer编码器
        # 【做什么】：将词序列编码为上下文表示
        encoder = TransformerEncoder(
            num_layers=self.config['num_layers'],
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            d_ff=self.config['d_ff'],
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            dropout_rate=self.config['dropout_rate']
        )

        # 创建padding掩码
        # 【是什么】：遮蔽padding位置
        # 【为什么】：padding位置不应该被关注
        mask = create_padding_mask(inputs)

        # 编码
        encoder_output = encoder(inputs, mask=mask)
        # 形状: (batch, seq_len, d_model)

        # ============================================
        # 池化层
        # ============================================
        # 【是什么】：将序列表示转换为固定长度向量
        # 【为什么】：分类需要固定长度的输入

        pooling_type = self.config['pooling']

        if pooling_type == 'avg':
            # ============================================
            # 全局平均池化
            # ============================================
            # 【是什么】：对序列维度求平均
            # 【做什么】：将(batch, seq_len, d_model)变为(batch, d_model)
            # 【为什么】：
            #   - 简单有效
            #   - 考虑所有位置
            #   - 对序列长度不敏感
            pooled = layers.GlobalAveragePooling1D(name='global_avg_pooling')(encoder_output)

        elif pooling_type == 'max':
            # ============================================
            # 全局最大池化
            # ============================================
            # 【是什么】：对序列维度取最大值
            # 【为什么】：
            #   - 关注最显著的特征
            #   - 适合关键词检测
            pooled = layers.GlobalMaxPooling1D(name='global_max_pooling')(encoder_output)

        elif pooling_type == 'cls':
            # ============================================
            # [CLS] Token池化
            # ============================================
            # 【是什么】：使用第一个位置的输出
            # 【为什么】：
            #   - 类似BERT的做法
            #   - [CLS]位置学习全局表示
            #   - 需要在输入时添加[CLS] token
            pooled = encoder_output[:, 0, :]  # (batch, d_model)

        elif pooling_type == 'attention':
            # ============================================
            # 注意力池化
            # ============================================
            # 【是什么】：学习每个位置的权重
            # 【做什么】：加权求和
            # 【为什么】：
            #   - 自动学习重要位置
            #   - 最灵活的方式
            #   - 效果通常最好

            # 注意力权重
            attention = layers.Dense(1, activation='tanh', name='attention_weights')(encoder_output)
            attention = layers.Flatten()(attention)
            attention = layers.Activation('softmax', name='attention_softmax')(attention)
            attention = layers.RepeatVector(self.config['d_model'])(attention)
            attention = layers.Permute([2, 1])(attention)

            # 加权求和
            pooled = layers.Multiply()([encoder_output, attention])
            pooled = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_pooling')(pooled)

        else:
            # 默认使用平均池化
            pooled = layers.GlobalAveragePooling1D()(encoder_output)

        # ============================================
        # 分类头
        # ============================================
        # 【是什么】：全连接层 + 输出层
        # 【做什么】：将编码表示映射到类别概率

        # Dropout
        x = layers.Dropout(self.config['dropout_rate'], name='classifier_dropout')(pooled)

        # 中间层（可选）
        if self.model_type in ['improved', 'advanced']:
            # ============================================
            # 添加中间全连接层
            # ============================================
            # 【为什么】：
            #   - 增加非线性能力
            #   - 更好地适应分类任务
            x = layers.Dense(
                self.config['d_model'] // 2,
                activation='relu',
                name='classifier_hidden'
            )(x)
            x = layers.Dropout(self.config['dropout_rate'], name='classifier_dropout2')(x)

        # 输出层
        # 【是什么】：全连接层 + Softmax
        # 【做什么】：输出每个类别的概率
        if self.num_classes == 2:
            # 二分类：使用sigmoid
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            # 多分类：使用softmax
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        # 创建模型
        model = keras.Model(inputs=inputs, outputs=outputs, name=f'transformer_classifier_{self.model_type}')

        return model

    def compile_model(self, learning_rate=1e-4):
        """
        编译模型

        Args:
            learning_rate: 学习率
        """
        # ============================================
        # 优化器
        # ============================================
        # 【是什么】：Adam优化器
        # 【为什么】：
        #   - 自适应学习率
        #   - 对超参数不敏感
        #   - Transformer标准选择
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # ============================================
        # 损失函数
        # ============================================
        if self.num_classes == 2:
            # 二分类：binary crossentropy
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        else:
            # 多分类：categorical crossentropy
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train(self, X_train, y_train, X_val, y_val,
              epochs=10, batch_size=32, learning_rate=1e-4,
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
        predictions = self.model.predict(X)

        if self.num_classes == 2:
            # 二分类：返回0或1
            return (predictions > 0.5).astype(int).flatten()
        else:
            # 多分类：返回类别索引
            return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        """
        预测概率

        Args:
            X: 输入数据

        Returns:
            预测概率
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
    print("Transformer文本分类模型测试")
    print("="*60)

    # 测试参数
    vocab_size = 10000
    max_len = 128
    num_classes = 2
    batch_size = 4

    # 创建随机数据
    X_train = np.random.randint(0, vocab_size, (100, max_len))
    y_train = np.random.randint(0, num_classes, (100,))
    X_val = np.random.randint(0, vocab_size, (20, max_len))
    y_val = np.random.randint(0, num_classes, (20,))

    # 测试三种模型
    for model_type in ['simple', 'improved', 'advanced']:
        print(f"\n{'='*60}")
        print(f"测试 {model_type} 模型")
        print(f"{'='*60}")

        # 创建模型
        classifier = TransformerTextClassifier(
            vocab_size=vocab_size,
            max_len=max_len,
            num_classes=num_classes,
            model_type=model_type
        )

        # 打印摘要
        print(f"\n模型结构:")
        classifier.summary()

        # 训练
        print(f"\n训练模型...")
        history = classifier.train(
            X_train, y_train,
            X_val, y_val,
            epochs=2,
            batch_size=batch_size,
            verbose=0
        )

        # 评估
        metrics = classifier.evaluate(X_val, y_val)
        print(f"\n验证集性能:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # 预测
        predictions = classifier.predict(X_val[:5])
        print(f"\n预测结果: {predictions}")

    print("\n✓ 所有测试通过！")
