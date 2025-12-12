"""
Transformer NER模型

本模块实现：
1. 基础Transformer NER模型
2. Transformer + CRF模型
3. BiLSTM + CRF模型（对比）

每个模型都有详细的注释说明设计思路和参数选择。
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.metrics import classification_report, f1_score


class TransformerNERModel:
    """
    Transformer NER模型

    【是什么】：基于Transformer的命名实体识别模型
    【做什么】：序列标注，为每个词预测实体标签
    【为什么】：
        - Transformer捕获长距离依赖
        - 自注意力机制理解上下文
        - 适合序列标注任务
    """

    def __init__(self,
                 vocab_size,
                 num_tags,
                 max_len=128,
                 model_type='transformer',
                 **kwargs):
        """
        初始化NER模型

        Args:
            vocab_size: 词汇表大小
            num_tags: 标签数量
            max_len: 最大序列长度
            model_type: 模型类型
                - 'transformer': Transformer编码器
                - 'transformer_crf': Transformer + CRF
                - 'bilstm_crf': BiLSTM + CRF
            **kwargs: 其他参数
        """
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.max_len = max_len
        self.model_type = model_type

        # 配置
        self.config = self._get_model_config(model_type)
        self.config.update(kwargs)

        # 模型
        self.model = self._build_model()

    def _get_model_config(self, model_type):
        """获取模型配置"""
        configs = {
            'transformer': {
                # ============================================
                # Transformer配置
                # ============================================
                'd_model': 128,
                # 【为什么=128】：
                #   - NER任务相对简单
                #   - 128维足够捕获实体特征
                #   - 训练速度快

                'num_heads': 4,
                # 【为什么=4】：
                #   - d_model=128，每个头32维
                #   - 4个头学习不同的上下文模式

                'num_layers': 2,
                # 【为什么=2】：
                #   - 2层足够学习实体边界
                #   - 避免过拟合

                'd_ff': 512,
                # 【为什么=512】：
                #   - d_model的4倍
                #   - 标准配置

                'dropout': 0.1,
                # 【为什么=0.1】：
                #   - 轻度正则化
                #   - NER数据通常不大

                'use_crf': False,
            },

            'transformer_crf': {
                # ============================================
                # Transformer + CRF配置
                # ============================================
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 3,
                # 【为什么=3】：
                #   - CRF会增加模型能力
                #   - 可以用更深的编码器

                'd_ff': 512,
                'dropout': 0.2,
                # 【为什么=0.2】：
                #   - 更深的模型需要更强正则化

                'use_crf': True,
                # 【为什么使用CRF】：
                #   - 考虑标签之间的依赖关系
                #   - B-PER后面不能直接跟I-ORG
                #   - 提升标注一致性
            },

            'bilstm_crf': {
                # ============================================
                # BiLSTM + CRF配置（对比基线）
                # ============================================
                'lstm_units': 128,
                # 【为什么=128】：
                #   - 与Transformer的d_model对齐
                #   - 公平对比

                'num_layers': 2,
                'dropout': 0.2,
                'use_crf': True,
                # 【为什么BiLSTM+CRF】：
                #   - 经典NER架构
                #   - 作为对比基线
            }
        }

        return configs.get(model_type, configs['transformer'])

    def _build_model(self):
        """构建模型"""
        # ============================================
        # 输入层
        # ============================================
        input_ids = layers.Input(shape=(self.max_len,), dtype=tf.int32, name='input_ids')
        mask = layers.Input(shape=(self.max_len,), dtype=tf.float32, name='mask')

        # ============================================
        # 词嵌入层
        # ============================================
        # 【是什么】：将词ID转换为稠密向量
        # 【为什么】：
        #   - 词嵌入捕获词的语义
        #   - 可训练的表示
        if self.model_type.startswith('transformer'):
            d_model = self.config['d_model']
            embedding = layers.Embedding(
                self.vocab_size,
                d_model,
                mask_zero=True,
                name='embedding'
            )(input_ids)

            # 位置编码
            # 【为什么需要】：
            #   - Transformer没有位置信息
            #   - 需要显式添加位置编码
            positions = tf.range(start=0, limit=self.max_len, delta=1)
            position_embedding = layers.Embedding(
                self.max_len,
                d_model,
                name='position_embedding'
            )(positions)

            x = embedding + position_embedding
            x = layers.Dropout(self.config['dropout'])(x)

            # ============================================
            # Transformer编码器层
            # ============================================
            for i in range(self.config['num_layers']):
                # Multi-Head Attention
                attention_output = layers.MultiHeadAttention(
                    num_heads=self.config['num_heads'],
                    key_dim=d_model // self.config['num_heads'],
                    dropout=self.config['dropout'],
                    name=f'attention_{i}'
                )(x, x, attention_mask=mask[:, tf.newaxis, tf.newaxis, :])

                attention_output = layers.Dropout(self.config['dropout'])(attention_output)
                x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

                # Feed Forward
                ffn_output = layers.Dense(
                    self.config['d_ff'],
                    activation='relu',
                    name=f'ffn_1_{i}'
                )(x)
                ffn_output = layers.Dropout(self.config['dropout'])(ffn_output)
                ffn_output = layers.Dense(d_model, name=f'ffn_2_{i}')(ffn_output)
                ffn_output = layers.Dropout(self.config['dropout'])(ffn_output)

                x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

        else:
            # ============================================
            # BiLSTM编码器
            # ============================================
            embedding = layers.Embedding(
                self.vocab_size,
                self.config['lstm_units'],
                mask_zero=True,
                name='embedding'
            )(input_ids)

            x = embedding

            for i in range(self.config['num_layers']):
                x = layers.Bidirectional(
                    layers.LSTM(
                        self.config['lstm_units'] // 2,
                        return_sequences=True,
                        dropout=self.config['dropout'],
                        name=f'lstm_{i}'
                    ),
                    name=f'bilstm_{i}'
                )(x)

        # ============================================
        # 输出层
        # ============================================
        if self.config.get('use_crf', False):
            # ============================================
            # CRF层
            # ============================================
            # 【是什么】：条件随机场
            # 【做什么】：考虑标签之间的转移概率
            # 【为什么】：
            #   - 标签有依赖关系（如B-PER后只能跟I-PER或O）
            #   - 提升标注一致性
            #   - 避免非法标签序列

            # Dense层输出logits
            logits = layers.Dense(self.num_tags, name='logits')(x)

            # CRF层
            crf = tfa.layers.CRF(self.num_tags, name='crf')
            outputs = crf(logits)

            # 创建模型
            model = keras.Model(
                inputs=[input_ids, mask],
                outputs=outputs,
                name=f'ner_{self.model_type}'
            )

            # 保存CRF层（用于解码）
            self.crf_layer = crf

        else:
            # ============================================
            # Softmax输出
            # ============================================
            # 【是什么】：每个位置独立预测标签
            # 【为什么】：
            #   - 简单直接
            #   - 不考虑标签依赖

            outputs = layers.Dense(
                self.num_tags,
                activation='softmax',
                name='output'
            )(x)

            model = keras.Model(
                inputs=[input_ids, mask],
                outputs=outputs,
                name=f'ner_{self.model_type}'
            )

        return model

    def compile_model(self, learning_rate=0.001):
        """编译模型"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        if self.config.get('use_crf', False):
            # CRF模型使用特殊的损失函数
            # 【是什么】：负对数似然损失
            # 【为什么】：
            #   - CRF需要考虑整个序列
            #   - 不能用普通的交叉熵
            self.model.compile(
                optimizer=optimizer,
                loss=self.crf_layer.loss,
                metrics=[self.crf_layer.accuracy]
            )
        else:
            # 普通模型使用交叉熵
            self.model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

    def train(self, X_train, y_train, mask_train,
              X_val, y_val, mask_val,
              epochs=50, batch_size=32, learning_rate=0.001,
              callbacks=None, verbose=1):
        """
        训练模型

        Args:
            X_train: 训练输入
            y_train: 训练标签
            mask_train: 训练掩码
            X_val: 验证输入
            y_val: 验证标签
            mask_val: 验证掩码
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
            [X_train, mask_train],
            y_train,
            validation_data=([X_val, mask_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, X, mask):
        """
        预测

        Args:
            X: 输入序列
            mask: 掩码

        Returns:
            预测的标签
        """
        if self.config.get('use_crf', False):
            # CRF模型需要特殊的解码
            # 【是什么】：Viterbi解码
            # 【为什么】：
            #   - 找到最优标签序列
            #   - 考虑转移概率
            predictions = self.model.predict([X, mask])
        else:
            # 普通模型直接argmax
            predictions = self.model.predict([X, mask])
            predictions = np.argmax(predictions, axis=-1)

        return predictions

    def evaluate(self, X, y, mask):
        """评估模型"""
        results = self.model.evaluate([X, mask], y, verbose=0)

        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value

        return metrics

    def calculate_f1_score(self, y_true, y_pred, mask, idx2tag):
        """
        计算F1分数

        【重要】：NER任务通常使用实体级别的F1分数

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            mask: 掩码
            idx2tag: 标签映射

        Returns:
            F1分数字典
        """
        # 展平并过滤填充位置
        y_true_flat = []
        y_pred_flat = []

        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                if mask[i][j] == 1:  # 非填充位置
                    y_true_flat.append(y_true[i][j])
                    y_pred_flat.append(y_pred[i][j])

        # 转换为标签名称
        y_true_tags = [idx2tag.get(idx, 'O') for idx in y_true_flat]
        y_pred_tags = [idx2tag.get(idx, 'O') for idx in y_pred_flat]

        # 计算F1分数
        f1_micro = f1_score(y_true_tags, y_pred_tags, average='micro')
        f1_macro = f1_score(y_true_tags, y_pred_tags, average='macro')

        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro
        }

    def save_model(self, filepath):
        """保存模型"""
        self.model.save(filepath)
        print(f"✓ 模型已保存: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        if self.config.get('use_crf', False):
            # CRF模型需要自定义对象
            self.model = keras.models.load_model(
                filepath,
                custom_objects={'CRF': tfa.layers.CRF}
            )
        else:
            self.model = keras.models.load_model(filepath)
        print(f"✓ 模型已加载: {filepath}")

    def summary(self):
        """打印模型摘要"""
        self.model.summary()


if __name__ == '__main__':
    """测试模型"""
    print("="*60)
    print("Transformer NER模型测试")
    print("="*60)

    # 测试参数
    vocab_size = 1000
    num_tags = 9
    max_len = 50
    batch_size = 4

    # 创建随机数据
    X_train = np.random.randint(0, vocab_size, (100, max_len))
    y_train = np.random.randint(0, num_tags, (100, max_len))
    mask_train = np.ones((100, max_len), dtype=np.float32)

    X_val = np.random.randint(0, vocab_size, (20, max_len))
    y_val = np.random.randint(0, num_tags, (20, max_len))
    mask_val = np.ones((20, max_len), dtype=np.float32)

    # 测试三种模型
    for model_type in ['transformer', 'transformer_crf', 'bilstm_crf']:
        print(f"\n{'='*60}")
        print(f"测试 {model_type} 模型")
        print(f"{'='*60}")

        # 创建模型
        ner_model = TransformerNERModel(
            vocab_size=vocab_size,
            num_tags=num_tags,
            max_len=max_len,
            model_type=model_type
        )

        # 打印摘要
        print(f"\n模型结构:")
        ner_model.summary()

        # 训练
        print(f"\n训练模型...")
        history = ner_model.train(
            X_train, y_train, mask_train,
            X_val, y_val, mask_val,
            epochs=2,
            batch_size=batch_size,
            verbose=0
        )

        # 评估
        metrics = ner_model.evaluate(X_val, y_val, mask_val)
        print(f"\n验证集性能:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # 预测
        predictions = ner_model.predict(X_val[:2], mask_val[:2])
        print(f"\n预测形状: {predictions.shape}")
        print(f"预测示例: {predictions[0][:10]}")

    print("\n✓ 所有测试通过！")
