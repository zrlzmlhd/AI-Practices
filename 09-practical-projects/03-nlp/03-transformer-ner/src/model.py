"""
Transformer命名实体识别模型

本模块实现三种NER模型架构:
1. 基础Transformer编码器模型
2. Transformer + CRF层模型
3. BiLSTM + CRF模型(作为对比基线)

每个模型都包含详细的参数说明和设计思路注释。

技术说明:
- CRF层依赖tensorflow-addons包,如遇兼容性问题可使用基础模型
- 所有模型都支持掩码机制来处理填充位置
- 提供完整的训练、预测和评估接口
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except (ImportError, ModuleNotFoundError):
    HAS_TFA = False
    print("警告: tensorflow_addons未安装或不兼容，CRF功能将不可用")
from sklearn.metrics import classification_report, f1_score


class TransformerNERModel:
    """
    基于Transformer的命名实体识别模型

    实现完整的NER模型,包括:
    - Transformer编码器: 捕获长距离依赖和上下文信息
    - CRF层(可选): 建模标签之间的转移约束
    - BiLSTM编码器(可选): 作为传统模型对比基线

    Transformer通过自注意力机制能够有效捕获句子中的长距离依赖关系,
    相比传统RNN模型在处理长序列时更有优势。CRF层可以确保输出标签序列
    满足BIO标注规则(如B-PER后不能直接跟I-ORG)。
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
        """获取不同模型类型的配置参数"""
        configs = {
            'transformer': {
                # ============================================
                # Transformer编码器配置
                # ============================================
                'd_model': 128,
                # 模型维度设为128: NER任务相对简单,128维足够捕获实体特征,
                # 同时保证训练效率

                'num_heads': 4,
                # 注意力头数设为4: d_model=128时每个头32维,4个头可以学习
                # 不同的上下文模式(如实体边界、词性、语义等)

                'num_layers': 2,
                # 编码器层数设为2: 2层足够学习实体边界和类型特征,
                # 避免过深导致过拟合

                'd_ff': 512,
                # 前馈网络维度512: 标准配置为d_model的4倍

                'dropout': 0.1,
                # Dropout率0.1: 轻度正则化,适合小规模NER数据集

                'use_crf': False,
            },

            'transformer_crf': {
                # ============================================
                # Transformer + CRF配置
                # ============================================
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 3,
                # 增加到3层: CRF层已经提供了标签依赖建模能力,
                # 可以使用更深的编码器来提取更丰富的特征

                'd_ff': 512,
                'dropout': 0.2,
                # Dropout率0.2: 更深的模型需要更强的正则化

                'use_crf': True,
                # 使用CRF层: 建模标签转移概率,确保标签序列满足BIO约束
                # (如B-PER后只能跟I-PER或O,不能跟I-ORG)
            },

            'bilstm_crf': {
                # ============================================
                # BiLSTM + CRF配置(传统NER架构,用作对比基线)
                # ============================================
                'lstm_units': 128,
                # LSTM单元数128: 与Transformer的d_model对齐,保证公平对比

                'num_layers': 2,
                'dropout': 0.2,
                'use_crf': True,
                # BiLSTM+CRF是经典的NER架构,曾是该任务的主流方法
            }
        }

        return configs.get(model_type, configs['transformer'])

    def _build_model(self):
        """构建NER模型架构"""
        # ============================================
        # 输入层
        # ============================================
        input_ids = layers.Input(shape=(self.max_len,), dtype=tf.int32, name='input_ids')
        mask = layers.Input(shape=(self.max_len,), dtype=tf.float32, name='mask')

        # ============================================
        # 词嵌入层
        # ============================================
        # 将词ID转换为稠密向量表示,捕获词的语义信息
        # 使用mask_zero=True自动处理填充位置
        if self.model_type.startswith('transformer'):
            d_model = self.config['d_model']
            embedding = layers.Embedding(
                self.vocab_size,
                d_model,
                mask_zero=True,
                name='embedding'
            )(input_ids)

            # 位置编码: Transformer没有内置的位置信息,
            # 需要显式添加位置编码来提供序列顺序信息
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
            # CRF层(条件随机场)
            # ============================================
            # CRF层对整个标签序列建模,考虑标签之间的转移概率。
            # 这可以避免非法的标签序列(如B-PER后直接跟I-ORG),
            # 提升标注的一致性和准确率。

            # Dense层输出每个位置的标签logits
            logits = layers.Dense(self.num_tags, name='logits')(x)

            # CRF层学习标签转移矩阵
            crf = tfa.layers.CRF(self.num_tags, name='crf')
            outputs = crf(logits)

            # 创建模型
            model = keras.Model(
                inputs=[input_ids, mask],
                outputs=outputs,
                name=f'ner_{self.model_type}'
            )

            # 保存CRF层引用,用于后续的Viterbi解码
            self.crf_layer = crf

        else:
            # ============================================
            # Softmax输出层
            # ============================================
            # 每个位置独立预测标签,不考虑标签之间的依赖关系。
            # 实现简单直接,训练速度快。

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
        """编译模型,设置优化器和损失函数"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        if self.config.get('use_crf', False):
            # CRF模型使用负对数似然损失(Negative Log-Likelihood)
            # CRF需要对整个序列建模,不能使用普通的逐位置交叉熵损失
            self.model.compile(
                optimizer=optimizer,
                loss=self.crf_layer.loss,
                metrics=[self.crf_layer.accuracy]
            )
        else:
            # 基础模型使用稀疏分类交叉熵损失
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
        预测标签序列

        Args:
            X: 输入序列,shape=(batch_size, max_len)
            mask: 掩码矩阵,shape=(batch_size, max_len)

        Returns:
            predictions: 预测的标签ID,shape=(batch_size, max_len)
        """
        if self.config.get('use_crf', False):
            # CRF模型使用Viterbi算法解码最优标签序列
            # Viterbi算法通过动态规划找到全局最优的标签序列
            predictions = self.model.predict([X, mask])
        else:
            # 基础模型对每个位置独立取概率最大的标签
            predictions = self.model.predict([X, mask])
            predictions = np.argmax(predictions, axis=-1)

        return predictions

    def evaluate(self, X, y, mask):
        """评估模型性能"""
        results = self.model.evaluate([X, mask], y, verbose=0)

        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value

        return metrics

    def calculate_f1_score(self, y_true, y_pred, mask, idx2tag):
        """
        计算F1分数

        NER任务通常使用实体级别的F1分数作为主要评估指标,
        相比准确率能更好地反映模型在实体识别上的性能。

        Args:
            y_true: 真实标签,shape=(batch_size, max_len)
            y_pred: 预测标签,shape=(batch_size, max_len)
            mask: 掩码矩阵,shape=(batch_size, max_len)
            idx2tag: 标签ID到标签名的映射

        Returns:
            F1分数字典,包含micro和macro平均
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
    model_types = ['transformer']
    if HAS_TFA:
        model_types.extend(['transformer_crf', 'bilstm_crf'])
    else:
        print("\n" + "="*60)
        print("注意: CRF模型测试已跳过(tensorflow-addons不可用)")
        print("="*60)

    for model_type in model_types:
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
