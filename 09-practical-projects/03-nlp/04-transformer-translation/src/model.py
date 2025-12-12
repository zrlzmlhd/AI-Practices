"""
Transformer机器翻译模型实现

基于"Attention is All You Need" (Vaswani et al., 2017)的完整Transformer架构
包含编码器、解码器以及位置编码等核心组件

架构组成：
- PositionalEncoding: 为序列注入位置信息
- TransformerEncoder: 多层自注意力编码器
- TransformerDecoder: 多层交叉注意力解码器
- TransformerTranslationModel: 完整的端到端翻译模型
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Optional, Tuple, Any


class PositionalEncoding(layers.Layer):
    """
    正弦位置编码层

    Transformer架构本身不包含序列顺序信息，位置编码用于为模型提供位置线索。
    使用正弦和余弦函数生成固定的位置编码：

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    优点：
    - 对任意长度序列都能生成唯一编码
    - 相对位置关系可以通过线性变换获得
    - 不需要训练，减少参数量
    """

    def __init__(self, max_len: int, d_model: int, **kwargs):
        """
        Args:
            max_len: 支持的最大序列长度
            d_model: 模型维度（必须是偶数）
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = self._generate_positional_encoding(max_len, d_model)

    def _generate_positional_encoding(self, max_len: int, d_model: int) -> tf.Tensor:
        """
        生成正弦位置编码矩阵

        Returns:
            形状为(1, max_len, d_model)的位置编码张量
        """
        # 位置索引 [0, 1, 2, ..., max_len-1]
        positions = np.arange(max_len)[:, np.newaxis]

        # 维度索引 [0, 1, 2, ..., d_model-1]
        dimensions = np.arange(d_model)[np.newaxis, :]

        # 计算角度
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates

        # 偶数维度使用sin，奇数维度使用cos
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Args:
            x: 输入张量，形状为(batch_size, seq_len, d_model)

        Returns:
            添加位置编码后的张量
        """
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


class TransformerEncoder(layers.Layer):
    """
    Transformer编码器

    由N个相同的编码器层堆叠而成，每层包含：
    1. Multi-Head Self-Attention: 捕获序列内部依赖关系
    2. Position-wise Feed-Forward Network: 非线性变换
    3. Residual Connection + Layer Normalization: 稳定训练

    编码器的作用是将源语言序列转换为上下文感知的表示向量。
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_len: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Args:
            num_layers: 编码器层数
            d_model: 模型维度
            num_heads: 多头注意力的头数
            d_ff: 前馈网络隐藏层维度
            vocab_size: 词汇表大小
            max_len: 最大序列长度
            dropout_rate: Dropout比率
        """
        super(TransformerEncoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        # 词嵌入层
        self.embedding = layers.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(max_len, d_model)

        # 构建多层编码器
        self.encoder_layers = [
            self._build_encoder_layer(d_model, num_heads, d_ff, dropout_rate, i)
            for i in range(num_layers)
        ]

        self.dropout = layers.Dropout(dropout_rate)

    def _build_encoder_layer(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float,
        layer_id: int
    ) -> Dict[str, layers.Layer]:
        """
        构建单个编码器层

        Returns:
            包含该层所有子层的字典
        """
        # Multi-Head Self-Attention
        mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'encoder_mha_{layer_id}'
        )

        # Position-wise Feed-Forward Network
        ffn = keras.Sequential([
            layers.Dense(d_ff, activation='relu', name=f'encoder_ffn1_{layer_id}'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model, name=f'encoder_ffn2_{layer_id}')
        ], name=f'encoder_ffn_{layer_id}')

        # Layer Normalization
        layernorm1 = layers.LayerNormalization(epsilon=1e-6, name=f'encoder_ln1_{layer_id}')
        layernorm2 = layers.LayerNormalization(epsilon=1e-6, name=f'encoder_ln2_{layer_id}')

        # Dropout
        dropout1 = layers.Dropout(dropout_rate, name=f'encoder_dropout1_{layer_id}')
        dropout2 = layers.Dropout(dropout_rate, name=f'encoder_dropout2_{layer_id}')

        return {
            'mha': mha,
            'ffn': ffn,
            'layernorm1': layernorm1,
            'layernorm2': layernorm2,
            'dropout1': dropout1,
            'dropout2': dropout2
        }

    def call(
        self,
        x: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        编码器前向传播

        Args:
            x: 输入序列（词ID），形状为(batch_size, seq_len)
            mask: 填充掩码，形状为(batch_size, 1, 1, seq_len)
            training: 是否为训练模式

        Returns:
            编码后的表示，形状为(batch_size, seq_len, d_model)
        """
        # 词嵌入 + 缩放
        # 缩放因子sqrt(d_model)用于平衡嵌入和位置编码的量级
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 添加位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        # 通过所有编码器层
        for layer_dict in self.encoder_layers:
            # 子层1: Multi-Head Self-Attention
            attn_output = layer_dict['mha'](x, x, attention_mask=mask)
            attn_output = layer_dict['dropout1'](attn_output, training=training)
            x = layer_dict['layernorm1'](x + attn_output)  # 残差连接

            # 子层2: Feed-Forward Network
            ffn_output = layer_dict['ffn'](x, training=training)
            ffn_output = layer_dict['dropout2'](ffn_output, training=training)
            x = layer_dict['layernorm2'](x + ffn_output)  # 残差连接

        return x


class TransformerDecoder(layers.Layer):
    """
    Transformer解码器

    由N个相同的解码器层堆叠而成，每层包含：
    1. Masked Multi-Head Self-Attention: 自回归地处理目标序列
    2. Multi-Head Cross-Attention: 关注编码器输出
    3. Position-wise Feed-Forward Network: 非线性变换
    4. Residual Connection + Layer Normalization: 稳定训练

    解码器采用自回归方式生成翻译，每次生成一个词，
    并通过掩码机制确保只能看到已生成的词。
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_len: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Args:
            num_layers: 解码器层数
            d_model: 模型维度
            num_heads: 多头注意力的头数
            d_ff: 前馈网络隐藏层维度
            vocab_size: 词汇表大小
            max_len: 最大序列长度
            dropout_rate: Dropout比率
        """
        super(TransformerDecoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        # 词嵌入层
        self.embedding = layers.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(max_len, d_model)

        # 构建多层解码器
        self.decoder_layers = [
            self._build_decoder_layer(d_model, num_heads, d_ff, dropout_rate, i)
            for i in range(num_layers)
        ]

        self.dropout = layers.Dropout(dropout_rate)

    def _build_decoder_layer(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float,
        layer_id: int
    ) -> Dict[str, layers.Layer]:
        """
        构建单个解码器层

        Returns:
            包含该层所有子层的字典
        """
        # Masked Multi-Head Self-Attention
        # 确保预测位置i时只能使用位置<i的信息
        masked_mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'decoder_masked_mha_{layer_id}'
        )

        # Multi-Head Cross-Attention
        # Query来自解码器，Key和Value来自编码器
        # 实现源语言和目标语言之间的对齐
        cross_mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'decoder_cross_mha_{layer_id}'
        )

        # Position-wise Feed-Forward Network
        ffn = keras.Sequential([
            layers.Dense(d_ff, activation='relu', name=f'decoder_ffn1_{layer_id}'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model, name=f'decoder_ffn2_{layer_id}')
        ], name=f'decoder_ffn_{layer_id}')

        # Layer Normalization
        layernorm1 = layers.LayerNormalization(epsilon=1e-6, name=f'decoder_ln1_{layer_id}')
        layernorm2 = layers.LayerNormalization(epsilon=1e-6, name=f'decoder_ln2_{layer_id}')
        layernorm3 = layers.LayerNormalization(epsilon=1e-6, name=f'decoder_ln3_{layer_id}')

        # Dropout
        dropout1 = layers.Dropout(dropout_rate, name=f'decoder_dropout1_{layer_id}')
        dropout2 = layers.Dropout(dropout_rate, name=f'decoder_dropout2_{layer_id}')
        dropout3 = layers.Dropout(dropout_rate, name=f'decoder_dropout3_{layer_id}')

        return {
            'masked_mha': masked_mha,
            'cross_mha': cross_mha,
            'ffn': ffn,
            'layernorm1': layernorm1,
            'layernorm2': layernorm2,
            'layernorm3': layernorm3,
            'dropout1': dropout1,
            'dropout2': dropout2,
            'dropout3': dropout3
        }

    def call(
        self,
        x: tf.Tensor,
        encoder_output: tf.Tensor,
        look_ahead_mask: Optional[tf.Tensor] = None,
        padding_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        解码器前向传播

        Args:
            x: 目标序列（词ID），形状为(batch_size, seq_len)
            encoder_output: 编码器输出，形状为(batch_size, src_len, d_model)
            look_ahead_mask: 前瞻掩码，防止看到未来词
            padding_mask: 填充掩码，忽略填充位置
            training: 是否为训练模式

        Returns:
            解码后的表示，形状为(batch_size, seq_len, d_model)
        """
        # 词嵌入 + 缩放
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 添加位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        # 通过所有解码器层
        for layer_dict in self.decoder_layers:
            # 子层1: Masked Self-Attention
            # 保证自回归特性：生成第i个词时只能看到前i-1个词
            attn1 = layer_dict['masked_mha'](
                x, x,
                attention_mask=look_ahead_mask
            )
            attn1 = layer_dict['dropout1'](attn1, training=training)
            x = layer_dict['layernorm1'](x + attn1)

            # 子层2: Cross-Attention
            # 连接编码器和解码器，获取源语言上下文
            attn2 = layer_dict['cross_mha'](
                x, encoder_output,
                attention_mask=padding_mask
            )
            attn2 = layer_dict['dropout2'](attn2, training=training)
            x = layer_dict['layernorm2'](x + attn2)

            # 子层3: Feed-Forward Network
            ffn_output = layer_dict['ffn'](x, training=training)
            ffn_output = layer_dict['dropout3'](ffn_output, training=training)
            x = layer_dict['layernorm3'](x + ffn_output)

        return x


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Transformer论文中的学习率调度策略

    实现Warmup预热策略，结合平方根衰减：
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    训练初期（Warmup阶段）：学习率线性增长，帮助模型稳定
    训练后期（Decay阶段）：学习率衰减，提高收敛精度

    Args:
        d_model: 模型维度，用于缩放学习率基准值
        warmup_steps: 预热步数，默认4000步
    """

    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """计算当前步的学习率"""
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        """返回配置字典，用于序列化"""
        return {
            'd_model': int(self.d_model.numpy()),
            'warmup_steps': self.warmup_steps
        }

    @classmethod
    def from_config(cls, config):
        """从配置字典创建实例"""
        return cls(**config)


class TransformerTranslationModel:
    """
    完整的Transformer机器翻译模型

    端到端的序列到序列模型，包含：
    - 编码器：理解源语言
    - 解码器：生成目标语言
    - 输出层：预测下一个词的概率分布

    训练方式：Teacher Forcing
    推理方式：贪心解码或Beam Search
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_len: int = 50,
        num_layers: int = 4,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        初始化翻译模型

        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            max_len: 最大序列长度
            num_layers: 编码器/解码器层数
            d_model: 模型维度
            num_heads: 注意力头数（d_model必须能被num_heads整除）
            d_ff: 前馈网络维度（通常为4*d_model）
            dropout_rate: Dropout比率
        """
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # 构建模型
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        """
        构建完整的Transformer模型

        模型结构：
        源语言输入 -> Encoder -> 编码表示
                                    |
        目标语言输入 -> Decoder <---+ -> 输出概率分布

        Returns:
            Keras Model实例
        """
        # 定义输入层（接受可变长度序列）
        encoder_inputs = layers.Input(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs'
        )

        decoder_inputs = layers.Input(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_inputs'
        )

        # 构建编码器
        encoder = TransformerEncoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            vocab_size=self.src_vocab_size,
            max_len=self.max_len,
            dropout_rate=self.dropout_rate,
            name='encoder'
        )

        # 编码源语言（不传入掩码，让编码器内部处理）
        encoder_output = encoder(encoder_inputs, training=True)

        # 构建解码器
        decoder = TransformerDecoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            vocab_size=self.tgt_vocab_size,
            max_len=self.max_len,
            dropout_rate=self.dropout_rate,
            name='decoder'
        )

        # 解码目标语言（不传入掩码，让解码器内部处理）
        decoder_output = decoder(
            decoder_inputs,
            encoder_output,
            training=True
        )

        # 输出层：将解码器输出映射到词汇表概率分布
        outputs = layers.Dense(
            self.tgt_vocab_size,
            activation='softmax',
            name='output_layer'
        )(decoder_output)

        # 构建模型
        model = keras.Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=outputs,
            name='transformer_translation'
        )

        return model

    def _create_padding_mask(self, seq: tf.Tensor) -> tf.Tensor:
        """
        创建填充掩码

        将填充位置（ID=0）标记为1，其他位置标记为0

        Args:
            seq: 输入序列

        Returns:
            掩码张量，形状为(batch_size, 1, 1, seq_len)
        """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def _create_look_ahead_mask(self, size: int) -> tf.Tensor:
        """
        创建前瞻掩码（下三角掩码）

        确保位置i只能attend到位置<=i的词

        Args:
            size: 序列长度

        Returns:
            掩码张量，形状为(1, 1, size, size)
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask[tf.newaxis, tf.newaxis, :, :]

    def compile_model(self, learning_rate: Optional[float] = None):
        """
        编译模型

        使用Transformer论文中的学习率调度策略：
        - Warmup阶段：学习率线性增长
        - 衰减阶段：学习率按平方根倒数衰减

        损失函数使用稀疏交叉熵，忽略填充位置的损失

        Args:
            learning_rate: 固定学习率（可选），不使用则采用自定义调度
        """
        # 选择学习率
        if learning_rate is None:
            lr_schedule = CustomSchedule(self.d_model)
        else:
            lr_schedule = learning_rate

        # Adam优化器（使用论文推荐的参数）
        optimizer = keras.optimizers.Adam(
            lr_schedule,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )

        # 带掩码的损失函数
        def masked_loss(y_true, y_pred):
            """
            忽略填充位置的损失
            """
            loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=False,
                reduction='none'
            )
            loss = loss_fn(y_true, y_pred)

            # 创建掩码：非填充位置为1，填充位置为0
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
            loss *= mask

            return tf.reduce_sum(loss) / tf.reduce_sum(mask)

        # 带掩码的准确率
        def masked_accuracy(y_true, y_pred):
            """
            忽略填充位置的准确率
            """
            y_pred = tf.argmax(y_pred, axis=-1)
            y_pred = tf.cast(y_pred, y_true.dtype)

            match = tf.cast(tf.equal(y_true, y_pred), tf.float32)
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

            return tf.reduce_sum(match * mask) / tf.reduce_sum(mask)

        # 编译模型
        self.model.compile(
            optimizer=optimizer,
            loss=masked_loss,
            metrics=[masked_accuracy]
        )

    def train(
        self,
        src_train: np.ndarray,
        tgt_train: np.ndarray,
        src_val: np.ndarray,
        tgt_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        callbacks: Optional[List[keras.callbacks.Callback]] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        训练模型

        使用Teacher Forcing策略：
        - 解码器输入：目标序列去掉最后一个词 [<SOS>, w1, w2, ..., wn]
        - 解码器输出：目标序列去掉第一个词 [w1, w2, ..., wn, <EOS>]

        Args:
            src_train: 训练集源语言
            tgt_train: 训练集目标语言
            src_val: 验证集源语言
            tgt_val: 验证集目标语言
            epochs: 训练轮数
            batch_size: 批大小
            callbacks: 回调函数列表
            verbose: 日志详细程度

        Returns:
            训练历史对象
        """
        # 编译模型
        self.compile_model()

        # 准备Teacher Forcing数据
        decoder_input_train = tgt_train[:, :-1]
        decoder_output_train = tgt_train[:, 1:]

        decoder_input_val = tgt_val[:, :-1]
        decoder_output_val = tgt_val[:, 1:]

        # 训练
        history = self.model.fit(
            [src_train, decoder_input_train],
            decoder_output_train,
            validation_data=([src_val, decoder_input_val], decoder_output_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def translate(
        self,
        src_sequence: np.ndarray,
        tgt_word2idx: Dict[str, int],
        tgt_idx2word: Dict[int, str],
        max_len: Optional[int] = None
    ) -> str:
        """
        翻译单个句子（贪心解码）

        自回归生成：
        1. 从<SOS>开始
        2. 每步预测下一个词
        3. 将预测词加入序列
        4. 直到生成<EOS>或达到最大长度

        Args:
            src_sequence: 源语言序列（已编码）
            tgt_word2idx: 目标语言词到ID映射
            tgt_idx2word: 目标语言ID到词映射
            max_len: 最大生成长度

        Returns:
            翻译结果字符串
        """
        if max_len is None:
            max_len = self.max_len

        # 扩展batch维度
        src_sequence = tf.expand_dims(src_sequence, 0)

        # 初始化输出序列
        sos_id = tgt_word2idx['<SOS>']
        eos_id = tgt_word2idx['<EOS>']
        output_sequence = [sos_id]

        # 自回归生成
        for _ in range(max_len):
            # 准备解码器输入
            decoder_input = tf.expand_dims(output_sequence, 0)
            decoder_input = tf.pad(
                decoder_input,
                [[0, 0], [0, self.max_len - len(output_sequence)]]
            )

            # 预测下一个词
            predictions = self.model([src_sequence, decoder_input], training=False)

            # 取当前位置的预测结果
            predicted_id = tf.argmax(predictions[0, len(output_sequence) - 1, :]).numpy()

            # 如果生成<EOS>，停止生成
            if predicted_id == eos_id:
                break

            output_sequence.append(int(predicted_id))

        # 解码为文本（跳过<SOS>）
        translated_words = [
            tgt_idx2word.get(idx, '<UNK>')
            for idx in output_sequence[1:]
        ]

        return ' '.join(translated_words)

    def save_model(self, filepath: str):
        """保存模型权重"""
        self.model.save(filepath)
        print(f"模型已保存: {filepath}")

    def load_model(self, filepath: str):
        """加载模型权重"""
        self.model = keras.models.load_model(filepath)
        print(f"模型已加载: {filepath}")

    def summary(self):
        """打印模型结构摘要"""
        self.model.summary()


if __name__ == '__main__':
    """
    模块测试代码
    验证Transformer模型能否正常构建和训练
    """
    print(f"{'='*60}")
    print("Transformer翻译模型测试")
    print(f"{'='*60}")

    # 测试参数（小规模快速测试）
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_len = 20
    batch_size = 4
    num_samples = 100

    # 创建随机测试数据
    src_train = np.random.randint(1, src_vocab_size, (num_samples, max_len))
    tgt_train = np.random.randint(1, tgt_vocab_size, (num_samples, max_len))
    src_val = np.random.randint(1, src_vocab_size, (20, max_len))
    tgt_val = np.random.randint(1, tgt_vocab_size, (20, max_len))

    # 创建模型（小规模配置）
    print("\n创建模型...")
    translator = TransformerTranslationModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_len,
        num_layers=2,
        d_model=128,
        num_heads=4,
        d_ff=512
    )

    print("\n模型结构:")
    translator.summary()

    # 测试训练
    print("\n测试训练...")
    history = translator.train(
        src_train, tgt_train,
        src_val, tgt_val,
        epochs=2,
        batch_size=batch_size,
        verbose=1
    )

    print(f"\n{'='*60}")
    print("测试通过！")
    print(f"{'='*60}")
    print(f"最终训练损失: {history.history['loss'][-1]:.4f}")
    print(f"最终验证损失: {history.history['val_loss'][-1]:.4f}")
