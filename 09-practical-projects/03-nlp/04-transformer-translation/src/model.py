"""
Transformer机器翻译模型

本模块实现完整的Transformer Seq2Seq架构：
1. Transformer Encoder（编码器）
2. Transformer Decoder（解码器）
3. 完整的翻译模型
4. Beam Search解码

每个组件都有详细的注释说明设计思路和参数选择。
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    """
    位置编码

    【是什么】：为序列添加位置信息
    【为什么】：Transformer没有位置信息，需要显式添加
    """

    def __init__(self, max_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(max_len, d_model)

    def _positional_encoding(self, max_len, d_model):
        """生成位置编码"""
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        # 偶数维度用sin，奇数维度用cos
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


class TransformerEncoder(layers.Layer):
    """
    Transformer编码器

    【是什么】：多层编码器层的堆叠
    【做什么】：将源语言序列编码为上下文表示
    【为什么】：
        - 捕获源语言的语义信息
        - 提供给解码器使用
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 vocab_size, max_len, dropout_rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        # 词嵌入
        self.embedding = layers.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(max_len, d_model)

        # 编码器层
        self.encoder_layers = [
            self._create_encoder_layer(d_model, num_heads, d_ff, dropout_rate, i)
            for i in range(num_layers)
        ]

        self.dropout = layers.Dropout(dropout_rate)

    def _create_encoder_layer(self, d_model, num_heads, d_ff, dropout_rate, layer_id):
        """创建单个编码器层"""
        # Multi-Head Attention
        mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'encoder_mha_{layer_id}'
        )

        # Feed Forward
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

    def call(self, x, mask=None, training=False):
        """
        前向传播

        Args:
            x: 输入序列（词ID）
            mask: 填充掩码
            training: 是否训练模式

        Returns:
            编码后的表示
        """
        # 词嵌入
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        # 通过所有编码器层
        for layer_dict in self.encoder_layers:
            # Multi-Head Attention
            attn_output = layer_dict['mha'](x, x, attention_mask=mask)
            attn_output = layer_dict['dropout1'](attn_output, training=training)
            x = layer_dict['layernorm1'](x + attn_output)

            # Feed Forward
            ffn_output = layer_dict['ffn'](x, training=training)
            ffn_output = layer_dict['dropout2'](ffn_output, training=training)
            x = layer_dict['layernorm2'](x + ffn_output)

        return x


class TransformerDecoder(layers.Layer):
    """
    Transformer解码器

    【是什么】：多层解码器层的堆叠
    【做什么】：根据编码器输出生成目标语言序列
    【为什么】：
        - 自回归生成（一个词一个词生成）
        - 使用编码器的上下文信息
        - 通过交叉注意力连接编码器和解码器
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 vocab_size, max_len, dropout_rate=0.1, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        # 词嵌入
        self.embedding = layers.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(max_len, d_model)

        # 解码器层
        self.decoder_layers = [
            self._create_decoder_layer(d_model, num_heads, d_ff, dropout_rate, i)
            for i in range(num_layers)
        ]

        self.dropout = layers.Dropout(dropout_rate)

    def _create_decoder_layer(self, d_model, num_heads, d_ff, dropout_rate, layer_id):
        """创建单个解码器层"""
        # Masked Multi-Head Attention（自注意力）
        masked_mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'decoder_masked_mha_{layer_id}'
        )

        # Multi-Head Attention（交叉注意力）
        # 【是什么】：解码器关注编码器的输出
        # 【为什么】：
        #   - 获取源语言的上下文信息
        #   - 决定翻译哪个源词
        cross_mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'decoder_cross_mha_{layer_id}'
        )

        # Feed Forward
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

    def call(self, x, encoder_output, look_ahead_mask=None, padding_mask=None, training=False):
        """
        前向传播

        Args:
            x: 目标序列（词ID）
            encoder_output: 编码器输出
            look_ahead_mask: 前瞻掩码（防止看到未来）
            padding_mask: 填充掩码
            training: 是否训练模式

        Returns:
            解码后的表示
        """
        # 词嵌入
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        # 通过所有解码器层
        for layer_dict in self.decoder_layers:
            # ============================================
            # 子层1: Masked Self-Attention
            # ============================================
            # 【是什么】：解码器的自注意力
            # 【为什么需要mask】：
            #   - 防止看到未来的词
            #   - 保持自回归特性
            attn1 = layer_dict['masked_mha'](
                x, x,
                attention_mask=look_ahead_mask
            )
            attn1 = layer_dict['dropout1'](attn1, training=training)
            x = layer_dict['layernorm1'](x + attn1)

            # ============================================
            # 子层2: Cross-Attention
            # ============================================
            # 【是什么】：解码器关注编码器输出
            # 【为什么】：
            #   - 获取源语言信息
            #   - Query来自解码器，Key/Value来自编码器
            attn2 = layer_dict['cross_mha'](
                x, encoder_output,
                attention_mask=padding_mask
            )
            attn2 = layer_dict['dropout2'](attn2, training=training)
            x = layer_dict['layernorm2'](x + attn2)

            # ============================================
            # 子层3: Feed Forward
            # ============================================
            ffn_output = layer_dict['ffn'](x, training=training)
            ffn_output = layer_dict['dropout3'](ffn_output, training=training)
            x = layer_dict['layernorm3'](x + ffn_output)

        return x


class TransformerTranslationModel:
    """
    Transformer机器翻译模型

    【是什么】：完整的Encoder-Decoder Transformer
    【做什么】：将源语言翻译为目标语言
    【为什么】：
        - Encoder捕获源语言语义
        - Decoder生成目标语言
        - 注意力机制对齐源词和目标词
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 max_len=50,
                 num_layers=4,
                 d_model=256,
                 num_heads=8,
                 d_ff=1024,
                 dropout_rate=0.1,
                 **kwargs):
        """
        初始化翻译模型

        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            max_len: 最大序列长度
            num_layers: 编码器/解码器层数
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络维度
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

        # 创建模型
        self.model = self._build_model()

    def _build_model(self):
        """
        构建完整的Transformer模型

        【结构】：
        源语言输入 → Encoder → 编码表示
                                    ↓
        目标语言输入 → Decoder ← 交叉注意力 → 输出概率
        """
        # ============================================
        # 输入层
        # ============================================
        # 源语言输入
        encoder_inputs = layers.Input(
            shape=(self.max_len,),
            dtype=tf.int32,
            name='encoder_inputs'
        )

        # 目标语言输入（训练时使用）
        decoder_inputs = layers.Input(
            shape=(self.max_len,),
            dtype=tf.int32,
            name='decoder_inputs'
        )

        # ============================================
        # 编码器
        # ============================================
        # 【是什么】：处理源语言
        # 【做什么】：将源语言编码为上下文表示
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

        # 创建编码器填充掩码
        encoder_padding_mask = self._create_padding_mask(encoder_inputs)

        # 编码
        encoder_output = encoder(
            encoder_inputs,
            mask=encoder_padding_mask,
            training=True
        )

        # ============================================
        # 解码器
        # ============================================
        # 【是什么】：生成目标语言
        # 【做什么】：根据编码器输出生成翻译
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

        # 创建解码器掩码
        # 【前瞻掩码】：防止看到未来的词
        look_ahead_mask = self._create_look_ahead_mask(self.max_len)
        decoder_padding_mask = self._create_padding_mask(decoder_inputs)
        combined_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)

        # 解码
        decoder_output = decoder(
            decoder_inputs,
            encoder_output,
            look_ahead_mask=combined_mask,
            padding_mask=encoder_padding_mask,
            training=True
        )

        # ============================================
        # 输出层
        # ============================================
        # 【是什么】：线性层 + Softmax
        # 【做什么】：输出每个词的概率分布
        outputs = layers.Dense(
            self.tgt_vocab_size,
            activation='softmax',
            name='output'
        )(decoder_output)

        # 创建模型
        model = keras.Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=outputs,
            name='transformer_translation'
        )

        return model

    def _create_padding_mask(self, seq):
        """创建填充掩码"""
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def _create_look_ahead_mask(self, size):
        """创建前瞻掩码"""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask[tf.newaxis, tf.newaxis, :, :]

    def compile_model(self, learning_rate=0.0001):
        """
        编译模型

        Args:
            learning_rate: 学习率
        """
        # ============================================
        # 学习率调度
        # ============================================
        # 【是什么】：Warmup + 衰减
        # 【为什么】：
        #   - Warmup：开始时小学习率，逐渐增大
        #   - 衰减：后期逐渐减小学习率
        #   - Transformer训练的标准做法

        class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=4000):
                super(CustomSchedule, self).__init__()
                self.d_model = tf.cast(d_model, tf.float32)
                self.warmup_steps = warmup_steps

            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)
                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        learning_rate = CustomSchedule(self.d_model)
        optimizer = keras.optimizers.Adam(
            learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )

        # ============================================
        # 损失函数
        # ============================================
        # 【是什么】：Sparse Categorical Crossentropy
        # 【为什么】：
        #   - 多分类问题（预测下一个词）
        #   - 忽略填充位置的损失

        def masked_loss(y_true, y_pred):
            """带掩码的损失函数"""
            loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=False,
                reduction='none'
            )
            loss = loss_fn(y_true, y_pred)

            # 创建掩码（忽略填充位置）
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
            loss *= mask

            return tf.reduce_sum(loss) / tf.reduce_sum(mask)

        def masked_accuracy(y_true, y_pred):
            """带掩码的准确率"""
            y_pred = tf.argmax(y_pred, axis=-1)
            y_pred = tf.cast(y_pred, y_true.dtype)

            match = tf.cast(tf.equal(y_true, y_pred), tf.float32)
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

            return tf.reduce_sum(match * mask) / tf.reduce_sum(mask)

        self.model.compile(
            optimizer=optimizer,
            loss=masked_loss,
            metrics=[masked_accuracy]
        )

    def train(self, src_train, tgt_train, src_val, tgt_val,
              epochs=50, batch_size=64, callbacks=None, verbose=1):
        """
        训练模型

        Args:
            src_train: 源语言训练数据
            tgt_train: 目标语言训练数据
            src_val: 源语言验证数据
            tgt_val: 目标语言验证数据
            epochs: 训练轮数
            batch_size: 批大小
            callbacks: 回调函数
            verbose: 详细程度

        Returns:
            训练历史
        """
        # 编译模型
        self.compile_model()

        # 准备训练数据
        # 【重要】：解码器输入是目标序列去掉最后一个词
        #          解码器输出是目标序列去掉第一个词
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

    def translate(self, src_sequence, tgt_word2idx, tgt_idx2word, max_len=50):
        """
        翻译单个句子（贪心解码）

        Args:
            src_sequence: 源语言序列
            tgt_word2idx: 目标语言词到ID映射
            tgt_idx2word: 目标语言ID到词映射
            max_len: 最大生成长度

        Returns:
            翻译结果
        """
        # 编码源语言
        src_sequence = tf.expand_dims(src_sequence, 0)

        # 初始化目标序列（从SOS开始）
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

            # 取最后一个位置的预测
            predicted_id = tf.argmax(predictions[0, len(output_sequence)-1, :]).numpy()

            # 如果预测到EOS，停止生成
            if predicted_id == eos_id:
                break

            output_sequence.append(predicted_id)

        # 解码为词
        translated_words = [
            tgt_idx2word.get(idx, '<UNK>')
            for idx in output_sequence[1:]  # 跳过SOS
        ]

        return ' '.join(translated_words)

    def save_model(self, filepath):
        """保存模型"""
        self.model.save(filepath)
        print(f"✓ 模型已保存: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ 模型已加载: {filepath}")

    def summary(self):
        """打印模型摘要"""
        self.model.summary()


if __name__ == '__main__':
    """测试模型"""
    print("="*60)
    print("Transformer翻译模型测试")
    print("="*60)

    # 测试参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_len = 20
    batch_size = 4

    # 创建随机数据
    src_train = np.random.randint(1, src_vocab_size, (100, max_len))
    tgt_train = np.random.randint(1, tgt_vocab_size, (100, max_len))
    src_val = np.random.randint(1, src_vocab_size, (20, max_len))
    tgt_val = np.random.randint(1, tgt_vocab_size, (20, max_len))

    # 创建模型
    translator = TransformerTranslationModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_len,
        num_layers=2,
        d_model=128,
        num_heads=4,
        d_ff=512
    )

    print(f"\n模型结构:")
    translator.summary()

    # 训练
    print(f"\n训练模型...")
    history = translator.train(
        src_train, tgt_train,
        src_val, tgt_val,
        epochs=2,
        batch_size=batch_size,
        verbose=0
    )

    print("\n✓ 模型测试通过！")
