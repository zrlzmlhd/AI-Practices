"""
Transformer编码器实现

本模块实现完整的Transformer编码器，包括：
1. Feed Forward Network（前馈神经网络）
2. Encoder Layer（编码器层）
3. Transformer Encoder（完整编码器）

每个组件都有详细的注释说明原理和实现细节。
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from attention import MultiHeadAttention, PositionalEncoding


class FeedForwardNetwork(layers.Layer):
    """
    前馈神经网络（Feed Forward Network）

    【是什么】：两层全连接网络
    【做什么】：对每个位置独立进行非线性变换
    【为什么】：
        - 增加模型的非线性表达能力
        - 每个位置独立处理（Position-wise）
        - 类似于1x1卷积的作用

    结构：
        Linear(d_model -> d_ff) -> ReLU -> Dropout -> Linear(d_ff -> d_model)

    参数说明：
        - d_model: 模型维度（如512）
        - d_ff: 中间层维度（通常是d_model的4倍，如2048）
    """

    def __init__(self, d_model, d_ff, dropout_rate=0.1, **kwargs):
        """
        初始化前馈网络

        Args:
            d_model: 模型维度
            d_ff: 中间层维度
            dropout_rate: Dropout比率
        """
        super(FeedForwardNetwork, self).__init__(**kwargs)

        # ============================================
        # 第一层：扩展维度
        # ============================================
        # 【是什么】：全连接层，d_model -> d_ff
        # 【做什么】：将维度扩展到更高维空间
        # 【为什么】：
        #   - 更高维度提供更强的表达能力
        #   - 类似于"瓶颈"结构的反向
        #   - 通常d_ff = 4 * d_model
        self.dense1 = layers.Dense(d_ff, activation='relu', name='ffn_layer1')

        # ============================================
        # Dropout层
        # ============================================
        # 【是什么】：随机丢弃神经元
        # 【做什么】：防止过拟合
        # 【为什么】：
        #   - 增加模型泛化能力
        #   - 类似于集成学习的效果
        self.dropout = layers.Dropout(dropout_rate, name='ffn_dropout')

        # ============================================
        # 第二层：恢复维度
        # ============================================
        # 【是什么】：全连接层，d_ff -> d_model
        # 【做什么】：将维度恢复到原始大小
        # 【为什么】：
        #   - 保持输入输出维度一致
        #   - 便于残差连接
        self.dense2 = layers.Dense(d_model, name='ffn_layer2')

    def call(self, x, training=False):
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch, seq_len, d_model)
            training: 是否训练模式

        Returns:
            输出张量，形状 (batch, seq_len, d_model)
        """
        # 第一层：扩展 + ReLU
        x = self.dense1(x)  # (batch, seq_len, d_ff)

        # Dropout
        x = self.dropout(x, training=training)

        # 第二层：恢复维度
        x = self.dense2(x)  # (batch, seq_len, d_model)

        return x


class EncoderLayer(layers.Layer):
    """
    Transformer编码器层（Encoder Layer）

    【是什么】：Transformer编码器的基本单元
    【做什么】：包含多头自注意力和前馈网络
    【为什么】：
        - 自注意力：捕获序列内部的依赖关系
        - 前馈网络：增加非线性表达能力
        - 残差连接：帮助梯度传播
        - Layer Normalization：稳定训练

    结构：
        Input
          ↓
        Multi-Head Attention
          ↓
        Add & Norm (残差连接 + 层归一化)
          ↓
        Feed Forward Network
          ↓
        Add & Norm
          ↓
        Output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        """
        初始化编码器层

        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络中间层维度
            dropout_rate: Dropout比率
        """
        super(EncoderLayer, self).__init__(**kwargs)

        # ============================================
        # 子层1: 多头自注意力
        # ============================================
        # 【是什么】：Multi-Head Self-Attention
        # 【做什么】：让每个位置关注序列中的所有位置
        # 【为什么】：
        #   - 捕获长距离依赖
        #   - 并行处理整个序列
        #   - 多头学习不同的关注模式
        self.mha = MultiHeadAttention(d_model, num_heads, name='multi_head_attention')

        # ============================================
        # 子层2: 前馈网络
        # ============================================
        # 【是什么】：Position-wise Feed Forward Network
        # 【做什么】：对每个位置独立进行非线性变换
        # 【为什么】：
        #   - 增加模型的非线性能力
        #   - 每个位置独立处理
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate, name='feed_forward')

        # ============================================
        # Layer Normalization
        # ============================================
        # 【是什么】：层归一化
        # 【做什么】：归一化每个样本的特征
        # 【为什么】：
        #   - 稳定训练过程
        #   - 加速收敛
        #   - 减少对初始化的依赖
        #
        # 【Layer Norm vs Batch Norm】：
        #   - Batch Norm: 对batch维度归一化（适合CNN）
        #   - Layer Norm: 对特征维度归一化（适合RNN/Transformer）
        #   - Transformer用Layer Norm因为序列长度可变
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name='layernorm1')
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name='layernorm2')

        # ============================================
        # Dropout
        # ============================================
        self.dropout1 = layers.Dropout(dropout_rate, name='dropout1')
        self.dropout2 = layers.Dropout(dropout_rate, name='dropout2')

    def call(self, x, mask=None, training=False):
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch, seq_len, d_model)
            mask: 注意力掩码
            training: 是否训练模式

        Returns:
            输出张量，形状 (batch, seq_len, d_model)
        """
        # ============================================
        # 子层1: 多头自注意力 + 残差连接 + Layer Norm
        # ============================================
        # 【残差连接】：x + SubLayer(x)
        # 【为什么】：
        #   - 帮助梯度传播（解决梯度消失）
        #   - 允许网络学习恒等映射
        #   - 使深层网络更容易训练

        # 多头自注意力
        attn_output, _ = self.mha(x, x, x, mask)  # (batch, seq_len, d_model)

        # Dropout
        attn_output = self.dropout1(attn_output, training=training)

        # 残差连接 + Layer Norm
        # 【顺序】：Add -> Norm (Post-LN)
        # 也可以用 Norm -> Add (Pre-LN)
        out1 = self.layernorm1(x + attn_output)  # (batch, seq_len, d_model)

        # ============================================
        # 子层2: 前馈网络 + 残差连接 + Layer Norm
        # ============================================

        # 前馈网络
        ffn_output = self.ffn(out1, training=training)  # (batch, seq_len, d_model)

        # Dropout
        ffn_output = self.dropout2(ffn_output, training=training)

        # 残差连接 + Layer Norm
        out2 = self.layernorm2(out1 + ffn_output)  # (batch, seq_len, d_model)

        return out2


class TransformerEncoder(layers.Layer):
    """
    Transformer编码器（Transformer Encoder）

    【是什么】：多层编码器层的堆叠
    【做什么】：将输入序列编码为上下文表示
    【为什么】：
        - 多层堆叠：逐层提取更抽象的特征
        - 第1层：学习局部模式
        - 第2-3层：学习中等距离依赖
        - 第4-6层：学习长距离依赖和抽象语义

    结构：
        Input Embedding
          ↓
        Positional Encoding
          ↓
        Encoder Layer 1
          ↓
        Encoder Layer 2
          ↓
        ...
          ↓
        Encoder Layer N
          ↓
        Output
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 vocab_size, max_len, dropout_rate=0.1, **kwargs):
        """
        初始化Transformer编码器

        Args:
            num_layers: 编码器层数
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络中间层维度
            vocab_size: 词汇表大小
            max_len: 最大序列长度
            dropout_rate: Dropout比率
        """
        super(TransformerEncoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        # ============================================
        # 词嵌入层
        # ============================================
        # 【是什么】：将词ID转换为稠密向量
        # 【做什么】：查表操作，每个词对应一个向量
        # 【为什么】：
        #   - 将离散的词转换为连续表示
        #   - 相似的词有相似的向量
        #   - 可以学习词的语义信息
        self.embedding = layers.Embedding(
            vocab_size, d_model,
            name='token_embedding'
        )

        # ============================================
        # 位置编码
        # ============================================
        # 【是什么】：添加位置信息
        # 【做什么】：让模型知道词的顺序
        # 【为什么】：
        #   - Self-Attention是无序的
        #   - 需要显式添加位置信息
        self.pos_encoding = PositionalEncoding(max_len, d_model, name='positional_encoding')

        # ============================================
        # 编码器层堆叠
        # ============================================
        # 【是什么】：多个编码器层
        # 【做什么】：逐层提取特征
        # 【为什么】：
        #   - 浅层：学习局部模式
        #   - 深层：学习全局语义
        #   - 类似于CNN的层次特征提取
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate,
                        name=f'encoder_layer_{i}')
            for i in range(num_layers)
        ]

        # Dropout
        self.dropout = layers.Dropout(dropout_rate, name='encoder_dropout')

    def call(self, x, mask=None, training=False):
        """
        前向传播

        Args:
            x: 输入序列（词ID），形状 (batch, seq_len)
            mask: 注意力掩码
            training: 是否训练模式

        Returns:
            编码后的表示，形状 (batch, seq_len, d_model)
        """
        seq_len = tf.shape(x)[1]

        # ============================================
        # 步骤1: 词嵌入
        # ============================================
        # 【做什么】：将词ID转换为向量
        # 例如：[1, 234, 56] -> [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
        x = self.embedding(x)  # (batch, seq_len, d_model)

        # ============================================
        # 步骤2: 缩放
        # ============================================
        # 【是什么】：将嵌入向量乘以sqrt(d_model)
        # 【为什么】：
        #   - 嵌入向量的值通常较小（如[-1, 1]）
        #   - 位置编码的值也在[-1, 1]
        #   - 缩放后，嵌入向量的贡献更大
        #   - 防止位置编码淹没词嵌入信息
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # ============================================
        # 步骤3: 添加位置编码
        # ============================================
        # 【做什么】：将位置信息加到词嵌入上
        x = self.pos_encoding(x)

        # Dropout
        x = self.dropout(x, training=training)

        # ============================================
        # 步骤4: 通过所有编码器层
        # ============================================
        # 【做什么】：逐层提取特征
        # 【效果】：
        #   - 第1层：学习词级别的模式
        #   - 第2-3层：学习短语级别的模式
        #   - 第4-6层：学习句子级别的语义
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask, training)

        return x  # (batch, seq_len, d_model)


if __name__ == '__main__':
    """
    测试Transformer编码器
    """
    print("="*60)
    print("Transformer编码器测试")
    print("="*60)

    # 测试参数
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    max_len = 100

    # 模型参数
    num_layers = 2
    d_model = 128
    num_heads = 4
    d_ff = 512
    dropout_rate = 0.1

    # 创建随机输入（词ID）
    x = tf.random.uniform((batch_size, seq_len), maxval=vocab_size, dtype=tf.int32)

    # 测试前馈网络
    print("\n测试前馈网络...")
    ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)
    x_test = tf.random.normal((batch_size, seq_len, d_model))
    ffn_output = ffn(x_test)
    print(f"输入形状: {x_test.shape}")
    print(f"输出形状: {ffn_output.shape}")

    # 测试编码器层
    print("\n测试编码器层...")
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
    layer_output = encoder_layer(x_test)
    print(f"输入形状: {x_test.shape}")
    print(f"输出形状: {layer_output.shape}")

    # 测试完整编码器
    print("\n测试完整编码器...")
    encoder = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        max_len=max_len,
        dropout_rate=dropout_rate
    )
    encoder_output = encoder(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {encoder_output.shape}")

    print("\n✓ 所有测试通过！")
