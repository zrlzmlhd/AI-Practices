"""
Transformer注意力机制实现

本模块包含Transformer的核心组件：
1. Scaled Dot-Product Attention（缩放点积注意力）
2. Multi-Head Attention（多头注意力）
3. Positional Encoding（位置编码）

每个组件都有详细的注释说明原理和实现细节。
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ScaledDotProductAttention(layers.Layer):
    """
    缩放点积注意力（Scaled Dot-Product Attention）

    这是Transformer的核心机制，用于计算序列中每个位置对其他位置的注意力权重。

    公式：Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

    参数说明：
    - Q (Query): 查询向量，表示"我要找什么"
    - K (Key): 键向量，表示"我是什么"
    - V (Value): 值向量，表示"我的内容是什么"
    - d_k: Key的维度，用于缩放
    """

    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def call(self, query, key, value, mask=None):
        """
        计算缩放点积注意力

        Args:
            query: Query矩阵，形状 (batch, seq_len_q, d_k)
            key: Key矩阵，形状 (batch, seq_len_k, d_k)
            value: Value矩阵，形状 (batch, seq_len_v, d_v)
            mask: 掩码，用于遮蔽某些位置（如padding）

        Returns:
            output: 注意力输出，形状 (batch, seq_len_q, d_v)
            attention_weights: 注意力权重，形状 (batch, seq_len_q, seq_len_k)
        """

        # ============================================
        # 步骤1: 计算注意力分数 (Q · K^T)
        # ============================================
        # 【是什么】：Query和Key的点积
        # 【做什么】：计算Query和每个Key的相似度
        # 【为什么】：相似度高的Key应该获得更多关注
        #
        # 例子：句子 "我 爱 你"
        # Query="爱" 与所有Key的相似度：
        #   "我": 0.3 (中等相关)
        #   "爱": 1.0 (自己最相关)
        #   "你": 0.8 (高度相关，爱的对象)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        # 形状: (batch, seq_len_q, seq_len_k)

        # ============================================
        # 步骤2: 缩放 (除以 sqrt(d_k))
        # ============================================
        # 【是什么】：将分数除以sqrt(d_k)
        # 【做什么】：缩放点积值到合理范围
        # 【为什么】：
        #   - 问题：d_k很大时，点积值会很大
        #   - 后果：softmax后梯度很小（梯度消失）
        #   - 解决：除以sqrt(d_k)进行缩放
        #   - 效果：保持梯度在合理范围
        #
        # 数学原理：
        #   假设Q和K的元素是独立的，均值0，方差1
        #   则Q·K的方差是d_k
        #   除以sqrt(d_k)后，方差变为1
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

        # ============================================
        # 步骤3: 应用掩码（可选）
        # ============================================
        # 【是什么】：将某些位置的分数设为-inf
        # 【做什么】：遮蔽padding位置或未来位置
        # 【为什么】：
        #   - Padding位置：不应该被关注（没有实际内容）
        #   - 未来位置：解码器不能看到未来（防止作弊）
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # ============================================
        # 步骤4: Softmax归一化
        # ============================================
        # 【是什么】：将分数转换为概率分布
        # 【做什么】：归一化，使所有权重和为1
        # 【为什么】：
        #   - 概率解释：每个位置的重要性
        #   - 和为1：便于加权求和
        #
        # 例子：
        #   分数: [2.0, 5.0, 3.0]
        #   Softmax: [0.09, 0.67, 0.24]  # 和为1
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # 形状: (batch, seq_len_q, seq_len_k)

        # ============================================
        # 步骤5: 加权求和 (Attention_Weights · V)
        # ============================================
        # 【是什么】：用注意力权重对Value加权求和
        # 【做什么】：根据重要性组合信息
        # 【为什么】：
        #   - 重要的位置权重高，贡献大
        #   - 不重要的位置权重低，贡献小
        #   - 最终输出是所有位置的加权组合
        #
        # 例子：
        #   权重: [0.1, 0.6, 0.3]
        #   Value: [[1,2], [3,4], [5,6]]
        #   输出: 0.1*[1,2] + 0.6*[3,4] + 0.3*[5,6]
        #       = [3.4, 4.4]  # 重点关注第2个位置
        output = tf.matmul(attention_weights, value)
        # 形状: (batch, seq_len_q, d_v)

        return output, attention_weights


class MultiHeadAttention(layers.Layer):
    """
    多头注意力（Multi-Head Attention）

    核心思想：使用多个注意力头，每个头学习不同的关注模式

    为什么需要多头？
    - 单头：只能学习一种关系（如语义相似）
    - 多头：学习多种关系（语义、语法、位置等）

    例如8个头可能学习：
    - Head 1: 语义相似（"好"关注"棒"）
    - Head 2: 语法关系（动词关注主语）
    - Head 3: 位置关系（关注相邻词）
    - Head 4: 情感词（"好"关注"很"）
    - ... 等8个不同的关注模式
    """

    def __init__(self, d_model, num_heads, **kwargs):
        """
        初始化多头注意力

        Args:
            d_model: 模型维度（如512）
            num_heads: 注意力头数（如8）

        要求：d_model必须能被num_heads整除
        """
        super(MultiHeadAttention, self).__init__(**kwargs)

        # ============================================
        # 参数验证和初始化
        # ============================================

        # 【验证】：d_model必须能被num_heads整除
        # 【为什么】：需要将d_model平均分配给每个头
        # 例如：d_model=512, num_heads=8
        #      每个头的维度 d_k = 512/8 = 64
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.num_heads = num_heads
        self.d_model = d_model

        # 每个头的维度
        # 【是什么】：d_k = d_model / num_heads
        # 【为什么】：将总维度平均分配给每个头
        self.depth = d_model // num_heads

        # ============================================
        # 权重矩阵
        # ============================================

        # 【是什么】：线性变换矩阵
        # 【做什么】：将输入投影到Q、K、V空间
        # 【为什么】：
        #   - 学习如何生成Query、Key、Value
        #   - 每个头有独立的投影空间
        #   - 增加模型的表达能力

        # WQ: Query权重矩阵
        self.wq = layers.Dense(d_model, name='query_projection')

        # WK: Key权重矩阵
        self.wk = layers.Dense(d_model, name='key_projection')

        # WV: Value权重矩阵
        self.wv = layers.Dense(d_model, name='value_projection')

        # WO: 输出权重矩阵
        # 【作用】：将多个头的输出合并
        self.dense = layers.Dense(d_model, name='output_projection')

        # 注意力层
        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        """
        将输入分割成多个头

        Args:
            x: 输入张量，形状 (batch, seq_len, d_model)
            batch_size: 批大小

        Returns:
            分割后的张量，形状 (batch, num_heads, seq_len, depth)

        例子：
            输入: (32, 100, 512)  # 32个样本，100个词，512维
            输出: (32, 8, 100, 64)  # 32个样本，8个头，100个词，每头64维
        """
        # 重塑为 (batch, seq_len, num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        # 转置为 (batch, num_heads, seq_len, depth)
        # 【为什么转置】：方便并行计算每个头的注意力
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask=None):
        """
        计算多头注意力

        Args:
            query: Query输入
            key: Key输入
            value: Value输入
            mask: 掩码

        Returns:
            output: 多头注意力输出
            attention_weights: 注意力权重
        """
        batch_size = tf.shape(query)[0]

        # ============================================
        # 步骤1: 线性投影 (生成Q、K、V)
        # ============================================
        # 【做什么】：通过权重矩阵生成Q、K、V
        # 【为什么】：
        #   - 学习如何从输入生成查询、键、值
        #   - 每个头有不同的投影，学习不同的模式
        query = self.wq(query)  # (batch, seq_len, d_model)
        key = self.wk(key)      # (batch, seq_len, d_model)
        value = self.wv(value)  # (batch, seq_len, d_model)

        # ============================================
        # 步骤2: 分割成多个头
        # ============================================
        # 【做什么】：将d_model维度分成num_heads个头
        # 【为什么】：让每个头独立学习不同的关注模式
        #
        # 例子：d_model=512, num_heads=8
        #   输入: (batch, seq_len, 512)
        #   分割: (batch, 8, seq_len, 64)
        #   每个头处理64维
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # ============================================
        # 步骤3: 计算缩放点积注意力
        # ============================================
        # 【做什么】：每个头独立计算注意力
        # 【为什么】：并行计算，每个头学习不同的模式
        scaled_attention, attention_weights = self.attention(
            query, key, value, mask
        )
        # scaled_attention: (batch, num_heads, seq_len, depth)
        # attention_weights: (batch, num_heads, seq_len_q, seq_len_k)

        # ============================================
        # 步骤4: 转置回原来的形状
        # ============================================
        # 【做什么】：从 (batch, num_heads, seq_len, depth)
        #           转为 (batch, seq_len, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # ============================================
        # 步骤5: 拼接所有头
        # ============================================
        # 【做什么】：将多个头的输出拼接成一个向量
        # 【为什么】：
        #   - 合并所有头学到的信息
        #   - 恢复到原始维度d_model
        #
        # 例子：8个头，每个64维
        #   拼接: [head1(64), head2(64), ..., head8(64)]
        #   结果: 512维向量
        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model)
        )
        # 形状: (batch, seq_len, d_model)

        # ============================================
        # 步骤6: 最终线性变换
        # ============================================
        # 【做什么】：通过WO矩阵进行最终变换
        # 【为什么】：
        #   - 学习如何组合多个头的信息
        #   - 增加模型的表达能力
        #   - 类似于全连接层的作用
        output = self.dense(concat_attention)
        # 形状: (batch, seq_len, d_model)

        return output, attention_weights


class PositionalEncoding(layers.Layer):
    """
    位置编码（Positional Encoding）

    为什么需要位置编码？
    - 问题：Self-Attention是无序的
      "我爱你" 和 "你爱我" 的注意力计算结果相同
    - 解决：添加位置信息
      让模型知道每个词在句子中的位置

    位置编码公式：
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中：
    - pos: 词的位置 (0, 1, 2, ...)
    - i: 维度索引 (0, 1, 2, ..., d_model/2)
    """

    def __init__(self, max_len, d_model, **kwargs):
        """
        初始化位置编码

        Args:
            max_len: 最大序列长度
            d_model: 模型维度
        """
        super(PositionalEncoding, self).__init__(**kwargs)

        self.max_len = max_len
        self.d_model = d_model

        # 预计算位置编码
        self.pos_encoding = self.positional_encoding(max_len, d_model)

    def get_angles(self, pos, i, d_model):
        """
        计算位置编码的角度

        Args:
            pos: 位置索引
            i: 维度索引
            d_model: 模型维度

        Returns:
            角度值
        """
        # ============================================
        # 角度计算公式
        # ============================================
        # 【公式】：pos / 10000^(2i/d_model)
        # 【为什么用这个公式】：
        #   - 不同位置有不同的编码
        #   - 不同维度有不同的频率
        #   - 低维度：高频率（快速变化）
        #   - 高维度：低频率（缓慢变化）
        #
        # 效果：
        #   - 相邻位置的编码相似但不同
        #   - 可以通过三角函数计算相对位置
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, max_len, d_model):
        """
        生成位置编码矩阵

        Args:
            max_len: 最大序列长度
            d_model: 模型维度

        Returns:
            位置编码矩阵，形状 (1, max_len, d_model)
        """
        # 生成位置和维度的索引
        angle_rads = self.get_angles(
            np.arange(max_len)[:, np.newaxis],  # 位置: (max_len, 1)
            np.arange(d_model)[np.newaxis, :],  # 维度: (1, d_model)
            d_model
        )

        # ============================================
        # 应用sin和cos函数
        # ============================================
        # 【为什么用sin/cos】：
        #   1. 值域在[-1, 1]，不会太大
        #   2. 周期性函数，相似位置有相似编码
        #   3. 可以通过三角恒等式计算相对位置
        #      PE(pos+k) 可以表示为 PE(pos) 的线性组合
        #   4. 可以处理训练时未见过的长度

        # 偶数维度用sin
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 奇数维度用cos
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        # 形状: (1, max_len, d_model)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        """
        添加位置编码

        Args:
            x: 输入张量，形状 (batch, seq_len, d_model)

        Returns:
            添加位置编码后的张量
        """
        seq_len = tf.shape(x)[1]

        # ============================================
        # 添加位置编码
        # ============================================
        # 【做什么】：将位置编码加到词嵌入上
        # 【为什么是相加而不是拼接】：
        #   - 相加：保持维度不变，位置信息融入词向量
        #   - 拼接：维度翻倍，计算量大
        #   - 相加效果更好（实验证明）
        #
        # 例子：
        #   词嵌入: [0.2, 0.5, -0.3, ...]
        #   位置编码: [0.0, 0.1, 0.05, ...]
        #   结果: [0.2, 0.6, -0.25, ...]
        x = x + self.pos_encoding[:, :seq_len, :]

        return x


def create_padding_mask(seq):
    """
    创建padding掩码

    Args:
        seq: 输入序列，形状 (batch, seq_len)

    Returns:
        掩码，形状 (batch, 1, 1, seq_len)

    作用：遮蔽padding位置（值为0的位置）
    """
    # 找出padding位置（值为0）
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 添加额外维度以便广播
    # (batch, seq_len) -> (batch, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    创建前瞻掩码（用于解码器）

    Args:
        size: 序列长度

    Returns:
        掩码，形状 (size, size)

    作用：防止解码器看到未来的词
    """
    # 创建下三角矩阵
    # 例如 size=4:
    # [[0, 1, 1, 1],
    #  [0, 0, 1, 1],
    #  [0, 0, 0, 1],
    #  [0, 0, 0, 0]]
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


if __name__ == '__main__':
    """
    测试注意力机制
    """
    print("="*60)
    print("Transformer注意力机制测试")
    print("="*60)

    # 测试参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    # 创建随机输入
    x = tf.random.normal((batch_size, seq_len, d_model))

    # 测试多头注意力
    print("\n测试多头注意力...")
    mha = MultiHeadAttention(d_model, num_heads)
    output, attention_weights = mha(x, x, x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")

    # 测试位置编码
    print("\n测试位置编码...")
    pos_encoding = PositionalEncoding(max_len=100, d_model=d_model)
    x_with_pos = pos_encoding(x)

    print(f"输入形状: {x.shape}")
    print(f"添加位置编码后: {x_with_pos.shape}")

    print("\n✓ 所有测试通过！")
