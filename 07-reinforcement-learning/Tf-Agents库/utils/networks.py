#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
神经网络架构工具

提供强化学习中常用的神经网络构建函数，支持自定义配置。

================================================================================
核心思想 (Core Idea)
================================================================================
强化学习中的函数逼近器（Q网络、策略网络、价值网络）通常基于
多层感知机 (MLP) 或卷积神经网络 (CNN) 构建。本模块提供灵活的
网络构建工具，支持：
1. 全连接网络（用于低维状态）
2. 卷积网络（用于图像输入）
3. 各种正则化技术（Dropout、BatchNorm、LayerNorm）

================================================================================
数学原理 (Mathematical Theory)
================================================================================
多层感知机 (MLP):
$$f(x) = \sigma_L(W_L \cdot \sigma_{L-1}(W_{L-1} \cdot ... \cdot \sigma_1(W_1 x + b_1) + b_{L-1}) + b_L)$$

其中 $\sigma_i$ 为激活函数，常用选择：
- ReLU: $\sigma(x) = \max(0, x)$
- Tanh: $\sigma(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- Swish: $\sigma(x) = x \cdot \text{sigmoid}(x)$

卷积层:
$$(f * g)[n] = \sum_{m=-\infty}^{\infty} f[m] \cdot g[n-m]$$

================================================================================
"""

from __future__ import annotations

import tensorflow as tf
from typing import Tuple, Optional, List, Union, Callable
from enum import Enum


class ActivationType(Enum):
    """激活函数类型枚举"""
    RELU = "relu"
    TANH = "tanh"
    SWISH = "swish"
    ELU = "elu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"


def get_activation(activation: Union[str, ActivationType]) -> Callable:
    """
    获取激活函数

    Args:
        activation: 激活函数名称或枚举

    Returns:
        TensorFlow 激活函数

    Raises:
        ValueError: 未知的激活函数类型
    """
    if isinstance(activation, ActivationType):
        activation = activation.value

    activation_map = {
        "relu": tf.keras.activations.relu,
        "tanh": tf.keras.activations.tanh,
        "swish": tf.keras.activations.swish,
        "elu": tf.keras.activations.elu,
        "leaky_relu": tf.keras.layers.LeakyReLU(),
        "gelu": tf.keras.activations.gelu,
        "linear": tf.keras.activations.linear,
        "softmax": tf.keras.activations.softmax,
    }

    if activation not in activation_map:
        raise ValueError(f"Unknown activation: {activation}")

    return activation_map[activation]


def create_fc_network(
    input_shape: Tuple[int, ...],
    output_dim: int,
    hidden_layers: Tuple[int, ...] = (256, 256),
    activation: str = "relu",
    output_activation: Optional[str] = None,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = False,
    use_layer_norm: bool = False,
    kernel_initializer: str = "glorot_uniform",
    name: str = "fc_network"
) -> tf.keras.Model:
    """
    创建全连接神经网络

    核心思想 (Core Idea):
        多层感知机是最基础的函数逼近器，适用于低维状态空间。
        通过堆叠全连接层实现非线性映射。

    数学原理 (Mathematical Theory):
        前向传播:
        $$h_0 = x$$
        $$h_{l+1} = \sigma(W_l h_l + b_l), \quad l = 0, ..., L-1$$
        $$y = W_L h_L + b_L$$

        Glorot 初始化（Xavier）:
        $$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

    Args:
        input_shape: 输入形状
        output_dim: 输出维度
        hidden_layers: 隐藏层维度元组
        activation: 隐藏层激活函数
        output_activation: 输出层激活函数（None=线性）
        dropout_rate: Dropout 概率
        use_batch_norm: 是否使用批归一化
        use_layer_norm: 是否使用层归一化
        kernel_initializer: 权重初始化方法
        name: 网络名称

    Returns:
        Keras Sequential 模型

    Example:
        >>> net = create_fc_network(
        ...     input_shape=(4,),
        ...     output_dim=2,
        ...     hidden_layers=(128, 64),
        ...     activation="relu",
        ...     dropout_rate=0.1
        ... )
        >>> output = net(tf.random.normal((32, 4)))
        >>> print(output.shape)  # (32, 2)
    """
    layers = []

    # 输入层
    layers.append(tf.keras.layers.Input(shape=input_shape))

    # 隐藏层
    for units in hidden_layers:
        layers.append(tf.keras.layers.Dense(
            units,
            kernel_initializer=kernel_initializer
        ))

        # 归一化层（在激活之前）
        if use_batch_norm:
            layers.append(tf.keras.layers.BatchNormalization())
        if use_layer_norm:
            layers.append(tf.keras.layers.LayerNormalization())

        # 激活函数
        layers.append(tf.keras.layers.Activation(activation))

        # Dropout
        if dropout_rate > 0:
            layers.append(tf.keras.layers.Dropout(dropout_rate))

    # 输出层
    layers.append(tf.keras.layers.Dense(
        output_dim,
        activation=output_activation,
        kernel_initializer=kernel_initializer
    ))

    model = tf.keras.Sequential(layers, name=name)
    return model


def create_conv_network(
    input_shape: Tuple[int, int, int],
    output_dim: int,
    conv_layers: Tuple[Tuple[int, int, int], ...] = ((32, 8, 4), (64, 4, 2), (64, 3, 1)),
    fc_layers: Tuple[int, ...] = (512,),
    activation: str = "relu",
    output_activation: Optional[str] = None,
    use_batch_norm: bool = False,
    name: str = "conv_network"
) -> tf.keras.Model:
    """
    创建卷积神经网络

    核心思想 (Core Idea):
        卷积网络适用于图像输入（如 Atari 游戏），通过卷积层
        提取空间特征，再通过全连接层输出动作值或策略。

    数学原理 (Mathematical Theory):
        2D 卷积操作:
        $$y[i,j] = \sum_{m,n} x[i+m, j+n] \cdot k[m,n]$$

        输出尺寸计算:
        $$H_{out} = \frac{H_{in} - K + 2P}{S} + 1$$

        其中 K=卷积核大小，P=填充，S=步长。

    Args:
        input_shape: 输入图像形状 (H, W, C)
        output_dim: 输出维度
        conv_layers: 卷积层配置 ((filters, kernel_size, stride), ...)
        fc_layers: 全连接层维度
        activation: 激活函数
        output_activation: 输出层激活函数
        use_batch_norm: 是否使用批归一化
        name: 网络名称

    Returns:
        Keras Model 对象

    Example:
        >>> net = create_conv_network(
        ...     input_shape=(84, 84, 4),
        ...     output_dim=4,
        ...     conv_layers=((32, 8, 4), (64, 4, 2), (64, 3, 1))
        ... )
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # 卷积层
    for filters, kernel_size, stride in conv_layers:
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=stride,
            padding='valid'
        )(x)

        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation(activation)(x)

    # 展平
    x = tf.keras.layers.Flatten()(x)

    # 全连接层
    for units in fc_layers:
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.Activation(activation)(x)

    # 输出层
    outputs = tf.keras.layers.Dense(
        output_dim,
        activation=output_activation
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def create_dueling_network(
    input_shape: Tuple[int, ...],
    num_actions: int,
    hidden_layers: Tuple[int, ...] = (256, 256),
    value_layers: Tuple[int, ...] = (128,),
    advantage_layers: Tuple[int, ...] = (128,),
    activation: str = "relu",
    name: str = "dueling_network"
) -> tf.keras.Model:
    """
    创建 Dueling DQN 网络架构

    核心思想 (Core Idea):
        Dueling DQN 将 Q 值分解为状态价值 V(s) 和优势函数 A(s,a)，
        使网络能够独立学习哪些状态有价值，而不依赖于具体动作。

    数学原理 (Mathematical Theory):
        Q 值分解:
        $$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) +
        \left(A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a'; \theta, \alpha)\right)$$

        减去均值确保可辨识性：
        - $V(s)$ 表示状态本身的价值
        - $A(s,a)$ 表示动作相对于平均的优势

        这种分解在动作不影响状态价值的情况下特别有效。

    Args:
        input_shape: 输入形状
        num_actions: 动作数量
        hidden_layers: 共享隐藏层
        value_layers: 价值流隐藏层
        advantage_layers: 优势流隐藏层
        activation: 激活函数
        name: 网络名称

    Returns:
        Dueling 架构的 Keras Model

    Reference:
        Wang, Z. et al. (2016). Dueling Network Architectures for Deep
        Reinforcement Learning. ICML.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # 共享特征提取层
    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation=activation)(x)

    # 价值流 (Value Stream)
    value = x
    for units in value_layers:
        value = tf.keras.layers.Dense(units, activation=activation)(value)
    value = tf.keras.layers.Dense(1, name='value')(value)

    # 优势流 (Advantage Stream)
    advantage = x
    for units in advantage_layers:
        advantage = tf.keras.layers.Dense(units, activation=activation)(advantage)
    advantage = tf.keras.layers.Dense(num_actions, name='advantage')(advantage)

    # 聚合：Q = V + (A - mean(A))
    # 使用减均值而非减最大值确保可辨识性
    q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

    model = tf.keras.Model(inputs=inputs, outputs=q_values, name=name)
    return model


def create_noisy_layer(
    units: int,
    sigma_init: float = 0.5,
    name: str = "noisy_dense"
) -> tf.keras.layers.Layer:
    """
    创建 NoisyNet 层

    核心思想 (Core Idea):
        NoisyNet 在网络权重中添加可学习的噪声参数，
        实现参数化探索，替代 ε-贪婪策略。

    数学原理 (Mathematical Theory):
        标准全连接层:
        $$y = Wx + b$$

        Noisy 层:
        $$y = (\mu^w + \sigma^w \odot \epsilon^w)x + (\mu^b + \sigma^b \odot \epsilon^b)$$

        其中:
        - $\mu^w, \mu^b$: 可学习的均值参数
        - $\sigma^w, \sigma^b$: 可学习的噪声标准差
        - $\epsilon^w, \epsilon^b$: 采样的噪声

        因式分解噪声（降低参数量）:
        $$\epsilon_{ij} = f(\epsilon_i) \cdot f(\epsilon_j)$$
        $$f(x) = \text{sign}(x) \sqrt{|x|}$$

    Args:
        units: 输出维度
        sigma_init: 初始噪声标准差
        name: 层名称

    Returns:
        NoisyDense 层实例

    Reference:
        Fortunato, M. et al. (2018). Noisy Networks for Exploration. ICLR.
    """
    class NoisyDense(tf.keras.layers.Layer):
        """Noisy Dense 层实现"""

        def __init__(self, units, sigma_init=0.5, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.sigma_init = sigma_init

        def build(self, input_shape):
            self.input_dim = int(input_shape[-1])

            # 可学习的均值参数
            mu_range = 1 / (self.input_dim ** 0.5)
            self.mu_w = self.add_weight(
                "mu_w",
                shape=(self.input_dim, self.units),
                initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
                trainable=True
            )
            self.mu_b = self.add_weight(
                "mu_b",
                shape=(self.units,),
                initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
                trainable=True
            )

            # 可学习的噪声参数
            self.sigma_w = self.add_weight(
                "sigma_w",
                shape=(self.input_dim, self.units),
                initializer=tf.keras.initializers.Constant(
                    self.sigma_init / (self.input_dim ** 0.5)
                ),
                trainable=True
            )
            self.sigma_b = self.add_weight(
                "sigma_b",
                shape=(self.units,),
                initializer=tf.keras.initializers.Constant(
                    self.sigma_init / (self.units ** 0.5)
                ),
                trainable=True
            )

        def call(self, inputs, training=None):
            if training:
                # 因式分解噪声
                eps_i = self._f(tf.random.normal((self.input_dim, 1)))
                eps_j = self._f(tf.random.normal((1, self.units)))
                eps_w = eps_i * eps_j
                eps_b = tf.squeeze(eps_j)

                # 添加噪声
                w = self.mu_w + self.sigma_w * eps_w
                b = self.mu_b + self.sigma_b * eps_b
            else:
                w = self.mu_w
                b = self.mu_b

            return tf.matmul(inputs, w) + b

        def _f(self, x):
            """因式分解噪声函数"""
            return tf.sign(x) * tf.sqrt(tf.abs(x))

    return NoisyDense(units, sigma_init, name=name)


def create_actor_critic_networks(
    observation_shape: Tuple[int, ...],
    num_actions: int,
    shared_layers: Tuple[int, ...] = (256,),
    actor_layers: Tuple[int, ...] = (128,),
    critic_layers: Tuple[int, ...] = (128,),
    activation: str = "relu",
    continuous_action: bool = False
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    创建 Actor-Critic 网络对

    核心思想 (Core Idea):
        Actor-Critic 架构包含两个网络：
        - Actor (策略网络): 输出动作或动作分布
        - Critic (价值网络): 估计状态/动作价值

        共享底层特征可以提高样本效率。

    Args:
        observation_shape: 观测形状
        num_actions: 动作数量/维度
        shared_layers: 共享特征层
        actor_layers: Actor 独有层
        critic_layers: Critic 独有层
        activation: 激活函数
        continuous_action: 是否连续动作空间

    Returns:
        (actor_network, critic_network) 元组
    """
    # 共享输入
    inputs = tf.keras.layers.Input(shape=observation_shape)
    x = inputs

    # 共享特征层
    for units in shared_layers:
        x = tf.keras.layers.Dense(units, activation=activation)(x)

    # Actor 分支
    actor = x
    for units in actor_layers:
        actor = tf.keras.layers.Dense(units, activation=activation)(actor)

    if continuous_action:
        # 连续动作：输出均值和对数标准差
        mean = tf.keras.layers.Dense(num_actions, activation='tanh')(actor)
        log_std = tf.keras.layers.Dense(num_actions)(actor)
        actor_output = tf.keras.layers.Concatenate()([mean, log_std])
    else:
        # 离散动作：输出动作 logits
        actor_output = tf.keras.layers.Dense(num_actions, activation='softmax')(actor)

    actor_model = tf.keras.Model(inputs=inputs, outputs=actor_output, name='actor')

    # Critic 分支
    critic = x
    for units in critic_layers:
        critic = tf.keras.layers.Dense(units, activation=activation)(critic)
    critic_output = tf.keras.layers.Dense(1)(critic)

    critic_model = tf.keras.Model(inputs=inputs, outputs=critic_output, name='critic')

    return actor_model, critic_model


if __name__ == "__main__":
    print("=" * 60)
    print("神经网络架构测试")
    print("=" * 60)

    # 测试全连接网络
    print("\n1. 全连接网络")
    fc_net = create_fc_network(
        input_shape=(4,),
        output_dim=2,
        hidden_layers=(128, 64),
        activation="relu",
        dropout_rate=0.1
    )
    fc_net.summary()

    # 测试卷积网络
    print("\n2. 卷积网络")
    conv_net = create_conv_network(
        input_shape=(84, 84, 4),
        output_dim=4,
        conv_layers=((32, 8, 4), (64, 4, 2), (64, 3, 1)),
        fc_layers=(512,)
    )
    conv_net.summary()

    # 测试 Dueling 网络
    print("\n3. Dueling 网络")
    dueling_net = create_dueling_network(
        input_shape=(4,),
        num_actions=2,
        hidden_layers=(128, 64),
        value_layers=(32,),
        advantage_layers=(32,)
    )
    dueling_net.summary()

    # 测试 Actor-Critic 网络
    print("\n4. Actor-Critic 网络")
    actor, critic = create_actor_critic_networks(
        observation_shape=(4,),
        num_actions=2,
        shared_layers=(128,),
        actor_layers=(64,),
        critic_layers=(64,)
    )
    print("Actor:")
    actor.summary()
    print("\nCritic:")
    critic.summary()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
