"""
LSTM情感分析模型定义

本模块包含详细的LSTM模型实现，每一层都有详细的注释说明：
1. 这一层是什么
2. 这一层做什么
3. 为什么使用这一层
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


class LSTMSentimentAnalyzer:
    """
    LSTM情感分析器

    支持三种模型架构：
    1. simple_lstm: 单层LSTM（入门）
    2. bilstm: 双向LSTM（中级）
    3. stacked_lstm: 堆叠LSTM（高级）
    """

    def __init__(self, model_type='simple_lstm', max_words=10000, max_len=200, random_state=42):
        """
        初始化情感分析器

        Args:
            model_type: 模型类型 ['simple_lstm', 'bilstm', 'stacked_lstm']
            max_words: 词汇表大小（最常用的词数量）
            max_len: 序列最大长度（评论的最大词数）
            random_state: 随机种子（保证结果可复现）
        """
        self.model_type = model_type
        self.max_words = max_words
        self.max_len = max_len
        self.random_state = random_state
        self.model = None
        self.history = None

        # 设置随机种子，保证结果可复现
        set_seed(random_state)

    def create_simple_lstm(self, embedding_dim=128):
        """
        创建简单的单层LSTM模型（入门级）

        架构：Embedding → LSTM → Dense → Dropout → Output

        适用场景：
        - 学习LSTM基础
        - 数据量较小
        - 快速原型验证

        Args:
            embedding_dim: 词嵌入维度

        Returns:
            Keras模型
        """
        model = models.Sequential(name='simple_lstm')

        # ============================================
        # 第1层：Embedding（词嵌入层）
        # ============================================
        # 【是什么】：将整数索引转换为稠密向量
        # 【做什么】：
        #   输入：[5, 234, 12, ...]  (词的索引)
        #   输出：[[0.2, 0.5, ...], [0.1, -0.3, ...], ...]  (每个词的向量表示)
        # 【为什么】：
        #   - 词需要用向量表示才能输入神经网络
        #   - 稠密向量能捕获词之间的语义关系
        #   - 例如："好"和"棒"的向量会比较接近
        # 【参数选择】：
        #   - max_words=10000: 词汇表大小
        #     * 太小(1000): 很多词会变成"未知词"
        #     * 太大(50000): 稀有词太多，训练效果差
        #     * 10000: 经验最佳值，覆盖常用词
        #   - embedding_dim=128: 每个词的向量维度
        #     * 太小(32): 表达能力不足
        #     * 太大(512): 参数过多，容易过拟合
        #     * 128: 平衡表达能力和效率
        #   - input_length=max_len: 固定序列长度
        #     * 短评论会填充0
        #     * 长评论会截断
        model.add(layers.Embedding(
            input_dim=self.max_words,
            output_dim=embedding_dim,
            input_length=self.max_len,
            name='embedding_word_vectors'
        ))

        # ============================================
        # 第2层：LSTM（长短期记忆层）- 核心层
        # ============================================
        # 【是什么】：能够记住长期依赖关系的循环神经网络
        # 【做什么】：
        #   - 顺序处理每个词的向量
        #   - 通过三个门控制信息流动：
        #     1. 遗忘门：决定丢弃哪些旧信息
        #     2. 输入门：决定存储哪些新信息
        #     3. 输出门：决定输出哪些信息
        #   - 维护一个记忆单元，保存重要的上下文信息
        # 【为什么】：
        #   - 情感分析需要理解上下文关系
        #   - 例如："不好"需要理解"不"对"好"的否定作用
        #   - LSTM能记住前面的词，理解整个句子的意思
        # 【参数选择】：
        #   - units=128: LSTM单元数（记忆容量）
        #     * 太小(32): 记忆容量不足，无法捕获复杂模式
        #     * 太大(512): 计算量大，容易过拟合
        #     * 128: 平衡记忆能力和效率
        #     * 与embedding_dim相同：信息流动更顺畅
        #   - dropout=0.2: 输入dropout率
        #     * 作用：随机丢弃20%的输入连接
        #     * 为什么：防止过度依赖某些特定词
        #     * 20%：不会太影响训练，又能防止过拟合
        #   - recurrent_dropout=0.2: 循环dropout率
        #     * 作用：在时间步之间应用dropout
        #     * 为什么：防止在序列处理中过拟合
        #     * 注意：不能太大，否则破坏时序信息
        #   - return_sequences=False: 只返回最后一个输出
        #     * False: 用于分类任务（返回整个句子的表示）
        #     * True: 用于序列标注（返回每个词的表示）
        model.add(layers.LSTM(
            units=128,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=False,
            name='lstm_sequence_processing'
        ))

        # ============================================
        # 第3层：Dense（全连接层）- 特征组合
        # ============================================
        # 【是什么】：全连接的神经网络层
        # 【做什么】：
        #   - 将LSTM的128维输出组合成64维特征
        #   - 学习哪些特征组合对情感分析重要
        # 【为什么】：
        #   - 降维：去除冗余信息，减少过拟合
        #   - 特征组合：学习更高层次的抽象特征
        #   - 为最终分类做准备
        # 【参数选择】：
        #   - units=64: 神经元数量
        #     * 为什么是64: LSTM输出的一半，适度压缩
        #     * 压缩的好处：去除冗余，减少过拟合
        #   - activation='relu': ReLU激活函数
        #     * 作用：引入非线性，过滤负值
        #     * 公式：f(x) = max(0, x)
        #     * 为什么用ReLU：
        #       1. 计算简单，训练快
        #       2. 缓解梯度消失问题
        #       3. 生物学上更合理（神经元激活模式）
        model.add(layers.Dense(
            units=64,
            activation='relu',
            name='dense_feature_combination'
        ))

        # ============================================
        # 第4层：Dropout（正则化层）
        # ============================================
        # 【是什么】：随机丢弃神经元的正则化技术
        # 【做什么】：
        #   - 训练时：随机关闭50%的神经元
        #   - 测试时：使用所有神经元（输出乘以0.5）
        # 【为什么】：
        #   - 防止过拟合：强制网络不依赖特定神经元
        #   - 集成效果：相当于训练多个子网络
        #   - 提高泛化能力：学习更鲁棒的特征
        # 【参数选择】：
        #   - rate=0.5: 丢弃率50%
        #     * 为什么这么高：全连接层容易过拟合
        #     * 为什么不用在LSTM：LSTM已有dropout机制
        #     * 经验值：全连接层常用0.5，卷积层常用0.25
        model.add(layers.Dropout(
            rate=0.5,
            name='dropout_regularization'
        ))

        # ============================================
        # 第5层：Output（输出层）
        # ============================================
        # 【是什么】：二分类输出层
        # 【做什么】：
        #   - 输出一个0-1之间的概率值
        #   - >0.5: 正面情感
        #   - <0.5: 负面情感
        # 【为什么】：
        #   - 二分类问题只需要一个输出
        #   - sigmoid将任意实数映射到概率
        # 【参数选择】：
        #   - units=1: 单个输出神经元
        #     * 为什么是1：二分类只需一个概率值
        #     * 多分类：units=类别数，用softmax
        #   - activation='sigmoid': Sigmoid激活函数
        #     * 公式：σ(x) = 1 / (1 + e^(-x))
        #     * 作用：将实数映射到(0, 1)
        #     * 为什么用sigmoid：
        #       1. 输出可解释为概率
        #       2. 配合binary_crossentropy损失
        #       3. 二分类的标准选择
        model.add(layers.Dense(
            units=1,
            activation='sigmoid',
            name='output_probability'
        ))

        return model

    def create_bilstm(self, embedding_dim=128):
        """
        创建双向LSTM模型（中级）

        架构：Embedding → BiLSTM → Dense → Dropout → Output

        优势：
        - 同时看前后文，理解更全面
        - 例如："不好看" - 双向能同时看到"不"和"看"

        适用场景：
        - 需要理解完整上下文
        - 数据量充足
        - 对准确率要求较高

        Args:
            embedding_dim: 词嵌入维度

        Returns:
            Keras模型
        """
        model = models.Sequential(name='bilstm')

        # 词嵌入层（同simple_lstm）
        model.add(layers.Embedding(
            input_dim=self.max_words,
            output_dim=embedding_dim,
            input_length=self.max_len,
            name='embedding_word_vectors'
        ))

        # ============================================
        # 双向LSTM层
        # ============================================
        # 【是什么】：同时从前向后和从后向前处理序列的LSTM
        # 【做什么】：
        #   前向LSTM: "这" → "部" → "电影" → "不" → "好"
        #   后向LSTM: "好" → "不" → "电影" → "部" → "这"
        #   最终输出: 拼接前向和后向的结果 (64+64=128维)
        # 【为什么】：
        #   - 单向LSTM的局限：
        #     * 处理"不好"时，只看到了"不"
        #     * 可能误判为负面
        #   - 双向LSTM的优势：
        #     * 同时看到"不"和"好"
        #     * 理解"不好"是一个整体
        #     * 判断更准确
        # 【参数选择】：
        #   - units=64: 每个方向的LSTM单元数
        #     * 为什么是64而不是128：
        #       1. 双向会拼接，总输出是128维
        #       2. 保持与simple_lstm相同的输出维度
        #       3. 减少参数量，防止过拟合
        #   - 其他参数同simple_lstm
        model.add(layers.Bidirectional(
            layers.LSTM(
                units=64,
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=False
            ),
            name='bilstm_bidirectional_processing'
        ))

        # 全连接层、Dropout、输出层（同simple_lstm）
        model.add(layers.Dense(64, activation='relu', name='dense_feature_combination'))
        model.add(layers.Dropout(0.5, name='dropout_regularization'))
        model.add(layers.Dense(1, activation='sigmoid', name='output_probability'))

        return model

    def create_stacked_lstm(self, embedding_dim=128):
        """
        创建堆叠LSTM模型（高级）

        架构：Embedding → LSTM → LSTM → Dense → Dropout → Output

        优势：
        - 多层LSTM能学习更复杂的模式
        - 第一层：学习低级特征（词组）
        - 第二层：学习高级特征（句子结构）

        适用场景：
        - 复杂的文本理解任务
        - 数据量大
        - 需要捕获深层语义

        Args:
            embedding_dim: 词嵌入维度

        Returns:
            Keras模型
        """
        model = models.Sequential(name='stacked_lstm')

        # 词嵌入层
        model.add(layers.Embedding(
            input_dim=self.max_words,
            output_dim=embedding_dim,
            input_length=self.max_len,
            name='embedding_word_vectors'
        ))

        # ============================================
        # 第一层LSTM：学习低级特征
        # ============================================
        # 【是什么】：第一层LSTM，返回完整序列
        # 【做什么】：
        #   - 处理每个词，输出每个时间步的隐状态
        #   - 学习词组级别的特征
        #   - 例如："非常好"、"不太好"等词组模式
        # 【为什么】：
        #   - return_sequences=True: 返回所有时间步
        #     * 为什么：需要传递给下一层LSTM
        #     * 输出形状：(batch, 200, 128)
        #   - units=128: 较大的容量
        #     * 为什么：第一层需要捕获丰富的低级特征
        model.add(layers.LSTM(
            units=128,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=True,  # 关键：返回序列给下一层
            name='lstm_layer1_low_level_features'
        ))

        # ============================================
        # 第二层LSTM：学习高级特征
        # ============================================
        # 【是什么】：第二层LSTM，只返回最后输出
        # 【做什么】：
        #   - 基于第一层的输出，学习更抽象的特征
        #   - 学习句子级别的语义
        #   - 例如：整体情感倾向、转折关系等
        # 【为什么】：
        #   - 多层的优势：
        #     * 第一层：词组特征（"非常好"）
        #     * 第二层：句子特征（"虽然...但是..."）
        #   - units=64: 较小的容量
        #     * 为什么：高级特征更抽象，不需要太大容量
        #     * 防止过拟合
        #   - return_sequences=False: 只返回最后输出
        #     * 为什么：用于分类，只需要整体表示
        model.add(layers.LSTM(
            units=64,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=False,
            name='lstm_layer2_high_level_features'
        ))

        # 全连接层、Dropout、输出层
        model.add(layers.Dense(64, activation='relu', name='dense_feature_combination'))
        model.add(layers.Dropout(0.5, name='dropout_regularization'))
        model.add(layers.Dense(1, activation='sigmoid', name='output_probability'))

        return model

    def create_model(self):
        """
        根据model_type创建对应的模型

        Returns:
            编译好的Keras模型
        """
        print(f"\n{'='*60}")
        print(f"创建模型: {self.model_type}")
        print(f"{'='*60}")

        if self.model_type == 'simple_lstm':
            model = self.create_simple_lstm()
        elif self.model_type == 'bilstm':
            model = self.create_bilstm()
        elif self.model_type == 'stacked_lstm':
            model = self.create_stacked_lstm()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # ============================================
        # 编译模型
        # ============================================
        # 【优化器】：Adam
        #   - 为什么用Adam：
        #     1. 自适应学习率，训练稳定
        #     2. 对超参数不敏感
        #     3. 是目前最常用的优化器
        #   - learning_rate=0.001: 默认学习率
        #     * 太大(0.01): 训练不稳定，可能不收敛
        #     * 太小(0.0001): 训练太慢
        #     * 0.001: 经验最佳值
        # 【损失函数】：binary_crossentropy
        #   - 为什么用这个：
        #     1. 二分类的标准损失函数
        #     2. 配合sigmoid输出
        #     3. 公式：-[y*log(p) + (1-y)*log(1-p)]
        # 【评估指标】：accuracy
        #   - 为什么用准确率：
        #     1. 直观易懂
        #     2. 数据平衡时效果好
        #     3. 如果数据不平衡，应该用F1-score
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=10, batch_size=32, callbacks=None):
        """
        训练模型

        Args:
            X_train: 训练数据（已经过预处理的序列）
            y_train: 训练标签（0或1）
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            callbacks: 回调函数列表

        Returns:
            训练历史
        """
        print(f"\n开始训练模型...")
        print(f"训练样本数: {len(X_train)}")
        print(f"序列长度: {self.max_len}")
        print(f"词汇表大小: {self.max_words}")

        # 创建模型
        self.model = self.create_model()

        # 打印模型结构
        print(f"\n模型结构:")
        self.model.summary()

        # 准备验证数据
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            print(f"验证样本数: {len(X_val)}")

        # 训练模型
        print(f"\n开始训练 (epochs={epochs}, batch_size={batch_size})...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✓ 模型训练完成！")

        return self.history

    def predict(self, X):
        """
        预测类别

        Args:
            X: 输入数据

        Returns:
            预测类别（0或1）
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """
        预测概率

        Args:
            X: 输入数据

        Returns:
            预测概率（0-1之间）
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        return self.model.predict(X).flatten()

    def evaluate(self, X, y):
        """
        评估模型

        Args:
            X: 测试数据
            y: 测试标签

        Returns:
            评估指标字典
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
        # 早停：验证集loss不再下降时停止训练
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

        # 学习率调度：验证集loss不下降时降低学习率
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
    """
    测试模型创建
    """
    print("="*60)
    print("LSTM情感分析模型测试")
    print("="*60)

    # 测试三种模型
    for model_type in ['simple_lstm', 'bilstm', 'stacked_lstm']:
        print(f"\n\n{'='*60}")
        print(f"测试模型: {model_type}")
        print(f"{'='*60}")

        analyzer = LSTMSentimentAnalyzer(model_type=model_type)
        model = analyzer.create_model()

        print(f"\n✓ {model_type} 模型创建成功！")
        print(f"总参数量: {model.count_params():,}")

    print("\n\n✓ 所有模型测试完成！")
