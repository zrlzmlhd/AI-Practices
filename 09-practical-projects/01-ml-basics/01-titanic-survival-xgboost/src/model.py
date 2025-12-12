"""
XGBoost模型定义

本模块包含详细的XGBoost模型实现，每个参数都有详细的注释说明：
1. 这个参数是什么
2. 这个参数做什么
3. 为什么选择这个值
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed


class TitanicXGBoostClassifier:
    """
    Titanic生存预测XGBoost分类器

    支持三种模型配置：
    1. basic: 基础XGBoost（默认参数）
    2. tuned: 调优后的XGBoost
    3. advanced: 高级XGBoost（更多正则化）
    """

    def __init__(self, model_type='basic', random_state=42):
        """
        初始化分类器

        Args:
            model_type: 模型类型 ['basic', 'tuned', 'advanced']
            random_state: 随机种子
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.feature_importance = None

        set_seed(random_state)

    def create_basic_model(self):
        """
        创建基础XGBoost模型（使用默认参数）

        适用场景：
        - 快速原型验证
        - 建立基线模型
        - 数据量较小

        Returns:
            XGBoost分类器
        """
        # ============================================
        # 基础XGBoost模型
        # ============================================
        # 使用较少的参数，快速训练

        model = xgb.XGBClassifier(
            # ============================================
            # 树的参数
            # ============================================

            # max_depth: 树的最大深度
            # 【是什么】：决策树可以生长的最大层数
            # 【做什么】：控制树的复杂度
            # 【为什么=3】：
            #   - Titanic数据集较小（~900样本）
            #   - 浅树防止过拟合
            #   - 3层足够捕获主要的特征交互
            #   - 如果是大数据集，可以用6-10
            max_depth=3,

            # ============================================
            # 提升参数
            # ============================================

            # learning_rate: 学习率（也叫eta）
            # 【是什么】：每棵树的贡献权重
            # 【做什么】：控制模型学习速度
            # 【为什么=0.1】：
            #   - 默认值0.3太大，容易过拟合
            #   - 0.1是常用的稳定值
            #   - 较小的学习率需要更多树（n_estimators）
            #   - 学习率和树的数量需要平衡
            learning_rate=0.1,

            # n_estimators: 树的数量
            # 【是什么】：要训练多少棵树
            # 【做什么】：更多的树可以学习更复杂的模式
            # 【为什么=100】：
            #   - 配合learning_rate=0.1
            #   - 100棵树对小数据集足够
            #   - 使用early_stopping可以自动找到最佳数量
            n_estimators=100,

            # ============================================
            # 其他参数
            # ============================================

            # objective: 目标函数
            # 【是什么】：要优化的损失函数
            # 【做什么】：定义模型的学习目标
            # 【为什么='binary:logistic'】：
            #   - 二分类问题（生存/遇难）
            #   - 输出概率值（0-1之间）
            objective='binary:logistic',

            # eval_metric: 评估指标
            # 【是什么】：用于评估模型性能的指标
            # 【做什么】：在训练过程中监控模型表现
            # 【为什么='logloss'】：
            #   - 对数损失，适合概率预测
            #   - 也可以用'auc'（ROC曲线下面积）
            eval_metric='logloss',

            # random_state: 随机种子
            # 【是什么】：控制随机性的种子
            # 【做什么】：保证结果可复现
            random_state=self.random_state,

            # use_label_encoder: 是否使用标签编码器
            # 【为什么=False】：避免警告，我们已经手动处理了标签
            use_label_encoder=False
        )

        return model

    def create_tuned_model(self):
        """
        创建调优后的XGBoost模型

        适用场景：
        - 追求更高准确率
        - 有足够的训练时间
        - 需要防止过拟合

        Returns:
            XGBoost分类器
        """
        model = xgb.XGBClassifier(
            # ============================================
            # 树的参数（更精细的控制）
            # ============================================

            # max_depth: 稍微增加深度
            # 【为什么=4】：比基础模型深一层，捕获更复杂的交互
            max_depth=4,

            # min_child_weight: 最小叶子权重
            # 【是什么】：叶子节点所需的最小样本权重和
            # 【做什么】：控制叶子节点的最小样本数
            # 【为什么=1】：
            #   - 允许较小的叶子节点
            #   - 可以学习更细致的模式
            #   - 如果过拟合，可以增加到3-5
            min_child_weight=1,

            # gamma: 分裂所需的最小损失减少
            # 【是什么】：节点分裂所需的最小增益
            # 【做什么】：控制树的生长（正则化）
            # 【为什么=0.1】：
            #   - 轻微的正则化
            #   - 防止过度分裂
            #   - 0表示不限制，0.1-0.2是常用值
            gamma=0.1,

            # ============================================
            # 提升参数
            # ============================================

            learning_rate=0.05,  # 降低学习率，更稳定
            n_estimators=200,    # 增加树的数量，配合较小的学习率

            # ============================================
            # 采样参数（防止过拟合）
            # ============================================

            # subsample: 行采样比例
            # 【是什么】：每棵树使用多少比例的样本
            # 【做什么】：随机采样，增加模型多样性
            # 【为什么=0.8】：
            #   - 使用80%的样本训练每棵树
            #   - 类似于随机森林的bagging
            #   - 防止过拟合，提高泛化能力
            #   - 0.5-1.0都是常用值
            subsample=0.8,

            # colsample_bytree: 列采样比例
            # 【是什么】：每棵树使用多少比例的特征
            # 【做什么】：随机选择特征，增加多样性
            # 【为什么=0.8】：
            #   - 使用80%的特征训练每棵树
            #   - 防止某些特征过度主导
            #   - 提高模型鲁棒性
            colsample_bytree=0.8,

            # ============================================
            # 正则化参数
            # ============================================

            # reg_alpha: L1正则化
            # 【是什么】：权重的L1范数惩罚
            # 【做什么】：使某些权重变为0（特征选择）
            # 【为什么=0.1】：
            #   - 轻微的L1正则化
            #   - 可以自动进行特征选择
            #   - 产生稀疏解
            reg_alpha=0.1,

            # reg_lambda: L2正则化
            # 【是什么】：权重的L2范数惩罚
            # 【做什么】：使权重变小（平滑）
            # 【为什么=1】：
            #   - XGBoost默认值
            #   - 防止权重过大
            #   - 提高模型稳定性
            reg_lambda=1,

            # ============================================
            # 其他参数
            # ============================================

            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.random_state,
            use_label_encoder=False
        )

        return model

    def create_advanced_model(self):
        """
        创建高级XGBoost模型（更强的正则化）

        适用场景：
        - 数据量很小，容易过拟合
        - 需要最好的泛化能力
        - 可以接受较长的训练时间

        Returns:
            XGBoost分类器
        """
        model = xgb.XGBClassifier(
            # 树的参数：更保守
            max_depth=3,              # 浅树
            min_child_weight=3,       # 更大的最小叶子权重
            gamma=0.2,                # 更强的分裂限制

            # 提升参数：更慢更稳
            learning_rate=0.01,       # 很小的学习率
            n_estimators=500,         # 很多树来补偿小学习率

            # 采样参数：更多随机性
            subsample=0.7,            # 只用70%的样本
            colsample_bytree=0.7,     # 只用70%的特征

            # 正则化参数：更强
            reg_alpha=0.5,            # 更强的L1正则化
            reg_lambda=2,             # 更强的L2正则化

            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.random_state,
            use_label_encoder=False
        )

        return model

    def create_model(self):
        """
        根据model_type创建对应的模型

        Returns:
            XGBoost分类器
        """
        print(f"\n{'='*60}")
        print(f"创建模型: {self.model_type}")
        print(f"{'='*60}")

        if self.model_type == 'basic':
            model = self.create_basic_model()
        elif self.model_type == 'tuned':
            model = self.create_tuned_model()
        elif self.model_type == 'advanced':
            model = self.create_advanced_model()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              early_stopping_rounds=10, verbose=True):
        """
        训练模型

        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            early_stopping_rounds: 早停轮数
            verbose: 是否显示训练过程

        Returns:
            训练历史
        """
        print(f"\n开始训练模型...")
        print(f"训练样本数: {len(X_train)}")

        # 保存特征名
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()

        # 创建模型
        self.model = self.create_model()

        # 打印模型参数
        print(f"\n模型参数:")
        params = self.model.get_params()
        for key in ['max_depth', 'learning_rate', 'n_estimators',
                   'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
            if key in params:
                print(f"  {key}: {params[key]}")

        # 准备验证集
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            print(f"验证样本数: {len(X_val)}")

        # 训练模型
        print(f"\n开始训练...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose=verbose
        )

        # 获取特征重要性
        self.feature_importance = self.model.feature_importances_

        print("\n✓ 模型训练完成！")

        # 如果使用了early stopping，显示最佳迭代
        if hasattr(self.model, 'best_iteration'):
            print(f"最佳迭代: {self.model.best_iteration}")

        return self.model

    def predict(self, X):
        """
        预测类别

        Args:
            X: 输入数据

        Returns:
            预测类别
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        return self.model.predict(X)

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

        return self.model.predict_proba(X)

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

        # 预测
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]

        # 计算指标
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)

        return {
            'accuracy': accuracy,
            'auc': auc
        }

    def get_feature_importance(self, top_n=None):
        """
        获取特征重要性

        Args:
            top_n: 返回前N个重要特征

        Returns:
            特征重要性DataFrame
        """
        if self.feature_importance is None:
            raise ValueError("模型未训练，请先调用train()方法")

        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        })

        # 排序
        importance_df = importance_df.sort_values('importance', ascending=False)

        # 返回前N个
        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df

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

        # 保存模型和元数据
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }

        joblib.dump(model_data, filepath)
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

        # 加载模型和元数据
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']

        print(f"✓ 模型已加载: {filepath}")


if __name__ == '__main__':
    """
    测试模型创建
    """
    print("="*60)
    print("XGBoost模型测试")
    print("="*60)

    # 测试三种模型
    for model_type in ['basic', 'tuned', 'advanced']:
        print(f"\n\n{'='*60}")
        print(f"测试模型: {model_type}")
        print(f"{'='*60}")

        classifier = TitanicXGBoostClassifier(model_type=model_type)
        model = classifier.create_model()

        print(f"\n✓ {model_type} 模型创建成功！")
        print(f"模型参数:")
        params = model.get_params()
        for key, value in params.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")

    print("\n\n✓ 所有模型测试完成！")
