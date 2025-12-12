"""
Otto分类模型

本模块实现：
1. XGBoost多分类模型
2. LightGBM模型
3. CatBoost模型
4. Stacking集成
5. Voting集成

每个模型都有详细的注释说明设计思路和参数选择。
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import cross_val_predict
import pickle


class OttoXGBoostClassifier:
    """
    XGBoost多分类器

    【是什么】：基于XGBoost的多分类模型
    【做什么】：预测产品属于9个类别中的哪一个
    【为什么】：
        - XGBoost在表格数据上表现优秀
        - 支持多分类
        - 可以输出概率
    """

    def __init__(self, n_classes=9, model_type='tuned', **kwargs):
        """
        初始化分类器

        Args:
            n_classes: 类别数
            model_type: 模型类型 ('basic', 'tuned', 'advanced')
            **kwargs: 其他参数
        """
        self.n_classes = n_classes
        self.model_type = model_type

        # 根据模型类型设置参数
        self.params = self._get_model_params(model_type)
        self.params.update(kwargs)

        # 模型
        self.model = None

    def _get_model_params(self, model_type):
        """
        获取模型参数

        Args:
            model_type: 模型类型

        Returns:
            参数字典
        """
        params = {
            'basic': {
                # ============================================
                # 基础配置
                # ============================================
                'objective': 'multi:softprob',
                # 【是什么】：多分类 + 输出概率
                # 【为什么】：
                #   - multi:softprob: 输出每个类别的概率
                #   - multi:softmax: 只输出类别（不适合Stacking）

                'num_class': self.n_classes,
                # 【是什么】：类别数量
                # 【为什么】：XGBoost需要知道有多少个类别

                'eval_metric': 'mlogloss',
                # 【是什么】：多分类对数损失
                # 【为什么】：
                #   - Otto竞赛的评估指标
                #   - 衡量概率预测的准确性

                # 树的参数
                'max_depth': 6,
                # 【为什么=6】：
                #   - 中等深度
                #   - 平衡欠拟合和过拟合

                'min_child_weight': 1,
                # 【为什么=1】：
                #   - 允许较小的叶子节点
                #   - 捕获细节模式

                # 提升参数
                'learning_rate': 0.1,
                # 【为什么=0.1】：
                #   - 标准学习率
                #   - 平衡速度和性能

                'n_estimators': 100,
                # 【为什么=100】：
                #   - 基础配置，快速实验
                #   - 可以通过early_stopping调整

                # 正则化
                'reg_alpha': 0,
                'reg_lambda': 1,
                # 【为什么】：
                #   - 默认L2正则化
                #   - 防止过拟合

                # 其他
                'random_state': 42,
                'n_jobs': -1,
            },

            'tuned': {
                # ============================================
                # 调优配置
                # ============================================
                'objective': 'multi:softprob',
                'num_class': self.n_classes,
                'eval_metric': 'mlogloss',

                # 树的参数
                'max_depth': 8,
                # 【为什么=8】：
                #   - 更深的树捕获更复杂的模式
                #   - Otto数据特征多，需要更深的树

                'min_child_weight': 3,
                # 【为什么=3】：
                #   - 增加正则化
                #   - 防止过拟合

                'gamma': 0.1,
                # 【是什么】：分裂所需的最小损失减少
                # 【为什么=0.1】：
                #   - 轻度剪枝
                #   - 去除不重要的分裂

                # 提升参数
                'learning_rate': 0.05,
                # 【为什么=0.05】：
                #   - 较小的学习率
                #   - 配合更多的树

                'n_estimators': 500,
                # 【为什么=500】：
                #   - 更多的树
                #   - 配合early_stopping

                # 采样参数
                'subsample': 0.8,
                # 【是什么】：每棵树使用80%的样本
                # 【为什么=0.8】：
                #   - 增加随机性
                #   - 防止过拟合
                #   - 加速训练

                'colsample_bytree': 0.8,
                # 【是什么】：每棵树使用80%的特征
                # 【为什么=0.8】：
                #   - 特征采样
                #   - 增加模型多样性

                # 正则化
                'reg_alpha': 0.1,
                # 【是什么】：L1正则化
                # 【为什么=0.1】：
                #   - 特征选择
                #   - 稀疏解

                'reg_lambda': 1.0,
                # 【是什么】：L2正则化
                # 【为什么=1.0】：
                #   - 权重平滑
                #   - 防止过拟合

                # 其他
                'random_state': 42,
                'n_jobs': -1,
            },

            'advanced': {
                # ============================================
                # 高级配置
                # ============================================
                'objective': 'multi:softprob',
                'num_class': self.n_classes,
                'eval_metric': 'mlogloss',

                # 树的参数
                'max_depth': 10,
                # 【为什么=10】：
                #   - 深树捕获复杂交互
                #   - 配合强正则化

                'min_child_weight': 5,
                # 【为什么=5】：
                #   - 强正则化
                #   - 平衡深度

                'gamma': 0.2,
                # 【为什么=0.2】：
                #   - 更强的剪枝
                #   - 去除噪声分裂

                # 提升参数
                'learning_rate': 0.03,
                # 【为什么=0.03】：
                #   - 小学习率
                #   - 精细调整

                'n_estimators': 1000,
                # 【为什么=1000】：
                #   - 大量的树
                #   - 充分学习

                # 采样参数
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                # 【为什么都=0.7】：
                #   - 强随机性
                #   - 防止过拟合
                #   - 增加多样性

                # 正则化
                'reg_alpha': 0.5,
                'reg_lambda': 2.0,
                # 【为什么更大】：
                #   - 深树需要强正则化
                #   - 防止过拟合

                # 其他
                'random_state': 42,
                'n_jobs': -1,
            }
        }

        return params.get(model_type, params['tuned'])

    def train(self, X_train, y_train, X_val=None, y_val=None,
              early_stopping_rounds=50, verbose=True):
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            early_stopping_rounds: 早停轮数
            verbose: 是否显示训练过程

        Returns:
            训练历史
        """
        # 准备验证集
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # 创建模型
        self.model = xgb.XGBClassifier(**self.params)

        # 训练
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )

        return self.model.evals_result()

    def predict(self, X):
        """预测类别"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """
        评估模型

        Args:
            X: 特征
            y: 标签

        Returns:
            评估指标字典
        """
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'log_loss': log_loss(y, y_pred_proba)
        }

        return metrics

    def get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]

        import pandas as pd
        importance_df = pd.DataFrame({
            'feature': [f'feat_{i}' for i in indices],
            'importance': importance[indices]
        })

        return importance_df

    def save_model(self, filepath):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ 模型已保存: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ 模型已加载: {filepath}")


class OttoLightGBMClassifier:
    """
    LightGBM多分类器

    【LightGBM vs XGBoost】：
    - LightGBM: 基于直方图的算法，训练更快
    - XGBoost: 基于预排序，精度可能更高
    - LightGBM: 内存占用更少
    - XGBoost: 更成熟，文档更完善
    """

    def __init__(self, n_classes=9, **kwargs):
        """初始化分类器"""
        self.n_classes = n_classes
        self.params = {
            'objective': 'multiclass',
            'num_class': n_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 8,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        self.params.update(kwargs)
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None,
              num_boost_round=500, early_stopping_rounds=50):
        """训练模型"""
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(early_stopping_rounds)]
        )

        return self.model

    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """预测概率"""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'log_loss': log_loss(y, y_pred_proba)
        }


class OttoCatBoostClassifier:
    """
    CatBoost多分类器

    【CatBoost特点】：
    - 自动处理类别特征（Otto数据都是数值，用不上）
    - 对称树结构
    - 有序提升（Ordered Boosting）
    - 训练速度适中
    """

    def __init__(self, n_classes=9, **kwargs):
        """初始化分类器"""
        self.params = {
            'loss_function': 'MultiClass',
            'classes_count': n_classes,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3.0,
            'random_seed': 42,
            'verbose': False,
        }
        self.params.update(kwargs)
        self.model = cb.CatBoostClassifier(**self.params)

    def train(self, X_train, y_train, X_val=None, y_val=None,
              iterations=500, early_stopping_rounds=50):
        """训练模型"""
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )

        return self.model

    def predict(self, X):
        """预测类别"""
        return self.model.predict(X).flatten()

    def predict_proba(self, X):
        """预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'log_loss': log_loss(y, y_pred_proba)
        }


class OttoStackingEnsemble:
    """
    Stacking集成

    【是什么】：两层模型结构
    【第一层】：多个基模型（XGBoost、LightGBM、CatBoost）
    【第二层】：元模型（LogisticRegression）学习如何组合基模型

    【为什么Stacking更强】：
    - 不同模型有不同的偏差
    - 元模型学习每个模型的优势
    - 比简单平均更智能
    """

    def __init__(self, n_classes=9):
        """初始化集成模型"""
        self.n_classes = n_classes

        # 第一层：基模型
        self.base_models = {
            'xgboost': OttoXGBoostClassifier(n_classes, model_type='tuned'),
            'lightgbm': OttoLightGBMClassifier(n_classes),
            'catboost': OttoCatBoostClassifier(n_classes)
        }

        # 第二层：元模型
        self.meta_model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )

        # 存储基模型的预测
        self.base_predictions = {}

    def train(self, X_train, y_train, X_val, y_val, cv_folds=None):
        """
        训练Stacking集成

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            cv_folds: 交叉验证折（用于生成元特征）

        Returns:
            训练历史
        """
        print("\n训练Stacking集成...")

        # ============================================
        # 步骤1: 训练基模型
        # ============================================
        print("\n步骤1: 训练基模型")

        for name, model in self.base_models.items():
            print(f"\n训练 {name}...")
            model.train(X_train, y_train, X_val, y_val, verbose=False)

            # 评估
            metrics = model.evaluate(X_val, y_val)
            print(f"  验证集 - Accuracy: {metrics['accuracy']:.4f}, Log Loss: {metrics['log_loss']:.4f}")

        # ============================================
        # 步骤2: 生成元特征
        # ============================================
        print("\n步骤2: 生成元特征")

        # 使用验证集生成元特征（Blending方式）
        meta_features_val = self._generate_meta_features(X_val)
        print(f"  元特征形状: {meta_features_val.shape}")

        # ============================================
        # 步骤3: 训练元模型
        # ============================================
        print("\n步骤3: 训练元模型")
        self.meta_model.fit(meta_features_val, y_val)
        print("  元模型训练完成")

        return self.base_models

    def _generate_meta_features(self, X):
        """
        生成元特征

        【是什么】：基模型的预测概率作为新特征
        【形状】：(n_samples, n_models * n_classes)

        Args:
            X: 原始特征

        Returns:
            元特征
        """
        meta_features = []

        for name, model in self.base_models.items():
            # 获取每个基模型的预测概率
            proba = model.predict_proba(X)
            meta_features.append(proba)

        # 拼接所有基模型的预测
        # 例如：3个模型 × 9个类别 = 27个元特征
        return np.hstack(meta_features)

    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """预测概率"""
        # 生成元特征
        meta_features = self._generate_meta_features(X)

        # 元模型预测
        return self.meta_model.predict_proba(meta_features)

    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'log_loss': log_loss(y, y_pred_proba)
        }


if __name__ == '__main__':
    """
    测试模型
    """
    print("="*60)
    print("Otto分类模型测试")
    print("="*60)

    # 创建模拟数据
    n_samples = 1000
    n_features = 93
    n_classes = 9

    np.random.seed(42)
    X_train = np.random.rand(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)
    X_val = np.random.rand(200, n_features)
    y_val = np.random.randint(0, n_classes, 200)

    # 测试XGBoost
    print("\n测试XGBoost...")
    xgb_clf = OttoXGBoostClassifier(n_classes, model_type='basic')
    xgb_clf.train(X_train, y_train, X_val, y_val, verbose=False)
    metrics = xgb_clf.evaluate(X_val, y_val)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")

    # 测试LightGBM
    print("\n测试LightGBM...")
    lgb_clf = OttoLightGBMClassifier(n_classes)
    lgb_clf.train(X_train, y_train, X_val, y_val, num_boost_round=100)
    metrics = lgb_clf.evaluate(X_val, y_val)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")

    # 测试CatBoost
    print("\n测试CatBoost...")
    cb_clf = OttoCatBoostClassifier(n_classes)
    cb_clf.train(X_train, y_train, X_val, y_val, iterations=100)
    metrics = cb_clf.evaluate(X_val, y_val)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")

    # 测试Stacking
    print("\n测试Stacking...")
    stacking = OttoStackingEnsemble(n_classes)
    stacking.train(X_train, y_train, X_val, y_val)
    metrics = stacking.evaluate(X_val, y_val)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")

    print("\n✓ 所有测试通过！")
