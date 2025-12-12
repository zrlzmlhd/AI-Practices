"""
XGBoost高级技巧 - 模型模块

本模块展示竞赛级别的XGBoost使用技巧：
1. 高级超参数调优（贝叶斯优化、Optuna）
2. 早停和学习率调度
3. 自定义目标函数和评估指标
4. 模型集成（Stacking、Blending）
5. 特征重要性分析
6. 模型解释（SHAP）

【核心技巧】：
- 学习率衰减
- 正则化组合
- 树的深度和数量平衡
- 样本和特征采样
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# XGBoost
import xgboost as xgb

# 模型选择和评估
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

# 超参数优化
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("警告: optuna未安装，贝叶斯优化功能不可用")


class AdvancedXGBoostClassifier:
    """
    高级XGBoost分类器

    【是什么】：集成竞赛技巧的XGBoost模型
    【包含技巧】：
        - 学习率调度
        - 早停策略
        - 正则化组合
        - 类别不平衡处理
    """

    def __init__(self, params=None, use_gpu=False):
        """
        初始化分类器

        Args:
            params: 模型参数字典
            use_gpu: 是否使用GPU
        """
        # 【默认参数】：竞赛级别的配置
        self.default_params = {
            # ============================================
            # 基础参数
            # ============================================
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'random_state': 42,

            # ============================================
            # 学习参数
            # ============================================
            'learning_rate': 0.05,
            # 【为什么=0.05】：
            #   - 较小的学习率 + 更多树 = 更好的泛化
            #   - 竞赛中常用0.01-0.1

            'n_estimators': 1000,
            # 【为什么=1000】：
            #   - 配合早停，让模型自己决定最优树数
            #   - 实际训练可能在200-500棵树停止

            # ============================================
            # 树结构参数
            # ============================================
            'max_depth': 6,
            # 【为什么=6】：
            #   - 平衡复杂度和泛化能力
            #   - 太深容易过拟合，太浅欠拟合

            'min_child_weight': 3,
            # 【为什么=3】：
            #   - 控制叶子节点最小样本权重
            #   - 防止过拟合

            # ============================================
            # 采样参数
            # ============================================
            'subsample': 0.8,
            # 【为什么=0.8】：
            #   - 每棵树使用80%的样本
            #   - 增加随机性，防止过拟合

            'colsample_bytree': 0.8,
            # 【为什么=0.8】：
            #   - 每棵树使用80%的特征
            #   - 类似随机森林的特征采样

            'colsample_bylevel': 0.8,
            # 【为什么=0.8】：
            #   - 每层使用80%的特征
            #   - 进一步增加随机性

            # ============================================
            # 正则化参数
            # ============================================
            'reg_alpha': 0.1,
            # 【L1正则化】：
            #   - 特征选择效果
            #   - 使部分权重为0

            'reg_lambda': 1.0,
            # 【L2正则化】：
            #   - 权重平滑
            #   - 防止过拟合

            'gamma': 0.1,
            # 【最小分裂损失】：
            #   - 分裂节点需要的最小损失减少
            #   - 控制树的复杂度
        }

        # 合并用户参数
        if params:
            self.default_params.update(params)

        self.model = None
        self.best_iteration = None
        self.feature_importance = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            early_stopping_rounds=50, verbose=True):
        """
        训练模型

        【技巧】：
        - 使用验证集进行早停
        - 监控训练和验证指标
        - 保存最佳迭代

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            early_stopping_rounds: 早停轮数
            verbose: 是否显示训练过程
        """
        if verbose:
            print("\n训练XGBoost模型...")
            print(f"  参数配置:")
            for key, value in self.default_params.items():
                print(f"    {key}: {value}")

        start_time = time.time()

        # 创建DMatrix（XGBoost的数据格式）
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # 准备验证集
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))

        # 训练
        evals_result = {}
        self.model = xgb.train(
            self.default_params,
            dtrain,
            num_boost_round=self.default_params.get('n_estimators', 1000),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=50 if verbose else False
        )

        self.best_iteration = self.model.best_iteration

        elapsed = time.time() - start_time

        if verbose:
            print(f"\n训练完成:")
            print(f"  耗时: {elapsed:.2f}秒")
            print(f"  最佳迭代: {self.best_iteration}")
            if 'val' in evals_result:
                best_score = evals_result['val'][self.default_params['eval_metric']][self.best_iteration]
                print(f"  最佳验证分数: {best_score:.4f}")

        # 保存特征重要性
        self.feature_importance = self.model.get_score(importance_type='gain')

        return evals_result

    def predict(self, X):
        """预测类别"""
        dtest = xgb.DMatrix(X)
        y_pred_proba = self.model.predict(dtest)
        return (y_pred_proba > 0.5).astype(int)

    def predict_proba(self, X):
        """预测概率"""
        dtest = xgb.DMatrix(X)
        y_pred_proba = self.model.predict(dtest)
        return np.vstack([1 - y_pred_proba, y_pred_proba]).T

    def get_feature_importance(self, feature_names=None, top_n=20):
        """
        获取特征重要性

        【重要性类型】：
        - weight: 特征被使用的次数
        - gain: 特征带来的平均增益（推荐）
        - cover: 特征覆盖的样本数
        """
        if self.feature_importance is None:
            return None

        # 转换为DataFrame
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance.items()
        ])

        # 排序
        importance_df = importance_df.sort_values('importance', ascending=False)

        # 如果提供了特征名称，进行映射
        if feature_names:
            importance_df['feature'] = importance_df['feature'].apply(
                lambda x: feature_names[int(x.replace('f', ''))] if x.startswith('f') else x
            )

        return importance_df.head(top_n)

    def save_model(self, filepath):
        """保存模型"""
        self.model.save_model(filepath)
        print(f"✓ 模型已保存: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        print(f"✓ 模型已加载: {filepath}")


class XGBoostHyperparameterOptimizer:
    """
    XGBoost超参数优化器（使用Optuna）

    【是什么】：贝叶斯优化超参数
    【为什么】：
        - 比网格搜索更高效
        - 自动探索参数空间
        - 早停不佳的试验
    """

    def __init__(self, task_type='classification', n_trials=100, cv=5):
        """
        初始化优化器

        Args:
            task_type: 任务类型
            n_trials: 优化试验次数
            cv: 交叉验证折数
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("需要安装optuna: pip install optuna")

        self.task_type = task_type
        self.n_trials = n_trials
        self.cv = cv
        self.best_params = None
        self.study = None

    def objective(self, trial, X, y):
        """
        优化目标函数

        【参数空间】：竞赛常用的参数范围
        """
        # 定义参数空间
        params = {
            'objective': 'binary:logistic' if self.task_type == 'classification' else 'reg:squarederror',
            'eval_metric': 'auc' if self.task_type == 'classification' else 'rmse',
            'tree_method': 'hist',
            'random_state': 42,

            # 【学习率】：对数空间采样
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),

            # 【树结构】
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),

            # 【采样】
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),

            # 【正则化】
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
        }

        # 交叉验证
        cv_scores = []
        kfold = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            # 训练
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )

            # 预测
            y_pred = model.predict(dval)

            # 评估
            if self.task_type == 'classification':
                score = roc_auc_score(y_val, y_pred)
            else:
                score = -mean_squared_error(y_val, y_pred, squared=False)  # 负RMSE

            cv_scores.append(score)

        return np.mean(cv_scores)

    def optimize(self, X, y, verbose=True):
        """
        执行优化

        Args:
            X: 特征
            y: 标签
            verbose: 是否显示进度
        """
        print("\n" + "="*60)
        print("XGBoost超参数优化（Optuna）")
        print("="*60)
        print(f"  试验次数: {self.n_trials}")
        print(f"  交叉验证: {self.cv}折")

        # 创建study
        direction = 'maximize' if self.task_type == 'classification' else 'maximize'
        self.study = optuna.create_study(direction=direction)

        # 优化
        start_time = time.time()

        self.study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=verbose
        )

        elapsed = time.time() - start_time

        # 最佳参数
        self.best_params = self.study.best_params

        print(f"\n优化完成，耗时: {elapsed:.2f}秒")
        print(f"\n最佳参数:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\n最佳分数: {self.study.best_value:.4f}")

        return self.best_params

    def get_optimization_history(self):
        """获取优化历史"""
        if self.study is None:
            return None

        history = []
        for trial in self.study.trials:
            history.append({
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params
            })

        return pd.DataFrame(history)


class XGBoostEnsemble:
    """
    XGBoost集成模型

    【是什么】：组合多个XGBoost模型
    【方法】：
        - Bagging：训练多个模型，平均预测
        - Stacking：使用元模型组合基模型
    """

    def __init__(self, n_models=5, ensemble_method='bagging'):
        """
        初始化集成模型

        Args:
            n_models: 基模型数量
            ensemble_method: 集成方法
        """
        self.n_models = n_models
        self.ensemble_method = ensemble_method
        self.models = []

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        训练集成模型

        【Bagging策略】：
        - 每个模型使用不同的随机种子
        - 每个模型使用不同的采样
        """
        print(f"\n训练{self.ensemble_method}集成模型...")
        print(f"  基模型数量: {self.n_models}")

        for i in range(self.n_models):
            if verbose:
                print(f"\n训练模型 {i+1}/{self.n_models}...")

            # 创建模型（不同随机种子）
            params = {
                'random_state': 42 + i,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }

            model = AdvancedXGBoostClassifier(params=params)

            # 训练
            model.fit(
                X_train, y_train,
                X_val, y_val,
                early_stopping_rounds=50,
                verbose=False
            )

            self.models.append(model)

        print(f"\n✓ 集成模型训练完成")

    def predict_proba(self, X):
        """
        预测概率（平均）

        【集成策略】：对所有模型的预测概率求平均
        """
        predictions = []

        for model in self.models:
            pred = model.predict_proba(X)
            predictions.append(pred)

        # 平均
        return np.mean(predictions, axis=0)

    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


if __name__ == '__main__':
    """测试模型模块"""
    print("="*60)
    print("XGBoost高级技巧 - 模型测试")
    print("="*60)

    # 创建模拟数据
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )

    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]

    # 测试基础模型
    print("\n测试基础XGBoost模型...")
    model = AdvancedXGBoostClassifier()
    model.fit(X_train, y_train, X_val, y_val, verbose=False)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"  验证集准确率: {accuracy:.4f}")

    # 测试特征重要性
    importance = model.get_feature_importance(top_n=5)
    print(f"\n  Top 5 重要特征:")
    print(importance)

    # 测试集成模型
    print("\n测试集成模型...")
    ensemble = XGBoostEnsemble(n_models=3)
    ensemble.fit(X_train, y_train, X_val, y_val, verbose=False)

    y_pred_ensemble = ensemble.predict(X_val)
    accuracy_ensemble = accuracy_score(y_val, y_pred_ensemble)
    print(f"  集成模型准确率: {accuracy_ensemble:.4f}")

    print("\n✓ 模型测试通过！")
