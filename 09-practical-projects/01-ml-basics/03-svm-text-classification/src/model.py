"""
SVM文本分类模型模块

本模块实现：
1. 多种SVM核函数（线性、RBF、多项式）
2. 超参数调优（网格搜索、随机搜索）
3. 模型集成（投票、Stacking）
4. 模型解释（特征重要性）

【核心概念】：
- SVM：支持向量机，寻找最优分类超平面
- 核技巧：将数据映射到高维空间，使其线性可分
- C参数：正则化强度（越小正则化越强）
- gamma参数：RBF核的影响范围
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time

# SVM模型
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# 模型选择和评估
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 模型集成
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


class SVMTextClassifier:
    """
    SVM文本分类器

    【是什么】：基于支持向量机的文本分类模型
    【支持的核函数】：
        - linear：线性核（适合高维稀疏数据）
        - rbf：径向基核（适合非线性数据）
        - poly：多项式核
    """

    def __init__(self, kernel='linear', C=1.0, gamma='scale', **kwargs):
        """
        初始化SVM分类器

        Args:
            kernel: 核函数类型
            C: 正则化参数
                【为什么】：
                - C越大：更关注训练准确率，可能过拟合
                - C越小：更关注泛化能力，可能欠拟合
            gamma: RBF核参数
                【为什么】：
                - gamma越大：决策边界越复杂
                - gamma越小：决策边界越平滑
            **kwargs: 其他参数
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.kwargs = kwargs

        # 创建模型
        if kernel == 'linear':
            # 【LinearSVC】：线性核的优化实现
            # 【为什么用LinearSVC】：
            #   - 比SVC(kernel='linear')更快
            #   - 适合高维稀疏数据（如TF-IDF）
            self.model = LinearSVC(
                C=C,
                max_iter=kwargs.get('max_iter', 1000),
                random_state=42
            )
            # 【CalibratedClassifierCV】：校准概率输出
            # 【为什么】：LinearSVC不直接输出概率
            self.model = CalibratedClassifierCV(self.model, cv=3)

        else:
            # 【SVC】：支持多种核函数
            self.model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                probability=True,  # 输出概率
                random_state=42,
                **kwargs
            )

    def fit(self, X_train, y_train, verbose=True):
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            verbose: 是否显示训练信息
        """
        if verbose:
            print(f"\n训练SVM模型 (kernel={self.kernel}, C={self.C})...")
            start_time = time.time()

        self.model.fit(X_train, y_train)

        if verbose:
            elapsed = time.time() - start_time
            print(f"  训练完成，耗时: {elapsed:.2f}秒")

    def predict(self, X):
        """预测"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, label_names=None):
        """
        评估模型

        Args:
            X_test: 测试特征
            y_test: 测试标签
            label_names: 类别名称

        Returns:
            评估指标字典
        """
        y_pred = self.predict(X_test)

        # 准确率
        accuracy = accuracy_score(y_test, y_pred)

        # 分类报告
        report = classification_report(
            y_test, y_pred,
            target_names=label_names,
            output_dict=True
        )

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }

    def get_feature_importance(self, feature_names, top_n=20):
        """
        获取特征重要性（仅线性核）

        【原理】：线性SVM的权重向量表示特征重要性
        【解释】：权重绝对值越大，特征越重要

        Args:
            feature_names: 特征名称列表
            top_n: 返回前N个重要特征

        Returns:
            每个类别的重要特征
        """
        if self.kernel != 'linear':
            print("特征重要性仅支持线性核")
            return None

        # 获取权重
        if isinstance(self.model, CalibratedClassifierCV):
            # LinearSVC包装在CalibratedClassifierCV中
            coef = self.model.calibrated_classifiers_[0].estimator.coef_
        else:
            coef = self.model.coef_

        # 每个类别的重要特征
        importance_dict = {}

        for i, class_coef in enumerate(coef):
            # 获取权重最大的特征
            top_indices = np.argsort(np.abs(class_coef))[-top_n:][::-1]
            top_features = [(feature_names[idx], class_coef[idx])
                           for idx in top_indices]

            importance_dict[f'class_{i}'] = top_features

        return importance_dict

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


class SVMHyperparameterTuner:
    """
    SVM超参数调优器

    【是什么】：自动搜索最优超参数
    【支持的方法】：
        - 网格搜索（Grid Search）：遍历所有组合
        - 随机搜索（Random Search）：随机采样组合
    """

    def __init__(self, kernel='linear', search_method='grid', cv=5):
        """
        初始化调优器

        Args:
            kernel: 核函数类型
            search_method: 搜索方法 ('grid' 或 'random')
            cv: 交叉验证折数
        """
        self.kernel = kernel
        self.search_method = search_method
        self.cv = cv
        self.best_model = None
        self.best_params = None
        self.search_results = None

    def get_param_grid(self):
        """
        获取参数网格

        【参数说明】：
        - C: 正则化强度
        - gamma: RBF核参数
        - degree: 多项式核的度数
        """
        if self.kernel == 'linear':
            return {
                'C': [0.01, 0.1, 1, 10, 100],
                'max_iter': [1000]
            }

        elif self.kernel == 'rbf':
            return {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }

        elif self.kernel == 'poly':
            return {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'degree': [2, 3, 4]
            }

    def tune(self, X_train, y_train, n_iter=20, verbose=True):
        """
        执行超参数调优

        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_iter: 随机搜索的迭代次数
            verbose: 是否显示详细信息
        """
        print("\n" + "="*60)
        print(f"SVM超参数调优 (kernel={self.kernel}, method={self.search_method})")
        print("="*60)

        # 创建基础模型
        if self.kernel == 'linear':
            base_model = LinearSVC(random_state=42)
        else:
            base_model = SVC(kernel=self.kernel, probability=True, random_state=42)

        # 参数网格
        param_grid = self.get_param_grid()

        if verbose:
            print(f"\n参数空间:")
            for param, values in param_grid.items():
                print(f"  {param}: {values}")

        # 搜索
        start_time = time.time()

        if self.search_method == 'grid':
            # 【网格搜索】：遍历所有参数组合
            # 【优点】：找到全局最优
            # 【缺点】：计算量大
            searcher = GridSearchCV(
                base_model,
                param_grid,
                cv=self.cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1 if verbose else 0
            )

        else:  # random
            # 【随机搜索】：随机采样参数组合
            # 【优点】：计算量小
            # 【缺点】：可能错过最优解
            searcher = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=self.cv,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=1 if verbose else 0
            )

        searcher.fit(X_train, y_train)

        elapsed = time.time() - start_time

        # 保存结果
        self.best_model = searcher.best_estimator_
        self.best_params = searcher.best_params_
        self.search_results = pd.DataFrame(searcher.cv_results_)

        # 打印结果
        print(f"\n调优完成，耗时: {elapsed:.2f}秒")
        print(f"\n最佳参数:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\n最佳交叉验证准确率: {searcher.best_score_:.4f}")

        return self.best_model, self.best_params

    def get_top_configs(self, top_n=5):
        """获取最佳的N个配置"""
        if self.search_results is None:
            return None

        results = self.search_results.sort_values('rank_test_score')
        top_results = results.head(top_n)[['params', 'mean_test_score', 'std_test_score']]

        return top_results


class SVMEnsembleClassifier:
    """
    SVM集成分类器

    【是什么】：组合多个SVM模型
    【方法】：
        - Voting：投票集成
        - Stacking：堆叠集成
    """

    def __init__(self, ensemble_method='voting'):
        """
        初始化集成分类器

        Args:
            ensemble_method: 集成方法 ('voting' 或 'stacking')
        """
        self.ensemble_method = ensemble_method
        self.model = None

    def build_voting_ensemble(self):
        """
        构建投票集成

        【是什么】：多个模型投票决定最终预测
        【包含】：
            - 线性SVM（快速，适合高维）
            - RBF SVM（非线性）
            - 逻辑回归（概率校准好）
            - 朴素贝叶斯（快速基线）
        """
        print("\n构建投票集成模型...")

        # 线性SVM
        linear_svm = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=1000, random_state=42),
            cv=3
        )

        # RBF SVM
        rbf_svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)

        # 逻辑回归
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        # 朴素贝叶斯
        nb = MultinomialNB(alpha=0.1)

        # 投票集成
        self.model = VotingClassifier(
            estimators=[
                ('linear_svm', linear_svm),
                ('rbf_svm', rbf_svm),
                ('lr', lr),
                ('nb', nb)
            ],
            voting='soft',  # 软投票（基于概率）
            n_jobs=-1
        )

        print("  包含模型: Linear SVM, RBF SVM, Logistic Regression, Naive Bayes")

    def fit(self, X_train, y_train, verbose=True):
        """训练集成模型"""
        if self.model is None:
            if self.ensemble_method == 'voting':
                self.build_voting_ensemble()
            else:
                raise ValueError(f"不支持的集成方法: {self.ensemble_method}")

        if verbose:
            print(f"\n训练集成模型...")
            start_time = time.time()

        self.model.fit(X_train, y_train)

        if verbose:
            elapsed = time.time() - start_time
            print(f"  训练完成，耗时: {elapsed:.2f}秒")

    def predict(self, X):
        """预测"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, label_names=None):
        """评估模型"""
        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=label_names,
            output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }


if __name__ == '__main__':
    """测试模型模块"""
    print("="*60)
    print("SVM文本分类模型测试")
    print("="*60)

    # 创建模拟数据
    from sklearn.datasets import make_classification

    X_train, y_train = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=50,
        n_classes=3,
        random_state=42
    )

    X_test, y_test = make_classification(
        n_samples=200,
        n_features=100,
        n_informative=50,
        n_classes=3,
        random_state=43
    )

    # 测试线性SVM
    print("\n测试线性SVM...")
    svm = SVMTextClassifier(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)
    results = svm.evaluate(X_test, y_test)
    print(f"  准确率: {results['accuracy']:.4f}")

    # 测试超参数调优
    print("\n测试超参数调优...")
    tuner = SVMHyperparameterTuner(kernel='linear', search_method='grid')
    best_model, best_params = tuner.tune(X_train, y_train, verbose=False)
    print(f"  最佳参数: {best_params}")

    # 测试集成模型
    print("\n测试集成模型...")
    ensemble = SVMEnsembleClassifier(ensemble_method='voting')
    ensemble.fit(X_train, y_train, verbose=False)
    results = ensemble.evaluate(X_test, y_test)
    print(f"  准确率: {results['accuracy']:.4f}")

    print("\n✓ 模型测试通过！")
