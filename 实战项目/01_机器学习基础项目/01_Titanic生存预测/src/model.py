"""
模型定义和训练模块
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed


class TitanicPredictor:
    """Titanic生存预测器"""

    def __init__(self, model_type='random_forest', random_state=42):
        """
        初始化预测器

        Args:
            model_type: 模型类型
            random_state: 随机种子
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None

        set_seed(random_state)

    def create_model(self):
        """
        创建模型

        Returns:
            模型对象
        """
        models = {
            'logistic': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=5
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=5,
                learning_rate=0.1,
                eval_metric='logloss'
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=5,
                learning_rate=0.1,
                verbose=-1
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True,
                kernel='rbf'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=5,
                learning_rate=0.1
            ),
        }

        if self.model_type not in models:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        return models[self.model_type]

    def create_ensemble_model(self):
        """
        创建集成模型

        Returns:
            集成模型对象
        """
        # 创建基础模型
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            max_depth=10
        )

        xgb = XGBClassifier(
            n_estimators=100,
            random_state=self.random_state,
            max_depth=5,
            eval_metric='logloss'
        )

        lr = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )

        # 创建投票分类器
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb),
                ('lr', lr)
            ],
            voting='soft'
        )

        return ensemble

    def train(self, X_train, y_train, use_ensemble=False):
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            use_ensemble: 是否使用集成模型

        Returns:
            训练好的模型
        """
        print(f"\n开始训练模型: {self.model_type}")
        print(f"训练样本数: {len(X_train)}")
        print(f"特征数: {X_train.shape[1]}")

        # 保存特征名称
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values

        # 创建模型
        if use_ensemble:
            self.model = self.create_ensemble_model()
            print("使用集成模型 (Random Forest + XGBoost + Logistic Regression)")
        else:
            self.model = self.create_model()

        # 训练模型
        self.model.fit(X_train, y_train)
        print("✓ 模型训练完成")

        return self.model

    def predict(self, X):
        """
        预测

        Args:
            X: 特征

        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        预测概率

        Args:
            X: 特征

        Returns:
            预测概率
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """
        评估模型

        Args:
            X: 特征
            y: 真实标签

        Returns:
            dict: 评估指标
        """
        y_pred = self.predict(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }

        return metrics

    def cross_validate(self, X, y, cv=5):
        """
        交叉验证

        Args:
            X: 特征
            y: 标签
            cv: 折数

        Returns:
            dict: 交叉验证结果
        """
        if self.model is None:
            self.model = self.create_model()

        if isinstance(X, pd.DataFrame):
            X = X.values

        scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='accuracy'
        )

        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }

    def tune_hyperparameters(self, X_train, y_train, param_grid=None, cv=5):
        """
        超参数调优

        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格
            cv: 折数

        Returns:
            最佳模型
        """
        print(f"\n开始超参数调优...")

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values

        # 默认参数网格
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0]
                }
            else:
                print("⚠ 未提供参数网格，使用默认参数")
                self.model = self.create_model()
                self.model.fit(X_train, y_train)
                return self.model

        # 网格搜索
        base_model = self.create_model()
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"✓ 最佳参数: {grid_search.best_params_}")
        print(f"✓ 最佳得分: {grid_search.best_score_:.4f}")

        self.model = grid_search.best_estimator_
        return self.model

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

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names
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

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data.get('feature_names')

        print(f"✓ 模型已加载: {filepath}")

    def get_feature_importance(self, top_n=10):
        """
        获取特征重要性

        Args:
            top_n: 返回前N个重要特征

        Returns:
            pd.DataFrame: 特征重要性
        """
        if self.model is None:
            raise ValueError("模型未训练")

        if not hasattr(self.model, 'feature_importances_'):
            print("⚠ 该模型不支持特征重要性")
            return None

        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
        else:
            feature_names = self.feature_names

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)


def compare_models(X_train, y_train, X_val, y_val, random_state=42):
    """
    比较多个模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        random_state: 随机种子

    Returns:
        pd.DataFrame: 模型对比结果
    """
    print("\n" + "=" * 60)
    print("模型对比")
    print("=" * 60)

    model_types = [
        'logistic',
        'decision_tree',
        'random_forest',
        'xgboost',
        'lightgbm',
        'gradient_boosting'
    ]

    results = []

    for model_type in model_types:
        print(f"\n训练 {model_type}...")

        predictor = TitanicPredictor(model_type=model_type, random_state=random_state)
        predictor.train(X_train, y_train)

        # 评估
        metrics = predictor.evaluate(X_val, y_val)

        results.append({
            'Model': model_type,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1']
        })

        print(f"  准确率: {metrics['accuracy']:.4f}")

    # 创建结果DataFrame
    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

    print("\n" + "=" * 60)
    print("模型对比结果")
    print("=" * 60)
    print(results_df.to_string(index=False))

    return results_df


if __name__ == '__main__':
    from data import prepare_data

    print("=" * 60)
    print("Titanic模型训练测试")
    print("=" * 60)

    # 准备数据
    X_train, X_val, y_train, y_val, _ = prepare_data()

    # 训练单个模型
    predictor = TitanicPredictor(model_type='random_forest')
    predictor.train(X_train, y_train)

    # 评估
    metrics = predictor.evaluate(X_val, y_val)
    print(f"\n模型性能:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # 特征重要性
    importance = predictor.get_feature_importance()
    if importance is not None:
        print(f"\n特征重要性 (Top 5):")
        print(importance.head())

    print("\n✓ 模型训练测试完成！")
