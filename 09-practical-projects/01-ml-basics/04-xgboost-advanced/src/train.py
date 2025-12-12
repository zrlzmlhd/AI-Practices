"""
XGBoost高级技巧训练脚本

使用方法:
    python src/train.py --mode basic
    python src/train.py --mode optimize --n_trials 50
    python src/train.py --mode ensemble --n_models 5

【训练模式】:
- basic: 基础训练
- optimize: 超参数优化
- ensemble: 集成模型
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_california_housing_data, prepare_advanced_features
from src.model import AdvancedXGBoostClassifier, XGBoostHyperparameterOptimizer, XGBoostEnsemble


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练XGBoost高级模型')

    # 训练模式
    parser.add_argument('--mode', type=str, default='basic',
                       choices=['basic', 'optimize', 'ensemble'],
                       help='训练模式')

    # 特征工程
    parser.add_argument('--create_interactions', action='store_true', default=True,
                       help='创建交互特征')
    parser.add_argument('--create_polynomials', action='store_true', default=True,
                       help='创建多项式特征')
    parser.add_argument('--create_statistical', action='store_true', default=True,
                       help='创建统计特征')
    parser.add_argument('--feature_selection', action='store_true', default=True,
                       help='特征选择')
    parser.add_argument('--top_k_features', type=int, default=100,
                       help='保留的特征数')

    # 优化参数
    parser.add_argument('--n_trials', type=int, default=50,
                       help='优化试验次数')

    # 集成参数
    parser.add_argument('--n_models', type=int, default=5,
                       help='集成模型数量')

    # 保存路径
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def plot_feature_importance(importance_df, save_path):
    """绘制特征重要性"""
    plt.figure(figsize=(12, 8))

    # 取前20个特征
    top_features = importance_df.head(20)

    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('重要性（Gain）', fontsize=12, fontweight='bold')
    plt.title('Top 20 特征重要性', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 特征重要性图已保存: {save_path}")
    plt.close()


def plot_training_history(evals_result, save_path):
    """绘制训练历史"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 训练集
    if 'train' in evals_result:
        metric_name = list(evals_result['train'].keys())[0]
        train_scores = evals_result['train'][metric_name]
        ax.plot(train_scores, label='训练集', linewidth=2)

    # 验证集
    if 'val' in evals_result:
        metric_name = list(evals_result['val'].keys())[0]
        val_scores = evals_result['val'][metric_name]
        ax.plot(val_scores, label='验证集', linewidth=2)

    ax.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name.upper(), fontsize=12, fontweight='bold')
    ax.set_title('训练历史', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练历史图已保存: {save_path}")
    plt.close()


def plot_optimization_history(history_df, save_path):
    """绘制优化历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 优化过程
    axes[0].plot(history_df['trial'], history_df['value'], marker='o', linewidth=2)
    axes[0].set_xlabel('试验次数', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('分数', fontsize=12, fontweight='bold')
    axes[0].set_title('优化过程', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 最佳分数累积
    best_scores = history_df['value'].cummax()
    axes[1].plot(history_df['trial'], best_scores, marker='o', linewidth=2, color='green')
    axes[1].set_xlabel('试验次数', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('最佳分数', fontsize=12, fontweight='bold')
    axes[1].set_title('最佳分数演化', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 优化历史图已保存: {save_path}")
    plt.close()


def main():
    """主训练流程"""
    args = parse_args()

    print("="*60)
    print("XGBoost高级技巧 - 模型训练")
    print("="*60)
    print(f"\n训练配置:")
    print(f"  训练模式: {args.mode}")
    print(f"  特征工程: 交互={args.create_interactions}, 多项式={args.create_polynomials}, 统计={args.create_statistical}")
    print(f"  特征选择: {args.feature_selection} (Top {args.top_k_features})")

    # 创建保存目录
    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / args.model_dir
    result_dir = project_dir / args.result_dir
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    # ============================================
    # 1. 准备数据
    # ============================================
    print("\n" + "="*60)
    print("步骤1: 数据准备")
    print("="*60)

    # 加载数据
    X, y, feature_names = load_california_housing_data()

    # 转换为二分类任务
    y_binary = (y > y.median()).astype(int)
    print(f"\n转换为二分类任务:")
    print(f"  类别0（低房价）: {np.sum(y_binary == 0)}")
    print(f"  类别1（高房价）: {np.sum(y_binary == 1)}")

    # 高级特征工程
    (X_train, y_train), (X_test, y_test), selected_features = \
        prepare_advanced_features(
            X, y_binary,
            task_type='classification',
            create_interactions=args.create_interactions,
            create_polynomials=args.create_polynomials,
            create_statistical=args.create_statistical,
            feature_selection=args.feature_selection,
            top_k_features=args.top_k_features
        )

    # 进一步划分验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    print(f"\n最终数据划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")

    # 保存特征名称
    features_path = model_dir / 'selected_features.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(selected_features, f)

    # ============================================
    # 2. 训练模型
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 训练模型")
    print("="*60)

    if args.mode == 'basic':
        # ============================================
        # 基础训练
        # ============================================
        print("\n基础XGBoost训练...")

        model = AdvancedXGBoostClassifier()
        evals_result = model.fit(
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            early_stopping_rounds=50,
            verbose=True
        )

        # 保存模型
        model_path = model_dir / 'xgboost_basic_model.json'
        model.save_model(model_path)

        # 绘制训练历史
        history_path = result_dir / 'training_history.png'
        plot_training_history(evals_result, history_path)

        # 特征重要性
        importance = model.get_feature_importance(selected_features, top_n=20)
        if importance is not None:
            importance_path = result_dir / 'feature_importance.png'
            plot_feature_importance(importance, importance_path)

            # 保存重要性数据
            importance.to_csv(result_dir / 'feature_importance.csv', index=False)

    elif args.mode == 'optimize':
        # ============================================
        # 超参数优化
        # ============================================
        print(f"\n超参数优化（{args.n_trials}次试验）...")

        optimizer = XGBoostHyperparameterOptimizer(
            task_type='classification',
            n_trials=args.n_trials,
            cv=5
        )

        best_params = optimizer.optimize(
            X_train.values, y_train.values,
            verbose=True
        )

        # 保存最佳参数
        params_path = model_dir / 'best_params.pkl'
        with open(params_path, 'wb') as f:
            pickle.dump(best_params, f)
        print(f"✓ 最佳参数已保存: {params_path}")

        # 使用最佳参数训练最终模型
        print("\n使用最佳参数训练最终模型...")
        model = AdvancedXGBoostClassifier(params=best_params)
        evals_result = model.fit(
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            early_stopping_rounds=50,
            verbose=True
        )

        # 保存模型
        model_path = model_dir / 'xgboost_optimized_model.json'
        model.save_model(model_path)

        # 绘制优化历史
        history = optimizer.get_optimization_history()
        opt_history_path = result_dir / 'optimization_history.png'
        plot_optimization_history(history, opt_history_path)

        # 保存优化历史
        history.to_csv(result_dir / 'optimization_history.csv', index=False)

    elif args.mode == 'ensemble':
        # ============================================
        # 集成模型
        # ============================================
        print(f"\n训练集成模型（{args.n_models}个基模型）...")

        ensemble = XGBoostEnsemble(
            n_models=args.n_models,
            ensemble_method='bagging'
        )

        ensemble.fit(
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            verbose=True
        )

        # 保存集成模型
        model_path = model_dir / 'xgboost_ensemble_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        print(f"✓ 集成模型已保存: {model_path}")

        model = ensemble

    # ============================================
    # 3. 评估模型
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 评估模型")
    print("="*60)

    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    # 验证集评估
    y_val_pred = model.predict(X_val.values)
    y_val_proba = model.predict_proba(X_val.values)[:, 1]

    val_accuracy = accuracy_score(y_val.values, y_val_pred)
    val_auc = roc_auc_score(y_val.values, y_val_proba)

    print(f"\n验证集性能:")
    print(f"  准确率: {val_accuracy:.4f}")
    print(f"  AUC: {val_auc:.4f}")

    # 测试集评估
    y_test_pred = model.predict(X_test.values)
    y_test_proba = model.predict_proba(X_test.values)[:, 1]

    test_accuracy = accuracy_score(y_test.values, y_test_pred)
    test_auc = roc_auc_score(y_test.values, y_test_proba)

    print(f"\n测试集性能:")
    print(f"  准确率: {test_accuracy:.4f}")
    print(f"  AUC: {test_auc:.4f}")

    # 分类报告
    report = classification_report(y_test.values, y_test_pred, target_names=['低房价', '高房价'])
    print(f"\n分类报告:")
    print(report)

    # ============================================
    # 4. 保存结果
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 保存结果")
    print("="*60)

    results_path = result_dir / 'training_results.txt'
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("XGBoost高级技巧 - 训练结果\n")
        f.write("="*60 + "\n\n")

        f.write(f"训练模式: {args.mode}\n\n")

        f.write("验证集性能:\n")
        f.write(f"  准确率: {val_accuracy:.4f}\n")
        f.write(f"  AUC: {val_auc:.4f}\n\n")

        f.write("测试集性能:\n")
        f.write(f"  准确率: {test_accuracy:.4f}\n")
        f.write(f"  AUC: {test_auc:.4f}\n\n")

        f.write("分类报告:\n")
        f.write(report)

    print(f"✓ 训练结果已保存: {results_path}")

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("训练总结")
    print("="*60)
    print(f"✓ 模型已保存: {model_path}")
    print(f"✓ 特征名称已保存: {features_path}")
    print(f"✓ 测试集准确率: {test_accuracy:.4f}")
    print(f"✓ 测试集AUC: {test_auc:.4f}")
    print(f"\n使用以下命令进行评估:")
    print(f"  python src/evaluate.py --model_path {model_path} --features_path {features_path}")


if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)

    main()
