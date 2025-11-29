"""
模型训练脚本
"""

import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed
from data import prepare_data
from model import TitanicPredictor, compare_models


def train_single_model(model_type='random_forest', tune=False, use_ensemble=False):
    """
    训练单个模型

    Args:
        model_type: 模型类型
        tune: 是否进行超参数调优
        use_ensemble: 是否使用集成模型
    """
    print("=" * 60)
    print("Titanic生存预测 - 模型训练")
    print("=" * 60)

    # 准备数据
    X_train, X_val, y_train, y_val, preprocessor = prepare_data()

    # 创建预测器
    predictor = TitanicPredictor(model_type=model_type, random_state=42)

    # 训练模型
    if tune:
        print("\n进行超参数调优...")
        predictor.tune_hyperparameters(X_train, y_train)
    else:
        predictor.train(X_train, y_train, use_ensemble=use_ensemble)

    # 评估模型
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    train_metrics = predictor.evaluate(X_train, y_train)
    val_metrics = predictor.evaluate(X_val, y_val)

    print("\n训练集性能:")
    for metric, value in train_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    print("\n验证集性能:")
    for metric, value in val_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    # 特征重要性
    importance = predictor.get_feature_importance(top_n=10)
    if importance is not None:
        print("\n特征重要性 (Top 10):")
        print(importance.to_string(index=False))

    # 保存模型
    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    if use_ensemble:
        model_path = model_dir / 'ensemble_model.pkl'
    else:
        model_path = model_dir / f'{model_type}_model.pkl'

    predictor.save_model(model_path)

    # 保存最佳模型
    best_model_path = model_dir / 'best_model.pkl'
    predictor.save_model(best_model_path)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"模型已保存到: {model_path}")

    return predictor, val_metrics


def train_and_compare_models():
    """训练并比较多个模型"""
    print("=" * 60)
    print("Titanic生存预测 - 模型对比")
    print("=" * 60)

    # 准备数据
    X_train, X_val, y_train, y_val, preprocessor = prepare_data()

    # 比较模型
    results_df = compare_models(X_train, y_train, X_val, y_val)

    # 保存对比结果
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'model_comparison.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ 对比结果已保存: {results_path}")

    # 训练最佳模型
    best_model_type = results_df.iloc[0]['Model']
    print(f"\n最佳模型: {best_model_type}")
    print("训练最佳模型...")

    predictor = TitanicPredictor(model_type=best_model_type, random_state=42)
    predictor.train(X_train, y_train)

    # 保存最佳模型
    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / 'best_model.pkl'
    predictor.save_model(best_model_path)

    return results_df


def train_ensemble_model():
    """训练集成模型"""
    print("=" * 60)
    print("Titanic生存预测 - 集成模型训练")
    print("=" * 60)

    # 准备数据
    X_train, X_val, y_train, y_val, preprocessor = prepare_data()

    # 创建并训练集成模型
    predictor = TitanicPredictor(model_type='random_forest', random_state=42)
    predictor.train(X_train, y_train, use_ensemble=True)

    # 评估
    print("\n" + "=" * 60)
    print("集成模型评估")
    print("=" * 60)

    train_metrics = predictor.evaluate(X_train, y_train)
    val_metrics = predictor.evaluate(X_val, y_val)

    print("\n训练集性能:")
    for metric, value in train_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    print("\n验证集性能:")
    for metric, value in val_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    # 保存模型
    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / 'ensemble_model.pkl'
    predictor.save_model(model_path)

    best_model_path = model_dir / 'best_model.pkl'
    predictor.save_model(best_model_path)

    print("\n" + "=" * 60)
    print("集成模型训练完成！")
    print("=" * 60)

    return predictor, val_metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Titanic生存预测模型训练')

    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['logistic', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'svm', 'gradient_boosting'],
        help='模型类型'
    )

    parser.add_argument(
        '--tune',
        action='store_true',
        help='是否进行超参数调优'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='比较多个模型'
    )

    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='训练集成模型'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 根据参数执行不同的训练流程
    if args.compare:
        train_and_compare_models()
    elif args.ensemble:
        train_ensemble_model()
    else:
        train_single_model(
            model_type=args.model,
            tune=args.tune,
            use_ensemble=False
        )


if __name__ == '__main__':
    main()
