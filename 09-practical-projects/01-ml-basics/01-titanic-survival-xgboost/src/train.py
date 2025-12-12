"""
XGBoost Titanic模型训练脚本

使用方法:
    python src/train.py --model_type basic --n_estimators 100
    python src/train.py --model_type tuned --n_estimators 200
    python src/train.py --model_type advanced --n_estimators 500
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from data import load_titanic_data
from model import TitanicXGBoostClassifier
from utils.visualization import plot_training_history


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练XGBoost Titanic模型')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='tuned',
                       choices=['basic', 'tuned', 'advanced'],
                       help='模型类型')

    # 训练参数
    parser.add_argument('--early_stopping_rounds', type=int, default=10,
                       help='早停轮数')

    # 数据参数
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')

    # 保存路径
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("="*60)
    print("XGBoost Titanic生存预测 - 模型训练")
    print("="*60)
    print(f"\n配置:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # 创建保存目录
    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / args.model_dir
    result_dir = project_dir / args.result_dir
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    # 加载数据
    print("\n" + "="*60)
    print("步骤1: 加载数据")
    print("="*60)

    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = load_titanic_data(
            test_size=args.test_size,
            random_state=args.random_state
        )
    except FileNotFoundError:
        print("\n✗ 数据文件不存在！")
        print("\n请先下载数据:")
        print("  cd data")
        print("  python download_data.py")
        return

    # 创建模型
    print("\n" + "="*60)
    print("步骤2: 创建模型")
    print("="*60)

    classifier = TitanicXGBoostClassifier(
        model_type=args.model_type,
        random_state=args.random_state
    )

    # 训练模型
    print("\n" + "="*60)
    print("步骤3: 训练模型")
    print("="*60)

    classifier.train(
        X_train, y_train,
        X_val, y_val,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=True
    )

    # 评估模型
    print("\n" + "="*60)
    print("步骤4: 评估模型")
    print("="*60)

    # 训练集评估
    train_metrics = classifier.evaluate(X_train, y_train)
    print(f"\n训练集性能:")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  AUC: {train_metrics['auc']:.4f}")

    # 验证集评估
    val_metrics = classifier.evaluate(X_val, y_val)
    print(f"\n验证集性能:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  AUC: {val_metrics['auc']:.4f}")

    # 测试集评估
    test_metrics = classifier.evaluate(X_test, y_test)
    print(f"\n测试集性能:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")

    # 特征重要性
    print("\n" + "="*60)
    print("步骤5: 特征重要性分析")
    print("="*60)

    importance_df = classifier.get_feature_importance(top_n=10)
    print(f"\nTop 10 重要特征:")
    print(importance_df.to_string(index=False))

    # 保存特征重要性
    importance_path = result_dir / f'{args.model_type}_feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"\n✓ 特征重要性已保存: {importance_path}")

    # 保存模型
    print("\n" + "="*60)
    print("步骤6: 保存模型")
    print("="*60)

    model_path = model_dir / f'{args.model_type}_model.pkl'
    classifier.save_model(model_path)

    # 保存评估结果
    results = {
        'model_type': args.model_type,
        'train_accuracy': train_metrics['accuracy'],
        'train_auc': train_metrics['auc'],
        'val_accuracy': val_metrics['accuracy'],
        'val_auc': val_metrics['auc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_auc': test_metrics['auc'],
    }

    results_path = result_dir / f'{args.model_type}_results.txt'
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ 评估结果已保存: {results_path}")

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"\n模型保存路径: {model_path}")
    print(f"测试集准确率: {test_metrics['accuracy']:.4f}")
    print(f"测试集AUC: {test_metrics['auc']:.4f}")

    # 给出建议
    print(f"\n下一步:")
    print(f"  1. 查看特征重要性: {importance_path}")
    print(f"  2. 评估模型: python src/evaluate.py --model_path {model_path}")
    print(f"  3. 尝试其他模型类型: python src/train.py --model_type advanced")


if __name__ == '__main__':
    main()
