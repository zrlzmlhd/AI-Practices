"""
Otto分类模型训练脚本

使用方法:
    python src/train.py --model_type xgboost_basic
    python src/train.py --model_type xgboost_tuned
    python src/train.py --model_type lightgbm
    python src/train.py --model_type catboost
    python src/train.py --model_type stacking
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import prepare_otto_data
from src.model import (
    OttoXGBoostClassifier,
    OttoLightGBMClassifier,
    OttoCatBoostClassifier,
    OttoStackingEnsemble
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Otto分类模型')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='xgboost_tuned',
                       choices=['xgboost_basic', 'xgboost_tuned', 'xgboost_advanced',
                               'lightgbm', 'catboost', 'stacking'],
                       help='模型类型')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/train.csv',
                       help='数据文件路径')
    parser.add_argument('--create_features', action='store_true',
                       help='是否创建工程特征')

    # 训练参数
    parser.add_argument('--early_stopping_rounds', type=int, default=50,
                       help='早停轮数')

    # 其他参数
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
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
    print("Otto分类 - 模型训练")
    print("="*60)
    print(f"\n配置:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # 设置随机种子
    np.random.seed(args.random_state)

    # 创建保存目录
    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / args.model_dir
    result_dir = project_dir / args.result_dir
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    # ============================================
    # 步骤1: 准备数据
    # ============================================
    print("\n" + "="*60)
    print("步骤1: 准备数据")
    print("="*60)

    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), processor = prepare_otto_data(
            data_path=args.data_path,
            create_features=args.create_features,
            random_state=args.random_state
        )
    except FileNotFoundError as e:
        print(f"\n✗ 数据文件不存在: {e}")
        print("\n请先下载数据:")
        print("  cd data")
        print("  python download_data.py")
        return

    # 保存数据处理器
    processor_path = model_dir / f'{args.model_type}_processor.pkl'
    processor.save_processor(processor_path)

    # ============================================
    # 步骤2: 创建模型
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 创建模型")
    print("="*60)

    n_classes = len(np.unique(y_train))

    if args.model_type.startswith('xgboost'):
        model_variant = args.model_type.split('_')[1]  # basic/tuned/advanced
        classifier = OttoXGBoostClassifier(n_classes, model_type=model_variant)
        print(f"创建XGBoost模型 ({model_variant})")

    elif args.model_type == 'lightgbm':
        classifier = OttoLightGBMClassifier(n_classes)
        print("创建LightGBM模型")

    elif args.model_type == 'catboost':
        classifier = OttoCatBoostClassifier(n_classes)
        print("创建CatBoost模型")

    elif args.model_type == 'stacking':
        classifier = OttoStackingEnsemble(n_classes)
        print("创建Stacking集成模型")

    # ============================================
    # 步骤3: 训练模型
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 训练模型")
    print("="*60)

    print(f"\n开始训练...")

    if args.model_type == 'stacking':
        # Stacking需要特殊的训练流程
        classifier.train(X_train, y_train, X_val, y_val)
    else:
        # 单模型训练
        if args.model_type.startswith('xgboost'):
            classifier.train(
                X_train, y_train,
                X_val, y_val,
                early_stopping_rounds=args.early_stopping_rounds,
                verbose=True
            )
        elif args.model_type == 'lightgbm':
            classifier.train(
                X_train, y_train,
                X_val, y_val,
                early_stopping_rounds=args.early_stopping_rounds
            )
        elif args.model_type == 'catboost':
            classifier.train(
                X_train, y_train,
                X_val, y_val,
                early_stopping_rounds=args.early_stopping_rounds
            )

    # ============================================
    # 步骤4: 评估模型
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 评估模型")
    print("="*60)

    # 训练集评估
    train_metrics = classifier.evaluate(X_train, y_train)
    print(f"\n训练集性能:")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Log Loss: {train_metrics['log_loss']:.4f}")

    # 验证集评估
    val_metrics = classifier.evaluate(X_val, y_val)
    print(f"\n验证集性能:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Log Loss: {val_metrics['log_loss']:.4f}")

    # 测试集评估
    test_metrics = classifier.evaluate(X_test, y_test)
    print(f"\n测试集性能:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Log Loss: {test_metrics['log_loss']:.4f}")

    # ============================================
    # 步骤5: 特征重要性（仅XGBoost）
    # ============================================
    if args.model_type.startswith('xgboost'):
        print("\n" + "="*60)
        print("步骤5: 特征重要性分析")
        print("="*60)

        importance_df = classifier.get_feature_importance(top_n=20)
        print(f"\nTop 20 重要特征:")
        print(importance_df.to_string(index=False))

        # 保存特征重要性
        importance_path = result_dir / f'{args.model_type}_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        print(f"\n✓ 特征重要性已保存: {importance_path}")

    # ============================================
    # 步骤6: 保存结果
    # ============================================
    print("\n" + "="*60)
    print("步骤6: 保存结果")
    print("="*60)

    # 保存模型
    model_path = model_dir / f'{args.model_type}_model.pkl'
    classifier.save_model(model_path)

    # 保存评估结果
    results = {
        'model_type': args.model_type,
        'n_classes': n_classes,
        'n_features': X_train.shape[1],
        'train_accuracy': train_metrics['accuracy'],
        'train_log_loss': train_metrics['log_loss'],
        'val_accuracy': val_metrics['accuracy'],
        'val_log_loss': val_metrics['log_loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_log_loss': test_metrics['log_loss'],
    }

    results_path = result_dir / f'{args.model_type}_results.txt'
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ 评估结果已保存: {results_path}")

    # ============================================
    # 步骤7: 示例预测
    # ============================================
    print("\n" + "="*60)
    print("步骤7: 示例预测")
    print("="*60)

    # 随机选择几个测试样本
    num_examples = 5
    indices = np.random.choice(len(X_test), num_examples, replace=False)

    print(f"\n随机选择 {num_examples} 个测试样本:")
    for i, idx in enumerate(indices):
        x = X_test[idx:idx+1]
        y_true = y_test[idx]
        y_pred = classifier.predict(x)[0]
        y_proba = classifier.predict_proba(x)[0]

        # 转换回原始标签
        true_label = processor.inverse_transform_labels([y_true])[0]
        pred_label = processor.inverse_transform_labels([y_pred])[0]

        print(f"\n样本 {i+1}:")
        print(f"  真实标签: {true_label}")
        print(f"  预测标签: {pred_label}")
        print(f"  预测概率: {y_proba}")
        print(f"  最高概率: {y_proba.max():.4f}")
        print(f"  预测{'正确' if y_pred == y_true else '错误'}")

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"\n模型保存路径: {model_path}")
    print(f"数据处理器保存路径: {processor_path}")
    print(f"\n测试集性能:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Log Loss: {test_metrics['log_loss']:.4f}")

    # 给出建议
    print(f"\n下一步:")
    print(f"  1. 查看评估结果: {results_path}")
    print(f"  2. 评估模型: python src/evaluate.py --model_path {model_path} --processor_path {processor_path}")
    print(f"  3. 尝试其他模型:")
    print(f"     python src/train.py --model_type lightgbm")
    print(f"     python src/train.py --model_type catboost")
    print(f"     python src/train.py --model_type stacking")

    # 性能分析
    print(f"\n性能分析:")
    if test_metrics['log_loss'] < 0.5:
        print(f"  ✓✓ Log Loss < 0.5，性能优秀！")
    elif test_metrics['log_loss'] < 0.6:
        print(f"  ✓ Log Loss < 0.6，性能良好")
        print(f"     可以尝试:")
        print(f"     - 使用Stacking集成")
        print(f"     - 创建更多特征 (--create_features)")
        print(f"     - 调整超参数")
    else:
        print(f"  ⚠ Log Loss较高，建议:")
        print(f"    - 检查数据质量")
        print(f"    - 创建工程特征")
        print(f"    - 使用更复杂的模型")
        print(f"    - 尝试模型集成")

    # 过拟合检查
    if train_metrics['log_loss'] < val_metrics['log_loss'] * 0.7:
        print(f"\n  ⚠ 检测到过拟合，建议:")
        print(f"    - 增加正则化")
        print(f"    - 减少模型复杂度")
        print(f"    - 使用更多训练数据")


if __name__ == '__main__':
    main()
