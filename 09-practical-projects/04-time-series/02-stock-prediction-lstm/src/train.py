"""
LSTM股票预测模型训练脚本

使用方法:
    python src/train.py --model_type lstm_basic --epochs 100
    python src/train.py --model_type lstm_attention --epochs 150
    python src/train.py --model_type multitask --epochs 150
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import prepare_stock_data
from src.model import StockLSTMPredictor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练LSTM股票预测模型')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='lstm_attention',
                       choices=['lstm_basic', 'lstm_attention', 'multitask'],
                       help='模型类型')

    # 数据参数
    parser.add_argument('--data_path', type=str,
                       default='data/stock_data.csv',
                       help='数据文件路径')
    parser.add_argument('--lookback', type=int, default=60,
                       help='回看窗口大小（天）')
    parser.add_argument('--forecast_horizon', type=int, default=1,
                       help='预测范围（天）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='早停耐心值')

    # 其他参数
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def create_callbacks(model_path, patience=20):
    """创建训练回调函数"""
    callbacks = []

    # ModelCheckpoint
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)

    # EarlyStopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    callbacks.append(early_stopping)

    # ReduceLROnPlateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        mode='min',
        verbose=1
    )
    callbacks.append(reduce_lr)

    return callbacks


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("="*60)
    print("LSTM股票预测 - 模型训练")
    print("="*60)
    print(f"\n配置:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # 设置随机种子
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

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
        train_data, val_data, test_data, processor = prepare_stock_data(
            data_path=args.data_path,
            lookback=args.lookback,
            forecast_horizon=args.forecast_horizon
        )
    except FileNotFoundError as e:
        print(f"\n✗ 数据文件不存在: {e}")
        print("\n请先下载数据:")
        print("  cd data")
        print("  python download_data.py")
        return

    X_train, y_price_train, y_trend_train = train_data
    X_val, y_price_val, y_trend_val = val_data
    X_test, y_price_test, y_trend_test = test_data

    # 保存数据处理器
    processor_path = model_dir / f'{args.model_type}_processor.pkl'
    processor.save_processor(processor_path)

    # ============================================
    # 步骤2: 创建模型
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 创建模型")
    print("="*60)

    input_shape = (X_train.shape[1], X_train.shape[2])

    predictor = StockLSTMPredictor(
        input_shape=input_shape,
        model_type=args.model_type
    )

    # 打印模型摘要
    print(f"\n模型结构:")
    predictor.summary()

    # 计算参数量
    total_params = predictor.model.count_params()
    print(f"\n总参数量: {total_params:,}")

    # ============================================
    # 步骤3: 训练模型
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 训练模型")
    print("="*60)

    # 创建回调函数
    model_path = model_dir / f'{args.model_type}_model.h5'
    callbacks = create_callbacks(
        model_path=model_path,
        patience=args.early_stopping_patience
    )

    # 准备训练数据
    if args.model_type == 'multitask':
        y_train = {'price': y_price_train, 'trend': y_trend_train}
        y_val = {'price': y_price_val, 'trend': y_trend_val}
    else:
        y_train = y_price_train
        y_val = y_price_val

    # 训练
    print(f"\n开始训练...")
    history = predictor.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        callbacks=callbacks,
        verbose=1
    )

    # ============================================
    # 步骤4: 评估模型
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 评估模型")
    print("="*60)

    # 准备测试数据
    if args.model_type == 'multitask':
        y_test = {'price': y_price_test, 'trend': y_trend_test}
    else:
        y_test = y_price_test

    # 训练集评估
    if args.model_type == 'multitask':
        y_train_eval = {'price': y_price_train, 'trend': y_trend_train}
    else:
        y_train_eval = y_price_train

    train_metrics = predictor.evaluate(X_train, y_train_eval)
    print(f"\n训练集性能:")
    for name, value in train_metrics.items():
        print(f"  {name}: {value:.4f}")

    # 验证集评估
    val_metrics = predictor.evaluate(X_val, y_val)
    print(f"\n验证集性能:")
    for name, value in val_metrics.items():
        print(f"  {name}: {value:.4f}")

    # 测试集评估
    test_metrics = predictor.evaluate(X_test, y_test)
    print(f"\n测试集性能:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    # ============================================
    # 步骤5: 原始尺度评估
    # ============================================
    print("\n" + "="*60)
    print("步骤5: 原始尺度评估")
    print("="*60)

    # 预测
    if args.model_type == 'multitask':
        y_pred_price, y_pred_trend = predictor.predict(X_test)
    else:
        y_pred_price = predictor.predict(X_test)

    # 反归一化
    y_test_original = processor.inverse_transform_price(y_price_test)
    y_pred_original = processor.inverse_transform_price(y_pred_price)

    # 计算原始尺度的指标
    original_metrics = predictor.calculate_metrics(y_test_original, y_pred_original)
    print(f"\n测试集性能（原始尺度）:")
    for name, value in original_metrics.items():
        print(f"  {name}: {value:.4f}")

    # 趋势准确率（多任务）
    if args.model_type == 'multitask':
        from sklearn.metrics import accuracy_score
        trend_accuracy = accuracy_score(y_trend_test, y_pred_trend)
        print(f"  趋势准确率: {trend_accuracy:.4f}")

    # ============================================
    # 步骤6: 保存结果
    # ============================================
    print("\n" + "="*60)
    print("步骤6: 保存结果")
    print("="*60)

    # 保存训练历史
    history_path = result_dir / f'{args.model_type}_history.npz'
    np.savez(
        history_path,
        **history.history
    )
    print(f"✓ 训练历史已保存: {history_path}")

    # 保存评估结果
    results = {
        'model_type': args.model_type,
        'total_params': total_params,
        'lookback': args.lookback,
        'forecast_horizon': args.forecast_horizon,
        'test_mae_original': original_metrics['mae'],
        'test_rmse_original': original_metrics['rmse'],
    }

    if 'mape' in original_metrics:
        results['test_mape'] = original_metrics['mape']
    if 'direction_accuracy' in original_metrics:
        results['test_direction_accuracy'] = original_metrics['direction_accuracy']
    if args.model_type == 'multitask':
        results['test_trend_accuracy'] = trend_accuracy

    results_path = result_dir / f'{args.model_type}_results.txt'
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ 评估结果已保存: {results_path}")

    # 保存预测结果
    predictions_path = result_dir / f'{args.model_type}_predictions.npz'
    save_dict = {
        'y_true': y_test_original[:100],
        'y_pred': y_pred_original[:100]
    }
    if args.model_type == 'multitask':
        save_dict['y_trend_true'] = y_trend_test[:100]
        save_dict['y_trend_pred'] = y_pred_trend[:100]

    np.savez(predictions_path, **save_dict)
    print(f"✓ 预测结果已保存: {predictions_path}")

    # ============================================
    # 步骤7: 示例预测
    # ============================================
    print("\n" + "="*60)
    print("步骤7: 示例预测")
    print("="*60)

    # 随机选择一个测试样本
    idx = np.random.randint(0, len(X_test))
    sample_pred = y_pred_original[idx]
    sample_true = y_test_original[idx]

    print(f"\n样本 {idx} 的预测结果:")
    print(f"  真实价格: ${sample_true:.2f}")
    print(f"  预测价格: ${sample_pred:.2f}")
    print(f"  误差: ${abs(sample_pred - sample_true):.2f}")
    print(f"  误差率: {abs(sample_pred - sample_true) / sample_true * 100:.2f}%")

    if args.model_type == 'multitask':
        print(f"  真实趋势: {'上涨' if y_trend_test[idx] == 1 else '下跌'}")
        print(f"  预测趋势: {'上涨' if y_pred_trend[idx] == 1 else '下跌'}")

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"\n模型保存路径: {model_path}")
    print(f"数据处理器保存路径: {processor_path}")
    print(f"\n测试集性能（原始尺度）:")
    print(f"  MAE:  ${original_metrics['mae']:.2f}")
    print(f"  RMSE: ${original_metrics['rmse']:.2f}")
    if 'mape' in original_metrics:
        print(f"  MAPE: {original_metrics['mape']:.2f}%")
    if 'direction_accuracy' in original_metrics:
        print(f"  方向准确率: {original_metrics['direction_accuracy']:.2%}")

    # 给出建议
    print(f"\n下一步:")
    print(f"  1. 查看训练历史: {history_path}")
    print(f"  2. 评估模型: python src/evaluate.py --model_path {model_path} --processor_path {processor_path}")
    print(f"  3. 尝试其他模型类型:")
    print(f"     python src/train.py --model_type lstm_basic")
    print(f"     python src/train.py --model_type multitask")

    # 性能分析
    print(f"\n性能分析:")
    if 'mape' in original_metrics and original_metrics['mape'] < 5:
        print(f"  ✓✓ MAPE < 5%，性能优秀！")
    elif 'mape' in original_metrics and original_metrics['mape'] < 10:
        print(f"  ✓ MAPE < 10%，性能良好")
    else:
        print(f"  ⚠ 性能有待提升，建议:")
        print(f"    - 增加lookback窗口")
        print(f"    - 使用注意力机制")
        print(f"    - 尝试多任务学习")
        print(f"    - 增加训练数据")

    # 风险提示
    print(f"\n⚠️  风险提示:")
    print(f"  本模型仅供学习使用，不构成投资建议。")
    print(f"  股票投资有风险，入市需谨慎。")


if __name__ == '__main__':
    main()
