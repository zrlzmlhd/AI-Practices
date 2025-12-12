"""
LSTM温度预测模型训练脚本

使用方法:
    python src/train.py --model_type simple --epochs 50
    python src/train.py --model_type stacked --epochs 100
    python src/train.py --model_type gru --epochs 100
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

from src.data import prepare_temperature_data
from src.model import TemperatureLSTMPredictor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练LSTM温度预测模型')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='stacked',
                       choices=['simple', 'stacked', 'gru'],
                       help='模型类型')

    # 数据参数
    parser.add_argument('--data_path', type=str,
                       default='data/jena_climate_2009_2016.csv',
                       help='数据文件路径')
    parser.add_argument('--lookback', type=int, default=168,
                       help='回看窗口大小（小时）')
    parser.add_argument('--forecast_horizon', type=int, default=24,
                       help='预测范围（小时）')
    parser.add_argument('--sampling_rate', type=int, default=6,
                       help='采样率（每隔多少条取一条）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='早停耐心值')

    # 其他参数
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def create_callbacks(model_path, patience=10):
    """
    创建训练回调函数

    Args:
        model_path: 模型保存路径
        patience: 早停耐心值

    Returns:
        回调函数列表
    """
    callbacks = []

    # ============================================
    # ModelCheckpoint: 保存最佳模型
    # ============================================
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)

    # ============================================
    # EarlyStopping: 早停
    # ============================================
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    callbacks.append(early_stopping)

    # ============================================
    # ReduceLROnPlateau: 学习率衰减
    # ============================================
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
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
    print("LSTM温度预测 - 模型训练")
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
        (X_train, y_train), (X_val, y_val), (X_test, y_test), processor = prepare_temperature_data(
            data_path=args.data_path,
            lookback=args.lookback,
            forecast_horizon=args.forecast_horizon,
            sampling_rate=args.sampling_rate
        )
    except FileNotFoundError as e:
        print(f"\n✗ 数据文件不存在: {e}")
        print("\n请先下载数据:")
        print("  cd data")
        print("  python download_data.py")
        return

    # 保存数据处理器
    scaler_path = model_dir / f'{args.model_type}_scaler.pkl'
    processor.save_scaler(scaler_path)

    # ============================================
    # 步骤2: 创建模型
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 创建模型")
    print("="*60)

    input_shape = (X_train.shape[1], X_train.shape[2])

    predictor = TemperatureLSTMPredictor(
        input_shape=input_shape,
        forecast_horizon=args.forecast_horizon,
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

    # 训练集评估
    train_metrics = predictor.evaluate(X_train, y_train)
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
    # 步骤5: 详细评估（原始尺度）
    # ============================================
    print("\n" + "="*60)
    print("步骤5: 原始尺度评估")
    print("="*60)

    # 预测
    y_pred_test = predictor.predict(X_test)

    # 反归一化
    y_test_original = processor.inverse_transform_target(y_test)
    y_pred_original = processor.inverse_transform_target(y_pred_test)

    # 计算原始尺度的指标
    original_metrics = predictor.calculate_metrics(y_test_original, y_pred_original)
    print(f"\n测试集性能（原始尺度）:")
    for name, value in original_metrics.items():
        print(f"  {name}: {value:.4f}")

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
        'train_loss': train_metrics['loss'],
        'train_mae': train_metrics['mae'],
        'train_rmse': train_metrics['rmse'],
        'val_loss': val_metrics['loss'],
        'val_mae': val_metrics['mae'],
        'val_rmse': val_metrics['rmse'],
        'test_loss': test_metrics['loss'],
        'test_mae': test_metrics['mae'],
        'test_rmse': test_metrics['rmse'],
        'test_mae_original': original_metrics['mae'],
        'test_rmse_original': original_metrics['rmse'],
    }

    if 'mape' in original_metrics:
        results['test_mape'] = original_metrics['mape']

    results_path = result_dir / f'{args.model_type}_results.txt'
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ 评估结果已保存: {results_path}")

    # 保存预测结果（用于可视化）
    predictions_path = result_dir / f'{args.model_type}_predictions.npz'
    np.savez(
        predictions_path,
        y_true=y_test_original[:100],  # 保存前100个样本
        y_pred=y_pred_original[:100]
    )
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

    print(f"\n样本 {idx} 的预测结果（未来24小时温度）:")
    print(f"\n时间    真实温度(°C)  预测温度(°C)  误差(°C)")
    print("-" * 50)
    for hour in range(min(24, len(sample_true))):
        error = sample_pred[hour] - sample_true[hour]
        print(f"{hour+1:2d}小时   {sample_true[hour]:7.2f}      {sample_pred[hour]:7.2f}      {error:+6.2f}")

    # 计算平均误差
    mae_sample = np.mean(np.abs(sample_pred - sample_true))
    print(f"\n该样本的平均绝对误差: {mae_sample:.2f}°C")

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"\n模型保存路径: {model_path}")
    print(f"数据处理器保存路径: {scaler_path}")
    print(f"\n测试集性能（原始尺度）:")
    print(f"  MAE:  {original_metrics['mae']:.2f}°C")
    print(f"  RMSE: {original_metrics['rmse']:.2f}°C")
    if 'mape' in original_metrics:
        print(f"  MAPE: {original_metrics['mape']:.2f}%")

    # 给出建议
    print(f"\n下一步:")
    print(f"  1. 查看训练历史: {history_path}")
    print(f"  2. 评估模型: python src/evaluate.py --model_path {model_path} --scaler_path {scaler_path}")
    print(f"  3. 尝试其他模型类型:")
    print(f"     python src/train.py --model_type simple")
    print(f"     python src/train.py --model_type gru")

    # 性能分析
    print(f"\n性能分析:")
    mae_threshold = 2.0  # 2°C
    rmse_threshold = 3.0  # 3°C

    if original_metrics['mae'] < mae_threshold and original_metrics['rmse'] < rmse_threshold:
        print(f"  ✓✓ 模型性能优秀！")
        print(f"     MAE < {mae_threshold}°C, RMSE < {rmse_threshold}°C")
    elif original_metrics['mae'] < mae_threshold * 1.5:
        print(f"  ✓ 模型性能良好")
        print(f"     可以尝试:")
        print(f"     - 增加训练轮数")
        print(f"     - 使用更复杂的模型（stacked）")
        print(f"     - 调整学习率")
    else:
        print(f"  ⚠ 模型性能有待提升，建议:")
        print(f"    - 检查数据质量")
        print(f"    - 增加lookback窗口")
        print(f"    - 使用更多特征")
        print(f"    - 增加模型复杂度")

    # 过拟合检查
    if train_metrics['mae'] < val_metrics['mae'] * 0.7:
        print(f"\n  ⚠ 检测到过拟合，建议:")
        print(f"    - 增加Dropout")
        print(f"    - 减少模型复杂度")
        print(f"    - 增加训练数据")
        print(f"    - 使用正则化")


if __name__ == '__main__':
    main()
