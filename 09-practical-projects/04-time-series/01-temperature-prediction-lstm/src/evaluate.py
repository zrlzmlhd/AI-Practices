"""
模型评估脚本

使用方法:
    python src/evaluate.py --model_path models/stacked_model.h5 --scaler_path models/stacked_scaler.pkl
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import TimeSeriesDataProcessor, prepare_temperature_data


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估LSTM温度预测模型')

    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--scaler_path', type=str, required=True,
                       help='数据处理器文件路径')
    parser.add_argument('--data_path', type=str,
                       default='data/jena_climate_2009_2016.csv',
                       help='数据文件路径')
    parser.add_argument('--lookback', type=int, default=168,
                       help='回看窗口大小（小时）')
    parser.add_argument('--forecast_horizon', type=int, default=24,
                       help='预测范围（小时）')
    parser.add_argument('--sampling_rate', type=int, default=6,
                       help='采样率')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def plot_predictions(y_true, y_pred, num_samples=5, save_path=None):
    """
    绘制预测结果对比图

    Args:
        y_true: 真实值
        y_pred: 预测值
        num_samples: 显示样本数
        save_path: 保存路径
    """
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]
        hours = np.arange(len(y_true[i]))

        ax.plot(hours, y_true[i], 'b-', label='True', linewidth=2)
        ax.plot(hours, y_pred[i], 'r--', label='Predicted', linewidth=2)
        ax.fill_between(hours, y_true[i], y_pred[i], alpha=0.3)

        ax.set_xlabel('Hours', fontsize=10)
        ax.set_ylabel('Temperature (°C)', fontsize=10)
        ax.set_title(f'Sample {i+1}: Temperature Prediction', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 预测对比图已保存: {save_path}")


def plot_error_distribution(y_true, y_pred, save_path=None):
    """
    绘制误差分布图

    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
    """
    errors = (y_pred - y_true).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 误差直方图
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error (°C)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14)
    axes[0].grid(alpha=0.3)

    # 误差统计
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    axes[0].text(0.02, 0.98,
                f'Mean: {mean_error:.2f}°C\nStd: {std_error:.2f}°C',
                transform=axes[0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 误差箱线图
    axes[1].boxplot(errors, vert=True)
    axes[1].set_ylabel('Prediction Error (°C)', fontsize=12)
    axes[1].set_title('Error Box Plot', fontsize=14)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 误差分布图已保存: {save_path}")


def plot_scatter(y_true, y_pred, save_path=None):
    """
    绘制真实值vs预测值散点图

    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    plt.figure(figsize=(8, 8))

    # 散点图
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10)

    # 理想线（y=x）
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')

    plt.xlabel('True Temperature (°C)', fontsize=12)
    plt.ylabel('Predicted Temperature (°C)', fontsize=12)
    plt.title('True vs Predicted Temperature', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')

    # 计算R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true_flat, y_pred_flat)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 散点图已保存: {save_path}")


def plot_hourly_performance(y_true, y_pred, save_path=None):
    """
    绘制不同预测时间的性能

    Args:
        y_true: 真实值 (samples, forecast_horizon)
        y_pred: 预测值 (samples, forecast_horizon)
        save_path: 保存路径
    """
    forecast_horizon = y_true.shape[1]

    # 计算每个小时的MAE
    hourly_mae = []
    for hour in range(forecast_horizon):
        mae = np.mean(np.abs(y_true[:, hour] - y_pred[:, hour]))
        hourly_mae.append(mae)

    plt.figure(figsize=(12, 6))

    hours = np.arange(1, forecast_horizon + 1)
    plt.plot(hours, hourly_mae, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Forecast Hour', fontsize=12)
    plt.ylabel('MAE (°C)', fontsize=12)
    plt.title('Prediction Performance by Forecast Hour', fontsize=14)
    plt.grid(alpha=0.3)

    # 标注最好和最差的小时
    best_hour = np.argmin(hourly_mae) + 1
    worst_hour = np.argmax(hourly_mae) + 1
    plt.axvline(best_hour, color='green', linestyle='--', alpha=0.5, label=f'Best: Hour {best_hour}')
    plt.axvline(worst_hour, color='red', linestyle='--', alpha=0.5, label=f'Worst: Hour {worst_hour}')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 逐小时性能图已保存: {save_path}")

    return hourly_mae


def analyze_performance(y_true, y_pred):
    """
    详细性能分析

    Args:
        y_true: 真实值
        y_pred: 预测值
    """
    print("\n" + "="*60)
    print("详细性能分析")
    print("="*60)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    # 基本统计
    print(f"\n误差统计:")
    print(f"  平均误差: {np.mean(errors):.2f}°C")
    print(f"  误差标准差: {np.std(errors):.2f}°C")
    print(f"  最大正误差: {np.max(errors):.2f}°C")
    print(f"  最大负误差: {np.min(errors):.2f}°C")

    # 百分位数
    print(f"\n误差百分位数:")
    for p in [50, 75, 90, 95, 99]:
        percentile = np.percentile(abs_errors, p)
        print(f"  {p}%: {percentile:.2f}°C")

    # 误差范围统计
    print(f"\n误差范围分布:")
    ranges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, float('inf'))]
    for low, high in ranges:
        if high == float('inf'):
            count = np.sum(abs_errors >= low)
            percentage = count / len(abs_errors.flatten()) * 100
            print(f"  ≥{low}°C: {count} ({percentage:.1f}%)")
        else:
            count = np.sum((abs_errors >= low) & (abs_errors < high))
            percentage = count / len(abs_errors.flatten()) * 100
            print(f"  {low}-{high}°C: {count} ({percentage:.1f}%)")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("="*60)
    print("LSTM温度预测 - 模型评估")
    print("="*60)
    print(f"\n模型路径: {args.model_path}")
    print(f"数据处理器路径: {args.scaler_path}")

    # 创建结果目录
    project_dir = Path(__file__).parent.parent
    result_dir = project_dir / args.result_dir
    result_dir.mkdir(exist_ok=True)

    # ============================================
    # 步骤1: 加载数据处理器
    # ============================================
    print("\n" + "="*60)
    print("步骤1: 加载数据处理器")
    print("="*60)

    processor = TimeSeriesDataProcessor(
        data_path=args.data_path,
        sampling_rate=args.sampling_rate
    )
    processor.load_scaler(args.scaler_path)

    # ============================================
    # 步骤2: 准备数据
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 准备数据")
    print("="*60)

    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = prepare_temperature_data(
            data_path=args.data_path,
            lookback=args.lookback,
            forecast_horizon=args.forecast_horizon,
            sampling_rate=args.sampling_rate
        )
    except FileNotFoundError as e:
        print(f"\n✗ 数据文件不存在: {e}")
        return

    # ============================================
    # 步骤3: 加载模型
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 加载模型")
    print("="*60)

    model = keras.models.load_model(args.model_path)
    print(f"✓ 模型已加载")

    # ============================================
    # 步骤4: 预测
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 模型预测")
    print("="*60)

    print("\n预测测试集...")
    y_pred_test = model.predict(X_test, verbose=1)

    # 反归一化
    y_test_original = processor.inverse_transform_target(y_test)
    y_pred_original = processor.inverse_transform_target(y_pred_test)

    # ============================================
    # 步骤5: 计算评估指标
    # ============================================
    print("\n" + "="*60)
    print("步骤5: 计算评估指标")
    print("="*60)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true_flat = y_test_original.flatten()
    y_pred_flat = y_pred_original.flatten()

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_flat, y_pred_flat)

    # MAPE
    mask = y_true_flat != 0
    mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100

    print(f"\n测试集性能:")
    print(f"  MAE:  {mae:.2f}°C")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    # ============================================
    # 步骤6: 绘制预测对比图
    # ============================================
    print("\n" + "="*60)
    print("步骤6: 绘制预测对比图")
    print("="*60)

    model_name = Path(args.model_path).stem

    # 选择一些有代表性的样本
    indices = np.random.choice(len(y_test_original), 5, replace=False)
    pred_path = result_dir / f'{model_name}_predictions.png'
    plot_predictions(
        y_test_original[indices],
        y_pred_original[indices],
        num_samples=5,
        save_path=pred_path
    )

    # ============================================
    # 步骤7: 绘制误差分布
    # ============================================
    print("\n" + "="*60)
    print("步骤7: 绘制误差分布")
    print("="*60)

    error_path = result_dir / f'{model_name}_error_distribution.png'
    plot_error_distribution(y_test_original, y_pred_original, error_path)

    # ============================================
    # 步骤8: 绘制散点图
    # ============================================
    print("\n" + "="*60)
    print("步骤8: 绘制散点图")
    print("="*60)

    scatter_path = result_dir / f'{model_name}_scatter.png'
    plot_scatter(y_test_original, y_pred_original, scatter_path)

    # ============================================
    # 步骤9: 逐小时性能分析
    # ============================================
    print("\n" + "="*60)
    print("步骤9: 逐小时性能分析")
    print("="*60)

    hourly_path = result_dir / f'{model_name}_hourly_performance.png'
    hourly_mae = plot_hourly_performance(y_test_original, y_pred_original, hourly_path)

    print(f"\n逐小时MAE:")
    for hour, mae_val in enumerate(hourly_mae[:12], 1):  # 显示前12小时
        print(f"  第{hour:2d}小时: {mae_val:.2f}°C")

    # ============================================
    # 步骤10: 详细性能分析
    # ============================================
    analyze_performance(y_test_original, y_pred_original)

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)

    print(f"\n生成的文件:")
    print(f"  1. 预测对比图: {pred_path}")
    print(f"  2. 误差分布图: {error_path}")
    print(f"  3. 散点图: {scatter_path}")
    print(f"  4. 逐小时性能图: {hourly_path}")

    print(f"\n模型性能总结:")
    print(f"  MAE:  {mae:.2f}°C")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  R²:   {r2:.4f}")

    # 性能评价
    if mae < 2.0 and rmse < 3.0:
        print(f"\n  ✓✓ 模型性能优秀！")
    elif mae < 3.0 and rmse < 4.0:
        print(f"\n  ✓ 模型性能良好")
    else:
        print(f"\n  ⚠ 模型性能有待提升")


if __name__ == '__main__':
    main()
