"""
模型评估脚本

使用方法:
    python src/evaluate.py --model_path models/lstm_attention_model.h5 --processor_path models/lstm_attention_processor.pkl
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

from src.data import prepare_stock_data
from src.model import AttentionLayer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估LSTM股票预测模型')

    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--processor_path', type=str, required=True,
                       help='数据处理器文件路径')
    parser.add_argument('--data_path', type=str,
                       default='data/stock_data.csv',
                       help='数据文件路径')
    parser.add_argument('--lookback', type=int, default=60,
                       help='回看窗口大小')
    parser.add_argument('--forecast_horizon', type=int, default=1,
                       help='预测范围')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def plot_predictions(y_true, y_pred, num_samples=200, save_path=None):
    """绘制预测结果对比图"""
    plt.figure(figsize=(15, 6))

    indices = np.arange(min(num_samples, len(y_true)))
    plt.plot(indices, y_true[:num_samples], 'b-', label='True Price', linewidth=2, alpha=0.7)
    plt.plot(indices, y_pred[:num_samples], 'r--', label='Predicted Price', linewidth=2, alpha=0.7)

    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.title('Stock Price Prediction', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 预测对比图已保存: {save_path}")


def plot_error_distribution(y_true, y_pred, save_path=None):
    """绘制误差分布图"""
    errors = y_pred - y_true
    percentage_errors = (errors / y_true) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 绝对误差分布
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error ($)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Absolute Error Distribution', fontsize=14)
    axes[0].grid(alpha=0.3)

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    axes[0].text(0.02, 0.98,
                f'Mean: ${mean_error:.2f}\nStd: ${std_error:.2f}',
                transform=axes[0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 百分比误差分布
    axes[1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error (%)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Percentage Error Distribution', fontsize=14)
    axes[1].grid(alpha=0.3)

    mean_pct_error = np.mean(percentage_errors)
    std_pct_error = np.std(percentage_errors)
    axes[1].text(0.02, 0.98,
                f'Mean: {mean_pct_error:.2f}%\nStd: {std_pct_error:.2f}%',
                transform=axes[1].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 误差分布图已保存: {save_path}")


def plot_scatter(y_true, y_pred, save_path=None):
    """绘制真实值vs预测值散点图"""
    plt.figure(figsize=(8, 8))

    plt.scatter(y_true, y_pred, alpha=0.5, s=20)

    # 理想线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')

    plt.xlabel('True Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title('True vs Predicted Price', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    # R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 散点图已保存: {save_path}")


def plot_attention_weights(attention_weights, num_samples=5, save_path=None):
    """绘制注意力权重热图"""
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]
        weights = attention_weights[i].reshape(1, -1)

        sns.heatmap(weights, ax=ax, cmap='YlOrRd', cbar=True,
                   xticklabels=10, yticklabels=False)
        ax.set_xlabel('Time Steps', fontsize=10)
        ax.set_title(f'Sample {i+1}: Attention Weights', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 注意力权重图已保存: {save_path}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("="*60)
    print("LSTM股票预测 - 模型评估")
    print("="*60)
    print(f"\n模型路径: {args.model_path}")
    print(f"数据处理器路径: {args.processor_path}")

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

    import pickle
    with open(args.processor_path, 'rb') as f:
        processor_data = pickle.load(f)

    print(f"✓ 数据处理器已加载")

    # ============================================
    # 步骤2: 准备数据
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 准备数据")
    print("="*60)

    try:
        train_data, val_data, test_data, processor = prepare_stock_data(
            data_path=args.data_path,
            lookback=args.lookback,
            forecast_horizon=args.forecast_horizon
        )
    except FileNotFoundError as e:
        print(f"\n✗ 数据文件不存在: {e}")
        return

    X_test, y_price_test, y_trend_test = test_data

    # ============================================
    # 步骤3: 加载模型
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 加载模型")
    print("="*60)

    model = keras.models.load_model(args.model_path, custom_objects={'AttentionLayer': AttentionLayer})
    print(f"✓ 模型已加载")

    # ============================================
    # 步骤4: 预测
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 模型预测")
    print("="*60)

    print("\n预测测试集...")
    predictions = model.predict(X_test, verbose=1)

    # 处理多任务输出
    if isinstance(predictions, list):
        y_pred_price = predictions[0].flatten()
        y_pred_trend = (predictions[1] > 0.5).astype(int).flatten()
        is_multitask = True
    else:
        y_pred_price = predictions.flatten()
        is_multitask = False

    # 反归一化
    y_test_original = processor.inverse_transform_price(y_price_test)
    y_pred_original = processor.inverse_transform_price(y_pred_price)

    # ============================================
    # 步骤5: 计算评估指标
    # ============================================
    print("\n" + "="*60)
    print("步骤5: 计算评估指标")
    print("="*60)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)

    # MAPE
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

    # 方向准确率
    true_direction = np.diff(y_test_original) > 0
    pred_direction = np.diff(y_pred_original) > 0
    direction_accuracy = np.mean(true_direction == pred_direction)

    print(f"\n测试集性能:")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  MSE:  ${mse:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    print(f"  方向准确率: {direction_accuracy:.2%}")

    if is_multitask:
        from sklearn.metrics import accuracy_score
        trend_accuracy = accuracy_score(y_trend_test, y_pred_trend)
        print(f"  趋势准确率: {trend_accuracy:.2%}")

    # ============================================
    # 步骤6: 绘制预测对比图
    # ============================================
    print("\n" + "="*60)
    print("步骤6: 绘制预测对比图")
    print("="*60)

    model_name = Path(args.model_path).stem
    pred_path = result_dir / f'{model_name}_predictions.png'
    plot_predictions(y_test_original, y_pred_original, num_samples=200, save_path=pred_path)

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
    # 步骤9: 注意力权重可视化（如果有）
    # ============================================
    if 'attention' in model_name:
        print("\n" + "="*60)
        print("步骤9: 注意力权重可视化")
        print("="*60)

        try:
            # 创建注意力模型
            attention_layer = None
            for layer in model.layers:
                if isinstance(layer, AttentionLayer):
                    attention_layer = layer
                    break

            if attention_layer is not None:
                # 获取注意力权重
                attention_model = keras.Model(
                    inputs=model.input,
                    outputs=attention_layer.output[1]
                )
                attention_weights = attention_model.predict(X_test[:5])

                attention_path = result_dir / f'{model_name}_attention_weights.png'
                plot_attention_weights(attention_weights, num_samples=5, save_path=attention_path)
            else:
                print("  未找到注意力层")
        except Exception as e:
            print(f"  注意力权重可视化失败: {e}")

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
    if 'attention' in model_name:
        print(f"  4. 注意力权重图: {attention_path}")

    print(f"\n模型性能总结:")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    if mape < 5:
        print(f"\n  ✓✓ 模型性能优秀！")
    elif mape < 10:
        print(f"\n  ✓ 模型性能良好")
    else:
        print(f"\n  ⚠ 模型性能有待提升")

    print(f"\n⚠️  风险提示:")
    print(f"  本模型仅供学习使用，不构成投资建议。")
    print(f"  股票投资有风险，入市需谨慎。")


if __name__ == '__main__':
    main()
