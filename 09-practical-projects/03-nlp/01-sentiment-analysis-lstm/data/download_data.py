"""
IMDB数据集下载脚本

使用方法:
    cd data
    python download_data.py
"""

import sys
from pathlib import Path
from tensorflow import keras

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def download_imdb_data():
    """
    下载IMDB数据集

    Keras会自动下载并缓存数据集到 ~/.keras/datasets/
    """
    print("="*60)
    print("下载IMDB数据集")
    print("="*60)

    print("\n正在下载数据集...")
    print("数据将被缓存到: ~/.keras/datasets/imdb.npz")

    try:
        # 下载数据集
        (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()

        print(f"\n✓ 下载成功！")
        print(f"  训练集大小: {len(X_train)}")
        print(f"  测试集大小: {len(X_test)}")

        # 下载词汇索引
        print("\n正在下载词汇索引...")
        word_index = keras.datasets.imdb.get_word_index()
        print(f"✓ 词汇索引下载成功！")
        print(f"  词汇表大小: {len(word_index)}")

        print("\n" + "="*60)
        print("数据集下载完成！")
        print("="*60)
        print("\n你现在可以运行训练脚本:")
        print("  cd ..")
        print("  python src/train.py --model_type simple_lstm --epochs 10")

    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n请检查网络连接，或手动下载数据集:")
        print("  https://ai.stanford.edu/~amaas/data/sentiment/")


if __name__ == '__main__':
    download_imdb_data()
