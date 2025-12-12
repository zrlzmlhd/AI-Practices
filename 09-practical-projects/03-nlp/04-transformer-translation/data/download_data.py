"""
数据集下载工具

使用方法:
    cd data
    python download_data.py

支持的数据集：
- WMT系列：机器翻译Workshop数据集
- IWSLT：口语翻译数据集
- Tatoeba：开源多语言句对集合
"""

import sys
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def download_file(url: str, dest_path: Path, desc: Optional[str] = None):
    """
    下载文件并显示进度条

    Args:
        url: 下载链接
        dest_path: 保存路径
        desc: 进度条描述
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f, tqdm(
        desc=desc or dest_path.name,
        total=total_size,
        unit='B',
        unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_data():
    """
    下载翻译数据集

    提供多个数据源选项，用户可根据需求选择
    """
    print(f"{'='*60}")
    print("翻译数据集下载工具")
    print(f"{'='*60}")

    print("\n推荐数据集:")
    print("1. WMT English-German: 英德翻译大规模数据集")
    print("2. WMT English-Chinese: 英中翻译数据集")
    print("3. IWSLT English-German: TED演讲翻译")
    print("4. Tatoeba: 多语言短句翻译")

    print("\n请手动下载数据集并放置在data/目录：")
    print("- WMT: http://www.statmt.org/wmt/")
    print("- IWSLT: https://wit3.fbk.eu/")
    print("- Tatoeba: https://tatoeba.org/downloads")

    print("\n文件格式要求：")
    print("- 源语言文件: train.en (每行一个句子)")
    print("- 目标语言文件: train.zh (每行一个句子)")
    print("- 行数必须相同，对应行互为翻译")


if __name__ == '__main__':
    download_data()
