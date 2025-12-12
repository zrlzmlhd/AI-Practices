"""
Titanic数据集下载脚本
"""
import requests
from pathlib import Path

def download_titanic_data():
    print("="*60)
    print("下载Titanic数据集")
    print("="*60)
    
    train_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    try:
        print("\n正在下载数据...")
        response = requests.get(train_url, timeout=30)
        response.raise_for_status()
        
        data_dir = Path(__file__).parent
        train_path = data_dir / 'train.csv'
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"\n✓ 下载成功！")
        print(f"  保存路径: {train_path}")
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n请手动下载: https://www.kaggle.com/c/titanic/data")

if __name__ == '__main__':
    download_titanic_data()
