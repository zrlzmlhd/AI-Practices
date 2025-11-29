#!/usr/bin/env python3
"""
Notebook注释增强脚本
为notebooks添加详细的中文注释和文档
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

# ============================================================
# 注释模板库
# ============================================================

COMMENT_TEMPLATES = {
    # 导入相关
    'import numpy': '# NumPy: 用于数值计算和数组操作',
    'import pandas': '# Pandas: 用于数据处理和分析',
    'import matplotlib': '# Matplotlib: 用于数据可视化',
    'import seaborn': '# Seaborn: 基于Matplotlib的高级可视化库',
    'import tensorflow': '# TensorFlow: 深度学习框架',
    'import keras': '# Keras: 高级神经网络API',
    'from sklearn': '# Scikit-learn: 机器学习库',
    'import torch': '# PyTorch: 深度学习框架',

    # 数据处理
    'train_test_split': '# 将数据划分为训练集和测试集',
    'StandardScaler': '# 标准化处理：将特征缩放到均值为0，标准差为1',
    'MinMaxScaler': '# 归一化处理：将特征缩放到[0, 1]范围',
    'LabelEncoder': '# 标签编码：将分类标签转换为数值',
    'OneHotEncoder': '# 独热编码：将分类变量转换为二进制向量',

    # 模型相关
    'LinearRegression': '# 线性回归模型',
    'LogisticRegression': '# 逻辑回归模型（用于分类）',
    'DecisionTreeClassifier': '# 决策树分类器',
    'RandomForestClassifier': '# 随机森林分类器',
    'SVC': '# 支持向量机分类器',
    'KMeans': '# K-Means聚类算法',
    'PCA': '# 主成分分析（降维）',

    # 深度学习
    'Sequential': '# 顺序模型：层的线性堆叠',
    'Dense': '# 全连接层',
    'Conv2D': '# 二维卷积层',
    'MaxPooling2D': '# 二维最大池化层',
    'Dropout': '# Dropout层：防止过拟合',
    'BatchNormalization': '# 批标准化层：加速训练，稳定梯度',
    'LSTM': '# 长短期记忆网络层',
    'GRU': '# 门控循环单元层',
    'Embedding': '# 嵌入层：将整数索引转换为稠密向量',

    # 训练相关
    '.fit(': '# 训练模型',
    '.predict(': '# 使用模型进行预测',
    '.evaluate(': '# 评估模型性能',
    '.compile(': '# 编译模型：配置损失函数、优化器和评估指标',

    # 评估指标
    'accuracy_score': '# 计算准确率',
    'precision_score': '# 计算精确率',
    'recall_score': '# 计算召回率',
    'f1_score': '# 计算F1分数',
    'confusion_matrix': '# 混淆矩阵',
    'classification_report': '# 分类报告',
    'mean_squared_error': '# 均方误差',
    'r2_score': '# R²决定系数',
}

# 代码块说明模板
CODE_BLOCK_EXPLANATIONS = {
    'np.random.seed': '''
# ============================================================
# 设置随机种子
# 作用：确保每次运行代码时产生相同的随机数，保证结果可重复
# ============================================================''',

    'plt.figure': '''
# ============================================================
# 创建图形
# ============================================================''',

    'model.fit': '''
# ============================================================
# 模型训练
# 将训练数据输入模型，通过优化算法调整模型参数
# ============================================================''',

    'model.compile': '''
# ============================================================
# 编译模型
# 配置训练过程中使用的优化器、损失函数和评估指标
# ============================================================''',
}


def load_notebook(path: Path) -> dict:
    """加载notebook文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(path: Path, notebook: dict) -> None:
    """保存notebook文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)


def add_inline_comments(source: str) -> str:
    """为代码添加行内注释"""
    lines = source.split('\n')
    new_lines = []

    # 记录已添加的注释，避免重复
    added_comments = set()

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 跳过空行和已有注释的行
        if not stripped or stripped.startswith('#'):
            new_lines.append(line)
            continue

        # 检查是否需要添加注释
        comment_added = False
        for pattern, comment in COMMENT_TEMPLATES.items():
            if pattern in line and '#' not in line:
                # 检查前一行是否已有相同注释
                prev_line = new_lines[-1].strip() if new_lines else ''
                if prev_line == comment.strip():
                    # 已有注释，跳过
                    new_lines.append(line)
                    comment_added = True
                    break

                # 检查是否已在本cell添加过此注释
                comment_key = f"{comment}:{i}"
                if comment_key not in added_comments:
                    new_lines.append(f"{comment}")
                    added_comments.add(comment_key)

                new_lines.append(line)
                comment_added = True
                break

        if not comment_added:
            new_lines.append(line)

    return '\n'.join(new_lines)


def add_block_comments(source: str) -> str:
    """为代码块添加分隔注释"""
    for pattern, block_comment in CODE_BLOCK_EXPLANATIONS.items():
        if pattern in source and block_comment.strip() not in source:
            # 在模式之前添加块注释
            source = source.replace(pattern, f"{block_comment}\n{pattern}")

    return source


def enhance_notebook_comments(notebook: dict) -> Tuple[dict, int]:
    """增强notebook中的注释"""
    comments_added = 0

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source_list = cell.get('source', [])
            if isinstance(source_list, list):
                source = ''.join(source_list)
            else:
                source = source_list

            original_length = len(source)

            # 添加行内注释
            source = add_inline_comments(source)

            # 添加块注释
            source = add_block_comments(source)

            if len(source) > original_length:
                comments_added += 1
                # 更新cell源码
                new_source = source.split('\n')
                cell['source'] = [line + '\n' for line in new_source[:-1]] + [new_source[-1]]

    return notebook, comments_added


def create_header_cell(title: str, description: str) -> dict:
    """创建标题markdown单元格"""
    content = f"""# {title}

{description}

---

## 📚 本节内容

完成本节学习后，你将：
- 理解核心概念和原理
- 掌握代码实现方法
- 能够应用到实际问题

## ⏱️ 预计时间

15-25分钟
"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }


def process_notebook(path: Path, dry_run: bool = False) -> dict:
    """处理单个notebook"""
    report = {
        'path': str(path),
        'comments_added': 0,
        'error': None
    }

    try:
        notebook = load_notebook(path)
        notebook, comments_added = enhance_notebook_comments(notebook)
        report['comments_added'] = comments_added

        if comments_added > 0 and not dry_run:
            save_notebook(path, notebook)
            report['status'] = 'modified'
        elif comments_added > 0:
            report['status'] = 'would_modify'
        else:
            report['status'] = 'no_change'

    except Exception as e:
        report['error'] = str(e)
        report['status'] = 'error'

    return report


def find_notebooks(root_dir: Path, exclude_optimized: bool = True) -> List[Path]:
    """查找需要处理的notebook文件"""
    notebooks = list(root_dir.rglob('*.ipynb'))
    filtered = []

    for nb in notebooks:
        if '.ipynb_checkpoints' in str(nb):
            continue
        if exclude_optimized and '优化版' in str(nb):
            continue
        filtered.append(nb)

    return filtered


def main():
    import argparse

    parser = argparse.ArgumentParser(description='为Notebooks添加详细注释')
    parser.add_argument('--dry-run', action='store_true', help='仅检查，不实际修改')
    parser.add_argument('--path', type=str, default='.', help='项目根目录')
    parser.add_argument('--include-optimized', action='store_true', help='包含优化版notebooks')
    args = parser.parse_args()

    root = Path(args.path)
    notebooks = find_notebooks(root, exclude_optimized=not args.include_optimized)

    print(f"找到 {len(notebooks)} 个notebook文件")
    print("=" * 60)

    total_comments = 0
    modified_count = 0

    for nb_path in notebooks:
        report = process_notebook(nb_path, dry_run=args.dry_run)

        if report['comments_added'] > 0:
            print(f"\n📓 {report['path']}")
            print(f"  ✅ 添加注释: {report['comments_added']}处")
            total_comments += report['comments_added']
            modified_count += 1

        if report['error']:
            print(f"\n📓 {report['path']}")
            print(f"  ❌ 错误: {report['error']}")

    print("\n" + "=" * 60)
    print(f"📊 总结:")
    print(f"  - 处理文件: {len(notebooks)}")
    print(f"  - 添加注释: {total_comments}处")
    print(f"  - 修改文件: {modified_count}")

    if args.dry_run:
        print("\n⚠️  这是dry-run模式，没有实际修改文件")


if __name__ == '__main__':
    main()
