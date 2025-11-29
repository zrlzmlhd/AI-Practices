#!/usr/bin/env python3
"""
Notebook批量优化脚本
用于自动修复常见问题：
1. 替换弃用的API (fit_generator -> fit)
2. 添加随机种子设置
3. 修复import语句
4. 规范化代码风格
5. 添加标准配置头
"""

import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional


# ============================================================
# 标准代码模板
# ============================================================

STANDARD_IMPORTS_ML = '''# ============================================================
# 导入必要的库
# ============================================================

# 数值计算
import numpy as np

# 数据处理
import pandas as pd

# 可视化
import matplotlib.pyplot as plt

# ============================================================
# 配置参数
# ============================================================

# 设置随机种子，确保结果可重复
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 可视化配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

print("✓ 环境配置完成")'''

STANDARD_IMPORTS_DL = '''# ============================================================
# 导入必要的库
# ============================================================

# 数值计算
import numpy as np

# 可视化
import matplotlib.pyplot as plt

# 深度学习框架
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# 配置参数
# ============================================================

# 设置随机种子，确保结果可重复
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# GPU配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU可用: {len(gpus)}个")
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")
else:
    print("⚠ 未检测到GPU，使用CPU")

# 可视化配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

print("✓ 环境配置完成")
print(f"✓ TensorFlow版本: {tf.__version__}")'''


def load_notebook(path: Path) -> dict:
    """加载notebook文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(path: Path, notebook: dict) -> None:
    """保存notebook文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)


def fix_deprecated_apis(source: str) -> Tuple[str, List[str]]:
    """
    修复弃用的API调用
    返回: (修复后的代码, 修复列表)
    """
    fixes = []

    # 替换 fit_generator -> fit
    if 'fit_generator' in source:
        source = source.replace('fit_generator', 'fit')
        fixes.append('fit_generator -> fit')

    # 替换 predict_generator -> predict
    if 'predict_generator' in source:
        source = source.replace('predict_generator', 'predict')
        fixes.append('predict_generator -> predict')

    # 替换 evaluate_generator -> evaluate
    if 'evaluate_generator' in source:
        source = source.replace('evaluate_generator', 'evaluate')
        fixes.append('evaluate_generator -> evaluate')

    # 替换 keras.preprocessing.image -> keras.utils
    if 'keras.preprocessing.image' in source:
        source = source.replace('keras.preprocessing.image', 'keras.utils')
        fixes.append('keras.preprocessing.image -> keras.utils')

    # 修复 tf.keras.optimizers.schedules 的旧API
    if 'schedules.ExponentialDecay' in source and 'learning_rate_schedule' not in source:
        # 这个需要更复杂的处理，暂时只标记
        pass

    return source, fixes


def fix_hardcoded_paths(source: str) -> Tuple[str, List[str]]:
    """
    检测硬编码路径
    返回: (原代码, 警告列表)
    """
    warnings = []

    # 检测Windows路径
    windows_pattern = r'r?["\']C:\\[^"\']+["\']'
    if re.search(windows_pattern, source):
        warnings.append('检测到Windows硬编码路径')

    # 检测Linux绝对路径
    linux_pattern = r'r?["\']/home/[^"\']+["\']'
    if re.search(linux_pattern, source):
        warnings.append('检测到Linux硬编码路径')

    # 检测macOS用户路径
    macos_pattern = r'r?["\']/Users/[^"\']+["\']'
    if re.search(macos_pattern, source):
        warnings.append('检测到macOS硬编码路径')

    return source, warnings


def fix_common_issues(source: str) -> Tuple[str, List[str]]:
    """
    修复常见代码问题
    返回: (修复后的代码, 修复列表)
    """
    fixes = []

    # 修复 np.int -> int (numpy 1.24+已弃用)
    if 'np.int,' in source or 'np.int)' in source or 'np.int]' in source:
        source = re.sub(r'\bnp\.int\b', 'int', source)
        fixes.append('np.int -> int')

    # 修复 np.float -> float
    if 'np.float,' in source or 'np.float)' in source or 'np.float]' in source:
        source = re.sub(r'\bnp\.float\b', 'float', source)
        fixes.append('np.float -> float')

    # 修复 np.bool -> bool
    if 'np.bool,' in source or 'np.bool)' in source or 'np.bool]' in source:
        source = re.sub(r'\bnp\.bool\b', 'bool', source)
        fixes.append('np.bool -> bool')

    return source, fixes


def check_has_random_seed(notebook: dict) -> bool:
    """检查notebook是否已设置随机种子"""
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'random.seed' in source or 'np.random.seed' in source or 'tf.random.set_seed' in source:
                return True
    return False


def check_is_deep_learning(notebook: dict) -> bool:
    """检查是否是深度学习notebook"""
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'tensorflow' in source.lower() or 'keras' in source.lower() or 'torch' in source.lower():
                return True
    return False


def add_random_seed_to_imports(source: str, is_dl: bool = False) -> str:
    """在import语句后添加随机种子设置"""
    lines = source.split('\n')
    new_lines = []
    seed_added = False

    for i, line in enumerate(lines):
        new_lines.append(line)
        # 在numpy import后添加seed
        if not seed_added and ('import numpy' in line or 'import np' in line):
            if is_dl:
                # 检查后面是否已有seed
                remaining = '\n'.join(lines[i+1:])
                if 'random.seed' not in remaining[:200]:
                    new_lines.append('')
                    new_lines.append('# 设置随机种子')
                    new_lines.append('RANDOM_SEED = 42')
                    new_lines.append('np.random.seed(RANDOM_SEED)')
                    seed_added = True
            else:
                remaining = '\n'.join(lines[i+1:])
                if 'random.seed' not in remaining[:200]:
                    new_lines.append('')
                    new_lines.append('# 设置随机种子')
                    new_lines.append('np.random.seed(42)')
                    seed_added = True

    return '\n'.join(new_lines)


def process_notebook(path: Path, dry_run: bool = False, add_seed: bool = False) -> dict:
    """
    处理单个notebook
    返回处理报告
    """
    report = {
        'path': str(path),
        'fixes': [],
        'warnings': [],
        'error': None
    }

    try:
        notebook = load_notebook(path)

        # 检查特征
        has_seed = check_has_random_seed(notebook)
        is_dl = check_is_deep_learning(notebook)

        if not has_seed:
            report['warnings'].append('缺少随机种子设置')

        modified = False
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                original_source = source

                # 修复弃用API
                source, fixes = fix_deprecated_apis(source)
                report['fixes'].extend(fixes)

                # 修复常见问题
                source, common_fixes = fix_common_issues(source)
                report['fixes'].extend(common_fixes)

                # 检查硬编码路径
                _, warnings = fix_hardcoded_paths(source)
                report['warnings'].extend(warnings)

                # 如果需要添加随机种子
                if add_seed and not has_seed:
                    source = add_random_seed_to_imports(source, is_dl)
                    if source != original_source:
                        report['fixes'].append('添加随机种子')
                        has_seed = True

                if source != original_source:
                    if isinstance(cell['source'], list):
                        cell['source'] = source.split('\n')
                        cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                    else:
                        cell['source'] = source
                    modified = True

        if modified and not dry_run:
            save_notebook(path, notebook)
            report['status'] = 'modified'
        elif modified:
            report['status'] = 'would_modify'
        else:
            report['status'] = 'no_change'

    except Exception as e:
        report['error'] = str(e)
        report['status'] = 'error'

    return report


def find_notebooks(root_dir: Path) -> List[Path]:
    """查找所有notebook文件"""
    notebooks = list(root_dir.rglob('*.ipynb'))
    # 过滤掉checkpoint文件
    return [nb for nb in notebooks if '.ipynb_checkpoints' not in str(nb)]


def generate_quality_report(reports: List[dict]) -> str:
    """生成质量报告"""
    total = len(reports)
    with_issues = sum(1 for r in reports if r['fixes'] or r['warnings'])
    modified = sum(1 for r in reports if r['status'] in ['modified', 'would_modify'])
    errors = sum(1 for r in reports if r['error'])

    missing_seed = sum(1 for r in reports if '缺少随机种子设置' in r.get('warnings', []))
    hardcoded_paths = sum(1 for r in reports if any('硬编码路径' in w for w in r.get('warnings', [])))

    report = f"""
# Notebook质量报告

## 概览
- 总文件数: {total}
- 有问题的文件: {with_issues}
- 已修改文件: {modified}
- 处理错误: {errors}

## 问题分布
- 缺少随机种子: {missing_seed}
- 硬编码路径: {hardcoded_paths}

## 详细问题列表
"""

    for r in reports:
        if r['fixes'] or r['warnings']:
            report += f"\n### {r['path']}\n"
            if r['fixes']:
                report += f"- 修复: {', '.join(set(r['fixes']))}\n"
            if r['warnings']:
                report += f"- 警告: {', '.join(set(r['warnings']))}\n"

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='批量优化Jupyter Notebooks')
    parser.add_argument('--dry-run', action='store_true', help='仅检查，不实际修改')
    parser.add_argument('--path', type=str, default='.', help='项目根目录')
    parser.add_argument('--add-seed', action='store_true', help='自动添加随机种子')
    parser.add_argument('--report', type=str, help='输出质量报告到文件')
    args = parser.parse_args()

    root = Path(args.path)
    notebooks = find_notebooks(root)

    print(f"找到 {len(notebooks)} 个notebook文件")
    print("=" * 60)

    reports = []
    total_fixes = 0
    total_warnings = 0
    modified_count = 0

    for nb_path in notebooks:
        report = process_notebook(nb_path, dry_run=args.dry_run, add_seed=args.add_seed)
        reports.append(report)

        if report['fixes'] or report['warnings'] or report['error']:
            print(f"\n📓 {report['path']}")

            if report['fixes']:
                print(f"  ✅ 修复: {', '.join(set(report['fixes']))}")
                total_fixes += len(report['fixes'])

            if report['warnings']:
                print(f"  ⚠️  警告: {', '.join(set(report['warnings']))}")
                total_warnings += len(report['warnings'])

            if report['error']:
                print(f"  ❌ 错误: {report['error']}")

            if report['status'] in ['modified', 'would_modify']:
                modified_count += 1

    print("\n" + "=" * 60)
    print(f"📊 总结:")
    print(f"  - 处理文件: {len(notebooks)}")
    print(f"  - 修复问题: {total_fixes}")
    print(f"  - 警告数量: {total_warnings}")
    print(f"  - 修改文件: {modified_count}")

    if args.dry_run:
        print("\n⚠️  这是dry-run模式，没有实际修改文件")
        print("   移除 --dry-run 参数以应用修改")

    if args.report:
        quality_report = generate_quality_report(reports)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(quality_report)
        print(f"\n📝 质量报告已保存到: {args.report}")


if __name__ == '__main__':
    main()
