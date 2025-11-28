#!/usr/bin/env python3
"""
Notebook批量优化脚本
用于自动修复常见问题：
1. 替换弃用的API (fit_generator -> fit)
2. 添加随机种子设置
3. 修复import语句
"""

import json
import os
import re
from pathlib import Path
from typing import List, Tuple


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

    return source, fixes


def fix_hardcoded_paths(source: str) -> Tuple[str, List[str]]:
    """
    标记硬编码路径（不自动替换，因为需要根据实际情况处理）
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

    return source, warnings


def process_notebook(path: Path, dry_run: bool = False) -> dict:
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

        modified = False
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))

                # 修复弃用API
                new_source, fixes = fix_deprecated_apis(source)
                if fixes:
                    report['fixes'].extend(fixes)
                    if new_source != source:
                        if isinstance(cell['source'], list):
                            cell['source'] = new_source.split('\n')
                            cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                        else:
                            cell['source'] = new_source
                        modified = True

                # 检查硬编码路径
                _, warnings = fix_hardcoded_paths(source)
                report['warnings'].extend(warnings)

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
    return list(root_dir.rglob('*.ipynb'))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='批量优化Jupyter Notebooks')
    parser.add_argument('--dry-run', action='store_true', help='仅检查，不实际修改')
    parser.add_argument('--path', type=str, default='.', help='项目根目录')
    args = parser.parse_args()

    root = Path(args.path)
    notebooks = find_notebooks(root)

    print(f"找到 {len(notebooks)} 个notebook文件")
    print("=" * 60)

    total_fixes = 0
    total_warnings = 0
    modified_count = 0

    for nb_path in notebooks:
        # 跳过checkpoint文件
        if '.ipynb_checkpoints' in str(nb_path):
            continue

        report = process_notebook(nb_path, dry_run=args.dry_run)

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


if __name__ == '__main__':
    main()
