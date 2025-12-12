#!/usr/bin/env python
"""
Transformer文本分类项目 - 完整测试脚本

运行所有测试以验证项目功能完整性
"""

import sys
import subprocess


def run_test(name, command):
    """运行单个测试"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✓ {name} 通过")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} 失败")
        print(e.stdout)
        print(e.stderr)
        return False


def main():
    print("="*60)
    print("Transformer文本分类 - 完整测试套件")
    print("="*60)

    tests = [
        ("模块导入", "python -c 'import sys; sys.path.insert(0, \"src\"); from attention import *; from transformer import *; from data import *; from model import *; print(\"✓ 所有模块导入成功\")'"),

        ("attention模块", "python -c 'import sys; sys.path.insert(0, \"src\"); from attention import *; import tensorflow as tf; import numpy as np; mha = MultiHeadAttention(128, 4); x = tf.random.normal((2, 10, 128)); output, _ = mha(x, x, x); assert output.shape == (2, 10, 128); print(\"✓ attention模块测试通过\")'"),

        ("transformer模块", "python -c 'import sys; sys.path.insert(0, \"src\"); from transformer import *; import tensorflow as tf; encoder = TransformerEncoder(2, 128, 4, 512, 1000, 100); x = tf.random.uniform((2, 10), maxval=1000, dtype=tf.int32); output = encoder(x); assert output.shape == (2, 10, 128); print(\"✓ transformer模块测试通过\")'"),

        ("data模块", "python -c 'import sys; sys.path.insert(0, \"src\"); from data import *; preprocessor = TextPreprocessor(); text = \"<b>Test</b> Text!\"; cleaned = preprocessor(text); assert \"<b>\" not in cleaned; print(\"✓ data模块测试通过\")'"),

        ("model模块", "python -c 'import sys; sys.path.insert(0, \"src\"); from model import *; import numpy as np; classifier = TransformerTextClassifier(1000, 128, 2, \"simple\"); X = np.random.randint(0, 1000, (10, 128)); classifier.compile_model(); preds = classifier.predict(X); assert len(preds) == 10; print(\"✓ model模块测试通过\")'"),
    ]

    passed = 0
    failed = 0

    for name, command in tests:
        if run_test(name, command):
            passed += 1
        else:
            failed += 1

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总计: {passed + failed}")

    if failed == 0:
        print("\n✓✓ 所有测试通过！项目功能完整，可以正常使用。")
        return 0
    else:
        print(f"\n✗ {failed} 个测试失败，请检查错误信息。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
