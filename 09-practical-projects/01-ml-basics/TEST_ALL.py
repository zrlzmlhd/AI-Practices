"""
所有项目的综合测试脚本

用于验证所有项目的代码能否正常运行
"""
import sys
from pathlib import Path

def test_project(project_name, test_command):
    """测试单个项目"""
    print("=" * 80)
    print(f"测试项目: {project_name}")
    print("=" * 80)

    try:
        import subprocess
        result = subprocess.run(
            test_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"✓ {project_name} 测试通过")
            return True
        else:
            print(f"✗ {project_name} 测试失败")
            print(f"错误信息: {result.stderr[:500]}")
            return False

    except Exception as e:
        print(f"✗ {project_name} 测试异常: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("开始测试所有项目")
    print("=" * 80 + "\n")

    base_path = Path(__file__).parent

    tests = [
        ("01-titanic-survival-xgboost",
         f"cd {base_path}/01-titanic-survival-xgboost/src && python data.py"),

        ("02-otto-classification-xgboost",
         f"cd {base_path}/02-otto-classification-xgboost/src && python data.py"),

        ("03-svm-text-classification",
         f"cd {base_path}/03-svm-text-classification/src && python data.py"),

        ("04-xgboost-advanced",
         f"cd {base_path}/04-xgboost-advanced/src && python data.py"),
    ]

    results = []
    for project_name, command in tests:
        success = test_project(project_name, command)
        results.append((project_name, success))
        print()

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for project_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {project_name:40s} {status}")

    print(f"\n总计: {passed}/{total} 项目通过测试")

    if passed == total:
        print("\n🎉 所有项目测试通过！")
        return 0
    else:
        print(f"\n⚠ 还有 {total - passed} 个项目需要修复")
        return 1


if __name__ == '__main__':
    sys.exit(main())
