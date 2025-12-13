# Contributing

感谢你对 AI-Practices 项目的关注！我们欢迎任何形式的贡献。

---

## Code of Conduct

为了营造一个开放和友好的环境，我们承诺：

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情

---

## How to Contribute

### 1. Report Bugs

如果你发现了 bug，请：

1. 检查 [Issues](https://github.com/zimingttkx/AI-Practices/issues) 确保该 bug 尚未被报告
2. 创建新 issue，包含清晰的标题和描述
3. 提供重现步骤和环境信息
4. 附上屏幕截图（如果适用）

### 2. Suggest Features

1. 创建 Feature Request issue
2. 描述功能需求和使用场景
3. 说明预期行为

### 3. Submit Code

#### Development Workflow

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/YOUR-USERNAME/AI-Practices.git
cd AI-Practices

# 2. 创建新分支
git checkout -b feature/your-feature-name

# 3. 进行修改并提交
git add .
git commit -m "feat: add new feature"

# 4. 推送并创建 PR
git push origin feature/your-feature-name
```

#### Branch Naming

| Prefix | Purpose |
|:-------|:--------|
| `feature/` | 新功能 |
| `fix/` | Bug 修复 |
| `docs/` | 文档更新 |
| `refactor/` | 代码重构 |

#### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat` - 新功能
- `fix` - Bug 修复
- `docs` - 文档更新
- `style` - 代码格式
- `refactor` - 重构
- `test` - 测试
- `chore` - 构建/工具

**Example:**
```
feat(ml-basics): add ridge regression example

- 添加 Ridge 回归的完整实现
- 包含交叉验证和超参数调优
- 添加可视化结果

Closes #123
```

---

## Code Style

### Python

- 遵循 PEP 8
- 使用 Black 格式化: `black --line-length 88`
- 使用 Type Hints
- 添加 Docstrings (Google Style)

### Jupyter Notebooks

- 清晰的标题和目录
- 详细的注释
- 清理输出后提交
- 设置随机种子保证可复现

---

## Priority Areas

### High Priority

- 改进现有代码 - 添加详细注释、优化代码结构
- 完善文档 - 补充理论说明、添加使用示例
- 添加可视化 - 改进图表质量、添加交互式可视化

### Medium Priority

- 新增教程 - 填补知识空白、添加实战项目
- 性能优化 - 提高代码效率、减少内存占用

---

## Recognition

所有贡献者将被列入项目贡献者名单。感谢每一位贡献者的付出！

---

<div align="center">

**[[← Home|Home]]**

</div>
