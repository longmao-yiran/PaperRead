# Initial Paper Folders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在论文阅读项目根目录创建并验证 12 个任务级论文分类目录。

**Architecture:** 目录全部位于项目根目录，使用 PowerShell 的幂等目录创建能力；已有目录直接复用，不覆盖或删除内容。创建后逐项验证路径存在且类型为目录，并核对目标目录集合。

**Tech Stack:** PowerShell、Windows 文件系统

---

### Task 1: 创建论文分类目录

**Files:**
- Create: `图像恢复/`
- Create: `RAW图像去噪/`
- Create: `低光成像/`
- Create: `ISP与噪声建模/`
- Create: `扩散模型/`
- Create: `Flow Matching/`
- Create: `MeanFlow/`
- Create: `视觉语言模型/`
- Create: `目标检测/`
- Create: `视觉问答/`
- Create: `图像编辑/`
- Create: `智能构图/`

- [x] **Step 1: 创建目标目录**

在项目根目录运行：

```powershell
$folders = @(
    '图像恢复', 'RAW图像去噪', '低光成像', 'ISP与噪声建模',
    '扩散模型', 'Flow Matching', 'MeanFlow', '视觉语言模型',
    '目标检测', '视觉问答', '图像编辑', '智能构图'
)
$folders | ForEach-Object {
    New-Item -ItemType Directory -Path $_ -Force | Out-Null
}
```

预期结果：命令退出码为 0；已存在目录被复用，其他目录被创建。

- [x] **Step 2: 逐项验证目录**

```powershell
$missing = $folders | Where-Object {
    -not (Test-Path -LiteralPath $_ -PathType Container)
}
if ($missing.Count -gt 0) {
    throw "缺少目录: $($missing -join ', ')"
}
$folders | ForEach-Object {
    Get-Item -LiteralPath $_ | Select-Object Name, FullName, PSIsContainer
}
```

预期结果：命令退出码为 0，输出 12 条记录，每条记录的 `PSIsContainer` 均为 `True`。

- [x] **Step 3: 核对项目现有文件保持不变**

```powershell
if (-not (Test-Path -LiteralPath 'AGENTS.md' -PathType Leaf)) {
    throw 'AGENTS.md 不存在'
}
Get-Item -LiteralPath 'AGENTS.md' | Select-Object Name, Length
```

预期结果：命令退出码为 0，并输出非空的 `AGENTS.md` 文件信息。

- [x] **Step 4: 提交变更（仅当 Git 可用）**

```powershell
git add docs/superpowers/specs/2026-07-13-initial-paper-folders-design.md docs/superpowers/plans/2026-07-13-initial-paper-folders.md
git commit -m "chore: initialize paper category folders"
```

当前环境未发现 Git 可执行程序，因此本步骤记录为环境限制，不阻塞目录创建与验证。
