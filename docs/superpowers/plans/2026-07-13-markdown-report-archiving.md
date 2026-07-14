# Markdown Report Archiving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Markdown 论文阅读报告归档规范融入项目根目录 `AGENTS.md`，并确保仅在用户主动要求时保存。

**Architecture:** 删除 PDF 归档章节末尾与新需求冲突的旧规则，并在其后新增独立的“Markdown 阅读报告归档”章节。新章节复用既有飞书问答融合规则和 Markdown 块级公式规范，同时明确文件路径、命名、冲突检查、非空验证及最终报告要求。

**Tech Stack:** Markdown、PowerShell 文本验证、Git

---

### Task 1: 融合 Markdown 报告归档规则

**Files:**
- Modify: `AGENTS.md`
- Reference: `docs/superpowers/specs/2026-07-13-markdown-report-archiving-design.md`

- [x] **Step 1: 验证旧规则存在且新章节不存在**

```powershell
$content = Get-Content -Raw -LiteralPath 'AGENTS.md'
if ($content -notmatch '只归档原始 PDF，不自动把论文阅读报告保存为文件') {
    throw '未找到待替换的旧规则'
}
if ($content -match '## Markdown 阅读报告归档') {
    throw 'Markdown 阅读报告归档章节已经存在'
}
```

预期结果：命令退出码为 0，证明测试能锁定本次变更前状态。

- [x] **Step 2: 删除冲突旧句并新增完整章节**

将旧句替换为以下内容：

```markdown
## Markdown 阅读报告归档

每次生成论文总结时，除非用户明确要求不保存，否则自动保存 Markdown 阅读报告。

1. Markdown 报告的总结内容与输出到飞书文档的版本一致。若此前对话中有针对论文的提问与解答，应遵循“默认工作要求”，将这些内容融合到研究背景、研究方法、图表分析、主要发现或局限性等对应章节中，不单独新增“问答”章节。
2. Markdown 报告中的公式遵循“公式输出规则”的 Markdown 格式：所有数学表达使用独立块级 `$$`，开始标记、单行 KaTeX 内容和结束标记分别独占一行；不要使用飞书文档专用的 `<equation>...</equation>` 标签。
3. Markdown 文件与归档 PDF 保存在同一目录，文件主名与 PDF 完全一致，仅将 `.pdf` 后缀替换为 `.md`。例如 `扩散模型/Example Paper_CVPR2026.pdf` 对应 `扩散模型/Example Paper_CVPR2026.md`。
4. 若同名 Markdown 文件已存在，先核对文件内容：内容相同则复用现有文件；内容不同则报告冲突，不得静默覆盖。
5. 保存后确认 Markdown 文件非空。保存失败时明确报告原因，不得声称已经归档。
6. 在最终阅读结果中明确列出 Markdown 报告的本地保存路径。
```

- [x] **Step 3: 验证规则完整且没有冲突**

```powershell
$content = Get-Content -Raw -LiteralPath 'AGENTS.md'
$required = @(
    '## Markdown 阅读报告归档',
    '总结内容与输出到飞书文档的版本一致',
    '不要使用飞书文档专用的 `<equation>...</equation>` 标签',
    '仅将 `.pdf` 后缀替换为 `.md`',
    '内容不同则报告冲突，不得静默覆盖',
    '确认 Markdown 文件非空',
    'Markdown 报告的本地保存路径'
)
$missing = @($required | Where-Object { -not $content.Contains($_) })
if ($missing.Count -gt 0) {
    throw "缺少规则: $($missing -join ', ')"
}
if ($content -match '只归档原始 PDF，不自动把论文阅读报告保存为文件') {
    throw '冲突旧规则仍然存在'
}
```

预期结果：命令退出码为 0，所有新增要求存在且旧冲突规则已删除。

- [x] **Step 4: 检查差异范围**

```powershell
git diff -- AGENTS.md
```

预期结果：差异仅包括删除冲突旧句并在相同位置加入新的 Markdown 归档章节；不改动 PDF 归档、简单总结、公式规则和默认总结结构的其他内容。

- [x] **Step 5: 提交变更（Git 身份可用时）**

```powershell
git add -- AGENTS.md docs/superpowers/plans/2026-07-13-markdown-report-archiving.md
git commit -m "docs: require markdown paper reports"
```

若 Git 仍未配置 `user.name` 或 `user.email`，保留修改并明确报告，不擅自设置身份或提交。

### Task 2: 将保存行为改为用户主动触发

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-07-13-markdown-report-archiving-design.md`

- [x] **Step 1: 验证当前自动保存表述存在**

```powershell
$content = Get-Content -Raw -LiteralPath 'AGENTS.md'
if (-not $content.Contains('每次生成论文总结时，除非用户明确要求不保存，否则自动保存 Markdown 阅读报告。')) {
    throw '未找到待替换的自动保存规则'
}
```

预期结果：命令退出码为 0。

- [x] **Step 2: 替换触发条件**

将自动保存句替换为：

```markdown
默认不保存 Markdown 阅读报告。仅当用户主动要求输出或保存 Markdown 文档时，才按照本节规则生成并归档。
```

编号 1–6 的内容、公式、路径、命名、冲突处理和验证规则保持不变。

- [x] **Step 3: 验证新旧规则互斥**

```powershell
$content = Get-Content -Raw -LiteralPath 'AGENTS.md'
if (-not $content.Contains('默认不保存 Markdown 阅读报告。仅当用户主动要求输出或保存 Markdown 文档时，才按照本节规则生成并归档。')) {
    throw '缺少用户主动触发规则'
}
if ($content.Contains('除非用户明确要求不保存，否则自动保存 Markdown 阅读报告')) {
    throw '自动保存旧规则仍然存在'
}
```

预期结果：命令退出码为 0。
