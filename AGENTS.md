# 项目代理规则

## 当前目录与运行约定

- `content/` 是公开内容真源：`content/papers/` 保存候选账本、公开论文归档、标注；`content/writings/` 保存文章 bundle。不再建立根目录 `data/`。
- `config/` 保存主题、模型、会议与任务参数；`src/` 保存全部可复用业务逻辑及 CLI；根目录不保留业务脚本。
- `docs/` 只用于网站文件与生成产物；历史实施计划不再发布。当前架构和运行契约集中在本文件及各模块 AGENTS.md。
- `ops/` 只保留本机 WSL 服务配置、薄启动入口和一份运行说明，继续 Git 忽略。业务编排属于 `src/papers/`。
- `src/papers/batch/` 保存历史批处理与兼容现有缓存/断点的逻辑；`src/papers/conferences.py` 负责配置驱动的会议时间线。
- 采集只更新候选账本。每日本地任务负责筛选、摘要、标注和发布；周末批处理共用模型锁，按批次让行每日任务。
- 临时实施计划和验收只放在忽略的 `build/reports/`，完成后删除测试文件。不得清空原文、Markdown、缓存、断点或 `.git.pre-reinit-backup/`。
- 会议配置保留逐届官方来源和录用论文库，默认展示当前年及前两年；双年会只记录实际举办届次。
- 模型系列按官方主要发布节点组织，技术精读沿用本地 Markdown 真源及现有发布器；缺少披露的技术字段统一显示 `/`，不得猜测或用商业许可替代技术局限。

## 修改前同步远程代码

- 每次开始修改项目文件前，必须先执行 `git pull --ff-only`，确认本地分支已同步远程最新代码。
- 用户已授权本项目常规 Git 操作：status、diff、fetch、pull --ff-only、创建/切换功能分支、按特性暂存与 commit、快进合并及普通 push，无需重复确认。权限不足时可申请工具权限后重试原常规命令。
- 修改前检查未提交改动，保留并排除与当前任务无关的文件；只有改动重叠、真实合并冲突、远端分叉或需破坏性操作时停止并说明。禁止强推、reset --hard、覆盖用户改动或擅自 stash。网络临时失败可有界重试，同步仍未成功时不修改业务文件。

## 定期维护与推理预算

- arXiv 每日采集继续由 GitHub Actions 执行，游标与可恢复进度必须随候选账本保存，不能依赖 runner 的临时磁盘跨运行续传。
- 会议在每周末检查官方录用列表；模型系列在每月 1 日独立核查。两项维护分别记录结果和提交，失败项保留待重试。
- 本地论文模型共用持久化时间预算：累计推理运行 7200 秒后休息 600 秒，只按时长限制，不以论文数量强制休息。达到预算后不启动新请求，已在途请求完成后开始完整休息；并发请求的时间不能重复累加。
- 本项目共享模型请求统一从同一 WSL/Linux 环境发起；Windows 原生调用明确拒绝并引导到 WSL，避免两套不互通的文件锁与进程编号破坏共享时间预算。
- 工作流实施计划和本地回归测试沿用忽略的 build/reports/；不得修改 README.md 或提交测试文件。

## 新特性测试文件仅限本地

- 可以为新特性创建和运行本地测试，但所有为测试新特性而新增或修改的测试相关文件都必须仅保留在本地，特性完成后进行删除
- 不得暂存、提交、推送、上传这些文件，也不得将其包含在 Pull Request、补丁或任何其他发送到远程/云端的变更中。
- 测试相关文件包括但不限于：单元测试、集成测试、端到端测试、临时测试脚本、测试配置、fixture、mock、snapshot、golden file、测试数据及覆盖率产物。

## 按特性拆分本地提交

- 功能实现并验证完成后，应按相互独立的特性拆分为多笔本地 Git commit，避免把无关功能混入同一提交。
- 每笔 commit 只暂存该特性对应的源码、配置、文档或生成产物；提交前必须检查暂存区内容和 `git diff --cached --check`。

## README 保持固定

- 任何更改都不再修改 `README.md`。

## Writings 目录约定

- `content/writings/<slug>/index.md` 是公开文章唯一真源；本地图片只放在同 bundle 的 `assets/`。
- `src/writings/` 只包含文章校验、渲染与发布逻辑；跨主题能力留在 `src/shared/`。
- `docs/writings/` 只保存生成产物，受管范围以 `manifest.json` 为准，不手工编辑受管文件。
- `build/` 只保存本地报告和临时产物，必须保持忽略且不得提交。

## Writings 导入器约定

- `src/writings/importers/` 只负责把外部导出物转换为标准 writing bundle；发布器不得反向依赖 importer。
- 导入计划、私有映射、预览、报告和解压内容只放在已忽略的 `build/notion-import/` 或 `build/reports/`，不得进入 `content/` 或 `docs/`。
- importer 只能通过显式 apply 修改 `content/writings/<slug>/`，不得生成、提交或推送站点产物。

## WeChat Reading 导入器约定

- `src/writings/importers/weread/` 只负责本地微信读书 Markdown 归一化、loopback 模型调用、私有缓存、预览与 CLI 编排。
- 可复用的路径安全、状态和事务逻辑保留在 `src/writings/importers/`；Notion 与 WeChat adapter 不得相互依赖。
- 微信读书计划、原始归一化内容、提示词、模型响应、缓存、预览、状态和报告只放在已忽略的 `build/weread-import/` 或 `build/reports/`。
- 只有显式 apply 可以修改 `content/writings/<slug>/`；adapter 不得直接生成 `docs/`。
