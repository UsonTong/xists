# xists Roadmap

> 本文档是 xists 的长期发展方针。它不是普通 TODO 列表，而是给未来维护者和 agent 使用的决策文档：不仅说明“做什么”，也说明“为什么这样做”“为什么不先做别的”“做到什么程度才算完成”。

---

## 0. 项目定位

xists 是一个本地语义搜索工具，用来解决 GitHub 原生搜索和普通 agent 搜索都不擅长的问题：**根据项目真实用途进行语义搜索**。

GitHub 原生搜索主要依赖：

- repo 名字
- README 字面关键词
- topics / tags
- stars / forks
- 简单文本匹配

这些信号对“语义问题”不够好。例如用户可能问：

- “有没有开源 Firebase 替代品？”
- “有没有本地 LLM serving/runtime？”
- “有没有 Rust 写的现代编辑器？”
- “有没有适合 agent 使用的浏览器自动化工具？”
- “有没有类似 LangChain 但更轻量的工具？”

这些问题通常不是单纯关键词匹配能解决的。一个项目的 README 可能没有写出用户查询中的词，但它确实解决同一个问题；另一个项目可能反复出现查询关键词，却只是教程、demo、wrapper 或无关插件。

xists 的核心思路是：

```text
原始项目资料
  → LLM 总结成结构化 profile
  → 生成适合 embedding 的 search_text
  → 建立本地 embedding index
  → 通过 CLI / API / MCP / 其他形态进行语义搜索
```

换句话说：

> xists 不试图直接搜索 GitHub 原始文本，而是先把项目变成高质量、结构化、可检索的语义记录，再搜索这些记录。

---

## 1. 当前状态与问题

> **状态更新（2026-07-28）**：v0.2.0 至 v0.8.0 的主体路线已完成，随后发布了 `v0.8.1`（project discovery skill）和 `v0.8.2`（identity evidence 与 index 兼容性修复）。`v0.9.0` 收口中文检索与可复现的正式检索基线；当前公开版本为 `0.9.0`。后续发布路线为 `v0.9.x → v0.10.0 → v1.0.0rcN → v1.0.0`。本节以下内容描述 roadmap 制定时（`0.1.0`）的状态，作为决策背景保留。

本 roadmap 制定时版本是 `0.1.0`，处于 demo 状态。

当前项目已经证明了一件事：

> “LLM profile + embedding search” 这条路线是可行的。

但当前代码和文档还带有明显 demo 痕迹，不能直接作为长期稳定架构继续堆功能。

### 1.1 当前已经有的能力

当前项目已经具备：

- GitHub repo ingest
- LLM profile 生成
- embedding index build
- 本地 search
- eval run / inspect
- doctor / stats / records inspect 等辅助命令
- JSON 文件作为 records/index 存储
- 基本 CLI 工作流

这些都是有价值的，不需要推倒重来。

### 1.2 当前最大问题：搜索逻辑过度复杂

`src/xists/search/query.py` 当前约 2100 行，包含大量为了提高 demo eval 分数而逐渐堆出来的启发式逻辑。

典型复杂度包括：

- 大量手工术语集合，例如 type cue、artifact cue、learning resource cue 等
- 针对特定场景的 role mismatch penalty
- metadata multiplier / cap 的多层阈值
- semantic winner 与 rerank winner 的复杂比较逻辑
- phrase match / coverage match / partial match 的复杂组合
- 许多很难判断泛化效果的权重和 magic number

这些逻辑不是完全错误。它们确实可能提升当前 200 个 demo 仓库上的 eval 分数。

但它们有四个长期风险：

1. **不可理解**：维护者很难快速判断某个结果为什么排第一。
2. **不可修改**：改一个权重可能破坏另一个 query family。
3. **不可泛化**：为 200 个 demo repos 调出来的规则，不一定适用于 1k、10k repos。
4. **不可接管**：项目会继续变成“Claude 调参产物”，而不是用户能掌控的工具。

所以第一个大方向不是加功能，而是降低复杂度。

### 1.3 当前第二个问题：数据层不够表达语义

当前很多应该属于 record/profile 的信息，被迫由 query.py 去猜。

例如用户搜索 `vllm` 时，系统应该知道：

- `vllm` 是项目别名或项目名
- 它是 LLM inference / serving runtime
- 它和 `llama.cpp` 接近但不是完全同类使用场景
- 它不是普通 LLM app framework

如果 records 中没有这些结构化信息，搜索算法只能从 repo name、README、topics、summary 中猜。猜不准时，就会继续往 query.py 里加规则。

因此长期方向应该是：

> 把搜索智能尽量前移到 profile/schema/search_text，而不是堆在 ranking algorithm 里。

### 1.4 当前第三个问题：CLI 还没有被当成一等产品体验

xists 当前是 CLI。未来可以有 MCP、skill、Python API、Web API，但这些都不应该削弱 CLI 的重要性。

CLI 对 xists 很关键，因为：

- 个人工具首先需要一个好用的本地入口。
- 数据源维护者需要 CLI 来检查、修复、刷新 records/index。
- agent 集成出问题时，CLI 是最好的复现和调试入口。
- CLI 是所有核心能力的参考实现。

因此 roadmap 必须明确：

> CLI 不是临时壳，不是 debug fallback，而是一等用户体验。

---

## 2. 核心原则

后续任何 agent 或维护者在做功能时，都应该先检查是否符合这些原则。

### 2.1 数据驱动，而不是规则驱动

xists 的质量应该主要来自：

- 原始数据质量
- LLM profile 质量
- search_text 质量
- embedding 模型质量
- 少量可解释的通用 ranking 逻辑

不应该主要来自：

- 为某个 eval case 特别写的规则
- 大量人工术语表
- 多层 magic number
- 很难解释的 rerank 过程

允许保留少量通用规则，例如：

- exact repo/name/alias match 应该强烈提升
- archived/disabled repo 应该降权
- 明确语言不匹配可以轻微降权
- 结果应该暴露 score_breakdown / why

但不应该把 query.py 变成领域知识库。

### 2.2 CLI-first，但不 CLI-only

xists 首先要有优秀 CLI：

- 默认输出适合人读
- `--format json` 适合程序和 agent
- 错误信息有 next steps
- 常用命令短且一致
- help 文档清楚
- 搜索结果有解释
- 数据检查报告可读

但 xists 不应该只有 CLI。后续需要 Python API、MCP server 等集成形态。

正确关系是：

```text
core Python API
  ├── CLI
  ├── MCP server
  ├── skill / agent integration
  └── future API/server
```

CLI 是一等体验，但核心逻辑不应该只能通过 subprocess CLI 调用。

### 2.3 records/index 是用户资产

`records.json` 和 `index.json` 不是随便生成的临时缓存。未来用户可能会：

- 自己长期维护 records
- 分享 records 给别人
- 下载别人维护的 records
- 基于 records 重新 build index
- 合并多个 records

因此必须重视：

- schema version
- index version
- profile prompt version
- embedding input version
- migration / refresh path
- validate / stats / inspect

一旦别人开始维护数据源，破坏 schema 就是在破坏用户资产。

### 2.4 先本地 1k-10k repos，不追求全 GitHub

xists 不是要在 1.0.0 做全 GitHub 搜索。

1.0.0 的合理目标是：

> 在普通开发机上稳定支持 1k-10k repos 的本地语义搜索。

这个规模已经能覆盖大量真实用途：

- GitHub stars top 5k
- AI/LLM tools top 2k
- frontend/devtools top 1k
- 用户自己的 curated repos

不要过早引入复杂数据库、服务化系统或 ANN 引擎。只有当真实使用超过当前架构能力时，再考虑 FAISS/hnswlib/LanceDB/sqlite-vss 等。

### 2.5 Embedding 模型是数据资产的一部分

index 中的向量与 embedding 模型绑死：换模型 = 全部向量作废。因此模型选择不是实现细节，而是数据资产契约的一部分。

规则：

- 每个 release 钦定一个默认 embedding 模型（当前为 `BAAI/bge-m3`），写进文档；更换默认模型视为重大变更，必须说明理由并提供全量 rebuild 指引。
- index 必须永远记录 embedding_model / dimension / embedding_input_version，搜索前必须校验（均已实现，此处固化为原则）。
- 共享 index 的前提是使用方运行同一模型；文档必须说明这一点，并引导模型不匹配的用户改为共享 records + 自行 rebuild。
- 默认模型的选择标准：多语言（中文查询是一等用例）、可本地部署、社区可长期获取。
- 不做多模型并存的抽象层，直到真实需求出现。

### 2.6 MCP 是外壳，不是地基

MCP 很重要，因为它能让 agent 使用 xists。但 MCP 不应该太早成为核心开发目标。

如果搜索架构、schema、CLI、API 不稳定，MCP 只是把不稳定包装给 agent。

正确顺序是：

```text
清理搜索
→ 稳定 schema
→ 数据质量工具
→ 本地规模边界
→ 规模化 ingest 与数据更新
→ Python API + CLI 打磨 + 打包首发
→ MCP / agent 集成
```

---

## 3. 版本路线总览

```text
v0.2.0  清理搜索，夺回控制权                         [已完成]
         ↓
v0.3.0  Schema v2，把智能转移到数据层                 [已完成]
         ↓
v0.4.0  数据质量工具，让数据源可维护                  [已完成]
         ↓
v0.5.x  本地规模与 index 稳定                         [已完成]
         ↓
v0.6.x  规模化 ingest、数据更新与 eval                [已完成]
         ↓
v0.7.x  Python API、CLI、打包与 workspace              [已完成]
         ↓
v0.8.0  MCP / agent 集成                              [已完成]
         ↓
v0.8.1  Project discovery skill                       [已完成]
         ↓
v0.8.2  搜索正确性与索引兼容性补丁                    [已完成]
         ↓
v0.9.0  中文检索与正式检索基线                        [当前公开版本]
         ↓
v0.9.x  中文检索稳定窗口，只接收兼容性修复
         ↓
v0.10.0 PreparedIndex、NumPy 与索引/checkpoint 架构
         ↓
v1.0.0rc1 / rc2  冻结 API、CLI、schema 与发布证据
         ↓
v1.0.0  稳定发布
```

### 3.1 为什么不把本轮整改做成一个版本

本轮问题同时涉及排序正确性、中文 query analysis、检索评测、运行性能、持久化格式、checkpoint、CLI 拆分和 CI。把它们合并到一个版本会让行为变化、性能变化和格式变化互相干扰，也无法在回归时快速定位原因。

因此采用以下切分原则：

- **patch（`0.x.y`）**：修复现有错误，不增加新的数据迁移要求，不改变索引持久化格式。
- **minor（`0.x.0`）**：增加用户可感知能力、改变默认搜索行为、引入新运行时架构，或要求重建索引。
- **release candidate（`1.0.0rcN`）**：冻结 1.0 契约并验证发布证据，不再加入计划外功能。
- **major（`1.0.0`）**：开始正式承担 API、CLI JSON、record schema 和 index 生命周期的兼容性承诺。

软件版本与数据协议版本必须分开：

- `xists 0.10.0` 是软件发布版本。
- `INDEX_VERSION = 4`（如发生）是索引文件协议版本。
- 只引入内存态 `PreparedIndex` 而不改变磁盘格式时，不得无理由提升 `INDEX_VERSION`。
- 改变磁盘结构时必须提升 `INDEX_VERSION`，并提供兼容读取、转换工具或明确的 rebuild 指引。

### 3.2 执行规格约定

v0.5.0 / v0.6.0 的历史执行规格见 §11 / §12；v0.7.0 与 v0.8.0 的历史执行规格见 §13 / §14。自 2026-07-26 起，新的整改与发布执行规格以 §15 为准。

每个版本都必须有明确目标和范围、明确不做什么、对应测试和评测证据、可执行验收标准、独立 release commit/tag，以及 staged 文件清单审查。

行为修改必须先有回归测试；正确性、性能、存储格式和纯重构不得混在同一个功能提交中。若 §15 与开工时的真实代码状态不符，先更新现状盘点和验收标准，不得按照过期假设盲目实现。

# v0.2.0 — 清理搜索，夺回控制权

## 目标

把当前复杂的 demo ranking 系统重构成一个简单、可解释、可维护的混合搜索基线。

v0.2.0 的核心不是“让 eval 分数更高”，而是：

> 让维护者能理解和控制搜索行为。

## 为什么 v0.2.0 必须先做这个

如果不先清理 query.py，后续所有功能都会建立在复杂、脆弱、难以理解的 ranking 系统上。

例如：

- 新 schema 字段不知道该怎样接入旧 ranking。
- MCP 返回的结果难以解释。
- CLI explain 模式会暴露一堆难以理解的内部权重。
- 每次扩数据集都可能触发更多调参。

所以第一步必须是降低搜索核心复杂度。

## 目标架构

搜索改为两阶段：

```text
query
  → identity search
      repo_id / name / aliases exact match
      if matched, boost/pin to top

  → semantic search
      embed query
      cosine similarity against index vectors
      apply lightweight metadata adjustments
      return ranked results with explanations
```

注意：identity match 不应该完全替代 semantic search。即使用户搜 `vllm` 命中 vLLM，也可以继续返回类似项目，例如 llama.cpp、text-generation-inference 等，方便用户探索替代项。

## 保留逻辑

保留以下通用、可解释逻辑：

- cosine similarity
- confidence bucket
- repo_id/name exact match
- aliases exact match（如果当前 schema 还没有 aliases，可先支持 fallback）
- archived/disabled penalty
- 简单 language match/mismatch
- score_breakdown
- why / diagnostics 简化版

## 删除或大幅简化逻辑

删除或重写：

- 大量人工术语集合
- `_role_mismatch_penalty`
- `_metadata_multiplier`
- `_metadata_bonus_cap`
- `_profile_phrase_match`
- `_metadata_match_strength` 的复杂规则
- `_rerank_results` 中 semantic winner vs rerank winner 的复杂博弈

如果某些逻辑确实需要保留，必须满足：

1. 能用一句话解释。
2. 不针对具体项目或具体 eval case。
3. 有测试覆盖。
4. 不引入大量 magic number。

## CLI 体验要求

搜索 CLI 是 v0.2.0 的重点之一。

命令：

```bash
xists search "open source firebase alternative" --index demo-index.json
xists search "open source firebase alternative" --index demo-index.json --format text
xists search "open source firebase alternative" --index demo-index.json --format json
```

默认输出应该适合人读。JSON 输出适合程序。

text 输出应该包含：

- query
- result count
- repo id
- URL（如果有）
- confidence
- score
- summary
- why

示例方向：

```text
query: open source firebase alternative
results: 5

1. supabase/supabase
   url: https://github.com/supabase/supabase
   confidence: high
   score: 0.72
   summary: Open source Firebase alternative built on Postgres.
   why: strong semantic match; matched project identity text; popular repository

2. appwrite/appwrite
   url: https://github.com/appwrite/appwrite
   confidence: high
   score: 0.68
   summary: Backend server for web, mobile, and Flutter developers.
   why: strong semantic match; similar backend-as-a-service profile
```

JSON 输出应该稳定，不要为了可读性牺牲机器消费。

## Eval 处理

v0.2.0 不追求保持当前 88%+ exact top-1。

原因：当前分数可能部分来自过拟合的启发式规则。

v0.2.0 应建立 smoke eval：

- 20-30 个 case
- 覆盖基本意图，而不是覆盖所有历史 miss
- 重点是防止明显退化

case 类型：

- exact name: `vllm`, `supabase`, `yt-dlp`
- functional: `local llm serving`, `open source firebase alternative`
- language/ecosystem: `rust code editor`, `python web framework`
- ambiguous/exploratory: `workflow automation`, `browser automation for agents`
- weak/no-result: 查询没有明显匹配时不要乱给 high confidence

## 不做什么

v0.2.0 不做：

- 新 schema 大改
- MCP server
- 大规模性能优化
- 新数据源
- Web UI

## 验收标准

v0.2.0 完成必须满足：

- `query.py` 大幅减少复杂度，目标约 400-600 行
- 搜索逻辑能被 README/docs 简单解释
- exact repo/name 查询稳定命中
- 搜索结果仍包含 score/confidence/why
- CLI text 输出可读
- CLI JSON 输出稳定
- smoke eval 通过
- 没有为单个 eval case 添加特殊规则

---

# v0.3.0 — Schema v2，数据驱动搜索

## 目标

让 records/profile 更能表达项目语义，从而减少 ranking algorithm 的负担。

一句话：

> 让数据更聪明，让搜索代码更笨。

## 为什么 v0.3.0 做 schema

v0.2.0 清理后，搜索算法会变简单。此时如果搜索质量下降，不应该第一反应是把复杂规则加回来，而应该提高 profile/search_text 质量。

很多搜索问题本质上是数据表达不足：

- 用户搜项目名，records 没有 aliases。
- 用户搜替代品，records 没有 replaces/related_projects。
- 用户搜工具类型，records 没有 project_type。
- embedding 输入混合了人类摘要和检索文本，导致不够精准。

因此 v0.3.0 要升级 schema。

## Schema v2 方向

新增或稳定字段：

```json
{
  "schema_version": 2,
  "repo_id": "vllm-project/vllm",
  "url": "https://github.com/vllm-project/vllm",
  "name": "vllm",
  "source": "github",
  "source_metadata": {},
  "llm_profile": {
    "summary": "A high-throughput LLM inference and serving engine.",
    "use_cases": [],
    "capabilities": [],
    "not_for": [],
    "aliases": ["vllm"],
    "project_type": "runtime",
    "ecosystem": ["python", "llm"],
    "replaces": [],
    "related_projects": ["ggml-org/llama.cpp"],
    "search_text": "local LLM serving runtime, high throughput inference server, OpenAI-compatible model serving...",
    "confidence": "high",
    "abstained": false
  }
}
```

字段说明：

- `aliases`: 用于 identity search。解决 embedding 不擅长精确实体的问题。
- `project_type`: 表达项目类型，例如 library/tool/framework/platform/runtime/tutorial/collection/app/service/dataset。
- `ecosystem`: 表达语言、技术生态或领域，例如 python/javascript/rust/llm/web/devtools。
- `replaces`: 表示替代、继承、接替关系，例如 yt-dlp 替代 youtube-dl。
- `related_projects`: 表示相近但不一定替代的项目。
- `search_text`: 专门给 embedding 用的检索文本。

## search_text 的重要性

当前 profile 同时给人看、给 embedding 吃。这是不理想的。

人类摘要应该简洁、准确、少重复。

embedding 文本应该：

- 包含同义表达
- 包含用户可能搜索的自然语言短语
- 可以适度冗余
- 可以明确项目不容易从 README 看出的用途
- 不需要像 summary 那样优雅

示例：

```json
{
  "summary": "A high-throughput LLM inference and serving engine.",
  "search_text": "local LLM serving runtime, high throughput inference server, OpenAI-compatible model serving, production inference engine, vLLM, model serving for transformers"
}
```

这样可以减少 query.py 中 phrase matching 的必要性。

## Versioning

v0.3.0 必须明确以下版本：

- `RECORD_SCHEMA_VERSION`
- `PROFILE_PROMPT_VERSION`
- `EMBEDDING_INPUT_VERSION`
- `INDEX_VERSION`

它们含义不同：

- record schema 变了，不一定需要重新抓 GitHub。
- profile prompt 变了，需要重新 profile。
- embedding input 变了，需要 rebuild index。
- index format 变了，可能需要 rebuild index 或 migration。

## 旧数据处理

不能让旧 records 静默坏掉。

至少需要：

```bash
xists records validate --records records.json
xists profile refresh --records records.json --output records-v2.json
xists index build --records records-v2.json --output index.json
```

如果完整 migration 暂时不做，也必须给出清晰错误和 next steps。

错误示例：

```text
records schema version is 1, but xists expects version 2 for this command.

Next steps:
  1. Refresh profiles:
     xists profile refresh --records records.json --output records-v2.json
  2. Rebuild index:
     xists index build --records records-v2.json --output index.json
```

## CLI 体验要求

新增或增强：

```bash
xists records validate --records records.json
xists records inspect --records records.json --repo vllm
xists profile refresh --records records.json --output records-v2.json
```

`records inspect` 应该显示：

- repo_id
- summary
- aliases
- project_type
- ecosystem
- search_text preview
- confidence
- abstained

## 不做什么

v0.3.0 不做：

- 多数据源完整实现
- MCP server
- ANN index
- Web UI

## 验收标准

- Schema v2 文档完成
- profile prompt 能生成新增字段
- search 使用 aliases/search_text
- 旧 records 有明确 refresh/migration 路径
- index rebuild 逻辑正确处理 embedding input version
- CLI 能解释 schema mismatch
- 测试覆盖 schema loading/validation/search_text indexing

---

# v0.4.0 — 数据质量工具，让数据源可维护

## 目标

让用户和社区可以制作、检查、修复、分享数据源。

如果 xists 的愿景是“任何人都可以维护一份数据源”，那仅仅支持 ingest 是不够的。用户还需要知道这份数据源质量如何。

## 为什么这是独立版本

数据质量工具不如 MCP 吸引人，但它决定 xists 是否能从个人 demo 变成可复用工具。

没有数据质量工具时：

- 用户不知道 records 是否缺字段。
- 用户不知道 profile 是否大量 abstained。
- 用户不知道 index 是否 stale。
- 用户不知道别人的 records 是否可靠。
- agent 搜索失败时难以判断是 query 问题、profile 问题还是 index 问题。

因此 v0.4.0 要把“维护数据源”变成一等工作流。

## Artifact 分层

明确三种 artifact：

```text
repos.txt       # 可审查项目列表，只包含 repo ids / urls
records.json   # 可复用语义数据，包含原始 metadata + LLM profile + search_text
index.json     # 与 embedding model 绑定的可搜索向量索引
```

用途：

- 分享 `repos.txt`: 适合让别人自己重新 ingest/profile。
- 分享 `records.json`: 适合复用 LLM 成本，别人只需 rebuild index。
- 分享 `index.json`: 下载即搜，但绑定 embedding model。

这三层必须在文档中解释清楚。

## 数据检查命令

目标命令：

```bash
xists records validate --records records.json
xists records stats --records records.json
xists records inspect --records records.json --repo supabase
xists index verify --records records.json --index index.json
xists index stats --index index.json
```

检查项目包括：

- schema version
- duplicate repo_id
- missing repo_id/url/name
- missing summary
- missing aliases
- missing search_text
- search_text too short
- profile abstained
- low confidence profile
- archived/disabled ratio
- missing README ratio
- index model mismatch
- embedding dimension mismatch
- stale fingerprint
- index contains records not present in records.json
- records not present in index.json

## CLI 输出要求

默认输出必须适合人读，而不是巨大 JSON。

示例：

```text
records: data/ai-repos.json
schema: 2
repos: 1280

quality:
  ok: 1194
  missing_search_text: 32
  missing_aliases: 418
  profile_abstained: 21
  low_confidence: 44
  archived: 67
  duplicates: 0

index:
  status: stale
  stale_vectors: 32
  missing_vectors: 7

next steps:
  - run xists profile refresh for records missing search_text
  - rebuild index after refreshing profiles
  - review 21 abstained profiles manually
```

JSON 模式：

```bash
xists records validate --records records.json --format json
```

必须稳定，方便 CI 或 agent 使用。

## 维护者工作流

v0.4.0 后，维护一份数据源应该像这样：

```bash
xists records validate --records records.json
xists records stats --records records.json
xists profile refresh --records records.json --only-missing-search-text --output records.new.json
xists index build --records records.new.json --output index.json
xists index verify --records records.new.json --index index.json
```

## 不做什么

v0.4.0 不做：

- 大规模 ANN
- MCP
- Web UI
- 自动社区 registry

## 验收标准

- records validate 可发现常见数据问题
- records stats 输出有用概览
- index verify 能发现 stale/mismatch
- text 输出适合人读
- JSON 输出适合自动化
- 文档解释 repos/records/index 三层 artifact
- 数据维护工作流可跑通

---

# v0.5.0 — 本地规模与 index 稳定

> 本版本的逐任务执行规格见 §11。

## 目标

明确并验证 xists 在本地文件架构下的可用规模。

1.0.0 前的目标是：

> 稳定支持 1k-10k repos 的本地语义搜索。

## 为什么不直接做 100k/1M

支持全 GitHub 级别搜索会引入很多复杂度：

- ANN index
- 数据库或对象存储
- 分片
- 增量更新
- 后台服务
- 资源管理
- 部署问题

这些不是当前个人工具阶段最重要的问题。

当前真正需要的是：

- 搜索体验稳定
- schema 稳定
- CLI 好用
- 1k-10k repos 足够快

## 性能边界

目标：

- 1k repos：搜索近似即时
- 10k repos：仍可交互使用
- validate/stats/index verify 在可接受时间内完成
- 不需要启动数据库服务
- 不强制引入 FAISS/hnswlib 等新依赖

继续使用 JSON + numpy brute-force vector search，直到实际数据证明不够用。

## index 格式要求

index.json 必须包含：

- index_version
- record_schema_version
- embedding_model
- embedding_base_url 或 provider 信息
- embedding_input_version
- dimension
- built_at
- record_count
- skipped records
- per-vector fingerprint

搜索前必须检查：

- index embedding model 是否与当前配置一致
- dimension 是否一致
- embedding input version 是否兼容
- vectors 是否完整

## CLI 体验要求

```bash
xists index stats --index index.json
xists index verify --records records.json --index index.json
xists search "local llm serving" --index index.json --top-k 5
```

`index stats` 应展示：

- vector count
- model
- dimension
- built_at
- record count
- skipped count
- stale/missing 状态（如果传 records）
- estimated memory footprint（可选）

## 何时考虑 ANN

不要在 v0.5.0 默认引入 ANN。

只有当满足以下条件时，才开新 roadmap：

- 真实数据超过 50k-100k repos
- brute-force 明显影响交互体验
- 用户愿意接受额外依赖
- index 格式已有稳定抽象

## 验收标准

- 1k-10k repos 搜索可用
- index stats/verify 清晰
- index mismatch 报错可行动
- 不引入强制数据库依赖
- benchmark 或 smoke performance test 有记录

---

# v0.6.0 — 规模化 ingest 与数据更新

> 本版本的逐任务执行规格见 §12。

## 目标

让 xists 能可靠地**生产和维护**千级到万级规模的数据源。

v0.5.0 验证的是消费侧（搜索 1k-10k repos 是否够快），v0.6.0 验证生产侧：

> ingest + profile 上万个 repos，在普通开发环境下是否现实可行。

## 为什么需要这个版本

roadmap 此前只回答了"1k-10k repos 搜起来怎么样"，没有回答"1k-10k 条 records 怎么生产出来"。现实约束是：

- GitHub API 认证限额约 5k 请求/小时，2 万 repos 的 ingest 必然跨小时运行并遭遇限流。
- 即使使用本地 LLM，2 万个 profile 也意味着数十小时连续运行。
- 任何长任务中途都会失败：网络中断、单个 repo 异常、endpoint 超时。

没有断点续跑和增量刷新，规模化数据源在工程上不成立，"用户维护自己的数据源"的愿景也只停留在几百个 repo 的玩具规模。

## 必须具备的能力

- **resume/checkpoint**：ingest 和 profile refresh 都必须可中断、可续跑，重跑时跳过已完成条目。
- **限流与退避**：GitHub API 限流时自动等待重试，而不是失败退出。
- **增量刷新**：基于 fingerprint 只重新处理发生变化的 repo（metadata 变更、profile prompt 升级、embedding input 升级）。
- **成本/耗时预估**：`--dry-run` 报告将处理多少 repo、跳过多少、预计调用量，让用户在开跑前知道代价。
- **失败隔离**：单个 repo 失败记录到报告中，不中断整批任务；结束时可只重试失败项。

## 规模压力实验（阶梯）

用逐级放大的真实 corpus 验证 ranking 是否稳定：

```text
200（现有 demo）→ 2k → 10k → 20k
```

原则：

- **分层选取，刻意加噪声**：每一级都应包含 tutorial、awesome-list、废弃 fork、archived repo 等干扰项。规模的价值在于干扰项密度和同名冲突，不在数字本身。随便抓 top stars 反而质量偏高、压力不足。
- **每级先评估再爬升**：跑 eval、观察 ranking 行为变化（identity 冲突、confidence 是否虚高、语义近邻拥挤时的表现），确认没有系统性问题再进入下一级。
- **20k 是压力测试，不是产品承诺**：产品目标仍是 1k-10k；20k 提供 2x 余量，验证架构不在边界上刚好及格。

## Eval 方法论升级

corpus 变大后，200 repos 时代的 eval 方式不再适用：

- `exact top-1` 指标改为 **recall@k + LLM judge**（judge 基建已存在）。
- query 集按类型分层扩展：exact name / functional / language-ecosystem / ambiguous / weak-no-result。
- **query 集必须包含中文 case**。本文档 §0 的示例查询全部是中文，bge-m3 也是多语言模型，但历史 eval cases 全是英文——维护者自己最真实的使用方式从未被测过。每个类型分层至少配一个中文 case。
- 必须包含"正确答案不在 corpus 中"的 case，验证系统不给虚高 confidence。
- **红线不变**：禁止为 eval 分数往 query.py 加规则。规模化 eval 暴露的问题，第一响应永远是改 profile/search_text/schema。这个实验正是检验"数据驱动"原则能否扛住压力的试金石。

## CLI 体验要求

```bash
xists ingest github --input repos.txt --output records.json --resume
xists profile refresh --records records.json --resume --dry-run
xists profile refresh --records records.json --only-changed --output records.json
```

要求：

- 长任务有进度输出（已完成/总数/失败数）。
- 中断后重跑同一命令即可续跑，不需要用户手工计算剩余清单。
- `--dry-run` 输出适合人读，且有 `--format json`。
- 失败报告可行动：哪些 repo 失败、为什么、如何只重试失败项。

## 不做什么

v0.6.0 不做：

- daemon / webhook / 实时更新
- 任务队列系统或数据库依赖
- 分布式 / 并行集群
- 自动社区 registry

## 验收标准

- 能在可中断的普通开发环境下生产 2 万级 corpus：一条命令启动，中断后续跑，最终完成。
- 增量刷新只处理变化的 repo，有测试覆盖（fingerprint 未变则跳过）。
- `--dry-run` 预估可用且有测试。
- 单 repo 失败不中断整批，失败报告可读、可重试。
- 阶梯实验的操作手册（`docs/scaling-experiment.md`，见 §12 T6）完成。实验本身的执行与各级 eval 结果记录**不是 v0.6.0 的收版条件**，而是 v1.0.0 的发布门槛（见 v1.0.0"发布前置"节）——此处与 §12 T6 保持一致。
- 没有为 eval 分数向 query.py 添加规则。

---

# v0.7.0 — 稳定 Python API + 优秀 CLI + 打包首发

## 目标

让 xists 既可以被人舒服地用 CLI 使用，也可以被程序稳定调用。

这一版是 agent 集成前的关键地基。

## 为什么 API 和 CLI 放在同一版本

CLI 和 API 不应该互相割裂。

CLI 是用户体验，API 是集成基础。两者应该共享同一套核心逻辑和数据结构。

如果先做 MCP 而没有稳定 API，MCP 可能会通过 subprocess 调 CLI，这可以作为临时实现，但不是长期好架构。

## Python API 方向

目标 API：

```python
from xists import load_index, search

index = load_index("index.json")
result = search(
    "open source firebase alternative",
    index,
    top_k=5,
)
```

返回结构应接近 CLI JSON：

```json
{
  "query": "open source firebase alternative",
  "results": [
    {
      "repo_id": "supabase/supabase",
      "url": "https://github.com/supabase/supabase",
      "summary": "Open source Firebase alternative built on Postgres.",
      "confidence": "high",
      "score": 0.72,
      "why": ["strong semantic match", "matches backend-as-a-service profile"],
      "best_for": ["auth", "database", "realtime apps"],
      "not_for": ["simple static sites"]
    }
  ]
}
```

注意：搜索必须先 embed query，上面的示例签名省略了 embedding 配置的来源。实际 API 设计必须显式支持注入 embedding 配置或客户端（例如 `search(query, index, top_k=5, embedder=...)` 或显式的 config 参数），不允许在 import 或调用时隐式读取 `.env` 之类的全局状态——库的调用方必须能完全控制 endpoint 从哪来。

## Agent-friendly JSON

agent 不需要只拿到 repo_id 和 score。

agent 需要的是候选项目理解包：

- repo_id
- url
- summary
- why
- best_for
- not_for
- confidence
- metadata evidence
- maybe related/replaces

这样 agent 才能判断：

- 是否要打开该 repo
- 是否要继续搜索
- 是否要比较多个候选
- 是否应该告诉用户没有明显结果

## CLI 打磨方向

稳定命令族：

```bash
xists doctor
xists ingest github
xists profile refresh
xists records validate
xists records stats
xists records inspect
xists index build
xists index stats
xists index verify
xists search
xists eval run
xists eval inspect
```

CLI 要求：

- 常用路径短
- 子命令命名一致
- 默认输出适合人读
- `--format json` 稳定
- 错误有 next steps
- `--help` 能解释用途
- README 有 happy path
- docs 有 troubleshooting

## Explain 模式

增加或完善：

```bash
xists search "vllm" --explain
```

输出应解释：

- 是否命中 repo/name/alias
- semantic score
- metadata adjustment
- 使用了哪些 profile 字段
- 为什么 top result 排前面
- 是否有可能是 ambiguous query

## Packaging 与首发

v0.7.0 完成时应**首发 PyPI**（0.x 语义版本），不等 v1.0.0。理由：

- 发包本身是获取真实反馈的手段，waterfall 走到 1.0 才见用户，风险大于任何技术缺口。
- 下一版本（MCP）的接入文档需要包已可 `pip install` 才能真实验证。

必须完成：

- `pyproject.toml` 补全 `license`、`classifiers`、`urls`。
- 仓库包含 LICENSE 文件。License 由维护者在 v0.7.0 开工前选定并记录在本文档变更记录中，执行 agent 不得代为选择（License 选择与"分享 records"愿景及 §9 的 LLM 派生内容责任相关，是产品决策）。
- CI（如 GitHub Actions）在 push/PR 上运行 `pytest`。
- 大型 demo artifacts（`demo-records.json`、`demo-index.json`）从 git 仓库移出，改为 GitHub Release asset 提供下载，解决冷启动的第一份数据问题。**发布 asset 前必须先把 demo 数据刷新到当前 schema / embedding input 版本**（当前仓库内的 artifacts 是 schema v1 过期产物，validate/verify 必然失败），asset 必须通过 `records validate` 与 `index verify` 才允许发布——冷启动给出的第一份数据不能是坏的。
- README 覆盖从 `pip install xists` 到第一次搜索成功的完整路径，包括 embedding endpoint 的最低成本配置方案。
- §9"安全与隐私"的三个问题（token 处理、数据外发、共享数据源的内容责任）的答案写入用户文档。这是 §9 明文规定的发包前置条件，列入本清单防止漏项。
- 发布步骤固化进 `docs/release.md`：构建、版本一致性检查、打 tag、上传 PyPI（或 CI publish workflow），使发布流程可凭文档重复执行（对应 §5.9"PyPI 发布流程可重复"）。
- PyPI 包名 `xists` 已确认可用（2026-07 查证未被占用），首发时占用。抢注风险与提前占位选项见 §8。

## 不做什么

v0.7.0 不做：

- MCP server 作为主任务
- Web UI
- 远程服务化

## 验收标准

- Python API 有测试
- CLI 与 API 输出结构一致或可映射
- CLI text/json 稳定
- search explain 有用
- agent-friendly JSON 字段完整
- 文档覆盖 CLI 主流程和 API 示例
- 包已发布到 PyPI，全新虚拟环境中 `pip install xists` 后 happy path 可跑通
- CI 在主分支上稳定通过

---

# v0.8.0 — MCP / agent 集成

## 目标

让 agent 可以直接使用 xists 搜索项目。

但要注意：

> MCP 是集成形态，不是 xists 的核心价值本身。

xists 的核心价值仍然是高质量语义项目索引和优秀 CLI/API。

## 为什么放到 v0.8.0

如果太早做 MCP，会出现这些问题：

- 返回结果不稳定
- schema 还会变
- CLI 还不能复现 MCP 行为
- API 还没稳定，只能 subprocess CLI
- agent 拿到的信息不足，只是一堆链接

等 v0.2-v0.7 做完后，MCP 才能成为稳定包装。

## 依赖策略

MCP server 的实现允许引入 MCP SDK（如 `mcp` 包），但必须作为 optional extra 安装：

```bash
pip install "xists[mcp]"
```

规则：

- 核心包运行时依赖保持 numpy-only，不因 MCP 而改变——这是对 §8"零重依赖"原则的正式裁决：该原则约束的是核心包，MCP 作为可选集成形态例外。
- 未安装 extra 时，CLI / Python API 必须完整可用；MCP 入口给出可行动错误，错误信息包含 `pip install "xists[mcp]"`。
- 不允许为绕开 SDK 依赖而手写 stdio JSON-RPC 协议实现——那是把维护成本换到更糟的地方。

## MCP tools 方向

可能的 tools：

- `search_projects(query, top_k)`
- `inspect_project(repo_id)`
- `index_stats()`
- `validate_index()`（可选）

`search_projects` 返回 agent-friendly JSON。

`inspect_project` 用于 agent 对某个 repo 做进一步了解，不需要再读整个 records/index。

## Agent 使用原则

xists 给 agent 的不是“搜索结果链接列表”，而是“候选项目理解包”。

每个结果应尽可能包含：

- repo_id
- url
- summary
- use_cases
- capabilities
- best_for
- not_for
- why
- confidence
- related/replaces

## CLI 仍然是一等入口

MCP 输出应该能用 CLI 复现：

```bash
xists search "browser automation for agents" --format json
```

这样当 agent 行为异常时，用户可以直接在终端 debug。

## 文档要求

需要说明：

- 如何启动 MCP server
- 如何配置 Claude Code / Cursor / Cline 等
- MCP tool 返回什么
- 如何用 CLI 复现 MCP 搜索
- 如何更新 index

## 验收标准

- MCP server 可运行
- agent 能完成基本搜索任务
- CLI 和 MCP 输出一致
- 文档有接入示例
- 不破坏 CLI-first 使用体验
- MCP 使用 core API，而不是长期依赖 subprocess CLI

---

# v1.0.0 — 稳定发布

## 发布含义

1.0.0 不代表 xists 已经是大型搜索平台。

1.0.0 代表：

> 用户可以开始认真维护自己的 xists 数据源，并相信 CLI、schema、index、API、MCP 在一段时间内是稳定的。

## 必须满足

### 搜索

- 搜索架构简洁
- query.py 可维护
- 结果可解释
- exact name / alias 查询可靠
- semantic search 可用
- 不依赖大量场景硬编码

### Schema / 数据

- Schema v2 稳定
- versioning 清楚
- profile refresh/migration 路径清楚
- records validate/stats/inspect 可用
- index verify/stats 可用

### CLI

- CLI 是一等体验
- 默认输出适合人读
- JSON 输出稳定
- 错误信息可行动
- help/docs 完整
- 常用工作流顺畅

### API / 集成

- Python API 稳定
- MCP server 可用
- agent-friendly JSON 稳定
- CLI 可复现 MCP 行为

### 规模

- 本地 1k-10k repos 搜索可靠
- 不强制数据库服务
- index 格式稳定

### 发布前置（证据与节奏）

- 阶梯实验至少完成 2k 与 10k 两级真实 corpus（按 `docs/scaling-experiment.md` 执行），各级 eval 无系统性回归、失败已归因，结果记录在 docs。这是"本地 1k-10k repos 搜索可靠"声明的证据——缺此不得发布 1.0.0。20k 级是可选压力测试，不是发布门槛。
- PyPI 首发（v0.7.0）上线距 1.0.0 发布至少 4 周，期间收到的数据兼容性 / schema 相关问题已处理，或有明确记录的处置决定。稳定承诺不能没有经过任何真实使用期就做出。

## 稳定性与废弃政策（随 1.0.0 生效）

1.0.0 承诺的"稳定"必须有可检验的定义。发布时把本政策原样写进用户文档：

- **语义化版本**：破坏 record schema、index 格式、Python API 签名或 CLI JSON 输出结构的变更，必须升 major 版本。
- **废弃流程**：CLI 参数、命令、API 入口废弃前，至少保留一个 minor 版本并输出 deprecation warning，warning 中给出替代写法。
- **schema 演进**：schema v3 及以后必须附 migration 命令或明确的 refresh/rebuild 指引；validate 必须能识别所有历史 schema 版本并给出 next steps。
- **默认 embedding 模型变更**视为重大变更（见 §2.5），必须提供全量 rebuild 指引。
- **兼容窗口**：新 minor 版本必须能读取上一个 minor 版本产生的 records/index，或明确拒绝并给出修复命令；静默误读视为 bug。

## 1.0.0 文档必须覆盖

- xists 是什么 / 不是什么
- 稳定性与废弃政策（上一节原文）
- 为什么 GitHub 原生搜索不够
- LLM profile + embedding 的工作原理
- repos/records/index 三层 artifact 区别
- 如何制作数据源
- 如何刷新 profile
- 如何 build index
- 如何 search
- 如何 validate records/index
- 如何接入 agent
- 如何升级 schema/index
- 常见错误和 next steps

---

## 4. 1.0.0 前暂时不做

这些不是永远不做，而是 1.0.0 前不应作为主线。

### 不做 Web UI

理由：

- 当前核心是本地语义搜索管道。
- Web UI 会引入前端、状态管理、部署等额外复杂度。
- CLI/API/MCP 更贴近当前个人工具和 agent 工作流。

### 不做实时索引更新

理由：

- 批量 ingest + profile refresh + index build 已经足够。
- 实时更新需要 daemon/webhook/scheduler，复杂度高。

### 不做用户系统 / 多租户

理由：

- xists 首先是个人本地工具。
- 多用户会引入权限、存储、部署、安全等问题。

### 不追求全 GitHub 规模

理由：

- 当前核心价值可以在 curated 1k-10k repos 上成立。
- 全 GitHub 规模会过早引入分布式系统复杂度。

### 不为了 eval 分数写特殊规则

理由：

- eval 是回归保护，不是优化目标。
- 如果 eval miss 反映真实数据问题，应优先改 profile/search_text/schema。

### 不把 MCP 放在核心之前

理由：

- MCP 是外壳。
- 核心搜索、schema、CLI、API 不稳定时，MCP 只会放大不稳定。

---

## 5. 详细验收标准（防止 coding agent 走偏）

本节是给后续 coding agent 的硬性验收清单。实现某个版本时，不允许只完成“看起来相关”的代码改动；必须逐项满足对应版本的验收条件。

如果某条验收标准暂时无法完成，必须在提交说明或最终回复中明确写出：

- 哪条没有完成
- 为什么没有完成
- 当前替代方案是什么
- 后续应该如何补齐

不能用模糊表述，例如“基本完成”“大致可用”“后续优化”。

### 5.1 全局 Definition of Done

任何版本、任何功能改动都必须满足以下全局标准。

#### 代码层面

- 代码必须保持简单、可读、局部化。
- 不允许为了修复单个 eval case 增加项目名特判或 query 特判。
- 不允许引入新的大规模术语表，除非 roadmap 明确要求，且有独立文档解释用途。
- 不允许把新的核心逻辑只写在 CLI 层；可复用逻辑应该在模块函数中，CLI 只负责参数解析和格式化输出。
- 不允许让 MCP/API/CLI 各自实现一套不同搜索逻辑。
- 不允许静默忽略 schema/index/version mismatch；必须明确报错或给出 warning + next steps。

#### CLI 层面

每个新增或修改的 CLI 命令必须满足：

- `--help` 能解释它做什么。
- 默认输出适合人在终端阅读。
- 如果该命令可能被脚本或 agent 使用，应提供 `--format json` 或保持现有 JSON 输出稳定。
- 错误信息必须可行动，至少包含：问题是什么、用户下一步应运行什么命令。
- 不允许输出未截断的大型 payload，例如完整 embedding vector、完整 README、完整 records/index。
- 命令命名应与现有结构一致，例如 `records validate`、`index verify`、`profile refresh`，不要随意新增风格不同的命令。

#### 测试层面

每个版本必须至少包含：

- 正常路径测试。
- 失败路径测试。
- CLI 参数或输出测试。
- schema/version/mismatch 相关测试（如果涉及数据格式）。
- 不依赖真实 GitHub、真实 LLM、真实 embedding endpoint 的单元测试。

如果某项功能必须依赖外部服务，应通过 mock、fixture 或 injected function 测试。

#### 文档层面

每个版本完成时必须同步更新：

- README 或 docs 中的用户路径。
- `docs/usage.md` 中对应命令说明。
- `docs/record-schema.md` 或 index/schema 文档（如果涉及格式）。
- `ROADMAP.md` 状态或后续说明（如果实现偏离本路线）。ROADMAP.md 位于仓库根目录；执行 agent 应直接更新它，并在完成报告中单列"偏离 roadmap 的事项"供维护者审阅。

不允许代码行为和文档长期不一致。

#### 数据兼容层面

涉及 records/index/profile 的改动必须明确：

- 是否改变 schema。
- 是否改变 profile prompt。
- 是否改变 embedding input。
- 是否需要重建 index。
- 旧文件遇到新代码时会发生什么。
- 用户应该运行什么命令修复。

### 5.2 v0.2.0 验收标准：清理搜索

v0.2.0 的目标是清理搜索，不是提高 demo eval 分数。coding agent 不允许把它做成“继续调参版本”。

#### 必须完成

- `src/xists/search/query.py` 的复杂度必须显著下降。
  - 目标行数约 400-600 行。
  - 如果超过 700 行，必须解释为什么无法继续拆分。
- 搜索流程必须能用以下结构解释：
  - identity match
  - semantic vector search
  - lightweight metadata adjustment
  - confidence / explanation output
- identity match 必须独立存在，而不是埋在复杂 metadata score 中。
- repo_id/name 查询必须稳定置顶精确匹配结果。
- 如果 schema 暂时没有 aliases，也必须预留 aliases 读取路径；缺失时 fallback 到 repo_id/name。
- 搜索结果必须保留或提供等价字段：
  - `repo_id`
  - `score`
  - `confidence`
  - `score_breakdown` 或等价分数解释
  - `why`
- CLI `xists search ... --format text` 必须可读。
- CLI `xists search ... --format json` 必须稳定。
- smoke eval 必须保留并通过。

#### 必须删除或不再使用

以下逻辑如果继续存在，必须有非常明确的新理由；默认应删除或大幅简化：

- 大量领域术语集合。
- `_role_mismatch_penalty` 的场景硬编码。
- `_metadata_multiplier` 多层阈值。
- `_metadata_bonus_cap` 多层阈值。
- `_rerank_results` 中复杂 winner/challenger 博弈。
- 为某类具体项目写的特殊 ranking 分支。

#### 禁止行为

- 禁止为了让旧 eval 继续 88%+ exact top-1 而恢复复杂规则。
- 禁止新增新的大型 cue set 来替代旧 cue set。
- 禁止让 exact match 和 semantic score 混在一个不可解释的大分数函数里。
- 禁止只改测试不改搜索结构。

#### 测试要求

至少覆盖：

- 精确 repo_id 查询命中。
- repo name 查询命中。
- alias 查询路径（即使 fixture 手动构造 alias）。
- functional query 返回语义相关结果。
- archived repo 降权。
- top_k=0 或空 index 的行为。
- index/model/dimension mismatch 报错。
- text/json CLI 输出中包含关键字段。

#### 文档要求

必须更新文档解释：

- 当前搜索如何工作。
- identity match 和 semantic search 的关系。
- score/confidence/why 含义。
- eval 分数下降为什么可以接受。

### 5.3 v0.3.0 验收标准：Schema v2 / search_text

v0.3.0 的目标是把搜索智能转移到数据层。coding agent 不允许只加字段但不让搜索实际使用这些字段。

#### 必须完成

- 明确定义 `RECORD_SCHEMA_VERSION = 2` 或等价机制。
- `llm_profile` 或 record 中必须支持：
  - `aliases`
  - `project_type`
  - `ecosystem`
  - `replaces`
  - `related_projects`
  - `search_text`
- `PROFILE_PROMPT_VERSION` 必须升级。
- profile prompt 必须明确要求 LLM 生成这些字段，并要求不确定时使用空数组或 `unknown`，不能编造事实。
- embedding input 必须优先使用 `search_text`。
- 如果 `search_text` 缺失，必须有明确 fallback，并在 validate 中报告。
- search 必须实际使用 aliases 进行 identity match。
- docs 必须说明每个新增字段的含义和允许值。

#### 兼容要求

旧 records 遇到新代码时必须满足其一：

1. 仍可搜索，但 validate 明确报告 schema 旧、字段缺失、建议 refresh。
2. 对要求 Schema v2 的命令明确失败，并给出 next steps。

不能静默把旧 records 当新 records 使用。

#### CLI 要求

以下命令或等价能力必须存在：

```bash
xists records validate --records records.json
xists records inspect --records records.json --repo vllm
xists profile refresh --records records.json --output records-v2.json
```

如果 `profile refresh` 暂时不能完整实现，必须至少提供明确的替代刷新流程，并在 roadmap 中说明。

`records inspect` 默认输出必须显示：

- repo_id
- summary
- aliases
- project_type
- ecosystem
- search_text preview
- confidence
- abstained

#### 测试要求

至少覆盖：

- 新 profile prompt 输出解析。
- 缺字段 profile 的容错。
- `search_text` 被用于 embedding input。
- aliases 能命中 identity search。
- schema version mismatch。
- profile refresh 或替代路径不会破坏原始 GitHub metadata。

#### 禁止行为

- 禁止只在 docs 里写 Schema v2，但代码仍按旧字段工作。
- 禁止让 LLM 自由输出任意 schema；必须 validate。
- 禁止 schema mismatch 静默通过。
- 禁止把 `search_text` 当成人类摘要展示的唯一来源；summary 和 search_text 目的不同。

### 5.4 v0.4.0 验收标准：数据质量工具

v0.4.0 的目标是让数据源可维护。coding agent 不允许只做一个浅层 JSON schema check。

#### 必须完成

`records validate` 必须至少检查：

- JSON 顶层结构是否正确。
- schema version 是否存在且兼容。
- repo_id 是否存在。
- repo_id 是否重复。
- URL/name/source 是否缺失。
- summary 是否缺失。
- aliases 是否缺失或为空。
- search_text 是否缺失或过短。
- profile 是否 abstained。
- confidence 是否为允许值。
- archived/disabled 状态是否可见。

`records stats` 必须至少展示：

- repo 总数。
- schema version。
- profile confidence 分布。
- abstained 数量。
- missing search_text 数量。
- missing aliases 数量。
- archived/disabled 数量。
- top languages/ecosystems/project types（如果有）。

`index verify` 必须至少检查：

- index version。
- embedding model。
- dimension。
- record_count 是否和 vectors 对应。
- records 中有但 index 中没有的 repo。
- index 中有但 records 中没有的 repo。
- fingerprint stale 情况。
- embedding input version mismatch。

#### CLI 输出要求

默认 text 输出必须像报告，而不是裸 JSON。

必须包含 summary 和 next steps。例如：

```text
status: warning
problems:
  - 32 records missing search_text
  - 7 records missing vectors in index
next steps:
  - run xists profile refresh ...
  - run xists index build ...
```

JSON 输出必须包含机器可读字段：

- `ok`
- `status`
- `counts`
- `problems`
- `next_steps`

#### 测试要求

至少覆盖：

- 完整 records 通过 validate。
- 缺 search_text 报 warning/error。
- duplicate repo_id 报 error。
- old schema 报 warning/error。
- stale index 被 verify 发现。
- text 输出包含 next steps。
- JSON 输出可被测试稳定断言。

#### 禁止行为

- 禁止 validate 只检查 JSON 是否能 parse。
- 禁止只输出问题不给 next steps。
- 禁止 records stats 打印完整 records。
- 禁止 index stats 打印完整 vectors。

### 5.5 v0.5.0 验收标准：本地规模与 index 稳定

v0.5.0 的目标是确认 1k-10k repos 的本地体验，不是引入复杂向量数据库。

#### 必须完成

- 明确记录 index 格式字段：
  - index_version
  - record_schema_version
  - embedding_model
  - embedding_input_version
  - dimension
  - built_at
  - record_count
  - vectors count
  - fingerprints
- 搜索前必须检查 index 与当前 embedding config 是否兼容。
- `index stats` 必须显示 index 关键状态。
- `index verify` 必须能和 records 对比。
- 必须有至少一种方式测试或模拟 1k+ records/index 的行为。
- 不允许默认引入必须安装的数据库服务。

#### 性能验收

至少记录以下本地基线之一：

- 1k fixture/synthetic index 搜索耗时。
- 10k synthetic index 搜索耗时。
- index stats/verify 在大 fixture 上的耗时。

不要求 benchmark 非常严格，但必须防止明显 O(n) 之外的意外低效逻辑，例如对每个 query 重复 parse 大型 metadata 多次。

#### CLI 要求

`index stats` 默认输出必须包含：

- vector_count
- record_count
- model
- dimension
- embedding_input_version
- built_at
- skipped_count
- stale/missing 概览（如果传 records）

#### 禁止行为

- 禁止为了“未来规模”提前强制引入 FAISS/hnswlib/LanceDB。
- 禁止搜索时打印大型向量。
- 禁止 index mismatch 只给 Python traceback。
- 禁止没有测试就修改 index 格式。

### 5.6 v0.6.0 验收标准：规模化 ingest 与数据更新

v0.6.0 的目标是让数据源生产在万级规模上工程可行。coding agent 不允许只加一个简单 retry 循环就宣称完成。

#### 必须完成

- ingest 和 profile refresh 支持中断后续跑：
  - 已完成条目重跑时跳过。
  - 续跑不依赖用户手工维护剩余清单。
- GitHub API 限流触发时自动等待并重试，不失败退出。
- 增量刷新基于 fingerprint 判断，未变化的 repo 不重新 profile / 不重新 embedding。
- `--dry-run` 输出将处理数量、跳过数量、预计调用量。
- 单个 repo 失败被记录且不中断整批；结束后可只重试失败项。
- 阶梯实验的操作手册（`docs/scaling-experiment.md`）完成，含 corpus 分层配方、每级执行清单、记录模板与爬升判据。实验的实际执行是 v1.0.0 的发布门槛，不在 v0.6.0 代码验收范围内（与 §12 T6 一致）。
- eval 指标从 exact top-1 升级为 recall@k + judge，query 集分层，包含 no-result case。
- 分层 query 集每个类型至少包含一个中文 case，且报告中可区分中文 case 的命中情况。

#### 测试要求

至少覆盖：

- checkpoint 写入与续跑跳过逻辑（用 fixture 模拟中断）。
- 限流响应触发退避重试（mock API 响应）。
- fingerprint 未变化时跳过 profile/embedding。
- 单 repo 失败不影响批任务退出码语义和失败报告。
- dry-run 不产生实际写入和外部调用。

#### 禁止行为

- 禁止为了让大 corpus 的 eval 分数好看而向 query.py 添加规则或术语表。
- 禁止把 20k 压力测试结果当作产品规模承诺写进文档。
- 禁止引入数据库、队列服务等重依赖来实现 checkpoint；本地文件即可。
- 禁止长任务无进度输出。

### 5.7 v0.7.0 验收标准：Python API + 优秀 CLI + 打包首发

v0.7.0 的目标是让 xists 可被程序调用，同时保持 CLI 体验优秀，并完成 PyPI 首发。

#### Python API 必须完成

必须提供稳定入口，例如：

```python
from xists import load_index, search
```

API 必须：

- 不依赖 argparse。
- 不直接 print。
- 返回结构化 dict/list。
- 抛出可理解的异常或返回明确错误。
- 与 CLI JSON 输出保持一致或可直接映射。

#### CLI 必须完成

CLI 命令族应保持一致：

```bash
xists doctor
xists ingest github
xists profile refresh
xists records validate
xists records stats
xists records inspect
xists index build
xists index stats
xists index verify
xists search
```

每个核心命令必须有：

- help 文档。
- 正常路径测试。
- 至少一个失败路径测试。
- text/json 输出策略。

#### Agent-friendly JSON 必须包含

搜索结果 JSON 至少包含：

- query
- results
- repo_id
- url（如果有）
- summary
- confidence
- score
- why
- score_breakdown 或 equivalent evidence

如果 profile 中有以下字段，应透出：

- best_for / use_cases
- not_for
- capabilities
- project_type
- ecosystem
- aliases 或 entity_match evidence

#### Explain 模式验收

如果实现 `--explain`，必须显示：

- identity match 是否发生。
- semantic score。
- metadata adjustment。
- 使用的主要字段。
- confidence 原因。

如果暂时不实现 `--explain`，必须保证普通结果中的 `why` 足够有用，并在 roadmap 中保留 explain 后续项。

#### 禁止行为

- 禁止 MCP 或 CLI 直接复制搜索逻辑，绕过 core API。
- 禁止 API 返回和 CLI JSON 完全不同的结构。
- 禁止 API 内部调用 subprocess 执行 `xists search`。
- 禁止 CLI 默认输出难以阅读的大 JSON，除非命令明确是 JSON-first。

#### Packaging 必须完成

- `pyproject.toml` 含 license、classifiers、urls；仓库含 LICENSE（license 由维护者选定，见 v0.7.0 章节）。
- CI 在 push/PR 上运行 pytest。
- 大型 demo artifacts 移出 git，以 Release asset 分发；asset 必须是当前 schema / embedding input 版本，且发布前通过 `records validate` 与 `index verify`。
- README 覆盖 `pip install xists` 到第一次搜索成功的完整路径。
- §9 安全与隐私三问的答案已写入用户文档。
- 发布步骤已固化进 `docs/release.md`。
- 已发布到 PyPI，全新虚拟环境安装后 happy path 可跑通。

### 5.8 v0.8.0 验收标准：MCP / agent 集成

v0.8.0 的目标是让 agent 稳定使用 xists，不是简单把 CLI 包一层就结束。

#### 必须完成

MCP server 至少提供：

- `search_projects`
- `inspect_project` 或等价能力
- `index_stats` 或等价能力

MCP tools 必须使用 core Python API。

MCP 相关依赖必须作为 optional extra（`xists[mcp]`）提供，核心包运行时依赖保持 numpy-only；未安装 extra 时 CLI/API 完整可用，MCP 入口报错含安装命令（见 v0.8.0"依赖策略"）。

`search_projects` 返回必须与 CLI JSON 语义一致，至少包含：

- repo_id
- url
- summary
- confidence
- why
- score/evidence

#### Agent 体验要求

agent 拿到结果后，不应该只得到链接。必须得到足够判断材料：

- 这个项目做什么。
- 为什么和 query 相关。
- 它适合什么。
- 它不适合什么（如果 profile 有）。
- 是否需要继续搜索。

#### CLI 复现要求

MCP 搜索结果必须能用 CLI 复现：

```bash
xists search "same query" --format json
```

如果 MCP 加了额外字段，必须说明来源，不能凭空生成不可追溯内容。

#### 文档要求

必须包含：

- 如何启动 MCP server。
- 如何配置至少一种 agent 客户端。
- tool 列表。
- tool 输入输出示例。
- 如何用 CLI debug MCP 结果。

#### 禁止行为

- 禁止 MCP server 长期通过 subprocess 调 CLI。
- 禁止 MCP 返回不稳定临时 schema。
- 禁止 MCP 隐藏搜索错误，只返回空结果。
- 禁止 MCP 结果与 CLI/API 搜索结果明显不一致。

### 5.9 v1.0.0 最终验收标准

发布 1.0.0 前必须逐项确认。

#### 搜索稳定性

- exact repo/name/alias 查询可靠。
- functional query 可用。
- weak/no-result query 不会轻易 high confidence。
- 搜索结果有解释。
- query.py 没有重新膨胀成不可维护状态。

#### 数据稳定性

- Schema v2 稳定。
- records validate/stats/inspect 可用。
- profile refresh 或迁移路径可用。
- index build/stats/verify 可用。
- version mismatch 报错清晰。

#### CLI 稳定性

- README happy path 可完整跑通。
- docs/usage.md 覆盖主要命令。
- 默认 text 输出可读。
- JSON 输出稳定。
- 错误信息有 next steps。

#### API / MCP 稳定性

- Python API 有测试。
- MCP server 可运行。
- MCP 输出与 CLI/API 一致。
- agent-friendly JSON 字段稳定。

#### 测试与发布

- `pytest` 通过。
- demo 或 smoke workflow 通过。
- release docs 更新。
- 版本号一致。
- 不提交 `.env`、token、私人 records/index。
- PyPI 发布流程可重复（自 v0.7.0 起持续发布）。
- 全新环境 `pip install xists` 冒烟测试通过。
- 稳定性与废弃政策已写入用户文档。
- 阶梯实验 2k 与 10k 两级已执行且结果记录在 docs（见 v1.0.0"发布前置"节）。
- 距 v0.7.0 首发至少 4 周，期间的兼容性问题已处理或有记录的处置决定。

---

## 6. 给后续 agent 的执行提醒

如果你是后续接手实现的 agent，请注意：

1. **不要急着加功能。** 先读本 roadmap，确认当前版本目标。
2. **不要为了某个 case 往 query.py 加规则。** 先判断是不是 profile/search_text/schema 问题。
3. **不要破坏 CLI 体验。** 新能力必须考虑 text 输出、JSON 输出、错误提示和 help。
4. **不要把 records/index 当临时文件。** 它们是用户资产，涉及 schema/version/migration。
5. **不要过早引入重依赖。** 1.0.0 前保持本地、轻量、可理解。
6. **每个版本都要有测试和文档。** 尤其是 CLI 行为和数据格式。
7. **如果要偏离 roadmap，先更新 roadmap。** 不要让实现和方针脱节。ROADMAP.md 就在本仓库根目录，直接更新它，并在完成报告中列出偏离项供维护者审阅。

---

## 7. 一句话总结

xists 的核心不是 MCP，不是 Web UI，也不是 eval 分数。

xists 的核心是：

> 一条可靠、可维护、CLI 体验优秀、数据可共享的语义项目搜索管道。

先把管道打稳，再做集成。

---

## 8. 风险与缓解

以下是可能让项目失效的外部依赖和长期风险。每条都必须有已知的缓解姿势；新风险出现时追加到本节。

### embedding endpoint 依赖

- **风险**：搜索路径依赖本地/远程 embedding 服务，服务不可用则整个工具不可用。
- **缓解**：`doctor` 必须能诊断 endpoint 状态并给出 next steps（已有）；文档提供至少一条最低成本的 endpoint 搭建路径（v0.7 打包要求）；错误信息永远区分"endpoint 不可用"和"index 不匹配"这两种失败。

### 默认 embedding 模型停止分发

- **风险**：默认模型（bge-m3）若下架或停止维护，新用户无法 rebuild 出兼容 index。
- **缓解**：records 层不含向量、与模型无关，是真正的长期资产；模型不可得时按 §2.5 钦定新默认模型并发布全量 rebuild 指引。这也是文档必须强调"分享 records 优于分享 index"的原因。

### LLM profile 质量漂移

- **风险**：更换 profile 用的 LLM 后，新旧 records 的 profile 风格和质量不一致，混用导致排序不公平且难以察觉。
- **缓解**：records 记录 profile prompt version（已有）；`records stats` 展示 confidence 分布（已有）；文档建议同一份数据源用同一 LLM 一次性生成，跨模型混用前先跑 stats 对比新旧批次。

### GitHub API 政策变化

- **风险**：限额收紧、字段变更或认证方式调整导致 ingest 失效。
- **缓解**：record schema 与抓取方式解耦（`source` 字段预留多源）；records 是抓取结果的持久化，政策变化不影响已有资产；REST/GraphQL 双后端（已有）分散单点风险。

### 单人维护

- **风险**：xists 是个人工具，维护者中断投入则项目停滞。
- **缓解**：这正是本 roadmap 作为决策文档存在的理由——"为什么"和验收清单让任何 agent 或维护者可接手；1.0 前核心包保持零重依赖（MCP SDK 仅为可选 extra，见 v0.8.0 依赖策略）和纯本地文件架构，把项目"复活成本"压到最低。本文档自身也必须纳入版本控制（公开或私有仓库均可），不得只存在于单机目录——它是接手的前提，丢了它其余缓解都失效。

### PyPI 包名被抢注

- **风险**：首发定在 v0.7.0，从查证可用（2026-07）到实际发布之间可能间隔数月，`xists` 名称可能被他人注册，导致改名并波及包名、CLI 名、文档。
- **缓解**：维护者可选择提前发布一个 0.0.x 占位包（仅含说明与项目链接）锁定名称；若不占位，则接受改名风险，并在 v0.7.0 开工前重新查证一次。是否占位由维护者决定，执行 agent 不得代为发布。

---

## 9. 安全与隐私

发包前必须能回答以下三个问题，答案写进用户文档。

### Token 处理

- GitHub token 只从环境变量或 `--token-file` 读取，永不写入 records/index/report 等任何产物。
- `.env`、`.secrets/` 永不提交（.gitignore 保证，发布检查清单复核）。
- 错误信息和日志不允许输出 token 内容。

### 数据外发

- ingest 和 profile 会把 repo 的 README/metadata 全文发送给 LLM endpoint 和 embedding endpoint。
- 文档必须写明：配置远程 API 时这些内容会离开本机；对隐私敏感的用户应使用本地 endpoint。
- xists 自身不上传遥测，不回传任何数据。

### 共享数据源的内容责任

- records 包含 LLM 生成的摘要和源自 README 的派生内容；分享 records 即分享这些内容。
- 文档建议：只分享基于公开 repo 的数据源；含私有 repo 的 records 视为敏感文件，不应公开分发。

---

## 10. 变更记录

roadmap 是活文档。每次修订在此追加一条：日期、变更内容、原因。

- **2026-07-26** — 根据 `v0.8.1` 后项目审查重排发布路线：新增 `v0.8.2` 搜索正确性补丁、`v0.9.0` 中文检索与正式基线、`v0.9.x` 稳定窗口、`v0.10.0` 性能与索引架构版本，以及 `v1.0.0rcN` 发布候选阶段；新增 §15 的范围、禁止项、验收、提交隔离和发布流程。明确本地 `PROJECT_AUDIT.md` 仅作为整改输入，不得进入任何 commit、wheel、sdist 或 release asset。原因：审查问题跨越行为、评测、性能和格式，合并为单一版本会放大回归风险，也不符合语义化发布和可审查提交原则。

- **2026-07-23** — 补写 §13「v0.7.0 执行规格（草案，待维护者审阅）」：以 `origin/main` 的 `d6ebf57`（v0.6.2）为现状基线，冻结 v0.7 的公共 Python API、CLI JSON 契约、打包与首发工作分解、禁止项、验收命令及完成报告模板；同时明确 LICENSE、是否提前占用 PyPI 名称、Release asset 的最终托管与发布权限仍必须由维护者决定。原因：v0.6.x 已完成，按 §3 的执行规格约定，v0.7 开工前必须先有基于真实代码状态、可审阅且可验收的施工规格，不能仅凭版本目标直接编码。

- **2026-07-17** — 第三次修订（1.0.0 就绪性审查）。(1) 裁决阶梯实验归属：v0.6.0 验收只含操作手册，实验执行（至少 2k + 10k 两级）改为 v1.0.0 发布门槛，v1.0.0 新增"发布前置"节，§5.6 / v0.6.0 验收标准同步修改，消除与 §12 T6 的矛盾；(2) v1.0.0 增加首发后至少 4 周真实使用期门槛；(3) §9 安全与隐私三问列入 v0.7.0 packaging 验收，防止按清单执行时漏项；(4) v0.7.0 要求 Release asset 的 demo 数据先刷新到当前 schema 并通过 validate/verify；(5) v0.8.0 新增"依赖策略"：MCP SDK 作为可选 extra（`xists[mcp]`），核心包保持 numpy-only，§8 表述同步；(6) §3 新增执行规格约定：v0.7.0 及以后各版本开工前必须先补写 §11/§12 式执行规格；(7) §12 T4 增加多线程 ingest checkpoint 空洞的处理要求；(8) v0.7.0 补充 License 由维护者选定、发布流程固化进 docs/release.md；(9) §8 新增 PyPI 包名抢注风险，单人维护条目要求本文档纳入版本控制；(10) v0.7.0 API 示例补充 embedding 配置必须显式注入的说明。原因：完整审查发现按原文执行到 1.0.0，会带着未经真实规模验证的可靠性声明、缺失的隐私告知、过期的冷启动数据和未裁决的依赖冲突完成发布。
- **2026-07-17** — 第二次修订。重写 §12 T1 的 checkpoint 设计：partial 文件从"每刷新一条就整体重写完整 records 快照"改为 JSONL 逐条追加（`<output>.partial.jsonl`），并在 §12 硬规则/禁止事项中同步允许 JSONL、禁止快照式逐条重写。原因：20k 规模下快照方案累计写入约 1.4TB（O(n²)），每次全量 JSON 序列化的 CPU 与 LLM 调用同量级，checkpoint 会成为长任务自身的瓶颈；JSONL 追加使单条落盘成本恒定，且截断的末行天然可检测，最终 output 格式不变。
- **2026-07-17** — 初版后第一次修订。(1) v0.2.0-v0.4.0 标记完成，§1 加状态说明；(2) 插入 v0.6.0"规模化 ingest 与数据更新"（断点续跑、限流等待、增量刷新、dry-run、2k→10k→20k 阶梯实验、recall@k eval 升级），原 v0.6/v0.7 顺移为 v0.7/v0.8；(3) v0.7 并入 Packaging 与 PyPI 首发（首发不等 1.0）；(4) 新增核心原则 §2.5 embedding 模型策略；(5) v1.0.0 增加稳定性与废弃政策；(6) eval 要求覆盖中文查询；(7) 新增 §8 风险与缓解、§9 安全与隐私、§10 变更记录；(8) 修复重复的 §6 编号；(9) §5.1 明确 roadmap 不在仓库时的偏离上报方式。原因：为 PyPI 发包补齐产品侧缺口（冷启动、数据生产规模化、发布政策），并使文档可由低能力 agent 安全执行（v0.5.0/v0.6.0 执行规格见 §11、§12）。

---

## 11. v0.5.0 执行规格（实现方案与验收标准）

> 本节是给执行 coding agent 的操作规格，必须逐字遵守。对应章节："v0.5.0 — 本地规模与 index 稳定" 与 §5.5。
> 如果本节与代码现状不符，**停止工作并在最终报告中说明差异**，不要自行猜测或绕过。

---

### 0. 执行硬规则（违反任何一条即视为任务失败）

1. 按 T1 → T2 → T3 → T4 → T5 的顺序执行，不允许跳步、不允许合并步骤。
2. 每完成一个任务，必须运行 `python -m pytest tests/ -q`，全部通过才能进入下一个任务。出现失败时必须先修复，不允许注释掉或跳过失败的测试。
3. **禁止修改以下文件的现有逻辑**（只读参考）：
   - `src/xists/search/query.py` 的排序/打分逻辑（第 300 行以后的 ranking 部分）
   - `src/xists/records.py` 的 `RECORD_SCHEMA_VERSION`
   - `src/xists/profile/llm.py` 的 `PROFILE_PROMPT_VERSION`
   - `src/xists/search/embed.py` 的 `EMBEDDING_INPUT_VERSION` 和 fingerprint 逻辑
   - `src/xists/search/index.py` 的 `INDEX_VERSION` 和 index 输出格式（新增字段除外，见 T4）
4. 禁止添加任何新的第三方依赖。当前运行时依赖只有 `numpy`，保持不变。
5. 禁止引入 FAISS / hnswlib / LanceDB / sqlite-vss / 任何数据库。v0.5.0 明确继续使用 JSON + numpy brute-force。
6. 每个任务只做该任务描述的事。如果你发现"顺便可以改进"的东西，写进最终报告的"建议"部分，不要动手改。
7. 所有新测试不允许调用真实 GitHub API、真实 LLM、真实 embedding endpoint。用 fixture 和构造数据。

### 1. 开工前必读（按顺序读完再动手）

1. 本文档第 2 节（核心原则）、v0.5.0 章节、§5.1、§5.5
2. `src/xists/search/index.py`（全文，约 130 行）
3. `src/xists/search/query.py` 的第 460-530 行（搜索前兼容性检查部分）
4. `src/xists/cli.py` 中 `_index_stats_report`（约 836 行起）和 `_format_index_stats_text`（约 878 行起）
5. `tests/test_search.py` 和 `tests/test_cli.py` 中与 `index stats` / `index verify` 相关的现有测试（搜索关键词 `index_stats`、`index_verify`）

### 2. 现状盘点 — 以下能力已存在，禁止重新实现

执行前先逐条核实（核实方法附后）。如果某条与描述不符，停止并报告。

| 已有能力 | 位置 | 核实方法 |
|---|---|---|
| index 包含 index_version / record_schema_version / embedding_model / embedding_base_url / embedding_input_version / dimension / built_at / record_count / skipped / per-vector fingerprint | `src/xists/search/index.py` `build_index()` 返回值 | 读代码 |
| 搜索前检查 embedding_input_version、record_schema_version，不匹配时报可行动错误 | `src/xists/search/query.py` ~479、~486 行 | 读代码 + 现有测试 |
| `index stats` 命令（text 默认输出，不打印向量） | `src/xists/cli.py` | `python -m xists.cli index stats --help` |
| `index verify` 命令（stale/missing/mismatch 检测 + next steps） | `src/xists/cli.py` | `python -m xists.cli index verify --help` |
| index build 有 checkpoint 写入 | `src/xists/cli.py` `_index_write_checkpoint` | 读代码 |

**因此 v0.5.0 的实际工作量是差距部分：兼容检查补全核实、synthetic fixture、性能基线、内存估算展示、性能冒烟测试。**

---

### 3. 任务分解

#### T1 — 核实并补全搜索前兼容性检查

**目标**：确认搜索前对 index 的四项检查都存在：(a) embedding model 与当前配置一致；(b) dimension 一致；(c) embedding_input_version 兼容；(d) record_schema_version 兼容。

**步骤**：
1. 读 `src/xists/search/query.py` 中执行搜索前检查的函数，列出实际检查了哪几项。
2. 已确认存在 (c) 和 (d)。重点核实 (a) 和 (b)：如果搜索时 index 的 `embedding_model` 与 `EmbeddingConfig` 当前配置的 model 不同，或 query 向量维度与 index `dimension` 不同，是否有明确报错？
3. 缺哪项就补哪项。报错信息必须包含：实际值、期望值、用户下一步应运行的命令（参考现有 embedding_input_version 报错的措辞风格）。
4. 检查逻辑必须写在 `query.py` 的检查函数中（与现有检查同一位置），不允许写在 CLI 层。

**测试**（加到 `tests/test_search.py`）：
- 构造 model 不匹配的 index → 搜索抛出含 model 名的错误。
- 构造 dimension 不匹配 → 同上。
- 已有的 input_version / schema_version mismatch 测试保持通过。

**完成判据**：四项检查各有至少一个测试；报错文案含 next step。

#### T2 — synthetic fixture 生成脚本

**目标**：提供一个不依赖任何外部服务的脚本，生成 1k / 10k 规模的合成 records + index，用于性能测量。

**实现**：新建 `scripts/generate_synthetic_index.py`。

要求：
1. 用法：`python scripts/generate_synthetic_index.py --count 1000 --dimension 1024 --output-records /tmp/syn-records.json --output-index /tmp/syn-index.json --seed 42`
2. 生成的 records 必须是合法 schema v2：`repo_id` 形如 `synthetic/repo-000001`，`llm_profile` 含 summary / aliases / project_type / ecosystem / search_text 等字段（内容可以是模板化假文本，但必须通过 `xists records validate`）。
3. 生成的 index 必须与 `build_index()` 输出格式**逐字段一致**（index_version、record_schema_version、embedding_model、embedding_input_version、dimension、built_at、record_count、skipped、vectors 含 repo_id / embedding_input_fingerprint / metadata / vector）。向量用 `numpy.random.RandomState(seed)` 生成并归一化。fingerprint 必须调用真实的 `embedding_input_fingerprint()` 函数计算，不允许填假值。
4. `embedding_model` 字段填 `synthetic-test-model`，方便测试区分。
5. 脚本不允许 import 任何网络相关模块，不允许调用 embedding endpoint。
6. 复用 `src/xists/` 中已有的函数（`entry_metadata`、`embedding_input_fingerprint` 等），不允许复制粘贴它们的实现。

**测试**（新建 `tests/test_generate_synthetic_index.py`）：
- 生成 count=50 的小 fixture，断言：records 通过 `records validate` 的核心检查；index 字段齐全；vectors 数量 = 50；同一 seed 两次生成结果一致。

**完成判据**：`--count 1000` 在 30 秒内完成；生成物能被 `index stats`、`index verify` 正常消费（手动跑一次确认，写进报告）。

#### T3 — 性能基线测量与记录

**目标**：记录 1k / 10k 规模下的实际耗时，写成文档。这是测量任务，不是优化任务。

**步骤**：
1. 用 T2 脚本生成 1k 和 10k 两套 fixture（dimension 用 1024）。
2. 测量并记录以下项目，各跑 3 次取中位数（用 `time.perf_counter`，可以写一个临时脚本 `scripts/bench_search.py` 并保留）：
   - 搜索延迟：注意搜索需要 embed query，synthetic 场景下没有真实 endpoint。做法：直接调用 `query.py` 中的核心搜索函数，query 向量用随机归一化向量代替（在 bench 脚本内构造），只测"加载 index + 相似度计算 + 排序"部分。分别记录含 index 加载和不含加载（index 已在内存）两个数字。
   - `index stats` 耗时（subprocess 计时即可）。
   - `index verify` 耗时（对照对应 records）。
3. 新建 `docs/performance.md`，内容包括：测量环境（CPU 型号、内存、Python 版本）、fixture 规模与维度、每项的中位数耗时表格、结论（是否满足"1k 近似即时、10k 仍可交互"）。
4. **判断标准**：10k 时不含加载的搜索核心计算应在 1 秒以内（numpy brute-force 在这个规模理应远快于此）。如果超过 1 秒，**不要自行优化**，在报告中说明测量数据并停在这一步等待人工决策。唯一允许的例外：如果发现明显的重复工作（例如每次搜索对每个向量重复 json parse 或重复 normalize），可以修复并在报告中说明修复前后的数字对比。

**完成判据**：`docs/performance.md` 存在且含真实测量数字；bench 脚本保留在 `scripts/`。

#### T4 — index stats 增加内存占用估算

**目标**：`index stats` 输出中增加 `estimated_memory_mb`（ROADMAP 标注为可选项，此处正式实现）。

**实现**：
1. 在 `_index_stats_report` 中计算：`vector_count × dimension × 8 / 1024 / 1024`（float64 假设），保留 1 位小数。
2. text 输出加一行 `estimated memory: X.X MB`；JSON 输出加字段 `estimated_memory_mb`。
3. dimension 或 vector 数据缺失时该字段为 null，text 输出显示 `estimated memory: unknown`，不允许抛异常。

**测试**：text 和 JSON 各断言一次；缺 dimension 的 index 走 unknown 分支。

**完成判据**：现有 index stats 测试不回归；新字段有测试。

#### T5 — 性能冒烟测试（防退化护栏）

**目标**：在 pytest 中加入一个轻量护栏，防止未来改动引入明显的复杂度退化。

**实现**（加到 `tests/test_search.py` 或新建 `tests/test_performance_smoke.py`）：
1. 测试内用 T2 的生成逻辑（import 脚本中的函数）在内存中构造 count=2000、dimension=64 的 index（小维度保证测试快）。
2. 断言：对该 index 执行 20 次核心搜索（随机 query 向量）总耗时 < 5 秒。这个阈值故意宽松——它的目的是抓 O(n²) 级别的意外退化，不是精确 benchmark。
3. 测试必须离线可跑、总时长控制在 10 秒以内。

**完成判据**：该测试在本机稳定通过（连续跑 3 次）。

---

### 4. 验收核对表（全部完成后逐条执行并把输出写进报告）

```bash
# 1. 全量测试
python -m pytest tests/ -q                                    # 期望：全部通过，0 failed

# 2. synthetic fixture 全链路
python scripts/generate_synthetic_index.py --count 1000 --dimension 1024 \
  --output-records /tmp/syn-records.json --output-index /tmp/syn-index.json --seed 42
python -m xists.cli records validate --records /tmp/syn-records.json   # 期望：ok: true
python -m xists.cli index stats --index /tmp/syn-index.json            # 期望：含 estimated memory 行，不打印向量
python -m xists.cli index verify --records /tmp/syn-records.json --index /tmp/syn-index.json
                                                              # 期望：status: ok

# 3. mismatch 报错可行动（用手工改坏的 index 副本验证 model/dimension 检查）
# 把 /tmp/syn-index.json 的 embedding_model 改成 other-model 后搜索 → 期望：明确报错 + next steps

# 4. 文档
ls docs/performance.md                                        # 期望：存在，含真实数字
```

### 5. 明确禁止（v0.5.0 特有）

- 禁止为了 benchmark 数字修改排序逻辑或减少返回字段。
- 禁止把 synthetic fixture 提交到 git（生成脚本提交，生成物不提交；确认 `.gitignore` 覆盖或输出到 /tmp）。
- 禁止在测试中生成 10k 级 fixture（太慢）；10k 只在 T3 手动测量中使用。
- 禁止修改 `demo-records.json` / `demo-index.json`。

### 6. 完成报告模板（最终回复必须按此结构）

```
### v0.5.0 完成报告
#### 已完成
- T1: [完成情况，含核实结论：四项检查中哪些原本就有、哪些是新补的]
- T2: ...
- T3: [附 docs/performance.md 中的核心数字]
- T4: ...
- T5: ...
#### 未完成或偏离（没有则写"无"）
- [哪条验收标准没满足 / 为什么 / 当前替代方案 / 建议如何补齐]
#### 测试
- pytest 结果：X passed
#### 验收核对表执行结果
- [逐条命令 + 实际输出摘要]
#### 建议（本次未动手的改进点）
```

不允许使用"基本完成""大致可用"等模糊表述。

---

## 12. v0.6.0 执行规格（实现方案与验收标准）

> 本节是给执行 coding agent 的操作规格，必须逐字遵守。对应章节："v0.6.0 — 规模化 ingest 与数据更新" 与 §5.6。
> 前置条件：v0.5.0 已完成（`scripts/generate_synthetic_index.py` 和 `docs/performance.md` 已存在）。
> 如果本节与代码现状不符，**停止并在报告中说明差异**，不要自行猜测。

---

### 0. 执行硬规则（违反任何一条即视为任务失败）

1. 按 T1 → T6 顺序执行。每个任务完成后跑 `python -m pytest tests/ -q`，全部通过才进入下一个。
2. **绝对禁区**：`src/xists/search/query.py` 的排序逻辑。本版本任何任务都不需要碰它。如果你认为需要改 query.py 才能完成某个任务，说明你理解错了任务——停止并报告。
3. 禁止新增第三方依赖（运行时依赖保持只有 numpy）。checkpoint 用本地 JSON/JSONL 文件实现，禁止引入数据库、队列、缓存服务。
4. 禁止修改 `RECORD_SCHEMA_VERSION` / `PROFILE_PROMPT_VERSION` / `EMBEDDING_INPUT_VERSION` / `INDEX_VERSION` 的值。
5. 所有测试离线可跑：GitHub API、LLM、embedding 一律 mock / fixture / injected function。仓库中已有大量此类测试范例（`tests/test_github_ingest.py`、`tests/test_llm_profile.py`），先模仿再动手。
6. 新增 CLI 参数必须同时更新 `--help` 文案和 `docs/usage.md`。
7. 长任务的进度输出打到 stderr 或与现有 `_print_ingest_progress` 风格一致；`--format json` 模式下 stdout 只输出最终 JSON。

### 1. 开工前必读

1. 本文档 v0.6.0 章节 + §5.1 + §5.6
2. `src/xists/ingest/github.py`（重点：`RETRYABLE_HTTP_STATUSES`、指数退避 `time.sleep(2**attempt)`、TokenPool、GraphQL rateLimit 查询）
3. `src/xists/cli.py` 的 ingest 部分（~89-344 行：`_ingest_one`、`_ingest_graphql_batch`、checkpoint 写入、`_print_ingest_progress`）
4. `src/xists/cli.py` 的 profile refresh 部分和 `src/xists/records.py` 的 refresh 选择逻辑（`only_missing_search_text` 等）
5. `src/xists/eval/` 全部 + `scripts/generate_stratified_eval.py` + `scripts/check_eval_report.py`
6. `tests/test_github_ingest.py`（mock 模式参考）

### 2. 现状盘点 — 已存在的能力，禁止重新实现

执行前逐条核实；与描述不符则停止并报告。

| 已有能力 | 位置 |
|---|---|
| ingest 默认增量：已存在于 output 的 repo 会跳过，`--force` 才全量重跑 | `cli.py` ingest |
| ingest 单线程模式逐 repo 写 checkpoint；多线程模式全部完成后写 | `cli.py` ~277、~295 行注释 |
| HTTP 429/5xx 指数退避重试 | `ingest/github.py` `RETRYABLE_HTTP_STATUSES` + `2**attempt` |
| 多 token 轮换（TokenPool，GITHUB_TOKENS） | `ingest/github.py` |
| GraphQL 批量模式（低配额消耗）+ rateLimit 查询 | `ingest/github.py` |
| ingest 进度输出 + `--report` 失败报告文件 | `cli.py` |
| profile refresh 默认只刷新过期记录（prompt version 判断），`--force` 全量，`--only-missing-search-text` 过滤 | `cli.py` + `records.py` |
| index build 有 checkpoint（`_index_write_checkpoint`） | `cli.py` ~344 |
| eval run + `--llm-judge`（top-1 不一致时 LLM 成对裁决） | `eval/` |
| 分层 eval 生成脚本 | `scripts/generate_stratified_eval.py` |

**因此 v0.6.0 的实际差距是**：profile refresh 的断点续跑、dry-run 预估、失败隔离与只重试失败项、GitHub 限流的"等到重置"策略（现在只有指数退避，长任务遇到配额耗尽会失败）、recall@k 指标、以及阶梯实验的操作手册。以下任务只做差距部分。

---

### 3. 任务分解

#### T1 — profile refresh 断点续跑

**现状问题**：refresh 是全内存处理、最后一次性写 output。中断即丢失全部进度。

**checkpoint 格式（为什么是 JSONL 追加，不是完整快照）**：records.json 是单个 JSON 文档，物理上不可追加——逐条落盘只能整体重写。按 demo-records.json 约 7KB/repo 推算，20k repos 的完整快照约 140MB；"每条重写一次快照"累计写入约 1.4TB，且每次全量序列化的 CPU 开销与 LLM 调用本身同量级，checkpoint 会成为它要保护的长任务自身的瓶颈。因此 checkpoint 用 JSONL：每行一条独立记录，逐条 append 成本恒定（约 7KB/条）。JSONL 只是 checkpoint 的内部格式，最终 output 仍是正常 records.json，schema 与下游命令不受影响。

**实现**：
1. checkpoint 文件为 `<output>.partial.jsonl`。每成功刷新完一条记录，把该条完整 record（含新 profile）序列化为一行 JSON，append 到该文件并 flush。只写实际刷新成功的记录：被选择逻辑跳过的记录不写（重跑时跳过它们是零成本的），失败的记录不写（见 T3，失败记录保留旧 profile，续跑时自然重试）。
2. 增加 `--resume` 参数：启动时如果 `<output>.partial.jsonl` 存在，逐行解析，建立 repo_id → 已刷新 record 的映射（同一 repo_id 出现多行时取最后一行）。最后一行解析失败（进程被杀导致的截断写）时丢弃该行、把该条视为未完成，不允许因此报错退出。处理每条记录时：repo_id 在映射中 → 直接使用映射结果，不调用 LLM；不在映射中 → 走现有的刷新选择逻辑（prompt version 判断、`--only-missing-search-text` 等过滤）。
3. 不带 `--resume` 时如果 partial 文件存在，报错提示：要么加 `--resume` 续跑，要么删除 partial 文件重来。不允许静默覆盖。
4. 全部完成后：按输入 records 的原始顺序组装最终结果（未刷新的保留原样，已刷新的用 partial/本次结果替换），原子写最终 output（先写临时文件再 rename，模仿 `_index_write_checkpoint` 的做法），成功后删除 partial 文件。
5. 进度输出：`refreshed X/Y (skipped S, failed F)`，每 10 条输出一次。

**测试**（`tests/test_cli.py` 或新文件，LLM 用 injected mock）：
- mock LLM 在第 N 条抛异常 → partial JSONL 存在、恰含前 N-1 条已刷新记录、每行可独立解析。
- 带 `--resume` 重跑 → mock LLM 只被调用剩余条数次，最终 output 与不中断一次跑完的结果一致。
- partial 末行被人为截断（fixture 直接截断文件字节）→ `--resume` 不崩溃，该条被重新刷新。
- 不带 `--resume` 且 partial 存在 → 报错退出，错误信息含两条 next steps。
- 正常完成 → partial 被清理，output 是合法 records.json，记录顺序与输入一致。

#### T2 — dry-run 预估

**实现**：
1. `profile refresh` 增加 `--dry-run`：执行选择逻辑但不调用 LLM、不写任何文件。输出：总记录数、将刷新数、将跳过数（按原因分类：已是当前版本 / 被过滤条件排除）、预计 LLM 调用次数。
2. `ingest github` 增加 `--dry-run`：读取 repos 文件与既有 output，输出：清单总数、已存在将跳过数、将抓取数、预计 GitHub API 请求数（REST 模式 ≈ 每 repo 的请求数 × 将抓取数；GraphQL 模式 ≈ ceil(将抓取数 / batch-size)。先读代码确认每 repo 实际发几个请求，用真实数字）。
3. 两者都支持 `--format json`（字段：`total`、`to_process`、`to_skip`、`skip_reasons`、`estimated_calls`）和默认 text（人读，含一行说明"this was a dry run, nothing was written"）。

**测试**：
- dry-run 后断言：output/partial/report 文件都不存在、mock LLM/HTTP 零调用。
- JSON 输出字段断言。
- text 输出含 dry run 声明。

#### T3 — 失败隔离与只重试失败项

**现状**：ingest 已有 `--report` 失败报告。需要补齐闭环。

**实现**：
1. 核实 ingest 单 repo 失败不中断整批（读代码 + 已有测试确认；缺则修）。
2. `profile refresh` 对齐同样行为：单条 LLM 失败记入失败列表，继续处理后续；结束时失败记录保留旧 profile 不变（不允许写入半成品 profile）。
3. `profile refresh` 增加 `--report <path>`：写 JSON 失败报告，格式与 ingest report 对齐（先读 ingest report 的实际结构，保持字段风格一致；至少含 repo_id、error、attempted_at）。
4. `ingest github` 和 `profile refresh` 增加 `--retry-failed <report.json>`：只处理报告中列出的 repo_id。与 `--dry-run` 可组合。
5. 退出码语义：全部成功 = 0；有失败但批任务完成 = 0 且 stderr 汇总失败数与报告路径；批任务本身无法进行（文件不存在、endpoint 全挂）= 非 0。在 `docs/usage.md` 写明这个语义。

**测试**：
- mock 第 3 条失败 → 其余条目正常完成、报告含该条、退出码 0、该条旧 profile 未被改动。
- `--retry-failed` 只处理报告内条目。
- endpoint 完全不可用 → 非 0 退出 + 可行动错误。

#### T4 — GitHub 配额耗尽的等待策略

**现状**：429/5xx 有指数退避（最多几次尝试），但配额耗尽（rate limit reset 在几十分钟后）会重试几次后失败。数十小时的批任务需要"等到重置时间"。

**实现**：
1. 在 `ingest/github.py` 中：收到 403/429 且响应头含 `x-ratelimit-remaining: 0` 时，读取 `x-ratelimit-reset`（epoch 秒），计算等待时长。GraphQL 模式用响应中已有的 `rateLimit.resetAt`。
2. 若 TokenPool 有其他 token，先换 token；全部 token 都耗尽时，等待最早的 reset 时间 + 5 秒缓冲，期间每 60 秒向 stderr 输出一行 `rate limited, resuming at <ISO时间>`。
3. 增加 `--max-rate-limit-wait <seconds>` 参数，默认 3600；超过则报错退出（此时单线程模式的 checkpoint 保证已完成部分不丢失）。
4. sleep 必须可注入（函数参数或 module 级可替换），测试中不真实等待。
5. **多线程模式的 checkpoint 空洞必须在本任务一并处理**。现状：多线程 ingest 全部完成后才写 checkpoint（cli.py ~277 行注释），数十小时任务中途失败（含超过 max-rate-limit-wait 退出）会丢失全部进度，而大规模 ingest 恰恰最可能开多线程。二选一：(a) 多线程模式每完成约 25 个 repo 落盘一次 checkpoint（写法与单线程 checkpoint 一致，注意线程安全：由主线程或加锁写入）；(b) 不改代码，但 `--help`、`docs/usage.md` 和 `docs/scaling-experiment.md` 明确规定大规模 ingest 必须使用单线程模式，且多线程模式下触发限流等待时 stderr 警告这一点。选择哪种及理由写进完成报告；选 (a) 时必须有测试（mock 中断后 checkpoint 含已完成条目）。

**测试**：
- mock 响应带 remaining=0 + reset=now+30 → 调用注入的 sleep 且时长约 35 秒（不真等）。
- 多 token 场景：先换 token 不 sleep。
- 超过 max wait → 报错退出且错误含 reset 时间和 next steps。

#### T5 — eval 增加 recall@k

**范围警告**：本任务只改 `src/xists/eval/` 的指标计算和报告展示。禁止改动搜索本身，禁止调整任何 case 的期望答案。

**实现**：
1. 先读 eval 报告现有结构（跑 `python -m xists.cli eval inspect --help` 并读 `eval/` 代码，弄清当前指标怎么算、报告什么字段）。
2. eval cases 增加可选字段 `acceptable`（repo_id 数组）：除 `expected` 外也算命中的答案。缺省时行为与现在完全一致。
3. 报告 summary 增加：`recall_at_1`、`recall_at_5`（top-k 结果中含 expected 或 acceptable 任一即算命中）。原有指标全部保留不动。
4. `eval inspect` 的 text 输出展示新指标；`scripts/check_eval_report.py` 的阈值检查逻辑保持兼容（读代码确认它不会因新字段崩溃）。
5. `docs/usage.md` 或 eval 相关文档补充 `acceptable` 字段说明和两个指标的定义。

**测试**：
- 构造 4 个 case 的迷你 eval（含一个靠 acceptable 命中、一个 top-5 命中但 top-1 未中、一个全 miss；其中至少一个 case 的 query 是中文）→ 断言 recall_at_1 / recall_at_5 精确值。
- 无 acceptable 字段的旧 cases 文件 → 正常运行，新指标仍计算。

#### T6 — 阶梯实验操作手册（文档任务，不写代码）

**实现**：新建 `docs/scaling-experiment.md`，内容必须包含：

1. **目标与红线**：验证 ranking 在 2k → 10k → 20k 上的稳定性；红线原文引用本文档 §5.6："禁止为 eval 分数往 query.py 加规则"。
2. **corpus 分层配方**（每级给出具体构成比例，可执行的选取来源说明）：
   - 60% 各领域 top-star 真实项目（AI/LLM、web、devtools、infra、data 各若干）
   - 20% 中长尾（1k-10k stars）
   - 20% 刻意噪声：tutorial 仓库、awesome-list、archived 项目、知名项目的 fork、名字与热门项目相近的小项目
3. **每级执行清单**（按序的完整命令，直接可复制）：ingest（含 --dry-run 先预估）→ records validate → records stats → profile refresh → index build → index verify → eval run → eval inspect。
   query 集要求：每个类型分层（exact name / functional / ecosystem / ambiguous / no-result）至少含一个中文 case——本文档 §0 的示例查询全是中文，这是维护者的真实使用方式，必须被 eval 覆盖。
4. **每级记录模板**：corpus 构成、耗时、失败数、recall@1 / recall@5、发现的问题分类（identity 冲突 / confidence 虚高 / 语义拥挤 / 数据质量）。
5. **爬升判据**：上一级 eval 无系统性回归、失败率 < 2%、发现的问题都已归因（数据问题 → 记录待修；不允许归因为"需要往 query.py 加规则"）。
6. **明确说明**：实验本身需要真实 GitHub token、LLM endpoint、embedding endpoint 和数十小时运行时间，由维护者择时执行，不属于本版本代码验收范围。

---

### 4. 验收核对表（全部完成后逐条执行，输出写进报告）

```bash
python -m pytest tests/ -q                          # 全部通过
python -m xists.cli profile refresh --help          # 含 --resume / --dry-run / --report / --retry-failed
python -m xists.cli ingest github --help            # 含 --dry-run / --retry-failed / --max-rate-limit-wait
python -m xists.cli profile refresh --records demo-records.json --dry-run
                                                    # 不写任何文件，输出预估（demo records 为 v1，将刷新数应为 200）
python -m xists.cli eval run --help                 # 不需要新参数，但 eval inspect 输出含 recall 指标
ls docs/scaling-experiment.md docs/usage.md         # 存在且已更新
git status                                          # 无生成物（partial、report、fixture）被意外加入
```

### 5. 明确禁止（v0.6.0 特有）

- 禁止改 `query.py`（整个版本零改动，验收时用 `git diff --stat` 证明）。
- 禁止为让 demo eval 或迷你 eval 分数好看而调整任何搜索行为。
- 禁止 checkpoint/partial 文件用 pickle（一律 JSON 或 JSONL；profile refresh 的 partial 按 T1 规定用 JSONL）。
- 禁止 checkpoint 采用"每条重写完整快照"的写法（O(n²) 写入量，见 T1 格式说明）；逐条落盘必须是追加式。
- 禁止在测试中真实 sleep 超过 1 秒。
- 禁止把 20k 实验写成代码验收项（它是操作手册 + 维护者手动执行）。

### 6. 完成报告模板

```
### v0.6.0 完成报告
#### 已完成
- T1..T6: [各自完成情况；T1-T4 注明"现状核实结论：哪些已存在、哪些是新实现"]
#### 未完成或偏离（没有则写"无"）
- [哪条 / 为什么 / 替代方案 / 如何补齐]
#### 测试
- pytest 结果：X passed
- git diff --stat 中 query.py 的行数变化：必须为 0
#### 验收核对表执行结果
- [逐条命令 + 实际输出摘要]
#### 建议
```

不允许使用"基本完成""大致可用"等模糊表述。

---

## 13. v0.7.0 执行规格（草案，待维护者审阅）

> 对应章节："v0.7.0 — 稳定 API、优秀 CLI 与 PyPI 首发"。本节不是授权发布，也不是立即开工指令；它把 v0.7 的边界、顺序和验收冻结成可审阅的施工规格。维护者确认本节后，才可开始 T1-T3；下列与发行有关的前置决定必须在 T4 前完成，且授权发布前不得执行 T5。
>
> 现状基线：`origin/main` 的 `d6ebf57`（v0.6.2）。不得以已经分叉的本地分支、实验分支或未跟踪 `data/` 产物作为 v0.7 行为基线。

### 0. 开工门槛与必须由维护者决定的事项

开始实现前，维护者必须明确记录以下决定：

1. **LICENSE**：选择许可证名称及完整文本。执行者不得自行猜测或选择。
2. **PyPI 节奏**：是否先发布一个 0.0.x 占用名称，还是直接等到 v0.7.0。两者都可行，但实际上传一律需要维护者当次明确授权。
3. **Roadmap 的版本控制归属**：当前权威 Roadmap 位于仓库外的 Downloads；是否迁入仓库并纳入版本控制，须由维护者决定。不得擅自复制、覆盖或移动两份已经分叉的 Roadmap。
4. **发布权限**：创建 GitHub Release、上传 Release asset、推送 tag、上传 PyPI 均是外部状态变更；仅在所有验收通过且维护者明确授权后执行。

在上述决定未齐全时，允许完成不依赖它们的 API/CLI 测试、文档草稿和打包预检；不得添加 LICENSE、不得上传、不得声称已公开发布。

### 1. v0.7.0 硬规则

1. 按 T1 → T5 顺序执行。每个独立逻辑点一个 Conventional Commit；如果维护者继续采用直接合并工作流，每个任务分支完成验证后由维护者直接合并 main，不创建 PR。
2. 每个任务至少执行相关离线 pytest 与 `git diff --check`；所有新增测试必须 mock GitHub、LLM 和 embedding endpoint，不能依赖真实凭据或网络。
3. 运行时依赖继续保持只有 `numpy`。构建工具可以作为开发/发布环境依赖记录，但不得为此引入数据库、ANN、Web 框架或本地模型运行时。
4. 不改变 `src/xists/search/query.py` 的排名行为，不为 demo、私有 2k/10k、中文、AI、工程或任一领域特化排序规则、阈值或评测答案。
5. 公共 Python API 不得隐式读取 `.env`、环境变量或当前目录，不得 `print`、`sys.exit`、启动子进程，也不得在 import 时网络访问、读取文件或下载模型。embedding 配置必须显式注入。
6. CLI、公共 API 与未来 MCP（本版本不实现）必须复用同一核心搜索逻辑；公共 API 不得调用 CLI 或 subprocess，CLI 才能调用公共 API。
7. 不提交 `.env`、token、私有 records/index/eval 产物、刷新后的 demo 数据或 `data/scale-*`。不触碰 `experiment/multiview-retrieval`。
8. 任何 index schema、profile prompt、embedding input、模型或格式兼容性变化都必须显式写出迁移/重建影响；不得静默吞掉 model/dimension mismatch。

### 2. 真实代码现状与 v0.7 缺口

执行者先按下表核实。若真实代码与本表不符，停止相关任务并在完成报告说明差异，不要按记忆重写能力。

| 能力 | 当前位置 | v0.7 处理 |
|---|---|---|
| 内部 index JSON 加载 | `src/xists/search/index.py:load_index(path)` | 可以包装为公共 API；先冻结输入校验和错误契约 |
| 内部单/批查询 | `src/xists/search/query.py:rank`、`rank_many` | 保持为核心实现；公共 API 调用它们，不重写排名 |
| 显式 embedding 配置 | `src/xists/search/embed.py:EmbeddingConfig` | 作为 API 的显式注入配置候选 |
| CLI 的环境读取 | `src/xists/cli.py:main`、`load_env_file(Path(".env"))` | 仅留在 CLI 装配层，绝不可泄漏到公共 API |
| CLI 已有 JSON 结果 | `src/xists/cli.py` 各命令 | 冻结核心命令的 JSON 契约及 stdout/stderr/exit code 行为 |
| 离线 CI fixture | `examples/ci-smoke`、`scripts/smoke_check.py` | 必须保留并覆盖安装后的最小 smoke |
| 打包元数据与 release 文档 | `pyproject.toml`、`docs/release.md` | 当前不完整：metadata、版本来源、安装路径和 release runbook 都需补齐 |

额外核实项：当前 `src/xists/__init__.py` 与 `pyproject.toml` 仍为 `0.6.0`，而稳定线已存在 v0.6.2 tag。因此 v0.7 必须先定义**单一版本来源或可自动验证的同步关系**；绝不能把互相矛盾的版本号带进 PyPI。根目录 demo artifacts 当前被 git 跟踪，迁移为 Release asset 前必须先提供可下载或可本地重建的路径，不能直接删除。

### 3. 任务分解

#### T1 — 冻结公共 Python API 与错误契约

**目标**：提供一个不依赖 CLI 环境副作用、可被 Python 程序稳定调用的最小搜索 API。具体模块位置可在实现审阅时决定（例如 `xists/api.py`），但名字、输入、输出和错误在实现前必须写进测试并冻结。

最小候选形态：

```python
from pathlib import Path

from xists import load_index, search

index = load_index(Path("index.json"))
result = search(
    "open source firebase alternative",
    index,
    embedding_config=embedding_config,
    top_k=5,
)
```

**实现要求**：

1. `load_index` 接受 `str | Path`，返回 JSON-compatible 的已加载 index 对象；损坏 JSON、缺失文件、无效 schema 必须给出明确、可行动的异常。
2. `search` 接受非空 `query: str`、已加载 index、显式 `EmbeddingConfig` 和受验证的 `top_k`。若审阅时选择 injected embedding callable 而非 `EmbeddingConfig`，必须二选一并在文档、类型和测试中保持一致，不能同时形成两个模糊入口。
3. API 只公开当前能稳定承诺的基础 `rank` 能力。rerank、query transform 或实验性功能若无法冻结签名、字段与错误，留给后续版本，不得为凑功能公开。
4. 成功值与 search CLI JSON 的语义一致。顶层至少冻结 `query`、`query_intent`、`abstained`、`results`；如存在 `latency_ms`，明确它的单位和是否稳定。每个结果至少冻结 `repo_id`、`url`、`summary`、`confidence`、`score`、`why`；对可空字段采用一种固定表示。现有 `semantic_score`、`metadata_score`、`entity_match` 等字段必须明确是稳定保留、转换还是内部实现细节。
5. 无效 query、无效 top_k、index dimension/model mismatch、embedding endpoint/config 失败必须以可捕获异常表示；不得 `print` 或退出进程。错误信息须说明下一步，例如重新 build index 或检查 endpoint/model 配置。

**测试**：使用 fixture index 与 mock embedder/config，覆盖正常搜索、空/非法输入、损坏 index、dimension mismatch、model mismatch、endpoint 失败，以及 API import/search 全程不读取 `.env`、不访问网络。

#### T2 — CLI 复用 API，冻结 search 的 JSON 与文本体验

**目标**：确保 CLI 和 Python API 不会在日后各自演化出不同的搜索语义。

**实现要求**：

1. `xists search` 通过公共 API 或其同一层核心调用执行，不得让 API 调用 CLI。CLI 仍可在最外层读取 `.env` 并把配置显式传入 API。
2. 默认 text 输出保持面向人类可读；`--format json` 的 stdout 必须是唯一、完整 JSON payload。错误只写 stderr，并使用非零退出码。
3. 同一 fixture、同一 query、同一配置下，CLI JSON 和 API 的核心字段、结果排序、abstention 语义必须一致。
4. `why` 只能返回已有、真实的 evidence；如果现有信息不足以支持单独 `--explain`，不要虚构该参数或解释，把它列为后续版本候选。

**测试**：比较 API 与 CLI JSON；覆盖默认 text、JSON 成功、配置/文件失败的 stderr 与 exit code。

#### T3 — 统一核心 CLI 的机器可读契约、帮助与错误体验

**范围**：审计面向脚本或 agent 的核心诊断命令：`doctor`、`records validate/stats/inspect`、`index stats/verify`、`search`。不要求为了形式统一给所有历史命令强加 JSON，但每个核心命令必须有明确、稳定的机器可读策略。

**实现要求**：

1. 在 `docs/usage.md` 说明每个核心命令的 text/JSON 支持、stdout/stderr 分工和成功、validation failure、usage/config/file error 的 exit code 语义。
2. 参数新增或变更必须有准确 `--help`，并覆盖正常与失败路径测试。
3. 保持现有 eval 的 JSON-first 行为兼容，不输出 embedding vector、完整 README 或完整 records 等大 payload。
4. 需要加入 JSON 的地方要定义稳定字段，不得仅把人类文本包进 `message` 后称为 API。

#### T4 — 打包、安全文档与 Release asset 准备

**实现要求**：

1. 仅在维护者决定 LICENSE 后，添加 LICENSE 文件，并在 `pyproject.toml` 补齐正确的 license、classifiers、项目 URL 等 metadata。
2. 解决版本源不一致：选择单一版本来源，或实现构建前/CI 同步检查；`xists version`、wheel metadata、tag 和 release 文档中的 v0.7.0 必须一致。
3. 使 `python -m build` 能构建 sdist 与 wheel，并验证二者内容不包含 `.env`、私有数据、token、测试缓存或不应发布的生成物。构建工具安装方式必须写入开发/发布文档。
4. 更新 README（以及若维护者确认在维护范围内的中文 README）：至少有 `pip install xists`、首次真实搜索的最小 happy path、endpoint 配置方式、index/records 的获得或 build 方式。必须清楚披露远端 endpoint 会接收 query 与用于 embedding 的内容、本地 endpoint 选择、不做遥测、token 只局部读取，以及共享数据的内容责任。
5. 重写 `docs/release.md`：版本检查、build、clean venv、tag、main CI、Release asset hash/version、上传顺序、回滚/失败处理，以及不发布私有数据和凭据。
6. 先审计所有根目录 demo artifact 的 tracked 状态与消费者。迁移为 Release asset 前，必须提供准确下载位置和校验方式，或本地可复建路径；release-prep 工具/文档可先完成，但真实生成、GitHub 上传与公开发布属于 T5 的授权动作。

#### T5 — Release candidate 与授权后首发

**前置条件**：T1-T4 已完成、所有验收通过、维护者已决定 LICENSE 与 PyPI 节奏，并明确授权当次 tag / GitHub Release / PyPI 上传。

**执行要求**：

1. 版本、tag `v0.7.0`、package metadata 和 `xists version` 完全一致。
2. 在新建的临时 venv 中安装**构建出的 wheel**（不得 editable install），运行 `xists --help`、`xists version` 与 committed smoke fixture；如果可构造不访问外部 embedding 的 API smoke，也必须运行。
3. 将要发布的真实 demo asset 在上传前通过 `records validate` 和 `index verify`，并记录 hash、index/model 兼容性和构建版本。此项需要真实数据/endpoint 时由维护者执行或明确授权。
4. 仅在授权后上传 PyPI、创建 GitHub Release、上传 assets 并推送 tag。未授权时，完成报告必须写“未执行”，不得暗示已发布。
5. 授权发布后，从干净环境执行 `pip install xists==0.7.0` 并完成最小 happy path；若公开 PyPI 可见性存在延迟，记录实际状态和复查命令。

### 4. v0.7.0 验收核对表

以下命令是最低验收；执行者应按实际项目工具补足，但不得以真实 credential/生产数据替代离线测试。`build` 如未安装，先依照开发/发布文档安装，而不是假定环境天然具备它。

```bash
git diff --check
python -m pytest tests/ -q
python scripts/smoke_check.py
python -m build

python -m venv /tmp/xists-v070-venv
/tmp/xists-v070-venv/bin/python -m pip install dist/xists-*.whl
/tmp/xists-v070-venv/bin/xists --help
/tmp/xists-v070-venv/bin/xists version
```

验收还必须证明：

- 公共 API 与 CLI JSON 使用同一 fixture 时核心字段、排序和 abstention 一致。
- 所有新增 API/CLI 错误路径均不访问网络、不读取 `.env`，并有明确异常或 stderr/exit code。
- `git status` 未暂存/提交 `.env`、token、私有 `data/`、刷新后的 demo files、index/eval/partial/report 生成物。
- main CI 通过。
- 若进入授权发布阶段：Release asset 已经 validate/verify，PyPI 安装验证已完成；否则这两项必须明确标为“未执行（未获发布授权）”。

### 5. 明确禁止（v0.7.0 特有）

- 禁止实现 MCP server、Web UI、HTTP 服务、数据库、ANN 或本地模型下载。
- 禁止为打包/API 任务改动 `query.py` 排名规则、评测 case 或 private 2k/10k 的评分标准。
- 禁止公共 API 调用 CLI/subprocess，或读取 `.env`、环境变量、当前工作目录。
- 禁止未声明地改变 records/index schema、profile prompt、embedding input、模型或 index 格式版本。
- 禁止静默回退或吞掉 embedding/index mismatch。
- 禁止把私有规模评测当公开 benchmark，或默认上传任何 PyPI/GitHub Release 内容。

### 6. v0.7.0 完成报告模板

```md
### v0.7.0 完成报告
#### 前置决定
- LICENSE: [维护者明确决定]
- PyPI / GitHub Release: [授权与实际动作；未授权则写“未执行”]
- Roadmap 版本控制归属: [维护者决定]

#### 已完成
- T1: ...
- T2: ...
- T3: ...
- T4: ...
- T5: ...

#### 公共契约
- API signatures: ...
- CLI JSON 字段与兼容性: ...
- 错误行为: ...
- 兼容性 / rebuild 影响: ...

#### 未完成或偏离（没有则写“无”）
- ...

#### 验收
- pytest: ...
- smoke: ...
- build: ...
- clean venv: ...
- main CI: ...
- Release asset validate/verify: ...
- PyPI install: ...
```

不允许使用“基本完成”“大致可用”等模糊表述。

---

## 14. v0.8.0 执行规格（已审阅）

> 对应章节："v0.8.0 - MCP / agent 集成"。本节冻结 v0.8 的范围、实施顺序和验收标准。基线为 `main` 的 `v0.7.2`；当前 workspace 默认位于 `~/.xists`，公共 Python API 位于 `xists.api`。本版本的目标是稳定地包装既有搜索能力，不重新设计搜索、records 或 index。

### 0. 硬规则

1. 新建分支 `feat/mcp`。每个独立逻辑点一个 Conventional Commit；完成前不得 merge、打 tag、推送 release 或上传 PyPI，除非维护者当次明确授权。
2. MCP SDK 只能作为 optional extra：`pip install "xists[mcp]"`。核心安装 `pip install xists` 的运行时依赖仍然只能是 `numpy`。
3. 不手写 stdio JSON-RPC 或 MCP 协议实现；必须使用维护中的 MCP Python SDK。SDK import 必须惰性执行，未安装 extra 时 CLI、公共 API、普通 import 与其他所有命令保持可用。
4. MCP server 必须复用 `xists.api.load_index` 与 `xists.api.search`，不得启动 `xists search` 子进程，不得复制 ranking 逻辑，不得改变 `query.py` 的排名行为。
5. 只有 CLI/MCP 装配层可加载 workspace `.env` 与环境变量；公共 API 保持显式注入 `EmbeddingConfig` 的无副作用契约。
6. 不实现 HTTP server、Web UI、daemon、远程多租户服务、数据库、ANN、后台同步、hot reload、多 index 管理或本地模型下载。
7. 不提交 `.env`、token、私有 records/index/eval 数据、`data/scale-*`、MCP client 本地配置或真实模型调用结果。

### 1. 真实基线与接口边界

| 能力 | v0.7.2 位置 | v0.8 处理 |
|---|---|---|
| 公共 index 加载 | `xists.api.load_index` | MCP 启动时加载一次，错误转换为可行动的 MCP 错误 |
| 公共搜索 | `xists.api.search` | `search_projects` 的唯一搜索实现 |
| 显式 embedding 配置 | `xists.search.embed.EmbeddingConfig` | MCP 装配层从 workspace 配置构造后显式传入 API |
| workspace / `.env` 加载 | `xists.workspace`、CLI main | MCP CLI 入口复用同一配置优先级：shell > cwd `.env` > workspace `.env` |
| CLI JSON | `xists search --format json` | MCP 搜索的核心字段、排序、abstention 语义必须一致 |

MCP server 只支持 stdio transport。用户入口固定为：

```bash
xists mcp
```

server 在启动时加载 index 和配置；用户重建或替换 index 后重启 server 生效。本版本不承诺运行时刷新。所有协议日志与诊断必须写 stderr，stdout 专供 MCP transport。

### 2. 任务分解

#### T1 - Optional extra 与 MCP 启动入口

**目标**：提供可安装、可发现、不会破坏核心包的 MCP 入口。

**实现要求**：

1. 在 `pyproject.toml` 定义 `mcp` optional extra，使用一个经过测试的 MCP Python SDK 版本范围；不得把它加入基础 `dependencies`。
2. 增加 `xists mcp` 命令。它在启动 server 前解析 workspace、加载配置、验证 index 与 embedding 配置；不得把 protocol output、banner 或颜色写到 stdout。
3. 未安装 extra 时，命令以非零状态退出，stderr 明确包含 `pip install "xists[mcp]"`；`xists --help`、`xists doctor` 和 `import xists.api` 不依赖 MCP SDK。
4. 启动失败时，缺 workspace/index/config、模型与 index 不匹配、endpoint 初始化失败等错误必须复用或保留现有的可行动原因，不能伪装为空搜索结果。

**测试**：覆盖 extra 缺失、正常 CLI parser/help、配置/文件失败路径；所有测试不得启动真实 MCP client、访问网络或读取真实 `.env`。

#### T2 - 冻结 MCP 工具契约

**目标**：使 agent 获得候选项目理解包，而不是仅有链接和分数。

首批稳定工具：

1. `search_projects(query: str, top_k: int = 10)`：调用公共 API，返回 `query`、`query_intent`、`abstained`、`results`。结果保留 CLI/API 已有的 `repo_id`、`url`、`summary`、`confidence`、`score`、`why`，并在已有 profile 数据存在时提供 `use_cases`、`capabilities`、`best_for`、`not_for`、`related` / `replaces`。不得臆造缺失字段。
2. `inspect_project(repo_id: str)`：返回单个项目的完整、受限 profile。其数据必须与已加载 index 对应；实现前须核实 index 是否已含足够 profile。若只能读取 `records.json`，必须在启动时验证 records/index 的对应版本或 fingerprint，并在不一致时失败，而不是返回陈旧记录。
3. `index_stats()`：返回 index 的 schema/version、record count、embedding model、dimension 和可公开的统计信息；不得返回 embedding 向量、token、完整 README 或其他大 payload。

`top_k` 的默认值为 10，接受范围必须有明确上限并在 API/SDK 边界验证。所有工具返回 JSON-compatible 的结构化结果；错误必须区分 invalid input、configuration/index problem 与 upstream embedding failure。

**测试**：使用 fixture index、fixture records 和 mock embedding 配置，证明 `search_projects` 与公共 API/CLI JSON 的核心字段、结果排序和 abstention 一致；覆盖空 query、非法 top_k、缺失 repo、records/index 不一致及 index 损坏。

#### T3 - Server 装配与端到端离线测试

**目标**：确认工具注册、stdio transport 和错误边界真实可用，而不仅是普通 Python 函数测试。

**实现要求**：

1. 采用 MCP SDK 支持的内存或测试 transport 调用 tool，不依赖 Claude Code、Cursor、Cline 或真实网络。
2. tool handler 只使用启动期构造的 immutable server state（index、EmbeddingConfig、可验证的 profile source）；不得在每次请求中重新读取 `.env` 或当前目录。
3. stdout 不得包含日志或 ANSI 转义；CLI 的 MCP 进程启动后协议通信正常。
4. 保持普通 `xists search`、`doctor`、Python API 与所有现有 CLI 命令行为不变。

**测试**：新增 MCP 专项测试；完成后运行完整 pytest、既有 smoke check 和安装后的 MCP smoke。真实 endpoint 一律 mock。

#### T4 - 用户接入文档

**目标**：用户可以从 `pip install` 到 agent 调用完成一次可复现搜索，并在问题发生时用 CLI 排查。

**文档要求**：

1. README 与 `docs/usage.md` 说明 `pip install "xists[mcp]"`、`xists init`、数据/index 前置条件和 `xists mcp`。
2. 提供 Claude Code、Cursor、Cline 的最小 stdio 配置示例；示例中不包含 token 或绝对的用户私有路径。
3. 说明 MCP 读取的默认 workspace、配置优先级、index 更新后必须重启 server，以及远端 embedding endpoint 会收到查询文本。
4. 明确给出 CLI 复现命令：`xists search "<query>" --format json`。
5. 文档不得承诺所有查询都会命中；当 `abstained: true` 时，agent 应诚实处理未找到足够可信匹配的状态。

#### T5 - 发布候选与授权发布

**前置条件**：T1-T4 完成、所有验收通过、维护者明确批准 merge、`v0.8.0` tag、GitHub Release 与 PyPI 上传。

**执行要求**：

1. 版本源、wheel metadata、`xists version`、tag 和发布说明均为 `0.8.0`。
2. 在新建临时 venv 分别安装核心 wheel 和 `.[mcp]`，证明核心 CLI 不受 optional extra 影响，MCP 入口可启动并通过离线 smoke。
3. 构建物不得包含 token、workspace、私有数据、测试缓存或本地 MCP 配置。
4. 发布后从干净环境执行 `pip install "xists[mcp]==0.8.0"`，完成离线 MCP smoke；公开 PyPI 可见性延迟须如实记录。

### 3. 验收核对表

```bash
git diff --check
python -m pytest tests/ -q
python scripts/smoke_check.py
python -m build

# 新建临时 venv：分别验证基础 wheel 与 MCP extra
python -m venv /tmp/xists-v080-venv
/tmp/xists-v080-venv/bin/python -m pip install dist/xists-*.whl
/tmp/xists-v080-venv/bin/xists --help
/tmp/xists-v080-venv/bin/xists doctor --help

/tmp/xists-v080-venv/bin/python -m pip install "xists[mcp]"
/tmp/xists-v080-venv/bin/xists mcp --help
```

完成报告必须包含：依赖版本、工具 schema、CLI/API/MCP 一致性证据、基础 wheel 与 extra wheel 的安装结果、全部离线测试结果、数据兼容性结论，以及未执行的外部发布动作。不得使用“基本完成”“大致可用”等模糊表述。

---


## 15. v0.8.2 至 v1.0.0 整改与发布执行规格（2026-07-26）

> 本节取代“完成 v0.8.0 后直接发布 v1.0.0”的旧假设，是当前有效的版本发布规格。历史章节继续保留，用来解释项目决策演进。

### 15.0 执行硬规则

1. 修复搜索行为前先增加能复现问题的回归测试，再修改实现。
2. 正确性、中文 query analysis、性能、索引格式、checkpoint 和纯重构分开提交。
3. 不为单个 dev/holdout case 添加项目专用规则、特殊仓库列表或不可解释的 magic number。
4. 不未经 dev/holdout 对比就改变默认 embedding、profile、reranker、阈值或 canonical 配置。
5. 不删除未跟踪的 `data/` 实验资产；先分类，再由维护者决定保留、忽略或清理。
6. 不使用 `git add .` 或 `git add -A`。所有提交只按明确路径暂存，并检查 `git diff --cached --name-only`。
7. 本地 `PROJECT_AUDIT.md` 只作为问题来源，**不得提交，也不得进入 wheel、sdist、GitHub Release asset 或文档包**。
8. `.env`、token、私有 query、大规模 records/index、partial checkpoint 和 diagnostic 输出不得提交。
9. 每个 release 必须由独立的 `chore: prepare vX.Y.Z` commit 收口；未经维护者明确授权，不执行 tag push、GitHub Release 或 PyPI 上传。
10. 每个提交保持测试绿色；不得提交“先红后绿”的中间状态到共享分支。

### 15.1 当前基线

- 当前公开软件版本：`0.8.1`
- Git tag：`v0.8.1`
- record schema：`RECORD_SCHEMA_VERSION = 2`
- embedding input：`EMBEDDING_INPUT_VERSION = 3`
- index format：`INDEX_VERSION = 3`
- Python：3.11+
- 当前已知全量测试基线：307 passed
- 2k 内部 holdout：Recall@1 约 50%—56.7%，Recall@5 约 73.3%—78.3%
- 10k 内部 dev：较好结果约 Recall@1 46.7%，Recall@5 75.0%
- 10k 单查询 `rank()`：约 1.56s；CLI 本地排序与加载合计约 3.1s，另加 endpoint latency

这些数字是内部审查时的本地基线，不得包装成第三方可复现的公开性能承诺。正式公开结论必须按 v0.9.0 的基线文档要求提供 corpus/case 说明、运行命令和环境信息。

### 15.2 v0.8.2 — 搜索正确性与索引兼容性补丁

#### 目标与范围

用最小范围修复当前搜索错误，不混入中文分词、性能或持久化架构改造。

必须完成：

1. 将 identity evidence 分为 `repo_id`、`exact_value`、`contextual_name_mention` 和 `none`。
2. `repo_id` 和 `exact_value` 属于强身份；`contextual_name_mention` 只能作为较弱证据，不得无条件 pin 到第一名。
3. 修复“轻量级 Node.js Web 应用框架”把 `nodejs/node` 压过语义上更合适的 `expressjs/express` 的行为。
4. 保留“Kubernetes 云原生容器编排平台”等明确项目提及的可靠命中。
5. 搜索入口验证 `index_version`、vectors 类型、record/vector 数量、dimension 和向量维度。
6. 兼容性问题必须给出明确、可行动错误，不泄漏 `KeyError`、`IndexError` 或含糊异常。

测试至少覆盖：Node.js 生态词、Kubernetes 明确提及、完整 `owner/repo`、owner 片段、Python/React/Rust 等生态词、semantic 明显更高的候选、缺失/过期 index version、非 list vectors、数量不一致和非法维度。

明确不做：完整 CJK tokenizer、`PreparedIndex`、NumPy 排序重写、磁盘 index 格式变更、checkpoint 重写、CLI 拆分，以及默认 embedding/profile/reranker/threshold 变更。

#### 验收与提交

```bash
python -m pytest tests/test_search.py -q
python -m pytest tests/ -q
python scripts/smoke_check.py
python -m compileall -q src scripts tests
python -m pip check
```

此外完成一次冻结的 2k dev/holdout 对比：目标错误修复；Recall@1/Recall@5 不出现无法解释的整体退化；任何指标变化都记录失败分类。

```text
fix(search): distinguish contextual mentions from exact identity
fix(search): validate index version and vector structure
chore: prepare v0.8.2
```

### 15.3 v0.9.0 — 中文检索与正式检索基线

#### 必须完成

1. 保留 ASCII 技术词解析，单独提取连续 CJK 文本。
2. 使用轻量 CJK bigram/trigram 或结构化短语提取，不立即增加重型分词依赖。
3. token 去重并限制数量/长度，避免单字和 n-gram 爆炸。
4. 纯中文查询不再普遍得到空 `keywords` 和 `specificity = 0.0`。
5. 中文 term 参与 query intent、metadata overlap、specificity、matched terms 和解释输出。
6. 保持 `C++`、`C#`、`.NET`、`Node.js`、`owner/repo` 的既有解析能力。
7. 冻结并区分 2k dev、2k holdout、10k dev、10k holdout；dev 用于调参，holdout 只用于阶段验收。
8. 确定 canonical retrieval configuration，记录 embedding、profile、multiview、metadata 权重、reranker、candidate top-k 和 threshold。
9. 新增 `docs/current-retrieval-baseline.md`，记录数据版本、Git commit、命令、环境、Recall@1、Recall@5、MRR、运行时间和已知失败类型。
10. 修复新评测报告中残留旧 `xists_version` 的问题。

测试至少包括：`自托管大语言模型应用界面`、`轻量级 Node.js Web 应用框架`、`Kubernetes 云原生容器编排平台`，以及含中文标点、重复词和空白的查询；覆盖 exact、functional、ecosystem、ambiguous、no-result 和中文 query family。

#### 发布门槛

- 全量测试、smoke、compileall、pip check、wheel/sdist 构建全部通过；
- 纯中文 query intent 有可用 term，不再系统性为空；
- 2k dev/holdout 达到冻结基线，或对下降给出维护者认可的失败归因；
- 10k 至少完成 dev 验证，未完成 holdout 时必须如实记录；
- 默认配置和实验配置清楚区分；
- 文档版本和实际 `__version__` 一致。

```text
feat(search): add lightweight CJK query term extraction
fix(eval): record current retrieval configuration
docs(eval): document the canonical retrieval baseline
chore: prepare v0.9.0
```

### 15.4 v0.9.x — 稳定窗口

允许进入：CJK token 边界/标点/去重修复、ranking tie-break、错误信息、文档、评测元数据，以及不改变磁盘格式的小范围指标回归修复。

不得进入：新索引持久化格式、大规模 `rank()` 重写、checkpoint 架构替换、CLI 大拆分和公共 API 类型重设计。可以不强行发布 `v0.9.1`；只有出现需要交付的兼容修复时才发布 patch。

### 15.5 v0.10.0 — 性能、索引和 checkpoint 架构

#### 阶段 A：PreparedIndex 和统一 NumPy 路径

1. 引入内存态 `PreparedIndex`，加载时一次完成 vector 解码、NumPy matrix/normalization、identity mapping、metadata token 缓存和维度校验。
2. `rank()` 与 `rank_many()` 复用同一核心。
3. 增加 top-k 顺序、分数容差、稳定 tie-break、零向量、空索引、dimension mismatch 和 metadata rerank 的 parity 测试。
4. 保存 2k/10k before/after：load、cold query、warm query、rank_many、CLI 总耗时和 peak memory。

#### 阶段 B：持久化和 checkpoint

1. 比较 metadata JSON + `.npy`、memmap、manifest + shards 等方案，以测量结果选择格式。
2. 改变磁盘格式时提升 `INDEX_VERSION`，提供兼容读取、转换工具或明确 rebuild 指引。
3. 用 append-only 临时分片或 versioned manifest 替代每 N batch 全量重写。
4. checkpoint 必须原子提交，能从最后完整 batch 恢复并检测截断/损坏。
5. 正确性与磁盘格式分开提交；若阶段 B 风险过高，允许延后到 `v0.11.0`。

#### 发布门槛

- 2k/10k 指标不低于 v0.9.x canonical baseline；
- 排序 parity 通过，浮点差异被量化；
- warm query 有可复现改善，同时报告 cold load 和 peak memory；
- checkpoint 中断恢复通过；
- 迁移/重建提示可行动。

```text
perf(search): reuse prepared NumPy index data
feat(index): add versioned vector storage
fix(index): make checkpoint recovery atomic
docs(index): document index format and migration
chore: prepare v0.10.0
```

### 15.6 v1.0.0rcN — 契约冻结与候选验证

不得从 `v0.10.0` 直接发布 `v1.0.0`。至少发布一个符合 PEP 440 的候选版：Python package 使用 `1.0.0rc1`，Git tag 使用 `v1.0.0rc1`。

RC 前必须完成：

- 公共 Python API、错误契约、CLI 参数/default 和 JSON schema 冻结；
- record/index/eval schema 生命周期和迁移策略冻结；
- CLI 领域拆分完成，主入口只负责装配、注册和统一异常处理；
- search/index/eval 边界的核心 `dict[str, Any]` 收敛为明确类型；
- demo artifacts 与当前 schema 一致，README 演示可在干净环境运行；
- lint、type、coverage 和 Python 版本矩阵进入 CI；
- 2k/10k dev/holdout 结论完整；
- wheel/sdist、基础安装、MCP extra、CLI、Python API 和 MCP 离线 smoke 通过。

RC 阶段只修 release blocker。如 rc1 有阻断问题，发布 `1.0.0rc2`，不得覆盖或移动公开 tag。

### 15.7 v1.0.0 — 稳定承诺

`v1.0.0` 不代表长期 TODO 全部完成，而代表：CLI/Python API 兼容边界清楚；CLI JSON、record schema、index format、eval report 有版本策略；默认检索配置有正式基线；1k—10k 经过质量和性能验证；用户能判断何时 refresh、rebuild 或 migrate；RC 无未处理 release blocker。

1.0 后，破坏 CLI JSON、公开 API、record schema 或 index format 的变更必须升 major；兼容新增使用 minor；向后兼容修复使用 patch。

### 15.8 提交与发布流程

版本源是 `src/xists/__init__.py` 中的 `__version__`。release commit 必须同步验证 `xists --version`、wheel metadata、报告中的 `xists_version`、文档当前版本、Git tag 和 Release title。

每次只暂存明确路径。禁止 `git add .` 和 `git add -A`。提交前执行：

```bash
git status --short
git diff --check
git diff --cached --stat
git diff --cached --name-only
```

staged 清单不得出现 `PROJECT_AUDIT.md`、`data/scale-*`、`.env`、`*.partial.json`、`*-diagnostic.json`。

发布验证：

```bash
python -m pytest tests/ -q
python scripts/smoke_check.py
python -m compileall -q src scripts tests
python -m pip check
python -m build
python -m twine check dist/*
```

构建后检查 wheel/sdist 内容，并在临时 venv 安装实际 wheel，验证 `xists --version`、`xists --help`；含 MCP 的版本还要验证 `xists[mcp]` 和离线 MCP smoke。未经维护者授权，不执行 tag push、GitHub Release 或 PyPI 上传。

### 15.9 版本选择速查

| 变化 | 版本选择 | 示例 |
|---|---|---|
| 向后兼容 bugfix、错误信息或文档修复 | patch | `0.8.1 → 0.8.2` |
| 新增中文检索等用户可感知能力 | minor | `0.8.2 → 0.9.0` |
| 0.x 阶段的新持久化格式并有迁移说明 | minor + 提升 `INDEX_VERSION` | `0.9.x → 0.10.0` |
| 1.0 契约冻结验证 | prerelease | `1.0.0rc1` |
| 首次稳定兼容承诺 | major | `1.0.0` |
| 1.0 后破坏公开契约 | major | `1.x → 2.0.0` |

### 15.10 完成报告模板

```text
### vX.Y.Z 完成报告

#### 范围
- 本版本完成：
- 明确未做：

#### 行为与兼容性
- 用户可见变化：
- records/index 是否需要 refresh/rebuild：
- 软件/schema/index 版本：

#### 质量与性能证据
- pytest / smoke / compileall / pip check：
- dev / holdout：
- 已知失败分类：
- load / cold / warm / rank_many / peak memory（适用时）：

#### 构建与安装
- wheel/sdist / twine / 临时 venv / MCP extra：

#### 提交隔离
- staged 文件已审查：
- PROJECT_AUDIT.md 未提交：
- data 实验产物未提交：

#### 外部动作
- tag / GitHub Release / PyPI：
```

## 附录 A：未排期的长期 TODO

### 建立公开、可复现的检索回归评测集

**状态**：未排期。它不是 v0.6.1 的发布条件，也不是启动 v0.7.0 的前置条件；等项目需要对外发布可审计性能结论或引入外部贡献者时再安排。

**目的**：让公开 PR、release note 和性能说明中的检索质量结论能够被第三方理解、审阅和复跑，而不只依赖维护者本地的私有评测产物。

**原则**：维护者可以使用本地 2k/10k corpus、冻结 query 和 holdout 做内部验收；这对防止回归很有价值。但若 corpus、case、判定规则或运行命令未公开，诸如 `Recall@5`、`wrong high-confidence` 等数字不能被外部独立验证。因此，私有评测应在 PR 中表述为内部回归结论，不应单独作为公开性能宣称的证据。

**将来完成条件**：

- 提交一个规模可控、无私人 records/index、可在 CI 或普通开发机运行的公开 regression fixture。
- 公开 case 的来源或编写原则、分层方法、`expected` / `acceptable` / no-result 的判定规则，以及 dev 与 holdout 的隔离规则。
- 提供从 records 到 index 再到 eval 的可复制命令；所有外部服务调用必须能以 fixture 或 mock 替代。
- 至少覆盖 exact name、functional、ecosystem、ambiguous、no-result 和中文 query；分层按通用意图设计，禁止按 AI、工程或某一特定领域的得分做特化。
- 在文档中定义并解释公开指标（至少 Recall@1、Recall@5、no-result abstention、错误 high-confidence），并标明每项指标的样本量。
- PR 或 release note 中引用这些数字时，必须同时指出公开评测文件与复现命令；私有评测结果另行明确标注为内部验收。

**边界**：不提交生产 records、embedding index、含 token 的配置、私有 query 或大规模生成产物。公开 fixture 的目标是可审计的回归保护，不是替代维护者在真实 2k/10k corpus 上进行的规模和泛化验证。

---

## 附录 B：项目审查基线（2026-07-26，完整归档）

> 本附录完整归档原 `PROJECT_AUDIT.md` 的审查内容。其问题排期与发布决策以 §15 为准；保留本附录是为了避免问题证据、复现案例和文件定位在删除独立审查文件后丢失。

### xists 项目审查报告

> 审查日期：2026-07-26
> 审查版本：`v0.8.1`
> 审查分支：`main`

#### 一、审查结论

当前项目不是“代码不可用”，而是一个**工程基础较好、功能完整度较高，但检索质量、规模性能和实验资产治理尚未完全收口的 Beta 项目**。

| 维度 | 评价 |
|---|---|
| 代码可运行性 | 良好 |
| 测试完整度 | 良好 |
| 打包发布能力 | 良好 |
| CLI/API/MCP 完整度 | 良好 |
| 搜索正确性 | 中等，存在明确误排序问题 |
| 中文查询支持 | Embedding 可用，但规则层支持不足 |
| 10k 规模性能 | 勉强可用，仍有明显瓶颈 |
| 数据与实验资产管理 | 较乱，尚未收口 |
| 文档一致性 | 部分过期 |
| 1.0 就绪度 | 尚未达到 |

#### 二、已执行的检查

本次没有修改项目代码，主要进行了代码、测试、文档、数据和打包审计。

- 当前版本：`0.8.1`
- 当前分支：`main`
- 与 `origin/main` 差异：`0/0`
- 测试结果：**307 passed**
- `python scripts/smoke_check.py`：通过
- `python -m compileall`：通过
- `python -m pip check`：无依赖冲突
- 从源码重新构建 wheel 和 sdist：成功
- Wheel 元数据和版本一致
- 核心依赖保持为 `numpy`
- MCP 正确作为 optional dependency 提供

项目目前的基础工程质量不错。

#### 三、优先级最高的问题

##### P0/P1：混合中文查询中的项目名会被错误当成“精确仓库匹配”

相关代码：

- `src/xists/search/query.py:278`
- `src/xists/search/query.py:299`
- `src/xists/search/query.py:379`

当前逻辑中，只要中文查询包含一个英文项目名或别名，就可能返回：

```python
"contextual_name_mention"
```

但后续代码直接把所有非 `none` 的身份匹配都视作：

```python
exact_identity = identity_kind != "none"
```

于是“在自然语言中提到某个名字”与“用户明确搜索这个项目”被同等处理，并获得较大的身份加分和置顶资格。

例如查询：

```text
轻量级 Node.js Web 应用框架
```

用户意图显然是找 Express 一类 Web 框架，但 `nodejs/node` 的别名包含 `Node.js`，因此会被认为是身份匹配。最小复现中，即使：

```text
nodejs/node        semantic=0.1
expressjs/express  semantic=1.0
```

最终仍可能得到：

```text
nodejs/node        semantic=0.1  metadata=+0.57
expressjs/express  semantic=1.0  metadata=+0.04
```

真实评测报告中也出现了同类问题：

```text
查询：Node.js Web 框架生态
期望：expressjs/express
结果：nodejs/node
```

建议将身份匹配分级：

1. `repo_id` / 完整别名精确查询：允许置顶。
2. 独立项目名明确提及：有限加分。
3. `contextual_name_mention`：不能视作 exact，更不能无条件置顶。

例如：

```python
exact_identity = identity_kind in {"repo_id", "exact_value"}
contextual_identity = identity_kind in {
    "name_mention",
    "contextual_name_mention",
}
```

其中 `contextual_identity` 最多给小幅加分，并结合查询长度、上下文和语义分数判断。

建议补充以下回归查询：

```text
轻量级 Node.js Web 应用框架
Node.js ORM 框架
React 状态管理库
Python Web 框架
Rust Web 框架
```

##### P1：中文查询在 metadata/query-intent 层基本无法被正确分词

当前 Token 正则是：

```python
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._#-]*")
```

位置：`src/xists/search/query.py:24`。

纯中文内容不会进入：

- keyword 提取
- metadata term overlap
- query specificity
- query intent 分类
- profile term overlap
- 可解释的 `matched_terms`

Embedding 端可能支持中文，因此纯语义召回还能工作，但 metadata 和解释层基本失效。真实报告中存在：

```json
{
  "query": "自托管大语言模型应用界面",
  "query_intent": {
    "type": "functional",
    "specificity": 0.0,
    "keywords": []
  }
}
```

影响包括：

1. 中文查询只依赖 embedding，英文查询则能获得额外 metadata 加分。
2. 中英文检索行为不对称。
3. `query_intent` 和 `specificity` 对中文不可信。
4. `why` 很难解释中文查询为何匹配。
5. 混合中文查询可能触发错误身份置顶。

短期可采用轻量方案：

- 保留 ASCII 技术词抽取。
- 对 CJK 文本做字符 bigram/trigram。
- 对 profile/schema 中的结构化字段做精确短语匹配。
- 将中文 query intent 更多交给 query transform 或 embedding。
- 禁止 CJK 上下文项目名直接进入 exact identity。
- 增加中英文成对评测集。

##### P1：当前真实检索质量仍不足以支撑 1.0“可靠搜索”的定位

一个代表性的 2k holdout 报告结果是：

```text
recall@1:           50.0%
recall@5:           73.3%
serious mismatch:  50.0%
abstain rate:       18.3%
```

较好的 2k holdout 实验大致达到：

```text
recall@1: 约 56.7%
recall@5: 约 78.3%
```

10k 本地 dev 实验中较好的结果大致是：

```text
recall@1: 46.7%
recall@5: 75.0%
```

存在以下不足：

- 10k 报告主要是 dev 结果，缺少清晰、冻结后的 10k holdout 结论。
- profile、multiview、query transform、rerank 实验很多，但没有明确选出正式默认配置。
- 很多报告记录的 `xists_version` 还是 `0.6.0`，当前代码已经是 `0.8.1`。
- 本地实验很多，但缺少一份进入版本控制的最终对比与选择结论。

建议冻结一条正式基线：

```text
Corpus:
  scale-2k
  scale-10k

Dataset:
  dev
  holdout
  no-result
  Chinese/English paired cases

Configuration:
  profile prompt version
  embedding model
  query transform
  candidate count
  reranker
  threshold
  confidence calibration
```

只保留当前正式 baseline、一个 challenger、独立 holdout 和分类误差分析。

#### 四、性能和规模方面的不足

##### P1/P2：单查询搜索路径仍使用逐项 Python cosine

相关代码：

- `src/xists/search/query.py:785`
- `src/xists/search/query.py:825`

当前 `rank()` 每次查询都会：

1. 遍历全部 index entries。
2. 将每个 base64 vector 重新解码。
3. 使用 Python 循环计算 cosine。
4. 对每个 entry 重新提取和扩展 metadata token。

项目性能文档记录：

```text
10k index load：约 1.57 秒
10k rank()：约 1.56 秒
CLI 搜索：约 3.1 秒 + embedding endpoint 延迟
```

而 `rank_many()` 的 NumPy 矩阵路径约为 `0.86 秒`。真正面向用户的单查询路径反而比批量评测路径慢。

建议增加已准备的内存索引：

```python
class PreparedIndex:
    document: dict[str, Any]
    entries: list[IndexEntry]
    matrix: np.ndarray
    normalized_matrix: np.ndarray
    metadata_tokens: list[frozenset[str]]
```

加载时一次性完成 vector 解码、matrix 构造、归一化、metadata token 预计算和 identity variants 预计算。查询时直接执行矩阵乘法。

##### P1/P2：JSON index 在 multiview 实验下已经非常大

当前本地数据规模约为：

```text
整个项目目录：6.8 GB
data/：5.5 GB
```

最大的部分文件：

```text
1.3 GB  multiview partial index
986 MB  10k multiview index
575 MB  10k v3 index
573 MB  10k v2 index
336 MB  10k compact multiview index
230 MB  基础 10k index
```

JSON + base64 在 multiview 下会带来：

- 较高启动解析时间
- 较大内存峰值
- 重复 vector 解码
- checkpoint 写入代价高
- GB 级临时文件

短期可以将格式拆分为：

```text
index.json   # metadata / schema / repo ids
vectors.npy  # contiguous float32 matrix
```

也可考虑 `.npz`、memory-mapped NumPy、SQLite metadata + binary vector blob 或 shard 化索引。

##### P2：Index checkpoint 会周期性重写整个索引

相关代码：

- `src/xists/cli.py:721`
- `src/xists/cli.py:873`
- `src/xists/cli.py:909`

当前每 16 个 batch 重写一次完整 partial index。在数百 MB 到 GB 级索引下会产生很高的重复 I/O。

建议改为：

- 向量分片
- append-only JSONL/vector binary
- SQLite transaction
- 每批独立临时文件，最后合并
- metadata manifest + `.npy` memmap

#### 五、数据和仓库治理问题

##### P1：当前工作区存在 15 个未跟踪实验文件

包括 multiview partial index、profile failure report、repair/retry report、baseline report、canonical query 和多种 diagnostic 文件。

`.gitignore` 尚未覆盖：

```text
*.partial.json
*-baseline.json
*-diagnostic.json
*-failures.json
*-retry.json
*-canonical-queries.json
*-metadata-current.json
```

建议使用更明确的生成物目录：

```text
data/generated/
data/experiments/
data/checkpoints/
```

整体忽略目录，只将需要版本控制的 manifest/eval case 放入白名单目录。

##### P1：本地 `.env` 权限为 `644`

检查结果：

```text
.env                         644
.claude/worktrees/.env       644
```

如果其中存在 GitHub、LLM 或 embedding token，同一机器上的其他用户可能读取。建议：

```bash
chmod 600 .env
chmod 600 .claude/worktrees/.env
```

程序初始化 `.env` 时也可以主动设置 `0600`。

##### P2：根目录存在过期 demo records/index

本地 `demo-records.json` 和 `demo-index.json` 不是当前 schema：

- 200 条 records 都是 schema v1。
- 当前期望 schema v2。
- index version 为 1，当前期望为 3。
- embedding input version 为 2，当前期望为 3。
- 200 个向量都被判定为 stale。

README 已说明当前没有 current-schema demo asset，因此不属于发布内容错误，但根目录保留旧文件容易误导开发者。

建议：

- 删除或归档本地旧 demo 文件。
- 或重命名为 `legacy-demo-*`。
- 文档示例尽量使用 `examples/ci-smoke/` 中已经验证的当前 fixture。
- 新 demo asset 必须通过 `records validate` 和 `index verify`。

#### 六、文档和项目管理问题

##### P2：ROADMAP 状态明显过期

当前版本已经是 `0.8.1`，但 `ROADMAP.md:47` 仍写着：

```text
当前版本为 0.4.0
```

`v0.7.0` 和 `v0.8.0` 在总览中也仍未标记完成，和实际 tag、API、CLI、打包及 MCP 功能不一致。

建议在 ROADMAP 顶部增加实时状态表：

```text
Current release: 0.8.1
Completed: 0.2–0.8
Next milestone: 1.0 readiness
Open gates:
- retrieval quality
- 10k holdout evidence
- performance decision
- stable API/JSON policy
```

##### P2：README 仍用 v0.2.0 描述当前排序行为

README 和 usage 文档仍使用 `v0.2.0 keeps ranking simple...` 描述当前搜索，但现在已经存在 semantic、metadata、rerank、query transform、confidence calibration、RRF fusion 和 MCP。

建议按当前策略名称描述行为，不再用历史版本号描述当前默认逻辑。历史行为放入 release notes。

##### P2：缺少“当前正式实验结论”

本地存在大量实验结果，但缺少一份能够直接回答以下问题的文档：

- 当前推荐 embedding 是什么？
- 当前推荐 profile 版本是什么？
- 是否推荐 multiview？
- 是否推荐 query transform？
- 是否推荐 reranker？
- 默认阈值为什么是这个值？
- 当前 2k/10k holdout 指标是多少？
- 已知最主要的失败类型是什么？

建议增加：

```text
docs/current-retrieval-baseline.md
```

#### 七、代码可维护性问题

##### P2：`cli.py` 过大，职责过多

当前：

```text
src/xists/cli.py：2660 行
tests/test_cli.py：3158 行
```

`cli.py` 同时包含环境加载、workspace、ingest orchestration、checkpoint、profile refresh、index build、search formatting、doctor、records/index 工具、eval 和 parser construction。`build_parser()` 单个函数约 309 行。

建议拆分：

```text
src/xists/commands/
  init.py
  doctor.py
  ingest.py
  profile.py
  index.py
  search.py
  eval.py
  records.py
  parser.py
```

##### P2：部分核心函数仍然过大

例如：

```text
evaluate_dataset           307 行
records_validation_report  125 行
rank_many                  109 行
build_record                87 行
```

建议将 `evaluate_dataset()` 拆成：

```text
run_queries()
classify_results()
calculate_metrics()
build_report()
```

##### P2：“稳定 Python API”仍大量暴露 `dict[str, Any]`

`xists.api.search()` 的输入输出和 index 都是 `dict[str, Any]`，会造成 IDE 无法补全、字段错误只能运行时发现、JSON schema 变更难检测，以及 MCP/CLI/API 输出一致性依赖人工维护。

建议至少增加：

- `TypedDict`
- dataclass/domain models
- JSON schema fixture
- 输出 contract tests

##### P2：Search 没有直接验证 `index_version`

`ensure_index_matches_model()` 当前检查 embedding model、embedding input version 和 record schema version，但没有检查 `index_version`。

建议搜索入口同时检查：

- `index_version` 是否等于 `INDEX_VERSION`。
- `vectors` 是否为 list。
- `record_count` 是否与实际 vector count 一致。
- dimension 是否为正整数。
- repo_id 是否为空。
- 是否存在重复 repo/view 标识。

#### 八、测试与 CI

##### 做得好的地方

测试覆盖面比较广：

- CLI：105 个测试
- Search：55 个测试
- GitHub ingest：32 个测试
- LLM profile：16 个测试
- MCP：8 个测试
- Workspace：10 个测试
- 另有 packaging、eval、rerank、query transform 等测试

本次执行结果：

```text
307 passed in 5.15s
```

##### 仍有不足

CI 当前只有 Python 3.11/3.12、pytest、smoke test 和 package build/install，缺少：

- lint
- import/order 检查
- type checking
- coverage threshold
- dependency audit
- Python 3.13/3.14 CI

建议优先顺序：

1. `ruff check`
2. `pytest --cov`，先记录覆盖率
3. `mypy` 或 `pyright`，先覆盖 `api.py` 和核心数据模型
4. Python 3.13 CI
5. 定期 `pip-audit`
6. 补充 CJK/identity 回归测试

#### 九、建议整改顺序

##### 第一阶段：修复正确性问题

1. 修复 `contextual_name_mention` 被当成 exact identity 的问题。
2. 增加混合中文查询回归测试。
3. 改善中文 query intent/keyword 处理。
4. 搜索入口增加 `index_version` 校验。
5. 冻结一套 2k dev/holdout 基线，确认修复没有引发总体回归。

##### 第二阶段：收口实验与仓库资产

1. 增加 `docs/current-retrieval-baseline.md`。
2. 明确当前正式 profile、embedding、reranker 和 threshold。
3. 补完整 10k holdout 评测。
4. 将生成文件移动到 ignored artifact 目录。
5. 清理 15 个 untracked 文件和过期 demo。
6. 将 `.env` 权限改为 `600`。

##### 第三阶段：优化性能

1. 引入 `PreparedIndex`。
2. 统一 `rank()` 和 `rank_many()` 的 NumPy 打分路径。
3. 缓存 metadata token 和 identity variants。
4. 将向量从 JSON 拆为二进制矩阵。
5. 将 index checkpoint 改为分片或 append-only。

##### 第四阶段：为 1.0 做维护性收口

1. 拆分 `cli.py`。
2. 拆分 `evaluate_dataset()`。
3. 给 API 和输出增加 TypedDict/schema。
4. 更新 ROADMAP 当前状态。
5. 更新 README 中的旧版本描述。
6. 加入 lint、coverage 和 Python 3.13 CI。

#### 十、最终评价

项目目前最大的优势是：

- 功能链路完整。
- CLI/API/MCP 都已经可用。
- 测试数量和质量较好。
- 离线 smoke fixture 完整。
- 打包发布流程可靠。
- Schema/index 版本意识较强。
- 对失败恢复、断点续跑和错误提示投入充分。

最大的不足是：

1. 身份匹配规则存在明确误排序。
2. 中文查询只有 embedding 层相对可靠，规则和解释层较弱。
3. 真实 holdout 检索质量还不够稳定。
4. 10k/multiview 下 JSON 索引性能和体积已经接近架构边界。
5. 实验很多，但正式基线与结论尚未收口。
6. ROADMAP、demo 资产和当前版本状态不同步。

下一步不宜继续横向增加功能，建议集中完成：

> **搜索正确性修复 → 冻结正式评测基线 → 10k holdout 验证 → 索引内存化/矩阵化 → 文档与资产收口。**

完成这些后，项目才比较适合进入 `1.0.0` 稳定发布阶段。
