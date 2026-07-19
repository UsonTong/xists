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

> **状态更新（2026-07-17）**：v0.2.0、v0.3.0、v0.4.0 已完成，当前版本为 `0.4.0`。本节以下内容描述的是 roadmap 制定时（`0.1.0`）的状态，作为决策背景保留，其中的问题（query.py 过度复杂等）已在对应版本中解决。

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
v0.2.0  清理搜索，夺回控制权              [已完成]
         ↓
v0.3.0  Schema v2，把智能转移到数据层      [已完成]
         ↓
v0.4.0  数据质量工具，让数据源可维护       [已完成]
         ↓
v0.5.0  本地规模与 index 稳定              [已完成]
         ↓
v0.6.0  规模化 ingest 与数据更新           [已完成]
         ↓
v0.7.0  稳定 Python API + 优秀 CLI + 打包首发
         ↓
v0.8.0  MCP / agent 集成
         ↓
v1.0.0  稳定发布
```

每个版本都应该有：

- 明确目标
- 为什么现在做
- 不做什么
- CLI 体验要求
- 验收标准

**执行规格约定**：v0.5.0 / v0.6.0 的逐任务执行规格见 §11 / §12。v0.7.0、v0.8.0 与 v1.0.0 的执行规格（§13 起）**尚未编写，且刻意不提前编写**——执行规格中的"现状盘点"必须基于开工时的真实代码状态，提前写只会产出过期引用。规则：每个版本开工前，必须先由维护者（或在维护者审阅下）按 §11/§12 的结构补写对应执行规格（现状盘点、任务分解、明确禁止、验收核对表、完成报告模板）；执行规格未补写并通过审阅前，任何 agent 不得开工实现该版本。只有验收清单（§5.7-§5.9）不足以作为开工依据。

---

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
- `ROADMAP.md` 状态或后续说明（如果实现偏离本路线）。ROADMAP.md 已纳入本仓库（2026-07-17 起，仓库根目录），执行 agent 应直接更新它，并在完成报告中单列"偏离 roadmap 的事项"供维护者审阅。

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
- **缓解**：这正是本 roadmap 作为决策文档存在的理由——"为什么"和验收清单让任何 agent 或维护者可接手；1.0 前核心包保持零重依赖（MCP SDK 仅为可选 extra，见 v0.8.0 依赖策略）和纯本地文件架构，把项目"复活成本"压到最低。本文档自身必须纳入版本控制——它是接手的前提，丢了它其余缓解都失效。2026-07-17 起本文档已随 xists 仓库版本控制（仓库根目录 `ROADMAP.md`），不得再退回单机目录维护。

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

- **2026-07-19** — 2k 实验诊断后续：将 LangChain/LlamaIndex、agent framework、关系型数据库和 data store 四个过窄的评测预期改为有理由的 acceptable alternatives；修复完整 `owner/repo` 嵌入自然语言时 identity 漏检、混合中文查询误把 ASCII 短名称当 identity，以及 no-result 分数被标为 high confidence 的通用行为。未添加任何项目/case 特判。诊断记录见 `docs/scale-2k-diagnosis.md`。配置的 embedding endpoint 当前不可用，因此新的 2k 报告尚未实测；10k 实验保持阻塞，直到该报告完成并审阅。

- **2026-07-19** — v0.6.0 完成，§3 标记 [已完成]。T1 为 profile refresh 增加 JSONL checkpoint 和 `--resume`；T2 增加 ingest/profile dry-run；T3 为两者补齐失败隔离、报告和 `--retry-failed`；T4 增加 GitHub rate-limit reset 等待，并将 ingest checkpoint 从逐条重写完整 JSON 快照改为追加式 JSONL（含 resume 和截断尾行恢复）；T5 增加 recall@1/@5；T6 增加规模实验手册。运行时依赖未增加，schema/version 常量未变，`query.py` 在 v0.6 提交和验收改动中均为零行变化。完整验收证据见 `docs/v0.6.0-completion.md`；2k 实验结果属于后续 v1.0 发布前证据，不以其排名分数作为本版本收版条件。

- **2026-07-17** — v0.5.0 完成，§3 标记 [已完成]。按 §11 执行：T1 四项兼容检查核实补全（model/dimension/input_version/schema_version 检查均已存在，新补"index 缺 embedding_model 字段静默通过"的报错与 4 个 mismatch 测试）；T2 `scripts/generate_synthetic_index.py`；T3 `docs/performance.md`（1k/10k 基线，AMD Ryzen 7 7735H）；T4 `index stats` 增加 `estimated_memory_mb`；T5 `tests/test_performance_smoke.py`。两项需维护者知悉：(1) T3 按其"重复工作"例外条款给 `query.py` 的 `_expanded_token` 加了 `lru_cache`（与 `_tokenize`/`_keyword_tokens` 同惯例，纯 memoization 零打分变化，全部测试不变通过），10k 核心搜索从 2.28s/1.62s 降至 1.56s/0.86s（rank/rank_many）；(2) **待决策**：`rank_many`（numpy 矩阵路径）10k 核心 0.86s 达标，但 CLI `search` 实际使用的单查询 `rank()` 用纯 Python cosine 循环打分，10k 时 1.56s 超过 §11 T3 的 1 秒线——非 O(n²)，是线性但常数大的标量实现。统一 `rank()` 到矩阵路径可解决，但 float32 矩阵与 float64 标量的分数差异约 1e-4，属打分行为变更，依 T3 规定不擅自优化，留待维护者裁决（详见 docs/performance.md）。

- **2026-07-17** — 第四次修订（对照代码逐条核查）。(1) 修正 §12 现状盘点：多线程 ingest 实际已在每个 future 完成后写 checkpoint，早前"全部完成后才写"的判断源自 `cli.py` 一行过期注释而非代码行为（该注释已同步改正）；(2) §12 T4 第 5 点从"多线程 checkpoint 空洞二选一"改写为"ingest checkpoint 快照式 O(n²) 写入的 JSONL 追加式改造"，消除与本节禁止事项"禁止每条重写完整快照"的自相矛盾——第二次修订只治了 profile refresh，漏了写入量问题完全相同的 ingest，若不改，v0.6.0 将带着 O(n²) checkpoint 去实现"2 万级 corpus 生产"的目标；(3) §11 硬规则 3 明确 T1 涉及的兼容检查函数不在 query.py 只读范围内，消除可能让执行 agent 卡死的字面歧义；(4) §11 T4 内存估算从 float64（×8 字节）改为 float32（×4 字节），与 `_normalized_matrix` 实际 dtype 一致；(5) 本文档自维护者单机目录（~/Downloads）移入 xists 仓库根目录纳入版本控制，落实 §8"单人维护"条目对本文档自身的要求，§5.1/§6 中"roadmap 不在仓库中"的偏离上报流程同步更新为直接修改。原因：v0.5.0/v0.6.0 开工前按 §11/§12 自身规则逐条核实现状盘点，发现两处与代码不符（会触发 agent 依规停工）及一处未被任何任务覆盖的规模瓶颈。

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
   - `src/xists/search/query.py` 的排序/打分逻辑（第 300 行以后的 ranking 部分）。例外：T1 涉及的搜索前兼容性检查（`ensure_index_matches_model` 及相邻检查逻辑）允许按 T1 修改，不受本条限制
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
1. 在 `_index_stats_report` 中计算：`vector_count × dimension × 4 / 1024 / 1024`（搜索时矩阵为 float32，见 `query.py` 的 `_normalized_matrix`），保留 1 位小数。
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
| ingest 各模式（单线程 / 多线程 / GraphQL 批量）均在每条或每批完成后写 checkpoint；但写法是快照式全量重写（每次 `write_json(args.output, merged)`），O(n²) 写入问题见 T4 | `cli.py` ingest 主循环 |
| HTTP 429/5xx 指数退避重试 | `ingest/github.py` `RETRYABLE_HTTP_STATUSES` + `2**attempt` |
| 多 token 轮换（TokenPool，GITHUB_TOKENS） | `ingest/github.py` |
| GraphQL 批量模式（低配额消耗）+ rateLimit 查询 | `ingest/github.py` |
| ingest 进度输出 + `--report` 失败报告文件 | `cli.py` |
| profile refresh 默认只刷新过期记录（prompt version 判断），`--force` 全量，`--only-missing-search-text` 过滤 | `cli.py` + `records.py` |
| index build 有 checkpoint（`_index_write_checkpoint`） | `cli.py` ~344 |
| eval run + `--llm-judge`（top-1 不一致时 LLM 成对裁决） | `eval/` |
| 分层 eval 生成脚本 | `scripts/generate_stratified_eval.py` |

**因此 v0.6.0 的实际差距是**：profile refresh 的断点续跑、ingest checkpoint 的追加式改造（现为快照式 O(n²)，见 T4）、dry-run 预估、失败隔离与只重试失败项、GitHub 限流的"等到重置"策略（现在只有指数退避，长任务遇到配额耗尽会失败）、recall@k 指标、以及阶梯实验的操作手册。以下任务只做差距部分。

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
5. **ingest checkpoint 的 O(n²) 写入必须在本任务一并处理**。现状核实（2026-07-17）：所有 ingest 模式（单线程、多线程、GraphQL 批量）都已在每条/每批完成后写 checkpoint（`write_json(args.output, merged)` 在 `as_completed` 循环内；曾误导的"write checkpoint after all complete"注释已于本次修订时改正）。真正的问题是写法：每次全量重写 output JSON 快照。这正是 T1 为 profile refresh 论证过要禁止的 O(n²) 模式（20k repos × ~7KB/条 ≈ 每次 140MB 序列化，累计 ~1.4TB 写入），而大规模 ingest 恰恰是本版本的目标场景。处理方式：把 ingest checkpoint 改造成与 T1 相同的 JSONL 追加设计（`<output>.partial.jsonl`，每成功一条 append 一行，完成后组装最终 records.json 并原子写、删除 partial；`--resume` 语义与 T1 一致——ingest 现有的"已在 output 中则跳过"增量逻辑保留，partial 中的条目同样计入已完成）。多线程模式下 append 必须线程安全（由主线程在 `as_completed` 循环中写入即可，与现有结构吻合）。必须有测试：mock 中断后 partial 含已完成条目、`--resume` 续跑不重复抓取、末行截断可容忍。

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
