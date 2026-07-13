# Evaluation Analysis

This note records the current demo retrieval baselines and the main follow-up items for search-quality tuning.

## Baseline reports

| Report | Dataset | Cases | Exact top-1 | Effective/acceptable top-1 | Serious mismatch |
| --- | --- | ---: | ---: | ---: | ---: |
| `demo-eval-report.json` | `xists-baseline-100` | 100 | 89.0% | 100.0% | 0.0% |
| `demo-eval-report-extended.json` | `xists-baseline-112` | 112 | 88.4% | 100.0% | 0.0% |

The extended report was generated against `demo-index.json` with the configured `BAAI/bge-m3` embedding endpoint.

## Quality gate

The current smoke/regression gate is:

```bash
python scripts/check_eval_report.py demo-eval-report.json
python scripts/check_eval_report.py demo-eval-report-extended.json
```

Default thresholds intentionally protect the small demo baseline rather than claim broad production quality:

- exact top-1 must be at least `0.88` using `exact_top1_rate` or `exact_hit_at_1`.
- effective top-1 must be `1.0` using `effective_top1_rate` or `acceptable_hit_at_1`.
- serious top-1 mismatch must remain `0.0` using `serious_top1_error_rate`.

This gate is a first safety net. It prevents obvious regressions while the search system grows toward much larger indexes and richer ranking stages.

## Extended report findings

Commands used:

```bash
xists doctor --records demo-records.json --index demo-index.json --cases examples/eval-cases-extended.json --check-endpoints --strict
xists eval run --cases examples/eval-cases-extended.json --index demo-index.json --output demo-eval-report-extended.json
xists eval inspect --report demo-eval-report-extended.json --status serious_mismatch
xists eval inspect --report demo-eval-report-extended.json --status acceptable
```

Summary:

- Exact top-1: 99/112 cases.
- Effective/acceptable top-1: 112/112 cases.
- Serious mismatch: 0/112 cases.
- Acceptable-but-not-exact: 13/112 cases.

### Fixed gap

The previous `zed-modern-editor` miss is now fixed: `modern rust based editor`
returns `zed-industries/zed` at top-1 instead of letting the broad Rust language
repository outrank the editor/product cue.

## Acceptable-but-not-exact cases are diagnostic samples

The 13 acceptable-but-not-exact cases should not be hard-coded one by one. They are diagnostic samples that reveal where a larger index will create many more near-neighbor conflicts. At 10k-100k repositories, every broad category will contain many plausible alternatives, so the ranking system needs general capabilities: intent detection, structured metadata matching, capability-term evidence, entity/name handling, and result grouping.

| Category | Representative cases | Scalable risk | General ranking capability needed |
| --- | --- | --- | --- |
| Framework positioning confusion | `fastapi-api-framework` expected `fastapi/fastapi` but got `django/django`; `django-web-framework` expected `django/django` but got `pallets/flask`; `nestjs-node-framework` expected `nestjs/nest` but got `expressjs/express` | More frameworks make “web framework” matches semantically dense; API-first, batteries-included, minimal, dependency-injection, and full-stack cues can get flattened. | Extract framework positioning cues from query and metadata: API-first, admin/ORM/full-stack, micro/minimal, dependency injection, async, OpenAPI, TypeScript/Node/Python ecosystem. |
| AI coding-agent confusion | `codex-terminal-agent` expected `openai/codex` but got `anthropics/claude-code`; `openhands-agent` expected `OpenHands/OpenHands` but got `msitarzewski/agency-agents` | Many agent tools share “AI coding assistant/agent” language; terminal, autonomous, open-source, pair-programming, and provider-specific identity need separation. | Detect product form and operating mode: terminal agent, autonomous SWE agent, pair programmer, IDE assistant, hosted service, open-source constraint, provider/entity cue. |
| LLM serving/runtime confusion | `vllm-local-serving` expected `vllm-project/vllm` but got `ggml-org/llama.cpp`; `dify-llm-platform` expected `langgenius/dify` but got `langchain-ai/langchain` | LLM infrastructure repos overlap heavily; local inference, production serving, app platform, orchestration library, and model runner are different jobs. | Add capability facets for serving/runtime/app-platform/library/local-runner: throughput, OpenAI-compatible server, GGUF/local inference, workflow/app builder, chains/agents SDK. |
| Web extraction and automation confusion | `firecrawl-markdown` and `firecrawl-web-scraping` expected `firecrawl/firecrawl` but got `unclecode/crawl4ai`; `n8n-workflow-automation` expected `n8n-io/n8n` but got `langgenius/dify` | Web extraction, scraping, workflow automation, and AI app workflow tools share “LLM-ready”, “automation”, and “platform” terms. | Distinguish extraction target and workflow model: crawler/scraper to markdown, browser automation, low-code workflow, AI app workflow, integrations ecosystem. |
| Fork/replacement or close-substitute confusion | `yt-dlp-download` expected `yt-dlp/yt-dlp` but got `ytdl-org/youtube-dl` | Forks and successors can be acceptable but users may expect the actively maintained successor or named entity. | Track lineage and replacement relationships separately from semantic similarity: fork-of, successor-of, maintained status, exact repo/name entity boost. |
| Exploratory ambiguous UI/tooling queries | `lobe-chat-ui` expected `lobehub/lobehub` but got `ChatGPTNextWeb/NextChat`; `comfyui-image-workflow` expected `Comfy-Org/ComfyUI` but got `AUTOMATIC1111/stable-diffusion-webui` | Broad UI queries often have multiple valid answers; forcing a single exact answer can overfit and degrade exploration quality. | Represent ambiguity explicitly: show top-k grouped by product form, include why-different notes, optimize coverage/diversity and nDCG rather than only exact top-1. |

## Implications for the next ranking pass

The next ranking pass should preserve `serious_top1_error_rate = 0.0` while making the search output more diagnostic. Ranking changes should be driven by reusable evidence rather than fixture-specific exceptions:

- Query intent: exact-name, alternative, domain, functional, and ambiguous exploratory queries need different success criteria.
- Capability terms: terms such as `api`, `batteries included`, `dependency injection`, `terminal`, `autonomous`, `serving`, `markdown`, and `workflow` should be visible in diagnostics.
- Metadata evidence: language, topics, repo name/entity, summary, use cases, and tags should be shown so a miss can be explained without guessing.
- Similar-project differences: when top-1 is acceptable but not exact, the report should expose why both repositories are close and which constraints separated them.
- Evaluation layering: this demo report remains a smoke/regression suite; larger indexes need category-specific and ambiguity-aware benchmarks.
