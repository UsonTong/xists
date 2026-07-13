# Evaluation Analysis

This note records the current demo retrieval baselines and the main follow-up items for search-quality tuning.

## Baseline reports

| Report | Dataset | Cases | Exact top-1 | Acceptable top-1 | Serious mismatch |
| --- | --- | ---: | ---: | ---: | ---: |
| `demo-eval-report.json` | `xists-baseline-100` | 100 | 89.0% | 100.0% | 0.0% |
| `demo-eval-report-extended.json` | `xists-baseline-112` | 112 | 88.4% | 100.0% | 0.0% |

The extended report was generated against `demo-index.json` with the configured `BAAI/bge-m3` embedding endpoint.

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
- Acceptable top-1: 112/112 cases.
- Serious mismatch: 0/112 cases.
- Acceptable-but-not-exact: 13/112 cases.

### Fixed gap

The previous `zed-modern-editor` miss is now fixed: `modern rust based editor`
returns `zed-industries/zed` at top-1 instead of letting the broad Rust language
repository outrank the editor/product cue.

### Acceptable-but-not-exact themes

The 13 acceptable misses are mostly broad, same-family substitutions. The main clusters are:

- Web/API frameworks: FastAPI/Django/Flask and Nest/Express.
- AI coding agents: Codex/Claude Code/OpenHands/agency-agents.
- LLM app/runtime tools: Dify/LangChain and vLLM/llama.cpp.
- Workflow/web extraction/media alternatives: n8n/Dify, Firecrawl/crawl4ai, yt-dlp/youtube-dl.

The next ranking pass should preserve these as acceptable while keeping the editor/product cue fix stable.
