# Evaluation Analysis

This note records the current demo retrieval baselines and the main follow-up items for search-quality tuning.

## Baseline reports

| Report | Dataset | Cases | Exact top-1 | Acceptable top-1 | Serious mismatch |
| --- | --- | ---: | ---: | ---: | ---: |
| `demo-eval-report.json` | `xists-baseline-100` | 100 | 89.0% | 100.0% | 0.0% |
| `demo-eval-report-extended.json` | `xists-baseline-112` | 112 | 87.5% | 99.1% | 0.9% |

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

- Exact top-1: 98/112 cases.
- Acceptable top-1: 111/112 cases.
- Serious mismatch: 1/112 cases.
- Acceptable-but-not-exact: 13/112 cases.

### Serious mismatch to fix first

| Case | Query | Expected | Top-1 | Exact rank | Notes |
| --- | --- | --- | --- | ---: | --- |
| `zed-modern-editor` | `modern rust based editor` | `zed-industries/zed` | `rust-lang/rust` | 2 | Language-ecosystem signal (`rust`) overpowers the product type (`editor`). |

### Acceptable-but-not-exact themes

The 13 acceptable misses are mostly broad, same-family substitutions. The main clusters are:

- Web/API frameworks: FastAPI/Django/Flask and Nest/Express.
- AI coding agents: Codex/Claude Code/OpenHands/agency-agents.
- LLM app/runtime tools: Dify/LangChain and vLLM/llama.cpp.
- Workflow/web extraction/media alternatives: n8n/Dify, Firecrawl/crawl4ai, yt-dlp/youtube-dl.

The next ranking pass should preserve these as acceptable while improving exact matches where a query contains a strong product-type cue such as `editor`.
