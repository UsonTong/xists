# Usage

## Demo First

If you want the shortest path to a working run, start with the committed example inputs:

```bash
xists ingest github --repos repos.txt --output demo-records.json --report demo-report.json --github-api graphql --github-batch-size 10 --workers 4
xists index build --records demo-records.json --output demo-index.json
xists search "frontend ui library" --index demo-index.json
xists eval run --cases examples/eval-cases.json --index demo-index.json --output demo-eval-report.json
xists eval inspect --report demo-eval-report.json
```

`repos.txt` is the current 200-repository demo list. `examples/eval-cases.json` is an optional 112-case dataset for checking retrieval quality.

For the full walkthrough, see [docs/demo.md](demo.md).

## Installation

Install xists in development mode:

```bash
python -m pip install -e ".[dev]"
```

This makes the `xists` command available globally and installs the test dependency used by CI.

## Configuration

Create a local `.env` file from the example file:

```bash
cp .env.example .env
```

Edit `.env` and configure your credentials:

```env
# GitHub (required for ingest)
# Single token:
GITHUB_TOKEN=your_github_token_here
# Multiple tokens (comma-separated, for higher rate limits):
# GITHUB_TOKENS=tok1,tok2,tok3

# LLM profile generation (required for ingest)
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-pro

# Embedding vector calculation (required for index build, search, and eval)
# The endpoint only computes vectors. xists stores index.json and searches locally.
EMBEDDING_API_KEY=local
EMBEDDING_BASE_URL=http://localhost:6597/v1
EMBEDDING_MODEL=BAAI/bge-m3
```

All three sections are required for the full workflow. `.env` is ignored by Git.

The embedding endpoint is a vector calculator, not a query service. During `index build`, xists sends repository texts to the endpoint and stores the returned vectors in local `index.json`. During `search` and `eval run`, xists sends only the query text to get its query vector, then performs vector search and reranking locally against `index.json`. Remote embedding APIs are usable, but only for calculation.

The same embedding config is intentionally shared by indexing, search, and evaluation. The query vector must be computed by the same model used to build the index vectors. Do not change `EMBEDDING_MODEL` after building `index.json`; if you change it, rebuild the index with `xists index build --force`.

#### Multiple GitHub tokens

For large ingestion jobs, a single GitHub token may hit rate limits (5000 requests/hour). Configure multiple tokens to distribute requests across them with round-robin rotation:

```env
GITHUB_TOKENS=ghp_token1,ghp_token2,ghp_token3
```

Alternatively, use `--token-file` with one token per line:

```text
ghp_token1
ghp_token2
ghp_token3
```

When multiple tokens are configured, each API request rotates to the next token, effectively multiplying your available rate limit. This is especially useful with `--workers` for concurrent ingestion.

## Create a repository list

Create `repos.txt` in the project root:

```text
facebook/react
vuejs/core
https://github.com/EbookFoundation/free-programming-books
```

Supported formats:

- `owner/repo`
- `https://github.com/owner/repo`

Blank lines and lines starting with `#` are ignored.

## Recommended local file layout

For local experiments, keep generated files under `data/` so it is obvious which files are inputs and outputs:

```text
data/
  repos.txt
  records.json
  index.json
  eval-cases.json
  eval-report.json
```

The default root-level generated artifacts (`records.json`, `index.json`, `report.json`, `eval-report.json`) are ignored by Git. Do not commit `.env`, token files, or generated demo data unless you are intentionally updating a small fixture.

## Preflight check

Check the installed version:

```bash
xists --version
xists version
```

Run `doctor` before a full ingest/index/eval cycle:

```bash
xists doctor \
  --records demo-records.json \
  --index demo-index.json \
  --cases examples/eval-cases.json
```

It checks whether embedding, LLM, and GitHub configuration are present and whether the expected records, index, and evaluation case files exist. Add `--check-endpoints` to probe the embedding service with a real vector request, or `--strict` to make that probe fail the command. The output is JSON and does not include secret values. Failing or warning checks include `next_steps` when xists can suggest a concrete fix:

```json
{
  "ok": false,
  "checks": [
    {"name": "embedding_config", "status": "ok", "model": "BAAI/bge-m3"},
    {"name": "llm_config", "status": "ok", "model": "gpt-5.4"},
    {
      "name": "embedding_endpoint",
      "status": "error",
      "message": "Embedding request failed for all configured endpoints...",
      "next_steps": [
        "Start the embedding service referenced by EMBEDDING_BASE_URL.",
        "Confirm the base URL is the API root, for example http://localhost:6597/v1 for OpenAI-compatible servers.",
        "Run xists doctor --check-endpoints --strict before retrying index/search/eval commands."
      ]
    }
  ]
}
```

Warnings usually mean a file has not been generated yet or an ingest-only token is missing. Errors mean a required endpoint configuration is missing or an endpoint probe failed in strict mode. A good demo preflight is:

```bash
xists doctor \
  --records demo-records.json \
  --index demo-index.json \
  --cases examples/eval-cases.json \
  --check-endpoints \
  --strict
```

## Workflow

### Step 1: Ingest repositories

```bash
xists ingest github
```

This fetches data from GitHub, generates LLM profiles, and writes `records.json`.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--repos` | `repos.txt` | Input file with one repo per line |
| `--output` | `records.json` | Output records file |
| `--report` | `report.json` | Output report file |
| `--token-file` | (none) | File containing GitHub token(s), one per line |
| `--force` | off | Ignore existing records.json and reprocess all repos |
| `--workers` | `1` | Number of concurrent workers |

#### Incremental update

By default, `xists ingest github` is incremental. It skips repos that already exist in `records.json` and only processes new ones. To reprocess everything:

```bash
xists ingest github --force
```

#### Checkpoint / resume

Each record is written to disk as it completes. If the process is interrupted (Ctrl+C, crash, etc.), previously completed records are preserved. Simply run the command again to continue from where it left off.

#### Multi-threaded ingest

Ingest is I/O-bound (GitHub API + LLM calls), so concurrency helps significantly:

```bash
xists ingest github --workers 5
```

Performance example (10 repos):

| Mode | Time | Speedup |
|------|------|---------|
| `--workers 1` | 2m 41s | 1x |
| `--workers 5` | 31s | 5.2x |

### Inspect generated records

Before building or rebuilding an index, inspect records to verify that ingestion and LLM profiling produced usable metadata:

```bash
xists records inspect --records demo-records.json --limit 5
xists records inspect --records demo-records.json --repo react --limit 2
```

This prints a compact JSON summary with repo id, URL, language, topics, README presence, profile confidence, abstain state, and the generated summary. It intentionally omits large README/profile payloads.

### Step 2: Build the embedding index

```bash
xists index build
```

This reads `records.json`, computes embeddings via the configured endpoint, and writes `index.json`.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--records` | `records.json` | Input records file |
| `--output` | `index.json` | Output index file |
| `--force` | off | Ignore existing index.json and rebuild from scratch |

#### Incremental update

By default, `xists index build` is incremental. It reuses an existing vector only when the repo id, embedding model, vector dimension, and embedding input fingerprint still match the current record. If the record content or embedding text logic changes, xists re-embeds that record automatically.

#### Checkpoint / resume

Each batch (64 records) is written to disk as it completes. If interrupted, completed batches are preserved.

#### Model mismatch protection

If `index.json` was built with a different embedding model than the one configured, `index build` refuses to run and asks you to rebuild or match the model. This prevents silent corruption from mixing incompatible vectors.

#### Inspect index statistics

Use `index stats` when you want to confirm what is inside an index without printing large embedding vectors:

```bash
xists index stats --index demo-index.json --limit 5
```

The output includes model, dimension, record/vector counts, skipped count, missing metadata/fingerprint counts, and the most common languages/topics. This is useful before evaluation because it catches stale or incomplete indexes quickly.

### Step 3: Search

```bash
xists search "frontend UI library"
```

Returns ranked results with confidence tiers:

```json
{
  "query": "frontend UI library",
  "abstained": false,
  "results": [
    {
      "repo_id": "react/react",
      "score": 0.68,
      "semantic_score": 0.62,
      "metadata_score": 0.06,
      "score_breakdown": {"semantic": 0.62, "metadata": 0.06, "final": 0.68},
      "matched_terms": ["frontend"],
      "confidence": "high_confidence",
      "why": ["matched topic: frontend", "matched phrase: UI library"]
    },
    {
      "repo_id": "vuejs/core",
      "score": 0.61,
      "semantic_score": 0.58,
      "metadata_score": 0.03,
      "confidence": "high_confidence"
    }
  ],
  "considered": 8
}
```

`score` is the final ranking score. `semantic_score` is the cosine similarity
from the embedding search, and `metadata_score` is the bounded reranking bonus
from repository names, descriptions, topics, and generated profile phrases.
Exact repository/name queries receive stronger metadata evidence than ordinary
substring overlap.

`query_intent` describes the detected query shape. Each result includes:

- `score_breakdown`: rounded semantic, metadata, and final scores for easier debugging
- `matched_terms`: non-generic query terms found in the candidate metadata/profile
- `why`: short human-readable metadata/topic/name/phrase signals that affected the rank

#### Confidence tiers

| Tier | Final score | Meaning |
|------|-------------|---------|
| `high_confidence` | ≥ 0.55 | Strong match, likely relevant |
| `exploratory` | ≥ 0.35 | Worth investigating |
| `abstain` | < 0.35 | Too weak, not shown |

Weak semantic matches stay hidden unless metadata provides strong evidence,
such as a repository/name match or a unique exact generated profile phrase.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--index` | `index.json` | Index file to search |
| `--top-k` | `10` | Maximum results to return |

### Step 4: Evaluate retrieval quality

```bash
xists eval cases --cases examples/eval-cases.json
xists eval run --cases examples/eval-cases.json --index index.json --output eval-report.json
```

`eval cases` validates the dataset and summarizes tag/query-intent coverage before
any embedding calls are made. `eval run` runs that fixed dataset against the
current index and writes an evaluation report you can use to sanity-check search
behavior across repository lists, regenerated summaries, or search configuration
changes.


### Inspect evaluation failures

After `xists eval run`, inspect the report to see which queries missed or returned weak substitutes:

```bash
xists eval inspect --report eval-report.json
xists eval inspect --report eval-report.json --status serious_mismatch --limit 20
xists eval inspect --report eval-report.json --tag weak-signal
xists eval inspect --report eval-report.json --query-intent functional
```

The inspect output includes the report metrics, the readable `summary_text`, and a sorted list of cases with:

- query
- expected repo
- top-1 repo
- top-1 status
- confidence
- exact and acceptable ranks

This optional loop is useful when you want to compare search behavior after changes:

```bash
pytest
xists eval run --cases examples/eval-cases.json --index demo-index.json --output demo-eval-report.json
xists eval inspect --report demo-eval-report.json --status serious_mismatch
```

#### Maintaining evaluation cases

`examples/eval-cases.json` is intentionally small enough to review by hand. When
adding cases:

- keep `id` stable and descriptive; changing an id makes trend comparison harder
- choose an `expected_repo_id` that exists in `repos.txt` and the demo records
- add `acceptable_repo_ids` or `acceptable_families` for broad or ambiguous queries
- use tags to preserve coverage across tools, language ecosystems, alternatives,
  weak-signal queries, and major product areas
- run `xists eval cases --cases examples/eval-cases.json` and `pytest` before
  committing the dataset

Useful review commands:

```bash
xists eval cases --cases examples/eval-cases.json --tag alternative
xists eval cases --cases examples/eval-cases.json --query-intent functional
xists eval inspect --report demo-eval-report.json --tag weak-signal
```

#### Evaluation dataset shape

```json
{
  "schema_version": 1,
  "dataset_name": "frontend-retrieval-smoke",
  "families": {
    "react-family": ["react/react", "facebook/react", "preactjs/preact"]
  },
  "cases": [
    {
      "id": "react-ui-1",
      "query": "frontend ui library",
      "expected_repo_id": "react/react",
      "acceptable_repo_ids": ["facebook/react"],
      "acceptable_families": ["react-family"],
      "tags": ["frontend", "ui"],
      "notes": "forks and sibling repos are acceptable"
    }
  ]
}
```

`expected_repo_id` is the exact target for strict scoring. `acceptable_repo_ids` and `acceptable_families` let you count highly similar repos, forks, or same-family alternatives without weakening the exact metric.

#### Metrics

Core retrieval metrics:

- `exact_hit_at_1` / `exact_hit_at_k`: the expected repo is ranked first or appears anywhere in the top K
- `mrr_exact`: how early the expected repo appears on average
- `acceptable_hit_at_1` / `acceptable_hit_at_k`: the expected repo or an acceptable same-family alternative appears in the top results
- `mrr_acceptable`: how early the first acceptable result appears on average
- `abstain_rate`: the fraction of queries where search returns no result above the exploratory threshold

Top-1 outcome metrics:

- `exact_top1_rate`: the fraction of cases where the top result exactly matches `expected_repo_id`
- `acceptable_top1_rate`: the fraction of cases where top-1 is not exact, but is still acceptable either because the dataset marks it as an acceptable alternative or because the optional LLM judge marks it as a close enough substitute
- `serious_top1_error_rate`: the fraction of cases where top-1 is not exact and still misses a material query constraint after applying the dataset acceptable set and optional judge analysis
- `insufficient_evidence_top1_rate`: the fraction of cases where top-1 is not exact and the optional LLM judge reports that the available evidence is too thin to classify it as either acceptable or a serious mismatch
- `effective_top1_rate`: `exact_top1_rate + acceptable_top1_rate`, useful when you care about whether top-1 is good enough for the user even if it is not the dataset's exact reference answer

The hard metrics remain the source of truth for exact retrieval quality. The top-1 metrics are a final outcome classification layer: dataset-declared acceptable alternatives count as acceptable immediately, and the optional judge is only used to classify remaining non-exact mismatches. Without the judge enabled, non-exact results outside the dataset acceptable set default to serious mismatches.

This is a semantic expansion from older reports, where `acceptable_top1_rate` only counted judge-approved substitutes. As a result, datasets that already declare acceptable alternatives may now show a higher `acceptable_top1_rate` and a lower `serious_top1_error_rate`.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--cases` | `eval-cases.json` | Evaluation dataset JSON |
| `--index` | `index.json` | Index file to evaluate |
| `--output` | `eval-report.json` | Output evaluation report |
| `--top-k` | `10` | Maximum results to score per query |
| `--batch-size` | `64` | Number of queries to embed per batch |
| `--records` | (none) | Records JSON used for optional LLM top1-vs-expected judge |
| `--llm-judge` | off | Run an LLM pairwise judge only on top-1 mismatches |

When `--llm-judge` is enabled, you must also provide `--records`. The judge compares only `top1` vs `expected_repo_id`; it does not change exact/acceptable metrics. It adds a separate `judge_summary` section and per-case judge fields so you can distinguish “wrong” from “close but acceptable substitute”.

## Output files

### `records.json`

A JSON array of generated xists records. Each record includes GitHub metadata, README excerpt, structure signals, evidence, evidence gaps, and an LLM-generated profile.

See [Record Schema](record-schema.md) for the full format.

### `index.json`

A JSON object containing the embedding index. Includes metadata (model, dimension, timestamp, embedding input version) and a vectors array mapping each repo_id to its embedding vector and embedding input fingerprint.

### `report.json`

A JSON report for the ingest run. Includes started_at, finished_at, duration_seconds, workers, force, xists_version, safe LLM config (provider/model/prompt_version only), input count, skipped count, generated count, failed count, and details for each failure. It never records API keys, tokens, or endpoint secrets.

## File management

| File | Gitignored | Description |
|------|------------|-------------|
| `.env` | Yes | Credentials and configuration |
| `repos.txt` | No | User-maintained repository list |
| `records.json` | Yes | Generated records (derived data) |
| `index.json` | Yes | Generated index (derived data) |
| `report.json` | Yes | Last ingest report |

To start fresh, delete `records.json` and `index.json` and re-run the workflow.

## CI and release checklist

See [docs/release.md](release.md) for the full release checklist.

The repository includes GitHub Actions CI in `.github/workflows/ci.yml`. On every push and pull request it installs the package in editable dev mode and runs `pytest` on Python 3.11 and 3.12.

Before tagging a release:

```bash
python -m pip install -e ".[dev]"
pytest
xists doctor --records demo-records.json --index demo-index.json --cases examples/eval-cases.json
xists index stats --index demo-index.json
xists eval inspect --report demo-eval-report.json --status serious_mismatch
```

Release readiness expectations for `0.1.0`:

- package version and `xists.__version__` are aligned
- README remains the short project entry point unless intentionally changed
- docs cover the full local workflow and inspection commands
- generated records, indexes, reports, `.env`, and token files stay uncommitted
- `pytest` and GitHub Actions CI pass
