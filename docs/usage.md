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

`repos.txt` is the current 200-repository demo list. `examples/eval-cases.json` is the committed 100-case baseline, and `examples/eval-cases-extended.json` adds 12 more cases for broader coverage.

For the full walkthrough, see [docs/demo.md](demo.md).

## Command-line entry

The primary command is search. Once you have an index, start with a natural-language query:

```bash
xists search "self-hosted photo gallery"
```

Run `xists doctor` when you need to check local configuration or expected files. The `ingest`, `profile`, and `index` commands maintain the local data source that search uses.

## Installation

For development from a source checkout, install xists in editable mode:

```bash
python -m pip install -e ".[dev]"
```

This makes the `xists` command available globally and installs the test dependency used by CI.

Users who do not need a source checkout can install the published package with:

```bash
python -m pip install xists
```

The command-line workflow keeps its default files in one local workspace. Build
your own records/index pair with the configured endpoints, as described below,
or download a future validated Release asset.

## Local workspace

Run `xists init` once after installation. It creates the default workspace
without downloading models, contacting an endpoint, or generating data:

```bash
xists init
```

By default, xists uses `~/.xists`:

```text
~/.xists/
  .env
  repos.txt
  records.json
  index.json
  report.json
  eval-cases.json
  eval-report.json
```

The normal workflow can then be run from any directory:

```bash
# Add one GitHub owner/repo or URL per line to ~/.xists/repos.txt.
xists ingest github
xists index build
xists search "self-hosted photo gallery"
```

Set `XISTS_HOME` to place the whole workspace elsewhere, such as an external
disk or a larger local volume:

```bash
export XISTS_HOME=/mnt/xists-data
xists init
```

Explicit file arguments always override workspace defaults, for example
`xists search "query" --index /path/to/experimental-index.json`. For backward
compatibility, if the current directory already contains `repos.txt`,
`records.json`, `index.json`, or `eval-cases.json`, commands without explicit
file arguments keep using that complete legacy working set. xists never moves
or overwrites those files automatically.

## MCP server

MCP is an optional integration. The regular package and all CLI/API features
remain usable without it. After `xists init`, embedding configuration, and an
index are ready, install it and start the stdio server:

```bash
python -m pip install "xists[mcp]"
xists mcp
```

The server loads the selected index once at startup. It uses the same default
workspace and configuration precedence as every CLI command: shell environment,
current-directory `.env`, then workspace `.env`. Pass `--index PATH` for an
explicit index. Rebuild or replace an index only while the server is stopped,
then restart `xists mcp`; hot reload is not provided.

The stable tools are:

- `search_projects(query, top_k=10)`: agent-ready candidates, with `top_k`
  limited to 1 through 20.
- `inspect_project(repo_id)`: the stored, non-vector profile for an indexed
  project.
- `index_stats()`: compatibility and size metadata without vectors.

Use `xists search "<query>" --format json` to reproduce a search in the CLI.
An `abstained: true` result is an honest no-credible-match state, not a retry
signal. MCP performs the same embedding request as CLI search, so a remote
embedding endpoint receives the query text; xists keeps the index and vector
comparison local.

Configure a client as a stdio server. Claude Code, Cursor, and Cline all accept
an equivalent entry in their respective MCP settings file:

```json
{
  "mcpServers": {
    "xists": {
      "command": "xists",
      "args": ["mcp"]
    }
  }
}
```

If the optional SDK is missing, run `pip install "xists[mcp]"`. If startup
reports a missing index or embedding configuration, run `xists doctor`, build
the index, or edit the active workspace `.env` before starting the client again.

## Configuration

`xists init` creates the workspace `.env` template (`~/.xists/.env` by
default). Edit that file and configure your credentials:

```bash
xists init
```

Edit `~/.xists/.env` and configure your credentials:

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

# Optional query canonicalization for an English-dominant corpus
QUERY_TRANSFORM_API_KEY=your_query_transform_api_key_here
QUERY_TRANSFORM_BASE_URL=https://api.example.com/v1
QUERY_TRANSFORM_MODEL=your_chat_model
```

All three sections are required for the full workflow. Environment variables
set by the shell take precedence over a `.env` file in the current directory,
which in turn takes precedence over the workspace `.env`. `.env` is ignored by
Git.

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

Create `repos.txt` in the workspace (by default `~/.xists/repos.txt`):

```text
facebook/react
vuejs/core
https://github.com/EbookFoundation/free-programming-books
```

Supported formats:

- `owner/repo`
- `https://github.com/owner/repo`

Blank lines and lines starting with `#` are ignored.

## Custom file layouts

The default workspace keeps generated files out of your source checkout. For a
one-off experiment, pass explicit paths instead:

```text
data/
  repos.txt
  records.json
  index.json
  eval-cases.json
  eval-report.json
```

Do not commit `.env`, token files, or generated demo data unless you are intentionally updating a small fixture.

## Preflight check

Check the installed version:

```bash
xists --version
xists version
```

Run `doctor` before a full ingest/index/eval cycle. Without file arguments it
shows the active workspace and its full default paths:

```bash
xists doctor
```

For an explicit demo or custom data set, pass the files directly:

```bash
xists doctor \
  --records demo-records.json \
  --index demo-index.json \
  --cases examples/eval-cases.json
```

It checks whether embedding, LLM, and GitHub configuration are present and whether the expected records, index, and evaluation case files exist. Add `--check-endpoints` to probe the embedding service with a real vector request, or `--strict` to make that probe fail the command. The default output is a short terminal summary and does not include secret values. Failing or warning checks include concrete next steps.

Use `--format json` only when another program needs the complete structured report:

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
xists doctor --format json \
  --records demo-records.json \
  --index demo-index.json \
  --cases examples/eval-cases.json \
  --check-endpoints \
  --strict
```

## Core CLI output contract

The core inspection and search commands are safe to use from either a terminal
or a script. Their default stdout is concise terminal text. Add `--format json`
when a command supports it and another program needs one structured document.
Progress, diagnostics, and failures are written to stderr; failures return a
nonzero exit code and do not write partial JSON to stdout. A completed `records
validate` or `index verify` report remains stdout output even when it reports
validation or verification failures.

| Command | Success output | Exit codes |
|---|---|---|
| `doctor` | Human-readable text by default; one JSON document with `--format json` | `0` when no error checks exist; `1` when a required check fails |
| `ingest github`, `profile refresh`, `index build` | Human-readable completion summary by default; one JSON document with `--format json` | `0` on success; nonzero when the command cannot complete its selected work |
| `records inspect` | One JSON document on stdout (JSON-first) | `0` on success; `1` for invalid records JSON; `2` when the file is absent |
| `records validate`, `records stats` | Human-readable text by default; one JSON document with `--format json` | `0` on success; `1` for validation or JSON-read/structure errors; `2` when the file is absent |
| `index stats`, `index verify` | Human-readable text by default; one JSON document with `--format json` | `0` when valid; `1` when verification or JSON-read/structure validation fails; `2` when an input file is absent |
| `search` | Human-readable text by default; one JSON result with `--format json` | `0` on success; `1` for index compatibility, endpoint, transform, or ranking errors; `2` for missing embedding configuration or index file |

`doctor` may report missing optional data files as warnings and still return
`0`; in JSON mode its `ok` field and individual check statuses are the
authoritative machine-readable result. Evaluation commands retain their
existing JSON-first reports and are not changed by this contract.

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
| `--dry-run` | off | Estimate work without calling GitHub or writing files |
| `--format` | `text` | Output format: text for terminal review, or json for scripts and agents |
| `--workers` | `1` | Number of concurrent workers |
| `--retry-failed` | (none) | Retry only repository ids listed in a previous failure report |
| `--resume` | off | Resume from an existing partial JSONL checkpoint |
| `--max-rate-limit-wait` | `3600` | Maximum seconds to wait for exhausted GitHub API quota |

#### Incremental update

By default, `xists ingest github` is incremental. It skips repos that already exist in `records.json` and only processes new ones. To reprocess everything:

```bash
xists ingest github --force
```

#### Checkpoint / resume

Each successful record is appended to `<output>.partial.jsonl` as it completes. If the process is interrupted (Ctrl+C, crash, etc.), previously completed records are preserved. Re-run the same command with `--resume` to continue from that checkpoint.

#### Dry run

Use `--dry-run` to preview how many repos will be processed or skipped before starting a long ingest job. Dry runs do not call GitHub, do not require an LLM configuration, and never write output files.

```bash
xists ingest github --repos repos.txt --output records.json --dry-run
```

Add `--format json` when you want a machine-readable estimate.

#### Retry failures

Failure reports contain retryable `repo_id` entries. Retry only those entries with:

```bash
xists ingest github --repos repos.txt --output records.json --retry-failed report.json
```

An ingest that finishes its batch with some individual failures exits `0` and writes their details to the report; a job that cannot process any selected repository exits nonzero. Failed entries are summarized on stderr.

When GitHub returns `403` or `429` with an exhausted quota, xists first rotates to another configured token. If every token is exhausted, it reports the reset time on stderr and waits until reset plus five seconds. Set `--max-rate-limit-wait` to bound that wait; when the limit is exceeded, completed checkpoints remain on disk and the command exits nonzero.

During ingest, each successful record is appended to `<output>.partial.jsonl`; the final `records.json` is written atomically only after the batch completes. Re-run the same command with `--resume` after an interruption. A partial file without `--resume` is rejected so completed work is never overwritten accidentally.

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

Before building or rebuilding an index, validate and inspect records to verify that ingestion and LLM profiling produced usable metadata:

```bash
xists records validate --records demo-records.json
xists records stats --records demo-records.json
xists records inspect --records demo-records.json --limit 5
xists records inspect --records demo-records.json --repo react --limit 2
```

`records validate` reports schema/profile problems and actionable next steps. `records stats` summarizes data quality, confidence, languages, topics, project types, and ecosystems. `records inspect` prints compact per-repo details without large README/profile payloads.

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
| `--resume` | off | Continue from a partial index checkpoint |
| `--format` | `text` | Output format: text for terminal review, or json for scripts and agents |

#### Incremental update

By default, `xists index build` is incremental. It reuses an existing vector only when the repo id, embedding model, vector dimension, and embedding input fingerprint still match the current record. If the record content or embedding text logic changes, xists re-embeds that record automatically.

#### Dual-file binary vector storage (v4)

Starting in v0.11.0, `xists index build` defaults to `INDEX_VERSION = 4` dual-file binary storage:
- `index.json`: Repository metadata, schemas, and a `vectors_file` reference.
- `<stem>.vectors.npy`: Raw float32 vector matrix saved in binary NumPy format.

Loading a v4 index uses `numpy.load(..., mmap_mode='r')` for zero-copy memory mapping, achieving sub-50ms cold-start load times on 10k corpora and reducing disk size by over 40% compared to Base64 JSON.

For backward compatibility, xists seamlessly reads `INDEX_VERSION = 3` legacy single-file indexes without requiring immediate rebuilds.

#### Public index distribution & pull

Use `xists index pull` to quickly download or install pre-built indexes and datasets without needing API keys or running manual builds:

```bash
# Pull bundled zero-config starter dataset into current workspace
xists index pull demo

# Pull curated community index
xists index pull curated-1k

# Pull from custom URL with SHA-256 integrity verification
xists index pull https://example.com/custom-index.tar.gz --sha256 <expected_hash> --output-dir ./data --force
```

Available presets include `demo` (20 top open-source repositories with pre-built binary vectors) and `curated-1k`.

#### Offline & zero-config search mode

You can immediately test search without any embedding API credentials or external network access using the `--demo` or `--offline` flags:

```bash
# Search bundled starter demo dataset (fallback to offline metadata ranking if no API key is set)
xists search --demo "python web framework"

# Force offline lexical/metadata search across records
xists search --offline "vector database"
```


#### Resilient atomic checkpoints & self-healing

Completed batches are periodically written to `<output>.partial.json` with embedded CRC32 checksums. If a build is interrupted by sudden power loss, process termination, or network abort:
- Run with `--resume` to recover automatically.
- Checkpoints with truncated or corrupted tails are parsed for all intact vector entries, healed automatically, and resumed from the last valid batch.

#### Model mismatch protection

If `index.json` was built with a different embedding model than the one configured, `index build` refuses to run and asks you to rebuild or match the model. This prevents silent corruption from mixing incompatible vectors.

#### Inspect index statistics

Use `index stats` when you want to confirm what is inside an index without printing large embedding vectors:

```bash
xists index stats --index demo-index.json --limit 5
xists index verify --records demo-records.json --index demo-index.json
```

`index stats` includes model, dimension, record/vector counts, an estimated in-memory size of the vector matrix (float32), skipped count, missing metadata/fingerprint counts, and the most common languages/topics. `index verify` compares records and index fingerprints to catch stale, missing, extra, or incompatible vectors across both v4 binary and v3 Base64 indexes before search/eval.

### Maintain a data source

Use this loop when updating or sharing a curated repository source:

```bash
xists records validate --records records.json
xists records stats --records records.json
xists profile refresh --records records.json --only-missing-search-text --output records.new.json
xists index build --records records.new.json --output index.json
xists index verify --records records.new.json --index index.json
xists index stats --index index.json
```

The goal is to keep `records.json` reusable and understandable, then rebuild `index.json` whenever profile fields or embedding input fingerprints change.

#### Refresh profiles safely

`profile refresh` appends each successfully refreshed record to `<output>.partial.jsonl`. Re-run with `--resume` after an interruption; a normal completion atomically writes the JSON output and removes the partial file.

```bash
xists profile refresh --records records.json --output records.new.json --dry-run
xists profile refresh --records records.json --output records.new.json --report refresh-report.json
xists profile refresh --records records.json --output records.new.json --retry-failed refresh-report.json
```

Use `--dry-run` to preview refresh work without calling the LLM or writing files; add `--format json` for a machine-readable estimate. Profile generation allows up to 600 seconds for an individual OpenAI-compatible LLM response, which accommodates local or heavily loaded models during long refreshes.

The refresh failure report includes `repo_id`, `error`, and `attempted_at`. A completed batch with individual failures exits `0`, preserves each failed record's old profile, and writes a stderr summary. A run that cannot refresh any selected record exits nonzero so endpoint or configuration failures remain actionable.

### Step 3: Search

```bash
xists search "frontend UI library"
xists search "frontend UI library" --format json
```

Returns ranked results with confidence tiers. Text is the default output for terminal review; `--format json` is intended for scripts and agent integrations. Search combines embedding similarity with bounded, explainable metadata signals. Exact `owner/repo` and exact name/alias lookups are pinned first; names mentioned inside a broader natural-language request remain contextual evidence and do not receive exact-identity pinning. Other results use cosine similarity plus lightweight language, topic/profile overlap, repository-state, and popularity adjustments.

Default text output keeps the fields people usually inspect first:

```text
query: frontend UI library
intent: functional
abstained: False
results: 1
1. repo: react/react
   url: https://github.com/react/react
   confidence: high_confidence
   score: 0.680000
   summary: A JavaScript library for building user interfaces.
   why: matched metadata terms: frontend, ui; matched topics: frontend, ui
```

JSON output exposes the same ranking data for automation:

```json
{
  "query": "frontend UI library",
  "query_intent": {"type": "functional"},
  "abstained": false,
  "results": [
    {
      "repo_id": "react/react",
      "url": "https://github.com/react/react",
      "score": 0.68,
      "semantic_score": 0.62,
      "metadata_score": 0.06,
      "score_breakdown": {"semantic": 0.62, "metadata": 0.06, "final": 0.68},
      "matched_terms": ["frontend", "ui"],
      "diagnostics": {
        "identity_match": null,
        "language_match": null,
        "topic_matches": ["frontend", "ui"],
        "profile_matches": ["frontend", "ui"]
      },
      "confidence": "high_confidence",
      "why": ["matched metadata terms: frontend, ui", "matched topics: frontend, ui"]
    }
  ],
  "considered": 8
}
```

`score` is the final ranking score. `semantic_score` is the embedding cosine similarity, and `metadata_score` is a lightweight, bounded adjustment. Exact `owner/repo` and exact name/alias queries are pinned to the top so entity lookup works even when the embedding score is not the highest. A name only mentioned as part of a broader request is not an exact identity match.

`query_intent` describes the detected query shape. Each result includes:

- `score_breakdown`: rounded semantic, metadata, and final scores for easier debugging
- `matched_terms`: non-generic query terms found in the candidate metadata/profile
- `diagnostics`: compact structured evidence used by CLI/eval reports

Chinese and mixed-language queries retain ASCII technical identifiers and add
a bounded, deduplicated set of CJK bigrams/trigrams. These terms participate in
query specificity, metadata overlap, matched terms, and explanations. They are
not a heavyweight word segmenter and do not change the text sent to the
embedding endpoint.
- `why`: short human-readable reasons from identity/language/topic/profile/state signals

#### Confidence tiers

| Tier | Final score | Meaning |
|------|-------------|---------|
| `high_confidence` | ≥ 0.60 | Strong match, likely relevant |
| `exploratory` | ≥ 0.35 | Worth investigating |
| `abstain` | < 0.35 | Too weak, not shown |

Weak semantic matches stay hidden unless they are exact identity matches; metadata should help close calls, not replace semantic relevance.

#### Evidence-based confidence calibration

`--confidence-calibration evidence-v1` is an opt-in post-ranking experiment for
reranked searches. It preserves the candidate list, ordering, and abstention
decision, but downgrades a `high_confidence` result to `exploratory` when the
available evidence is contradictory or incomplete. The result includes
`confidence_evidence` with the calibration version, ranking evidence, supporting
signals, and any downgrade reasons. Evaluation reports record the selected mode
and preserve the top result's evidence for reproducible analysis.

```bash
xists eval run --cases eval-cases.json --index index.json --output eval-calibrated.json \
  --ranking-strategy rerank --confidence-calibration evidence-v1
```

#### English canonical queries

For an English-dominant corpus, an optional compatible chat endpoint can turn a query into a concise English retrieval expression while preserving technical identifiers. The original query remains available for exact repository identity matching. `canonical` embeds only the canonical expression; `merge` embeds both expressions and keeps the stronger similarity for each candidate.

```bash
xists search "查找 Vue 开源项目" --query-transform-mode merge
xists eval run --cases eval-cases.json --index index.json --output eval-merge.json --query-transform-mode merge
```

This is disabled by default. It requires `QUERY_TRANSFORM_API_KEY`, `QUERY_TRANSFORM_BASE_URL`, and `QUERY_TRANSFORM_MODEL`. Evaluation reports record the selected mode and model so experiments remain comparable.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--index` | `index.json` | Index file to search |
| `--top-k` | `10` | Maximum results to return |
| `--query-transform-mode` | `off` | `off`, English `canonical`, or original-plus-canonical `merge` retrieval |
| `--confidence-calibration` | `off` | `off` or post-ranking `evidence-v1` confidence calibration |
| `--format` | `text` | Output format: `text` for terminal review, or `json` for scripts and agents |

### Step 4: Evaluate retrieval quality

```bash
xists eval cases --cases examples/eval-cases.json
xists eval cases --cases examples/eval-cases-extended.json
xists eval cases --cases examples/eval-cases-smoke.json
xists eval run --cases examples/eval-cases-smoke.json --index index.json --output eval-smoke-report.json
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
- add `acceptable` (or the legacy `acceptable_repo_ids`) or `acceptable_families` for broad or ambiguous queries
- use tags to preserve coverage across tools, language ecosystems, alternatives,
  weak-signal queries, and major product areas
- run `xists eval cases --cases examples/eval-cases.json` and `pytest` before
  committing the dataset
- keep the baseline file and the extended file in sync with `demo-eval-report.json`
  and the release checklist; use the extended file for extra coverage without
  disturbing the committed baseline report

Useful review commands:

```bash
xists eval cases --cases examples/eval-cases.json --tag alternative
xists eval cases --cases examples/eval-cases-extended.json --tag weak-signal
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
      "acceptable": ["facebook/react"],
      "acceptable_families": ["react-family"],
      "tags": ["frontend", "ui"],
      "notes": "forks and sibling repos are acceptable"
    }
  ]
}
```

`expected_repo_id` is the exact target for strict scoring. `acceptable` is an optional repo-id array for alternatives that count as retrieval hits; `acceptable_repo_ids` is kept as a compatible alias. `acceptable_families` expands named repo families. All three let you count highly similar repos, forks, or same-family alternatives without weakening the exact metric.

#### Metrics

Core retrieval metrics:

- `exact_hit_at_1` / `exact_hit_at_k`: the expected repo is ranked first or appears anywhere in the top K
- `mrr_exact`: how early the expected repo appears on average
- `acceptable_hit_at_1` / `acceptable_hit_at_k`: the expected repo or an acceptable same-family alternative appears in the top results
- `recall_at_1` / `recall_at_5`: the expected repo or a dataset-declared acceptable alternative appears at rank 1 or anywhere in the first five results
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
| `--query-transform-mode` | `off` | `off`, English `canonical`, or original-plus-canonical `merge` retrieval |
| `--records` | (none) | Records JSON used for optional LLM top1-vs-expected judge |
| `--llm-judge` | off | Run an LLM pairwise judge only on top-1 mismatches |

When `--llm-judge` is enabled, you must also provide `--records`. The judge compares only `top1` vs `expected_repo_id`; it does not change exact/acceptable metrics. It adds a separate `judge_summary` section and per-case judge fields so you can distinguish “wrong” from “close but acceptable substitute”.

## Output files

xists uses three artifact layers:

- `repos.txt`: a reviewable repository list with one repo id or GitHub URL per line.
- `records.json`: reusable semantic data with source metadata, evidence, LLM profile fields, and search text.
- `index.json`: a derived vector index tied to one embedding model and embedding input version.

Share `repos.txt` when collaborators should regenerate everything, share `records.json` when they should reuse LLM/profile work and rebuild their own index, and share `index.json` only when they use the same embedding setup.

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

Release readiness expectations for `0.4.0`:

- package version and `xists.__version__` are aligned
- README remains the short project entry point unless intentionally changed
- docs cover the full local workflow, data-quality commands, and artifact layers
- generated records, indexes, reports, `.env`, and token files stay uncommitted
- `pytest` and GitHub Actions CI pass
