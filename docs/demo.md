# Demo Workflow

This guide walks through the smallest end-to-end xists workflow using the committed demo inputs.

## Files

- `repos.txt`: the current demo repository list in the project root
- `examples/eval-cases.json`: optional 100-case evaluation dataset for checking retrieval quality

## 1. Install

```bash
python -m pip install -e ".[dev]"
```

## 2. Configure credentials

Create a local `.env` file:

```bash
cp .env.example .env
```

Set the required GitHub, LLM, and embedding credentials in `.env`.

## 3. Preflight and endpoint checks

Before spending GitHub or LLM quota, verify the local files and configuration:

```bash
xists doctor \
  --records demo-records.json \
  --index demo-index.json \
  --cases examples/eval-cases.json
```

`doctor` returns JSON. Missing generated files are warnings because the demo may
not have been generated yet. Missing embedding or LLM variables are errors with
`next_steps` entries showing what to set in `.env`.

When your embedding service is supposed to be running, probe it with a real
request before building an index, searching, or running eval:

```bash
xists doctor \
  --records demo-records.json \
  --index demo-index.json \
  --cases examples/eval-cases.json \
  --check-endpoints
```

Use `--strict` in scripts or CI-style smoke checks when endpoint failures should
make the command exit non-zero:

```bash
xists doctor \
  --records demo-records.json \
  --index demo-index.json \
  --cases examples/eval-cases.json \
  --check-endpoints \
  --strict
```

If the probe cannot connect, start the embedding server referenced by
`EMBEDDING_BASE_URL`, confirm the URL is the API root such as
`http://localhost:6597/v1`, and rerun the strict doctor command before retrying
`index build`, `search`, or `eval run`.

## 4. Ingest example repositories

```bash
xists ingest github \
  --repos repos.txt \
  --output demo-records.json \
  --report demo-report.json \
  --github-api graphql \
  --github-batch-size 10 \
  --workers 4
```

This fetches GitHub data, generates LLM profiles, and writes checkpoints as records finish:

- `demo-records.json`
- `demo-report.json`

The ingest command prints progress to stderr while it runs. If your LLM endpoint is rate-limited, reduce `--workers` to `1` or `2`.

## 5. Build the example index

```bash
xists index build \
  --records demo-records.json \
  --output demo-index.json
```

## 6. Run a few searches

```bash
xists search "frontend ui library" --index demo-index.json
xists search "python web framework for building apis" --index demo-index.json
xists search "open source workflow automation platform" --index demo-index.json
```

Expected results vary by model and repository data, but the top results should usually come from the same neighborhood as the query.

## 7. Optionally evaluate the demo index

```bash
xists eval run \
  --cases examples/eval-cases.json \
  --index demo-index.json \
  --output demo-eval-report.json
```

The report includes:

- hard retrieval metrics such as `exact_hit_at_1`, `acceptable_hit_at_k`, and `mrr_exact`
- a readable `summary_text` block for exact top-1, acceptable top-1, serious mismatches, abstains, and wrong high-confidence cases
- `top_misses`, a sorted list of the most actionable failures
- confidence counts for top-1 predictions
- per-case top-1 outcomes in `results`

Inspect the failures that matter most:

```bash
xists eval inspect --report demo-eval-report.json
xists eval inspect --report demo-eval-report.json --status serious_mismatch
```

## 8. Inspect misses and iterate

After changing the repository list, regenerating summaries, or adjusting search behavior, rerun:

```bash
xists index build --records demo-records.json --output demo-index.json --force
xists eval run --cases examples/eval-cases.json --index demo-index.json --output demo-eval-report.json
xists eval inspect --report demo-eval-report.json --status serious_mismatch
```

That gives you a stable way to sanity-check search behavior before moving on to larger repository lists or evaluation datasets.

## Common run failures

- `Embedding is required ... Missing environment variables`: copy `.env.example`
  to `.env` and set `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, and
  `EMBEDDING_MODEL`.
- `Embedding request failed for all configured endpoints`: the embedding server
  is not reachable at `EMBEDDING_BASE_URL`, the URL points at the wrong API root,
  or the server exposes only one of the supported OpenAI-compatible `/embeddings`
  or TEI `/embed` shapes. Run `xists doctor --check-endpoints --strict` after
  fixing it.
- `LLM endpoint is not configured`: set `LLM_API_KEY`, `LLM_BASE_URL`, and
  `LLM_MODEL`; this is required for ingest because profiles are LLM-generated.
- `GitHub token is not configured`: set `GITHUB_TOKEN`/`GITHUB_TOKENS` or pass
  `--token-file` before ingesting. Existing records, indexes, search, and eval do
  not need a GitHub token.
