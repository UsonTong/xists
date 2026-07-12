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

## 3. Ingest example repositories

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

## 4. Build the example index

```bash
xists index build \
  --records demo-records.json \
  --output demo-index.json
```

## 5. Run a few searches

```bash
xists search "frontend ui library" --index demo-index.json
xists search "python web framework for building apis" --index demo-index.json
xists search "open source workflow automation platform" --index demo-index.json
```

Expected results vary by model and repository data, but the top results should usually come from the same neighborhood as the query.

## 6. Optionally evaluate the demo index

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

## 7. Inspect misses and iterate

After changing the repository list, regenerating summaries, or adjusting search behavior, rerun:

```bash
xists index build --records demo-records.json --output demo-index.json --force
xists eval run --cases examples/eval-cases.json --index demo-index.json --output demo-eval-report.json
xists eval inspect --report demo-eval-report.json --status serious_mismatch
```

That gives you a stable way to sanity-check search behavior before moving on to larger repository lists or evaluation datasets.
