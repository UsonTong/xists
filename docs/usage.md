# Usage

## Installation

Install xists in development mode:

```bash
pip install -e .
```

This makes the `xists` command available globally.

## Configuration

Create a local `.env` file from the example file:

```bash
cp .env.example .env
```

Edit `.env` and configure your credentials:

```env
# GitHub (required for ingest)
GITHUB_TOKEN=your_github_token_here

# LLM (required for ingest)
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5.4

# Embedding (required for index build and search)
EMBEDDING_API_KEY=your_embedding_api_key_here
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
```

All three sections are required for the full workflow. `.env` is ignored by Git.

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
| `--token-file` | (none) | File containing GitHub token instead of env var |
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

By default, `xists index build` is incremental. It skips repos that already have vectors in `index.json` and only embeds new records.

#### Checkpoint / resume

Each batch (64 records) is written to disk as it completes. If interrupted, completed batches are preserved.

#### Model mismatch protection

If `index.json` was built with a different embedding model than the one configured, `index build` refuses to run and asks you to rebuild or match the model. This prevents silent corruption from mixing incompatible vectors.

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
    {"repo_id": "react/react", "score": 0.68, "confidence": "high_confidence"},
    {"repo_id": "vuejs/core", "score": 0.61, "confidence": "high_confidence"}
  ],
  "considered": 8
}
```

#### Confidence tiers

| Tier | Cosine similarity | Meaning |
|------|-------------------|---------|
| `high_confidence` | ≥ 0.55 | Strong match, likely relevant |
| `exploratory` | ≥ 0.35 | Worth investigating |
| `abstain` | < 0.35 | Too weak, not shown |

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--index` | `index.json` | Index file to search |
| `--top-k` | `10` | Maximum results to return |

## Output files

### `records.json`

A JSON array of generated xists records. Each record includes GitHub metadata, README excerpt, structure signals, evidence, evidence gaps, and an LLM-generated profile.

See [Record Schema](record-schema.md) for the full format.

### `index.json`

A JSON object containing the embedding index. Includes metadata (model, dimension, timestamp) and a vectors array mapping each repo_id to its embedding vector.

### `report.json`

A JSON report for the ingest run. Includes input count, skipped count (incremental), generated count, failed count, and details for each failure.

## File management

| File | Gitignored | Description |
|------|------------|-------------|
| `.env` | Yes | Credentials and configuration |
| `repos.txt` | No | User-maintained repository list |
| `records.json` | Yes | Generated records (derived data) |
| `index.json` | Yes | Generated index (derived data) |
| `report.json` | Yes | Last ingest report |

To start fresh, delete `records.json` and `index.json` and re-run the workflow.
