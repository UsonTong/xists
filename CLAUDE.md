# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

Environment management uses `uv` (or standard `pip`):

- **Install Dependencies**: `uv sync --all-extras` (or `python -m pip install -e ".[dev]"`)
- **Run All Tests**: `uv run pytest` (or `pytest`)
- **Run Single Test File**: `uv run pytest tests/test_search.py`
- **Run Specific Test**: `uv run pytest tests/test_search.py -k "test_rank_basic"`
- **Lint Check**: `uv run ruff check src tests`
- **Lint Auto-fix**: `uv run ruff check --fix src tests`
- **Format Check**: `uv run ruff format --check src tests`
- **Format Apply**: `uv run ruff format src tests`
- **Type Check**: `uv run mypy src/xists`
- **Run CI Smoke Check**: `python scripts/smoke_check.py`
- **Run Retrieval Regression**: `python scripts/run_retrieval_regression.py`
- **Build Package Artifacts**: `python -m build`
- **CLI Development Run**: `python -m xists.cli --help` or `uv run xists --help`

## Architecture & Code Organization

`xists` is a local semantic search engine for curated GitHub repositories. It transforms raw repo metadata and READMEs into structured LLM profiles, builds local embedding indexes, and provides fast semantic retrieval via CLI, Python API, or MCP server.

### Core Subsystems (`src/xists/`)

- **Domain Types (`types.py`)**: Central `TypedDict` schema definitions (`Record`, `GitHubMetadata`, `LLMProfile`, `StructureSignals`, `IndexManifest`, `SearchResponse`, `EvalCase`, `EvalReport`).
- **Programmatic API (`api.py`)**: Stable, side-effect-free public entry points (`search()`, `load_index()`). Does not load `.env`, read process env, or print to stdout.
- **Workspace Resolver (`workspace.py`)**: Resolves workspace paths (`~/.xists` by default, or legacy local repo directory if data files exist), handles `.env` template generation, and initializes demo starter data.
- **Record Pipeline (`records.py`, `ingest/`, `profile/`)**:
  - `ingest/github.py`: Collects GitHub repo metadata and READMEs via REST or GraphQL with rate-limiting and checkpointing.
  - `profile/llm.py`: Calls LLM endpoints to generate compact JSON profiles (`summary`, `use_cases`, `capabilities`, `search_text`).
  - `records.py`: Schema validation, normalization, and fingerprinting.
- **Search & Index Engine (`search/`)**:
  - `index.py`: Dual-file index storage (v4 manifest `index.json` + binary float32 `index.vectors.npy`; backwards-compatible with v3 Base64).
  - `embed.py`: Batched embedding API requests with fingerprinting and concurrency support.
  - `query.py`: Core vector retrieval via NumPy cosine similarity matrix math blended with metadata scores (`rank()`).
  - `local_embed.py`: Offline lexical/metadata fallback search without remote embedding APIs.
  - `pull.py`, `append.py`, `merge.py`, `prune.py`: Prebuilt index downloading, incremental record appending, multi-index fusion, and lifecycle pruning of archived/disabled repos.
  - `rerank.py`, `confidence.py`, `transform.py`: Optional cross-encoder reranking, confidence calibration, and query transformation.
- **Evaluation Engine (`eval/`)**:
  - `schema.py`, `run.py`, `inspect.py`, `judge.py`: Retrieval benchmark harness computing Recall@K, MRR, mismatch inspection, and optional LLM pairwise judging.
- **MCP Server (`mcp_server.py`)**: Model Context Protocol stdio server exposing `search_projects`, `inspect_project`, and `index_stats` tools for AI agents.
- **CLI Subcommands (`cli/`)**:
  - Modular sub-parsers: `search`, `doctor`, `ingest`, `profile`, `index`, `records`, `eval`, `workspace`.
  - Common utilities (`cli/common.py`): atomic file writes (`write_json_atomic`), resilient JSONL checkpointing for long-running batch operations, error formatters.
- **Starter Bundle (`starter/`)**: Zero-config offline starter dataset (200 curated repos and precomputed binary vectors).

### Key Design & Testing Conventions

- **Typing**: Python 3.11+ standard library types (`list[str]`, `dict[str, Any]`, `TypedDict`). Pass `mypy src/xists` with 0 errors.
- **Resilient I/O**: CLI batch commands (`ingest`, `profile`, `index build`) support `--resume` from JSONL checkpoints and write final files atomically.
- **Test Isolation**: Unit and integration tests in `tests/` must never require real network or paid API credentials; use `tmp_path`, `monkeypatch`, mocks, or synthetic fixture embeddings.
- **Commit Messages**: Follow Conventional Commits format (`feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`).
