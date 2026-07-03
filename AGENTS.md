# Repository Guidelines

## Project Structure & Module Organization

This is a Python package using a `src/` layout. Core code lives in `src/xists/`: `cli.py` defines the `xists` command, `ingest/` collects GitHub records, `profile/` builds LLM profiles, `search/` builds and queries embedding indexes, and `eval/` evaluates retrieval quality. Tests are in `tests/` and mirror feature areas (`test_cli.py`, `test_search.py`, etc.). Documentation lives in `docs/`; small example inputs are in `examples/`. Utility scripts, such as stratified eval generation, are in `scripts/`. Root-level `demo-*` JSON files are demo artifacts; generated outputs such as `records.json`, `index.json`, and `eval-report.json` should not be committed unless updating fixtures.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev]"` installs the package, console script, and pytest.
- `pytest` runs the full test suite used by CI.
- `python -m xists.cli --help` or `xists --help` shows available commands.
- `xists doctor --records demo-records.json --index demo-index.json --cases examples/eval-cases.json` checks local config and expected data files.
- `xists index build --records demo-records.json --output demo-index.json` rebuilds an embedding index from records.
- `xists eval run --cases examples/eval-cases.json --index demo-index.json --output demo-eval-report.json` runs retrieval evaluation.

## Coding Style & Naming Conventions

Use Python 3.11+ syntax and standard-library typing (`list[str]`, `dict[str, Any]`). Follow PEP 8 with 4-space indentation, clear function names, and small helpers for reusable parsing or normalization logic. Module and function names use `snake_case`; classes and exceptions use `PascalCase`. Keep CLI defaults aligned with tests and `docs/usage.md`. No formatter is configured; keep imports tidy and code explicit.

## Testing Guidelines

The project uses pytest. Add or update tests under `tests/` for every behavior change, using file names `test_*.py` and descriptive test functions such as `test_eval_run_writes_report`. Prefer `tmp_path`, `monkeypatch`, and `unittest.mock.patch` for filesystem, environment, and network/API isolation. Do not require real GitHub, LLM, or embedding credentials in tests.

## Commit & Pull Request Guidelines

Git history uses Conventional Commit-style subjects, especially `feat: ...` and `fix: ...`; keep messages imperative and scoped to one change. Pull requests should include a short description, commands run (usually `pytest`), linked issues when applicable, and before/after notes for CLI output, ranking, or generated reports. Update docs and examples when user-facing commands, schemas, or default paths change.

## Security & Configuration Tips

Copy `.env.example` to `.env` for local credentials. Never commit `.env`, `.secrets/`, token files, or generated private data. Keep experimental outputs in `data/` or ignored root-level artifact names.
