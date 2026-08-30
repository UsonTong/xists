"""Shared CLI helpers, formatting utilities, and filesystem operations."""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xists.ingest.github import parse_github_repo
from xists.search.embed import EmbeddingError
from xists.search.index import load_index
from xists.search.transform import (
    QueryTransformError,
    query_transform_config_from_env,
    query_variants,
    transform_queries,
)
from xists.terminal import style
from xists.workspace import resolve_workspace


def load_env_file(
    path: Path,
    *,
    override: bool = False,
    protected_keys: set[str] | None = None,
) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or "=" not in value:
            continue
        key, env_value = value.split("=", 1)
        key = key.strip()
        env_value = env_value.strip().strip('"').strip("'")
        if key and key not in (protected_keys or set()) and (override or key not in os.environ):
            os.environ[key] = env_value


def load_workspace_environment(workspace: Any, *, cwd: Path | None = None) -> None:
    """Load workspace and per-directory configuration without replacing shell values."""
    shell_keys = set(os.environ)
    load_env_file(workspace.env_file)
    current_env_file = (Path.cwd() if cwd is None else cwd) / ".env"
    load_env_file(current_env_file, override=True, protected_keys=shell_keys)


def load_repo_ids(path: Path) -> list[str]:
    repo_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        repo_ids.append(parse_github_repo(value))
    return repo_ids


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _prepare_query_transforms(
    queries: list[str], mode: str
) -> tuple[list[list[str]] | None, list[str] | None, str | None]:
    if mode == "off":
        return None, None, None
    config = query_transform_config_from_env()
    canonical_queries = transform_queries(config, queries)
    return (
        [
            query_variants(query, canonical, mode)
            for query, canonical in zip(queries, canonical_queries)
        ],
        canonical_queries,
        config.model,
    )


def _load_canonical_queries(path: Path, cases: list[dict[str, Any]]) -> list[str]:
    """Load a complete, case-id keyed canonical-query snapshot for an eval run."""
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise QueryTransformError(f"Canonical query file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise QueryTransformError(f"Canonical query file is not valid JSON: {error}") from error
    if not isinstance(values, dict):
        raise QueryTransformError(
            "Canonical query file must be an object mapping case id to query text"
        )

    expected_ids = [str(case["id"]) for case in cases]
    expected_id_set = set(expected_ids)
    actual_id_set = set(values)
    missing = [case_id for case_id in expected_ids if case_id not in actual_id_set]
    unexpected = sorted(str(case_id) for case_id in actual_id_set - expected_id_set)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing case ids: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected case ids: {', '.join(unexpected)}")
        raise QueryTransformError(
            "Canonical query file must match eval cases exactly; " + "; ".join(details)
        )

    canonical_queries = [values[case_id] for case_id in expected_ids]
    if any(not isinstance(query, str) or not query.strip() for query in canonical_queries):
        raise QueryTransformError("Canonical query file values must be non-empty strings")
    return [query.strip() for query in canonical_queries]


def _format_dry_run_text(title: str, report: dict[str, Any]) -> str:
    skip_reasons = report.get("skip_reasons") or {}
    lines = [
        f"{title} dry run",
        "this was a dry run, nothing was written",
        f"total: {report.get('total')}",
        f"to_process: {report.get('to_process')}",
        f"to_skip: {report.get('to_skip')}",
        "skip_reasons:",
    ]
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items()):
            lines.append(f"  {reason}: {count}")
    else:
        lines.append("  none: 0")
    lines.append(f"estimated_calls: {report.get('estimated_calls')}")
    return "\n".join(lines)


def _failed_repo_ids_from_report(path: Path) -> set[str]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Failure report not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Failure report is not valid JSON: {error}") from error
    failed = report.get("failed") if isinstance(report, dict) else None
    if not isinstance(failed, list):
        raise ValueError(f"Failure report must contain a failed list: {path}")
    repo_ids: set[str] = set()
    for item in failed:
        if isinstance(item, dict):
            repo_id = item.get("repo_id")
            if isinstance(repo_id, str) and repo_id.strip():
                repo_ids.add(repo_id.strip())
    return repo_ids


def _failure_entry(repo_id: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "reason": reason,
        "error": reason,
        "attempted_at": datetime.now(UTC).isoformat(),
        **extra,
    }


def _check_payload(name: str, status: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, **extra}


def _embedding_config_next_steps() -> list[str]:
    config_file = resolve_workspace().env_file
    return [
        f"Run xists init to create {config_file}, if needed.",
        f"Set EMBEDDING_API_KEY, EMBEDDING_BASE_URL, and EMBEDDING_MODEL in {config_file}.",
        "Run xists doctor --check-endpoints after setting the embedding variables.",
    ]


def _llm_config_next_steps() -> list[str]:
    config_file = resolve_workspace().env_file
    return [
        f"Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL in {config_file} or the environment.",
        "LLM configuration is required for xists ingest github and optional eval --llm-judge runs.",
    ]


def _github_token_next_steps() -> list[str]:
    config_file = resolve_workspace().env_file
    return [
        f"Set GITHUB_TOKEN or GITHUB_TOKENS in {config_file}, or pass --token-file.",
        "GitHub tokens are required for xists ingest github but not for local search/eval on existing files.",
    ]


def _embedding_endpoint_next_steps() -> list[str]:
    return [
        "Start the embedding service referenced by EMBEDDING_BASE_URL.",
        "Confirm the base URL is the API root, for example http://localhost:6597/v1 for OpenAI-compatible servers.",
        "Run xists doctor --check-endpoints --strict before retrying index/search/eval commands.",
    ]


def _print_embedding_error(error: EmbeddingError, *, command: str) -> None:
    next_steps = "\n".join(f"- {step}" for step in _embedding_endpoint_next_steps())
    print(
        f"xists {command} could not use the configured embedding endpoint.\n"
        f"{error}\n"
        "Next steps:\n"
        f"{next_steps}",
        file=sys.stderr,
    )


def _read_json_file(path: Path, *, kind: str) -> Any:
    """Read a JSON file and attach its role and path to parse errors."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {kind} JSON: {path}: {error}") from error


def _read_records_file(path: Path) -> list[dict[str, Any]]:
    records = _read_json_file(path, kind="records")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Records file must contain a JSON list of objects: {path}")
    return records


def _read_index_file(path: Path) -> dict[str, Any]:
    try:
        index = load_index(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read index JSON: {path}: {error}") from error
    if not isinstance(index, dict):
        raise ValueError(f"Index file must contain a JSON object: {path}")
    return index


def _format_command_summary(title: str, rows: list[tuple[str, Any]], *, stream: Any = None) -> str:
    lines = [style(title, "title", stream=stream)]
    max_label_len = max(len(label) for label, _ in rows)
    for label, value in rows:
        formatted_label = f"   {label}:".ljust(max_label_len + 6)
        formatted_value = value if isinstance(value, str) else str(value)
        lines.append(f"{formatted_label} {formatted_value}")
    return "\n".join(lines)


def _counter_items(counter: Counter[str], key_name: str, limit: int) -> list[dict[str, Any]]:
    return [{key_name: key, "count": count} for key, count in counter.most_common(limit)]


def _safe_divide(numerator: Any, denominator: int) -> float:
    if denominator <= 0 or numerator is None:
        return 0.0
    return round(float(numerator) / float(denominator), 3)


def _format_top_items(items: list[dict[str, Any]], key_name: str) -> str:
    if not items:
        return "none"
    return ", ".join(f"{item[key_name]} ({item['count']})" for item in items)


def _format_search_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value) if value is not None else "n/a"


def _terminal_width() -> int:
    return max(20, shutil.get_terminal_size(fallback=(88, 24)).columns)


def _wrap_terminal_text(value: str, *, width: int, indent: int = 0) -> list[str]:
    available_width = max(8, width - indent)
    text = " ".join(value.split())
    if not text:
        return [" " * indent]
    return [
        " " * indent + line
        for line in textwrap.wrap(
            text,
            width=available_width,
            break_long_words=True,
            break_on_hyphens=False,
        )
    ]


__all__ = [
    "_check_payload",
    "_counter_items",
    "_embedding_config_next_steps",
    "_embedding_endpoint_next_steps",
    "_failed_repo_ids_from_report",
    "_failure_entry",
    "_format_command_summary",
    "_format_dry_run_text",
    "_format_search_number",
    "_format_top_items",
    "_github_token_next_steps",
    "_llm_config_next_steps",
    "_load_canonical_queries",
    "_prepare_query_transforms",
    "_print_embedding_error",
    "_read_index_file",
    "_read_json_file",
    "_read_records_file",
    "_safe_divide",
    "_terminal_width",
    "_wrap_terminal_text",
    "load_env_file",
    "load_repo_ids",
    "load_workspace_environment",
    "write_json",
    "write_json_atomic",
]
