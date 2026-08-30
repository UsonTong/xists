"""Doctor command handler and environment diagnostic tools."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from xists.cli.common import (
    _check_payload,
    _embedding_config_next_steps,
    _embedding_endpoint_next_steps,
    _github_token_next_steps,
    _llm_config_next_steps,
)
from xists.ingest.github import (
    github_token_from_env,
    github_token_from_file,
)
from xists.profile.llm import (
    LLMNotConfiguredError,
    llm_config_from_env,
)
from xists.search.embed import (
    EmbeddingError,
    EmbeddingNotConfiguredError,
    embedding_config_from_env,
    probe_embedding_endpoint,
)
from xists.terminal import TerminalRole, style
from xists.workspace import resolve_workspace


def _format_doctor_text(payload: dict[str, Any], *, stream: Any = None) -> str:
    stream = stream or sys.stdout
    checks = payload.get("checks") or []
    workspace = payload.get("workspace") or {}
    check_labels = {
        "embedding_config": "Embedding configuration",
        "embedding_endpoint": "Embedding endpoint",
        "llm_config": "LLM configuration",
        "github_token": "GitHub token",
        "records_file": "Records file",
        "index_file": "Index file",
        "eval_cases_file": "Evaluation cases file",
    }
    lines = [style("Doctor", "title", stream=stream)]
    lines.append(
        style(
            "Ready" if payload.get("ok") else "Needs attention",
            "success" if payload.get("ok") else "warning",
            stream=stream,
        )
    )
    if isinstance(workspace, dict):
        mode = (
            "legacy current directory" if workspace.get("mode") == "legacy" else "default workspace"
        )
        lines.extend(
            [
                "",
                style("Workspace", "title", stream=stream),
                f"  Mode      {mode}",
                f"  Location  {workspace.get('root') or 'unknown'}",
            ]
        )
        paths = workspace.get("paths")
        if isinstance(paths, dict):
            for label, key in (("Records", "records"), ("Index", "index"), ("Cases", "cases")):
                if paths.get(key):
                    lines.append(f"  {label.ljust(9)} {paths[key]}")
    next_steps: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        status = str(check.get("status") or "unknown").upper()
        role: TerminalRole = (
            "success" if status == "OK" else "warning" if status == "WARN" else "error"
        )
        name = str(check.get("name") or "check")
        label = check_labels.get(name, name.replace("_", " ").title())
        lines.append(
            f"  {style(status.ljust(5), role, stream=stream)} {label}: {check.get('message') or ''}"
        )
        for step in check.get("next_steps") or []:
            if isinstance(step, str) and step not in next_steps:
                next_steps.append(step)
    if next_steps:
        lines.extend(["", style("Next steps", "title", stream=stream)])
        lines.extend(f"  {position}. {step}" for position, step in enumerate(next_steps, start=1))
    return "\n".join(lines)


def doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []
    embedding_config = None
    workspace = getattr(args, "workspace", resolve_workspace())

    try:
        config = embedding_config_from_env()
        embedding_config = config
        checks.append(
            _check_payload(
                "embedding_config",
                "ok",
                "embedding endpoint is configured",
                model=config.model,
                base_url=config.base_url,
            )
        )
    except EmbeddingNotConfiguredError as error:
        checks.append(
            _check_payload(
                "embedding_config",
                "error",
                str(error),
                next_steps=_embedding_config_next_steps(),
            )
        )

    check_endpoints = bool(
        getattr(args, "check_endpoints", False) or getattr(args, "strict", False)
    )
    strict = bool(getattr(args, "strict", False))
    if check_endpoints and embedding_config is not None:
        try:
            _probe_embedding_endpoint = getattr(
                sys.modules.get("xists.cli"), "probe_embedding_endpoint", probe_embedding_endpoint
            )
            probe = _probe_embedding_endpoint(embedding_config)
            checks.append(
                _check_payload(
                    "embedding_endpoint",
                    "ok",
                    "embedding endpoint responded to a probe request",
                    model=probe.get("model"),
                    dimension=probe.get("dimension"),
                    resolved_url=probe.get("resolved_url"),
                    response_kind=probe.get("response_kind"),
                )
            )
        except EmbeddingError as error:
            checks.append(
                _check_payload(
                    "embedding_endpoint",
                    "error" if strict else "warn",
                    str(error),
                    model=embedding_config.model,
                    base_url=embedding_config.base_url,
                    hint="Start the embedding service, fix EMBEDDING_BASE_URL, or rerun without --strict.",
                    next_steps=_embedding_endpoint_next_steps(),
                )
            )

    try:
        llm_cfg = llm_config_from_env()
        checks.append(
            _check_payload(
                "llm_config",
                "ok",
                "LLM endpoint is configured",
                model=llm_cfg.model,
                base_url=llm_cfg.base_url,
            )
        )
    except LLMNotConfiguredError as error:
        checks.append(
            _check_payload("llm_config", "error", str(error), next_steps=_llm_config_next_steps())
        )

    try:
        tokens = (
            github_token_from_file(args.token_file) if args.token_file else github_token_from_env()
        )
        if tokens:
            checks.append(
                _check_payload(
                    "github_token", "ok", "GitHub token is configured", token_count=len(tokens)
                )
            )
        else:
            checks.append(
                _check_payload(
                    "github_token",
                    "warn",
                    "GitHub token is not configured",
                    token_count=0,
                    next_steps=_github_token_next_steps(),
                )
            )
    except Exception as error:
        checks.append(
            _check_payload(
                "github_token", "error", str(error), next_steps=_github_token_next_steps()
            )
        )

    for name, path in (
        ("records_file", args.records),
        ("index_file", args.index),
        ("eval_cases_file", args.cases),
    ):
        if path.exists():
            checks.append(_check_payload(name, "ok", f"{path} exists", path=str(path)))
        else:
            command_hint = {
                "records_file": f"Run xists ingest github to create {path}, or pass --records to point at an existing records file.",
                "index_file": f"Run xists index build to create {path}, or pass --index to point at an existing index file.",
                "eval_cases_file": "Pass --cases examples/eval-cases.json for the committed demo evaluation dataset.",
            }[name]
            checks.append(
                _check_payload(
                    name,
                    "warn",
                    f"{path} does not exist yet",
                    path=str(path),
                    next_steps=[command_hint],
                )
            )

    ok = all(check["status"] != "error" for check in checks)
    payload = {
        "ok": ok,
        "workspace": {
            "mode": workspace.mode,
            "root": str(workspace.root),
            "config_file": str(workspace.env_file),
            "paths": {
                "records": str(args.records.expanduser().resolve()),
                "index": str(args.index.expanduser().resolve()),
                "cases": str(args.cases.expanduser().resolve()),
            },
        },
        "checks": checks,
    }
    if getattr(args, "format", "text") == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_format_doctor_text(payload, stream=sys.stdout))
    return 0 if ok else 1


__all__ = [
    "_format_doctor_text",
    "doctor",
]
