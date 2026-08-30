"""Records validation, statistics, and inspection CLI commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from xists.cli.common import (
    _counter_items,
    _format_top_items,
    _read_records_file,
    _safe_divide,
)
from xists.profile.llm import PROFILE_PROMPT_VERSION
from xists.records import (
    RECORD_SCHEMA_VERSION,
    record_profile,
    records_validation_report,
)


def records_inspect(args: argparse.Namespace) -> int:
    if not args.records.exists():
        print(
            f"Records file not found: {args.records}. Run 'xists ingest github' first.",
            file=sys.stderr,
        )
        return 2

    try:
        records = _read_records_file(args.records)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    filtered = records
    if args.repo:
        needle = args.repo.lower()
        filtered = [
            record
            for record in records
            if needle in str(record.get("repo_id") or record.get("repo_id_requested") or "").lower()
        ]

    inspected: list[dict[str, Any]] = []
    for record in filtered[: max(args.limit, 0)]:
        github_raw = record.get("github")
        github: dict[str, Any] = github_raw if isinstance(github_raw, dict) else {}
        profile_raw = record.get("llm_profile")
        profile: dict[str, Any] = profile_raw if isinstance(profile_raw, dict) else {}
        inspected.append(
            {
                "schema_version": record.get("schema_version"),
                "repo_id": record.get("repo_id") or record.get("repo_id_requested"),
                "name": record.get("name"),
                "url": record.get("url"),
                "language": github.get("language"),
                "topics": github.get("topics") or [],
                "has_readme": bool(record.get("readme")),
                "profile_confidence": profile.get("confidence"),
                "profile_abstained": bool(profile.get("abstained")),
                "summary": profile.get("summary"),
                "aliases": profile.get("aliases") or [],
                "project_type": profile.get("project_type"),
                "ecosystem": profile.get("ecosystem") or [],
                "search_text_preview": (profile.get("search_text") or "")[:160],
            }
        )

    payload = {
        "records": str(args.records),
        "record_count": len(records),
        "matching_count": len(filtered),
        "inspected_count": len(inspected),
        "filter": {"repo": args.repo, "limit": args.limit},
        "items": inspected,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _records_stats_report(records: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    validation = records_validation_report(
        records, expected_profile_prompt_version=PROFILE_PROMPT_VERSION
    )
    languages: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    project_types: Counter[str] = Counter()
    ecosystems: Counter[str] = Counter()
    confidence: Counter[str] = Counter()

    for record in records:
        github_raw = record.get("github")
        github: dict[str, Any] = github_raw if isinstance(github_raw, dict) else {}
        language = github.get("language")
        if isinstance(language, str) and language.strip():
            languages[language] += 1
        for topic in github.get("topics") or []:
            if isinstance(topic, str) and topic.strip():
                topics[topic] += 1

        profile = record_profile(record)
        if profile.get("project_type"):
            project_types[str(profile["project_type"])] += 1
        for ecosystem in profile.get("ecosystem") or []:
            ecosystems[ecosystem] += 1
        confidence[str(profile.get("confidence") or "low")] += 1

    quality = validation.get("quality") or {}
    return {
        "record_count": len(records),
        "schema_version": RECORD_SCHEMA_VERSION,
        "schema_versions": validation.get("schema_versions") or {},
        "profile_prompt_version": PROFILE_PROMPT_VERSION,
        "prompt_versions": validation.get("prompt_versions") or {},
        "quality": quality,
        "confidence": dict(confidence),
        "ratios": {
            "abstained": _safe_divide(quality.get("profile_abstained", 0), len(records)),
            "low_confidence": _safe_divide(quality.get("low_confidence", 0), len(records)),
            "archived": _safe_divide(quality.get("archived", 0), len(records)),
            "disabled": _safe_divide(quality.get("disabled", 0), len(records)),
            "missing_readme": _safe_divide(quality.get("missing_readme", 0), len(records)),
        },
        "top_languages": _counter_items(languages, "language", limit),
        "top_topics": _counter_items(topics, "topic", limit),
        "top_project_types": _counter_items(project_types, "project_type", limit),
        "top_ecosystems": _counter_items(ecosystems, "ecosystem", limit),
    }


def _format_records_stats_text(report: dict[str, Any], records_path: Path) -> str:
    quality = report.get("quality") or {}
    ratios = report.get("ratios") or {}
    lines = [
        f"records: {records_path}",
        f"schema: expected {report['schema_version']}",
        f"repos: {report['record_count']}",
        f"profile_prompt_version: {report['profile_prompt_version']}",
        "",
        "quality:",
    ]
    for key in (
        "missing_search_text",
        "missing_aliases",
        "search_text_too_short",
        "profile_abstained",
        "low_confidence",
        "archived",
        "disabled",
        "missing_readme",
        "duplicates",
    ):
        ratio = ratios.get("abstained" if key == "profile_abstained" else key)
        suffix = f" ({ratio:.2%})" if isinstance(ratio, float) else ""
        lines.append(f"  {key}: {quality.get(key, 0)}{suffix}")
    lines.extend(
        [
            "",
            "distribution:",
            f"  confidence: {report.get('confidence') or {}}",
            f"  schema_versions: {report.get('schema_versions') or {}}",
            f"  prompt_versions: {report.get('prompt_versions') or {}}",
            "",
            "top:",
            f"  languages: {_format_top_items(report.get('top_languages') or [], 'language')}",
            f"  topics: {_format_top_items(report.get('top_topics') or [], 'topic')}",
            f"  project_types: {_format_top_items(report.get('top_project_types') or [], 'project_type')}",
            f"  ecosystems: {_format_top_items(report.get('top_ecosystems') or [], 'ecosystem')}",
        ]
    )
    return "\n".join(lines)


def records_stats(args: argparse.Namespace) -> int:
    if not args.records.exists():
        print(
            f"Records file not found: {args.records}. Run 'xists ingest github' first.",
            file=sys.stderr,
        )
        return 2
    try:
        records = _read_records_file(args.records)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    report = _records_stats_report(records, limit=args.limit)
    report["records"] = str(args.records)
    if getattr(args, "format", "text") == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format_records_stats_text(report, args.records))
    return 0


def _records_next_steps(records_path: Path, report: dict[str, Any] | None = None) -> list[str]:
    errors = (report or {}).get("errors") or {}
    warnings = (report or {}).get("warnings") or {}
    steps: list[str] = []
    if any(
        key in errors
        for key in (
            "schema_version_mismatch",
            "missing_llm_profile",
            "missing_summary",
            "missing_search_text",
        )
    ):
        steps.append(
            f"Refresh profiles: xists profile refresh --records {records_path} --output records-v2.json"
        )
    if errors.get("duplicate_repo_id"):
        steps.append(
            "Review duplicate repo_id entries and keep one canonical record per repository."
        )
    if warnings.get("search_text_too_short") or warnings.get("missing_aliases"):
        steps.append("Review weak profiles or refresh them with xists profile refresh.")
    if warnings.get("profile_abstained") or warnings.get("low_confidence_profile"):
        steps.append(
            "Inspect low-confidence or abstained profiles before sharing this records file."
        )
    if errors:
        steps.append(
            "Rebuild the index after records are fixed: xists index build --records records-v2.json --output index.json"
        )
    return steps or ["No required action; records passed validation."]


def _format_records_validation_text(report: dict[str, Any], records_path: Path) -> str:
    quality = report.get("quality") or {}
    lines = [
        f"records: {records_path}",
        f"schema: expected {report['schema_version']}",
        f"repos: {report['record_count']}",
        f"ok: {str(report['ok']).lower()}",
        "",
        "quality:",
    ]
    for key in (
        "ok",
        "missing_search_text",
        "missing_aliases",
        "search_text_too_short",
        "profile_abstained",
        "low_confidence",
        "archived",
        "disabled",
        "missing_readme",
        "duplicates",
    ):
        lines.append(f"  {key}: {quality.get(key, 0)}")
    for label in ("errors", "warnings"):
        items = report.get(label) or {}
        lines.append("")
        lines.append(f"{label}:")
        if items:
            for key, value in sorted(items.items()):
                lines.append(f"  {key}: {value}")
        else:
            lines.append("  none")
    next_steps = report.get("next_steps") or []
    if next_steps:
        lines.append("")
        lines.append("next steps:")
        for step in next_steps:
            lines.append(f"  - {step}")
    return "\n".join(lines)


def records_validate(args: argparse.Namespace) -> int:
    if not args.records.exists():
        print(
            f"Records file not found: {args.records}. Run 'xists ingest github' first.",
            file=sys.stderr,
        )
        return 2
    try:
        records = _read_records_file(args.records)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    report = records_validation_report(
        records, expected_profile_prompt_version=PROFILE_PROMPT_VERSION
    )
    report["records"] = str(args.records)
    report["next_steps"] = [] if report["ok"] else _records_next_steps(args.records, report)
    if getattr(args, "format", "text") == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format_records_validation_text(report, args.records))
    return 0 if report["ok"] else 1


__all__ = [
    "_format_records_stats_text",
    "_format_records_validation_text",
    "_records_next_steps",
    "_records_stats_report",
    "records_inspect",
    "records_stats",
    "records_validate",
]
