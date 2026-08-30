"""Profile refresh command handler and checkpoint management."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xists.cli.common import (
    _failed_repo_ids_from_report,
    _failure_entry,
    _format_command_summary,
    _format_dry_run_text,
    write_json,
    write_json_atomic,
)
from xists.profile.llm import (
    PROFILE_PROMPT_VERSION,
    LLMError,
    LLMNotConfiguredError,
    attach_llm_profile,
    generate_llm_profile,
    llm_config_from_env,
)
from xists.records import RECORD_SCHEMA_VERSION, profile_refresh_reason, record_repo_id


def _profile_refresh_checkpoint_path(output: Path) -> Path:
    return Path(f"{output}.partial.jsonl")


def _load_profile_refresh_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    refreshed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return refreshed

    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            if all(not candidate.strip() for candidate in lines[index + 1 :]):
                break
            raise
        if isinstance(record, dict):
            repo_id = record_repo_id(record)
            if repo_id:
                refreshed[repo_id] = record
    return refreshed


def _append_profile_refresh_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _profile_refresh_report_payload(
    *,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_records: list[dict[str, Any]],
    refreshed: int,
    resumed: int,
    skipped: int,
    failed: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "records": str(args.records),
        "output": str(args.output),
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "profile_prompt_version": PROFILE_PROMPT_VERSION,
        "input_count": len(records),
        "refreshed_count": refreshed,
        "resumed_count": resumed,
        "skipped_count": skipped,
        "failed_count": len(failed),
        "failed": failed,
        "attempted_at": datetime.now(UTC).isoformat(),
        "output_count": len(output_records),
    }


def profile_refresh(args: argparse.Namespace) -> int:
    if not args.records.exists():
        print(
            f"Records file not found: {args.records}. Run 'xists ingest github' first.",
            file=sys.stderr,
        )
        return 2

    records = json.loads(args.records.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print(f"Records file must contain a JSON list: {args.records}", file=sys.stderr)
        return 1

    retry_failed: set[str] | None = None
    if getattr(args, "retry_failed", None):
        try:
            retry_failed = _failed_repo_ids_from_report(args.retry_failed)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1

    if getattr(args, "dry_run", False):
        skipped_reasons: Counter[str] = Counter()
        to_process = 0
        to_skip = 0
        for record in records:
            repo_id = record_repo_id(record) or "<unknown>"
            reason = (
                "force"
                if args.force
                else profile_refresh_reason(
                    record,
                    only_missing_search_text=bool(args.only_missing_search_text),
                    expected_prompt_version=PROFILE_PROMPT_VERSION,
                )
            )
            if retry_failed is not None:
                reason = "retry_failed" if repo_id in retry_failed else None
            if reason is None:
                to_skip += 1
                continue
            to_process += 1
            skipped_reasons[reason] += 1
        report = {
            "total": len(records),
            "to_process": to_process,
            "to_skip": to_skip,
            "skip_reasons": dict(skipped_reasons),
            "estimated_calls": to_process,
        }
        if getattr(args, "format", "text") == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(_format_dry_run_text("profile refresh", report))
        return 0

    try:
        _llm_config_from_env = getattr(
            sys.modules.get("xists.cli"), "llm_config_from_env", llm_config_from_env
        )
        config = _llm_config_from_env()
    except LLMNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    workers = getattr(args, "workers", 1) or 1
    if workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 1

    checkpoint_path = _profile_refresh_checkpoint_path(args.output)
    if checkpoint_path.exists() and not getattr(args, "resume", False):
        print(
            f"Checkpoint file already exists: {checkpoint_path}\n"
            "Next steps:\n"
            f"  1. Re-run with --resume to continue from the checkpoint\n"
            f"  2. Delete {checkpoint_path} if you want to restart from scratch",
            file=sys.stderr,
        )
        return 1

    resumed_records = (
        _load_profile_refresh_checkpoint(checkpoint_path) if getattr(args, "resume", False) else {}
    )

    refreshed = 0
    resumed = 0
    skipped = 0
    failed: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []
    processed = 0
    total = len(records)

    def prepare_record(record: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        updated = dict(record)
        repo_id = record_repo_id(updated) or "<unknown>"
        reason = (
            "force"
            if args.force
            else profile_refresh_reason(
                updated,
                only_missing_search_text=bool(args.only_missing_search_text),
                expected_prompt_version=PROFILE_PROMPT_VERSION,
            )
        )
        if retry_failed is not None:
            reason = "retry_failed" if repo_id in retry_failed else None
        return updated, reason

    try:
        _generate_llm_profile = getattr(
            sys.modules.get("xists.cli"), "generate_llm_profile", generate_llm_profile
        )
        if workers == 1:
            for record in records:
                updated, reason = prepare_record(record)
                repo_id = record_repo_id(updated) or "<unknown>"
                if repo_id in resumed_records:
                    output_records.append(dict(resumed_records[repo_id]))
                    resumed += 1
                elif reason is None:
                    updated["schema_version"] = RECORD_SCHEMA_VERSION
                    output_records.append(updated)
                    skipped += 1
                else:
                    try:
                        profile = _generate_llm_profile(updated, config)
                        attach_llm_profile(updated, profile)
                        updated["schema_version"] = RECORD_SCHEMA_VERSION
                        _append_profile_refresh_checkpoint(checkpoint_path, updated)
                        output_records.append(updated)
                        refreshed += 1
                    except LLMError as error:
                        failed.append(_failure_entry(repo_id, str(error), refresh_reason=reason))
                        output_records.append(updated)

                processed += 1
                if processed % 10 == 0 or processed == total:
                    print(
                        f"profile refresh progress: {processed}/{total} processed "
                        f"({refreshed} refreshed, {resumed} resumed, {len(failed)} failed, {skipped} skipped)",
                        file=sys.stderr,
                        flush=True,
                    )
        else:
            output_by_repo_id: dict[str, dict[str, Any]] = {}
            pending: list[tuple[dict[str, Any], str]] = []
            for record in records:
                updated, reason = prepare_record(record)
                repo_id = record_repo_id(updated) or "<unknown>"
                if repo_id in resumed_records:
                    output_by_repo_id[repo_id] = dict(resumed_records[repo_id])
                    resumed += 1
                elif reason is None:
                    updated["schema_version"] = RECORD_SCHEMA_VERSION
                    output_by_repo_id[repo_id] = updated
                    skipped += 1
                else:
                    pending.append((updated, reason))

            processed = resumed + skipped
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_generate_llm_profile, updated, config): (updated, reason)
                    for updated, reason in pending
                }
                for future in as_completed(futures):
                    updated, reason = futures[future]
                    repo_id = record_repo_id(updated) or "<unknown>"
                    try:
                        profile = future.result()
                        attach_llm_profile(updated, profile)
                        updated["schema_version"] = RECORD_SCHEMA_VERSION
                        _append_profile_refresh_checkpoint(checkpoint_path, updated)
                        refreshed += 1
                    except LLMError as error:
                        failed.append(_failure_entry(repo_id, str(error), refresh_reason=reason))
                    output_by_repo_id[repo_id] = updated
                    processed += 1
                    if processed % 10 == 0 or processed == total:
                        print(
                            f"profile refresh progress: {processed}/{total} processed "
                            f"({refreshed} refreshed, {resumed} resumed, {len(failed)} failed, {skipped} skipped)",
                            file=sys.stderr,
                            flush=True,
                        )

            output_records = [
                output_by_repo_id[record_repo_id(record) or "<unknown>"] for record in records
            ]

        write_json_atomic(args.output, output_records)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    except Exception as error:
        failed_id = (
            record_repo_id(records[processed]) if processed < len(records) else "<unknown>"
        ) or "<unknown>"
        if args.report:
            write_json(
                args.report,
                _profile_refresh_report_payload(
                    args=args,
                    records=records,
                    output_records=output_records,
                    refreshed=refreshed,
                    resumed=resumed,
                    skipped=skipped,
                    failed=[
                        *failed,
                        _failure_entry(
                            failed_id,
                            str(error),
                            refresh_reason="interrupted",
                        ),
                    ],
                ),
            )
        print(f"profile refresh failed: {error}", file=sys.stderr)
        return 1
    summary = _profile_refresh_report_payload(
        args=args,
        records=records,
        output_records=output_records,
        refreshed=refreshed,
        resumed=resumed,
        skipped=skipped,
        failed=failed,
    )
    if args.report:
        write_json(args.report, summary)
    if getattr(args, "format", "text") == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            _format_command_summary(
                "Profiles refreshed",
                [
                    ("Records", args.records),
                    ("Output", args.output),
                    ("Refreshed", refreshed),
                    ("Resumed", resumed),
                    ("Skipped", skipped),
                    ("Failed", len(failed)),
                    ("Report", args.report),
                ],
                stream=sys.stdout,
            )
        )
    if failed:
        print(
            f"profile refresh finished with {len(failed)} failed records; report written to {args.report or 'stdout'}",
            file=sys.stderr,
            flush=True,
        )
    return 1 if failed and refreshed == 0 and total > 0 else 0


__all__ = [
    "_append_profile_refresh_checkpoint",
    "_load_profile_refresh_checkpoint",
    "_profile_refresh_checkpoint_path",
    "_profile_refresh_report_payload",
    "profile_refresh",
]
