"""Ingest command handler for GitHub repositories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any

from xists import __version__
from xists.cli.common import (
    _failed_repo_ids_from_report,
    _failure_entry,
    _format_command_summary,
    _format_dry_run_text,
    load_repo_ids,
    write_json,
    write_json_atomic,
)
from xists.ingest.github import (
    GitHubAPIError,
    TokenPool,
    collect_record,
    collect_record_graphql,
    collect_records_graphql,
    github_token_from_env,
    github_token_from_file,
)
from xists.profile.llm import (
    PROFILE_PROMPT_VERSION,
    LLMError,
    LLMNotConfiguredError,
    attach_llm_profile,
    generate_llm_profile,
    llm_config_from_env,
)
from xists.records import record_repo_id


def _ingest_checkpoint_path(output: Path) -> Path:
    return Path(f"{output}.partial.jsonl")


def _load_ingest_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
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


def _append_ingest_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _collect_with_rate_limit(
    token_pool: TokenPool,
    operation: Any,
    *,
    max_rate_limit_wait: float,
) -> Any:
    while True:
        token = token_pool.next_token()
        if token is None:
            token_pool.wait_for_available_token(max_rate_limit_wait)
            continue
        try:
            return operation(token)
        except GitHubAPIError as error:
            if error.rate_limit_reset is None:
                raise
            token_pool.mark_rate_limited(token, error.rate_limit_reset)


def _collect_with_fallback(
    repo_id: str,
    token_pool: TokenPool,
    github_api: str,
    *,
    max_rate_limit_wait: float,
) -> dict[str, Any]:
    _collect_record_graphql = getattr(
        sys.modules.get("xists.cli"), "collect_record_graphql", collect_record_graphql
    )
    _collect_record = getattr(sys.modules.get("xists.cli"), "collect_record", collect_record)
    if github_api == "graphql":
        try:
            return _collect_with_rate_limit(
                token_pool,
                lambda token: _collect_record_graphql(repo_id, token=token),
                max_rate_limit_wait=max_rate_limit_wait,
            )
        except GitHubAPIError as graph_error:
            try:
                return _collect_with_rate_limit(
                    token_pool,
                    lambda token: _collect_record(repo_id, token=token),
                    max_rate_limit_wait=max_rate_limit_wait,
                )
            except GitHubAPIError as rest_error:
                raise GitHubAPIError(
                    f"GraphQL failed: {graph_error}; REST fallback failed: {rest_error}",
                    status=rest_error.status or graph_error.status,
                ) from rest_error

    try:
        return _collect_with_rate_limit(
            token_pool,
            lambda token: _collect_record(repo_id, token=token),
            max_rate_limit_wait=max_rate_limit_wait,
        )
    except GitHubAPIError as rest_error:
        try:
            return _collect_with_rate_limit(
                token_pool,
                lambda token: _collect_record_graphql(repo_id, token=token),
                max_rate_limit_wait=max_rate_limit_wait,
            )
        except GitHubAPIError as graph_error:
            raise GitHubAPIError(
                f"REST failed: {rest_error}; GraphQL fallback failed: {graph_error}",
                status=rest_error.status or graph_error.status,
            ) from graph_error


def _ingest_one(
    repo_id: str,
    token_pool: TokenPool,
    llm_config: Any,
    github_api: str = "rest",
    max_rate_limit_wait: float = 3600,
) -> dict[str, Any]:
    """Ingest a single repo. Returns a result dict with either 'record' or 'error'."""
    try:
        record = _collect_with_fallback(
            repo_id,
            token_pool,
            github_api,
            max_rate_limit_wait=max_rate_limit_wait,
        )
        _generate_llm_profile = getattr(
            sys.modules.get("xists.cli"), "generate_llm_profile", generate_llm_profile
        )
        profile = _generate_llm_profile(record, llm_config)
        attach_llm_profile(record, profile)
        return {"repo_id": repo_id, "record": record}
    except GitHubAPIError as error:
        return {
            "repo_id": repo_id,
            "error": _failure_entry(repo_id, str(error), status=error.status),
        }
    except LLMError as error:
        return {"repo_id": repo_id, "error": _failure_entry(repo_id, str(error), status=None)}
    except Exception as error:
        return {"repo_id": repo_id, "error": _failure_entry(repo_id, str(error), status=None)}


def _ingest_graphql_batch(
    repo_ids: list[str],
    token_pool: TokenPool,
    llm_config: Any,
    max_rate_limit_wait: float = 3600,
) -> list[dict[str, Any]]:
    """Fetch multiple repos in one GraphQL request, then generate LLM profiles per record."""
    _collect_records_graphql = getattr(
        sys.modules.get("xists.cli"), "collect_records_graphql", collect_records_graphql
    )
    _ingest_one_fn = getattr(sys.modules.get("xists.cli"), "_ingest_one", _ingest_one)
    _generate_llm_profile = getattr(
        sys.modules.get("xists.cli"), "generate_llm_profile", generate_llm_profile
    )
    try:
        records: list[dict[str, Any]] = _collect_with_rate_limit(
            token_pool,
            lambda token: _collect_records_graphql(repo_ids, token=token),
            max_rate_limit_wait=max_rate_limit_wait,
        )
    except GitHubAPIError as error:
        if len(repo_ids) == 1:
            return [
                {
                    "repo_id": repo_ids[0],
                    "error": _failure_entry(repo_ids[0], str(error), status=error.status),
                }
            ]
        return [
            _ingest_one_fn(
                repo_id,
                token_pool,
                llm_config,
                github_api="graphql",
                max_rate_limit_wait=max_rate_limit_wait,
            )
            for repo_id in repo_ids
        ]
    except Exception as error:
        if len(repo_ids) == 1:
            return [
                {
                    "repo_id": repo_ids[0],
                    "error": _failure_entry(repo_ids[0], str(error), status=None),
                }
            ]
        return [
            _ingest_one_fn(
                repo_id,
                token_pool,
                llm_config,
                github_api="graphql",
                max_rate_limit_wait=max_rate_limit_wait,
            )
            for repo_id in repo_ids
        ]

    results: list[dict[str, Any]] = []
    for repo_id, record in zip(repo_ids, records):
        try:
            profile = _generate_llm_profile(record, llm_config)
            attach_llm_profile(record, profile)
            results.append({"repo_id": repo_id, "record": record})
        except LLMError as error:
            results.append(
                {"repo_id": repo_id, "error": _failure_entry(repo_id, str(error), status=None)}
            )
        except Exception as error:
            results.append(
                {"repo_id": repo_id, "error": _failure_entry(repo_id, str(error), status=None)}
            )
    return results


def _summarize_error(result: dict[str, Any]) -> str:
    error = result.get("error") or {}
    reason = error.get("reason") or "unknown error"
    status = error.get("status")
    repo_id = error.get("repo_id") or result.get("repo_id") or "<unknown>"
    if status is not None:
        return f"{repo_id}: {reason} (status={status})"
    return f"{repo_id}: {reason}"


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _print_ingest_progress(
    *, processed: int, total: int, generated: int, failed: int, skipped: int
) -> None:
    print(
        f"ingest progress: {processed}/{total} processed "
        f"({generated} generated, {failed} failed, {skipped} skipped)",
        file=sys.stderr,
        flush=True,
    )


def ingest_github(args: argparse.Namespace) -> int:
    started_at = datetime.now(UTC)
    start_time = time.perf_counter()
    checkpoint_path = _ingest_checkpoint_path(args.output)
    resume = bool(getattr(args, "resume", False))

    retry_failed: set[str] | None = None
    if getattr(args, "retry_failed", None):
        try:
            retry_failed = _failed_repo_ids_from_report(args.retry_failed)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1

    if checkpoint_path.exists() and not resume:
        print(
            f"Checkpoint file already exists: {checkpoint_path}\n"
            "Next steps:\n"
            f"  1. Re-run with --resume to continue from the checkpoint\n"
            f"  2. Delete {checkpoint_path} if you want to restart from scratch",
            file=sys.stderr,
        )
        return 1

    repo_ids = load_repo_ids(args.repos)
    checkpoint_records = _load_ingest_checkpoint(checkpoint_path) if resume else {}
    existing: list[dict[str, Any]] = []
    if not args.force and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    existing_ids = {record.get("repo_id") for record in existing if isinstance(record, dict)}
    completed_ids = existing_ids | set(checkpoint_records)

    if getattr(args, "dry_run", False):
        if retry_failed is None:
            skipped = [repo_id for repo_id in repo_ids if repo_id in completed_ids]
            to_ingest = [repo_id for repo_id in repo_ids if repo_id not in completed_ids]
            skip_reasons: dict[str, int] = {}
            if skipped:
                existing_count = sum(repo_id in existing_ids for repo_id in skipped)
                checkpoint_count = len(skipped) - existing_count
                if existing_count:
                    skip_reasons["already_exists"] = existing_count
                if checkpoint_count:
                    skip_reasons["checkpoint_completed"] = checkpoint_count
        else:
            skipped = [repo_id for repo_id in repo_ids if repo_id not in retry_failed]
            to_ingest = [repo_id for repo_id in repo_ids if repo_id in retry_failed]
            skip_reasons = {"not_in_failure_report": len(skipped)} if skipped else {}
        github_api = getattr(args, "github_api", "rest")
        github_batch_size = getattr(args, "github_batch_size", 1) or 1
        workers = getattr(args, "workers", 1) or 1
        if github_api == "graphql" and github_batch_size > 1:
            estimated_calls = ceil(len(to_ingest) / github_batch_size)
        else:
            estimated_calls = len(to_ingest) * 3
        report = {
            "total": len(repo_ids),
            "to_process": len(to_ingest),
            "to_skip": len(skipped),
            "skip_reasons": skip_reasons,
            "estimated_calls": estimated_calls,
            "github_api": github_api,
            "github_batch_size": github_batch_size,
            "workers": workers,
        }
        if getattr(args, "format", "text") == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(_format_dry_run_text("ingest github", report))
        return 0

    try:
        _llm_config_from_env = getattr(
            sys.modules.get("xists.cli"), "llm_config_from_env", llm_config_from_env
        )
        llm_config = _llm_config_from_env()
    except LLMNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    tokens = github_token_from_file(args.token_file) if args.token_file else github_token_from_env()
    token_pool = TokenPool(
        tokens,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )

    if retry_failed is None:
        skipped = [repo_id for repo_id in repo_ids if repo_id in completed_ids]
        to_ingest = [repo_id for repo_id in repo_ids if repo_id not in completed_ids]
    else:
        skipped = [repo_id for repo_id in repo_ids if repo_id not in retry_failed]
        to_ingest = [repo_id for repo_id in repo_ids if repo_id in retry_failed]
        # Retry results replace any stale record with the same repository id.
        existing = [record for record in existing if record.get("repo_id") not in retry_failed]

    completed_records = dict(checkpoint_records)
    failed: list[dict[str, Any]] = []
    generated = 0
    with_readme = 0
    without_readme = 0
    abstained = 0
    workers = getattr(args, "workers", 1) or 1
    github_api = getattr(args, "github_api", "rest")
    github_batch_size = getattr(args, "github_batch_size", 1) or 1
    max_rate_limit_wait = getattr(args, "max_rate_limit_wait", 3600)
    if max_rate_limit_wait < 0:
        print("--max-rate-limit-wait must be zero or greater", file=sys.stderr)
        return 2
    total_to_process = len(to_ingest)

    print(
        f"ingest starting: {len(repo_ids)} input repos, {len(skipped)} skipped, "
        f"{total_to_process} to process, api={github_api}, "
        f"batch_size={github_batch_size}, workers={workers}",
        file=sys.stderr,
        flush=True,
    )

    def process_result(result: dict[str, Any]) -> None:
        nonlocal generated, with_readme, without_readme, abstained
        if "error" in result:
            failed.append(result["error"])
            print(f"ingest failed: {_summarize_error(result)}", file=sys.stderr, flush=True)
            return
        record = result["record"]
        repo_id = record_repo_id(record)
        if repo_id is None:
            raise ValueError("Ingested record is missing repo_id")
        _append_ingest_checkpoint(checkpoint_path, record)
        completed_records[repo_id] = record
        generated += 1
        if record.get("readme"):
            with_readme += 1
        else:
            without_readme += 1
        if (record.get("llm_profile") or {}).get("abstained"):
            abstained += 1

    if github_api == "graphql" and github_batch_size > 1 and to_ingest:
        batches = _chunks(to_ingest, github_batch_size)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _ingest_graphql_batch, batch, token_pool, llm_config, max_rate_limit_wait
                    ): batch
                    for batch in batches
                }
                for future in as_completed(futures):
                    for result in future.result():
                        process_result(result)
                    _print_ingest_progress(
                        processed=generated + len(failed),
                        total=total_to_process,
                        generated=generated,
                        failed=len(failed),
                        skipped=len(skipped),
                    )
        else:
            for batch in batches:
                for result in _ingest_graphql_batch(
                    batch, token_pool, llm_config, max_rate_limit_wait
                ):
                    process_result(result)
                _print_ingest_progress(
                    processed=generated + len(failed),
                    total=total_to_process,
                    generated=generated,
                    failed=len(failed),
                    skipped=len(skipped),
                )
    elif workers > 1 and to_ingest:
        # Multi-threaded: process repos concurrently, checkpoint after each future completes.
        _ingest_one_fn = getattr(sys.modules.get("xists.cli"), "_ingest_one", _ingest_one)
        with ThreadPoolExecutor(max_workers=workers) as single_executor:
            single_futures = {
                single_executor.submit(
                    _ingest_one_fn, repo_id, token_pool, llm_config, github_api, max_rate_limit_wait
                ): repo_id
                for repo_id in to_ingest
            }
            for single_future in as_completed(single_futures):
                process_result(single_future.result())
                _print_ingest_progress(
                    processed=generated + len(failed),
                    total=total_to_process,
                    generated=generated,
                    failed=len(failed),
                    skipped=len(skipped),
                )
    else:
        # Single-threaded: process one by one with checkpoint after each.
        _ingest_one_fn = getattr(sys.modules.get("xists.cli"), "_ingest_one", _ingest_one)
        for repo_id in to_ingest:
            process_result(
                _ingest_one_fn(repo_id, token_pool, llm_config, github_api, max_rate_limit_wait)
            )
            _print_ingest_progress(
                processed=generated + len(failed),
                total=total_to_process,
                generated=generated,
                failed=len(failed),
                skipped=len(skipped),
            )

    records_by_repo_id = {
        record.get("repo_id"): record
        for record in existing
        if isinstance(record, dict) and record.get("repo_id")
    }
    records_by_repo_id.update(completed_records)
    ordered_ids = list(
        dict.fromkeys(
            [
                *(
                    record.get("repo_id")
                    for record in existing
                    if isinstance(record, dict) and record.get("repo_id")
                ),
                *repo_ids,
                *completed_records,
            ]
        )
    )
    merged = [
        records_by_repo_id[repo_id] for repo_id in ordered_ids if repo_id in records_by_repo_id
    ]
    write_json_atomic(args.output, merged)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    finished_at = datetime.now(UTC)
    report = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": time.perf_counter() - start_time,
        "xists_version": __version__,
        "workers": workers,
        "token_count": len(tokens),
        "force": bool(args.force),
        "github_api": github_api,
        "github_batch_size": github_batch_size,
        "llm": {
            "provider": "openai_compatible",
            "model": llm_config.model,
            "prompt_version": PROFILE_PROMPT_VERSION,
        },
        "input_count": len(repo_ids),
        "skipped_count": len(skipped),
        "resumed_count": len(checkpoint_records),
        "generated_count": generated,
        "failed_count": len(failed),
        "failed": failed,
        "records_with_readme": with_readme,
        "records_without_readme": without_readme,
        "records_abstained": abstained,
    }

    if args.report:
        write_json(args.report, report)

    payload = {
        "records": str(args.output),
        "report": str(args.report) if args.report else None,
        "total_records": len(merged),
        **report,
    }
    if getattr(args, "format", "text") == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            _format_command_summary(
                "Ingest complete",
                [
                    ("Records", args.output),
                    ("Projects", len(merged)),
                    ("Generated", generated),
                    ("Skipped", len(skipped)),
                    ("Failed", len(failed)),
                    ("Report", args.report),
                ],
                stream=sys.stdout,
            )
        )
    if failed:
        print(
            f"ingest finished with {len(failed)} failed repos; report written to {args.report or 'stdout'}",
            file=sys.stderr,
            flush=True,
        )
    return 1 if failed and generated == 0 and total_to_process > 0 else 0


__all__ = [
    "_append_ingest_checkpoint",
    "_chunks",
    "_collect_with_fallback",
    "_collect_with_rate_limit",
    "_ingest_checkpoint_path",
    "_ingest_graphql_batch",
    "_ingest_one",
    "_load_ingest_checkpoint",
    "_print_ingest_progress",
    "_summarize_error",
    "ingest_github",
]
