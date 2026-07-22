"""Command-line interface for xists."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

from xists import __version__
from xists.eval.inspect import inspect_report, load_report
from xists.eval.run import evaluate_dataset
from xists.eval.schema import EvaluationDatasetError, load_dataset
from xists.ingest.github import (
    GitHubAPIError,
    TokenPool,
    collect_record,
    collect_record_graphql,
    collect_records_graphql,
    github_token_from_env,
    github_token_from_file,
    parse_github_repo,
)
from xists.profile.llm import (
    LLMError,
    LLMNotConfiguredError,
    PROFILE_PROMPT_VERSION,
    attach_llm_profile,
    generate_llm_profile,
    llm_config_from_env,
)
from xists.records import (
    RECORD_SCHEMA_VERSION,
    profile_refresh_reason,
    record_profile,
    record_repo_id,
    records_validation_report,
)
from xists.search.embed import (
    EMBEDDING_INPUT_VERSION,
    EmbeddingError,
    EmbeddingNotConfiguredError,
    call_embeddings,
    embedding_config_from_env,
    embedding_input_fingerprint,
    embedding_text_from_record,
    probe_embedding_endpoint,
)
from xists.search.index import INDEX_VERSION, decode_vector, encode_vector, entry_metadata, load_index
from xists.search.query import IndexMismatchError, _query_intent, rank


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or "=" not in value:
            continue
        key, env_value = value.split("=", 1)
        key = key.strip()
        env_value = env_value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = env_value


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


def _ingest_checkpoint_path(output: Path) -> Path:
    return Path(f"{output}.partial.jsonl")


def _load_ingest_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    return _load_profile_refresh_checkpoint(path)


def _append_ingest_checkpoint(path: Path, record: dict[str, Any]) -> None:
    _append_profile_refresh_checkpoint(path, record)


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
    repo_ids = {
        item.get("repo_id")
        for item in failed
        if isinstance(item, dict) and isinstance(item.get("repo_id"), str) and item.get("repo_id").strip()
    }
    return {repo_id for repo_id in repo_ids if repo_id}


def _failure_entry(repo_id: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "reason": reason,
        "error": reason,
        "attempted_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


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
        "attempted_at": datetime.now(timezone.utc).isoformat(),
        "output_count": len(output_records),
    }


def _collect_with_rate_limit(
    token_pool: TokenPool,
    operation: Any,
    *,
    max_rate_limit_wait: float,
) -> dict[str, Any]:
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
    if github_api == "graphql":
        try:
            return _collect_with_rate_limit(
                token_pool,
                lambda token: collect_record_graphql(repo_id, token=token),
                max_rate_limit_wait=max_rate_limit_wait,
            )
        except GitHubAPIError as graph_error:
            try:
                return _collect_with_rate_limit(
                    token_pool,
                    lambda token: collect_record(repo_id, token=token),
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
            lambda token: collect_record(repo_id, token=token),
            max_rate_limit_wait=max_rate_limit_wait,
        )
    except GitHubAPIError as rest_error:
        try:
            return _collect_with_rate_limit(
                token_pool,
                lambda token: collect_record_graphql(repo_id, token=token),
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
        profile = generate_llm_profile(record, llm_config)
        attach_llm_profile(record, profile)
        return {"repo_id": repo_id, "record": record}
    except GitHubAPIError as error:
        return {"repo_id": repo_id, "error": _failure_entry(repo_id, str(error), status=error.status)}
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
    try:
        records = _collect_with_rate_limit(
            token_pool,
            lambda token: collect_records_graphql(repo_ids, token=token),
            max_rate_limit_wait=max_rate_limit_wait,
        )
    except GitHubAPIError as error:
        if len(repo_ids) == 1:
            return [
                {"repo_id": repo_ids[0], "error": _failure_entry(repo_ids[0], str(error), status=error.status)}
            ]
        results: list[dict[str, Any]] = []
        for repo_id in repo_ids:
            results.append(_ingest_one(repo_id, token_pool, llm_config, github_api="graphql", max_rate_limit_wait=max_rate_limit_wait))
        return results
    except Exception as error:
        if len(repo_ids) == 1:
            return [
                {"repo_id": repo_ids[0], "error": _failure_entry(repo_ids[0], str(error), status=None)}
            ]
        results = []
        for repo_id in repo_ids:
            results.append(_ingest_one(repo_id, token_pool, llm_config, github_api="graphql", max_rate_limit_wait=max_rate_limit_wait))
        return results

    results: list[dict[str, Any]] = []
    for repo_id, record in zip(repo_ids, records):
        try:
            profile = generate_llm_profile(record, llm_config)
            attach_llm_profile(record, profile)
            results.append({"repo_id": repo_id, "record": record})
        except LLMError as error:
            results.append({"repo_id": repo_id, "error": _failure_entry(repo_id, str(error), status=None)})
        except Exception as error:
            results.append({"repo_id": repo_id, "error": _failure_entry(repo_id, str(error), status=None)})
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
    return [items[i : i + size] for i in range(0, len(items), size)]


def _print_ingest_progress(*, processed: int, total: int, generated: int, failed: int, skipped: int) -> None:
    print(
        f"ingest progress: {processed}/{total} processed "
        f"({generated} generated, {failed} failed, {skipped} skipped)",
        file=sys.stderr,
        flush=True,
    )


def ingest_github(args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc)
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
        llm_config = llm_config_from_env()
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
                    executor.submit(_ingest_graphql_batch, batch, token_pool, llm_config, max_rate_limit_wait): batch
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
                for result in _ingest_graphql_batch(batch, token_pool, llm_config, max_rate_limit_wait):
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
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_ingest_one, repo_id, token_pool, llm_config, github_api, max_rate_limit_wait): repo_id
                for repo_id in to_ingest
            }
            for future in as_completed(futures):
                process_result(future.result())
                _print_ingest_progress(
                    processed=generated + len(failed),
                    total=total_to_process,
                    generated=generated,
                    failed=len(failed),
                    skipped=len(skipped),
                )
    else:
        # Single-threaded: process one by one with checkpoint after each.
        for repo_id in to_ingest:
            process_result(_ingest_one(repo_id, token_pool, llm_config, github_api, max_rate_limit_wait))
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
    ordered_ids = list(dict.fromkeys([
        *(record.get("repo_id") for record in existing if isinstance(record, dict) and record.get("repo_id")),
        *repo_ids,
        *completed_records,
    ]))
    merged = [records_by_repo_id[repo_id] for repo_id in ordered_ids if repo_id in records_by_repo_id]
    write_json_atomic(args.output, merged)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    finished_at = datetime.now(timezone.utc)
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

    print(json.dumps(
        {"records": str(args.output), "report": str(args.report) if args.report else None, "total_records": len(merged), **report},
        ensure_ascii=False,
        indent=2,
    ))
    if failed:
        print(
            f"ingest finished with {len(failed)} failed repos; report written to {args.report or 'stdout'}",
            file=sys.stderr,
            flush=True,
        )
    return 1 if failed and generated == 0 and total_to_process > 0 else 0


def _index_write_checkpoint(
    output: Path,
    *,
    index_version: int,
    record_schema_version: int,
    embedding_model: str,
    embedding_base_url: str,
    embedding_input_version: int,
    dimension: int | None,
    record_count: int,
    skipped: list[str],
    vectors: list[dict[str, Any]],
) -> None:
    # Index files can be large enough that a reader may otherwise observe a
    # partially truncated JSON document while a checkpoint is being rewritten.
    # Replacing a completed sibling file keeps every visible checkpoint valid.
    write_json_atomic(
        output,
        {
            "index_version": index_version,
            "record_schema_version": record_schema_version,
            "embedding_model": embedding_model,
            "embedding_base_url": embedding_base_url,
            "embedding_input_version": embedding_input_version,
            "dimension": dimension,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "record_count": record_count,
            "skipped": skipped,
            "vectors": vectors,
        },
    )


def _index_checkpoint_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.partial.json")


def index_build(args: argparse.Namespace) -> int:
    try:
        config = embedding_config_from_env()
    except EmbeddingNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    if not args.records.exists():
        print(f"Records file not found: {args.records}", file=sys.stderr)
        return 2

    records = json.loads(args.records.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print(f"Records file must contain a JSON list: {args.records}", file=sys.stderr)
        return 1
    validation = records_validation_report(records, expected_profile_prompt_version=PROFILE_PROMPT_VERSION)
    if not validation["ok"]:
        print(
            f"Records schema/quality validation failed for {args.records}.\n"
            f"Expected record schema_version {RECORD_SCHEMA_VERSION}; "
            f"errors: {validation['errors']}.\n"
            "Next steps:\n"
            f"  1. Refresh profiles: xists profile refresh --records {args.records} --output records-v2.json\n"
            "  2. Rebuild index: xists index build --records records-v2.json --output index.json",
            file=sys.stderr,
        )
        return 1
    batch_size = 64

    checkpoint_path = _index_checkpoint_path(args.output)
    if args.resume and not checkpoint_path.exists():
        print(f"Checkpoint file not found: {checkpoint_path}", file=sys.stderr)
        return 1
    if checkpoint_path.exists() and not args.resume:
        print(
            f"Checkpoint file already exists: {checkpoint_path}\n"
            "Next steps:\n"
            f"  1. Re-run with --resume to continue from the checkpoint\n"
            f"  2. Delete {checkpoint_path} if you want to restart from scratch",
            file=sys.stderr,
        )
        return 1

    # Load existing index for fingerprint-aware incremental update (skip with --force).
    vectors: list[dict[str, Any]] = []
    skipped: list[str] = []
    dimension: int | None = None
    reusable_vectors: dict[str, dict[str, Any]] = {}

    source_index_path: Path | None = None
    if args.resume:
        source_index_path = checkpoint_path
    elif not args.force and args.output.exists():
        source_index_path = args.output
    if source_index_path is not None:
        existing_index = json.loads(source_index_path.read_text(encoding="utf-8"))
        if existing_index.get("embedding_model") and existing_index["embedding_model"] != config.model:
            print(
                f"Index was built with model '{existing_index['embedding_model']}' "
                f"but configured model is '{config.model}'. "
                f"Delete {args.output} and rebuild, or set EMBEDDING_MODEL to match.",
                file=sys.stderr,
            )
            return 1
        reusable = (
            existing_index.get("embedding_input_version") == EMBEDDING_INPUT_VERSION
            and existing_index.get("record_schema_version") == RECORD_SCHEMA_VERSION
        )
        if reusable:
            dimension = existing_index.get("dimension")
            reusable_vectors = {
                entry.get("repo_id"): entry
                for entry in existing_index.get("vectors", [])
                if entry.get("repo_id")
            }

    # Prepare embeddable records and reuse unchanged vectors.
    embeddable: list[dict[str, Any]] = []
    for record in records:
        text = embedding_text_from_record(record)
        repo_id = record.get("repo_id") or record.get("repo_id_requested")
        if not text:
            skipped.append(repo_id or "<unknown>")
            continue
        fingerprint = embedding_input_fingerprint(record)
        metadata = entry_metadata(record)
        existing = reusable_vectors.get(repo_id)
        if existing and existing.get("embedding_input_fingerprint") == fingerprint:
            vector = decode_vector(existing.get("vector"), dimension=dimension)
            if vector is None:
                embeddable.append({"repo_id": repo_id, "text": text, "fingerprint": fingerprint, "metadata": metadata})
                continue
            if dimension is None:
                dimension = int(vector.size)
            if vector.size == dimension:
                existing = {**existing, "metadata": metadata}
                vectors.append(existing)
                continue
        embeddable.append({"repo_id": repo_id, "text": text, "fingerprint": fingerprint, "metadata": metadata})

    def write_partial_checkpoint() -> None:
        _index_write_checkpoint(
            checkpoint_path,
            index_version=INDEX_VERSION,
            record_schema_version=RECORD_SCHEMA_VERSION,
            embedding_model=config.model,
            embedding_base_url=config.base_url,
            embedding_input_version=EMBEDDING_INPUT_VERSION,
            dimension=dimension,
            record_count=len(vectors),
            skipped=skipped,
            vectors=vectors,
        )

    new_count = 0
    checkpoint_every_batches = 16
    for batch_number, start in enumerate(range(0, len(embeddable), batch_size), start=1):
        batch = embeddable[start : start + batch_size]
        try:
            results = call_embeddings(config, [item["text"] for item in batch])
        except EmbeddingError as error:
            write_partial_checkpoint()
            _print_embedding_error(error, command="index build")
            return 1
        if len(results) != len(batch):
            write_partial_checkpoint()
            print(
                f"Embedding count mismatch: sent {len(batch)}, received {len(results)}",
                file=sys.stderr,
            )
            return 1
        for item, vector in zip(batch, results):
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                write_partial_checkpoint()
                print(
                    f"Inconsistent embedding dimension: {len(vector)} vs {dimension}",
                    file=sys.stderr,
                )
                return 1
            vectors.append(
                {
                    "repo_id": item["repo_id"],
                    "embedding_input_fingerprint": item["fingerprint"],
                    "metadata": item["metadata"],
                    "vector": encode_vector(vector),
                }
            )
            new_count += 1

        if batch_number % checkpoint_every_batches == 0:
            write_partial_checkpoint()

    _index_write_checkpoint(
        args.output,
        index_version=INDEX_VERSION,
        record_schema_version=RECORD_SCHEMA_VERSION,
        embedding_model=config.model,
        embedding_base_url=config.base_url,
        embedding_input_version=EMBEDDING_INPUT_VERSION,
        dimension=dimension,
        record_count=len(vectors),
        skipped=skipped,
        vectors=vectors,
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(
        json.dumps(
            {
                "index": str(args.output),
                "record_schema_version": RECORD_SCHEMA_VERSION,
                "embedding_model": config.model,
                "dimension": dimension,
                "record_count": len(vectors),
                "new_vectors": new_count,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def search(args: argparse.Namespace) -> int:
    try:
        config = embedding_config_from_env()
    except EmbeddingNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    if not args.index.exists():
        print(f"Index file not found: {args.index}. Run 'xists index build' first.", file=sys.stderr)
        return 2

    index = load_index(args.index)
    try:
        result = rank(args.query, index, config, top_k=args.top_k)
    except IndexMismatchError as error:
        print(str(error), file=sys.stderr)
        return 1
    except EmbeddingError as error:
        _print_embedding_error(error, command="search")
        return 1

    if getattr(args, "format", "json") == "text":
        print(_format_search_text(result, index))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _index_summaries_by_repo_id(index: dict[str, Any]) -> dict[str, str]:
    summaries: dict[str, str] = {}
    for item in index.get("vectors") or []:
        if not isinstance(item, dict):
            continue
        repo_id = item.get("repo_id")
        metadata = item.get("metadata")
        if not isinstance(repo_id, str) or not isinstance(metadata, dict):
            continue
        summary = metadata.get("summary") or metadata.get("description") or ""
        if isinstance(summary, str) and summary.strip():
            summaries[repo_id] = summary.strip()
    return summaries


def _index_metadata_by_repo_id(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata_by_repo: dict[str, dict[str, Any]] = {}
    for item in index.get("vectors") or []:
        if not isinstance(item, dict):
            continue
        repo_id = item.get("repo_id")
        metadata = item.get("metadata")
        if isinstance(repo_id, str) and isinstance(metadata, dict):
            metadata_by_repo[repo_id] = metadata
    return metadata_by_repo


def _format_search_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value) if value is not None else "n/a"


def _format_search_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(str(value) for value in values if str(value).strip())


def _format_search_text(result: dict[str, Any], index: dict[str, Any]) -> str:
    summaries = _index_summaries_by_repo_id(index)
    metadata_by_repo_id = _index_metadata_by_repo_id(index)
    intent = result.get("query_intent") or {}
    intent_type = intent.get("type") if isinstance(intent, dict) else None
    search_results = result.get("results") or []

    lines = [
        f"query: {result.get('query') or ''}",
        f"intent: {intent_type or 'unknown'}",
        f"abstained: {bool(result.get('abstained'))}",
        f"results: {len(search_results)}",
    ]
    if result.get("abstained") and result.get("abstain_reason"):
        lines.append(f"abstain_reason: {result['abstain_reason']}")

    if not search_results:
        return "\n".join(lines)

    for position, item in enumerate(search_results, start=1):
        if not isinstance(item, dict):
            continue
        repo_id = str(item.get("repo_id") or "<unknown>")
        metadata = metadata_by_repo_id.get(repo_id, {})
        url = item.get("url") or metadata.get("url") or "n/a"
        why = item.get("why") or []
        if isinstance(why, list):
            why_text = "; ".join(str(reason) for reason in why if str(reason).strip())
        else:
            why_text = str(why)

        lines.extend(
            [
                f"{position}. repo: {repo_id}",
                f"   url: {url}",
                f"   confidence: {item.get('confidence') or 'unknown'}",
                f"   score: {_format_search_number(item.get('score'))}",
                f"   summary: {summaries.get(repo_id, '(none)')}",
                f"   why: {why_text or '(none)'}",
            ]
        )

        matched_terms = item.get("matched_terms") or []
        if matched_terms:
            lines.append(f"   matched_terms: {', '.join(str(term) for term in matched_terms)}")

        diagnostics = item.get("diagnostics") or {}
        if isinstance(diagnostics, dict) and diagnostics:
            evidence_parts = []
            for label, key in (
                ("topics", "topic_matches"),
                ("capabilities", "capability_terms"),
                ("types", "type_cue_matches"),
                ("profile", "profile_matches"),
                ("state", "repository_state"),
            ):
                value = _format_search_list(diagnostics.get(key))
                if value:
                    evidence_parts.append(f"{label}={value}")
            for label, key in (
                ("entity", "entity_match"),
                ("identity", "identity_match"),
                ("language", "language_match"),
                ("language_mismatch", "language_mismatch"),
                ("phrase", "phrase_match"),
            ):
                value = diagnostics.get(key)
                if value:
                    evidence_parts.append(f"{label}={value}")
            if evidence_parts:
                lines.append(f"   diagnostics: {'; '.join(evidence_parts)}")

        breakdown = item.get("score_breakdown") or {}
        if isinstance(breakdown, dict) and breakdown:
            semantic = _format_search_number(breakdown.get("semantic"))
            metadata = _format_search_number(breakdown.get("metadata"))
            final = _format_search_number(breakdown.get("final"))
            lines.append(f"   score_breakdown: semantic={semantic}, metadata={metadata}, final={final}")

    return "\n".join(lines)


def _check_payload(name: str, status: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, **extra}


def _embedding_config_next_steps() -> list[str]:
    return [
        "Copy .env.example to .env if you have not configured the project yet.",
        "Set EMBEDDING_API_KEY, EMBEDDING_BASE_URL, and EMBEDDING_MODEL.",
        "Run xists doctor --check-endpoints after setting the embedding variables.",
    ]


def _llm_config_next_steps() -> list[str]:
    return [
        "Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL in .env or the environment.",
        "LLM configuration is required for xists ingest github and optional eval --llm-judge runs.",
    ]


def _github_token_next_steps() -> list[str]:
    return [
        "Set GITHUB_TOKEN or GITHUB_TOKENS in .env, or pass --token-file.",
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


def version(args: argparse.Namespace) -> int:
    print(json.dumps({"version": __version__}, ensure_ascii=False, indent=2))
    return 0


def doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []
    embedding_config = None

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

    check_endpoints = bool(getattr(args, "check_endpoints", False) or getattr(args, "strict", False))
    strict = bool(getattr(args, "strict", False))
    if check_endpoints and embedding_config is not None:
        try:
            probe = probe_embedding_endpoint(embedding_config)
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
        config = llm_config_from_env()
        checks.append(_check_payload("llm_config", "ok", "LLM endpoint is configured", model=config.model, base_url=config.base_url))
    except LLMNotConfiguredError as error:
        checks.append(_check_payload("llm_config", "error", str(error), next_steps=_llm_config_next_steps()))

    try:
        tokens = github_token_from_file(args.token_file) if args.token_file else github_token_from_env()
        if tokens:
            checks.append(_check_payload("github_token", "ok", "GitHub token is configured", token_count=len(tokens)))
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
        checks.append(_check_payload("github_token", "error", str(error), next_steps=_github_token_next_steps()))

    for name, path in (
        ("records_file", args.records),
        ("index_file", args.index),
        ("eval_cases_file", args.cases),
    ):
        if path.exists():
            checks.append(_check_payload(name, "ok", f"{path} exists", path=str(path)))
        else:
            command_hint = {
                "records_file": "Run xists ingest github to create records.json, or pass --records to point at an existing records file.",
                "index_file": "Run xists index build to create index.json, or pass --index to point at an existing index file.",
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
    print(json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def _index_stats_report(index: dict[str, Any], *, index_path: Path, limit: int) -> dict[str, Any]:
    vectors = index.get("vectors") or []
    languages: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    missing_metadata = 0
    missing_fingerprints = 0
    for entry in vectors:
        if not isinstance(entry, dict):
            continue
        if not entry.get("embedding_input_fingerprint"):
            missing_fingerprints += 1
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            missing_metadata += 1
            continue
        language = metadata.get("language")
        if isinstance(language, str) and language.strip():
            languages[language] += 1
        for topic in metadata.get("topics") or []:
            if isinstance(topic, str) and topic.strip():
                topics[topic] += 1

    dimension = index.get("dimension")
    estimated_memory_mb: float | None = None
    if isinstance(dimension, int) and dimension > 0:
        # Search materializes vectors as a float32 matrix (4 bytes per value).
        estimated_memory_mb = round(len(vectors) * dimension * 4 / 1024 / 1024, 1)

    payload = {
        "index": str(index_path),
        "index_version": index.get("index_version"),
        "record_schema_version": index.get("record_schema_version"),
        "embedding_model": index.get("embedding_model"),
        "embedding_base_url": index.get("embedding_base_url"),
        "embedding_input_version": index.get("embedding_input_version"),
        "dimension": index.get("dimension"),
        "built_at": index.get("built_at"),
        "record_count": index.get("record_count"),
        "vector_count": len(vectors),
        "estimated_memory_mb": estimated_memory_mb,
        "skipped_count": len(index.get("skipped") or []),
        "missing_metadata_count": missing_metadata,
        "missing_fingerprint_count": missing_fingerprints,
        "top_languages": _counter_items(languages, "language", limit),
        "top_topics": _counter_items(topics, "topic", limit),
    }
    return payload


def _format_index_stats_text(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"index: {report['index']}",
            f"index_version: {report.get('index_version')}",
            f"record_schema_version: {report.get('record_schema_version')}",
            f"embedding_model: {report.get('embedding_model')}",
            f"embedding_base_url: {report.get('embedding_base_url')}",
            f"embedding_input_version: {report.get('embedding_input_version')}",
            f"dimension: {report.get('dimension')}",
            f"built_at: {report.get('built_at')}",
            f"record_count: {report.get('record_count')}",
            f"vector_count: {report.get('vector_count')}",
            "estimated memory: "
            + (
                f"{report['estimated_memory_mb']} MB"
                if report.get("estimated_memory_mb") is not None
                else "unknown"
            ),
            f"skipped_count: {report.get('skipped_count')}",
            f"missing_metadata_count: {report.get('missing_metadata_count')}",
            f"missing_fingerprint_count: {report.get('missing_fingerprint_count')}",
            "top:",
            f"  languages: {_format_top_items(report.get('top_languages') or [], 'language')}",
            f"  topics: {_format_top_items(report.get('top_topics') or [], 'topic')}",
        ]
    )


def index_stats(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index file not found: {args.index}. Run 'xists index build' first.", file=sys.stderr)
        return 2

    index = load_index(args.index)
    payload = _index_stats_report(index, index_path=args.index, limit=args.limit)
    if getattr(args, "format", "text") == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_format_index_stats_text(payload))
    return 0


def records_inspect(args: argparse.Namespace) -> int:
    if not args.records.exists():
        print(f"Records file not found: {args.records}. Run 'xists ingest github' first.", file=sys.stderr)
        return 2

    records = json.loads(args.records.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print(f"Records file must contain a JSON list: {args.records}", file=sys.stderr)
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
        github = record.get("github") or {}
        profile = record.get("llm_profile") or {}
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


def _counter_items(counter: Counter[str], key_name: str, limit: int) -> list[dict[str, Any]]:
    return [{key_name: key, "count": value} for key, value in counter.most_common(limit)]


def _records_stats_report(records: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    validation = records_validation_report(records, expected_profile_prompt_version=PROFILE_PROMPT_VERSION)
    languages: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    project_types: Counter[str] = Counter()
    ecosystems: Counter[str] = Counter()
    confidence: Counter[str] = Counter()

    for record in records:
        github = record.get("github") if isinstance(record.get("github"), dict) else {}
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


def _safe_divide(numerator: Any, denominator: int) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator or 0) / denominator, 6)


def _format_top_items(items: list[dict[str, Any]], key_name: str) -> str:
    if not items:
        return "none"
    return ", ".join(f"{item[key_name]} ({item['count']})" for item in items)


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
    for key in ("missing_search_text", "missing_aliases", "search_text_too_short", "profile_abstained", "low_confidence", "archived", "disabled", "missing_readme", "duplicates"):
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
        print(f"Records file not found: {args.records}. Run 'xists ingest github' first.", file=sys.stderr)
        return 2
    records = json.loads(args.records.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print(f"Records file must contain a JSON list: {args.records}", file=sys.stderr)
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
    if any(key in errors for key in ("schema_version_mismatch", "missing_llm_profile", "missing_summary", "missing_search_text")):
        steps.append(f"Refresh profiles: xists profile refresh --records {records_path} --output records-v2.json")
    if errors.get("duplicate_repo_id"):
        steps.append("Review duplicate repo_id entries and keep one canonical record per repository.")
    if warnings.get("search_text_too_short") or warnings.get("missing_aliases"):
        steps.append("Review weak profiles or refresh them with xists profile refresh.")
    if warnings.get("profile_abstained") or warnings.get("low_confidence_profile"):
        steps.append("Inspect low-confidence or abstained profiles before sharing this records file.")
    if errors:
        steps.append("Rebuild the index after records are fixed: xists index build --records records-v2.json --output index.json")
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
        print(f"Records file not found: {args.records}. Run 'xists ingest github' first.", file=sys.stderr)
        return 2
    records = json.loads(args.records.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print(f"Records file must contain a JSON list: {args.records}", file=sys.stderr)
        return 1

    report = records_validation_report(records, expected_profile_prompt_version=PROFILE_PROMPT_VERSION)
    report["records"] = str(args.records)
    report["next_steps"] = [] if report["ok"] else _records_next_steps(args.records, report)
    if getattr(args, "format", "text") == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format_records_validation_text(report, args.records))
    return 0 if report["ok"] else 1


def profile_refresh(args: argparse.Namespace) -> int:
    if not args.records.exists():
        print(f"Records file not found: {args.records}. Run 'xists ingest github' first.", file=sys.stderr)
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
            reason = "force" if args.force else profile_refresh_reason(
                record,
                only_missing_search_text=bool(args.only_missing_search_text),
                expected_prompt_version=PROFILE_PROMPT_VERSION,
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
        config = llm_config_from_env()
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

    resumed_records = _load_profile_refresh_checkpoint(checkpoint_path) if getattr(args, "resume", False) else {}

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
        reason = "force" if args.force else profile_refresh_reason(
            updated,
            only_missing_search_text=bool(args.only_missing_search_text),
            expected_prompt_version=PROFILE_PROMPT_VERSION,
        )
        if retry_failed is not None:
            reason = "retry_failed" if repo_id in retry_failed else None
        return updated, reason

    try:
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
                        profile = generate_llm_profile(updated, config)
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
                    executor.submit(generate_llm_profile, updated, config): (updated, reason)
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
                output_by_repo_id[record_repo_id(record) or "<unknown>"]
                for record in records
            ]

        write_json_atomic(args.output, output_records)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    except Exception as error:
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
                            record_repo_id(records[processed]) if processed < len(records) else "<unknown>",
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
            "\n".join(
                [
                    f"records: {args.records}",
                    f"output: {args.output}",
                    f"schema: {RECORD_SCHEMA_VERSION}",
                    f"profile_prompt_version: {PROFILE_PROMPT_VERSION}",
                    f"refreshed: {refreshed}",
                    f"resumed: {resumed}",
                    f"skipped: {skipped}",
                    f"failed: {len(failed)}",
                ]
            )
        )
    if failed:
        print(
            f"profile refresh finished with {len(failed)} failed records; report written to {args.report or 'stdout'}",
            file=sys.stderr,
            flush=True,
        )
    return 1 if failed and refreshed == 0 and total > 0 else 0


def _index_verify_report(records: list[dict[str, Any]], index: dict[str, Any]) -> dict[str, Any]:
    record_validation = records_validation_report(records, expected_profile_prompt_version=PROFILE_PROMPT_VERSION)
    errors: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    if not record_validation["ok"]:
        errors["records_validation_failed"] = sum(record_validation["errors"].values())
    if index.get("index_version") != INDEX_VERSION:
        errors["index_version_mismatch"] += 1
    if index.get("record_schema_version") != RECORD_SCHEMA_VERSION:
        errors["record_schema_version_mismatch"] += 1
    if index.get("embedding_input_version") != EMBEDDING_INPUT_VERSION:
        errors["embedding_input_version_mismatch"] += 1

    vectors = [entry for entry in index.get("vectors") or [] if isinstance(entry, dict)]
    vector_by_id = {entry.get("repo_id"): entry for entry in vectors if entry.get("repo_id")}
    dimension = index.get("dimension")
    if not isinstance(dimension, int) or dimension <= 0:
        errors["dimension_missing"] += 1
    missing_fingerprints = [entry.get("repo_id") for entry in vectors if not entry.get("embedding_input_fingerprint")]
    if missing_fingerprints:
        errors["missing_fingerprints"] = len(missing_fingerprints)
    dimension_mismatches = [
        entry.get("repo_id")
        for entry in vectors
        if isinstance(dimension, int)
        and (decoded := decode_vector(entry.get("vector"))) is not None
        and decoded.size != dimension
    ]
    invalid_vectors = [entry.get("repo_id") for entry in vectors if decode_vector(entry.get("vector")) is None]
    if dimension_mismatches:
        errors["dimension_mismatch"] = len(dimension_mismatches)
    if invalid_vectors:
        errors["invalid_vectors"] = len(invalid_vectors)
    if isinstance(index.get("record_count"), int) and index.get("record_count") != len(vectors):
        warnings["record_count_mismatch"] += 1

    record_ids = {record_repo_id(record) for record in records if record_repo_id(record)}
    missing_vectors: list[str] = []
    stale_vectors: list[str] = []
    skipped_expected: list[str] = []
    for record in records:
        repo_id = record_repo_id(record)
        if not repo_id:
            continue
        fingerprint = embedding_input_fingerprint(record)
        if fingerprint is None or not embedding_text_from_record(record):
            skipped_expected.append(repo_id)
            continue
        entry = vector_by_id.get(repo_id)
        if entry is None:
            missing_vectors.append(repo_id)
        elif entry.get("embedding_input_fingerprint") != fingerprint:
            stale_vectors.append(repo_id)
    extra_vectors = sorted(str(repo_id) for repo_id in vector_by_id if repo_id not in record_ids)
    if missing_vectors:
        errors["missing_vectors"] = len(missing_vectors)
    if stale_vectors:
        errors["stale_vectors"] = len(stale_vectors)
    if extra_vectors:
        warnings["extra_vectors"] = len(extra_vectors)

    ok = not errors
    stale_only = set(errors).issubset({"missing_vectors", "stale_vectors"}) and (missing_vectors or stale_vectors or extra_vectors)
    return {
        "ok": ok,
        "status": "ok" if ok else ("stale" if stale_only else "invalid"),
        "index_version": index.get("index_version"),
        "expected_index_version": INDEX_VERSION,
        "record_schema_version": index.get("record_schema_version"),
        "expected_record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_input_version": index.get("embedding_input_version"),
        "expected_embedding_input_version": EMBEDDING_INPUT_VERSION,
        "record_count": len(records),
        "vector_count": len(vectors),
        "errors": dict(errors),
        "warnings": dict(warnings),
        "missing_fingerprints": missing_fingerprints,
        "dimension_mismatches": dimension_mismatches,
        "invalid_vectors": invalid_vectors,
        "missing_vectors": missing_vectors,
        "stale_vectors": stale_vectors,
        "extra_vectors": extra_vectors,
        "skipped_expected": skipped_expected,
        "records_validation": record_validation,
        "next_steps": [] if ok else [
            "Refresh profiles if records are old: xists profile refresh --records records.json --output records-v2.json",
            "Rebuild the index: xists index build --records records-v2.json --output index.json",
        ],
    }


def _format_index_verify_text(report: dict[str, Any], records_path: Path, index_path: Path) -> str:
    lines = [
        f"records: {records_path}",
        f"index: {index_path}",
        f"status: {report['status']}",
        f"ok: {str(report['ok']).lower()}",
        f"records: {report['record_count']}",
        f"vectors: {report['vector_count']}",
    ]
    for label in ("errors", "warnings"):
        lines.append(f"{label}:")
        items = report.get(label) or {}
        if items:
            for key, value in sorted(items.items()):
                lines.append(f"  {key}: {value}")
        else:
            lines.append("  none")
    if report.get("next_steps"):
        lines.append("next steps:")
        for step in report["next_steps"]:
            lines.append(f"  - {step}")
    return "\n".join(lines)


def index_verify(args: argparse.Namespace) -> int:
    if not args.records.exists():
        print(f"Records file not found: {args.records}. Run 'xists ingest github' first.", file=sys.stderr)
        return 2
    if not args.index.exists():
        print(f"Index file not found: {args.index}. Run 'xists index build' first.", file=sys.stderr)
        return 2
    records = json.loads(args.records.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print(f"Records file must contain a JSON list: {args.records}", file=sys.stderr)
        return 1
    index = load_index(args.index)
    report = _index_verify_report(records, index)
    report["records"] = str(args.records)
    report["index"] = str(args.index)
    if getattr(args, "format", "text") == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format_index_verify_text(report, args.records, args.index))
    return 0 if report["ok"] else 1


def eval_run(args: argparse.Namespace) -> int:
    try:
        config = embedding_config_from_env()
    except EmbeddingNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    llm_judge_config = None
    if args.llm_judge:
        try:
            llm_judge_config = llm_config_from_env()
        except LLMNotConfiguredError as error:
            print(str(error), file=sys.stderr)
            return 2
        if args.records is None:
            print("--records is required when --llm-judge is enabled", file=sys.stderr)
            return 2

    if not args.index.exists():
        print(f"Index file not found: {args.index}. Run 'xists index build' first.", file=sys.stderr)
        return 2

    try:
        report = evaluate_dataset(
            args.cases,
            args.index,
            config,
            top_k=args.top_k,
            batch_size=args.batch_size,
            llm_judge_config=llm_judge_config,
            records_path=args.records,
        )
    except EmbeddingError as error:
        _print_embedding_error(error, command="eval run")
        return 1
    except (EvaluationDatasetError, FileNotFoundError, IndexMismatchError, ValueError, LLMError) as error:
        print(str(error), file=sys.stderr)
        return 1

    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0



def eval_inspect(args: argparse.Namespace) -> int:
    try:
        report = load_report(args.report)
        payload = inspect_report(
            report,
            status=args.status,
            limit=args.limit,
            include_exact=args.include_exact,
            tag=args.tag,
            intent=args.query_intent,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def eval_cases(args: argparse.Namespace) -> int:
    try:
        dataset = load_dataset(args.cases)
    except EvaluationDatasetError as error:
        print(str(error), file=sys.stderr)
        return 1

    selected: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        if args.tag and args.tag not in case.get("tags", []):
            continue
        intent = _query_intent(case["query"]).get("type")
        if args.query_intent and intent != args.query_intent:
            continue
        selected.append(
            {
                "id": case["id"],
                "query": case["query"],
                "query_intent": intent,
                "expected_repo_id": case["expected_repo_id"],
                "acceptable": case.get("acceptable") or [],
                "acceptable_repo_ids": case.get("acceptable_repo_ids") or [],
                "acceptable_families": case.get("acceptable_families") or [],
                "tags": case.get("tags") or [],
                "notes": case.get("notes"),
            }
        )

    tag_counts = Counter(tag for case in dataset["cases"] for tag in case.get("tags", []))
    intent_counts = Counter(_query_intent(case["query"]).get("type") for case in dataset["cases"])
    payload = {
        "dataset_name": dataset.get("dataset_name"),
        "schema_version": dataset.get("schema_version"),
        "case_count": len(dataset["cases"]),
        "family_count": len(dataset.get("families") or {}),
        "tag_counts": [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(args.limit)],
        "query_intent_counts": [
            {"query_intent": intent, "count": count} for intent, count in intent_counts.most_common()
        ],
        "filter": {"tag": args.tag, "query_intent": args.query_intent, "limit": args.limit},
        "matching_count": len(selected),
        "inspected_count": min(len(selected), args.limit),
        "cases": selected[: args.limit],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="xists helps developers find what already exists.")
    parser.add_argument("--version", action="version", version=f"xists {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print the xists version as JSON")
    version_parser.set_defaults(func=version)

    doctor_parser = subparsers.add_parser("doctor", help="Check local configuration and expected data files")
    doctor_parser.add_argument("--records", type=Path, default=Path("records.json"), help="Records JSON to check")
    doctor_parser.add_argument("--index", type=Path, default=Path("index.json"), help="Embedding index to check")
    doctor_parser.add_argument("--cases", type=Path, default=Path("eval-cases.json"), help="Evaluation cases JSON to check")
    doctor_parser.add_argument("--token-file", type=Path, default=None, help="Optional file containing GitHub tokens")
    doctor_parser.add_argument(
        "--check-endpoints",
        action="store_true",
        help="Probe the configured embedding endpoint with a small real request",
    )
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when endpoint probes fail; implies --check-endpoints",
    )
    doctor_parser.set_defaults(func=doctor)

    ingest = subparsers.add_parser("ingest", help="Collect repository records")
    ingest_subparsers = ingest.add_subparsers(dest="source", required=True)

    github = ingest_subparsers.add_parser("github", help="Collect records from GitHub repositories")
    github.add_argument("--repos", type=Path, default=Path("repos.txt"), help="Text file with one GitHub owner/repo or URL per line")
    github.add_argument("--output", type=Path, default=Path("records.json"), help="Path to write records JSON")
    github.add_argument("--report", type=Path, default=Path("report.json"), help="Path to write generation report JSON")
    github.add_argument("--token-file", type=Path, default=None, help="Optional file containing a GitHub token")
    github.add_argument("--force", action="store_true", help="Ignore existing records.json and reprocess all repos")
    github.add_argument("--resume", action="store_true", help="Resume from an existing partial JSONL checkpoint")
    github.add_argument("--dry-run", action="store_true", help="Estimate ingest work without calling GitHub or writing files")
    github.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (default: 1)")
    github.add_argument(
        "--github-api",
        choices=("rest", "graphql"),
        default="rest",
        help="GitHub API backend: rest (default) or graphql (lower quota usage)",
    )
    github.add_argument(
        "--github-batch-size",
        type=int,
        default=1,
        help="Repos per GraphQL request when --github-api=graphql (default: 1, recommended: 25-50)",
    )
    github.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for dry-run reports: text (default) or json for scripts and agents",
    )
    github.add_argument("--retry-failed", type=Path, default=None, help="Only process repos listed in a failure report JSON")
    github.add_argument(
        "--max-rate-limit-wait",
        type=float,
        default=3600,
        help="Maximum seconds to wait for exhausted GitHub rate limits (default: 3600)",
    )
    github.set_defaults(func=ingest_github)

    index = subparsers.add_parser("index", help="Build the embedding index")
    index_subparsers = index.add_subparsers(dest="index_command", required=True)
    index_build_parser = index_subparsers.add_parser("build", help="Build an embedding index from records")
    index_build_parser.add_argument("--records", type=Path, default=Path("records.json"), help="Records JSON to index")
    index_build_parser.add_argument("--output", type=Path, default=Path("index.json"), help="Path to write the embedding index")
    index_build_parser.add_argument("--force", action="store_true", help="Ignore existing index.json and rebuild from scratch")
    index_build_parser.add_argument("--resume", action="store_true", help="Resume from an existing partial index checkpoint")
    index_build_parser.set_defaults(func=index_build)
    index_stats_parser = index_subparsers.add_parser("stats", help="Summarize an embedding index without printing vectors")
    index_stats_parser.add_argument("--index", type=Path, default=Path("index.json"), help="Embedding index to inspect")
    index_stats_parser.add_argument("--limit", type=int, default=10, help="Maximum languages/topics to print")
    index_stats_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    index_stats_parser.set_defaults(func=index_stats)
    index_verify_parser = index_subparsers.add_parser("verify", help="Check that records and index are in sync")
    index_verify_parser.add_argument("--records", type=Path, default=Path("records.json"), help="Records JSON to compare")
    index_verify_parser.add_argument("--index", type=Path, default=Path("index.json"), help="Embedding index to verify")
    index_verify_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    index_verify_parser.set_defaults(func=index_verify)

    records = subparsers.add_parser("records", help="Inspect generated repository records")
    records_subparsers = records.add_subparsers(dest="records_command", required=True)
    records_inspect_parser = records_subparsers.add_parser("inspect", help="Print a compact summary of records")
    records_inspect_parser.add_argument("--records", type=Path, default=Path("records.json"), help="Records JSON to inspect")
    records_inspect_parser.add_argument("--repo", default=None, help="Only show records whose owner/repo contains this text")
    records_inspect_parser.add_argument("--limit", type=int, default=20, help="Maximum records to print")
    records_inspect_parser.set_defaults(func=records_inspect)
    records_stats_parser = records_subparsers.add_parser("stats", help="Summarize records quality and metadata")
    records_stats_parser.add_argument("--records", type=Path, default=Path("records.json"), help="Records JSON to summarize")
    records_stats_parser.add_argument("--limit", type=int, default=10, help="Maximum languages/topics/project types to print")
    records_stats_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    records_stats_parser.set_defaults(func=records_stats)
    records_validate_parser = records_subparsers.add_parser("validate", help="Validate record schema and profile quality")
    records_validate_parser.add_argument("--records", type=Path, default=Path("records.json"), help="Records JSON to validate")
    records_validate_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    records_validate_parser.set_defaults(func=records_validate)

    profile = subparsers.add_parser("profile", help="Refresh LLM profiles for records")
    profile_subparsers = profile.add_subparsers(dest="profile_command", required=True)
    profile_refresh_parser = profile_subparsers.add_parser("refresh", help="Regenerate LLM profiles and schema v2 fields")
    profile_refresh_parser.add_argument("--records", type=Path, default=Path("records.json"), help="Records JSON to refresh")
    profile_refresh_parser.add_argument("--output", type=Path, default=Path("records-v2.json"), help="Path to write refreshed records JSON")
    profile_refresh_parser.add_argument("--force", action="store_true", help="Refresh every record instead of only outdated ones")
    profile_refresh_parser.add_argument("--workers", type=int, default=1, help="Concurrent LLM refresh workers (default: 1)")
    profile_refresh_parser.add_argument("--resume", action="store_true", help="Resume from an existing partial JSONL checkpoint")
    profile_refresh_parser.add_argument("--dry-run", action="store_true", help="Estimate refresh work without calling the LLM or writing files")
    profile_refresh_parser.add_argument("--report", type=Path, default=None, help="Path to write a refresh failure report JSON")
    profile_refresh_parser.add_argument("--retry-failed", type=Path, default=None, help="Only process repos listed in a failure report JSON")
    profile_refresh_parser.add_argument(
        "--only-missing-search-text",
        action="store_true",
        help="Only refresh records whose profile is missing search_text",
    )
    profile_refresh_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    profile_refresh_parser.set_defaults(func=profile_refresh)

    search_parser = subparsers.add_parser("search", help="Search the embedding index")
    search_parser.add_argument("query", help="Natural-language query")
    search_parser.add_argument("--index", type=Path, default=Path("index.json"), help="Embedding index to search")
    search_parser.add_argument("--top-k", type=int, default=10, help="Maximum number of results to return")
    search_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    search_parser.set_defaults(func=search)

    eval_parser = subparsers.add_parser("eval", help="Evaluate retrieval quality")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run_parser = eval_subparsers.add_parser("run", help="Run retrieval evaluation against an index")
    eval_run_parser.add_argument("--cases", type=Path, default=Path("eval-cases.json"), help="Evaluation dataset JSON")
    eval_run_parser.add_argument("--index", type=Path, default=Path("index.json"), help="Embedding index to evaluate")
    eval_run_parser.add_argument("--output", type=Path, default=Path("eval-report.json"), help="Path to write evaluation report JSON")
    eval_run_parser.add_argument("--top-k", type=int, default=10, help="Maximum results to score per query")
    eval_run_parser.add_argument("--batch-size", type=int, default=64, help="Number of queries to embed per batch")
    eval_run_parser.add_argument("--records", type=Path, default=None, help="Records JSON used for optional LLM judge comparisons")
    eval_run_parser.add_argument("--llm-judge", action="store_true", help="Run an LLM pairwise judge on top-1 mismatches")
    eval_run_parser.set_defaults(func=eval_run)

    eval_inspect_parser = eval_subparsers.add_parser("inspect", help="Inspect misses and summary from an evaluation report")
    eval_inspect_parser.add_argument("--report", type=Path, default=Path("eval-report.json"), help="Evaluation report JSON to inspect")
    eval_inspect_parser.add_argument("--status", choices=("exact", "acceptable", "serious_mismatch", "insufficient_evidence"), default=None, help="Only show cases with this top-1 status")
    eval_inspect_parser.add_argument("--limit", type=int, default=20, help="Maximum cases to print")
    eval_inspect_parser.add_argument("--include-exact", action="store_true", help="Include exact top-1 cases in the inspection output")
    eval_inspect_parser.add_argument("--tag", default=None, help="Only show cases carrying this tag")
    eval_inspect_parser.add_argument(
        "--query-intent",
        dest="query_intent",
        choices=("empty", "exact_name", "alternative", "domain", "functional"),
        default=None,
        help="Only show cases with this query intent",
    )
    eval_inspect_parser.set_defaults(func=eval_inspect)

    eval_cases_parser = eval_subparsers.add_parser("cases", help="Validate and summarize an evaluation dataset")
    eval_cases_parser.add_argument("--cases", type=Path, default=Path("eval-cases.json"), help="Evaluation dataset JSON")
    eval_cases_parser.add_argument("--tag", default=None, help="Only show cases carrying this tag")
    eval_cases_parser.add_argument(
        "--query-intent",
        dest="query_intent",
        choices=("empty", "exact_name", "alternative", "domain", "functional"),
        default=None,
        help="Only show cases with this query intent",
    )
    eval_cases_parser.add_argument("--limit", type=int, default=20, help="Maximum cases to print")
    eval_cases_parser.set_defaults(func=eval_cases)

    return parser


def main() -> int:
    load_env_file(Path(".env"))
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
