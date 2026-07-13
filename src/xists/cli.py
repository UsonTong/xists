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
from xists.search.index import INDEX_VERSION, entry_metadata, load_index
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


def _collect_with_fallback(repo_id: str, token_pool: TokenPool, github_api: str) -> dict[str, Any]:
    if github_api == "graphql":
        try:
            return collect_record_graphql(repo_id, token=token_pool.next_token())
        except GitHubAPIError as graph_error:
            try:
                return collect_record(repo_id, token=token_pool.next_token())
            except GitHubAPIError as rest_error:
                raise GitHubAPIError(
                    f"GraphQL failed: {graph_error}; REST fallback failed: {rest_error}",
                    status=rest_error.status or graph_error.status,
                ) from rest_error

    try:
        return collect_record(repo_id, token=token_pool.next_token())
    except GitHubAPIError as rest_error:
        try:
            return collect_record_graphql(repo_id, token=token_pool.next_token())
        except GitHubAPIError as graph_error:
            raise GitHubAPIError(
                f"REST failed: {rest_error}; GraphQL fallback failed: {graph_error}",
                status=rest_error.status or graph_error.status,
            ) from graph_error


def _ingest_one(repo_id: str, token_pool: TokenPool, llm_config: Any, github_api: str = "rest") -> dict[str, Any]:
    """Ingest a single repo. Returns a result dict with either 'record' or 'error'."""
    try:
        record = _collect_with_fallback(repo_id, token_pool, github_api)
        profile = generate_llm_profile(record, llm_config)
        attach_llm_profile(record, profile)
        return {"repo_id": repo_id, "record": record}
    except GitHubAPIError as error:
        return {"repo_id": repo_id, "error": {"repo_id": repo_id, "reason": str(error), "status": error.status}}
    except LLMError as error:
        return {"repo_id": repo_id, "error": {"repo_id": repo_id, "reason": str(error), "status": None}}
    except Exception as error:
        return {"repo_id": repo_id, "error": {"repo_id": repo_id, "reason": str(error), "status": None}}


def _ingest_graphql_batch(repo_ids: list[str], token_pool: TokenPool, llm_config: Any) -> list[dict[str, Any]]:
    """Fetch multiple repos in one GraphQL request, then generate LLM profiles per record."""
    try:
        records = collect_records_graphql(repo_ids, token=token_pool.next_token())
    except GitHubAPIError as error:
        if len(repo_ids) == 1:
            return [
                {"repo_id": repo_ids[0], "error": {"repo_id": repo_ids[0], "reason": str(error), "status": error.status}}
            ]
        results: list[dict[str, Any]] = []
        for repo_id in repo_ids:
            results.append(_ingest_one(repo_id, token_pool, llm_config, github_api="graphql"))
        return results
    except Exception as error:
        if len(repo_ids) == 1:
            return [
                {"repo_id": repo_ids[0], "error": {"repo_id": repo_ids[0], "reason": str(error), "status": None}}
            ]
        results = []
        for repo_id in repo_ids:
            results.append(_ingest_one(repo_id, token_pool, llm_config, github_api="graphql"))
        return results

    results: list[dict[str, Any]] = []
    for repo_id, record in zip(repo_ids, records):
        try:
            profile = generate_llm_profile(record, llm_config)
            attach_llm_profile(record, profile)
            results.append({"repo_id": repo_id, "record": record})
        except LLMError as error:
            results.append({"repo_id": repo_id, "error": {"repo_id": repo_id, "reason": str(error), "status": None}})
        except Exception as error:
            results.append({"repo_id": repo_id, "error": {"repo_id": repo_id, "reason": str(error), "status": None}})
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

    try:
        llm_config = llm_config_from_env()
    except LLMNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    repo_ids = load_repo_ids(args.repos)
    tokens = github_token_from_file(args.token_file) if args.token_file else github_token_from_env()
    token_pool = TokenPool(tokens)

    # Load existing records for incremental update (skip with --force).
    existing: list[dict[str, Any]] = []
    if not args.force and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    existing_ids = {record.get("repo_id") for record in existing}
    skipped = [repo_id for repo_id in repo_ids if repo_id in existing_ids]
    to_ingest = [repo_id for repo_id in repo_ids if repo_id not in existing_ids]

    merged: list[dict[str, Any]] = list(existing)
    failed: list[dict[str, Any]] = []
    generated = 0
    with_readme = 0
    without_readme = 0
    abstained = 0
    workers = getattr(args, "workers", 1) or 1
    github_api = getattr(args, "github_api", "rest")
    github_batch_size = getattr(args, "github_batch_size", 1) or 1
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
        merged.append(record)
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
                    executor.submit(_ingest_graphql_batch, batch, token_pool, llm_config): batch
                    for batch in batches
                }
                for future in as_completed(futures):
                    for result in future.result():
                        process_result(result)
                    write_json(args.output, merged)
                    _print_ingest_progress(
                        processed=generated + len(failed),
                        total=total_to_process,
                        generated=generated,
                        failed=len(failed),
                        skipped=len(skipped),
                    )
        else:
            for batch in batches:
                for result in _ingest_graphql_batch(batch, token_pool, llm_config):
                    process_result(result)
                write_json(args.output, merged)
                _print_ingest_progress(
                    processed=generated + len(failed),
                    total=total_to_process,
                    generated=generated,
                    failed=len(failed),
                    skipped=len(skipped),
                )
    elif workers > 1 and to_ingest:
        # Multi-threaded: process repos concurrently, write checkpoint after all complete.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_ingest_one, repo_id, token_pool, llm_config, github_api): repo_id
                for repo_id in to_ingest
            }
            for future in as_completed(futures):
                process_result(future.result())
                # Checkpoint after each thread completes.
                write_json(args.output, merged)
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
            process_result(_ingest_one(repo_id, token_pool, llm_config, github_api))
            write_json(args.output, merged)
            _print_ingest_progress(
                processed=generated + len(failed),
                total=total_to_process,
                generated=generated,
                failed=len(failed),
                skipped=len(skipped),
            )

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
    return 1 if failed else 0


def _index_write_checkpoint(
    output: Path,
    *,
    index_version: int,
    embedding_model: str,
    embedding_base_url: str,
    embedding_input_version: int,
    dimension: int | None,
    record_count: int,
    skipped: list[str],
    vectors: list[dict[str, Any]],
) -> None:
    write_json(
        output,
        {
            "index_version": index_version,
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
    batch_size = 64

    # Load existing index for fingerprint-aware incremental update (skip with --force).
    vectors: list[dict[str, Any]] = []
    skipped: list[str] = []
    dimension: int | None = None
    reusable_vectors: dict[str, dict[str, Any]] = {}

    if not args.force and args.output.exists():
        existing_index = json.loads(args.output.read_text(encoding="utf-8"))
        if existing_index.get("embedding_model") and existing_index["embedding_model"] != config.model:
            print(
                f"Index was built with model '{existing_index['embedding_model']}' "
                f"but configured model is '{config.model}'. "
                f"Delete {args.output} and rebuild, or set EMBEDDING_MODEL to match.",
                file=sys.stderr,
            )
            return 1
        dimension = existing_index.get("dimension")
        reusable_vectors = {entry.get("repo_id"): entry for entry in existing_index.get("vectors", []) if entry.get("repo_id")}

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
            vector = existing.get("vector") or []
            if dimension is None:
                dimension = len(vector)
            if len(vector) == dimension:
                existing = {**existing, "metadata": metadata}
                vectors.append(existing)
                continue
        embeddable.append({"repo_id": repo_id, "text": text, "fingerprint": fingerprint, "metadata": metadata})

    new_count = 0
    for start in range(0, len(embeddable), batch_size):
        batch = embeddable[start : start + batch_size]
        try:
            results = call_embeddings(config, [item["text"] for item in batch])
        except EmbeddingError as error:
            _print_embedding_error(error, command="index build")
            return 1
        if len(results) != len(batch):
            print(
                f"Embedding count mismatch: sent {len(batch)}, received {len(results)}",
                file=sys.stderr,
            )
            return 1
        for item, vector in zip(batch, results):
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
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
                    "vector": vector,
                }
            )
            new_count += 1

        # Checkpoint: write after each batch.
        _index_write_checkpoint(
            args.output,
            index_version=INDEX_VERSION,
            embedding_model=config.model,
            embedding_base_url=config.base_url,
            embedding_input_version=EMBEDDING_INPUT_VERSION,
            dimension=dimension,
            record_count=len(vectors),
            skipped=skipped,
            vectors=vectors,
        )

    if not embeddable:
        _index_write_checkpoint(
            args.output,
            index_version=INDEX_VERSION,
            embedding_model=config.model,
            embedding_base_url=config.base_url,
            embedding_input_version=EMBEDDING_INPUT_VERSION,
            dimension=dimension,
            record_count=len(vectors),
            skipped=skipped,
            vectors=vectors,
        )

    print(
        json.dumps(
            {
                "index": str(args.output),
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


def _format_search_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value) if value is not None else "n/a"


def _format_search_text(result: dict[str, Any], index: dict[str, Any]) -> str:
    summaries = _index_summaries_by_repo_id(index)
    intent = result.get("query_intent") or {}
    intent_type = intent.get("type") if isinstance(intent, dict) else None

    lines = [
        f"query: {result.get('query') or ''}",
        f"intent: {intent_type or 'unknown'}",
        f"abstained: {bool(result.get('abstained'))}",
    ]
    if result.get("abstained") and result.get("abstain_reason"):
        lines.append(f"abstain_reason: {result['abstain_reason']}")

    search_results = result.get("results") or []
    if not search_results:
        lines.append("results: none")
        return "\n".join(lines)

    lines.append("results:")
    for position, item in enumerate(search_results, start=1):
        if not isinstance(item, dict):
            continue
        repo_id = str(item.get("repo_id") or "<unknown>")
        why = item.get("why") or []
        if isinstance(why, list):
            why_text = "; ".join(str(reason) for reason in why if str(reason).strip())
        else:
            why_text = str(why)

        lines.extend(
            [
                f"{position}. repo: {repo_id}",
                f"   confidence: {item.get('confidence') or 'unknown'}",
                f"   score: {_format_search_number(item.get('score'))}",
                f"   summary: {summaries.get(repo_id, '(none)')}",
                f"   why: {why_text or '(none)'}",
            ]
        )

        matched_terms = item.get("matched_terms") or []
        if matched_terms:
            lines.append(f"   matched_terms: {', '.join(str(term) for term in matched_terms)}")

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


def index_stats(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index file not found: {args.index}. Run 'xists index build' first.", file=sys.stderr)
        return 2

    index = load_index(args.index)
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

    payload = {
        "index": str(args.index),
        "index_version": index.get("index_version"),
        "embedding_model": index.get("embedding_model"),
        "embedding_base_url": index.get("embedding_base_url"),
        "embedding_input_version": index.get("embedding_input_version"),
        "dimension": index.get("dimension"),
        "built_at": index.get("built_at"),
        "record_count": index.get("record_count"),
        "vector_count": len(vectors),
        "skipped_count": len(index.get("skipped") or []),
        "missing_metadata_count": missing_metadata,
        "missing_fingerprint_count": missing_fingerprints,
        "top_languages": [{"language": key, "count": value} for key, value in languages.most_common(args.limit)],
        "top_topics": [{"topic": key, "count": value} for key, value in topics.most_common(args.limit)],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
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
                "repo_id": record.get("repo_id") or record.get("repo_id_requested"),
                "name": record.get("name"),
                "url": record.get("url"),
                "language": github.get("language"),
                "topics": github.get("topics") or [],
                "has_readme": bool(record.get("readme")),
                "profile_confidence": profile.get("confidence"),
                "profile_abstained": bool(profile.get("abstained")),
                "summary": profile.get("summary"),
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
    github.set_defaults(func=ingest_github)

    index = subparsers.add_parser("index", help="Build the embedding index")
    index_subparsers = index.add_subparsers(dest="index_command", required=True)
    index_build_parser = index_subparsers.add_parser("build", help="Build an embedding index from records")
    index_build_parser.add_argument("--records", type=Path, default=Path("records.json"), help="Records JSON to index")
    index_build_parser.add_argument("--output", type=Path, default=Path("index.json"), help="Path to write the embedding index")
    index_build_parser.add_argument("--force", action="store_true", help="Ignore existing index.json and rebuild from scratch")
    index_build_parser.set_defaults(func=index_build)
    index_stats_parser = index_subparsers.add_parser("stats", help="Summarize an embedding index without printing vectors")
    index_stats_parser.add_argument("--index", type=Path, default=Path("index.json"), help="Embedding index to inspect")
    index_stats_parser.add_argument("--limit", type=int, default=10, help="Maximum languages/topics to print")
    index_stats_parser.set_defaults(func=index_stats)

    records = subparsers.add_parser("records", help="Inspect generated repository records")
    records_subparsers = records.add_subparsers(dest="records_command", required=True)
    records_inspect_parser = records_subparsers.add_parser("inspect", help="Print a compact summary of records")
    records_inspect_parser.add_argument("--records", type=Path, default=Path("records.json"), help="Records JSON to inspect")
    records_inspect_parser.add_argument("--repo", default=None, help="Only show records whose owner/repo contains this text")
    records_inspect_parser.add_argument("--limit", type=int, default=20, help="Maximum records to print")
    records_inspect_parser.set_defaults(func=records_inspect)

    search_parser = subparsers.add_parser("search", help="Search the embedding index")
    search_parser.add_argument("query", help="Natural-language query")
    search_parser.add_argument("--index", type=Path, default=Path("index.json"), help="Embedding index to search")
    search_parser.add_argument("--top-k", type=int, default=10, help="Maximum number of results to return")
    search_parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format: json (default) or text",
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
