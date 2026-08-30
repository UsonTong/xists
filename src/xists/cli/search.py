"""Search command handler and terminal formatting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from xists.api import search as public_search
from xists.cli.common import (
    _format_search_number,
    _prepare_query_transforms,
    _print_embedding_error,
    _read_index_file,
    _terminal_width,
    _wrap_terminal_text,
)
from xists.search.embed import (
    EmbeddingError,
    EmbeddingNotConfiguredError,
    embedding_config_from_env,
)
from xists.search.query import IndexMismatchError
from xists.search.rerank import (
    RerankerError,
    RerankerNotConfiguredError,
    rerank_documents,
    reranker_config_from_env,
)
from xists.search.transform import (
    QueryTransformError,
    QueryTransformNotConfiguredError,
)
from xists.starter import get_starter_index_path, starter_metadata_search
from xists.terminal import TerminalRole, style


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


def _append_search_detail(
    lines: list[str],
    label: str,
    value: str,
    *,
    width: int,
    stream: Any,
    role: TerminalRole = "body",
) -> None:
    prefix = f"   {label}: "
    wrapped = _wrap_terminal_text(value, width=width, indent=len(prefix))
    first_line = wrapped[0].lstrip()
    lines.append(
        f"   {style(label, 'muted', stream=stream)}: {style(first_line, role, stream=stream)}"
    )
    lines.extend(style(line, role, stream=stream) for line in wrapped[1:])


def _search_confidence_text(value: Any) -> str:
    return str(value or "unknown").replace("_", " ")


def _format_search_text(
    result: dict[str, Any], index: dict[str, Any], *, stream: Any = None
) -> str:
    stream = stream or sys.stdout
    summaries = _index_summaries_by_repo_id(index)
    metadata_by_repo_id = _index_metadata_by_repo_id(index)
    search_results = result.get("results") or []
    width = _terminal_width()

    lines = [style("Search", "title", stream=stream)]
    _append_search_detail(
        lines, "Query", str(result.get("query") or ""), width=width, stream=stream
    )

    if result.get("abstained") or not search_results:
        title = "No confident match" if result.get("abstained") else "No matching projects"
        message = (
            "The current index does not contain a sufficiently reliable match for this query."
            if result.get("abstained")
            else "The current index did not return a project for this query."
        )
        lines.extend(["", style(title, "warning", stream=stream)])
        lines.extend(_wrap_terminal_text(message, width=width))
        lines.extend(
            _wrap_terminal_text(
                "Try a broader description or search an index with more relevant projects.",
                width=width,
            )
        )
        return "\n".join(lines)

    lines.extend(["", style(f"{len(search_results)} matches", "success", stream=stream)])
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

        lines.append("")
        lines.extend(
            style(line, "title", stream=stream)
            for line in _wrap_terminal_text(f"{position}. {repo_id}", width=width)
        )
        summary = (
            item.get("summary")
            or summaries.get(repo_id)
            or metadata.get("description")
            or "No project summary is available."
        )
        _append_search_detail(lines, "About", str(summary), width=width, stream=stream)
        _append_search_detail(lines, "Link", str(url), width=width, stream=stream, role="link")
        confidence = _search_confidence_text(item.get("confidence"))
        score = _format_search_number(item.get("score"))
        _append_search_detail(
            lines,
            "Match",
            f"{confidence} (score {score})",
            width=width,
            stream=stream,
            role="success" if confidence == "high confidence" else "body",
        )
        if why_text:
            _append_search_detail(lines, "Why", why_text, width=width, stream=stream)

    return "\n".join(lines)


def search(args: argparse.Namespace) -> int:
    demo_mode = getattr(args, "demo", False)
    offline_mode = getattr(args, "offline", False)

    if offline_mode:
        records = None
        records_arg = getattr(args, "records", None)
        if not demo_mode and records_arg and Path(records_arg).is_file():
            try:
                records = json.loads(Path(records_arg).read_text(encoding="utf-8"))
            except Exception:
                pass
        result = starter_metadata_search(args.query, records=records, top_k=args.top_k)
        if getattr(args, "format", "json") == "text":
            print(_format_search_text(result, {}, stream=sys.stdout))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    try:
        config = embedding_config_from_env()
    except EmbeddingNotConfiguredError as error:
        if demo_mode:
            result = starter_metadata_search(args.query, top_k=args.top_k)
            if getattr(args, "format", "json") == "text":
                print(_format_search_text(result, {}, stream=sys.stdout))
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        print(str(error), file=sys.stderr)
        return 2

    target_index = get_starter_index_path() if demo_mode else args.index

    if not target_index.exists():
        print(
            f"Index file not found: {target_index}. Run 'xists index build' first.", file=sys.stderr
        )
        return 2

    try:
        index = _read_index_file(target_index)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    rerank = None
    if args.ranking_strategy == "rerank":
        try:
            reranker_config = reranker_config_from_env()
        except RerankerNotConfiguredError as error:
            print(str(error), file=sys.stderr)
            return 2

        def rerank(query: str, documents: list[str]) -> list[float]:
            return rerank_documents(reranker_config, query, documents)

    try:
        variants, rerank_queries, _ = _prepare_query_transforms(
            [args.query], args.query_transform_mode
        )
        rank_kwargs: dict[str, Any] = {}
        if variants is not None and rerank_queries is not None:
            rank_kwargs = {"query_variants": variants[0], "rerank_query": rerank_queries[0]}
        _public_search = getattr(sys.modules.get("xists.cli"), "public_search", public_search)
        result = _public_search(
            args.query,
            index,
            embedding_config=config,
            top_k=args.top_k,
            ranking_strategy=args.ranking_strategy,
            rerank=rerank,
            rerank_candidate_limit=args.rerank_candidates,
            exploratory_threshold=args.exploratory_threshold,
            rerank_abstain_threshold=args.rerank_abstain_threshold,
            confidence_calibration=args.confidence_calibration,
            **rank_kwargs,
        )
    except IndexMismatchError as error:
        print(str(error), file=sys.stderr)
        return 1
    except EmbeddingError as error:
        _print_embedding_error(error, command="search")
        return 1
    except (
        QueryTransformError,
        QueryTransformNotConfiguredError,
        RerankerError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1

    if getattr(args, "format", "json") == "text":
        print(_format_search_text(result, index, stream=sys.stdout))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "_append_search_detail",
    "_format_search_text",
    "_index_metadata_by_repo_id",
    "_index_summaries_by_repo_id",
    "_search_confidence_text",
    "search",
]
