"""Similar command handler and terminal formatting."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from xists.api import find_similar as public_find_similar
from xists.cli.common import (
    _format_search_number,
    _read_index_file,
    _terminal_width,
    _wrap_terminal_text,
)
from xists.cli.search import _append_search_detail, _search_confidence_text
from xists.starter import get_starter_index_path
from xists.terminal import style
from xists.types import SimilarResponse


def _format_project_badge(meta: dict[str, Any]) -> str:
    """Format short metadata badge: Language • ★ Stars • License • Type."""
    parts: list[str] = []
    lang = meta.get("language")
    if lang:
        parts.append(str(lang))
    stars = meta.get("stars")
    if stars is not None and isinstance(stars, (int, float)) and stars > 0:
        if stars >= 1000:
            parts.append(f"★ {stars / 1000:.1f}k")
        else:
            parts.append(f"★ {stars}")
    lic = meta.get("license")
    if lic:
        parts.append(str(lic))
    pt = meta.get("project_type")
    if pt:
        parts.append(str(pt).replace("_", " ").title())
    return " • ".join(parts)


def _format_similar_text(result: SimilarResponse | dict[str, Any], *, stream: Any = None) -> str:
    stream = stream or sys.stdout
    width = _terminal_width()
    target = result.get("target") or {}
    target_repo_id = str(result.get("target_repo_id") or target.get("repo_id") or "Target")
    search_results = result.get("results") or []

    lines = [style(f"Similar Projects for {target_repo_id}", "title", stream=stream)]

    # Target info
    target_summary = target.get("summary") or target.get("description")
    if target_summary:
        _append_search_detail(lines, "About", str(target_summary), width=width, stream=stream)
    target_url = target.get("url")
    if target_url:
        _append_search_detail(
            lines, "Link", str(target_url), width=width, stream=stream, role="link"
        )
    target_badge = _format_project_badge(target)
    if target_badge:
        _append_search_detail(lines, "Details", target_badge, width=width, stream=stream)

    active_filters = result.get("filters")
    if isinstance(active_filters, dict) and active_filters:
        filter_parts: list[str] = []
        for key, val in active_filters.items():
            if val is not None and val is not False:
                if isinstance(val, list):
                    filter_parts.append(f"{key}={','.join(str(x) for x in val)}")
                else:
                    filter_parts.append(f"{key}={val}")
        if filter_parts:
            _append_search_detail(
                lines, "Filters", ", ".join(filter_parts), width=width, stream=stream
            )

    if not search_results:
        lines.extend(["", style("No similar projects found", "warning", stream=stream)])
        lines.extend(
            _wrap_terminal_text(
                "The current index does not contain other repositories matching the specified criteria.",
                width=width,
            )
        )
        return "\n".join(lines)

    lines.extend(["", style(f"{len(search_results)} similar projects", "success", stream=stream)])

    for position, item in enumerate(search_results, start=1):
        if not isinstance(item, dict):
            continue
        repo_id = str(item.get("repo_id") or "<unknown>")
        url = str(item.get("url") or f"https://github.com/{repo_id}")
        summary = item.get("summary") or "No project summary is available."
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
        _append_search_detail(lines, "About", str(summary), width=width, stream=stream)
        _append_search_detail(lines, "Link", url, width=width, stream=stream, role="link")

        badge = _format_project_badge(item.get("metadata") or item)
        if badge:
            _append_search_detail(lines, "Details", badge, width=width, stream=stream)

        confidence = _search_confidence_text(item.get("confidence"))
        score = _format_search_number(item.get("score") or item.get("similarity"))
        _append_search_detail(
            lines,
            "Similarity",
            f"{confidence} (score {score})",
            width=width,
            stream=stream,
            role="success" if confidence == "high confidence" else "body",
        )

        if why_text:
            _append_search_detail(lines, "Why", why_text, width=width, stream=stream)

    return "\n".join(lines)


def similar(args: argparse.Namespace) -> int:
    demo_mode = getattr(args, "demo", False)

    filters: dict[str, Any] = {}
    if getattr(args, "language", None) is not None:
        filters["language"] = args.language
    if getattr(args, "ecosystem", None) is not None:
        filters["ecosystem"] = args.ecosystem
    if getattr(args, "project_type", None) is not None:
        filters["project_type"] = args.project_type
    if getattr(args, "min_stars", None) is not None:
        filters["min_stars"] = args.min_stars
    if getattr(args, "max_stars", None) is not None:
        filters["max_stars"] = args.max_stars
    if getattr(args, "license", None) is not None:
        filters["license"] = args.license
    if getattr(args, "topics", None) is not None:
        filters["topics"] = args.topics
    if getattr(args, "include_archived", False):
        filters["include_archived"] = True

    target_index = get_starter_index_path() if demo_mode else args.index

    if not target_index.exists():
        print(
            f"Index file not found: {target_index}. Run 'xists index build' first, or pass --demo.",
            file=sys.stderr,
        )
        return 2

    try:
        index = _read_index_file(target_index)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    _public_find_similar = getattr(
        sys.modules.get("xists.cli"), "public_find_similar", public_find_similar
    )

    try:
        result = _public_find_similar(
            args.repo_id,
            index,
            top_k=args.top_k,
            filters=filters or None,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    if getattr(args, "format", "json") == "text":
        print(_format_similar_text(result, stream=sys.stdout))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "_format_project_badge",
    "_format_similar_text",
    "similar",
]
