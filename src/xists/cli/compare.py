"""Compare command handler and terminal formatting."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from xists.api import compare_projects as public_compare_projects
from xists.cli.common import (
    _read_index_file,
    _terminal_width,
    _wrap_terminal_text,
)
from xists.cli.search import _append_search_detail
from xists.cli.similar import _format_project_badge
from xists.starter import get_starter_index_path
from xists.terminal import style
from xists.types import CompareResponse


def _format_compare_text(result: CompareResponse | dict[str, Any], *, stream: Any = None) -> str:
    stream = stream or sys.stdout
    width = _terminal_width()
    projects = result.get("projects") or []
    matrix = result.get("matrix") or {}
    analysis = result.get("analysis") or {}

    names = " vs ".join(str(p.get("repo_id") or "") for p in projects)
    lines = [style(f"Project Comparison: {names}", "title", stream=stream)]

    # 1. Project Snapshots
    lines.extend(["", style("1. Project Overviews", "title", stream=stream)])
    for idx, p in enumerate(projects, start=1):
        repo_id = str(p.get("repo_id") or "")
        summary = p.get("summary") or p.get("description") or "No summary available."
        url = str(p.get("url") or f"https://github.com/{repo_id}")

        lines.append("")
        lines.extend(
            style(line, "title", stream=stream)
            for line in _wrap_terminal_text(f"{idx}. {repo_id}", width=width)
        )
        _append_search_detail(lines, "About", str(summary), width=width, stream=stream)
        _append_search_detail(lines, "Link", url, width=width, stream=stream, role="link")

        badge = _format_project_badge(p)
        if badge:
            _append_search_detail(lines, "Details", badge, width=width, stream=stream)

        caps = p.get("capabilities") or []
        if caps:
            _append_search_detail(
                lines,
                "Capabilities",
                "; ".join(str(c) for c in caps[:4]),
                width=width,
                stream=stream,
            )

        use_cases = p.get("use_cases") or []
        if use_cases:
            _append_search_detail(
                lines,
                "Best for",
                "; ".join(str(u) for u in use_cases[:3]),
                width=width,
                stream=stream,
            )

        not_for = p.get("not_for") or []
        if not_for:
            _append_search_detail(
                lines,
                "Not for",
                "; ".join(str(n) for n in not_for[:3]),
                width=width,
                stream=stream,
                role="warning",
            )

    # 2. Pairwise Cosine Similarity Matrix
    if matrix and len(projects) >= 2:
        lines.extend(["", style("2. Pairwise Embedding Similarity Matrix", "title", stream=stream)])
        # Build simple tabular layout
        repo_ids = [p["repo_id"] for p in projects if p.get("repo_id")]
        short_names = [r.split("/")[-1] for r in repo_ids]
        col_width = max(max((len(s) for s in short_names), default=12), 12) + 2
        row_header_width = max((len(s) for s in short_names), default=16) + 2

        header_row = " " * row_header_width + "".join(f"{s:>{col_width}}" for s in short_names)
        lines.append(header_row)
        lines.append("-" * len(header_row))

        for r_id, s_name in zip(repo_ids, short_names):
            row_vals = []
            for j_id in repo_ids:
                sim = matrix.get(r_id, {}).get(j_id, 0.0)
                if r_id == j_id:
                    row_vals.append(f"{'1.000':>{col_width}}")
                else:
                    row_vals.append(f"{sim:.3f}".rjust(col_width))
            lines.append(f"{s_name:<{row_header_width}}" + "".join(row_vals))

    # 3. Shared Ground (Commonalities)
    lines.extend(["", style("3. Shared Ground", "title", stream=stream)])
    shared_ecosystems = analysis.get("shared_ecosystems") or []
    if shared_ecosystems:
        _append_search_detail(
            lines, "Shared Ecosystems", ", ".join(shared_ecosystems), width=width, stream=stream
        )

    shared_languages = analysis.get("shared_languages") or []
    if shared_languages:
        _append_search_detail(
            lines, "Shared Language", ", ".join(shared_languages), width=width, stream=stream
        )

    shared_caps = analysis.get("shared_capabilities") or []
    if shared_caps:
        _append_search_detail(
            lines, "Shared Capabilities", "; ".join(shared_caps), width=width, stream=stream
        )

    shared_topics = analysis.get("shared_topics") or []
    if shared_topics:
        _append_search_detail(
            lines, "Shared Topics", ", ".join(shared_topics), width=width, stream=stream
        )

    if not shared_ecosystems and not shared_languages and not shared_caps and not shared_topics:
        lines.append("   (No direct overlap in language, ecosystem, or topics)")

    # 4. Project Differentiators
    lines.extend(["", style("4. Distinctive Strengths & Boundaries", "title", stream=stream)])
    differentiators = analysis.get("differentiators") or {}
    for r_id, diff in differentiators.items():
        if not isinstance(diff, dict):
            continue
        lines.append(f"   {style(r_id, 'title', stream=stream)}:")
        unique_caps = diff.get("unique_capabilities") or []
        if unique_caps:
            _append_search_detail(
                lines,
                "  Unique Strengths",
                "; ".join(str(c) for c in unique_caps[:3]),
                width=width,
                stream=stream,
            )
        unique_use_cases = diff.get("unique_use_cases") or []
        if unique_use_cases:
            _append_search_detail(
                lines,
                "  Unique Use Cases",
                "; ".join(str(u) for u in unique_use_cases[:3]),
                width=width,
                stream=stream,
            )
        not_for = diff.get("not_for") or []
        if not_for:
            _append_search_detail(
                lines,
                "  Key Boundaries",
                "; ".join(str(n) for n in not_for[:2]),
                width=width,
                stream=stream,
                role="warning",
            )

    # 5. Direct Relationships
    direct_links = analysis.get("direct_links") or []
    if direct_links:
        lines.extend(["", style("5. Direct Knowledge Links", "title", stream=stream)])
        for link in direct_links:
            if isinstance(link, dict):
                src = link.get("source")
                tgt = link.get("target")
                rel = link.get("relation")
                lines.append(f"   • {src} → {tgt} ({rel})")

    return "\n".join(lines)


def compare(args: argparse.Namespace) -> int:
    demo_mode = getattr(args, "demo", False)
    repo_ids = getattr(args, "repo_ids", [])

    if not repo_ids or len(repo_ids) < 2:
        print("Please provide between 2 and 5 repositories to compare.", file=sys.stderr)
        return 1

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

    _public_compare_projects = getattr(
        sys.modules.get("xists.cli"), "public_compare_projects", public_compare_projects
    )

    try:
        result = _public_compare_projects(
            repo_ids,
            index,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    if getattr(args, "format", "json") == "text":
        print(_format_compare_text(result, stream=sys.stdout))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "_format_compare_text",
    "compare",
]
