#!/usr/bin/env python3
"""Check xists evaluation reports against retrieval-quality thresholds.

This script is intentionally small and dependency-free so it can run in local
pre-commit hooks or CI jobs after ``xists eval run`` produces a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_MIN_EXACT_TOP1 = 0.88
DEFAULT_MIN_EFFECTIVE_TOP1 = 1.0
DEFAULT_MAX_SERIOUS_MISMATCH = 0.0


class ReportCheckError(ValueError):
    """Raised when an evaluation report cannot be checked."""


def _load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReportCheckError(f"report not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ReportCheckError(f"report is not valid JSON: {error}") from error

    if not isinstance(payload, dict):
        raise ReportCheckError("report must be a JSON object")
    return payload


def _metric(metrics: dict[str, Any], names: tuple[str, ...]) -> tuple[str, float]:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return name, float(value)
    raise ReportCheckError(f"missing numeric metric: {' or '.join(names)}")


def check_report(
    report: dict[str, Any],
    *,
    min_exact_top1: float,
    min_effective_top1: float,
    max_serious_mismatch: float,
) -> dict[str, Any]:
    """Return a structured pass/fail summary for an eval report."""

    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ReportCheckError("report.metrics must be a JSON object")

    exact_name, exact_top1 = _metric(metrics, ("exact_top1_rate", "exact_hit_at_1"))
    effective_name, effective_top1 = _metric(metrics, ("effective_top1_rate", "acceptable_hit_at_1"))
    serious_name, serious_mismatch = _metric(metrics, ("serious_top1_error_rate",))

    checks = [
        {
            "name": "exact_top1",
            "metric": exact_name,
            "actual": exact_top1,
            "operator": ">=",
            "threshold": min_exact_top1,
            "ok": exact_top1 >= min_exact_top1,
        },
        {
            "name": "effective_top1",
            "metric": effective_name,
            "actual": effective_top1,
            "operator": ">=",
            "threshold": min_effective_top1,
            "ok": effective_top1 >= min_effective_top1,
        },
        {
            "name": "serious_mismatch",
            "metric": serious_name,
            "actual": serious_mismatch,
            "operator": "<=",
            "threshold": max_serious_mismatch,
            "ok": serious_mismatch <= max_serious_mismatch,
        },
    ]
    ok = all(check["ok"] for check in checks)
    return {
        "ok": ok,
        "dataset_name": report.get("dataset_name"),
        "case_count": report.get("case_count"),
        "checks": checks,
    }


def _format_check_line(check: dict[str, Any]) -> str:
    status = "PASS" if check["ok"] else "FAIL"
    return (
        f"{status} {check['name']}: {check['metric']}={check['actual']:.6f} "
        f"{check['operator']} {check['threshold']:.6f}"
    )


def render_text(summary: dict[str, Any], *, report_path: Path) -> str:
    """Render a stable, readable check summary."""

    status = "PASS" if summary["ok"] else "FAIL"
    header = (
        f"{status} {report_path}"
        f" dataset={summary.get('dataset_name') or '<unknown>'}"
        f" cases={summary.get('case_count') if summary.get('case_count') is not None else '<unknown>'}"
    )
    lines = [header]
    lines.extend(_format_check_line(check) for check in summary["checks"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check xists eval report quality thresholds.")
    parser.add_argument("report", type=Path, help="Evaluation report JSON produced by `xists eval run`")
    parser.add_argument(
        "--min-exact-top1",
        type=float,
        default=DEFAULT_MIN_EXACT_TOP1,
        help=f"Minimum exact top-1 rate, using exact_top1_rate or exact_hit_at_1 (default: {DEFAULT_MIN_EXACT_TOP1})",
    )
    parser.add_argument(
        "--min-effective-top1",
        type=float,
        default=DEFAULT_MIN_EFFECTIVE_TOP1,
        help=(
            "Minimum effective top-1 rate, using effective_top1_rate or acceptable_hit_at_1 "
            f"(default: {DEFAULT_MIN_EFFECTIVE_TOP1})"
        ),
    )
    parser.add_argument(
        "--max-serious-mismatch",
        type=float,
        default=DEFAULT_MAX_SERIOUS_MISMATCH,
        help=f"Maximum serious top-1 mismatch rate (default: {DEFAULT_MAX_SERIOUS_MISMATCH})",
    )
    parser.add_argument("--json", action="store_true", help="Print the check summary as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = _load_report(args.report)
        summary = check_report(
            report,
            min_exact_top1=args.min_exact_top1,
            min_effective_top1=args.min_effective_top1,
            max_serious_mismatch=args.max_serious_mismatch,
        )
    except ReportCheckError as error:
        print(f"FAIL {args.report}: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"report": str(args.report), **summary}, ensure_ascii=False, indent=2))
    else:
        print(render_text(summary, report_path=args.report))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
