"""Summarize an xists evaluation report across its dataset tags."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    normal = [result for result in results if "category-no-result" not in result.get("tags", [])]
    no_result = [result for result in results if "category-no-result" in result.get("tags", [])]
    high_confidence_wrong = [
        result
        for result in normal
        if result.get("top_result_confidence") == "high_confidence" and not result.get("acceptable_match")
    ]
    return {
        "case_count": len(results),
        "normal_case_count": len(normal),
        "no_result_case_count": len(no_result),
        "recall_at_1": _rate(sum(bool(result.get("acceptable_match")) for result in normal), len(normal)),
        "recall_at_5": _rate(
            sum((result.get("acceptable_rank") or 999) <= 5 for result in normal), len(normal)
        ),
        "wrong_high_confidence_rate": _rate(len(high_confidence_wrong), len(normal)),
        "no_result_abstain_rate": _rate(sum(bool(result.get("abstained")) for result in no_result), len(no_result)),
        "normal_false_abstain_rate": _rate(sum(bool(result.get("abstained")) for result in normal), len(normal)),
    }


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("report must contain a results list")
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        if not isinstance(result, dict):
            continue
        for tag in result.get("tags", []):
            if not isinstance(tag, str) or "-" not in tag:
                continue
            dimension, value = tag.split("-", 1)
            if dimension in {"domain", "category", "language"}:
                groups[dimension][value].append(result)
    return {
        "dataset_name": report.get("dataset_name"),
        "overall": _metrics([result for result in results if isinstance(result, dict)]),
        "slices": {
            dimension: {value: _metrics(items) for value, items in sorted(values.items())}
            for dimension, values in sorted(groups.items())
        },
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [f"# {summary.get('dataset_name') or 'Evaluation'} Slice Summary", ""]
    for dimension, values in summary["slices"].items():
        lines.extend([f"## {dimension.title()}", "", "| Slice | Cases | Recall@1 | Recall@5 | Wrong high-confidence | No-result abstain |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for value, metrics in values.items():
            lines.append(
                "| {value} | {case_count} | {recall_at_1:.1%} | {recall_at_5:.1%} | "
                "{wrong_high_confidence_rate:.1%} | {no_result_abstain_rate:.1%} |".format(
                    value=value, **metrics
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize xists evaluation metrics by tag slices")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = summarize_report(report)
    if args.format == "markdown":
        print(markdown_summary(summary))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
