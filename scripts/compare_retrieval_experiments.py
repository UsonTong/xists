"""Compare tagged retrieval evaluation reports without domain-specific logic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_eval_slices import summarize_report


def _latency_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    results = report.get("results")
    if not isinstance(results, list):
        return {"p50_ms": None, "p95_ms": None}
    values = [result.get("latency_ms") for result in results if isinstance(result, dict)]
    valid = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not valid:
        return {"p50_ms": None, "p95_ms": None}

    def percentile(percent: float) -> float:
        index = max(0, min(len(valid) - 1, round((len(valid) - 1) * percent)))
        return round(valid[index], 3)

    return {"p50_ms": percentile(0.5), "p95_ms": percentile(0.95)}


def compare_reports(named_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not named_reports:
        raise ValueError("at least one report is required")
    comparisons: dict[str, Any] = {}
    for name, report in named_reports.items():
        summary = summarize_report(report)
        comparisons[name] = {
            "ranking_strategy": report.get("ranking_strategy", "metadata"),
            "rerank_candidate_limit": report.get("rerank_candidate_limit"),
            "duration_seconds": report.get("duration_seconds"),
            "latency": _latency_metrics(report),
            "overall": summary["overall"],
            "slices": summary["slices"],
        }
    return {
        "dataset_name": next(iter(named_reports.values())).get("dataset_name"),
        "experiments": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare xists retrieval experiment reports")
    parser.add_argument("--report", action="append", required=True, metavar="NAME=PATH")
    args = parser.parse_args()
    reports: dict[str, dict[str, Any]] = {}
    for item in args.report:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--report must use NAME=PATH")
        reports[name] = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    print(json.dumps(compare_reports(reports), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
