"""Inspect xists evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MISS_STATUS_ORDER = {
    "serious_mismatch": 0,
    "insufficient_evidence": 1,
    "acceptable": 2,
    "exact": 3,
}


def _safe_rate_count(rate: Any, count: Any, total: int) -> str:
    if not isinstance(rate, (int, float)):
        rate = 0.0
    if not isinstance(count, int):
        count = 0
    return f"{rate * 100:.1f}% ({count}/{total})"


def build_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, human-readable summary from an eval report."""

    metrics = report.get("metrics") or {}
    confidence = report.get("confidence") or {}
    top1_summary = report.get("top1_summary") or {}
    case_count = int(report.get("case_count") or 0)

    exact_top1_count = sum(1 for result in report.get("results", []) if result.get("top1_status") == "exact")
    acceptable_substitute_top1_count = int(top1_summary.get("top1_miss_acceptable_count") or 0)
    acceptable_top1_count = exact_top1_count + acceptable_substitute_top1_count
    serious_mismatch_count = int(top1_summary.get("top1_miss_serious_count") or 0)
    insufficient_evidence_count = int(top1_summary.get("top1_miss_insufficient_evidence_count") or 0)
    abstain_count = sum(1 for result in report.get("results", []) if result.get("abstained"))
    wrong_high_confidence_count = int(confidence.get("wrong_high_confidence_top_1_count") or 0)

    return {
        "case_count": case_count,
        "exact_top_1": {
            "count": exact_top1_count,
            "rate": metrics.get("exact_top1_rate", metrics.get("exact_hit_at_1", 0.0)),
        },
        "acceptable_top_1": {
            "count": acceptable_top1_count,
            "rate": metrics.get("effective_top1_rate", metrics.get("acceptable_hit_at_1", 0.0)),
            "description": "top-1 was exact or an acceptable repo/family/judge substitute",
        },
        "acceptable_substitute_top_1": {
            "count": acceptable_substitute_top1_count,
            "rate": metrics.get("acceptable_top1_rate", 0.0),
            "description": "top-1 missed exact target but matched an acceptable repo/family or judge substitute",
        },
        "effective_top_1": {
            "count": acceptable_top1_count,
            "rate": metrics.get("effective_top1_rate", metrics.get("acceptable_hit_at_1", 0.0)),
            "description": "alias of acceptable_top_1 for backwards-readable reports",
        },
        "serious_mismatch": {
            "count": serious_mismatch_count,
            "rate": metrics.get("serious_top1_error_rate", 0.0),
        },
        "insufficient_evidence": {
            "count": insufficient_evidence_count,
            "rate": metrics.get("insufficient_evidence_top1_rate", 0.0),
        },
        "abstain": {
            "count": abstain_count,
            "rate": metrics.get("abstain_rate", 0.0),
        },
        "wrong_high_confidence": {
            "count": wrong_high_confidence_count,
            "rate": (round(wrong_high_confidence_count / case_count, 6) if case_count else 0.0),
            "description": "top-1 was high_confidence but outside the acceptable set",
        },
    }


def build_summary_text(summary: dict[str, Any]) -> list[str]:
    """Render a stable text summary that is easy to scan in JSON or CLI output."""

    case_count = int(summary.get("case_count") or 0)
    return [
        f"exact top-1: {_safe_rate_count((summary.get('exact_top_1') or {}).get('rate'), (summary.get('exact_top_1') or {}).get('count'), case_count)}",
        f"acceptable top-1: {_safe_rate_count((summary.get('acceptable_top_1') or {}).get('rate'), (summary.get('acceptable_top_1') or {}).get('count'), case_count)}",
        f"effective top-1: {_safe_rate_count((summary.get('effective_top_1') or {}).get('rate'), (summary.get('effective_top_1') or {}).get('count'), case_count)}",
        f"serious mismatch: {_safe_rate_count((summary.get('serious_mismatch') or {}).get('rate'), (summary.get('serious_mismatch') or {}).get('count'), case_count)}",
        f"insufficient evidence: {_safe_rate_count((summary.get('insufficient_evidence') or {}).get('rate'), (summary.get('insufficient_evidence') or {}).get('count'), case_count)}",
        f"abstain rate: {_safe_rate_count((summary.get('abstain') or {}).get('rate'), (summary.get('abstain') or {}).get('count'), case_count)}",
        f"wrong high-confidence: {(summary.get('wrong_high_confidence') or {}).get('count', 0)} cases",
    ]


def case_brief(result: dict[str, Any]) -> dict[str, Any]:
    """Return the fields needed to inspect a single evaluated case."""

    brief = {
        "id": result.get("id"),
        "query": result.get("query"),
        "query_intent": result.get("query_intent"),
        "tags": result.get("tags") or [],
        "top1_status": result.get("top1_status"),
        "abstained": bool(result.get("abstained")),
        "expected_repo_id": result.get("expected_repo_id"),
        "top_result_repo_id": result.get("top_result_repo_id"),
        "top_result_confidence": result.get("top_result_confidence"),
        "top_result_why": result.get("top_result_why") or [],
        "exact_rank": result.get("exact_rank"),
        "acceptable_rank": result.get("acceptable_rank"),
    }
    if result.get("judge_reason_short"):
        brief["judge_reason_short"] = result.get("judge_reason_short")
    return brief


def build_top_misses(results: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    """Collect non-exact cases, sorted to put the most actionable misses first."""

    misses = [case_brief(result) for result in results if result.get("top1_status") != "exact"]
    misses.sort(
        key=lambda item: (
            MISS_STATUS_ORDER.get(str(item.get("top1_status")), 99),
            item.get("top_result_confidence") != "high_confidence",
            item.get("acceptable_rank") is None,
            item.get("acceptable_rank") or 9999,
            str(item.get("id") or ""),
        )
    )
    if limit is None:
        return misses
    return misses[:limit]


def load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Evaluation report not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Evaluation report is not valid JSON: {error}") from error
    if not isinstance(report, dict):
        raise ValueError("Evaluation report must be a JSON object")
    return report


def inspect_report(
    report: dict[str, Any],
    *,
    status: str | None = None,
    limit: int = 20,
    include_exact: bool = False,
    tag: str | None = None,
    intent: str | None = None,
) -> dict[str, Any]:
    """Build a compact inspection payload from an evaluation report."""

    results = report.get("results") or []
    if not isinstance(results, list):
        raise ValueError("Evaluation report results must be a list")

    selected: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        top1_status = result.get("top1_status")
        if not include_exact and top1_status == "exact":
            continue
        if status and top1_status != status:
            continue
        tags = result.get("tags") or []
        if tag and tag not in tags:
            continue
        query_intent = result.get("query_intent") or {}
        if intent and query_intent.get("type") != intent:
            continue
        selected.append(case_brief(result))

    selected.sort(
        key=lambda item: (
            MISS_STATUS_ORDER.get(str(item.get("top1_status")), 99),
            item.get("top_result_confidence") != "high_confidence",
            item.get("acceptable_rank") is None,
            item.get("acceptable_rank") or 9999,
            str(item.get("id") or ""),
        )
    )

    summary = report.get("summary") if isinstance(report.get("summary"), dict) else build_summary(report)
    summary_text = report.get("summary_text") if isinstance(report.get("summary_text"), list) else build_summary_text(summary)

    return {
        "dataset_name": report.get("dataset_name"),
        "report_case_count": report.get("case_count"),
        "metrics": report.get("metrics") or {},
        "confidence": report.get("confidence") or {},
        "summary": summary,
        "summary_text": summary_text,
        "filter": {
            "status": status,
            "include_exact": include_exact,
            "limit": limit,
            "tag": tag,
            "intent": intent,
        },
        "inspected_count": min(len(selected), limit),
        "matching_count": len(selected),
        "cases": selected[:limit],
    }
