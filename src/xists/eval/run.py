"""Run retrieval evaluation against an xists embedding index."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from xists import __version__
from xists.eval.judge import judge_top1_vs_expected
from xists.eval.schema import load_dataset
from xists.profile.llm import LLMConfig
from xists.search.embed import EmbeddingConfig
from xists.search.index import load_index
from xists.search.query import rank_many


def _find_rank(results: list[dict[str, Any]], repo_ids: set[str]) -> int | None:
    for index, result in enumerate(results, start=1):
        if result.get("repo_id") in repo_ids:
            return index
    return None


def _safe_divide(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _round_metric(value: float) -> float:
    return round(value, 6)


def _load_records_by_repo_id(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Records file must be a JSON array: {path}")
    by_repo_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict) and record.get("repo_id"):
            by_repo_id[record["repo_id"]] = record
    return by_repo_id


def evaluate_dataset(
    cases_path: Path,
    index_path: Path,
    config: EmbeddingConfig,
    *,
    top_k: int = 10,
    batch_size: int = 64,
    embed_many: Any | None = None,
    llm_judge_config: LLMConfig | None = None,
    records_path: Path | None = None,
    judge_caller: Any | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started_clock = perf_counter()

    dataset = load_dataset(cases_path)
    index = load_index(index_path)
    records_by_repo_id: dict[str, dict[str, Any]] = {}
    judge_enabled = llm_judge_config is not None
    if judge_enabled:
        if records_path is None:
            raise ValueError("records_path is required when LLM judge is enabled")
        records_by_repo_id = _load_records_by_repo_id(records_path)
    queries = [case["query"] for case in dataset["cases"]]
    rank_many_fn = embed_many if embed_many is not None else rank_many
    if rank_many_fn is rank_many:
        ranked = rank_many_fn(queries, index, config, top_k=top_k, batch_size=batch_size)
    else:
        ranked = rank_many_fn(queries, index, config, top_k=top_k, batch_size=batch_size)

    if len(ranked) != len(dataset["cases"]):
        raise ValueError(f"Evaluation returned {len(ranked)} results for {len(dataset['cases'])} cases")

    exact_hit_at_1 = 0
    exact_hit_at_k = 0
    acceptable_hit_at_1 = 0
    acceptable_hit_at_k = 0
    abstained = 0
    mrr_exact_total = 0.0
    mrr_acceptable_total = 0.0
    top_1_high_confidence_count = 0
    top_1_exploratory_count = 0
    top_1_missing_count = 0
    wrong_high_confidence_top_1_count = 0
    top1_miss_count = 0
    top1_miss_acceptable_count = 0
    top1_miss_serious_count = 0
    top1_miss_insufficient_evidence_count = 0
    judge_summary = {
        "enabled": judge_enabled,
        "model": llm_judge_config.model if llm_judge_config else None,
        "prompt_version": None,
        "total_ran": 0,
        "acceptable_substitute_count": 0,
        "serious_mismatch_count": 0,
        "insufficient_evidence_count": 0,
        "small_difference_count": 0,
        "moderate_difference_count": 0,
        "large_difference_count": 0,
    }
    per_case: list[dict[str, Any]] = []

    for case, result in zip(dataset["cases"], ranked):
        results = result.get("results") or []
        acceptable_set = set(case["acceptable_set"])
        expected_repo_id = case["expected_repo_id"]
        exact_rank = _find_rank(results, {expected_repo_id})
        acceptable_rank = _find_rank(results, acceptable_set)
        top_result = results[0] if results else None
        top_result_repo_id = top_result.get("repo_id") if top_result else None
        top_result_confidence = top_result.get("confidence") if top_result else None
        exact_match = top_result_repo_id == expected_repo_id
        acceptable_match = top_result_repo_id in acceptable_set if top_result_repo_id else False

        if result.get("abstained"):
            abstained += 1
        if exact_match:
            exact_hit_at_1 += 1
        if exact_rank is not None:
            exact_hit_at_k += 1
            mrr_exact_total += 1.0 / exact_rank
        if acceptable_match:
            acceptable_hit_at_1 += 1
        if acceptable_rank is not None:
            acceptable_hit_at_k += 1
            mrr_acceptable_total += 1.0 / acceptable_rank

        if top_result_confidence == "high_confidence":
            top_1_high_confidence_count += 1
            if top_result_repo_id not in acceptable_set:
                wrong_high_confidence_top_1_count += 1
        elif top_result_confidence == "exploratory":
            top_1_exploratory_count += 1
        else:
            top_1_missing_count += 1

        judge_result = None
        top1_status = "exact"
        if not exact_match:
            top1_miss_count += 1
            if acceptable_match:
                top1_status = "acceptable"
                top1_miss_acceptable_count += 1
            elif judge_enabled and top_result_repo_id:
                expected_record = records_by_repo_id.get(expected_repo_id)
                if expected_record is None:
                    raise ValueError(f"Expected repo not found in records file: {expected_repo_id}")
                top1_record = records_by_repo_id.get(top_result_repo_id)
                if top1_record is None:
                    raise ValueError(f"Top result repo not found in records file: {top_result_repo_id}")
                judge_result = judge_top1_vs_expected(
                    case["query"],
                    expected_record=expected_record,
                    top1_record=top1_record,
                    config=llm_judge_config,
                    caller=judge_caller,
                )
                judge_summary["prompt_version"] = judge_result["prompt_version"]
                judge_summary["total_ran"] += 1
                judge_summary[f"{judge_result['verdict']}_count"] += 1
                judge_summary[f"{judge_result['difference_size']}_difference_count"] += 1
                if judge_result["verdict"] == "acceptable_substitute":
                    top1_status = "acceptable"
                    top1_miss_acceptable_count += 1
                elif judge_result["verdict"] == "insufficient_evidence":
                    top1_status = "insufficient_evidence"
                    top1_miss_insufficient_evidence_count += 1
                else:
                    top1_status = "serious_mismatch"
                    top1_miss_serious_count += 1
            else:
                top1_status = "serious_mismatch"
                top1_miss_serious_count += 1

        per_case.append(
            {
                "id": case["id"],
                "query": case["query"],
                "tags": case["tags"],
                "abstained": bool(result.get("abstained")),
                "expected_repo_id": expected_repo_id,
                "top_result_repo_id": top_result_repo_id,
                "exact_match": exact_match,
                "acceptable_match": acceptable_match,
                "exact_rank": exact_rank,
                "acceptable_rank": acceptable_rank,
                "top_result_confidence": top_result_confidence,
                "top1_status": top1_status,
                "judge_ran": judge_result is not None,
                "judge_verdict": judge_result.get("verdict") if judge_result else None,
                "judge_difference_size": judge_result.get("difference_size") if judge_result else None,
                "judge_confidence": judge_result.get("confidence") if judge_result else None,
                "judge_reason_short": judge_result.get("reason_short") if judge_result else None,
                "judge_query_specificity": judge_result.get("query_specificity") if judge_result else None,
                "judge_language_ecosystem_material": judge_result.get("language_ecosystem_material") if judge_result else None,
                "judge": judge_result,
            }
        )

    case_count = len(dataset["cases"])
    metrics = {
        "exact_hit_at_1": _round_metric(_safe_divide(exact_hit_at_1, case_count)),
        "exact_hit_at_k": _round_metric(_safe_divide(exact_hit_at_k, case_count)),
        "mrr_exact": _round_metric(_safe_divide(mrr_exact_total, case_count)),
        "acceptable_hit_at_1": _round_metric(_safe_divide(acceptable_hit_at_1, case_count)),
        "acceptable_hit_at_k": _round_metric(_safe_divide(acceptable_hit_at_k, case_count)),
        "mrr_acceptable": _round_metric(_safe_divide(mrr_acceptable_total, case_count)),
        "abstain_rate": _round_metric(_safe_divide(abstained, case_count)),
        "acceptable_minus_exact_hit_at_1": _round_metric(
            _safe_divide(acceptable_hit_at_1 - exact_hit_at_1, case_count)
        ),
        "acceptable_minus_exact_hit_at_k": _round_metric(
            _safe_divide(acceptable_hit_at_k - exact_hit_at_k, case_count)
        ),
        "mrr_acceptable_minus_exact": _round_metric(
            _safe_divide(mrr_acceptable_total - mrr_exact_total, case_count)
        ),
        "exact_top1_rate": _round_metric(_safe_divide(exact_hit_at_1, case_count)),
        "acceptable_top1_rate": _round_metric(_safe_divide(top1_miss_acceptable_count, case_count)),
        "serious_top1_error_rate": _round_metric(_safe_divide(top1_miss_serious_count, case_count)),
        "insufficient_evidence_top1_rate": _round_metric(
            _safe_divide(top1_miss_insufficient_evidence_count, case_count)
        ),
        "effective_top1_rate": _round_metric(
            _safe_divide(exact_hit_at_1 + top1_miss_acceptable_count, case_count)
        ),
    }

    finished_at = datetime.now(timezone.utc)
    return {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": _round_metric(perf_counter() - started_clock),
        "xists_version": __version__,
        "dataset_name": dataset["dataset_name"],
        "cases": str(cases_path),
        "index": str(index_path),
        "top_k": top_k,
        "batch_size": batch_size,
        "case_count": case_count,
        "metrics": metrics,
        "confidence": {
            "top_1_high_confidence_count": top_1_high_confidence_count,
            "top_1_exploratory_count": top_1_exploratory_count,
            "top_1_missing_count": top_1_missing_count,
            "wrong_high_confidence_top_1_count": wrong_high_confidence_top_1_count,
        },
        "top1_summary": {
            "top1_miss_count": top1_miss_count,
            "top1_miss_acceptable_count": top1_miss_acceptable_count,
            "top1_miss_serious_count": top1_miss_serious_count,
            "top1_miss_insufficient_evidence_count": top1_miss_insufficient_evidence_count,
            "top1_miss_acceptable_rate": _round_metric(_safe_divide(top1_miss_acceptable_count, top1_miss_count)),
            "top1_miss_serious_rate": _round_metric(_safe_divide(top1_miss_serious_count, top1_miss_count)),
            "top1_miss_insufficient_evidence_rate": _round_metric(
                _safe_divide(top1_miss_insufficient_evidence_count, top1_miss_count)
            ),
        },
        "judge_summary": judge_summary,
        "results": per_case,
    }
