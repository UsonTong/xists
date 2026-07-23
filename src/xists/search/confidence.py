"""Explainable post-ranking confidence calibration."""

from __future__ import annotations

from typing import Any

CONFIDENCE_CALIBRATION_MODES = ("off", "evidence-v1")
CONFIDENCE_CALIBRATION_VERSION = "evidence-v1"


def _identity_kind(result: dict[str, Any]) -> str:
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return "none"
    identity = diagnostics.get("identity_evidence")
    if not isinstance(identity, dict):
        return "none"
    kind = identity.get("kind")
    return str(kind) if isinstance(kind, str) else "none"


def _ranking_evidence(result: dict[str, Any]) -> dict[str, Any]:
    evidence = result.get("ranking_evidence")
    return evidence if isinstance(evidence, dict) else {}


def calibrate_confidence(
    results: list[dict[str, Any]],
    *,
    ranking_strategy: str,
    mode: str = "off",
) -> list[dict[str, Any]]:
    """Annotate confidence evidence and optionally downgrade unsafe high confidence.

    This function deliberately never reorders, removes, or adds results.  It only
    changes an existing ``high_confidence`` label to ``exploratory`` when its
    supporting evidence is ambiguous or contradictory.
    """

    if mode not in CONFIDENCE_CALIBRATION_MODES:
        raise ValueError(f"Unknown confidence calibration mode: {mode}")

    for position, result in enumerate(results):
        initial = str(result.get("confidence") or "missing")
        identity_kind = _identity_kind(result)
        ranking = _ranking_evidence(result)
        rerank_score = result.get("rerank_score")
        semantic_rank = ranking.get("semantic_rank")
        rerank_rank = ranking.get("rerank_rank")
        reranker_available = isinstance(rerank_score, (int, float)) and isinstance(semantic_rank, int) and isinstance(rerank_rank, int)
        support: list[str] = []
        downgrade: list[str] = []

        if identity_kind in {"repo_id", "exact_value", "name_mention"}:
            support.append("direct_identity")
        elif identity_kind == "contextual_name_mention":
            downgrade.append("identity_is_contextual")
        elif identity_kind == "substring_mention":
            downgrade.append("identity_is_substring_only")

        score_margin = None
        if position == 0 and len(results) > 1:
            first_score = result.get("score")
            second_score = results[1].get("score")
            if isinstance(first_score, (int, float)) and isinstance(second_score, (int, float)):
                score_margin = float(first_score) - float(second_score)
                if score_margin <= 0.0:
                    downgrade.append("top_candidate_not_separated")

        if ranking_strategy == "rerank" and identity_kind == "none":
            if not reranker_available:
                downgrade.append("reranker_evidence_unavailable")
            elif semantic_rank != rerank_rank:
                downgrade.append("semantic_and_rerank_disagree")
            elif semantic_rank != 1:
                downgrade.append("top_result_is_not_leading_in_both_rankers")
            else:
                support.append("semantic_and_rerank_agree")

        final = initial
        if mode == CONFIDENCE_CALIBRATION_VERSION and initial == "high_confidence" and downgrade:
            final = "exploratory"
            result["confidence"] = final

        result["confidence_evidence"] = {
            "version": CONFIDENCE_CALIBRATION_VERSION,
            "mode": mode,
            "initial_confidence": initial,
            "final_confidence": final,
            "identity_evidence": identity_kind,
            "ranking_strategy": ranking_strategy,
            "reranker_available": reranker_available if ranking_strategy == "rerank" else None,
            "semantic_rank": semantic_rank if isinstance(semantic_rank, int) else None,
            "rerank_rank": rerank_rank if isinstance(rerank_rank, int) else None,
            "top_score_margin": round(score_margin, 8) if score_margin is not None else None,
            "supporting_signals": support,
            "downgrade_reasons": downgrade,
        }
    return results
