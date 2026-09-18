"""Repository similarity search engine."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from xists.search.query import PreparedIndex
from xists.types import SearchFilter, SimilarResponse, SimilarResultItem


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    return []


def _resolve_repo_index(repo_id: str, prepared: PreparedIndex) -> int | None:
    """Resolve a repository identifier to its row index in PreparedIndex."""
    # 1. Exact match in repo_id_to_index
    if repo_id in prepared.repo_id_to_index:
        return prepared.repo_id_to_index[repo_id]

    target_lower = repo_id.strip().lower()

    # 2. Case-insensitive repo_id match
    for idx, cache in enumerate(prepared.metadata_caches):
        if cache.get("repo_id_lower") == target_lower:
            return idx

    # 3. Identity / alias match
    for idx, cache in enumerate(prepared.metadata_caches):
        if target_lower in cache.get("identity_values_lower", set()):
            return idx

    # 4. Suffix match (e.g. name only without owner)
    for idx, cache in enumerate(prepared.metadata_caches):
        r_id = cache.get("repo_id_lower", "")
        if r_id.endswith(f"/{target_lower}") or r_id == target_lower:
            return idx

    return None


def _confidence_bucket(score: float) -> str:
    if score >= 0.70:
        return "high_confidence"
    if score >= 0.50:
        return "medium_confidence"
    if score >= 0.35:
        return "exploratory"
    return "abstain"


def _generate_similar_why(
    target_cache: dict[str, Any],
    target_meta: dict[str, Any],
    cand_cache: dict[str, Any],
    cand_meta: dict[str, Any],
    cand_repo_id: str,
    sim_score: float,
) -> tuple[list[str], str | None]:
    """Generate explainability reasons and relation link for repository similarity."""
    why: list[str] = []
    relationship: str | None = None

    target_repo_lower = target_cache.get("repo_id_lower", "")
    cand_repo_lower = cand_repo_id.lower()

    target_replaces = {
        str(r).strip().lower() for r in _string_list(target_meta.get("replaces")) if str(r).strip()
    }
    target_related = {
        str(r).strip().lower()
        for r in _string_list(target_meta.get("related_projects"))
        if str(r).strip()
    }
    cand_replaces = {
        str(r).strip().lower() for r in _string_list(cand_meta.get("replaces")) if str(r).strip()
    }
    cand_related = {
        str(r).strip().lower()
        for r in _string_list(cand_meta.get("related_projects"))
        if str(r).strip()
    }

    # Direct relationship checks
    if cand_repo_lower in target_replaces or any(cand_repo_lower in r for r in target_replaces):
        why.append("Target explicitly lists this project as an alternative/replacement")
        relationship = "alternative"
    elif cand_repo_lower in target_related or any(cand_repo_lower in r for r in target_related):
        why.append("Target explicitly references this as a related project")
        relationship = "related_project"
    elif target_repo_lower in cand_replaces or any(target_repo_lower in r for r in cand_replaces):
        why.append("This project explicitly lists target as an alternative/replacement")
        relationship = "alternative"
    elif target_repo_lower in cand_related or any(target_repo_lower in r for r in cand_related):
        why.append("This project explicitly references target as a related project")
        relationship = "related_project"

    # Shared ecosystem
    target_ecos = target_cache.get("ecosystem_set") or frozenset()
    cand_ecos = cand_cache.get("ecosystem_set") or frozenset()
    shared_ecos = target_ecos & cand_ecos
    if shared_ecos:
        why.append(f"Shared ecosystem: {', '.join(sorted(shared_ecos))}")

    # Shared language
    target_lang = target_cache.get("language_alias")
    cand_lang = cand_cache.get("language_alias")
    if target_lang and target_lang == cand_lang:
        lang_display = target_meta.get("language") or target_lang.capitalize()
        why.append(f"Same primary language: {lang_display}")

    # Shared project type
    target_pt = target_cache.get("project_type_norm")
    cand_pt = cand_cache.get("project_type_norm")
    if target_pt and target_pt == cand_pt:
        pt_display = target_meta.get("project_type") or target_pt.replace("_", " ").title()
        why.append(f"Same project type: {pt_display}")

    # Shared topics
    target_topics = target_cache.get("topics_set") or frozenset()
    cand_topics = cand_cache.get("topics_set") or frozenset()
    shared_topics = target_topics & cand_topics
    if shared_topics:
        sample_topics = sorted(shared_topics)[:3]
        why.append(f"Shared topics: {', '.join(sample_topics)}")

    # Vector score explanation
    if sim_score >= 0.70:
        why.append(f"High embedding similarity ({sim_score:.3f})")
    elif sim_score >= 0.50:
        why.append(f"Moderate embedding similarity ({sim_score:.3f})")
    else:
        why.append(f"Embedding similarity ({sim_score:.3f})")

    return why, relationship


def find_similar_prepared(
    repo_id: str,
    prepared: PreparedIndex,
    *,
    top_k: int = 10,
    filters: SearchFilter | dict[str, Any] | str | None = None,
) -> SimilarResponse:
    """Find top-k similar repositories using precomputed embeddings in PreparedIndex."""
    start_time = time.perf_counter()

    target_idx = _resolve_repo_index(repo_id, prepared)
    if target_idx is None:
        raise ValueError(
            f"Repository '{repo_id}' not found in index. "
            f"Try searching for it first with 'xists search {repo_id}'."
        )

    target_entry = prepared.entries[target_idx]
    target_cache = prepared.metadata_caches[target_idx]
    target_meta = target_entry.get("metadata") or {}
    resolved_target_repo_id = str(target_entry.get("repo_id") or repo_id)

    target_summary: dict[str, Any] = {
        "repo_id": resolved_target_repo_id,
        "name": target_meta.get("name") or resolved_target_repo_id.split("/")[-1],
        "url": target_meta.get("url") or f"https://github.com/{resolved_target_repo_id}",
        "summary": target_meta.get("summary") or target_meta.get("description") or "",
        "language": target_meta.get("language"),
        "stars": target_meta.get("stars"),
        "license": target_meta.get("license"),
        "project_type": target_meta.get("project_type"),
        "ecosystem": list(target_meta.get("ecosystem") or []),
    }

    if len(prepared.entries) <= 1:
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "target_repo_id": resolved_target_repo_id,
            "target": target_summary,
            "total_candidates": 0,
            "considered": 0,
            "results": [],
            "filters": filters,
            "latency_ms": elapsed,
            "elapsed_ms": elapsed,
        }

    target_vec = prepared.get_vector(target_idx)
    similarities = prepared.score_query_vector(target_vec)

    # Exclude self
    valid_mask = np.ones(len(prepared.entries), dtype=bool)
    valid_mask[target_idx] = False

    # Apply facet filters
    filter_mask = prepared.compute_filter_mask(filters)
    if filter_mask is not None:
        valid_mask &= filter_mask

    candidate_indices = np.flatnonzero(valid_mask)
    if len(candidate_indices) == 0:
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "target_repo_id": resolved_target_repo_id,
            "target": target_summary,
            "total_candidates": 0,
            "considered": 0,
            "results": [],
            "filters": filters,
            "latency_ms": elapsed,
            "elapsed_ms": elapsed,
        }

    candidate_scores = similarities[candidate_indices]
    sorted_order = np.argsort(-candidate_scores)
    selected_k = min(len(candidate_indices), max(1, int(top_k)))
    top_indices = candidate_indices[sorted_order[:selected_k]]

    if hasattr(prepared.entries, "prefetch") and len(top_indices) > 0:
        prepared.entries.prefetch(top_indices)

    results: list[SimilarResultItem] = []
    for cand_idx in top_indices:
        cand_entry = prepared.entries[cand_idx]
        cand_cache = prepared.metadata_caches[cand_idx]
        cand_meta = cand_entry.get("metadata") or {}
        cand_repo_id = str(cand_entry.get("repo_id") or "")
        sim_score = float(similarities[cand_idx])

        why_reasons, relationship = _generate_similar_why(
            target_cache,
            target_meta,
            cand_cache,
            cand_meta,
            cand_repo_id,
            sim_score,
        )

        results.append(
            {
                "repo_id": cand_repo_id,
                "name": str(cand_meta.get("name") or cand_repo_id.split("/")[-1]),
                "url": str(cand_meta.get("url") or f"https://github.com/{cand_repo_id}"),
                "score": round(sim_score, 4),
                "similarity": round(sim_score, 4),
                "confidence": _confidence_bucket(sim_score),
                "summary": cand_meta.get("summary") or cand_meta.get("description"),
                "language": cand_meta.get("language"),
                "stars": cand_meta.get("stars"),
                "license": cand_meta.get("license"),
                "project_type": cand_meta.get("project_type"),
                "ecosystem": list(cand_meta.get("ecosystem") or []),
                "why": why_reasons,
                "relationship": relationship,
                "metadata": cand_meta,
            }
        )

    elapsed = round((time.perf_counter() - start_time) * 1000, 2)
    return {
        "target_repo_id": resolved_target_repo_id,
        "target": target_summary,
        "total_candidates": len(prepared.entries) - 1,
        "considered": len(candidate_indices),
        "results": results,
        "filters": filters,
        "latency_ms": elapsed,
        "elapsed_ms": elapsed,
    }


__all__ = [
    "_confidence_bucket",
    "_generate_similar_why",
    "_resolve_repo_index",
    "find_similar_prepared",
]
