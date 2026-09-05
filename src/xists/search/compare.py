"""Multi-repository side-by-side comparison engine."""

from __future__ import annotations

import time
from typing import Any

from xists.search.query import PreparedIndex
from xists.search.similar import _resolve_repo_index, _string_list
from xists.types import (
    CompareAnalysis,
    CompareItem,
    CompareResponse,
    DirectRelationship,
    PairwiseSimilarity,
    ProjectDifferentiator,
)


def _synthesize_compare_analysis(
    projects: list[CompareItem],
) -> CompareAnalysis:
    """Synthesize commonalities and per-project differentiators across compared projects."""
    if not projects:
        return {
            "shared_capabilities": [],
            "shared_ecosystems": [],
            "shared_topics": [],
            "shared_languages": [],
            "differentiators": {},
            "direct_links": [],
        }

    # 1. Shared Ecosystems
    ecosystem_sets = [
        {str(e).strip().lower() for e in p.get("ecosystem", []) if str(e).strip()} for p in projects
    ]
    shared_ecosystems = sorted(set.intersection(*ecosystem_sets)) if ecosystem_sets else []

    # 2. Shared Topics
    topic_sets = [
        {str(t).strip().lower() for t in p.get("topics", []) if str(t).strip()} for p in projects
    ]
    shared_topics = sorted(set.intersection(*topic_sets)) if topic_sets else []

    # 3. Shared Languages
    languages = [p.get("language") for p in projects if p.get("language")]
    if len(languages) == len(projects) and len({lang.lower() for lang in languages if lang}) == 1:
        shared_languages = [languages[0]] if languages[0] else []
    else:
        shared_languages = []

    # 4. Capabilities analysis
    capability_sets = [
        {str(c).strip().lower() for c in p.get("capabilities", []) if str(c).strip()}
        for p in projects
    ]
    shared_capabilities_lower = (
        set.intersection(*capability_sets) if capability_sets and all(capability_sets) else set()
    )
    shared_capabilities = sorted(shared_capabilities_lower)

    # 5. Differentiators per project
    differentiators: dict[str, ProjectDifferentiator | dict[str, Any]] = {}
    for idx, p in enumerate(projects):
        repo_id = p.get("repo_id", "")
        p_caps_lower = capability_sets[idx]
        p_use_cases_lower = {
            str(u).strip().lower() for u in p.get("use_cases", []) if str(u).strip()
        }

        # Other project capabilities / use cases
        other_caps = set.union(*(capability_sets[j] for j in range(len(projects)) if j != idx))
        other_use_cases = set.union(
            *(
                {str(u).strip().lower() for u in projects[j].get("use_cases", []) if str(u).strip()}
                for j in range(len(projects))
                if j != idx
            )
        )

        # Unique capabilities (present in p, not in any other)
        unique_caps = [
            c
            for c in p.get("capabilities", [])
            if str(c).strip().lower() in (p_caps_lower - other_caps)
        ]
        # Unique use cases
        unique_use_cases = [
            u
            for u in p.get("use_cases", [])
            if str(u).strip().lower() in (p_use_cases_lower - other_use_cases)
        ]

        differentiators[repo_id] = {
            "unique_capabilities": unique_caps,
            "unique_use_cases": unique_use_cases,
            "not_for": list(p.get("not_for", [])),
            "license": p.get("license"),
            "stars": int(p.get("stars", 0)),
        }

    # 6. Direct Relationships (links between compared repos)
    direct_links: list[DirectRelationship | dict[str, str]] = []
    repo_ids_lower = {p["repo_id"].lower(): p["repo_id"] for p in projects if p.get("repo_id")}

    for p in projects:
        src = p.get("repo_id", "")
        replaces = [str(r).strip().lower() for r in p.get("replaces", []) if str(r).strip()]
        related = [str(r).strip().lower() for r in p.get("related_projects", []) if str(r).strip()]

        for target_lower, orig_target in repo_ids_lower.items():
            if target_lower == src.lower():
                continue
            if target_lower in replaces or any(target_lower in r for r in replaces):
                direct_links.append(
                    {
                        "source": src,
                        "target": orig_target,
                        "relation": "replaces",
                    }
                )
            elif target_lower in related or any(target_lower in r for r in related):
                direct_links.append(
                    {
                        "source": src,
                        "target": orig_target,
                        "relation": "related_project",
                    }
                )

    return {
        "shared_capabilities": shared_capabilities,
        "shared_ecosystems": shared_ecosystems,
        "shared_topics": shared_topics,
        "shared_languages": shared_languages,
        "differentiators": differentiators,
        "direct_links": direct_links,
    }


def compare_projects_prepared(
    repo_ids: list[str],
    prepared: PreparedIndex,
) -> CompareResponse:
    """Compare 2 to 5 repositories side by side using indexed metadata and pairwise similarity."""
    start_time = time.perf_counter()

    if not isinstance(repo_ids, (list, tuple)):
        raise ValueError("Repositories must be provided as a list of repo identifiers.")

    cleaned_ids = [str(r).strip() for r in repo_ids if str(r).strip()]
    if len(cleaned_ids) < 2 or len(cleaned_ids) > 5:
        raise ValueError(
            f"Comparison requires between 2 and 5 repositories, got {len(cleaned_ids)}."
        )

    # Check for duplicates
    seen_ids: set[str] = set()
    for r in cleaned_ids:
        lower_r = r.lower()
        if lower_r in seen_ids:
            raise ValueError(f"Duplicate repository identifier provided for comparison: '{r}'.")
        seen_ids.add(lower_r)

    # Resolve all repos
    missing: list[str] = []
    indices: list[int] = []
    for r in cleaned_ids:
        idx = _resolve_repo_index(r, prepared)
        if idx is None:
            missing.append(r)
        else:
            indices.append(idx)

    if missing:
        raise ValueError(
            f"Repositories not found in index: {', '.join(missing)}. "
            "Try searching for available projects with 'xists search'."
        )

    # Extract snapshots
    projects: list[CompareItem] = []
    for idx in indices:
        entry = prepared.entries[idx]
        meta = entry.get("metadata") or {}
        repo_id = str(entry.get("repo_id") or "")
        projects.append(
            {
                "repo_id": repo_id,
                "name": str(meta.get("name") or repo_id.split("/")[-1]),
                "url": str(meta.get("url") or f"https://github.com/{repo_id}"),
                "summary": meta.get("summary") or meta.get("description"),
                "description": meta.get("description"),
                "language": meta.get("language"),
                "stars": int(meta.get("stars") or 0),
                "forks": int(meta.get("forks") or 0),
                "license": meta.get("license"),
                "project_type": meta.get("project_type"),
                "ecosystem": _string_list(meta.get("ecosystem")),
                "capabilities": _string_list(meta.get("capabilities")),
                "use_cases": _string_list(meta.get("use_cases")),
                "not_for": _string_list(meta.get("not_for")),
                "replaces": _string_list(meta.get("replaces")),
                "related_projects": _string_list(meta.get("related_projects")),
                "topics": _string_list(meta.get("topics")),
                "archived": bool(meta.get("archived") is True or meta.get("disabled") is True),
            }
        )

    # Compute pairwise cosine similarity matrix
    num_projects = len(indices)
    sub_matrix = prepared.normalized_matrix[indices]  # Shape (N, D)
    sim_matrix = sub_matrix @ sub_matrix.T  # Shape (N, N)

    matrix: dict[str, dict[str, float]] = {}
    pairwise: list[PairwiseSimilarity] = []

    for i in range(num_projects):
        repo_a = projects[i]["repo_id"]
        matrix[repo_a] = {}
        for j in range(num_projects):
            repo_b = projects[j]["repo_id"]
            sim_val = round(float(sim_matrix[i, j]), 4)
            matrix[repo_a][repo_b] = sim_val
            if i < j:
                pairwise.append(
                    {
                        "repo_a": repo_a,
                        "repo_b": repo_b,
                        "similarity": sim_val,
                    }
                )

    analysis = _synthesize_compare_analysis(projects)
    elapsed = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "repo_ids": [p["repo_id"] for p in projects],
        "projects": projects,
        "matrix": matrix,
        "pairwise": pairwise,
        "analysis": analysis,
        "latency_ms": elapsed,
        "elapsed_ms": elapsed,
    }


__all__ = [
    "_synthesize_compare_analysis",
    "compare_projects_prepared",
]
