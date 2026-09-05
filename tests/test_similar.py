"""Unit and integration tests for repository similarity search."""

from __future__ import annotations

import numpy as np
import pytest

from xists.api import find_similar
from xists.search.index import INDEX_VERSION
from xists.search.query import prepare_index
from xists.search.similar import (
    _confidence_bucket,
    _generate_similar_why,
    _resolve_repo_index,
    find_similar_prepared,
)


def _make_unit_vector(angle_rad: float) -> list[float]:
    return [float(np.cos(angle_rad)), float(np.sin(angle_rad))]


def _build_test_index(vectors: list[dict]) -> dict:
    return {
        "index_version": INDEX_VERSION,
        "record_schema_version": 2,
        "embedding_model": "test-embed-model",
        "embedding_input_version": 3,
        "dimension": 2,
        "record_count": len(vectors),
        "vectors": vectors,
    }


def test_confidence_bucket_thresholds():
    assert _confidence_bucket(0.85) == "high_confidence"
    assert _confidence_bucket(0.70) == "high_confidence"
    assert _confidence_bucket(0.69) == "medium_confidence"
    assert _confidence_bucket(0.50) == "medium_confidence"
    assert _confidence_bucket(0.49) == "exploratory"
    assert _confidence_bucket(0.35) == "exploratory"
    assert _confidence_bucket(0.34) == "abstain"


def test_find_similar_basic_and_self_exclusion():
    entries = [
        {
            "repo_id": "org/target",
            "vector": _make_unit_vector(0.0),  # (1, 0)
            "metadata": {
                "name": "target",
                "summary": "Target repository",
                "language": "Python",
                "stars": 5000,
                "license": "MIT",
                "project_type": "framework",
                "ecosystem": ["python", "web"],
                "topics": ["web", "api"],
            },
        },
        {
            "repo_id": "org/close",
            "vector": _make_unit_vector(0.1),  # Cosine sim ~ 0.995
            "metadata": {
                "name": "close",
                "summary": "Close match",
                "language": "Python",
                "stars": 8000,
                "license": "MIT",
                "project_type": "framework",
                "ecosystem": ["python", "web"],
                "topics": ["web", "microservice"],
                "replaces": ["org/target"],
            },
        },
        {
            "repo_id": "org/medium",
            "vector": _make_unit_vector(0.7),  # Cosine sim ~ 0.765
            "metadata": {
                "name": "medium",
                "summary": "Medium similarity",
                "language": "Go",
                "stars": 2000,
                "license": "Apache-2.0",
                "project_type": "library",
                "ecosystem": ["go"],
            },
        },
        {
            "repo_id": "org/far",
            "vector": _make_unit_vector(1.5),  # Cosine sim ~ 0.07
            "metadata": {
                "name": "far",
                "summary": "Far similarity",
                "language": "Rust",
                "stars": 100,
            },
        },
    ]

    index_dict = _build_test_index(entries)
    prepared = prepare_index(index_dict)

    res = find_similar_prepared("org/target", prepared, top_k=2)

    assert res["target_repo_id"] == "org/target"
    assert res["target"]["name"] == "target"
    assert res["total_candidates"] == 3
    assert res["considered"] == 3
    assert len(res["results"]) == 2

    # Top 1 should be org/close, NOT org/target (self excluded)
    assert res["results"][0]["repo_id"] == "org/close"
    assert res["results"][0]["score"] > 0.99
    assert res["results"][0]["confidence"] == "high_confidence"
    assert res["results"][0]["relationship"] == "alternative"

    # Top 2 should be org/medium
    assert res["results"][1]["repo_id"] == "org/medium"


def test_find_similar_why_reasons_generation():
    target_cache = {
        "repo_id_lower": "supabase/supabase",
        "ecosystem_set": frozenset({"typescript", "web", "postgres"}),
        "language_alias": "typescript",
        "project_type_norm": "platform",
        "topics_set": frozenset({"firebase", "auth", "database"}),
    }
    target_meta = {
        "language": "TypeScript",
        "project_type": "platform",
        "related_projects": ["pocketbase/pocketbase"],
    }

    cand_cache = {
        "repo_id_lower": "pocketbase/pocketbase",
        "ecosystem_set": frozenset({"go", "web"}),
        "language_alias": "go",
        "project_type_norm": "platform",
        "topics_set": frozenset({"database", "auth", "sqlite"}),
    }
    cand_meta = {
        "language": "Go",
        "project_type": "platform",
        "replaces": ["firebase"],
    }

    why, relationship = _generate_similar_why(
        target_cache,
        target_meta,
        cand_cache,
        cand_meta,
        "pocketbase/pocketbase",
        0.58,
    )

    assert relationship == "related_project"
    assert any("Target explicitly references this as a related project" in r for r in why)
    assert any("Shared ecosystem: web" in r for r in why)
    assert any("Same project type: platform" in r for r in why)
    assert any("Shared topics: auth, database" in r for r in why)
    assert any("Moderate embedding similarity" in r for r in why)


def test_find_similar_filters():
    entries = [
        {
            "repo_id": "org/target",
            "vector": [1.0, 0.0],
            "metadata": {
                "name": "target",
                "language": "Python",
                "stars": 5000,
                "license": "MIT",
                "project_type": "framework",
                "ecosystem": ["python"],
            },
        },
        {
            "repo_id": "org/python_mit",
            "vector": [0.95, 0.31],
            "metadata": {
                "name": "python_mit",
                "language": "Python",
                "stars": 10000,
                "license": "MIT",
                "project_type": "framework",
                "ecosystem": ["python"],
            },
        },
        {
            "repo_id": "org/python_gpl",
            "vector": [0.98, 0.19],
            "metadata": {
                "name": "python_gpl",
                "language": "Python",
                "stars": 15000,
                "license": "GPL-3.0",
                "project_type": "framework",
                "ecosystem": ["python"],
            },
        },
        {
            "repo_id": "org/rust_mit",
            "vector": [0.99, 0.14],
            "metadata": {
                "name": "rust_mit",
                "language": "Rust",
                "stars": 8000,
                "license": "MIT",
                "project_type": "framework",
                "ecosystem": ["cargo"],
            },
        },
    ]

    prepared = prepare_index(_build_test_index(entries))

    # 1. Filter by language alias 'py'
    res_py = find_similar_prepared("org/target", prepared, filters={"language": "py"})
    py_ids = [r["repo_id"] for r in res_py["results"]]
    assert "org/python_mit" in py_ids
    assert "org/python_gpl" in py_ids
    assert "org/rust_mit" not in py_ids

    # 2. Filter by license 'mit'
    res_mit = find_similar_prepared("org/target", prepared, filters={"license": "mit"})
    mit_ids = [r["repo_id"] for r in res_mit["results"]]
    assert "org/python_mit" in mit_ids
    assert "org/rust_mit" in mit_ids
    assert "org/python_gpl" not in mit_ids

    # 3. Filter by min_stars 12000
    res_stars = find_similar_prepared("org/target", prepared, filters={"min_stars": 12000})
    assert len(res_stars["results"]) == 1
    assert res_stars["results"][0]["repo_id"] == "org/python_gpl"


def test_find_similar_target_resolution_and_missing_error():
    entries = [
        {
            "repo_id": "fastapi/fastapi",
            "vector": [1.0, 0.0],
            "metadata": {
                "name": "fastapi",
                "aliases": ["FastAPI"],
            },
        },
        {
            "repo_id": "encode/starlette",
            "vector": [0.9, 0.43],
            "metadata": {"name": "starlette"},
        },
    ]
    prepared = prepare_index(_build_test_index(entries))

    # Exact match
    assert _resolve_repo_index("fastapi/fastapi", prepared) == 0
    # Case insensitive
    assert _resolve_repo_index("FastAPI/FastAPI", prepared) == 0
    # Suffix/name only
    assert _resolve_repo_index("fastapi", prepared) == 0

    # Missing repo raises ValueError
    with pytest.raises(ValueError, match="Repository 'unknown/repo' not found"):
        find_similar_prepared("unknown/repo", prepared)


def test_find_similar_with_public_api_and_dict_index():
    entries = [
        {"repo_id": "a/a", "vector": [1.0, 0.0], "metadata": {"name": "a"}},
        {"repo_id": "b/b", "vector": [0.8, 0.6], "metadata": {"name": "b"}},
    ]
    index_dict = _build_test_index(entries)

    # Calling public API with dict index
    res = find_similar("a/a", index_dict, top_k=5)
    assert res["target_repo_id"] == "a/a"
    assert len(res["results"]) == 1
    assert res["results"][0]["repo_id"] == "b/b"


def test_find_similar_empty_or_single_candidate():
    entries = [{"repo_id": "solo/solo", "vector": [1.0, 0.0], "metadata": {"name": "solo"}}]
    prepared = prepare_index(_build_test_index(entries))
    res = find_similar_prepared("solo/solo", prepared)
    assert res["results"] == []
    assert res["total_candidates"] == 0
