import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "diagnose_retrieval_candidates.py"
SPEC = importlib.util.spec_from_file_location("diagnose_retrieval_candidates", MODULE_PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(diagnostic)


def _index():
    return {
        "dimension": 2,
        "vectors": [
            {"repo_id": "first/repo", "metadata": {"description": "first"}, "vector": [1.0, 0.0]},
            {"repo_id": "second/repo", "metadata": {"description": "second"}, "vector": [0.8, 0.6]},
        ],
    }


def test_candidate_diagnostics_uses_stable_semantic_ranks():
    candidates = diagnostic.candidate_diagnostics(_index(), [1.0, 0.0], candidate_limit=2)

    assert [item["repo_id"] for item in candidates] == ["first/repo", "second/repo"]
    assert [item["semantic_rank"] for item in candidates] == [1, 2]
    assert [item["semantic_score"] for item in candidates] == pytest.approx([1.0, 0.8])


def test_rerank_diagnostics_preserves_semantic_order_and_adds_rerank_rank():
    candidates = diagnostic.rerank_diagnostics(
        _index(),
        [1.0, 0.0],
        "query",
        candidate_limit=2,
        reranker_config=object(),
        rerank=lambda _config, _query, documents: [float(len(documents)), 9.0],
    )

    assert [item["repo_id"] for item in candidates] == ["first/repo", "second/repo"]
    assert candidates[0]["semantic_rank"] == 1
    assert candidates[0]["rerank_score"] == 2.0
    assert candidates[0]["rerank_rank"] == 2
    assert candidates[1]["rerank_rank"] == 1
