import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "compare_retrieval_experiments.py"
SPEC = importlib.util.spec_from_file_location("compare_retrieval_experiments", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_compare_reports_includes_slice_and_latency_metrics():
    report = {
        "dataset_name": "sample",
        "ranking_strategy": "rerank",
        "rerank_candidate_limit": 50,
        "duration_seconds": 1.2,
        "results": [
            {
                "tags": ["domain-web", "category-functional", "language-en"],
                "acceptable_match": True,
                "acceptable_rank": 1,
                "top_result_confidence": "high_confidence",
                "abstained": False,
                "latency_ms": 20.0,
            },
            {
                "tags": ["domain-data", "category-no-result", "language-zh"],
                "acceptable_match": False,
                "acceptable_rank": None,
                "top_result_confidence": None,
                "abstained": True,
                "latency_ms": 40.0,
            },
        ],
    }

    result = MODULE.compare_reports({"rerank": report})

    experiment = result["experiments"]["rerank"]
    assert experiment["ranking_strategy"] == "rerank"
    assert experiment["latency"] == {"p50_ms": 20.0, "p95_ms": 40.0}
    assert experiment["slices"]["domain"]["web"]["recall_at_1"] == 1.0
