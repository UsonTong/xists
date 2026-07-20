import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "summarize_eval_slices.py"
SPEC = importlib.util.spec_from_file_location("summarize_eval_slices", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_summarize_report_groups_by_domain_category_and_language():
    report = {
        "dataset_name": "sample",
        "results": [
            {
                "tags": ["domain-web", "category-functional", "language-en"],
                "acceptable_match": True,
                "acceptable_rank": 1,
                "top_result_confidence": "high_confidence",
                "abstained": False,
            },
            {
                "tags": ["domain-data", "category-no-result", "language-zh"],
                "acceptable_match": False,
                "acceptable_rank": None,
                "top_result_confidence": None,
                "abstained": True,
            },
        ],
    }

    summary = MODULE.summarize_report(report)

    assert summary["slices"]["domain"]["web"]["recall_at_1"] == 1.0
    assert summary["slices"]["domain"]["data"]["no_result_abstain_rate"] == 1.0
    assert summary["slices"]["language"]["zh"]["no_result_case_count"] == 1
