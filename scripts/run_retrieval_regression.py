"""Run the committed public retrieval regression fixture without a model endpoint."""

from __future__ import annotations

import json
from pathlib import Path

from xists.eval.schema import load_dataset
from xists.search.embed import EmbeddingConfig
from xists.search.index import load_index
from xists.search.query import rank


ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "examples" / "retrieval-regression"

QUERY_VECTORS = {
    "express-lite": [0.0, 1.0, 0.0, 0.0],
    "open source firebase alternative": [1.0, 0.0, 0.0, 0.0],
    "Node.js web framework": [0.0, 1.0, 0.0, 0.0],
    "self hosted developer backend platform": [1.0, 0.0, 0.0, 0.0],
    "自托管大语言模型应用界面": [0.0, 0.0, 0.0, 1.0],
    "quantum chemistry notebook for protein folding games": [0.0, 0.0, 0.0, 0.0],
}


def run_regression() -> dict[str, object]:
    cases = load_dataset(FIXTURE_DIR / "cases.json")
    index = load_index(FIXTURE_DIR / "index.json")
    config = EmbeddingConfig(
        api_key="fixture",
        base_url="https://example.invalid/v1",
        model="fixture/regression-4d",
    )
    results: list[dict[str, object]] = []
    for case in cases["cases"]:
        query = case["query"]
        result = rank(query, index, config, top_k=3, embed=lambda _config, _query: QUERY_VECTORS[_query])
        expected = case["expected_repo_id"]
        if "no-result" in case["tags"]:
            passed = result["abstained"] is True and result["results"] == []
        else:
            passed = (
                result["abstained"] is False
                and bool(result["results"])
                and result["results"][0]["repo_id"] == expected
            )
        results.append(
            {
                "id": case["id"],
                "query": query,
                "expected_repo_id": expected,
                "top_result_repo_id": result["results"][0]["repo_id"] if result["results"] else None,
                "abstained": result["abstained"],
                "passed": passed,
            }
        )
    return {
        "fixture": str(FIXTURE_DIR.relative_to(ROOT)),
        "case_count": len(results),
        "passed": all(item["passed"] for item in results),
        "results": results,
    }


def main() -> int:
    report = run_regression()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
