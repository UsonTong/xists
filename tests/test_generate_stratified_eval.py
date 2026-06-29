import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_stratified_eval.py"
SPEC = importlib.util.spec_from_file_location("generate_stratified_eval", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_record(
    repo_id: str,
    *,
    stars: int = 1500,
    language: str = "Python",
) -> dict:
    return {
        "repo_id": repo_id,
        "name": repo_id.rsplit("/", 1)[-1],
        "github": {
            "description": "Framework for building data workflows and scheduled jobs.",
            "topics": ["workflow", "scheduler", "automation"],
            "language": language,
            "stars": stars,
        },
        "llm_profile": {
            "summary": "Open source workflow orchestration toolkit for data teams.",
            "use_cases": [
                "managing scheduled data pipelines",
                "running repeatable automation jobs",
            ],
            "capabilities": [
                "workflow orchestration",
                "job scheduling",
            ],
            "search_phrases": [
                "workflow orchestration toolkit",
                "scheduled data pipeline automation",
            ],
        },
    }


def test_alternative_template_produces_distinct_queries():
    record = MODULE._repo_record(make_record("acme/workflow-kit"))

    baseline = MODULE.build_cases_for_record(
        record,
        star_tier="star-1k-9k",
        template_style="baseline",
    )
    alternative = MODULE.build_cases_for_record(
        record,
        star_tier="star-1k-9k",
        template_style="alternative",
    )

    baseline_queries = {MODULE._normalize_query(case["query"]) for case in baseline}
    alternative_queries = {MODULE._normalize_query(case["query"]) for case in alternative}

    assert alternative_queries
    assert baseline_queries.isdisjoint(alternative_queries)


def test_generate_dataset_excludes_existing_queries_and_signatures(tmp_path):
    records = [
        make_record("acme/workflow-kit"),
        make_record("acme/workflow-cloud", stars=2200, language="Go"),
    ]
    excluded_dataset = {
        "schema_version": 1,
        "dataset_name": "existing",
        "families": {},
        "cases": [
            {
                "id": "existing-1",
                "query": "Python workflow orchestration toolkit",
                "expected_repo_id": "someone/existing",
            }
        ],
    }
    excluded_path = tmp_path / "existing.json"
    excluded_path.write_text(json.dumps(excluded_dataset), encoding="utf-8")

    dataset = MODULE.generate_dataset(
        records,
        limit=10,
        seed=42,
        template_style="alternative",
        excluded_paths=[excluded_path],
    )

    queries = [MODULE._normalize_query(case["query"]) for case in dataset["cases"]]
    signatures = [MODULE._query_signature(case["query"]) for case in dataset["cases"]]

    assert dataset["dataset_name"] == "xists-full-stratified-2000-alternative"
    assert "python workflow orchestration toolkit" not in queries
    assert MODULE._query_signature("Python workflow orchestration toolkit") not in signatures
    assert len(queries) == len(set(queries))
