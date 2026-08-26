import pytest
from pathlib import Path

from xists.records import RECORD_SCHEMA_VERSION, records_validation_report
from xists.search.index import INDEX_VERSION
from xists.starter import (
    get_starter_index_path,
    get_starter_records_path,
    load_starter_index,
    load_starter_records,
    starter_metadata_search,
)


def test_starter_paths_exist_and_are_valid():
    records_path = get_starter_records_path()
    index_path = get_starter_index_path()
    vectors_path = index_path.with_name(f"{index_path.stem}.vectors.npy")

    assert records_path.is_file()
    assert index_path.is_file()
    assert vectors_path.is_file()


def test_starter_records_load_and_pass_schema_validation():
    records = load_starter_records()
    assert len(records) >= 20

    report = records_validation_report(
        records,
        expected_schema_version=RECORD_SCHEMA_VERSION,
        expected_profile_prompt_version=2,
    )
    assert not report["errors"], f"Validation failed: {report['errors']}"
    assert report["quality"]["ok"] == len(records)


def test_starter_index_loads_with_mmap():
    index = load_starter_index(mmap=True)
    assert index.get("index_version") == INDEX_VERSION
    assert index.get("dimension") == 1024
    assert index.get("record_count") >= 20
    assert "_matrix" in index
    assert index["_matrix"].shape == (len(index.get("vectors") or []), 1024)


def test_starter_metadata_search_exact_matches():
    res = starter_metadata_search("fastapi")
    assert not res["abstained"]
    assert len(res["results"]) > 0
    top = res["results"][0]
    assert top["repo_id"] == "fastapi/fastapi"
    assert top["confidence"] == "high"


def test_starter_metadata_search_concept_queries():
    test_cases = [
        ("python web framework", "fastapi/fastapi"),
        ("vector database", "qdrant/qdrant"),
        ("local llm inference", "vllm-project/vllm"),
        ("workflow automation", "n8n-io/n8n"),
        ("code editor", "microsoft/vscode"),
        ("fast python linter", "astral-sh/ruff"),
    ]
    for query, expected_repo in test_cases:
        res = starter_metadata_search(query)
        assert not res["abstained"], f"Abstained on {query}"
        matching_repos = [r["repo_id"] for r in res["results"][:3]]
        assert expected_repo in matching_repos, f"Expected {expected_repo} in top 3 for '{query}', got {matching_repos}"


def test_starter_metadata_search_abstains_on_unrelated_queries():
    res = starter_metadata_search("xyz123nonsensequery_completely_unrelated")
    assert res["abstained"]
    assert len(res["results"]) == 0
