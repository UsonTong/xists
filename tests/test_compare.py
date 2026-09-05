"""Unit and integration tests for multi-project side-by-side comparison."""

from __future__ import annotations

import pytest

from xists.api import compare_projects
from xists.search.compare import _synthesize_compare_analysis, compare_projects_prepared
from xists.search.index import INDEX_VERSION
from xists.search.query import prepare_index


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


def test_compare_validation_bounds_and_duplicates():
    entries = [
        {"repo_id": "org/a", "vector": [1.0, 0.0], "metadata": {"name": "a"}},
        {"repo_id": "org/b", "vector": [0.0, 1.0], "metadata": {"name": "b"}},
    ]
    prepared = prepare_index(_build_test_index(entries))

    # Single repo rejected
    with pytest.raises(ValueError, match="between 2 and 5"):
        compare_projects_prepared(["org/a"], prepared)

    # 6 repos rejected
    with pytest.raises(ValueError, match="between 2 and 5"):
        compare_projects_prepared(["a", "b", "c", "d", "e", "f"], prepared)

    # Duplicate repos rejected
    with pytest.raises(ValueError, match="Duplicate repository identifier"):
        compare_projects_prepared(["org/a", "ORG/A"], prepared)

    # Missing repo rejected with informative message
    with pytest.raises(ValueError, match="Repositories not found in index: org/missing"):
        compare_projects_prepared(["org/a", "org/missing"], prepared)


def test_compare_pairwise_matrix_and_structure():
    entries = [
        {
            "repo_id": "vllm-project/vllm",
            "vector": [1.0, 0.0],
            "metadata": {
                "name": "vllm",
                "summary": "High-throughput LLM inference",
                "language": "Python",
                "stars": 80000,
                "license": "Apache-2.0",
                "project_type": "library",
                "ecosystem": ["python", "llm"],
                "capabilities": ["High-throughput inference", "PagedAttention"],
                "use_cases": ["Production model serving"],
                "not_for": ["Low-memory edge devices"],
                "related_projects": ["llama.cpp"],
            },
        },
        {
            "repo_id": "ggml-org/llama.cpp",
            "vector": [0.6, 0.8],  # Cosine sim with vLLM is 0.600
            "metadata": {
                "name": "llama.cpp",
                "summary": "LLM inference in C/C++",
                "language": "C++",
                "stars": 120000,
                "license": "MIT",
                "project_type": "library",
                "ecosystem": ["cpp", "llm"],
                "capabilities": ["CPU inference", "Quantization"],
                "use_cases": ["Local device inference"],
                "not_for": ["Distributed multi-node serving"],
                "replaces": ["vLLM"],
            },
        },
        {
            "repo_id": "ollama/ollama",
            "vector": [0.8, 0.6],  # Cosine sim with vLLM is 0.800, with llama.cpp is 0.960
            "metadata": {
                "name": "ollama",
                "summary": "Get up and running with LLMs locally",
                "language": "Go",
                "stars": 170000,
                "license": "MIT",
                "project_type": "runtime",
                "ecosystem": ["go", "llm"],
                "capabilities": ["CLI management", "REST API"],
                "use_cases": ["Local developer setup"],
                "not_for": ["Cluster orchestration"],
            },
        },
    ]

    prepared = prepare_index(_build_test_index(entries))

    # Compare 3 projects
    res = compare_projects_prepared(
        ["vllm-project/vllm", "ggml-org/llama.cpp", "ollama/ollama"],
        prepared,
    )

    assert res["repo_ids"] == [
        "vllm-project/vllm",
        "ggml-org/llama.cpp",
        "ollama/ollama",
    ]
    assert len(res["projects"]) == 3

    # Check matrix diagonal and symmetry
    matrix = res["matrix"]
    assert matrix["vllm-project/vllm"]["vllm-project/vllm"] == 1.0
    assert matrix["ggml-org/llama.cpp"]["ggml-org/llama.cpp"] == 1.0
    assert matrix["ollama/ollama"]["ollama/ollama"] == 1.0

    assert matrix["vllm-project/vllm"]["ggml-org/llama.cpp"] == 0.6
    assert matrix["ggml-org/llama.cpp"]["vllm-project/vllm"] == 0.6

    assert matrix["vllm-project/vllm"]["ollama/ollama"] == 0.8
    assert matrix["ollama/ollama"]["vllm-project/vllm"] == 0.8

    assert matrix["ggml-org/llama.cpp"]["ollama/ollama"] == 0.96
    assert matrix["ollama/ollama"]["ggml-org/llama.cpp"] == 0.96

    # Pairwise list
    assert len(res["pairwise"]) == 3
    pair_dict = {(p["repo_a"], p["repo_b"]): p["similarity"] for p in res["pairwise"]}
    assert pair_dict[("vllm-project/vllm", "ggml-org/llama.cpp")] == 0.6
    assert pair_dict[("vllm-project/vllm", "ollama/ollama")] == 0.8
    assert pair_dict[("ggml-org/llama.cpp", "ollama/ollama")] == 0.96


def test_compare_analysis_commonalities_and_differentiators():
    projects = [
        {
            "repo_id": "fastapi/fastapi",
            "name": "fastapi",
            "language": "Python",
            "stars": 80000,
            "license": "MIT",
            "project_type": "framework",
            "ecosystem": ["python", "web"],
            "capabilities": ["Auto OpenAPI", "Pydantic validation", "Async routing"],
            "use_cases": ["Building web APIs", "Microservices"],
            "not_for": ["Monolithic MVC apps"],
            "topics": ["python", "api"],
            "replaces": [],
            "related_projects": ["encode/starlette"],
        },
        {
            "repo_id": "encode/starlette",
            "name": "starlette",
            "language": "Python",
            "stars": 12000,
            "license": "BSD-3-Clause",
            "project_type": "framework",
            "ecosystem": ["python", "web"],
            "capabilities": ["ASGI toolkit", "Async routing"],
            "use_cases": ["Lightweight ASGI services", "Microservices"],
            "not_for": ["Full-stack templated sites"],
            "topics": ["python", "asgi"],
            "replaces": [],
            "related_projects": [],
        },
    ]

    analysis = _synthesize_compare_analysis(projects)

    # Commonalities
    assert analysis["shared_ecosystems"] == ["python", "web"]
    assert analysis["shared_languages"] == ["Python"]
    assert "async routing" in analysis["shared_capabilities"]
    assert analysis["shared_topics"] == ["python"]

    # Differentiators
    fastapi_diff = analysis["differentiators"]["fastapi/fastapi"]
    assert "Auto OpenAPI" in fastapi_diff["unique_capabilities"]
    assert "Pydantic validation" in fastapi_diff["unique_capabilities"]
    assert "Building web APIs" in fastapi_diff["unique_use_cases"]
    assert "Monolithic MVC apps" in fastapi_diff["not_for"]
    assert fastapi_diff["stars"] == 80000
    assert fastapi_diff["license"] == "MIT"

    starlette_diff = analysis["differentiators"]["encode/starlette"]
    assert "ASGI toolkit" in starlette_diff["unique_capabilities"]
    assert "Lightweight ASGI services" in starlette_diff["unique_use_cases"]
    assert "Full-stack templated sites" in starlette_diff["not_for"]

    # Direct links
    assert len(analysis["direct_links"]) == 1
    link = analysis["direct_links"][0]
    assert link["source"] == "fastapi/fastapi"
    assert link["target"] == "encode/starlette"
    assert link["relation"] == "related_project"


def test_compare_public_api_with_dict():
    entries = [
        {"repo_id": "a/a", "vector": [1.0, 0.0], "metadata": {"name": "a", "stars": 100}},
        {"repo_id": "b/b", "vector": [0.0, 1.0], "metadata": {"name": "b", "stars": 200}},
    ]
    index_dict = _build_test_index(entries)

    res = compare_projects(["a/a", "b/b"], index_dict)
    assert res["repo_ids"] == ["a/a", "b/b"]
    assert res["matrix"]["a/a"]["b/b"] == 0.0
