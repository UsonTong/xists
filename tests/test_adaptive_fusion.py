"""Comprehensive unit and regression tests for adaptive intent-guided hybrid fusion."""

import pytest

from xists.mcp_server import create_server
from xists.records import RECORD_SCHEMA_VERSION
from xists.search.embed import EMBEDDING_INPUT_VERSION, EmbeddingConfig
from xists.search.index import INDEX_VERSION
from xists.search.query import (
    _extract_alternative_target,
    _query_intent,
    _resolve_channel_weights,
    rank,
    rank_many,
)

CONFIG = EmbeddingConfig(
    api_key="test-key",
    base_url="http://localhost:8000/v1",
    model="bge-m3",
)


def make_test_index(vectors: list[dict]):
    return {
        "index_version": INDEX_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
        "embedding_model": "bge-m3",
        "dimension": 2,
        "record_count": len(vectors),
        "vectors": vectors,
    }


def test_intent_classification_and_target_extraction():
    # Exact name queries
    assert _query_intent("uv")["type"] == "exact_name"
    assert _query_intent("ruff")["type"] == "exact_name"
    assert _query_intent("astral-sh/uv")["type"] == "exact_name"

    # Alternative queries
    assert _query_intent("open source alternative to redis")["type"] == "alternative"
    assert _extract_alternative_target("open source alternative to redis") == "redis"
    assert _extract_alternative_target("tools like fzf") == "fzf"
    assert _extract_alternative_target("celery alternatives") == "celery"
    assert _extract_alternative_target("replace react with vue") == "react"

    # Functional and domain queries
    assert (
        _query_intent("high throughput distributed asynchronous actor runtime")["type"]
        == "functional"
    )
    assert _query_intent("web api framework in python")["type"] in {"domain", "functional"}


def test_channel_weight_resolution_defaults_and_overrides():
    # Defaults by intent
    exact_weights = _resolve_channel_weights("exact_name")
    assert exact_weights["dense_weight"] == 0.20
    assert exact_weights["sparse_weight"] == 0.80
    assert exact_weights["sparse_damping"] == 20.0
    assert exact_weights["dense_damping"] == 60.0

    func_weights = _resolve_channel_weights("functional")
    assert func_weights["dense_weight"] == 0.80
    assert func_weights["sparse_weight"] == 0.20
    assert func_weights["dense_damping"] == 30.0
    assert func_weights["sparse_damping"] == 60.0

    alt_weights = _resolve_channel_weights("alternative")
    assert alt_weights["dense_weight"] == 0.70
    assert alt_weights["sparse_weight"] == 0.30

    # Explicit overrides
    explicit_both = _resolve_channel_weights("exact_name", dense_weight=0.9, sparse_weight=0.1)
    assert explicit_both["dense_weight"] == 0.9
    assert explicit_both["sparse_weight"] == 0.1

    explicit_dense_only = _resolve_channel_weights("functional", dense_weight=0.6)
    assert explicit_dense_only["dense_weight"] == 0.6
    assert explicit_dense_only["sparse_weight"] == pytest.approx(0.4)

    explicit_sparse_only = _resolve_channel_weights("functional", sparse_weight=0.75)
    assert explicit_sparse_only["sparse_weight"] == 0.75
    assert explicit_dense_only["dense_weight"] == pytest.approx(0.6)


def test_exact_name_intent_favors_lexical_winner_over_dense_confuser():
    index = make_test_index(
        [
            {
                "repo_id": "astral-sh/uv",
                "vector": [0.6, 0.8],
                "metadata": {
                    "name": "uv",
                    "summary": "An extremely fast Python package and project manager.",
                    "topics": ["python", "packaging"],
                },
            },
            {
                "repo_id": "pypa/pip",
                "vector": [0.95, 0.3122],
                "metadata": {
                    "name": "pip",
                    "summary": "The PyPA recommended tool for installing Python packages.",
                    "topics": ["python", "pypi"],
                },
            },
        ]
    )

    # Query 'uv': dense embedding favors pip [0.95 vs 0.60], but exact name routing gives 80% weight to BM25
    result = rank(
        "uv",
        index,
        CONFIG,
        ranking_strategy="hybrid",
        embed=lambda c, q: [0.95, 0.3122],
    )

    assert result["abstained"] is False
    top = result["results"][0]
    assert top["repo_id"] == "astral-sh/uv"
    assert top["ranking_evidence"]["intent_type"] == "exact_name"
    assert top["ranking_evidence"]["sparse_weight"] == 0.8
    assert top["ranking_evidence"]["dense_weight"] == 0.2
    assert top["ranking_evidence"]["fusion"] == "adaptive_weighted_rrf"
    assert any("sparse priority" in reason or "exact" in reason for reason in top["why"])


def test_alternative_intent_penalizes_target_and_promotes_replacement():
    index = make_test_index(
        [
            {
                "repo_id": "antirez/redis",
                "vector": [0.9, 0.4358],
                "metadata": {
                    "name": "redis",
                    "summary": "In-memory database that persists on disk.",
                    "topics": ["database", "redis", "in-memory"],
                },
            },
            {
                "repo_id": "dragonflydb/dragonfly",
                "vector": [0.85, 0.5267],
                "metadata": {
                    "name": "dragonfly",
                    "summary": "A modern replacement for Redis and Memcached.",
                    "replaces": ["redis", "memcached"],
                    "topics": ["database", "in-memory"],
                },
            },
            {
                "repo_id": "valkey-io/valkey",
                "vector": [0.84, 0.5425],
                "metadata": {
                    "name": "valkey",
                    "summary": "A flexible distributed key-value datastore.",
                    "replaces": ["redis"],
                    "topics": ["key-value", "database"],
                },
            },
        ]
    )

    # Query "open source alternative to redis"
    # antirez/redis would normally win on BM25 ('redis') and dense, but should be down-ranked because it's the target!
    result = rank(
        "open source alternative to redis",
        index,
        CONFIG,
        ranking_strategy="hybrid",
        embed=lambda c, q: [0.9, 0.4358],
    )

    assert result["abstained"] is False
    results = result["results"]
    assert len(results) == 3

    # The declared replacements (dragonfly / valkey) should beat redis
    top_ids = [r["repo_id"] for r in results]
    assert "antirez/redis" != top_ids[0]
    assert top_ids[0] in {"dragonflydb/dragonfly", "valkey-io/valkey"}

    # Verify diagnostics and explanation
    redis_item = next(r for r in results if r["repo_id"] == "antirez/redis")
    assert redis_item["diagnostics"].get("alternative_target_penalty") == "redis"
    assert any("down-ranked target" in w for w in redis_item["why"])

    dragonfly_item = next(r for r in results if r["repo_id"] == "dragonflydb/dragonfly")
    assert dragonfly_item["diagnostics"].get("replaces_promotion") == "redis"
    assert any("promoted declared replacement" in w for w in dragonfly_item["why"])


def test_functional_intent_prioritizes_dense_channel():
    index = make_test_index(
        [
            {
                "repo_id": "tokio-rs/tokio",
                "vector": [0.95, 0.3122],
                "metadata": {
                    "name": "tokio",
                    "summary": "A runtime for writing reliable, asynchronous, and slim applications with Rust.",
                    "topics": ["async", "rust"],
                },
            },
            {
                "repo_id": "random/keyword-stuffer",
                "vector": [0.4, 0.9165],
                "metadata": {
                    "name": "keyword-stuffer",
                    "summary": "High throughput distributed event driven fast queue in rust python memory.",
                    "topics": ["queue", "distributed", "throughput"],
                },
            },
        ]
    )

    query = "high throughput distributed asynchronous actor runtime"
    result = rank(
        query,
        index,
        CONFIG,
        ranking_strategy="hybrid",
        embed=lambda c, q: [0.95, 0.3122],  # favors tokio
    )

    assert result["abstained"] is False
    assert result["results"][0]["repo_id"] == "tokio-rs/tokio"
    evidence = result["results"][0]["ranking_evidence"]
    assert evidence["intent_type"] == "functional"
    assert evidence["dense_weight"] == 0.8
    assert evidence["sparse_weight"] == 0.2


def test_explicit_weights_override_intent_routing():
    index = make_test_index(
        [
            {
                "repo_id": "tool/a",
                "vector": [0.9, 0.4358],
                "metadata": {"name": "a", "summary": "Semantic favorite"},
            },
            {
                "repo_id": "tool/b",
                "vector": [0.2, 0.9797],
                "metadata": {"name": "b", "summary": "BM25 keyword target match"},
            },
        ]
    )

    # Force 100% sparse weight
    sparse_forced = rank(
        "target match",
        index,
        CONFIG,
        ranking_strategy="hybrid",
        dense_weight=0.0,
        sparse_weight=1.0,
        embed=lambda c, q: [0.9, 0.4358],
    )
    assert sparse_forced["results"][0]["repo_id"] == "tool/b"
    assert sparse_forced["results"][0]["ranking_evidence"]["sparse_weight"] == 1.0
    assert sparse_forced["results"][0]["ranking_evidence"]["dense_weight"] == 0.0

    # Force 100% dense weight
    dense_forced = rank(
        "target match",
        index,
        CONFIG,
        ranking_strategy="hybrid",
        dense_weight=1.0,
        sparse_weight=0.0,
        embed=lambda c, q: [0.9, 0.4358],
    )
    assert dense_forced["results"][0]["repo_id"] == "tool/a"
    assert dense_forced["results"][0]["ranking_evidence"]["sparse_weight"] == 0.0
    assert dense_forced["results"][0]["ranking_evidence"]["dense_weight"] == 1.0


def test_single_and_batch_parity_with_adaptive_fusion():
    index = make_test_index(
        [
            {
                "repo_id": "astral-sh/ruff",
                "vector": [1.0, 0.0],
                "metadata": {"name": "ruff", "summary": "An extremely fast Python linter."},
            },
            {
                "repo_id": "astral-sh/uv",
                "vector": [0.0, 1.0],
                "metadata": {"name": "uv", "summary": "An extremely fast Python package manager."},
            },
        ]
    )

    queries = ["ruff", "fast python package manager", "open source alternative to pip"]
    embeds = {
        "ruff": [1.0, 0.0],
        "fast python package manager": [0.0, 1.0],
        "open source alternative to pip": [0.0, 1.0],
    }

    single_results = [
        rank(
            q,
            index,
            CONFIG,
            ranking_strategy="hybrid",
            embed=lambda c, query: embeds[query],
        )
        for q in queries
    ]

    batch_results = rank_many(
        queries,
        index,
        CONFIG,
        ranking_strategy="hybrid",
        embed_many=lambda c, qs, **kw: [embeds[q] for q in qs],
    )

    assert len(single_results) == len(batch_results)
    for s_res, b_res in zip(single_results, batch_results):
        assert [r["repo_id"] for r in s_res["results"]] == [r["repo_id"] for r in b_res["results"]]
        for r1, r2 in zip(s_res["results"], b_res["results"]):
            assert r1["score"] == pytest.approx(r2["score"], abs=1e-5)
            assert r1["ranking_evidence"] == r2["ranking_evidence"]


def test_mcp_server_forwards_channel_weights(monkeypatch):
    captured = {}

    def fake_search(query, index, **kwargs):
        captured.update(kwargs)
        return {
            "query": query,
            "abstained": False,
            "results": [
                {
                    "repo_id": "test/repo",
                    "score": 0.95,
                    "confidence": "high_confidence",
                    "ranking_evidence": {"fusion": "adaptive_weighted_rrf"},
                }
            ],
        }

    monkeypatch.setattr("xists.mcp_server.public_search", fake_search)

    server = create_server(
        make_test_index([{"repo_id": "test/repo", "vector": [1.0, 0.0], "metadata": {}}]),
        CONFIG,
    )
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
    search_tool = tools["search_projects"]

    search_tool.fn(query="test query", dense_weight=0.35, sparse_weight=0.65)
    assert captured["dense_weight"] == 0.35
    assert captured["sparse_weight"] == 0.65


def test_cross_language_conflict_penalizes_source_language():
    index = make_test_index(
        [
            {
                "repo_id": "pallets/flask",
                "vector": [0.8, 0.6],
                "metadata": {
                    "name": "flask",
                    "summary": "The Python micro framework for building web applications.",
                    "language": "Python",
                    "topics": ["python", "web", "microframework"],
                },
            },
            {
                "repo_id": "gin-gonic/gin",
                "vector": [0.75, 0.66],
                "metadata": {
                    "name": "gin",
                    "summary": "Gin is a HTTP web framework written in Go (Golang).",
                    "language": "Go",
                    "topics": ["go", "golang", "web", "framework"],
                },
            },
        ]
    )

    # Query asking for a port of flask in Go
    result = rank(
        "port of flask written in Go",
        index,
        CONFIG,
        ranking_strategy="hybrid",
        embed=lambda c, q: [0.78, 0.62],
    )

    assert result["abstained"] is False
    results = result["results"]
    assert len(results) >= 1
    # Gin (Go) should beat Flask (Python) because Flask is penalized for cross-language conflict
    assert results[0]["repo_id"] == "gin-gonic/gin"
    flask_entry = next((r for r in results if r["repo_id"] == "pallets/flask"), None)
    if flask_entry:
        assert "cross_language_penalty" in flask_entry["diagnostics"]
        assert any("cross-language conflict" in w for w in flask_entry["why"])


def test_canonical_distractor_defense_breaks_ties_by_stars():
    index = make_test_index(
        [
            {
                "repo_id": "vuejs-templates/webpack",
                "vector": [0.9, 0.4358],
                "metadata": {
                    "name": "webpack",
                    "summary": "A full-featured Webpack + vue-loader setup with hot reload.",
                    "stars": 9626,
                    "language": "JavaScript",
                },
            },
            {
                "repo_id": "webpack/webpack",
                "vector": [0.89, 0.4559],
                "metadata": {
                    "name": "webpack",
                    "summary": "A bundler for javascript and friends. Packs many modules into a few bundled assets.",
                    "stars": 63800,
                    "language": "JavaScript",
                },
            },
        ]
    )

    # Searching exact name "webpack" should rank canonical webpack/webpack #1 over the template
    res1 = rank(
        "webpack", index, CONFIG, ranking_strategy="hybrid", embed=lambda c, q: [0.9, 0.4358]
    )
    assert res1["results"][0]["repo_id"] == "webpack/webpack"

    # Searching "github webpack" should also recognize exact lookup and rank canonical repo #1
    res2 = rank(
        "github webpack", index, CONFIG, ranking_strategy="hybrid", embed=lambda c, q: [0.9, 0.4358]
    )
    assert res2["results"][0]["repo_id"] == "webpack/webpack"


def test_out_of_domain_abstention_on_fictional_queries():
    index = make_test_index(
        [
            {
                "repo_id": "astral-sh/uv",
                "vector": [0.8, 0.6],
                "metadata": {
                    "name": "uv",
                    "summary": "An extremely fast Python package and project manager.",
                    "language": "Rust",
                },
            },
            {
                "repo_id": "expressjs/express",
                "vector": [0.2, 0.9797],
                "metadata": {
                    "name": "express",
                    "summary": "Fast, unopinionated, minimalist web framework for node.",
                    "language": "JavaScript",
                },
            },
        ]
    )

    # Impossible fictional query with zero semantic connection and no keyword matches
    result = rank(
        "superconducting cold fusion reactor firmware written in bash",
        index,
        CONFIG,
        ranking_strategy="hybrid",
        embed=lambda c, q: [0.0, 0.0],
    )
    assert result["abstained"] is True
    assert len(result["results"]) == 0
