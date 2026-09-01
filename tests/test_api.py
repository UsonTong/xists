import json

import pytest

from xists.api import compare_projects, find_similar, load_index, search
from xists.records import RECORD_SCHEMA_VERSION
from xists.search.embed import EMBEDDING_INPUT_VERSION, EmbeddingConfig, EmbeddingError
from xists.search.query import IndexMismatchError

CONFIG = EmbeddingConfig(
    api_key="test-key",
    base_url="https://embeddings.example/v1",
    model="test-embedding",
)


def make_index(*, model: str = "test-embedding"):
    return {
        "index_version": 3,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_model": model,
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
        "dimension": 2,
        "record_count": 2,
        "vectors": [
            {
                "repo_id": "winner/repo",
                "vector": [1.0, 0.0],
                "metadata": {"name": "winner", "summary": "A useful project"},
            },
            {
                "repo_id": "other/repo",
                "vector": [0.0, 1.0],
                "metadata": {"name": "other"},
            },
        ],
    }


def test_load_index_accepts_str_and_path_and_returns_document(tmp_path):
    document = make_index()
    path = tmp_path / "index.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    assert load_index(str(path)) == document
    assert load_index(path) == document


def test_search_uses_explicit_configuration_without_environment_or_cli(monkeypatch):
    calls = []

    def fake_embeddings(config, inputs, *, timeout=60, input_type=None):
        calls.append((config, inputs, input_type))
        return [[1.0, 0.0] for _ in inputs]

    monkeypatch.setattr("xists.search.embed.call_embeddings", fake_embeddings)

    result = search("useful project", make_index(), embedding_config=CONFIG, top_k=1)

    assert result["results"][0]["repo_id"] == "winner/repo"
    assert result["considered"] == 2
    assert calls == [(CONFIG, ["useful project"], "query")]


def test_search_preserves_core_model_mismatch_error():
    with pytest.raises(IndexMismatchError, match="built with embedding model"):
        search("query", make_index(model="different-model"), embedding_config=CONFIG)


def test_search_preserves_actionable_embedding_endpoint_error(monkeypatch):
    def fail_embeddings(*_args, **_kwargs):
        raise EmbeddingError("Embedding endpoint request failed; check endpoint credentials.")

    monkeypatch.setattr("xists.search.embed.call_embeddings", fail_embeddings)

    with pytest.raises(EmbeddingError, match="check endpoint credentials"):
        search("query", make_index(), embedding_config=CONFIG)


def test_search_forwards_explicit_optional_ranking_arguments(monkeypatch):
    captured = {}

    def fake_rank(query, index, config, **kwargs):
        captured.update(query=query, index=index, config=config, **kwargs)
        return {"query": query, "abstained": True, "results": []}

    monkeypatch.setattr("xists.api.rank", fake_rank)

    def reranker(_query: str, _documents: list[str]) -> list[float]:
        return [0.5]

    result = search(
        "query",
        make_index(),
        embedding_config=CONFIG,
        top_k=3,
        ranking_strategy="rerank",
        rerank=reranker,
        rerank_candidate_limit=7,
        exploratory_threshold=0.4,
        rerank_abstain_threshold=0.2,
        confidence_calibration="evidence-v1",
        query_variants=["query", "canonical query"],
        rerank_query="canonical query",
    )

    assert result["abstained"] is True
    assert captured == {
        "query": "query",
        "index": make_index(),
        "config": CONFIG,
        "top_k": 3,
        "ranking_strategy": "rerank",
        "rerank": reranker,
        "rerank_candidate_limit": 7,
        "exploratory_threshold": 0.4,
        "rerank_abstain_threshold": 0.2,
        "confidence_calibration": "evidence-v1",
        "query_variants": ["query", "canonical query"],
        "rerank_query": "canonical query",
        "filters": None,
        "dense_weight": None,
        "sparse_weight": None,
    }


def test_search_forwards_channel_weights(monkeypatch):
    captured = {}

    def fake_rank(query, index, config, **kwargs):
        captured.update(query=query, index=index, config=config, **kwargs)
        return {"query": query, "abstained": True, "results": []}

    monkeypatch.setattr("xists.api.rank", fake_rank)

    result = search(
        "query",
        make_index(),
        embedding_config=CONFIG,
        ranking_strategy="hybrid",
        dense_weight=0.3,
        sparse_weight=0.7,
    )

    assert result["abstained"] is True
    assert captured["dense_weight"] == 0.3
    assert captured["sparse_weight"] == 0.7


def test_search_accepts_hybrid_ranking_strategy(monkeypatch):
    def fake_embeddings(config, inputs, *, timeout=60, input_type=None):
        return [[1.0, 0.0] for _ in inputs]

    monkeypatch.setattr("xists.search.embed.call_embeddings", fake_embeddings)

    result = search(
        "useful project",
        make_index(),
        embedding_config=CONFIG,
        ranking_strategy="hybrid",
        top_k=2,
    )
    assert result["abstained"] is False
    assert result["results"][0]["repo_id"] == "winner/repo"
    assert "bm25_score" in result["results"][0]


def test_search_forwards_filters_to_rank(monkeypatch):
    captured = {}

    def fake_rank(query, index, config, **kwargs):
        captured.update(kwargs)
        return {
            "query": query,
            "abstained": False,
            "results": [],
            "filters": kwargs.get("filters"),
        }

    monkeypatch.setattr("xists.api.rank", fake_rank)

    filters = {"language": "python", "min_stars": 1000, "license": "mit"}
    result = search(
        "web framework",
        make_index(),
        embedding_config=CONFIG,
        filters=filters,
    )
    assert captured.get("filters") == filters
    assert result["filters"] == filters


def test_api_find_similar_accepts_dict_and_prepared_index():
    index_dict = make_index()
    # 1. Dict input
    res_dict = find_similar("winner/repo", index_dict, top_k=5)
    assert res_dict["target_repo_id"] == "winner/repo"
    assert len(res_dict["results"]) == 1
    assert res_dict["results"][0]["repo_id"] == "other/repo"

    # 2. PreparedIndex input
    from xists.search.query import prepare_index

    prepared = prepare_index(index_dict)
    res_prep = find_similar("winner/repo", prepared, top_k=5)
    assert res_prep["target_repo_id"] == "winner/repo"
    assert len(res_prep["results"]) == 1
    assert res_prep["results"][0]["repo_id"] == "other/repo"


def test_api_compare_projects_accepts_dict_and_prepared_index():
    index_dict = make_index()
    # 1. Dict input
    res_dict = compare_projects(["winner/repo", "other/repo"], index_dict)
    assert res_dict["repo_ids"] == ["winner/repo", "other/repo"]
    assert res_dict["matrix"]["winner/repo"]["other/repo"] == 0.0

    # 2. PreparedIndex input
    from xists.search.query import prepare_index

    prepared = prepare_index(index_dict)
    res_prep = compare_projects(["winner/repo", "other/repo"], prepared)
    assert res_prep["repo_ids"] == ["winner/repo", "other/repo"]
    assert res_prep["matrix"]["winner/repo"]["other/repo"] == 0.0
