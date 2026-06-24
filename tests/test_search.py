import math

import pytest

from xists.search.embed import (
    EmbeddingConfig,
    EmbeddingError,
    EMBEDDING_INPUT_VERSION,
    EmbeddingNotConfiguredError,
    call_embeddings,
    embedding_config_from_env,
    embedding_input_fingerprint,
    embedding_text_from_record,
)
from xists.search.index import build_index
from xists.search.query import (
    IndexMismatchError,
    confidence_bucket,
    cosine_similarity,
    rank,
    rank_many,
)

CONFIG = EmbeddingConfig(api_key="k", base_url="http://localhost/v1", model="bge-m3")


def make_record(repo_id="react/react"):
    return {
        "repo_id": repo_id,
        "name": "react",
        "github": {
            "description": "The library for web and native user interfaces.",
            "topics": ["frontend", "ui"],
        },
        "llm_profile": {
            "summary": "React is a JavaScript UI library.",
            "use_cases": ["building web user interfaces"],
            "capabilities": ["declarative UI rendering"],
            "not_for": ["backend-only services"],
            "search_phrases": ["frontend UI library"],
        },
    }


def test_embedding_config_from_env_requires_all(monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(EmbeddingNotConfiguredError):
        embedding_config_from_env()


def test_embedding_config_from_env_builds(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
    config = embedding_config_from_env()
    assert config.embeddings_url == "http://localhost/v1/embeddings"


def test_embedding_text_excludes_not_for():
    text = embedding_text_from_record(make_record())
    assert "JavaScript UI library" in text
    assert "frontend UI library" in text
    assert "backend-only services" not in text


def test_embedding_text_empty_when_no_signal():
    assert embedding_text_from_record({"repo_id": None}) == ""


def test_embedding_input_fingerprint_changes_with_text():
    record = make_record()
    changed = make_record()
    changed["llm_profile"]["summary"] = "A changed summary."

    assert embedding_input_fingerprint(record) != embedding_input_fingerprint(changed)
    assert embedding_input_fingerprint({"repo_id": None}) is None


def test_call_embeddings_empty_input_returns_empty():
    assert call_embeddings(CONFIG, []) == []


def test_cosine_similarity_basic():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_confidence_bucket():
    assert confidence_bucket(0.9) == "high_confidence"
    assert confidence_bucket(0.4) == "exploratory"
    assert confidence_bucket(0.1) == "abstain"


def test_build_index_with_mock(monkeypatch):
    records = [make_record("react/react"), make_record("vuejs/core")]

    def fake_call(config, inputs, *, timeout=60):
        return [[1.0, 0.0, 0.0] for _ in inputs]

    monkeypatch.setattr("xists.search.index.call_embeddings", fake_call)
    index = build_index(records, CONFIG)
    assert index["embedding_model"] == "bge-m3"
    assert index["embedding_input_version"] == EMBEDDING_INPUT_VERSION
    assert index["dimension"] == 3
    assert index["record_count"] == 2
    assert index["skipped"] == []
    assert index["vectors"][0]["embedding_input_fingerprint"] == embedding_input_fingerprint(records[0])


def test_build_index_skips_empty_records(monkeypatch):
    records = [make_record("react/react"), {"repo_id": "empty/empty"}]

    def fake_call(config, inputs, *, timeout=60):
        return [[1.0, 0.0] for _ in inputs]

    monkeypatch.setattr("xists.search.index.call_embeddings", fake_call)
    index = build_index(records, CONFIG)
    assert index["record_count"] == 1
    assert index["skipped"] == ["empty/empty"]


def test_rank_returns_sorted_results():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {"repo_id": "react/react", "vector": [1.0, 0.0]},
            {"repo_id": "unrelated/repo", "vector": [0.0, 1.0]},
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("frontend ui", index, CONFIG, embed=fake_embed)
    assert result["abstained"] is False
    assert result["results"][0]["repo_id"] == "react/react"
    assert result["results"][0]["confidence"] == "high_confidence"


def test_rank_abstains_when_all_weak():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [{"repo_id": "react/react", "vector": [0.0, 1.0]}],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("frontend ui", index, CONFIG, embed=fake_embed)
    assert result["abstained"] is True
    assert result["results"] == []


def test_rank_rejects_model_mismatch():
    index = {"embedding_model": "other-model", "dimension": 2, "vectors": []}

    def fake_embed(config, query):
        return [1.0, 0.0]

    with pytest.raises(IndexMismatchError):
        rank("x", index, CONFIG, embed=fake_embed)


def test_rank_rejects_dimension_mismatch():
    index = {"embedding_model": "bge-m3", "dimension": 3, "vectors": []}

    def fake_embed(config, query):
        return [1.0, 0.0]

    with pytest.raises(IndexMismatchError):
        rank("x", index, CONFIG, embed=fake_embed)


def test_rank_many_batches_embeddings_and_matches_rank_order():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {"repo_id": "react/react", "vector": [1.0, 0.0]},
            {"repo_id": "vuejs/core", "vector": [0.8, 0.2]},
            {"repo_id": "postgres/postgres", "vector": [0.0, 1.0]},
        ],
    }
    batches = []

    def fake_embed_many(config, queries):
        batches.append(list(queries))
        return [[1.0, 0.0] if query == "frontend" else [0.0, 1.0] for query in queries]

    results = rank_many(["frontend", "database"], index, CONFIG, top_k=2, batch_size=1, embed_many=fake_embed_many)

    assert batches == [["frontend"], ["database"]]
    assert results[0]["results"][0]["repo_id"] == "react/react"
    assert results[0]["results"][1]["repo_id"] == "vuejs/core"
    assert results[1]["results"][0]["repo_id"] == "postgres/postgres"
    assert results[0]["considered"] == 3


def test_rank_many_abstains_when_all_weak():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [{"repo_id": "react/react", "vector": [0.0, 1.0]}],
    }

    def fake_embed_many(config, queries):
        return [[1.0, 0.0] for _ in queries]

    result = rank_many(["frontend"], index, CONFIG, embed_many=fake_embed_many)[0]

    assert result["abstained"] is True
    assert result["results"] == []
