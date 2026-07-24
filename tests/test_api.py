import json

import pytest

from xists.api import load_index, search
from xists.records import RECORD_SCHEMA_VERSION
from xists.search.embed import EmbeddingConfig, EmbeddingError
from xists.search.embed import EMBEDDING_INPUT_VERSION
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
