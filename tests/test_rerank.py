import pytest

from xists.search.rerank import (
    RerankerConfig,
    RerankerError,
    rerank_documents,
    rerank_text_from_entry,
)


def test_rerank_text_uses_only_generic_index_evidence():
    text = rerank_text_from_entry(
        {
            "repo_id": "owner/repo",
            "metadata": {
                "name": "repo",
                "description": "A useful project.",
                "topics": ["tooling"],
                "capabilities": ["automates tasks"],
            },
        }
    )

    assert text.splitlines() == ["owner/repo", "repo", "A useful project.", "tooling", "automates tasks"]


def test_rerank_documents_restores_tei_indexes_to_input_order():
    calls = []

    def fake_request(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        return [{"index": 1, "score": 0.2}, {"index": 0, "score": 0.8}]

    scores = rerank_documents(
        RerankerConfig(api_key="key", base_url="http://reranker/v1", model="model"),
        "query",
        ["first", "second"],
        request_json=fake_request,
    )

    assert scores == [0.8, 0.2]
    assert calls[0][0] == "http://reranker/rerank"
    assert calls[0][1] == {"query": "query", "texts": ["first", "second"], "raw_scores": True, "model": "model"}
    assert calls[0][2]["Authorization"] == "Bearer key"


def test_rerank_documents_rejects_duplicate_response_index():
    with pytest.raises(RerankerError, match="Duplicate"):
        rerank_documents(
            RerankerConfig(api_key=None, base_url="http://reranker"),
            "query",
            ["first", "second"],
            request_json=lambda *args: [{"index": 0, "score": 0.8}, {"index": 0, "score": 0.2}],
        )


def test_rerank_documents_supports_passages_protocol():
    calls = []

    def fake_request(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        return {"rankings": [{"index": 1, "logit": 0.2}, {"index": 0, "logit": 0.8}]}

    scores = rerank_documents(
        RerankerConfig(
            api_key="key",
            base_url="https://reranker.example/v1/retrieval/model/reranking",
            model="model",
            protocol="passages",
        ),
        "query",
        ["first", "second"],
        request_json=fake_request,
    )

    assert scores == [0.8, 0.2]
    assert calls[0][0] == "https://reranker.example/v1/retrieval/model/reranking"
    assert calls[0][1] == {
        "model": "model",
        "query": {"text": "query"},
        "passages": [{"text": "first"}, {"text": "second"}],
    }
