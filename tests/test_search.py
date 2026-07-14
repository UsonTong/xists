import math
from urllib.error import URLError

import pytest

from xists.search.embed import (
    EMBEDDING_INPUT_VERSION,
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingNotConfiguredError,
    call_embeddings,
    embedding_config_from_env,
    embedding_input_fingerprint,
    embedding_text_from_record,
)
from xists.search.index import build_index
from xists.search.query import (
    IndexMismatchError,
    _query_intent,
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
        "url": "https://github.com/react/react",
        "github": {
            "description": "The library for web and native user interfaces.",
            "topics": ["frontend", "ui"],
            "language": "JavaScript",
        },
        "llm_profile": {
            "summary": "React is a JavaScript UI library.",
            "use_cases": ["building web user interfaces"],
            "capabilities": ["declarative UI rendering"],
            "not_for": ["backend-only services"],
            "search_phrases": ["frontend UI library"],
            "aliases": ["reactjs"],
        },
    }


def make_index(vectors):
    return {"embedding_model": "bge-m3", "dimension": 2, "vectors": vectors}


def vector_for_cosine(score):
    return [score, math.sqrt(1.0 - score**2)]


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
    assert "JavaScript" in text
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


def test_call_embeddings_reports_all_attempted_endpoints(monkeypatch):
    from xists.search import embed as embed_module

    def fake_request_json(url, body, headers, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(embed_module, "_request_json", fake_request_json)

    with pytest.raises(EmbeddingError) as error:
        call_embeddings(EmbeddingConfig(api_key="k", base_url="http://localhost:6597/v1", model="bge-m3"), ["hello"])

    message = str(error.value)
    assert "all configured endpoints" in message
    assert "http://localhost:6597/v1/embeddings" in message
    assert "http://localhost:6597/embed" in message
    assert "Check that the embedding service is running" in message


def test_cosine_similarity_and_confidence_bucket():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert confidence_bucket(0.9) == "high_confidence"
    assert confidence_bucket(0.4) == "exploratory"
    assert confidence_bucket(0.1) == "abstain"


def test_build_index_includes_search_metadata(monkeypatch):
    records = [make_record("react/react"), make_record("vuejs/core")]

    def fake_call(config, inputs, *, timeout=60):
        return [[1.0, 0.0, 0.0] for _ in inputs]

    monkeypatch.setattr("xists.search.index.call_embeddings", fake_call)
    index = build_index(records, CONFIG)
    assert index["embedding_model"] == "bge-m3"
    assert index["embedding_input_version"] == EMBEDDING_INPUT_VERSION
    assert index["dimension"] == 3
    assert index["record_count"] == 2
    metadata = index["vectors"][0]["metadata"]
    assert metadata["language"] == "JavaScript"
    assert metadata["topics"] == ["frontend", "ui"]
    assert metadata["url"] == "https://github.com/react/react"
    assert metadata["aliases"] == ["reactjs"]
    assert index["vectors"][0]["embedding_input_fingerprint"] == embedding_input_fingerprint(records[0])


def test_build_index_skips_empty_records(monkeypatch):
    records = [make_record("react/react"), {"repo_id": "empty/empty"}]

    def fake_call(config, inputs, *, timeout=60):
        return [[1.0, 0.0] for _ in inputs]

    monkeypatch.setattr("xists.search.index.call_embeddings", fake_call)
    index = build_index(records, CONFIG)
    assert index["record_count"] == 1
    assert index["skipped"] == ["empty/empty"]


def test_rank_returns_sorted_semantic_results_with_stable_shape():
    index = make_index(
        [
            {"repo_id": "react/react", "vector": [1.0, 0.0], "metadata": {"summary": "React summary"}},
            {"repo_id": "unrelated/repo", "vector": [0.0, 1.0], "metadata": {}},
        ]
    )

    result = rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])
    top = result["results"][0]
    assert result["abstained"] is False
    assert result["considered"] == 2
    assert top["repo_id"] == "react/react"
    assert top["confidence"] == "high_confidence"
    assert top["semantic_score"] == pytest.approx(1.0)
    assert top["score_breakdown"] == {
        "semantic": round(top["semantic_score"], 6),
        "metadata": round(top["metadata_score"], 6),
        "final": round(top["score"], 6),
    }
    assert isinstance(top["why"], list)
    assert isinstance(top["matched_terms"], list)
    assert isinstance(top["diagnostics"], dict)


def test_exact_identity_is_pinned_even_when_embedding_is_weaker():
    index = make_index(
        [
            {"repo_id": "react/react", "vector": [0.0, 1.0], "metadata": {"name": "react"}},
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0], "metadata": {"name": "winner"}},
        ]
    )

    result = rank("react", index, CONFIG, top_k=2, embed=lambda config, query: [1.0, 0.0])
    assert result["query_intent"]["type"] == "exact_name"
    assert result["results"][0]["repo_id"] == "react/react"
    assert result["results"][0]["confidence"] == "high_confidence"
    assert result["results"][0]["diagnostics"]["identity_match"] == "exact"
    assert "matched exact repository identity" in result["results"][0]["why"]


def test_alias_identity_is_pinned():
    index = make_index(
        [
            {
                "repo_id": "vllm-project/vllm",
                "vector": [0.0, 1.0],
                "metadata": {"name": "vllm", "aliases": ["vllm"]},
            },
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0], "metadata": {}},
        ]
    )

    result = rank("vllm", index, CONFIG, top_k=2, embed=lambda config, query: [1.0, 0.0])
    assert result["results"][0]["repo_id"] == "vllm-project/vllm"


def test_identity_falls_back_to_repo_id_parts_for_legacy_indexes():
    index = make_index(
        [
            {"repo_id": "vllm-project/vllm", "vector": [0.0, 1.0]},
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0]},
        ]
    )

    result = rank("vllm", index, CONFIG, top_k=2, embed=lambda config, query: [1.0, 0.0])
    assert result["results"][0]["repo_id"] == "vllm-project/vllm"


def test_semantic_winner_is_not_overturned_by_ordinary_metadata():
    index = make_index(
        [
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0], "metadata": {"description": "General project."}},
            {
                "repo_id": "metadata/match",
                "vector": vector_for_cosine(0.6),
                "metadata": {
                    "description": "Python workflow automation platform.",
                    "topics": ["python", "workflow", "automation"],
                    "language": "Python",
                    "search_phrases": ["python workflow automation platform"],
                },
            },
        ]
    )

    result = rank("python workflow automation platform", index, CONFIG, top_k=2, embed=lambda config, query: [1.0, 0.0])
    assert result["results"][0]["repo_id"] == "semantic/winner"


def test_lightweight_metadata_can_break_a_close_tie():
    index = make_index(
        [
            {"repo_id": "generic/repo", "vector": vector_for_cosine(0.91), "metadata": {"summary": "Generic tool."}},
            {
                "repo_id": "fastapi/fastapi",
                "vector": vector_for_cosine(0.9),
                "metadata": {
                    "name": "fastapi",
                    "description": "FastAPI framework for Python APIs.",
                    "topics": ["python", "api", "framework"],
                    "language": "Python",
                    "search_phrases": ["python web framework for APIs"],
                },
            },
        ]
    )

    result = rank("python api framework", index, CONFIG, top_k=2, embed=lambda config, query: [1.0, 0.0])
    assert result["results"][0]["repo_id"] == "fastapi/fastapi"
    assert result["results"][0]["metadata_score"] > 0
    assert {"api", "framework"}.issubset(set(result["results"][0]["matched_terms"]))


def test_archived_repository_is_downranked():
    metadata = {
        "description": "CLI tool for project automation.",
        "topics": ["cli", "automation"],
        "language": "Python",
        "search_phrases": ["cli tool for project automation"],
    }
    index = make_index(
        [
            {"repo_id": "old/tool", "vector": [1.0, 0.0], "metadata": {**metadata, "archived": True}},
            {"repo_id": "new/tool", "vector": [1.0, 0.0], "metadata": {**metadata, "archived": False}},
        ]
    )

    result = rank("cli project automation", index, CONFIG, top_k=2, embed=lambda config, query: [1.0, 0.0])
    assert result["results"][0]["repo_id"] == "new/tool"
    archived = next(item for item in result["results"] if item["repo_id"] == "old/tool")
    assert archived["diagnostics"]["repository_state"] == ["archived"]
    assert archived["metadata_score"] < result["results"][0]["metadata_score"]


def test_weak_semantic_match_abstains_even_with_loose_metadata_overlap():
    index = make_index(
        [
            {
                "repo_id": "loose/overlap",
                "vector": vector_for_cosine(0.34),
                "metadata": {
                    "description": "Open source workflow automation platform.",
                    "topics": ["workflow", "automation", "platform"],
                    "summary": "Workflow automation for integrations.",
                },
            }
        ]
    )

    result = rank("open source workflow automation platform", index, CONFIG, embed=lambda config, query: [1.0, 0.0])
    assert result["abstained"] is True
    assert result["results"] == []


def test_rank_many_matches_rank_order():
    index = make_index(
        [
            {
                "repo_id": "fastapi/fastapi",
                "vector": [1.0, 0.0],
                "metadata": {"language": "Python", "topics": ["python", "api"], "search_phrases": ["python api framework"]},
            },
            {"repo_id": "react/react", "vector": [0.0, 1.0], "metadata": {"language": "JavaScript"}},
        ]
    )

    single = rank("python api", index, CONFIG, top_k=2, embed=lambda config, query: [1.0, 0.0])
    many = rank_many(["python api"], index, CONFIG, top_k=2, embed_many=lambda config, queries: [[1.0, 0.0]])[0]
    assert [item["repo_id"] for item in many["results"]] == [item["repo_id"] for item in single["results"]]


def test_rank_many_batches_embeddings():
    calls = []

    def fake_embed_many(config, queries):
        calls.append(list(queries))
        return [[1.0, 0.0] for _ in queries]

    index = make_index([{"repo_id": "a/b", "vector": [1.0, 0.0], "metadata": {}}])
    results = rank_many(["one", "two", "three"], index, CONFIG, batch_size=2, embed_many=fake_embed_many)
    assert calls == [["one", "two"], ["three"]]
    assert len(results) == 3


def test_rank_abstains_on_empty_index():
    result = rank("anything", make_index([]), CONFIG, embed=lambda config, query: [1.0, 0.0])
    assert result == {
        "query": "anything",
        "query_intent": _query_intent("anything"),
        "abstained": True,
        "results": [],
        "considered": 0,
    }


def test_rank_rejects_model_mismatch():
    index = {"embedding_model": "other-model", "dimension": 2, "vectors": []}
    with pytest.raises(IndexMismatchError):
        rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])


def test_rank_rejects_dimension_mismatch():
    index = make_index([{"repo_id": "react/react", "vector": [1.0, 0.0], "metadata": {}}])
    with pytest.raises(IndexMismatchError):
        rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0, 0.0])


def test_rank_many_rejects_dimension_mismatch():
    index = make_index([{"repo_id": "react/react", "vector": [1.0, 0.0], "metadata": {}}])
    with pytest.raises(IndexMismatchError):
        rank_many(["frontend ui"], index, CONFIG, embed_many=lambda config, queries: [[1.0, 0.0, 0.0]])


def test_query_intent_keeps_basic_labels():
    assert _query_intent("vllm")["type"] == "exact_name"
    assert _query_intent("open source firebase alternative")["type"] == "alternative"
    assert _query_intent("python web framework")["primary_language"] == "python"
    assert _query_intent("")["type"] == "empty"
