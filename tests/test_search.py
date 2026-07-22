import math
import json
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
from xists.search.index import INDEX_VERSION, build_index, decode_vector, encode_vector
from xists.records import RECORD_SCHEMA_VERSION
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
        "schema_version": RECORD_SCHEMA_VERSION,
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
            "project_type": "library",
            "ecosystem": ["javascript", "web"],
            "replaces": [],
            "related_projects": ["preact/preact"],
            "search_text": "react javascript ui library frontend ui library web user interfaces",
            "confidence": "high",
            "abstained": False,
            "prompt_version": 2,
        },
    }


def make_index(vectors):
    return {
        "index_version": 1,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_model": "bge-m3",
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
        "dimension": 2,
        "vectors": vectors,
    }


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


def test_embedding_config_reads_optional_input_type_field(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embeddings.example/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "dual-encoder")
    monkeypatch.setenv("EMBEDDING_INPUT_TYPE_FIELD", "input_type")

    config = embedding_config_from_env()

    assert config.input_type_field == "input_type"


def test_embedding_text_excludes_not_for():
    text = embedding_text_from_record(make_record())
    assert text.splitlines()[1].startswith("react javascript ui library")
    assert "JavaScript UI library" in text
    assert "frontend UI library" in text
    assert "JavaScript" in text
    assert "backend-only services" not in text


def test_embedding_text_prioritizes_search_text():
    record = make_record()
    record["llm_profile"]["search_text"] = "dedicated embedding text for semantic search"

    text = embedding_text_from_record(record)

    assert text.splitlines()[1] == "dedicated embedding text for semantic search"


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


def test_configured_embedding_request_sets_query_input_type(monkeypatch):
    from xists.search import embed as embed_module

    captured = {}

    def fake_request_json(url, body, headers, timeout):
        captured.update(
            url=url, payload=json.loads(body), headers=headers, timeout=timeout
        )
        return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    monkeypatch.setattr(embed_module, "_request_json", fake_request_json)

    vector = embed_module.embed_query(
        EmbeddingConfig(
            api_key="service-secret",
            base_url="https://embeddings.example/v1",
            model="dual-encoder",
            input_type_field="input_type",
        ),
        "Chinese repository search",
    )

    assert vector == [1.0, 0.0]
    assert captured["url"] == "https://embeddings.example/v1/embeddings"
    assert captured["payload"] == {
        "model": "dual-encoder",
        "input": ["Chinese repository search"],
        "input_type": "query",
    }
    assert captured["headers"]["Authorization"] == "Bearer service-secret"


def test_build_index_sends_passage_input_type(monkeypatch):
    captured = []

    def fake_call(config, inputs, *, timeout=60, input_type=None):
        captured.append(input_type)
        return [[1.0, 0.0] for _ in inputs]

    monkeypatch.setattr("xists.search.index.call_embeddings", fake_call)
    build_index(
        [make_record()],
        EmbeddingConfig(
            api_key="k",
            base_url="https://embeddings.example/v1",
            model="dual-encoder",
            input_type_field="input_type",
        ),
    )

    assert captured == ["passage"]


def test_embedding_request_without_configured_input_type_field_omits_it(monkeypatch):
    from xists.search import embed as embed_module

    captured = {}

    def fake_request_json(url, body, headers, timeout):
        captured.update(payload=json.loads(body))
        return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    monkeypatch.setattr(embed_module, "_request_json", fake_request_json)
    call_embeddings(CONFIG, ["hello"], input_type="passage")

    assert captured["payload"] == {"model": "bge-m3", "input": ["hello"]}


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
    assert confidence_bucket(0.599) == "exploratory"
    assert confidence_bucket(0.6) == "high_confidence"
    assert confidence_bucket(0.4) == "exploratory"
    assert confidence_bucket(0.1) == "abstain"


def test_build_index_includes_search_metadata(monkeypatch):
    records = [make_record("react/react"), make_record("vuejs/core")]

    def fake_call(config, inputs, *, timeout=60, input_type=None):
        return [[1.0, 0.0, 0.0] for _ in inputs]

    monkeypatch.setattr("xists.search.index.call_embeddings", fake_call)
    index = build_index(records, CONFIG)
    assert index["embedding_model"] == "bge-m3"
    assert index["embedding_input_version"] == EMBEDDING_INPUT_VERSION
    assert index["dimension"] == 3
    assert index["record_count"] == 2
    metadata = index["vectors"][0]["metadata"]
    assert index["record_schema_version"] == RECORD_SCHEMA_VERSION
    assert metadata["language"] == "JavaScript"
    assert metadata["topics"] == ["frontend", "ui"]
    assert metadata["url"] == "https://github.com/react/react"
    assert metadata["aliases"] == ["reactjs"]
    assert metadata["project_type"] == "library"
    assert metadata["ecosystem"] == ["javascript", "web"]
    assert metadata["search_text"].startswith("react javascript")
    assert index["vectors"][0]["embedding_input_fingerprint"] == embedding_input_fingerprint(records[0])


def test_build_index_skips_empty_records(monkeypatch):
    records = [make_record("react/react"), {"repo_id": "empty/empty"}]

    def fake_call(config, inputs, *, timeout=60, input_type=None):
        return [[1.0, 0.0] for _ in inputs]

    monkeypatch.setattr("xists.search.index.call_embeddings", fake_call)
    index = build_index(records, CONFIG)
    assert index["record_count"] == 1
    assert index["skipped"] == ["empty/empty"]


def test_compact_vector_round_trip_and_legacy_rank_compatibility():
    encoded = encode_vector([1.0, 0.0])

    assert isinstance(encoded, str)
    assert decode_vector(encoded).tolist() == pytest.approx([1.0, 0.0])
    assert decode_vector([1.0, 0.0]).tolist() == pytest.approx([1.0, 0.0])
    index = {
        **make_index(
            [
                {"repo_id": "winner/repo", "vector": encoded, "metadata": {}},
                {"repo_id": "other/repo", "vector": encode_vector([0.0, 1.0]), "metadata": {}},
            ]
        ),
        "index_version": INDEX_VERSION,
    }

    result = rank("query", index, CONFIG, embed=lambda *_: [1.0, 0.0])

    assert result["results"][0]["repo_id"] == "winner/repo"
    __import__("json").dumps(result)


def test_build_index_writes_compact_vectors(monkeypatch):
    monkeypatch.setattr("xists.search.index.call_embeddings", lambda *_args, **_kwargs: [[1.0, 0.0]])

    index = build_index([make_record()], CONFIG)

    assert index["index_version"] == INDEX_VERSION
    assert isinstance(index["vectors"][0]["vector"], str)
    assert decode_vector(index["vectors"][0]["vector"]).tolist() == pytest.approx([1.0, 0.0])


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


def test_repo_id_identity_is_pinned_inside_natural_language_query():
    index = make_index(
        [
            {"repo_id": "react/react", "vector": [0.0, 1.0], "metadata": {"name": "react"}},
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0], "metadata": {"name": "winner"}},
        ]
    )

    result = rank(
        "查找 React 前端库 react/react",
        index,
        CONFIG,
        top_k=2,
        embed=lambda config, query: [1.0, 0.0],
    )

    assert result["results"][0]["repo_id"] == "react/react"
    assert result["results"][0]["diagnostics"]["identity_match"] == "exact"


def test_cjk_context_does_not_pin_an_ascii_name_fragment():
    index = make_index(
        [
            {"repo_id": "shadcn-ui/ui", "vector": [0.0, 1.0], "metadata": {"name": "ui"}},
            {"repo_id": "react/react", "vector": [1.0, 0.0], "metadata": {"name": "react"}},
        ]
    )

    result = rank(
        "现代前端 UI 框架",
        index,
        CONFIG,
        top_k=2,
        embed=lambda config, query: [1.0, 0.0],
    )

    assert result["results"][0]["repo_id"] == "react/react"
    assert result["results"][0]["diagnostics"]["identity_match"] is None


def test_cjk_context_pins_a_distinct_ascii_name():
    index = make_index(
        [
            {"repo_id": "kubernetes/kubernetes", "vector": [0.0, 1.0], "metadata": {"name": "kubernetes"}},
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0], "metadata": {"name": "winner"}},
        ]
    )

    result = rank(
        "Kubernetes 云原生容器编排平台",
        index,
        CONFIG,
        top_k=2,
        embed=lambda config, query: [1.0, 0.0],
    )

    assert result["results"][0]["repo_id"] == "kubernetes/kubernetes"
    assert result["results"][0]["diagnostics"]["identity_match"] == "exact"


def test_cjk_context_does_not_pin_a_repo_owner_fragment():
    index = make_index(
        [
            {
                "repo_id": "python/cpython",
                "vector": [0.0, 1.0],
                "metadata": {"name": "cpython", "aliases": ["Python"]},
            },
            {"repo_id": "fastapi/fastapi", "vector": [1.0, 0.0], "metadata": {"name": "fastapi"}},
        ]
    )

    result = rank(
        "Python 异步 Web API 框架",
        index,
        CONFIG,
        top_k=2,
        embed=lambda config, query: [1.0, 0.0],
    )

    assert result["results"][0]["repo_id"] == "fastapi/fastapi"
    assert result["results"][0]["diagnostics"]["identity_match"] is None


def test_unsupported_semantic_match_is_exploratory_not_high_confidence():
    index = make_index(
        [{"repo_id": "unrelated/repo", "vector": vector_for_cosine(0.58), "metadata": {}}]
    )

    result = rank("unsupported specialized system", index, CONFIG, embed=lambda config, query: [1.0, 0.0])

    assert result["results"][0]["confidence"] == "exploratory"


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
    index = make_index([])
    index["embedding_model"] = "other-model"
    with pytest.raises(IndexMismatchError, match=r"other-model.*bge-m3"):
        rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])


def test_rank_rejects_missing_embedding_model():
    index = make_index([])
    del index["embedding_model"]
    with pytest.raises(IndexMismatchError, match=r"embedding_model.*bge-m3"):
        rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])


def test_rank_rejects_input_version_mismatch():
    index = make_index([])
    index["embedding_input_version"] = EMBEDDING_INPUT_VERSION + 1
    with pytest.raises(IndexMismatchError, match=r"embedding_input_version.*rebuild"):
        rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])


def test_rank_rejects_record_schema_version_mismatch():
    index = make_index([])
    index["record_schema_version"] = RECORD_SCHEMA_VERSION + 1
    with pytest.raises(IndexMismatchError, match=r"record_schema_version.*profile refresh"):
        rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])


def test_rank_many_rejects_index_vectors_not_matching_declared_dimension():
    index = make_index([{"repo_id": "react/react", "vector": [1.0, 0.0, 0.0], "metadata": {}}])
    with pytest.raises(IndexMismatchError, match=r"dimension 2.*Rebuild"):
        rank_many(["frontend ui"], index, CONFIG, embed_many=lambda config, queries: [[1.0, 0.0]])


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
