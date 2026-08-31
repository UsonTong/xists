import json
import math
from urllib.error import URLError

import numpy as np
import pytest

from xists.records import RECORD_SCHEMA_VERSION
from xists.search.confidence import calibrate_confidence
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
from xists.search.index import (
    INDEX_VERSION,
    build_index,
    decode_vector,
    encode_vector,
    load_index,
    save_index,
)
from xists.search.query import (
    IndexMismatchError,
    PreparedIndex,
    _query_intent,
    confidence_bucket,
    cosine_similarity,
    prepare_index,
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
        "index_version": INDEX_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_model": "bge-m3",
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
        "dimension": 2,
        "record_count": len(vectors),
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


def test_embedding_config_reads_multiple_keys_and_file(monkeypatch, tmp_path):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("# comment\nkey-file-1\nkey-file-2\n\nkey-file-1\n", encoding="utf-8")
    monkeypatch.setenv("EMBEDDING_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("EMBEDDING_API_KEYS", "key-env-1, key-env-2")
    monkeypatch.setenv("EMBEDDING_API_KEY", "key-single, key-file-2")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embeddings.example/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "fixture-model")

    config = embedding_config_from_env()
    assert config.api_key == "key-file-1"
    assert config.api_keys == ("key-file-1", "key-file-2", "key-env-1", "key-env-2", "key-single")

    cloned = config.with_api_key("custom-key")
    assert cloned.api_key == "custom-key"
    assert cloned.base_url == config.base_url
    assert cloned.model == config.model
    assert cloned.api_keys == config.api_keys


def test_request_json_retries_transient_http_errors(monkeypatch):
    from urllib.error import HTTPError

    from xists.search import embed as embed_module

    attempts = 0

    def fake_urlopen(request, timeout=30):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise HTTPError(request.full_url, 429, "Too Many Requests", hdrs=None, fp=None)

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return b'{"data": [{"index": 0, "embedding": [1.0, 0.0]}]}'

        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)

    res = embed_module._request_json("http://localhost/v1", b"{}", {}, timeout=5)
    assert attempts == 3
    assert res == {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}


def test_request_json_fails_immediately_on_non_retryable_error(monkeypatch):
    from urllib.error import HTTPError

    from xists.search import embed as embed_module

    attempts = 0

    def fake_urlopen(request, timeout=30):
        nonlocal attempts
        attempts += 1
        raise HTTPError(request.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(HTTPError) as exc_info:
        embed_module._request_json("http://localhost/v1", b"{}", {}, timeout=5)
    assert attempts == 1
    assert exc_info.value.code == 401


def test_embedding_text_truncates_long_description():
    record = make_record()
    record["llm_profile"] = None
    record["github"]["description"] = "A" * 3000

    text = embedding_text_from_record(record)
    assert len(text) <= 2050
    assert "A" * 2000 in text
    assert "A" * 2001 not in text


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
        captured.update(url=url, payload=json.loads(body), headers=headers, timeout=timeout)
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
        call_embeddings(
            EmbeddingConfig(api_key="k", base_url="http://localhost:6597/v1", model="bge-m3"),
            ["hello"],
        )

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
    assert index["vectors"][0]["embedding_input_fingerprint"] == embedding_input_fingerprint(
        records[0]
    )


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
    monkeypatch.setattr(
        "xists.search.index.call_embeddings", lambda *_args, **_kwargs: [[1.0, 0.0]]
    )

    index = build_index([make_record()], CONFIG)

    assert index["index_version"] == INDEX_VERSION
    assert isinstance(index["vectors"][0]["vector"], str)
    assert decode_vector(index["vectors"][0]["vector"]).tolist() == pytest.approx([1.0, 0.0])


def test_rank_returns_sorted_semantic_results_with_stable_shape():
    index = make_index(
        [
            {
                "repo_id": "react/react",
                "vector": [1.0, 0.0],
                "metadata": {"summary": "React summary"},
            },
            {"repo_id": "unrelated/repo", "vector": [0.0, 1.0], "metadata": {}},
        ]
    )

    result = rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])
    top = result["results"][0]
    assert result["abstained"] is False
    assert result["considered"] == 2
    assert isinstance(result["latency_ms"], float)
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


def test_explicit_chinese_project_lookup_without_spaces_is_treated_as_exact_identity():
    index = make_index(
        [
            {"repo_id": "vuejs/core", "vector": [0.0, 1.0], "metadata": {"name": "vue"}},
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0], "metadata": {"name": "winner"}},
        ]
    )

    result = rank(
        "查找Vue开源项目",
        index,
        CONFIG,
        top_k=2,
        embed=lambda _config, _query: [1.0, 0.0],
    )

    assert result["results"][0]["repo_id"] == "vuejs/core"
    assert result["query_intent"]["type"] == "exact_name"
    assert result["results"][0]["diagnostics"]["identity_evidence"]["kind"] == "exact_value"


@pytest.mark.parametrize("query", ["查找Vue开源项目。", "搜索 Vue 项目！", "寻找Vue开源项目？"])
def test_explicit_chinese_project_lookup_ignores_terminal_punctuation(query):
    index = make_index(
        [
            {"repo_id": "vuejs/core", "vector": [0.0, 1.0], "metadata": {"name": "vue"}},
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0], "metadata": {"name": "winner"}},
        ]
    )

    result = rank(query, index, CONFIG, top_k=2, embed=lambda _config, _query: [1.0, 0.0])

    assert result["results"][0]["repo_id"] == "vuejs/core"
    assert result["query_intent"]["type"] == "exact_name"
    assert result["results"][0]["diagnostics"]["identity_evidence"]["kind"] == "exact_value"


@pytest.mark.parametrize("query", ["搜索：Vue 项目", "查找:Vue开源项目", "寻找： Vue 项目"])
def test_explicit_chinese_project_lookup_ignores_leading_colons(query):
    index = make_index(
        [
            {"repo_id": "vuejs/core", "vector": [0.0, 1.0], "metadata": {"name": "vue"}},
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0], "metadata": {"name": "winner"}},
        ]
    )

    result = rank(query, index, CONFIG, top_k=2, embed=lambda _config, _query: [1.0, 0.0])

    assert result["results"][0]["repo_id"] == "vuejs/core"
    assert result["query_intent"]["type"] == "exact_name"
    assert result["results"][0]["diagnostics"]["identity_evidence"]["kind"] == "exact_value"


def test_explicit_chinese_project_lookup_is_treated_as_exact_identity():
    index = make_index(
        [
            {"repo_id": "vuejs/core", "vector": [0.0, 1.0], "metadata": {"name": "vue"}},
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0], "metadata": {"name": "winner"}},
        ]
    )

    result = rank(
        "查找 Vue 开源项目",
        index,
        CONFIG,
        top_k=2,
        embed=lambda _config, _query: [1.0, 0.0],
    )

    assert result["results"][0]["repo_id"] == "vuejs/core"
    assert result["query_intent"]["type"] == "exact_name"
    assert result["results"][0]["diagnostics"]["identity_evidence"]["kind"] == "exact_value"


def test_ambiguous_exact_values_do_not_claim_high_confidence():
    index = make_index(
        [
            {"repo_id": "current/vue", "vector": [1.0, 0.0], "metadata": {"aliases": ["vue"]}},
            {
                "repo_id": "legacy/vue",
                "vector": vector_for_cosine(0.9),
                "metadata": {"name": "vue"},
            },
        ]
    )

    result = rank("查找 Vue 开源项目", index, CONFIG, top_k=2, embed=lambda *_: [1.0, 0.0])

    assert [item["confidence"] for item in result["results"]] == ["exploratory", "exploratory"]
    assert all(item["diagnostics"]["identity_ambiguity_count"] == 2 for item in result["results"])
    assert all("ambiguous exact identity" in item["why"] for item in result["results"])


def test_repo_id_identity_remains_high_confidence_when_aliases_collide():
    index = make_index(
        [
            {"repo_id": "current/vue", "vector": [0.0, 1.0], "metadata": {"aliases": ["vue"]}},
            {"repo_id": "legacy/vue", "vector": [1.0, 0.0], "metadata": {"name": "vue"}},
        ]
    )

    result = rank("current/vue", index, CONFIG, top_k=2, embed=lambda *_: [1.0, 0.0])

    assert result["results"][0]["repo_id"] == "current/vue"
    assert result["results"][0]["confidence"] == "high_confidence"
    assert result["results"][0]["diagnostics"]["identity_evidence"]["kind"] == "repo_id"


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
            {
                "repo_id": "kubernetes/kubernetes",
                "vector": vector_for_cosine(0.82),
                "metadata": {"name": "kubernetes"},
            },
            {
                "repo_id": "semantic/winner",
                "vector": vector_for_cosine(0.80),
                "metadata": {"name": "winner"},
            },
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
    assert result["results"][0]["diagnostics"]["identity_match"] == "contextual"
    assert (
        result["results"][0]["diagnostics"]["identity_evidence"]["kind"]
        == "contextual_name_mention"
    )


def test_cjk_ecosystem_mention_cannot_overturn_a_clear_semantic_winner():
    index = make_index(
        [
            {
                "repo_id": "nodejs/node",
                "vector": vector_for_cosine(0.4),
                "metadata": {"name": "node", "aliases": ["Node.js"]},
            },
            {
                "repo_id": "expressjs/express",
                "vector": [1.0, 0.0],
                "metadata": {"name": "express"},
            },
        ]
    )

    result = rank(
        "轻量级 Node.js Web 应用框架",
        index,
        CONFIG,
        top_k=2,
        embed=lambda _config, _query: [1.0, 0.0],
    )

    assert result["results"][0]["repo_id"] == "expressjs/express"
    node = next(item for item in result["results"] if item["repo_id"] == "nodejs/node")
    assert node["diagnostics"]["identity_evidence"]["kind"] == "contextual_name_mention"
    assert node["diagnostics"]["identity_match"] == "contextual"


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

    result = rank(
        "unsupported specialized system", index, CONFIG, embed=lambda config, query: [1.0, 0.0]
    )

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


def test_semantic_strategy_does_not_apply_identity_or_metadata_adjustments():
    index = make_index(
        [
            {"repo_id": "react/react", "vector": [0.0, 1.0], "metadata": {"name": "react"}},
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0], "metadata": {"name": "winner"}},
        ]
    )

    result = rank(
        "react",
        index,
        CONFIG,
        top_k=2,
        embed=lambda config, query: [1.0, 0.0],
        ranking_strategy="semantic",
    )

    assert result["results"][0]["repo_id"] == "semantic/winner"
    assert result["results"][0]["metadata_score"] == 0.0
    assert result["results"][0]["diagnostics"]["identity_match"] is None


def test_rerank_strategy_fuses_semantic_recall_and_generic_rerank_evidence():
    index = make_index(
        [
            {
                "repo_id": "first/repo",
                "vector": [1.0, 0.0],
                "metadata": {"description": "First candidate"},
            },
            {
                "repo_id": "second/repo",
                "vector": vector_for_cosine(0.8),
                "metadata": {"description": "Second candidate"},
            },
        ]
    )
    calls = []

    def fake_rerank(query, documents):
        calls.append((query, documents))
        return [0.1, 0.9]

    result = rank(
        "general query",
        index,
        CONFIG,
        top_k=2,
        embed=lambda config, query: [1.0, 0.0],
        ranking_strategy="rerank",
        rerank=fake_rerank,
        rerank_candidate_limit=2,
    )

    assert result["results"][0]["repo_id"] == "first/repo"
    assert result["results"][0]["rerank_score"] == 0.1
    assert result["results"][0]["metadata_score"] == 0.0
    assert result["results"][0]["ranking_evidence"] == {
        "semantic_rank": 1,
        "rerank_rank": 2,
        "fusion": "reciprocal_rank",
    }
    assert calls == [
        ("general query", ["first/repo\nFirst candidate", "second/repo\nSecond candidate"])
    ]


def test_evidence_calibration_keeps_agreeing_rerank_winner_high_confidence():
    index = make_index(
        [
            {"repo_id": "first/repo", "vector": [1.0, 0.0], "metadata": {}},
            {"repo_id": "second/repo", "vector": vector_for_cosine(0.8), "metadata": {}},
        ]
    )

    result = rank(
        "general query",
        index,
        CONFIG,
        top_k=2,
        embed=lambda _config, _query: [1.0, 0.0],
        ranking_strategy="rerank",
        rerank=lambda _query, _documents: [0.9, 0.1],
        rerank_candidate_limit=2,
        confidence_calibration="evidence-v1",
    )

    top = result["results"][0]
    assert top["confidence"] == "high_confidence"
    assert top["confidence_evidence"]["supporting_signals"] == ["semantic_and_rerank_agree"]
    assert top["confidence_evidence"]["downgrade_reasons"] == []


def test_evidence_calibration_downgrades_conflicting_rerank_without_reordering():
    index = make_index(
        [
            {"repo_id": "first/repo", "vector": [1.0, 0.0], "metadata": {}},
            {"repo_id": "second/repo", "vector": vector_for_cosine(0.8), "metadata": {}},
        ]
    )
    kwargs = {
        "top_k": 2,
        "embed": lambda _config, _query: [1.0, 0.0],
        "ranking_strategy": "rerank",
        "rerank": lambda _query, _documents: [0.1, 0.9],
        "rerank_candidate_limit": 2,
    }

    baseline = rank("general query", index, CONFIG, **kwargs)
    calibrated = rank(
        "general query", index, CONFIG, confidence_calibration="evidence-v1", **kwargs
    )

    assert [item["repo_id"] for item in calibrated["results"]] == [
        item["repo_id"] for item in baseline["results"]
    ]
    assert calibrated["abstained"] is baseline["abstained"] is False
    assert baseline["results"][0]["confidence"] == "high_confidence"
    assert calibrated["results"][0]["confidence"] == "exploratory"
    assert set(calibrated["results"][0]["confidence_evidence"]["downgrade_reasons"]) == {
        "semantic_and_rerank_disagree",
        "top_candidate_not_separated",
    }


def test_evidence_calibration_downgrades_high_confidence_without_reranker_evidence():
    result = {
        "confidence": "high_confidence",
        "diagnostics": {"identity_evidence": {"kind": "none"}},
    }

    calibrated = calibrate_confidence([result], ranking_strategy="rerank", mode="evidence-v1")

    assert calibrated == [result]
    assert result["confidence"] == "exploratory"
    assert result["confidence_evidence"]["downgrade_reasons"] == ["reranker_evidence_unavailable"]


def test_evidence_calibration_keeps_contextual_identity_weak_in_rerank_results():
    index = make_index(
        [
            {"repo_id": "nodejs/node", "vector": [1.0, 0.0], "metadata": {"name": "node"}},
            {"repo_id": "other/repo", "vector": vector_for_cosine(0.8), "metadata": {}},
        ]
    )
    kwargs = {
        "top_k": 2,
        "embed": lambda _config, _query: [1.0, 0.0],
        "ranking_strategy": "rerank",
        "rerank": lambda _query, _documents: [0.1, 0.9],
        "rerank_candidate_limit": 2,
    }

    baseline = rank("Node Web 框架", index, CONFIG, **kwargs)
    calibrated = rank(
        "Node Web 框架", index, CONFIG, confidence_calibration="evidence-v1", **kwargs
    )

    assert [item["repo_id"] for item in calibrated["results"]] == [
        item["repo_id"] for item in baseline["results"]
    ]
    node = next(item for item in calibrated["results"] if item["repo_id"] == "nodejs/node")
    assert node["diagnostics"]["identity_evidence"]["kind"] == "contextual_name_mention"
    assert node["confidence_evidence"]["identity_evidence"] == "contextual_name_mention"


def test_evidence_calibration_preserves_direct_repository_identity():
    index = make_index(
        [
            {"repo_id": "nodejs/node", "vector": [1.0, 0.0], "metadata": {"name": "node"}},
            {"repo_id": "other/repo", "vector": vector_for_cosine(0.8), "metadata": {}},
        ]
    )

    result = rank(
        "nodejs/node",
        index,
        CONFIG,
        top_k=2,
        embed=lambda _config, _query: [1.0, 0.0],
        ranking_strategy="rerank",
        rerank=lambda _query, _documents: [0.1],
        rerank_candidate_limit=2,
        confidence_calibration="evidence-v1",
    )

    assert result["results"][0]["confidence"] == "high_confidence"
    assert result["results"][0]["confidence_evidence"]["identity_evidence"] == "repo_id"


def test_rerank_strategy_requires_a_reranker():
    index = make_index([{"repo_id": "one/repo", "vector": [1.0, 0.0], "metadata": {}}])

    with pytest.raises(ValueError, match="reranker"):
        rank(
            "query",
            index,
            CONFIG,
            embed=lambda config, query: [1.0, 0.0],
            ranking_strategy="rerank",
        )


def test_rerank_abstain_threshold_rejects_a_query_when_fused_top_score_is_too_low():
    index = make_index(
        [
            {"repo_id": "first/repo", "vector": [1.0, 0.0], "metadata": {}},
            {"repo_id": "second/repo", "vector": vector_for_cosine(0.8), "metadata": {}},
        ]
    )

    result = rank(
        "general query",
        index,
        CONFIG,
        embed=lambda config, query: [1.0, 0.0],
        ranking_strategy="rerank",
        rerank=lambda query, documents: [-9.0, -10.0],
        rerank_candidate_limit=2,
        rerank_abstain_threshold=-8.0,
    )

    assert result["abstained"] is True
    assert result["results"] == []


def test_rerank_abstain_threshold_preserves_an_exact_repository_identity():
    index = make_index(
        [
            {"repo_id": "owner/project", "vector": [1.0, 0.0], "metadata": {}},
            {"repo_id": "other/repo", "vector": vector_for_cosine(0.8), "metadata": {}},
        ]
    )

    result = rank(
        "owner/project",
        index,
        CONFIG,
        embed=lambda config, query: [1.0, 0.0],
        ranking_strategy="rerank",
        rerank=lambda query, documents: [-99.0],
        rerank_candidate_limit=2,
        rerank_abstain_threshold=-8.0,
    )

    assert result["abstained"] is False
    assert result["results"][0]["repo_id"] == "owner/project"


def test_semantic_winner_is_not_overturned_by_ordinary_metadata():
    index = make_index(
        [
            {
                "repo_id": "semantic/winner",
                "vector": [1.0, 0.0],
                "metadata": {"description": "General project."},
            },
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

    result = rank(
        "python workflow automation platform",
        index,
        CONFIG,
        top_k=2,
        embed=lambda config, query: [1.0, 0.0],
    )
    assert result["results"][0]["repo_id"] == "semantic/winner"


def test_lightweight_metadata_can_break_a_close_tie():
    index = make_index(
        [
            {
                "repo_id": "generic/repo",
                "vector": vector_for_cosine(0.91),
                "metadata": {"summary": "Generic tool."},
            },
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

    result = rank(
        "python api framework", index, CONFIG, top_k=2, embed=lambda config, query: [1.0, 0.0]
    )
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
            {
                "repo_id": "old/tool",
                "vector": [1.0, 0.0],
                "metadata": {**metadata, "archived": True},
            },
            {
                "repo_id": "new/tool",
                "vector": [1.0, 0.0],
                "metadata": {**metadata, "archived": False},
            },
        ]
    )

    result = rank(
        "cli project automation", index, CONFIG, top_k=2, embed=lambda config, query: [1.0, 0.0]
    )
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

    result = rank(
        "open source workflow automation platform",
        index,
        CONFIG,
        embed=lambda config, query: [1.0, 0.0],
    )
    assert result["abstained"] is True
    assert result["results"] == []


def test_exploratory_threshold_is_configurable_for_any_embedding_score_scale():
    index = make_index(
        [
            {
                "repo_id": "general/result",
                "vector": vector_for_cosine(0.34),
                "metadata": {"description": "General purpose retrieval result."},
            }
        ]
    )

    result = rank(
        "general retrieval",
        index,
        CONFIG,
        embed=lambda config, query: [1.0, 0.0],
        exploratory_threshold=0.32,
    )

    assert result["abstained"] is False
    assert result["results"][0]["confidence"] == "exploratory"


def test_rank_rejects_invalid_exploratory_threshold():
    with pytest.raises(ValueError, match="exploratory threshold"):
        rank("query", make_index([]), CONFIG, exploratory_threshold=1.01)


def test_rank_many_matches_rank_order():
    index = make_index(
        [
            {
                "repo_id": "fastapi/fastapi",
                "vector": [1.0, 0.0],
                "metadata": {
                    "language": "Python",
                    "topics": ["python", "api"],
                    "search_phrases": ["python api framework"],
                },
            },
            {
                "repo_id": "react/react",
                "vector": [0.0, 1.0],
                "metadata": {"language": "JavaScript"},
            },
        ]
    )

    single = rank("python api", index, CONFIG, top_k=2, embed=lambda config, query: [1.0, 0.0])
    many = rank_many(
        ["python api"], index, CONFIG, top_k=2, embed_many=lambda config, queries: [[1.0, 0.0]]
    )[0]
    assert [item["repo_id"] for item in many["results"]] == [
        item["repo_id"] for item in single["results"]
    ]


def test_rank_many_batches_embeddings():
    calls = []

    def fake_embed_many(config, queries):
        calls.append(list(queries))
        return [[1.0, 0.0] for _ in queries]

    index = make_index([{"repo_id": "a/b", "vector": [1.0, 0.0], "metadata": {}}])
    results = rank_many(
        ["one", "two", "three"], index, CONFIG, batch_size=2, embed_many=fake_embed_many
    )
    assert calls == [["one", "two"], ["three"]]
    assert len(results) == 3


def test_dual_query_variants_keep_the_stronger_cross_language_candidate():
    index = make_index(
        [
            {"repo_id": "wrong/repo", "vector": vector_for_cosine(0.8), "metadata": {}},
            {"repo_id": "target/repo", "vector": [0.0, 1.0], "metadata": {}},
        ]
    )
    vectors = {"原始查询": [1.0, 0.0], "canonical query": [0.0, 1.0]}

    result = rank(
        "原始查询",
        index,
        CONFIG,
        top_k=2,
        embed=lambda config, query: vectors[query],
        query_variants=["原始查询", "canonical query"],
    )

    assert result["results"][0]["repo_id"] == "target/repo"
    assert result["query_variants"] == ["原始查询", "canonical query"]


def test_rank_many_batches_all_query_variants_and_reranks_with_canonical_query():
    calls = []
    rerank_calls = []

    def fake_embed_many(config, queries):
        calls.append(list(queries))
        return [[1.0, 0.0] if query == "原始查询" else [0.0, 1.0] for query in queries]

    index = make_index(
        [
            {"repo_id": "wrong/repo", "vector": vector_for_cosine(0.8), "metadata": {}},
            {"repo_id": "target/repo", "vector": [0.0, 1.0], "metadata": {}},
        ]
    )
    results = rank_many(
        ["原始查询"],
        index,
        CONFIG,
        top_k=2,
        batch_size=2,
        embed_many=fake_embed_many,
        ranking_strategy="rerank",
        rerank=lambda query, documents: rerank_calls.append(query) or [0.9, 0.1],
        rerank_candidate_limit=2,
        query_variants=[["原始查询", "canonical query"]],
        rerank_queries=["canonical query"],
    )

    assert calls == [["原始查询", "canonical query"]]
    assert rerank_calls == ["canonical query"]
    assert results[0]["query_variants"] == ["原始查询", "canonical query"]


def test_rank_abstains_on_empty_index():
    result = rank("anything", make_index([]), CONFIG, embed=lambda config, query: [1.0, 0.0])
    assert result == {
        "query": "anything",
        "latency_ms": pytest.approx(result["latency_ms"]),
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


@pytest.mark.parametrize("version", [None, 1, 2, 5])
def test_rank_rejects_missing_or_incompatible_index_version(version):
    index = make_index([])
    index["index_version"] = version

    with pytest.raises(IndexMismatchError, match=r"index_version.*Rebuild"):
        rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])


@pytest.mark.parametrize("version", [3, 4])
def test_rank_accepts_supported_index_versions(version):
    index = make_index([{"repo_id": "a/b", "vector": [1.0, 0.0], "metadata": {}}])
    index["index_version"] = version
    res = rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])
    assert not res["abstained"]
    assert res["results"][0]["repo_id"] == "a/b"


def test_rank_rejects_non_list_vectors():
    index = make_index([])
    index["vectors"] = {"not": "a list"}

    with pytest.raises(IndexMismatchError, match=r"vectors.*list.*Rebuild"):
        rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])


def test_rank_rejects_record_count_not_matching_vectors():
    index = make_index([{"repo_id": "react/react", "vector": [1.0, 0.0], "metadata": {}}])
    index["record_count"] = 2

    with pytest.raises(IndexMismatchError, match=r"record_count.*vectors.*Rebuild"):
        rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])


def test_rank_rejects_declared_vector_count_not_matching_vectors():
    index = make_index([{"repo_id": "react/react", "vector": [1.0, 0.0], "metadata": {}}])
    index["vector_count"] = 2

    with pytest.raises(IndexMismatchError, match=r"vector_count.*vectors.*Rebuild"):
        rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])


@pytest.mark.parametrize("dimension", [0, -1, 2.5, True])
def test_rank_rejects_invalid_index_dimension(dimension):
    index = make_index([])
    index["dimension"] = dimension

    with pytest.raises(IndexMismatchError, match=r"dimension.*positive integer.*Rebuild"):
        rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])


def test_rank_accepts_a_valid_empty_index_without_dimension():
    index = make_index([])
    index["dimension"] = None

    result = rank("frontend ui", index, CONFIG, embed=lambda config, query: [1.0, 0.0])

    assert result["abstained"] is True
    assert result["results"] == []


def test_rank_rejects_invalid_vector_entry_before_embedding():
    index = make_index([{"repo_id": "react/react", "vector": [1.0, 0.0], "metadata": {}}])
    index["vectors"][0]["repo_id"] = ""

    with pytest.raises(IndexMismatchError, match=r"repo_id.*Rebuild"):
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
        rank_many(
            ["frontend ui"], index, CONFIG, embed_many=lambda config, queries: [[1.0, 0.0, 0.0]]
        )


def test_query_intent_keeps_basic_labels():
    assert _query_intent("vllm")["type"] == "exact_name"
    assert _query_intent("open source firebase alternative")["type"] == "alternative"
    assert _query_intent("python web framework")["primary_language"] == "python"
    assert _query_intent("")["type"] == "empty"


def test_query_intent_extracts_bounded_cjk_terms_without_losing_technical_identifiers():
    intent = _query_intent("自托管大语言模型应用界面，Node.js C++ C# .NET")

    cjk_terms = [
        term for term in intent["keywords"] if any("\u3400" <= char <= "\u9fff" for char in term)
    ]
    assert intent["type"] == "functional"
    assert intent["specificity"] > 0
    assert cjk_terms
    assert all(2 <= len(term) <= 3 for term in cjk_terms)
    assert len(cjk_terms) <= 32
    assert {"node.js", "c++", "c#", "net"}.issubset(intent["keywords"])


def test_query_intent_deduplicates_cjk_terms_across_whitespace_and_punctuation():
    intent = _query_intent("  自托管，自托管   大语言模型！！  ")

    assert len(intent["keywords"]) == len(set(intent["keywords"]))
    assert "自托管" in intent["keywords"]
    assert all(len(term) > 1 for term in intent["keywords"])


def test_cjk_terms_participate_in_metadata_overlap_and_explanations():
    index = make_index(
        [
            {
                "repo_id": "chat/ui",
                "vector": vector_for_cosine(0.90),
                "metadata": {
                    "name": "chat-ui",
                    "search_text": "自托管大语言模型应用界面",
                    "capabilities": ["大语言模型应用界面"],
                },
            },
            {"repo_id": "generic/result", "vector": vector_for_cosine(0.91), "metadata": {}},
        ]
    )

    result = rank(
        "自托管大语言模型应用界面",
        index,
        CONFIG,
        top_k=2,
        embed=lambda _config, _query: [1.0, 0.0],
    )

    top = result["results"][0]
    assert top["repo_id"] == "chat/ui"
    assert top["matched_terms"]
    assert any("matched metadata terms" in reason for reason in top["why"])


def test_prepared_index_creation_and_dict_protocol():
    raw_index = make_index(
        [
            {
                "repo_id": "fastapi/fastapi",
                "vector": [1.0, 0.0],
                "metadata": {"name": "fastapi", "language": "Python"},
            },
            {
                "repo_id": "expressjs/express",
                "vector": [0.0, 1.0],
                "metadata": {"name": "express", "language": "JavaScript"},
            },
        ]
    )

    prepared = prepare_index(raw_index, CONFIG)
    assert isinstance(prepared, PreparedIndex)
    assert len(prepared) == 2
    assert prepared.dimension == 2
    assert prepared.record_count == 2
    assert prepared.embedding_model == "bge-m3"
    assert prepared.repo_ids == ("fastapi/fastapi", "expressjs/express")
    assert prepared.repo_id_to_index == {"fastapi/fastapi": 0, "expressjs/express": 1}
    assert prepared.normalized_matrix.shape == (2, 2)

    # Dict compatibility
    assert prepared["vectors"] == raw_index["vectors"]
    assert prepared["dimension"] == 2
    assert prepared["record_count"] == 2
    assert prepared.get("dimension") == 2
    assert prepared.get("nonexistent", "fallback") == "fallback"
    assert "vectors" in prepared
    assert "dimension" in prepared
    assert len(list(iter(prepared))) > 0


def test_prepare_index_idempotent():
    raw_index = make_index([{"repo_id": "a/b", "vector": [1.0, 0.0], "metadata": {}}])
    prep1 = prepare_index(raw_index, CONFIG)
    prep2 = prepare_index(prep1, CONFIG)
    assert prep1 is prep2


def test_prepare_index_type_error():
    with pytest.raises(IndexMismatchError, match="must be a dictionary or PreparedIndex"):
        prepare_index("not-an-index")


def test_ranking_parity_single_and_batch_all_strategies():
    index = make_index(
        [
            {
                "repo_id": "fastapi/fastapi",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "fastapi",
                    "description": "FastAPI framework for Python.",
                    "language": "Python",
                    "topics": ["python", "api"],
                    "search_phrases": ["python api framework"],
                },
            },
            {
                "repo_id": "flask/flask",
                "vector": vector_for_cosine(0.85),
                "metadata": {
                    "name": "flask",
                    "description": "The Python micro framework.",
                    "language": "Python",
                    "topics": ["python", "web"],
                },
            },
            {
                "repo_id": "expressjs/express",
                "vector": [0.0, 1.0],
                "metadata": {
                    "name": "express",
                    "description": "Fast web framework for Node.js.",
                    "language": "JavaScript",
                },
            },
        ]
    )

    query = "python api framework"
    query_vector = [1.0, 0.0]

    # Strategy 1: metadata
    single_meta = rank(query, index, CONFIG, top_k=3, embed=lambda c, q: query_vector)
    many_meta = rank_many([query], index, CONFIG, top_k=3, embed_many=lambda c, qs: [query_vector])[
        0
    ]
    assert [r["repo_id"] for r in single_meta["results"]] == [
        r["repo_id"] for r in many_meta["results"]
    ]
    for r1, r2 in zip(single_meta["results"], many_meta["results"]):
        assert r1["score"] == pytest.approx(r2["score"], abs=1e-5)
        assert r1["semantic_score"] == pytest.approx(r2["semantic_score"], abs=1e-5)
        assert r1["metadata_score"] == pytest.approx(r2["metadata_score"], abs=1e-5)
        assert r1["confidence"] == r2["confidence"]

    # Strategy 2: semantic
    single_sem = rank(
        query, index, CONFIG, top_k=3, embed=lambda c, q: query_vector, ranking_strategy="semantic"
    )
    many_sem = rank_many(
        [query],
        index,
        CONFIG,
        top_k=3,
        embed_many=lambda c, qs: [query_vector],
        ranking_strategy="semantic",
    )[0]
    assert [r["repo_id"] for r in single_sem["results"]] == [
        r["repo_id"] for r in many_sem["results"]
    ]
    for r1, r2 in zip(single_sem["results"], many_sem["results"]):
        assert r1["score"] == pytest.approx(r2["score"], abs=1e-5)

    # Strategy 3: rerank
    def mock_rerank(q, docs):
        return [0.95, 0.50, 0.10][: len(docs)]

    single_rerank = rank(
        query,
        index,
        CONFIG,
        top_k=3,
        embed=lambda c, q: query_vector,
        ranking_strategy="rerank",
        rerank=mock_rerank,
        rerank_candidate_limit=3,
    )
    many_rerank = rank_many(
        [query],
        index,
        CONFIG,
        top_k=3,
        embed_many=lambda c, qs: [query_vector],
        ranking_strategy="rerank",
        rerank=mock_rerank,
        rerank_candidate_limit=3,
    )[0]
    assert [r["repo_id"] for r in single_rerank["results"]] == [
        r["repo_id"] for r in many_rerank["results"]
    ]
    for r1, r2 in zip(single_rerank["results"], many_rerank["results"]):
        assert r1["score"] == pytest.approx(r2["score"], abs=1e-5)


def test_ranking_parity_prepared_vs_raw_dict():
    raw_index = make_index(
        [
            {"repo_id": "repo/a", "vector": [1.0, 0.0], "metadata": {"name": "repo-a"}},
            {"repo_id": "repo/b", "vector": [0.0, 1.0], "metadata": {"name": "repo-b"}},
        ]
    )
    prepared = prepare_index(raw_index, CONFIG)

    res_raw = rank("repo-a", raw_index, CONFIG, embed=lambda c, q: [1.0, 0.0])
    res_prep = rank("repo-a", prepared, CONFIG, embed=lambda c, q: [1.0, 0.0])

    assert [r["repo_id"] for r in res_raw["results"]] == [r["repo_id"] for r in res_prep["results"]]
    assert res_raw["results"][0]["score"] == pytest.approx(
        res_prep["results"][0]["score"], abs=1e-5
    )


def test_zero_vector_similarity_handling():
    index = make_index(
        [
            {"repo_id": "nonzero/repo", "vector": [1.0, 0.0], "metadata": {}},
            {"repo_id": "zero/repo", "vector": [0.0, 0.0], "metadata": {}},
        ]
    )
    prepared = prepare_index(index, CONFIG)
    assert not math.isnan(prepared.normalized_matrix[1, 0])
    assert not math.isnan(prepared.normalized_matrix[1, 1])

    result = rank("test", prepared, CONFIG, embed=lambda c, q: [0.0, 0.0])
    assert result["abstained"] is True

    result_nonzero = rank("test", prepared, CONFIG, embed=lambda c, q: [1.0, 0.0])
    assert result_nonzero["results"][0]["repo_id"] == "nonzero/repo"


def test_ranking_tie_breaking_by_repo_id():
    index = make_index(
        [
            {"repo_id": "beta/repo", "vector": [1.0, 0.0], "metadata": {}},
            {"repo_id": "alpha/repo", "vector": [1.0, 0.0], "metadata": {}},
        ]
    )
    single = rank("test", index, CONFIG, top_k=2, embed=lambda c, q: [1.0, 0.0])
    many = rank_many(["test"], index, CONFIG, top_k=2, embed_many=lambda c, qs: [[1.0, 0.0]])[0]

    assert [r["repo_id"] for r in single["results"]] == ["beta/repo", "alpha/repo"]
    assert [r["repo_id"] for r in many["results"]] == ["beta/repo", "alpha/repo"]


def test_prepared_index_matches_model_validation():
    raw_index = make_index([{"repo_id": "a/b", "vector": [1.0, 0.0], "metadata": {}}])
    prepared = prepare_index(raw_index, CONFIG)

    diff_config = EmbeddingConfig(
        api_key="k", base_url="http://localhost/v1", model="different-model"
    )
    with pytest.raises(IndexMismatchError, match=r"different-model"):
        prepare_index(prepared, diff_config)


def test_save_and_load_index_v4_dual_file(tmp_path):
    index_file = tmp_path / "index.json"
    doc = make_index(
        [
            {"repo_id": "fastapi/fastapi", "vector": [1.0, 0.0], "metadata": {"name": "fastapi"}},
            {"repo_id": "expressjs/express", "vector": [0.0, 1.0], "metadata": {"name": "express"}},
        ]
    )

    save_index(index_file, doc, version=4)

    assert index_file.exists()
    vectors_file = tmp_path / "index.vectors.npy"
    assert vectors_file.exists()

    raw_json = json.loads(index_file.read_text(encoding="utf-8"))
    assert raw_json["index_version"] == 4
    assert raw_json["vectors_file"] == "index.vectors.npy"
    assert "vector" not in raw_json["vectors"][0]

    loaded = load_index(index_file, mmap=True)
    assert "_matrix" in loaded
    assert loaded["_matrix"].shape == (2, 2)
    assert np.allclose(loaded["_matrix"], [[1.0, 0.0], [0.0, 1.0]])

    prepared = prepare_index(loaded, CONFIG)
    assert isinstance(prepared, PreparedIndex)
    assert prepared.record_count == 2

    res = rank("fastapi", prepared, CONFIG, embed=lambda c, q: [1.0, 0.0])
    assert res["results"][0]["repo_id"] == "fastapi/fastapi"


def test_save_and_load_index_v3_legacy_file(tmp_path):
    index_file = tmp_path / "legacy_index.json"
    doc = make_index(
        [
            {"repo_id": "fastapi/fastapi", "vector": [1.0, 0.0], "metadata": {"name": "fastapi"}},
            {"repo_id": "expressjs/express", "vector": [0.0, 1.0], "metadata": {"name": "express"}},
        ]
    )

    save_index(index_file, doc, version=3)

    assert index_file.exists()
    assert not (tmp_path / "legacy_index.vectors.npy").exists()

    raw_json = json.loads(index_file.read_text(encoding="utf-8"))
    assert raw_json["index_version"] == 3
    assert "vectors_file" not in raw_json
    assert "vector" in raw_json["vectors"][0]

    loaded = load_index(index_file)
    prepared = prepare_index(loaded, CONFIG)
    assert isinstance(prepared, PreparedIndex)
    assert prepared.record_count == 2

    res = rank("fastapi", prepared, CONFIG, embed=lambda c, q: [1.0, 0.0])
    assert res["results"][0]["repo_id"] == "fastapi/fastapi"


def test_hybrid_ranking_strategy_basic_fusion():
    index = make_index(
        [
            {
                "repo_id": "astral-sh/uv",
                "vector": [0.6, 0.8],
                "metadata": {
                    "name": "uv",
                    "summary": "An extremely fast Python package manager written in Rust.",
                    "topics": ["python", "packaging"],
                },
            },
            {
                "repo_id": "pypa/pip",
                "vector": [0.9, 0.4358],
                "metadata": {
                    "name": "pip",
                    "summary": "The PyPA recommended tool for installing Python packages.",
                    "topics": ["python", "pypi"],
                },
            },
        ]
    )

    # Query 'uv' - dense vector is closer to pip [0.9], but BM25 matches 'uv' exclusively
    result = rank(
        "uv",
        index,
        CONFIG,
        ranking_strategy="hybrid",
        embed=lambda c, q: [0.9, 0.4358],  # dense favors pip
    )

    assert result["abstained"] is False
    results = result["results"]
    assert len(results) == 2

    # astral-sh/uv is #1 because BM25 rank #1 + dense rank #2 beats pip's dense rank #1 + BM25 rank None
    assert results[0]["repo_id"] == "astral-sh/uv"
    assert results[0]["ranking_evidence"]["bm25_rank"] == 1
    assert results[0]["ranking_evidence"]["semantic_rank"] == 2
    assert results[0]["score"] == pytest.approx(1.0 / (60 + 2) + 1.0 / (60 + 1), abs=1e-6)

    assert results[1]["repo_id"] == "pypa/pip"
    assert results[1]["ranking_evidence"]["bm25_rank"] is None
    assert results[1]["ranking_evidence"]["semantic_rank"] == 1
    assert results[1]["score"] == pytest.approx(1.0 / (60 + 1), abs=1e-6)


def test_hybrid_pure_semantic_recall():
    index = make_index(
        [
            {
                "repo_id": "expressjs/express",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "express",
                    "summary": "Fast minimalist web framework for Node.js.",
                },
            }
        ]
    )
    # Query with no lexical overlap in summary/name
    result = rank(
        "unrelated terms completely disjoint",
        index,
        CONFIG,
        ranking_strategy="hybrid",
        embed=lambda c, q: [1.0, 0.0],  # perfect dense vector similarity
    )
    assert result["abstained"] is False
    assert result["results"][0]["repo_id"] == "expressjs/express"
    assert result["results"][0]["bm25_score"] == 0.0
    assert result["results"][0]["confidence"] == "high_confidence"


def test_hybrid_ranking_parity_single_and_batch():
    index = make_index(
        [
            {
                "repo_id": "astral-sh/ruff",
                "vector": [1.0, 0.0],
                "metadata": {"name": "ruff", "summary": "Fast Python linter"},
            },
            {
                "repo_id": "astral-sh/uv",
                "vector": [0.0, 1.0],
                "metadata": {"name": "uv", "summary": "Fast Python package installer"},
            },
        ]
    )
    single = rank(
        "ruff",
        index,
        CONFIG,
        ranking_strategy="hybrid",
        embed=lambda c, q: [1.0, 0.0],
    )
    many = rank_many(
        ["ruff"],
        index,
        CONFIG,
        ranking_strategy="hybrid",
        embed_many=lambda c, qs: [[1.0, 0.0]],
    )[0]

    assert [r["repo_id"] for r in single["results"]] == [r["repo_id"] for r in many["results"]]
    for r1, r2 in zip(single["results"], many["results"]):
        assert r1["score"] == pytest.approx(r2["score"], abs=1e-6)
        assert r1["bm25_score"] == pytest.approx(r2["bm25_score"], abs=1e-6)


def test_compute_filter_mask_and_faceted_search_constraints():
    index = make_index(
        [
            {
                "repo_id": "fastapi/fastapi",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "fastapi",
                    "language": "Python",
                    "stars": 75000,
                    "license": "MIT",
                    "ecosystem": ["pypi", "python"],
                    "project_type": "framework",
                    "topics": ["python", "api", "framework", "async"],
                    "archived": False,
                },
            },
            {
                "repo_id": "actix/actix-web",
                "vector": [0.9, 0.4358],
                "metadata": {
                    "name": "actix-web",
                    "language": "Rust",
                    "stars": 21000,
                    "license": "Apache-2.0",
                    "ecosystem": ["cargo", "rust"],
                    "project_type": "framework",
                    "topics": ["rust", "async", "web", "framework"],
                    "archived": False,
                },
            },
            {
                "repo_id": "expressjs/express",
                "vector": [0.0, 1.0],
                "metadata": {
                    "name": "express",
                    "language": "JavaScript",
                    "stars": 64000,
                    "license": "MIT",
                    "ecosystem": ["npm", "javascript"],
                    "project_type": "framework",
                    "topics": ["javascript", "web", "framework"],
                    "archived": False,
                },
            },
            {
                "repo_id": "legacy/old-framework",
                "vector": [0.8, 0.6],
                "metadata": {
                    "name": "old-framework",
                    "language": "Python",
                    "stars": 5000,
                    "license": "BSD-3-Clause",
                    "ecosystem": ["pypi"],
                    "project_type": "framework",
                    "topics": ["python", "web"],
                    "archived": True,
                },
            },
            {
                "repo_id": "ziglang/zig",
                "vector": [0.5, 0.866],
                "metadata": {
                    "name": "zig",
                    "language": "Zig",
                    "stars": 35000,
                    "license": "MIT",
                    "ecosystem": [],
                    "project_type": "compiler",
                    "topics": ["compiler", "systems"],
                    "archived": False,
                },
            },
        ]
    )
    prepared = prepare_index(index, CONFIG)

    # 1. Language alias resolution (py -> Python)
    mask_py = prepared.compute_filter_mask({"language": "py"})
    assert mask_py is not None
    assert np.array_equal(
        mask_py, [True, False, False, False, False]
    )  # old-framework excluded (archived)

    # 2. Language alias (rust -> Rust, js -> JavaScript)
    mask_rust = prepared.compute_filter_mask({"language": "rust"})
    assert mask_rust is not None
    assert np.array_equal(mask_rust, [False, True, False, False, False])

    mask_js = prepared.compute_filter_mask({"language": "js"})
    assert mask_js is not None
    assert np.array_equal(mask_js, [False, False, True, False, False])

    # 3. Custom language without built-in alias (Zig)
    mask_zig = prepared.compute_filter_mask({"language": "zig"})
    assert mask_zig is not None
    assert np.array_equal(mask_zig, [False, False, False, False, True])

    # 4. Star ranges
    mask_stars = prepared.compute_filter_mask({"min_stars": 30000, "max_stars": 70000})
    assert mask_stars is not None
    assert np.array_equal(mask_stars, [False, False, True, False, True])  # express (64k), zig (35k)

    # 5. License filter (case-insensitive)
    mask_mit = prepared.compute_filter_mask({"license": "mit"})
    assert mask_mit is not None
    assert np.array_equal(mask_mit, [True, False, True, False, True])

    mask_apache = prepared.compute_filter_mask({"license": "apache-2.0"})
    assert mask_apache is not None
    assert np.array_equal(mask_apache, [False, True, False, False, False])

    # 6. Ecosystem filter
    mask_pypi = prepared.compute_filter_mask({"ecosystem": "pypi"})
    assert mask_pypi is not None
    assert np.array_equal(mask_pypi, [True, False, False, False, False])

    # 7. Project type filter (normalized)
    mask_compiler = prepared.compute_filter_mask({"project_type": "compiler"})
    assert mask_compiler is not None
    assert np.array_equal(mask_compiler, [False, False, False, False, True])

    # 8. Topics filter (subset)
    mask_topics = prepared.compute_filter_mask({"topics": ["async", "web"]})
    assert mask_topics is not None
    assert np.array_equal(
        mask_topics, [False, True, False, False, False]
    )  # actix-web has both async and web

    # 9. Include archived filter
    mask_archived = prepared.compute_filter_mask({"language": "py", "include_archived": True})
    assert mask_archived is not None
    assert np.array_equal(mask_archived, [True, False, False, True, False])

    # 10. Multi-constraint combination
    res = rank(
        "web framework",
        prepared,
        CONFIG,
        top_k=5,
        ranking_strategy="metadata",
        filters={"language": "rust", "min_stars": 20000, "license": "apache-2.0"},
        embed=lambda c, q: [1.0, 0.0],
    )
    assert res["abstained"] is False
    assert len(res["results"]) == 1
    assert res["results"][0]["repo_id"] == "actix/actix-web"
    assert res["filters"] == {
        "language": "rust",
        "min_stars": 20000,
        "license": "apache-2.0",
    }


def test_filters_across_all_ranking_strategies():
    index = make_index(
        [
            {
                "repo_id": "fastapi/fastapi",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "fastapi",
                    "summary": "FastAPI framework for Python",
                    "language": "Python",
                    "stars": 75000,
                    "license": "MIT",
                },
            },
            {
                "repo_id": "expressjs/express",
                "vector": [0.95, 0.312],
                "metadata": {
                    "name": "express",
                    "summary": "Express web framework for Node.js",
                    "language": "JavaScript",
                    "stars": 64000,
                    "license": "MIT",
                },
            },
            {
                "repo_id": "gin-gonic/gin",
                "vector": [0.9, 0.4358],
                "metadata": {
                    "name": "gin",
                    "summary": "Gin is a HTTP web framework written in Go",
                    "language": "Go",
                    "stars": 76000,
                    "license": "MIT",
                },
            },
        ]
    )

    query_vec = [1.0, 0.0]
    rust_filter = {"language": "rust"}
    py_filter = {"language": "py"}

    # 1. Empty match short-circuits
    res_empty = rank(
        "framework",
        index,
        CONFIG,
        ranking_strategy="semantic",
        filters=rust_filter,
        embed=lambda c, q: query_vec,
    )
    assert res_empty["abstained"] is True
    assert res_empty["results"] == []

    # 2. Semantic strategy with Python filter
    res_sem = rank(
        "framework",
        index,
        CONFIG,
        ranking_strategy="semantic",
        filters=py_filter,
        embed=lambda c, q: query_vec,
    )
    assert len(res_sem["results"]) == 1
    assert res_sem["results"][0]["repo_id"] == "fastapi/fastapi"

    # 3. Hybrid strategy with Python filter
    res_hyb = rank(
        "web framework",
        index,
        CONFIG,
        ranking_strategy="hybrid",
        filters=py_filter,
        embed=lambda c, q: query_vec,
    )
    assert len(res_hyb["results"]) == 1
    assert res_hyb["results"][0]["repo_id"] == "fastapi/fastapi"

    # 4. Rerank strategy with filter - only filtered candidates sent to reranker
    rerank_calls = []

    def mock_reranker(query, docs):
        rerank_calls.append((query, docs))
        return [0.99] * len(docs)

    res_rerank = rank(
        "web framework",
        index,
        CONFIG,
        ranking_strategy="rerank",
        rerank=mock_reranker,
        filters=py_filter,
        embed=lambda c, q: query_vec,
    )
    assert len(res_rerank["results"]) == 1
    assert res_rerank["results"][0]["repo_id"] == "fastapi/fastapi"
    assert len(rerank_calls) == 1
    assert len(rerank_calls[0][1]) == 1  # Only 1 candidate passed to cross-encoder

    # 5. rank_many batch with filter
    batch_res = rank_many(
        ["web framework"],
        index,
        CONFIG,
        filters=py_filter,
        embed_many=lambda c, qs: [query_vec],
    )
    assert len(batch_res[0]["results"]) == 1
    assert batch_res[0]["results"][0]["repo_id"] == "fastapi/fastapi"
