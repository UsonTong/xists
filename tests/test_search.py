import math
from urllib.error import URLError

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
            "language": "JavaScript",
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
    assert index["vectors"][0]["metadata"]["language"] == "JavaScript"
    assert index["vectors"][0]["metadata"]["topics"] == ["frontend", "ui"]
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


def test_rank_reranks_with_metadata():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "generic/framework",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "framework",
                    "description": "A general web application framework.",
                    "topics": ["web"],
                    "language": "Python",
                    "summary": "A broad framework for web projects.",
                    "use_cases": ["web application development"],
                    "capabilities": ["application framework"],
                    "search_phrases": ["frontend framework"],
                },
            },
            {
                "repo_id": "vuejs/core",
                "vector": [0.97, 0.03],
                "metadata": {
                    "name": "core",
                    "description": "Progressive JavaScript framework for building modern web interfaces.",
                    "topics": ["frontend", "vue", "javascript", "ui"],
                    "language": "JavaScript",
                    "summary": "Vue is a frontend framework for user interfaces.",
                    "use_cases": ["building modern web interfaces"],
                    "capabilities": ["reactive components"],
                    "search_phrases": ["progressive framework for building modern web interfaces"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank(
        "progressive framework for building modern web interfaces",
        index,
        CONFIG,
        top_k=1,
        embed=fake_embed,
    )

    assert result["results"][0]["repo_id"] == "vuejs/core"
    assert result["results"][0]["semantic_score"] == pytest.approx(0.999522, abs=0.000001)
    assert result["results"][0]["metadata_score"] > 0.0


def test_rank_keeps_legacy_index_behavior_without_metadata():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {"repo_id": "semantic/winner", "vector": [1.0, 0.0]},
            {"repo_id": "semantic/runner-up", "vector": [0.8, 0.2]},
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("frontend ui", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "semantic/winner"
    assert result["results"][0]["metadata_score"] == 0.0
    assert result["results"][1]["repo_id"] == "semantic/runner-up"


def test_rank_reranks_from_expanded_candidate_pool():
    vectors = [
        {
            "repo_id": f"generic/repo-{index}",
            "vector": [1.0 - index * 0.001, 0.0],
            "metadata": {
                "description": "Generic web framework.",
                "topics": ["web"],
                "language": "Python",
                "summary": "Generic framework.",
                "search_phrases": ["frontend framework"],
            },
        }
        for index in range(12)
    ]
    vectors.append(
        {
            "repo_id": "target/vue",
            "vector": [0.98, 0.0],
            "metadata": {
                "description": "Progressive JavaScript framework for building modern web interfaces.",
                "topics": ["frontend", "vue", "javascript"],
                "language": "JavaScript",
                "summary": "Vue builds modern web interfaces.",
                "search_phrases": ["progressive framework for building modern web interfaces"],
            },
        }
    )
    index = {"embedding_model": "bge-m3", "dimension": 2, "vectors": vectors}

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank(
        "progressive framework for building modern web interfaces",
        index,
        CONFIG,
        top_k=1,
        embed=fake_embed,
    )

    assert result["results"][0]["repo_id"] == "target/vue"


def test_rank_reranks_beyond_legacy_top_50_candidate_pool():
    vectors = [
        {
            "repo_id": f"generic/repo-{index}",
            "vector": [1.0 - index * 0.0001, 0.0],
            "metadata": {
                "description": "Generic infrastructure automation.",
                "topics": ["infrastructure", "automation"],
                "summary": "General automation tooling.",
                "search_phrases": ["infrastructure automation"],
            },
        }
        for index in range(60)
    ]
    vectors.append(
        {
            "repo_id": "ansible/ansible",
            "vector": [0.993, 0.0],
            "metadata": {
                "name": "ansible",
                "description": "IT automation and configuration management.",
                "topics": ["ansible", "automation"],
                "language": "Python",
                "summary": "Agentless configuration management over SSH.",
                "search_phrases": ["agentless configuration management"],
            },
        }
    )
    index = {"embedding_model": "bge-m3", "dimension": 2, "vectors": vectors}

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank(
        "agentless configuration management",
        index,
        CONFIG,
        top_k=1,
        embed=fake_embed,
    )

    assert result["results"][0]["repo_id"] == "ansible/ansible"


def test_rank_rewards_exact_generic_phrase_matches():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "SuperManito/Arcadia",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "Arcadia",
                    "language": "TypeScript",
                    "summary": "A one-stop code automation and operations platform built with TypeScript.",
                    "topics": ["typescript", "automation", "workflow"],
                    "search_phrases": ["workflow automation platform"],
                },
            },
            {
                "repo_id": "n8n-io/n8n",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "n8n",
                    "language": "TypeScript",
                    "summary": "A fair-code workflow automation platform.",
                    "topics": ["typescript", "automation", "workflow"],
                    "search_phrases": ["workflow automation platform"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("TypeScript workflow automation platform", index, CONFIG, top_k=2, embed=fake_embed)

    scores = {item["repo_id"]: item["metadata_score"] for item in result["results"]}
    assert scores["n8n-io/n8n"] == pytest.approx(0.098)
    assert scores["SuperManito/Arcadia"] == pytest.approx(0.098)


def test_rank_many_expands_candidates_for_short_queries(monkeypatch):
    from xists.search import query as query_module

    captured = []

    def fake_candidate_count(top_k, total, *, query_specificity=None, keyword_count=None, semantic_score=None, semantic_gap=None):
        captured.append(
            {
                "top_k": top_k,
                "total": total,
                "query_specificity": query_specificity,
                "keyword_count": keyword_count,
                "semantic_score": semantic_score,
                "semantic_gap": semantic_gap,
            }
        )
        return 750

    monkeypatch.setattr(query_module, "_candidate_count", fake_candidate_count)

    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {"repo_id": f"repo-{index}", "vector": [1.0 - index * 0.0001, 0.0]}
            for index in range(1200)
        ],
    }

    def fake_embed_many(config, queries):
        return [[1.0, 0.0] for _ in queries]

    result = rank_many(["figaro", "android api"], index, CONFIG, top_k=10, batch_size=2, embed_many=fake_embed_many)

    assert len(captured) == 2
    assert all(call["semantic_score"] == pytest.approx(1.0) for call in captured)
    assert all(call["query_specificity"] <= 0.45 for call in captured)
    assert captured[0]["keyword_count"] == 1
    assert captured[1]["keyword_count"] == 1
    assert result[0]["considered"] == 1200


def test_rank_many_groups_queries_by_candidate_count(monkeypatch):
    from xists.search import query as query_module

    calls = []

    def fake_candidate_count(top_k, total, *, query_specificity=None, keyword_count=None, semantic_score=None, semantic_gap=None):
        calls.append(query_specificity)
        if query_specificity <= 0.25:
            return 2
        return 3

    monkeypatch.setattr(query_module, "_candidate_count", fake_candidate_count)

    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {"repo_id": "a/short", "vector": [1.0, 0.0]},
            {"repo_id": "b/short", "vector": [0.9, 0.1]},
            {"repo_id": "c/short", "vector": [0.8, 0.2]},
            {"repo_id": "d/long", "vector": [0.2, 0.8]},
        ],
    }

    def fake_embed_many(config, queries):
        return [[1.0, 0.0] for _ in queries]

    results = rank_many(["figaro", "open source workflow automation platform"], index, CONFIG, top_k=1, batch_size=2, embed_many=fake_embed_many)

    assert calls == [0.25, 0.0]
    assert results[0]["results"][0]["repo_id"] == "a/short"
    assert results[1]["results"][0]["repo_id"] == "a/short"


def test_rank_uses_more_candidates_for_short_queries(monkeypatch):
    from xists.search import query as query_module

    captured = {}

    def fake_candidate_count(top_k, total, *, query_specificity=None, keyword_count=None, semantic_score=None, semantic_gap=None):
        captured["args"] = {
            "top_k": top_k,
            "total": total,
            "query_specificity": query_specificity,
            "keyword_count": keyword_count,
            "semantic_score": semantic_score,
            "semantic_gap": semantic_gap,
        }
        return 600

    monkeypatch.setattr(query_module, "_candidate_count", fake_candidate_count)

    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [{"repo_id": f"repo-{index}", "vector": [1.0 - index * 0.001, 0.0]} for index in range(900)],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    rank("figaro", index, CONFIG, top_k=10, embed=fake_embed)

    assert captured["args"]["query_specificity"] <= 0.25
    assert captured["args"]["keyword_count"] == 1
    assert captured["args"]["total"] == 900


def test_rank_does_not_reward_unmatched_specific_phrases():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "semantic/winner",
                "vector": [1.0, 0.0],
                "metadata": {
                    "description": "A general web framework.",
                    "topics": ["web"],
                    "language": "Python",
                    "summary": "General framework.",
                    "search_phrases": ["frontend framework"],
                },
            },
            {
                "repo_id": "noise/repo",
                "vector": [0.99, 0.0],
                "metadata": {
                    "description": "A general web framework.",
                    "topics": ["web"],
                    "language": "Python",
                    "summary": "General framework.",
                    "search_phrases": ["quantum wasm neural blockchain orchestrator"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("frontend framework", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "semantic/winner"
    unmatched = next(item for item in result["results"] if item["repo_id"] == "noise/repo")
    assert unmatched["metadata_score"] == pytest.approx(0.0)


def test_rank_protects_semantic_winner_for_short_generic_queries():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "pallets/flask",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "flask",
                    "description": "A lightweight WSGI web application framework.",
                    "topics": ["python", "web", "framework", "wsgi"],
                    "language": "Python",
                    "summary": "Flask is a Python web framework.",
                    "search_phrases": ["python web framework"],
                },
            },
            {
                "repo_id": "pyscript/pyscript",
                "vector": [0.99, 0.1],
                "metadata": {
                    "name": "pyscript",
                    "description": "Python web framework for running Python in the browser.",
                    "topics": ["python", "web", "framework"],
                    "language": "Python",
                    "summary": "Python web tools and browser integrations.",
                    "use_cases": ["python web framework"],
                    "capabilities": ["python web framework"],
                    "search_phrases": ["python web framework"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Python web framework", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "pallets/flask"


def test_rank_language_repository_does_not_win_from_language_token_only():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "n8n-io/n8n",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "n8n",
                    "description": "Workflow automation with integrations for popular apps and APIs.",
                    "topics": ["automation", "workflow", "integrations"],
                    "language": "TypeScript",
                    "summary": "n8n connects apps through pre-built integrations.",
                    "capabilities": ["400+ pre-built integrations for popular apps and APIs"],
                    "search_phrases": ["workflow automation integrations"],
                },
            },
            {
                "repo_id": "microsoft/TypeScript",
                "vector": [0.999, 0.045],
                "metadata": {
                    "name": "TypeScript",
                    "description": "TypeScript is a language for application-scale JavaScript.",
                    "topics": ["typescript", "javascript", "language"],
                    "language": "TypeScript",
                    "summary": "The TypeScript language and compiler.",
                    "capabilities": ["static type checking"],
                    "search_phrases": ["typescript language"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank(
        "TypeScript 400+ pre-built integrations for popular apps and APIs",
        index,
        CONFIG,
        top_k=2,
        embed=fake_embed,
    )

    assert result["results"][0]["repo_id"] == "n8n-io/n8n"


def test_rank_boosts_language_matched_exact_profile_phrase():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "BloopAI/bloop",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "bloop",
                    "description": "A fast code search engine written in Rust.",
                    "language": "Rust",
                    "summary": "Code search and AI assistant tooling.",
                    "search_phrases": ["code search engine with natural language"],
                },
            },
            {
                "repo_id": "meilisearch/meilisearch",
                "vector": [0.98, 0.0],
                "metadata": {
                    "name": "meilisearch",
                    "description": "Search engine API with typo tolerance.",
                    "topics": ["search-engine"],
                    "language": "Rust",
                    "summary": "Rust-based search engine API.",
                    "search_phrases": ["open source search engine"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Rust open source search engine", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "meilisearch/meilisearch"


def test_rank_boosts_exact_repo_identity_over_owner_sibling():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "browser-use/web-ui",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "web-ui",
                    "description": "Web UI for browser-use.",
                    "topics": ["browser-use", "ai-agent"],
                    "language": "Python",
                    "summary": "A web interface for browser-use.",
                    "search_phrases": ["browser-use web interface"],
                },
            },
            {
                "repo_id": "browser-use/browser-use",
                "vector": [0.98, 0.0],
                "metadata": {
                    "name": "browser-use",
                    "description": "Browser automation for AI agents.",
                    "topics": ["browser-use", "browser-automation"],
                    "language": "Python",
                    "summary": "A Python library for AI browser automation.",
                    "search_phrases": ["browser-use for agents"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("browser-use", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "browser-use/browser-use"


def test_rank_does_not_overboost_identity_substring_in_long_query():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "nginx/nginx",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "nginx",
                    "description": "Official NGINX web server.",
                    "topics": ["nginx", "web-server"],
                    "language": "C",
                    "summary": "Official NGINX open source repository.",
                    "search_phrases": ["open source web server"],
                },
            },
            {
                "repo_id": "reader/nginx-analysis",
                "vector": [0.99, 0.0],
                "metadata": {
                    "name": "nginx-analysis",
                    "description": "Annotated nginx source code analysis.",
                    "topics": ["nginx"],
                    "language": "C",
                    "summary": "Chinese source code analysis for nginx.",
                    "search_phrases": ["nginx high concurrency design"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("nginx high concurrency design", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "reader/nginx-analysis"


def test_rank_boosts_short_language_matched_exact_profile_phrase():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "quan-to/go-vsm",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "go-vsm",
                    "description": "Vector Space Model implementation in Go.",
                    "topics": ["go", "vector-space-model"],
                    "language": "Go",
                    "summary": "A Go information retrieval library.",
                    "search_phrases": ["vector space model implementation in Go"],
                },
            },
            {
                "repo_id": "milvus-io/milvus",
                "vector": [0.95, 0.0],
                "metadata": {
                    "name": "milvus",
                    "description": "Cloud-native vector database.",
                    "topics": ["vector-database", "golang"],
                    "language": "Go",
                    "summary": "Distributed vector database for ANN search.",
                    "search_phrases": ["vector database"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Go vector database", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "milvus-io/milvus"


def test_rank_treats_short_language_tokens_as_language_evidence():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "generic/vector",
                "vector": [1.0, 0.0],
                "metadata": {
                    "description": "Vector search library.",
                    "language": "Python",
                    "summary": "Vector search tooling.",
                    "search_phrases": ["vector database"],
                },
            },
            {
                "repo_id": "go/vector-db",
                "vector": [0.97, 0.0],
                "metadata": {
                    "description": "Vector database in Go.",
                    "language": "Go",
                    "summary": "Vector database tooling.",
                    "search_phrases": ["vector database"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Go vector database", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "go/vector-db"


def test_rank_treats_vue_language_prefix_as_language_evidence():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "generic/page-builder",
                "vector": [1.0, 0.0],
                "metadata": {
                    "description": "Visual drag-and-drop page builder.",
                    "topics": ["page-builder"],
                    "language": "JavaScript",
                    "summary": "A generic visual page builder.",
                    "search_phrases": ["visual drag-and-drop interface"],
                },
            },
            {
                "repo_id": "target/vue-builder",
                "vector": [0.97, 0.0],
                "metadata": {
                    "description": "Visual drag-and-drop interface for building pages.",
                    "topics": ["vue", "page-builder"],
                    "language": "Vue",
                    "summary": "Vue page builder.",
                    "search_phrases": ["visual drag-and-drop interface"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Vue visual drag-and-drop interface", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "target/vue-builder"


def test_rank_treats_multiword_language_prefix_as_language_evidence():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "generic/notebooks",
                "vector": [1.0, 0.0],
                "metadata": {
                    "description": "Interactive notebook examples.",
                    "topics": ["notebook"],
                    "language": "Python",
                    "summary": "Executable examples.",
                    "search_phrases": ["interactive notebooks with executable code examples"],
                },
            },
            {
                "repo_id": "target/jupyter-examples",
                "vector": [0.97, 0.0],
                "metadata": {
                    "description": "Interactive Jupyter notebooks with executable code examples.",
                    "topics": ["jupyter", "notebook"],
                    "language": "Jupyter Notebook",
                    "summary": "Executable notebook examples.",
                    "search_phrases": ["interactive notebooks with executable code examples"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank(
        "Jupyter Notebook interactive notebooks with executable code examples",
        index,
        CONFIG,
        top_k=2,
        embed=fake_embed,
    )

    assert result["results"][0]["repo_id"] == "target/jupyter-examples"


def test_rank_treats_shell_alias_as_language_evidence():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "generic/dockerfiles",
                "vector": [1.0, 0.0],
                "metadata": {
                    "description": "Docker image build and publish workflow.",
                    "topics": ["docker"],
                    "language": "Dockerfile",
                    "summary": "Container build workflow.",
                    "search_phrases": ["docker image build and publish workflow"],
                },
            },
            {
                "repo_id": "target/shell-release",
                "vector": [0.97, 0.0],
                "metadata": {
                    "description": "Docker image build and publish workflow.",
                    "topics": ["shell", "docker"],
                    "language": "Shell",
                    "summary": "Shell release automation.",
                    "search_phrases": ["docker image build and publish workflow"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("bash docker image build and publish workflow", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "target/shell-release"


def test_rank_uses_language_prefix_as_primary_language_constraint():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "target/java-db",
                "vector": [1.0, 0.0],
                "metadata": {
                    "description": "Android NoSQL database for Java and Kotlin apps.",
                    "topics": ["android", "nosql", "database", "java", "kotlin"],
                    "language": "Java",
                    "summary": "Java database for Android.",
                    "search_phrases": ["android nosql database java kotlin"],
                },
            },
            {
                "repo_id": "noise/kotlin-app",
                "vector": [0.99, 0.0],
                "metadata": {
                    "description": "Android app written in Kotlin.",
                    "topics": ["android", "java", "kotlin"],
                    "language": "Kotlin",
                    "summary": "Kotlin Android client.",
                    "search_phrases": ["android java kotlin"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Java android nosql database java kotlin", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "target/java-db"
    assert result["results"][0]["metadata_score"] > result["results"][1]["metadata_score"]


def test_rank_does_not_treat_negated_language_as_positive_language_evidence():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "semantic/css-patterns",
                "vector": [1.0, 0.0],
                "metadata": {
                    "description": "CSS-only background pattern library.",
                    "topics": ["css", "patterns"],
                    "language": "HTML",
                    "summary": "Pure CSS patterns without JavaScript.",
                    "capabilities": ["Pure CSS implementation with no JavaScript dependency"],
                    "search_phrases": ["pure CSS pattern library"],
                },
            },
            {
                "repo_id": "noise/javascript-free-ui",
                "vector": [0.96, 0.0],
                "metadata": {
                    "description": "CSS-only design system with no JavaScript dependency.",
                    "topics": ["css", "design-system"],
                    "language": "CSS",
                    "summary": "Retro UI components.",
                    "capabilities": ["Pure CSS implementation with no JavaScript dependency"],
                    "search_phrases": ["retro CSS design system"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Pure CSS implementation with no JavaScript dependency", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "semantic/css-patterns"


def test_rank_caps_repeated_metadata_token_evidence():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "semantic/search",
                "vector": [1.0, 0.0],
                "metadata": {
                    "description": "Distributed search engine.",
                    "topics": ["search"],
                    "summary": "Search engine.",
                    "search_phrases": ["distributed search engine"],
                },
            },
            {
                "repo_id": "repeated/search-noise",
                "vector": [0.99, 0.1],
                "metadata": {
                    "description": "Distributed distributed distributed search engine.",
                    "topics": ["distributed", "search", "engine"],
                    "summary": "Distributed search engine distributed search engine.",
                    "use_cases": ["distributed search engine"],
                    "capabilities": ["distributed search engine"],
                    "search_phrases": ["distributed search engine"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("distributed search engine", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "semantic/search"
    noise = next(item for item in result["results"] if item["repo_id"] == "repeated/search-noise")
    assert noise["metadata_score"] <= 0.04


def test_rank_allows_exact_profile_phrase_to_overtake_small_semantic_gap():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "semantic/near-match",
                "vector": [1.0, 0.0],
                "metadata": {
                    "description": "Generic deployment management.",
                    "topics": ["deployment"],
                    "summary": "Deployment tool.",
                    "search_phrases": ["deployment management"],
                },
            },
            {
                "repo_id": "ansible/ansible",
                "vector": [0.99, 0.0],
                "metadata": {
                    "name": "ansible",
                    "description": "IT automation and configuration management.",
                    "topics": ["python", "ansible"],
                    "language": "Python",
                    "summary": "Agentless configuration management over SSH.",
                    "search_phrases": ["agentless configuration management"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("agentless configuration management", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "ansible/ansible"


def test_rank_ignores_language_prefix_for_unique_exact_profile_phrase():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "semantic/near-match",
                "vector": [1.0, 0.0],
                "metadata": {
                    "description": "Generic infrastructure automation.",
                    "topics": ["python", "infrastructure", "automation"],
                    "language": "Python",
                    "summary": "Infrastructure automation framework.",
                    "search_phrases": ["infrastructure automation tool"],
                },
            },
            {
                "repo_id": "ansible/ansible",
                "vector": [0.96, 0.0],
                "metadata": {
                    "name": "ansible",
                    "description": "IT automation and configuration management.",
                    "topics": ["python", "ansible"],
                    "language": "Python",
                    "summary": "Agentless configuration management over SSH.",
                    "search_phrases": ["infrastructure as code tool"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Python infrastructure as code tool", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "ansible/ansible"
    assert result["results"][0]["metadata_score"] > result["results"][1]["metadata_score"]


def test_rank_keeps_semantic_winner_when_exact_phrase_is_not_unique():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "semantic/winner",
                "vector": [1.0, 0.0],
                "metadata": {
                    "description": "A Python web framework.",
                    "topics": ["python", "web", "framework"],
                    "language": "Python",
                    "search_phrases": ["python web framework"],
                },
            },
            {
                "repo_id": "semantic/runner-up",
                "vector": [0.99, 0.0],
                "metadata": {
                    "description": "Another Python web framework.",
                    "topics": ["python", "web", "framework"],
                    "language": "Python",
                    "use_cases": ["python web framework"],
                    "search_phrases": ["python web framework"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Python web framework", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "semantic/winner"


def test_rank_penalizes_alternative_target_identity_match():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "supabase/supabase",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "supabase",
                    "description": "Supabase is an open-source Firebase alternative.",
                    "topics": ["supabase", "firebase", "alternative"],
                    "language": "TypeScript",
                    "summary": "Hosted Postgres backend platform.",
                    "search_phrases": ["supabase vs firebase"],
                },
            },
            {
                "repo_id": "appwrite/appwrite",
                "vector": [0.94, 0.0],
                "metadata": {
                    "name": "appwrite",
                    "description": "Backend-as-a-service platform.",
                    "topics": ["firebase", "supabase", "self-hosted"],
                    "language": "TypeScript",
                    "summary": "Self-hosted Firebase or Supabase alternative.",
                    "use_cases": ["Self-hosting a Firebase or Supabase alternative"],
                    "search_phrases": ["self-hosted backend as a service"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Self-hosting a Firebase or Supabase alternative", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "appwrite/appwrite"


def test_rank_ignores_language_prefix_for_description_phrase():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "kubernetes/kubernetes",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "kubernetes",
                    "description": "Production-grade container orchestration.",
                    "topics": ["go", "kubernetes"],
                    "language": "Go",
                    "summary": "Kubernetes is a container orchestration platform.",
                },
            },
            {
                "repo_id": "helm/helm",
                "vector": [0.99, 0.0],
                "metadata": {
                    "name": "helm",
                    "description": "The Kubernetes Package Manager",
                    "topics": ["go", "kubernetes", "package-manager"],
                    "language": "Go",
                    "summary": "Helm manages Kubernetes charts.",
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Go The Kubernetes Package Manager", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "helm/helm"


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


def test_rank_abstains_when_weak_semantic_has_only_loose_metadata_overlap():
    weak_vector = [0.34, math.sqrt(1.0 - 0.34**2)]
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "loose/overlap",
                "vector": weak_vector,
                "metadata": {
                    "description": "Open source workflow automation platform.",
                    "topics": ["workflow", "automation", "platform"],
                    "summary": "Workflow automation for integrations.",
                },
            }
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("open source workflow automation platform", index, CONFIG, embed=fake_embed)

    assert result["abstained"] is True
    assert result["results"] == []


def test_rank_allows_strong_metadata_evidence_to_rescue_weak_semantic_match():
    weak_vector = [0.34, math.sqrt(1.0 - 0.34**2)]
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "meilisearch/meilisearch",
                "vector": weak_vector,
                "metadata": {
                    "name": "meilisearch",
                    "description": "Open source search engine.",
                    "topics": ["search"],
                    "summary": "Search engine.",
                },
            }
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("meilisearch", index, CONFIG, embed=fake_embed)

    assert result["abstained"] is False
    assert result["results"][0]["repo_id"] == "meilisearch/meilisearch"
    assert result["results"][0]["confidence"] == "high_confidence"


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


def test_rank_many_reranks_with_metadata():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "typesense/typesense",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "typesense",
                    "description": "Open source search engine.",
                    "topics": ["search"],
                    "language": "C++",
                    "summary": "A search engine.",
                    "use_cases": ["full-text search"],
                    "capabilities": ["typo tolerance"],
                    "search_phrases": ["open source search engine"],
                },
            },
            {
                "repo_id": "meilisearch/meilisearch",
                "vector": [0.98, 0.02],
                "metadata": {
                    "name": "meilisearch",
                    "description": "Open source search engine written in Rust with typo tolerance.",
                    "topics": ["rust", "search", "typo-tolerance"],
                    "language": "Rust",
                    "summary": "A Rust search engine focused on typo-tolerant search.",
                    "use_cases": ["building typo-tolerant search"],
                    "capabilities": ["typo tolerance"],
                    "search_phrases": ["open source search engine written in rust with typo tolerance"],
                },
            },
        ],
    }

    def fake_embed_many(config, queries):
        return [[1.0, 0.0] for _ in queries]

    result = rank_many(
        ["open source search engine written in rust with typo tolerance"],
        index,
        CONFIG,
        top_k=1,
        embed_many=fake_embed_many,
    )[0]

    assert result["results"][0]["repo_id"] == "meilisearch/meilisearch"


def test_rank_many_limits_results_after_expanded_candidate_rerank():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "generic/one",
                "vector": [1.0, 0.0],
                "metadata": {"description": "Generic framework.", "topics": ["web"]},
            },
            {
                "repo_id": "target/vue",
                "vector": [0.99, 0.0],
                "metadata": {
                    "description": "Progressive JavaScript framework for building modern web interfaces.",
                    "topics": ["frontend", "vue", "javascript"],
                    "language": "JavaScript",
                    "search_phrases": ["progressive framework for building modern web interfaces"],
                },
            },
            {
                "repo_id": "generic/two",
                "vector": [0.98, 0.0],
                "metadata": {"description": "Generic framework.", "topics": ["web"]},
            },
        ],
    }

    def fake_embed_many(config, queries):
        return [[1.0, 0.0] for _ in queries]

    result = rank_many(
        ["progressive framework for building modern web interfaces"],
        index,
        CONFIG,
        top_k=1,
        embed_many=fake_embed_many,
    )[0]

    assert len(result["results"]) == 1
    assert result["results"][0]["repo_id"] == "target/vue"


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


def test_rank_many_abstains_when_weak_semantic_has_only_loose_metadata_overlap():
    weak_vector = [0.34, math.sqrt(1.0 - 0.34**2)]
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "loose/overlap",
                "vector": weak_vector,
                "metadata": {
                    "description": "Open source workflow automation platform.",
                    "topics": ["workflow", "automation", "platform"],
                    "summary": "Workflow automation for integrations.",
                },
            }
        ],
    }

    def fake_embed_many(config, queries):
        return [[1.0, 0.0] for _ in queries]

    result = rank_many(
        ["open source workflow automation platform"],
        index,
        CONFIG,
        embed_many=fake_embed_many,
    )[0]

    assert result["abstained"] is True
    assert result["results"] == []


def test_rank_expands_candidates_for_short_name_queries(monkeypatch):
    from xists.search import query as query_module

    captured = {}

    def fake_candidate_count(top_k, total, *, query_specificity=None, keyword_count=None, semantic_score=None, semantic_gap=None):
        captured["args"] = {
            "top_k": top_k,
            "total": total,
            "query_specificity": query_specificity,
            "keyword_count": keyword_count,
            "semantic_score": semantic_score,
            "semantic_gap": semantic_gap,
        }
        return 6000

    monkeypatch.setattr(query_module, "_candidate_count", fake_candidate_count)

    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [{"repo_id": f"repo-{index}", "vector": [1.0 - index * 0.0001, 0.0]} for index in range(900)],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    rank("figaro", index, CONFIG, top_k=10, embed=fake_embed)

    assert captured["args"]["query_specificity"] <= 0.25
    assert captured["args"]["keyword_count"] == 1
    assert captured["args"]["total"] == 900


def test_rank_boosts_high_overlap_profile_phrase_without_exact_match():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "zsh-users/zsh",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "zsh",
                    "description": "A powerful shell with scripting and interactive features.",
                    "topics": ["shell", "terminal"],
                    "language": "Shell",
                    "summary": "The Z shell.",
                    "capabilities": ["interactive shell"],
                    "search_phrases": ["shell"],
                },
            },
            {
                "repo_id": "romkatv/powerlevel10k",
                "vector": [0.98, 0.02],
                "metadata": {
                    "name": "powerlevel10k",
                    "description": "A theme for Zsh with rich prompt customization.",
                    "topics": ["zsh", "prompt", "theme"],
                    "language": "Shell",
                    "summary": "Fast Zsh prompt theme.",
                    "capabilities": ["customizing zsh prompt"],
                    "search_phrases": ["customizing zsh prompt for better productivity"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("developer tool for customizing zsh prompt better productivity using zsh", index, CONFIG, top_k=1, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "romkatv/powerlevel10k"
    assert result["results"][0]["metadata_score"] > result["results"][0]["semantic_score"] - 0.98


def test_rank_partial_phrase_boost_stays_below_exact_phrase_boost():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "generic/automation",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "automation",
                    "description": "Automation platform for teams.",
                    "topics": ["workflow", "automation", "platform"],
                    "language": "Python",
                    "summary": "Workflow automation platform.",
                    "capabilities": ["workflow automation"],
                    "search_phrases": ["workflow automation platform"],
                },
            },
            {
                "repo_id": "specific/runner",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "runner",
                    "description": "Run scheduled jobs.",
                    "topics": ["scheduler"],
                    "language": "Python",
                    "summary": "Scheduled jobs runner.",
                    "capabilities": ["scheduled job runner"],
                    "search_phrases": ["scheduled job runner"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    partial = rank("developer tool for workflow automation platform teams", index, CONFIG, top_k=2, embed=fake_embed)
    exact = rank("workflow automation platform", index, CONFIG, top_k=2, embed=fake_embed)

    partial_score = next(item for item in partial["results"] if item["repo_id"] == "generic/automation")["metadata_score"]
    exact_score = next(item for item in exact["results"] if item["repo_id"] == "generic/automation")["metadata_score"]

    assert partial_score < exact_score


def test_rank_combines_multiple_profile_phrases_for_complex_query():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "rustdesk/rustdesk-server",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "rustdesk-server",
                    "description": "Self-hosted RustDesk server.",
                    "topics": ["remote-desktop", "remote-access"],
                    "language": "Rust",
                    "summary": "Self-hosted relay and rendezvous server for RustDesk.",
                    "use_cases": [
                        "Self-hosting a remote desktop server for RustDesk clients",
                        "Managing remote access connections without relying on third-party services",
                    ],
                    "search_phrases": ["self-hosted remote desktop server"],
                },
            },
            {
                "repo_id": "rustdesk/rustdesk",
                "vector": [0.98, 0.02],
                "metadata": {
                    "name": "rustdesk",
                    "description": "Open-source remote desktop application designed for self-hosting.",
                    "topics": ["remote-control", "remote-desktop", "rust"],
                    "language": "Rust",
                    "summary": "Cross-platform remote desktop client for self-hosted support sessions.",
                    "use_cases": [
                        "Providing remote technical support to other users",
                        "Self-hosting a remote desktop service without relying on third-party cloud services",
                    ],
                    "search_phrases": ["open source remote desktop software"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank(
        "Rust open source remote desktop for technical support and self-hosting",
        index,
        CONFIG,
        top_k=2,
        embed=fake_embed,
    )

    assert result["results"][0]["repo_id"] == "rustdesk/rustdesk"
    assert result["results"][0]["metadata_score"] > result["results"][1]["metadata_score"]


def test_rank_prefers_language_matched_candidate_when_metadata_signal_is_otherwise_close():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "noise/font-serif",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "font-serif",
                    "description": "Open source CJK serif font with variable font support.",
                    "topics": ["font", "cjk", "variable-fonts"],
                    "language": "Shell",
                    "summary": "Pan-CJK serif typeface.",
                    "search_phrases": ["Pan-CJK variable font"],
                },
            },
            {
                "repo_id": "target/font-sans",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "font-sans",
                    "description": "Open source CJK font with variable font support.",
                    "topics": ["font", "cjk", "variable-fonts"],
                    "language": "Python",
                    "summary": "Pan-CJK sans typeface.",
                    "search_phrases": ["Pan-CJK font", "variable font for Chinese Japanese Korean"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("Python pan-cjk font variable chinese", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "target/font-sans"
    assert result["results"][0]["metadata_score"] > result["results"][1]["metadata_score"]


def test_rank_description_signal_can_raise_metadata_score_without_overriding_semantics():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "adobe/react-spectrum",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "react-spectrum",
                    "description": "A collection of libraries and tools that help you build adaptive, accessible, and robust user experiences.",
                    "topics": ["react", "accessibility", "design-systems", "ui-components"],
                    "language": "TypeScript",
                    "summary": "React libraries for accessible design systems.",
                    "search_phrases": ["react accessible component library"],
                },
            },
            {
                "repo_id": "radix-ui/primitives",
                "vector": [0.98, 0.02],
                "metadata": {
                    "name": "primitives",
                    "description": "Unstyled UI primitives for React.",
                    "topics": ["react", "accessibility", "design-systems", "ui-components"],
                    "language": "TypeScript",
                    "summary": "Accessible low-level React components.",
                    "search_phrases": ["unstyled UI primitives", "accessible React component library"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank(
        "TypeScript ui ui-components ui-kit for accessible react-based design systems creating",
        index,
        CONFIG,
        top_k=2,
        embed=fake_embed,
    )

    adobe = next(item for item in result["results"] if item["repo_id"] == "adobe/react-spectrum")
    radix = next(item for item in result["results"] if item["repo_id"] == "radix-ui/primitives")
    assert adobe["metadata_score"] >= 0.15
    assert radix["metadata_score"] >= 0.12
    assert adobe["semantic_score"] > radix["semantic_score"]


def test_rank_keeps_profile_phrase_more_authoritative_than_description_only_match():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "amplication/amplication",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "amplication",
                    "description": "Platform engineering tool that generates backend service code from customizable templates.",
                    "topics": ["typescript", "code-generation", "graphql"],
                    "language": "TypeScript",
                    "summary": "Backend code generator and service template platform.",
                    "search_phrases": ["backend code generator", "service template platform"],
                },
            },
            {
                "repo_id": "noise/site-template",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "site-template",
                    "description": "Template repository for generating websites with AI.",
                    "topics": ["template", "typescript", "ai"],
                    "language": "TypeScript",
                    "summary": "Website cloning template.",
                    "search_phrases": ["AI website cloner"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("TypeScript backend code generator service template", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "amplication/amplication"


def test_rank_prefers_framework_over_flexbox_demo_when_query_mentions_framework():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "wcywin/flexbox",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "flexbox",
                    "description": "Responsive landing page made with Flexbox and media queries.",
                    "topics": ["flexbox", "landing-page", "css-framework"],
                    "language": "CSS",
                    "summary": "A static HTML/CSS flexbox demo.",
                    "use_cases": [
                        "Learning or referencing how to build a responsive landing page with Flexbox",
                        "Using as a template for a simple responsive landing page",
                    ],
                    "search_phrases": ["responsive landing page with flexbox"],
                },
            },
            {
                "repo_id": "jgthms/bulma",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "bulma",
                    "description": "Modern CSS framework based on Flexbox",
                    "topics": ["css", "flexbox", "css-framework"],
                    "language": "CSS",
                    "summary": "CSS-only framework for responsive interfaces.",
                    "use_cases": [
                        "Building responsive website layouts",
                        "Creating web interfaces without writing custom CSS",
                    ],
                    "capabilities": [
                        "Flexbox-based grid and column system",
                        "Responsive design utilities",
                    ],
                    "search_phrases": ["flexbox css framework", "responsive css framework"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank(
        "CSS library for responsive website layouts creating web modern css framework based flexbox",
        index,
        CONFIG,
        top_k=2,
        embed=fake_embed,
    )

    assert result["results"][0]["repo_id"] == "jgthms/bulma"


def test_rank_penalizes_template_repo_when_query_is_for_backend_generator():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "noise/site-template",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "site-template",
                    "description": "Template repository for generating websites with AI.",
                    "topics": ["template", "typescript", "ai"],
                    "language": "TypeScript",
                    "summary": "Website cloning template.",
                    "search_phrases": ["AI website cloner"],
                },
            },
            {
                "repo_id": "amplication/amplication",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "amplication",
                    "description": "Platform engineering tool that generates backend service code from customizable templates.",
                    "topics": ["typescript", "code-generation", "graphql"],
                    "language": "TypeScript",
                    "summary": "Backend code generator and service template platform.",
                    "search_phrases": ["backend code generator", "service template platform"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("TypeScript backend code generator service template", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "amplication/amplication"


def test_rank_prefers_base_repo_over_server_variant_when_query_does_not_mention_server():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "rustdesk/rustdesk-server",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "rustdesk-server",
                    "language": "Rust",
                    "summary": "Open-source self-hosted server for remote desktop connections.",
                    "topics": ["remote-desktop", "remote-access", "server"],
                    "search_phrases": ["self-hosted remote desktop server", "rustdesk server"],
                },
            },
            {
                "repo_id": "rustdesk/rustdesk",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "rustdesk",
                    "language": "Rust",
                    "summary": "Open-source remote desktop application and TeamViewer alternative for self-hosting.",
                    "topics": ["remote-desktop", "remote-access", "teamviewer", "p2p"],
                    "search_phrases": ["open source remote desktop software", "self-hosted TeamViewer alternative"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank(
        "Rust open source remote desktop designed self-hosting",
        index,
        CONFIG,
        top_k=2,
        embed=fake_embed,
    )

    assert result["results"][0]["repo_id"] == "rustdesk/rustdesk"


def test_rank_prefers_base_react_native_over_windows_variant_without_windows_cue():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "microsoft/react-native-windows",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "react-native-windows",
                    "language": "C++",
                    "summary": "Extension of React Native for Windows apps.",
                    "topics": ["react-native", "react", "windows", "desktop"],
                    "search_phrases": ["react native windows", "build windows apps with react native"],
                },
            },
            {
                "repo_id": "react/react-native",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "react-native",
                    "language": "C++",
                    "summary": "Framework for building native mobile apps with React.",
                    "topics": ["react-native", "react", "ios", "android", "mobile"],
                    "search_phrases": ["React for mobile apps", "cross-platform mobile framework"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("C++ framework native react", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "react/react-native"


def test_rank_prefers_specific_flutter_plugin_over_archived_plugin_collection():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "flutter-team-archive/plugins",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "plugins",
                    "language": "Dart",
                    "summary": "Archived Flutter team plugin collection for Android and iOS APIs.",
                    "topics": ["flutter", "dart", "plugin", "android", "ios", "flutter-plugin"],
                    "search_phrases": ["flutter official plugins", "flutter android ios plugins"],
                },
            },
            {
                "repo_id": "Wayaer/fl_pip",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "fl_pip",
                    "language": "Dart",
                    "summary": "Flutter picture-in-picture plugin for iOS and Android.",
                    "topics": ["android", "flutter-plugin", "ios", "picture-in-picture"],
                    "search_phrases": ["flutter picture in picture plugin", "pip flutter android ios"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank(
        "Dart android ios flutter picture plugin pip",
        index,
        CONFIG,
        top_k=2,
        embed=fake_embed,
    )

    assert result["results"][0]["repo_id"] == "Wayaer/fl_pip"


def test_rank_prefers_domain_specific_resource_collection_over_generic_awesome_list():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "vinta/awesome-python",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "awesome-python",
                    "language": "Python",
                    "summary": "Curated list of Python frameworks libraries and tools.",
                    "topics": ["awesome", "python", "collections", "python-libraries"],
                    "search_phrases": ["awesome python list", "curated python libraries and frameworks"],
                },
            },
            {
                "repo_id": "fighting41love/funNLP",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "funNLP",
                    "language": "Python",
                    "summary": "Curated collection of NLP resources datasets tools and models.",
                    "topics": ["python", "nlp"],
                    "search_phrases": ["Chinese NLP resource list", "Chinese word segmentation tools"],
                    "use_cases": ["Discovering NLP datasets tools and resources for text processing and segmentation"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank(
        "Python discovering libraries datasets resources text processing segmentation",
        index,
        CONFIG,
        top_k=2,
        embed=fake_embed,
    )

    assert result["results"][0]["repo_id"] == "fighting41love/funNLP"


def test_rank_promotes_stronger_exact_identity_evidence_for_low_specificity_query():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "generic/platform",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "platform",
                    "language": "Python",
                    "summary": "A platform for binary inspection and malware analysis.",
                    "topics": ["platform", "analysis", "binary"],
                    "search_phrases": ["malware analysis platform"],
                },
            },
            {
                "repo_id": "nsa/ghidra-like",
                "vector": [0.995, 0.1],
                "metadata": {
                    "name": "ghidra-like",
                    "language": "Java",
                    "summary": "Reverse engineering tooling for compiled binaries.",
                    "topics": ["reverse-engineering", "disassembler"],
                    "search_phrases": ["binary analysis platform"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("binary analysis platform", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "nsa/ghidra-like"


def test_rank_keeps_semantic_winner_when_no_candidate_has_stronger_structured_evidence():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "generic/platform",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "platform",
                    "language": "Python",
                    "summary": "A platform for binary inspection and malware analysis.",
                    "topics": ["platform", "analysis", "binary"],
                    "search_phrases": ["malware analysis platform"],
                },
            },
            {
                "repo_id": "other/platform",
                "vector": [0.98, 0.2],
                "metadata": {
                    "name": "other-platform",
                    "language": "Python",
                    "summary": "Another binary inspection platform.",
                    "topics": ["platform", "analysis", "binary"],
                    "search_phrases": ["binary inspection platform"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("binary analysis platform", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "generic/platform"


def test_entry_metadata_includes_repository_health_signals(monkeypatch):
    def fake_call(config, inputs, *, timeout=60):
        return [[1.0, 0.0] for _ in inputs]

    monkeypatch.setattr("xists.search.index.call_embeddings", fake_call)
    record = make_record("react/react")
    record["github"].update(
        {
            "stars": 123456,
            "forks": 12000,
            "archived": False,
            "disabled": False,
            "pushed_at": "2026-01-01T00:00:00Z",
        }
    )

    index = build_index([record], CONFIG)

    metadata = index["vectors"][0]["metadata"]
    assert metadata["stars"] == 123456
    assert metadata["forks"] == 12000
    assert metadata["archived"] is False
    assert metadata["disabled"] is False
    assert metadata["pushed_at"] == "2026-01-01T00:00:00Z"


def test_rank_returns_query_intent_and_result_explanations():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "fastapi/fastapi",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "fastapi",
                    "description": "FastAPI framework, high performance, easy to learn, fast to code.",
                    "topics": ["api", "apis", "python", "framework"],
                    "language": "Python",
                    "stars": 90000,
                    "summary": "FastAPI is a Python web framework for building APIs.",
                    "use_cases": ["building Python APIs"],
                    "capabilities": ["async API framework"],
                    "search_phrases": ["python web framework for building apis"],
                },
            }
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("python web framework for building apis", index, CONFIG, embed=fake_embed)

    assert result["query_intent"]["type"] in {"domain", "functional"}
    assert result["query_intent"]["primary_language"] == "python"
    top = result["results"][0]
    assert top["repo_id"] == "fastapi/fastapi"
    assert any(reason.startswith("matched topic:") for reason in top["why"])
    assert "matched language: Python" in top["why"]
    assert "popular repository" in top["why"]
    assert top["score_breakdown"] == {
        "semantic": round(top["semantic_score"], 6),
        "metadata": round(top["metadata_score"], 6),
        "final": round(top["score"], 6),
    }
    assert {"apis", "building"}.issubset(set(top["matched_terms"]))


def test_rank_marks_exact_name_query_intent():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "react/react",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "react",
                    "description": "The library for web and native user interfaces.",
                    "topics": ["frontend", "ui"],
                    "language": "JavaScript",
                },
            }
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("react", index, CONFIG, embed=fake_embed)

    assert result["query_intent"]["type"] == "exact_name"
    assert "exact repo/name match" in result["results"][0]["why"]


def test_repository_state_penalty_can_break_tie_against_archived_repo():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "old/tool",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "tool",
                    "description": "CLI tool for project automation.",
                    "topics": ["cli", "automation"],
                    "language": "Python",
                    "archived": True,
                    "search_phrases": ["cli tool for project automation"],
                },
            },
            {
                "repo_id": "new/tool",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "tool",
                    "description": "CLI tool for project automation.",
                    "topics": ["cli", "automation"],
                    "language": "Python",
                    "archived": False,
                    "search_phrases": ["cli tool for project automation"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("cli tool for project automation", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "new/tool"
    archived = next(item for item in result["results"] if item["repo_id"] == "old/tool")
    assert "archived repository penalty" in archived["why"]


def test_type_cue_strength_promotes_web_framework_over_static_site_framework():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "gohugoio/hugo",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "hugo",
                    "description": "The world fastest framework for building websites.",
                    "topics": ["go", "static-site-generator"],
                    "language": "Go",
                    "search_phrases": ["go static site generator", "hugo framework for websites"],
                },
            },
            {
                "repo_id": "gin-gonic/gin",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "gin",
                    "description": "High-performance HTTP web framework written in Go.",
                    "topics": ["go", "framework", "server", "router"],
                    "language": "Go",
                    "search_phrases": ["high performance Go web framework", "Go REST API framework"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("go web framework", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "gin-gonic/gin"


def test_type_cue_strength_promotes_rag_engine_over_document_tool_when_close():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "PaddlePaddle/PaddleOCR",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "PaddleOCR",
                    "description": "OCR toolkit that converts documents into structured data for RAG applications.",
                    "topics": ["ocr", "document-parsing", "rag"],
                    "language": "Python",
                    "search_phrases": ["RAG document extraction", "OCR for PDF"],
                },
            },
            {
                "repo_id": "infiniflow/ragflow",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "ragflow",
                    "description": "Open-source Retrieval-Augmented Generation RAG engine for LLM applications.",
                    "topics": ["rag", "retrieval-augmented-generation", "llm-apps"],
                    "language": "Go",
                    "search_phrases": ["open source RAG engine", "context layer for LLMs"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("rag engine for document question answering", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "infiniflow/ragflow"


def test_editor_query_prefers_editor_product_over_language_repo():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "rust-lang/rust",
                "vector": [0.97, 0.03],
                "metadata": {
                    "name": "rust",
                    "description": "Empowering everyone to build reliable and efficient software.",
                    "topics": ["compiler", "language", "rust"],
                    "language": "Rust",
                    "summary": "The official repository for the Rust programming language.",
                    "search_phrases": ["Rust programming language", "Rust compiler source code"],
                },
            },
            {
                "repo_id": "zed-industries/zed",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "zed",
                    "description": "High-performance code editor.",
                    "topics": ["text-editor", "rust-lang", "zed"],
                    "language": "Rust",
                    "summary": "Zed is a high-performance, multiplayer code editor written in Rust.",
                    "search_phrases": ["Rust-based editor", "multiplayer code editor"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("modern rust based editor", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "zed-industries/zed"
    assert result["results"][0]["metadata_score"] > result["results"][1]["metadata_score"]


def test_rank_prefers_framework_over_course_when_query_asks_for_llm_framework():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "learning/llm-course",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "llm-course",
                    "description": "Course and roadmap for learning large language models.",
                    "topics": ["course", "llm", "roadmap"],
                    "summary": "A course for building and deploying LLM applications with notebooks.",
                    "search_phrases": ["LLM course", "learn large language models"],
                },
            },
            {
                "repo_id": "langchain-ai/langchain",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "langchain",
                    "description": "The agent engineering platform.",
                    "topics": ["llm", "framework", "agents"],
                    "summary": "Framework for building LLM-powered applications.",
                    "search_phrases": ["LLM application development", "agent framework for LLM"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("framework for building llm applications", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "langchain-ai/langchain"


def test_rank_prefers_frontend_framework_over_backend_framework_for_frontend_query():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "nestjs/nest",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "nest",
                    "description": "Enterprise-grade server-side framework with TypeScript.",
                    "topics": ["typescript", "framework", "server"],
                    "summary": "Backend framework for server-side applications.",
                    "search_phrases": ["TypeScript backend framework"],
                },
            },
            {
                "repo_id": "angular/angular",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "angular",
                    "description": "Web application framework using TypeScript.",
                    "topics": ["typescript", "web-framework", "web"],
                    "summary": "Angular is a platform for enterprise frontend web applications.",
                    "use_cases": ["Building large-scale enterprise web applications"],
                    "search_phrases": ["enterprise typescript frontend framework"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("enterprise typescript frontend framework", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "angular/angular"


def test_rank_prefers_local_model_app_over_provider_aggregator_for_local_chat_query():
    index = {
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {
                "repo_id": "xtekky/gpt4free",
                "vector": [1.0, 0.0],
                "metadata": {
                    "name": "gpt4free",
                    "description": "Provider aggregator with API clients for GPT models.",
                    "topics": ["gpt", "api", "reverse-engineering"],
                    "summary": "Multi-provider LLM aggregator with OpenAI-compatible API.",
                    "search_phrases": ["free chatgpt api", "multi-provider llm aggregator"],
                },
            },
            {
                "repo_id": "nomic-ai/gpt4all",
                "vector": [0.99, 0.01],
                "metadata": {
                    "name": "gpt4all",
                    "description": "Run Local LLMs on Any Device.",
                    "topics": ["ai-chat", "llm-inference"],
                    "summary": "Desktop app for offline local AI chat with private models.",
                    "search_phrases": ["offline AI chat app", "private AI assistant no GPU"],
                },
            },
        ],
    }

    def fake_embed(config, query):
        return [1.0, 0.0]

    result = rank("local gpt chat client and models", index, CONFIG, top_k=2, embed=fake_embed)

    assert result["results"][0]["repo_id"] == "nomic-ai/gpt4all"


def test_rerank_breaks_final_score_tie_with_semantic_score(monkeypatch):
    from xists.search import query as query_module

    results = [
        {
            "repo_id": "semantic/winner",
            "score": 0.8,
            "metadata": {
                "name": "winner",
                "description": "Project automation utility.",
                "topics": [],
                "language": "Python",
            },
        },
        {
            "repo_id": "metadata/winner",
            "score": 0.7,
            "metadata": {
                "name": "winner",
                "description": "Project automation utility.",
                "topics": [],
                "language": "Python",
            },
        },
    ]

    def fake_metadata_score(query, item, **kwargs):
        return 0.1 if item["repo_id"] == "metadata/winner" else 0.0

    monkeypatch.setattr(query_module, "_metadata_score", fake_metadata_score)

    reranked = query_module._rerank_results("project automation", results)

    assert reranked[0]["repo_id"] == "semantic/winner"
    assert reranked[0]["score"] == pytest.approx(reranked[1]["score"])
    assert reranked[0]["semantic_score"] > reranked[1]["semantic_score"]
