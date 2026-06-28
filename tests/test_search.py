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
