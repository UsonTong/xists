from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from xists.search.cache import QueryEmbeddingCache, compute_cache_key
from xists.search.embed import EmbeddingConfig, call_embeddings, embed_query


def test_cache_initialization_and_table_creation(tmp_path: Path):
    db_path = tmp_path / "embeddings.db"
    cache = QueryEmbeddingCache(db_path, max_entries=1000)

    assert cache.enabled is True
    assert db_path.is_file()

    # Check WAL mode and tables
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute("PRAGMA journal_mode;")
        assert cursor.fetchone()[0].lower() in {"wal", "memory"}

        cursor = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='query_cache';"
        )
        assert cursor.fetchone()[0] == 1


def test_cache_single_get_set_and_access_count(tmp_path: Path):
    db_path = tmp_path / "embeddings.db"
    cache = QueryEmbeddingCache(db_path)

    # Miss before set
    assert cache.get("model-a", "find cli tool") is None

    vec = [0.1, 0.2, 0.3, 0.4]
    cache.set("model-a", "find cli tool", vec, dimension=4)

    cached_vec = cache.get("model-a", "find cli tool", dimension=4)
    assert cached_vec is not None
    assert len(cached_vec) == 4
    assert pytest.approx(cached_vec) == vec

    # Different query or model should miss
    assert cache.get("model-b", "find cli tool", dimension=4) is None
    assert cache.get("model-a", "find web framework", dimension=4) is None

    # Check access count updated
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute("SELECT access_count FROM query_cache;")
        # Set sets 1, get updates to 2
        assert cursor.fetchone()[0] == 2


def test_cache_batch_get_set(tmp_path: Path):
    db_path = tmp_path / "embeddings.db"
    cache = QueryEmbeddingCache(db_path)

    queries = ["query 1", "query 2", "query 3"]
    vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    cache.set_batch("test-model", queries, vectors, dimension=2)

    batch_res = cache.get_batch("test-model", ["query 1", "missing", "query 3"], dimension=2)
    assert len(batch_res) == 3
    assert pytest.approx(batch_res[0]) == [0.1, 0.2]
    assert batch_res[1] is None
    assert pytest.approx(batch_res[2]) == [0.5, 0.6]


def test_compute_cache_key_deterministic():
    k1 = compute_cache_key("bge-m3", "  hello world  ", dimension=1024, input_type="query")
    k2 = compute_cache_key("bge-m3", "hello world", dimension=1024, input_type="query")
    k3 = compute_cache_key("bge-m3", "hello world", dimension=512, input_type="query")
    k4 = compute_cache_key("bge-m3", "hello world", dimension=1024, input_type="passage")

    assert k1 == k2
    assert k1 != k3
    assert k1 != k4


def test_cache_lru_eviction(tmp_path: Path):
    db_path = tmp_path / "embeddings.db"
    cache = QueryEmbeddingCache(db_path, max_entries=100)

    # Insert 120 items to trigger eviction down to ~90
    for i in range(120):
        cache.set("model", f"q_{i}", [float(i), 1.0], dimension=2)

    stats = cache.stats()
    assert stats["entry_count"] <= 100
    assert stats["entry_count"] >= 90


def test_cache_clear_and_stats(tmp_path: Path):
    db_path = tmp_path / "embeddings.db"
    cache = QueryEmbeddingCache(db_path)

    cache.set("model", "q1", [1.0, 2.0], dimension=2)
    assert cache.stats()["entry_count"] == 1

    cache.clear()
    assert cache.stats()["entry_count"] == 0
    assert cache.get("model", "q1") is None


def test_cache_disabled_fallback():
    cache = QueryEmbeddingCache(None, enabled=False)
    assert cache.enabled is False
    assert cache.get("m", "q") is None
    cache.set("m", "q", [1.0, 2.0])
    assert cache.stats()["enabled"] is False


def test_embed_query_transparent_caching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "embeddings.db"
    cache = QueryEmbeddingCache(db_path)

    api_mock = MagicMock(return_value=([[0.42, 0.84]], {"attempted": []}))
    monkeypatch.setattr("xists.search.embed._call_embeddings_with_details", api_mock)

    config = EmbeddingConfig(api_key="k", base_url="http://test.local", model="test-m")

    # 1st call: hits remote API mock
    vec1 = embed_query(config, "fast search", cache=cache)
    assert pytest.approx(vec1) == [0.42, 0.84]
    assert api_mock.call_count == 1

    # 2nd call: hits SQLite cache, no API call
    vec2 = embed_query(config, "fast search", cache=cache)
    assert pytest.approx(vec2) == [0.42, 0.84]
    assert api_mock.call_count == 1


def test_call_embeddings_partial_cache_partitioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = tmp_path / "embeddings.db"
    cache = QueryEmbeddingCache(db_path)

    # Pre-populate query 2
    cache.set("test-m", "q2", [0.2, 0.2])

    api_mock = MagicMock(return_value=([[0.1, 0.1], [0.3, 0.3]], {"attempted": []}))
    monkeypatch.setattr("xists.search.embed._call_embeddings_with_details", api_mock)

    config = EmbeddingConfig(api_key="k", base_url="http://test.local", model="test-m")

    inputs = ["q1", "q2", "q3"]
    results = call_embeddings(config, inputs, cache=cache)

    assert len(results) == 3
    assert pytest.approx(results[0]) == [0.1, 0.1]
    assert pytest.approx(results[1]) == [0.2, 0.2]
    assert pytest.approx(results[2]) == [0.3, 0.3]

    # Check only missing ["q1", "q3"] were sent over network
    assert api_mock.call_count == 1
    call_args = api_mock.call_args[0]
    assert call_args[1] == ["q1", "q3"]
