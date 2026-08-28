import numpy as np
import pytest
from pathlib import Path

from xists.search.local_embed import (
    LOCAL_MODEL_REGISTRY,
    compute_deterministic_text_embedding,
    embed_query_local,
    get_cache_model_dir,
    is_onnxruntime_available,
)


def test_deterministic_text_embedding_returns_unit_vector():
    vec = compute_deterministic_text_embedding("python web framework", dimension=1024)
    assert len(vec) == 1024
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    assert abs(norm - 1.0) < 1e-5


def test_deterministic_text_embedding_similarity():
    vec1 = compute_deterministic_text_embedding("fastapi high performance web framework", dimension=1024)
    vec2 = compute_deterministic_text_embedding("fastapi python web framework", dimension=1024)
    vec_unrelated = compute_deterministic_text_embedding("cooking recipe Italian pasta sauce", dimension=1024)

    dot_similar = np.dot(vec1, vec2)
    dot_unrelated = np.dot(vec1, vec_unrelated)

    # Similar queries should have higher dot product than unrelated
    assert dot_similar > dot_unrelated
    assert dot_similar > 0.3


def test_local_model_registry_and_cache_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("XISTS_CACHE_DIR", str(tmp_path / "cache_models"))
    model_dir = get_cache_model_dir("bge-small-en-v1.5")
    assert model_dir.is_dir()
    assert "bge-small-en-v1.5" in LOCAL_MODEL_REGISTRY


def test_embed_query_local_fallback():
    vec = embed_query_local("vector search database", dimension=512)
    assert len(vec) == 512
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
