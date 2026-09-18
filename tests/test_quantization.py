"""Tests for vector quantization and chunked memory-safe dot-product operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xists.search.quantize import (
    chunked_dot_product,
    chunked_matrix_similarity,
    dequantize_vector_chunk,
    quantize_vector_matrix,
    save_quantized_vectors,
)


def test_quantize_vector_matrix_modes() -> None:
    rng = np.random.default_rng(42)
    # Unit-normalized 2D vectors
    raw = rng.normal(size=(50, 64)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    matrix = raw / norms

    # 1. Float32 mode
    f32_mat, scales = quantize_vector_matrix(matrix, mode="float32")
    assert f32_mat.dtype == np.float32
    assert scales is None
    assert np.allclose(f32_mat, matrix)

    # 2. Float16 mode
    f16_mat, scales = quantize_vector_matrix(matrix, mode="float16")
    assert f16_mat.dtype == np.float16
    assert scales is None
    assert np.allclose(f16_mat.astype(np.float32), matrix, atol=1e-3)

    # 3. SQ8 / int8 mode
    int8_mat, scales = quantize_vector_matrix(matrix, mode="sq8")
    assert int8_mat.dtype == np.int8
    assert scales is not None
    assert scales.shape == (50,)
    assert np.all(int8_mat >= -127)
    assert np.all(int8_mat <= 127)

    # Check reconstruction fidelity
    reconstructed = int8_mat.astype(np.float32) * scales[:, None]
    assert np.allclose(reconstructed, matrix, atol=0.015)


def test_quantize_vector_matrix_invalid() -> None:
    with pytest.raises(ValueError, match="Expected 2D matrix"):
        quantize_vector_matrix(np.zeros(10))

    with pytest.raises(ValueError, match="Unknown quantization mode"):
        quantize_vector_matrix(np.zeros((10, 10)), mode="invalid_mode")


def test_dequantize_vector_chunk() -> None:
    mat = np.ones((5, 10), dtype=np.float32)
    assert dequantize_vector_chunk(mat).dtype == np.float32

    mat_f16 = np.ones((5, 10), dtype=np.float16)
    assert dequantize_vector_chunk(mat_f16).dtype == np.float32

    mat_int8 = np.full((5, 10), 10, dtype=np.int8)
    scales = np.full(5, 0.1, dtype=np.float32)
    deq = dequantize_vector_chunk(mat_int8, scales=scales)
    assert deq.dtype == np.float32
    assert np.allclose(deq, 1.0)


def test_chunked_dot_product_fidelity() -> None:
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(200, 128)).astype(np.float32)
    matrix = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    query = rng.normal(size=128).astype(np.float32)
    query /= np.linalg.norm(query)

    expected = np.dot(matrix, query)

    # 1. Float32 chunked
    scores_f32 = chunked_dot_product(matrix, query, chunk_size=32)
    assert np.allclose(scores_f32, expected, atol=1e-5)

    # 2. Float16 chunked
    matrix_f16, _ = quantize_vector_matrix(matrix, mode="float16")
    scores_f16 = chunked_dot_product(matrix_f16, query, chunk_size=32)
    corr_f16 = np.corrcoef(expected, scores_f16)[0, 1]
    assert corr_f16 > 0.99999
    assert np.allclose(scores_f16, expected, atol=1e-3)

    # 3. SQ8 int8 chunked
    matrix_int8, scales = quantize_vector_matrix(matrix, mode="sq8")
    scores_sq8 = chunked_dot_product(matrix_int8, query, scales=scales, chunk_size=32)
    corr_sq8 = np.corrcoef(expected, scores_sq8)[0, 1]
    assert corr_sq8 > 0.999
    assert np.allclose(scores_sq8, expected, atol=0.015)

    # Test top-5 rankings alignment
    top5_expected = np.argsort(expected)[-5:]
    top5_sq8 = np.argsort(scores_sq8)[-5:]
    assert set(top5_expected) == set(top5_sq8)


def test_chunked_dot_product_empty() -> None:
    empty_mat = np.empty((0, 16), dtype=np.float32)
    query = np.ones(16, dtype=np.float32)
    scores = chunked_dot_product(empty_mat, query)
    assert len(scores) == 0


def test_chunked_matrix_similarity() -> None:
    rng = np.random.default_rng(456)
    matrix = rng.normal(size=(100, 32)).astype(np.float32)
    queries = rng.normal(size=(5, 32)).astype(np.float32)

    expected = np.dot(matrix, queries.T)

    # Float32 chunked
    sim_f32 = chunked_matrix_similarity(matrix, queries, chunk_size=20)
    assert np.allclose(sim_f32, expected, atol=1e-5)

    # SQ8 int8 chunked
    matrix_int8, scales = quantize_vector_matrix(matrix, mode="sq8")
    sim_sq8 = chunked_matrix_similarity(matrix_int8, queries, scales=scales, chunk_size=20)
    corr = np.corrcoef(expected.flatten(), sim_sq8.flatten())[0, 1]
    assert corr > 0.999


def test_save_quantized_vectors(tmp_path: Path) -> None:
    rng = np.random.default_rng(789)
    matrix = rng.normal(size=(20, 16)).astype(np.float32)

    # 1. Save Float16
    f16_path = tmp_path / "vecs_f16.npy"
    meta_f16 = save_quantized_vectors(f16_path, matrix, mode="float16")
    assert f16_path.exists()
    assert meta_f16["vector_dtype"] == "float16"
    assert meta_f16["vector_quantization"] == "none"
    loaded_f16 = np.load(f16_path)
    assert loaded_f16.dtype == np.float16

    # 2. Save SQ8
    sq8_path = tmp_path / "vecs_sq8.npy"
    meta_sq8 = save_quantized_vectors(sq8_path, matrix, mode="sq8")
    assert sq8_path.exists()
    assert meta_sq8["vector_dtype"] == "int8"
    assert meta_sq8["vector_quantization"] == "sq8"
    assert meta_sq8["scales_file"] is not None
    scales_path = tmp_path / meta_sq8["scales_file"]
    assert scales_path.exists()
    loaded_sq8 = np.load(sq8_path)
    loaded_scales = np.load(scales_path)
    assert loaded_sq8.dtype == np.int8
    assert loaded_scales.dtype == np.float32


def test_save_and_load_quantized_index_end_to_end(tmp_path: Path) -> None:
    from xists.search.embed import EMBEDDING_INPUT_VERSION, EmbeddingConfig
    from xists.search.index import save_index
    from xists.search.query import PreparedIndex, rank

    rng = np.random.default_rng(999)
    raw_vecs = rng.normal(size=(10, 32)).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    matrix = raw_vecs / norms

    vectors = [
        {
            "repo_id": f"org/repo-{i}",
            "embedding_input_fingerprint": f"fp-{i}",
            "metadata": {
                "name": f"repo-{i}",
                "description": f"Test repo {i} for quantization",
                "summary": f"Summary for repo {i}",
                "stars": i * 100,
            },
        }
        for i in range(10)
    ]

    index_doc = {
        "index_version": 4,
        "record_schema_version": 2,
        "embedding_model": "test-model",
        "embedding_base_url": "https://api.example.com",
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
        "dimension": 32,
        "built_at": "2026-09-13T00:00:00Z",
        "record_count": 10,
        "vectors": vectors,
        "_matrix": matrix,
    }

    config = EmbeddingConfig(
        model="test-model", base_url="https://api.example.com", api_key="test-key"
    )

    # 1. Test Float16 index
    f16_index_path = tmp_path / "f16" / "index.json"
    save_index(f16_index_path, index_doc, matrix=matrix, quantize="float16")
    prepared_f16 = PreparedIndex.from_path(f16_index_path, config)
    assert prepared_f16.vector_dtype == "float16"
    assert prepared_f16.vector_quantization == "none"

    query_vec = matrix[3]

    def dummy_embed(_cfg, _q, **_kw):
        return query_vec.tolist()

    res_f16 = rank("find repo 3", prepared_f16, config, embed=dummy_embed, top_k=3)
    assert len(res_f16["results"]) > 0
    assert res_f16["results"][0]["repo_id"] == "org/repo-3"

    # 2. Test SQ8 index
    sq8_index_path = tmp_path / "sq8" / "index.json"
    save_index(sq8_index_path, index_doc, matrix=matrix, quantize="sq8")
    prepared_sq8 = PreparedIndex.from_path(sq8_index_path, config)
    assert prepared_sq8.vector_dtype == "int8"
    assert prepared_sq8.vector_quantization == "sq8"
    assert prepared_sq8.scales is not None
    assert len(prepared_sq8.scales) == 10

    res_sq8 = rank("find repo 3", prepared_sq8, config, embed=dummy_embed, top_k=3)
    assert len(res_sq8["results"]) > 0
    assert res_sq8["results"][0]["repo_id"] == "org/repo-3"

    # Verify vector helper methods
    vec_0 = prepared_sq8.get_vector(0)
    assert vec_0.dtype == np.float32
    assert np.allclose(vec_0, matrix[0], atol=0.02)

    sub_vecs = prepared_sq8.get_vectors([0, 1, 2])
    assert sub_vecs.dtype == np.float32
    assert sub_vecs.shape == (3, 32)
    assert np.allclose(sub_vecs, matrix[:3], atol=0.02)
