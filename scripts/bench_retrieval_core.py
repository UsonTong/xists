"""Comprehensive benchmark suite for xists retrieval core optimization.

Measures and quantifies:
1. Memory-mapped (mmap) zero-copy vector loading latency and memory allocation vs full RAM loading.
2. BM25 inverted index incremental sync (append & prune) vs full corpus re-tokenization.
3. Composite boolean facet expression compilation and hardware bitmask pre-filtering throughput.

Usage:
    uv run python scripts/bench_retrieval_core.py [--count 5000] [--dim 1024]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from xists.search.bm25 import BM25Index
from xists.search.cache import QueryEmbeddingCache
from xists.search.embed import EmbeddingConfig
from xists.search.facets import evaluate_ast_mask, parse_facet_query
from xists.search.index import load_index, save_index
from xists.search.query import PreparedIndex, rank, rank_many

CONFIG = EmbeddingConfig(api_key="bench", base_url="http://bench.invalid/v1", model="bge-m3")


def generate_benchmark_dataset(
    count: int = 5000, dimension: int = 1024
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Generate synthetic repository entries and L2-normalized float32 vectors."""
    languages = ["Python", "Rust", "JavaScript", "TypeScript", "Go", "C++", "Java"]
    ecosystems = [["pypi"], ["cargo"], ["npm"], ["pypi", "cargo"], ["go"], ["crates.io"]]
    project_types = ["framework", "library", "cli_tool", "database", "compiler", "runtime"]
    licenses = ["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "MPL-2.0"]
    topic_pool = [
        "web",
        "async",
        "api",
        "cli",
        "ml",
        "distributed",
        "database",
        "parser",
        "cloud",
        "security",
    ]

    rng = np.random.RandomState(42)
    raw_matrix = rng.standard_normal((count, dimension)).astype(np.float32)
    norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
    matrix = np.divide(raw_matrix, norms, out=np.zeros_like(raw_matrix), where=norms != 0)

    entries: list[dict[str, Any]] = []
    for i in range(count):
        lang = languages[i % len(languages)]
        eco = ecosystems[i % len(ecosystems)]
        pt = project_types[i % len(project_types)]
        lic = licenses[i % len(licenses)]
        topics = [topic_pool[(i + j) % len(topic_pool)] for j in range((i % 3) + 1)]
        stars = int((i * 37) % 85000) + 100
        forks = int(stars * 0.08) + 10
        archived = (i % 25) == 0

        entries.append(
            {
                "repo_id": f"bench-org/repo-{i:05d}",
                "metadata": {
                    "name": f"repo-{i:05d}",
                    "summary": f"Synthetic benchmark repository {i} written in {lang} for {pt} and {topics[0]}.",
                    "language": lang,
                    "ecosystem": eco,
                    "project_type": pt,
                    "license": lic,
                    "stars": stars,
                    "forks": forks,
                    "archived": archived,
                    "topics": topics,
                },
            }
        )

    return entries, matrix


def benchmark_mmap_loading(
    entries: list[dict[str, Any]], matrix: np.ndarray, temp_dir: Path
) -> dict[str, Any]:
    """Measure load latency and allocated memory for mmap zero-copy vs full RAM load."""
    idx_path = temp_dir / "bench_index.json"
    doc = {
        "index_version": 4,
        "embedding_input_version": 3,
        "record_schema_version": 2,
        "embedding_model": "bge-m3",
        "dimension": matrix.shape[1],
        "record_count": len(entries),
        "vectors": entries,
    }
    save_index(idx_path, doc, matrix=matrix, version=4)

    # 1. Full memory load (mmap=False)
    t0 = time.perf_counter()
    full_doc = load_index(idx_path, mmap=False)
    full_prep = PreparedIndex.from_dict(full_doc, CONFIG, mmap=False)
    full_time_ms = (time.perf_counter() - t0) * 1000

    tracemalloc.start()
    _fd = load_index(idx_path, mmap=False)
    _fp = PreparedIndex.from_dict(_fd, CONFIG, mmap=False)
    _, full_peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 2. Zero-copy memory mapped load (mmap=True)
    t0 = time.perf_counter()
    mmap_doc = load_index(idx_path, mmap=True)
    mmap_prep = PreparedIndex.from_dict(mmap_doc, CONFIG, mmap=True)
    mmap_time_ms = (time.perf_counter() - t0) * 1000

    tracemalloc.start()
    _md = load_index(idx_path, mmap=True)
    _mp = PreparedIndex.from_dict(_md, CONFIG, mmap=True)
    _, mmap_peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert mmap_prep.is_mmap
    assert not full_prep.is_mmap

    return {
        "vector_count": len(entries),
        "dimension": matrix.shape[1],
        "raw_matrix_bytes": matrix.nbytes,
        "mmap_false": {
            "load_time_ms": round(full_time_ms, 3),
            "peak_allocated_kb": round(full_peak_ram / 1024, 2),
        },
        "mmap_true": {
            "load_time_ms": round(mmap_time_ms, 3),
            "peak_allocated_kb": round(mmap_peak_ram / 1024, 2),
        },
        "speedup": round(full_time_ms / max(mmap_time_ms, 0.001), 2),
        "ram_reduction_ratio": round(full_peak_ram / max(mmap_peak_ram, 1), 2),
    }


def benchmark_bm25_incremental(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure performance of incremental BM25 append and prune vs full rebuild."""
    split_idx = int(len(entries) * 0.9)
    initial_entries = entries[:split_idx]
    appended_entries = entries[split_idx:]

    # 1. Build initial index
    bm25 = BM25Index.build_from_entries(initial_entries)

    # 2. Incremental append (10% batch)
    t0 = time.perf_counter()
    bm25.append_entries(appended_entries)
    incr_append_ms = (time.perf_counter() - t0) * 1000

    # 3. Full rebuild of all items
    t0 = time.perf_counter()
    BM25Index.build_from_entries(entries)
    full_rebuild_ms = (time.perf_counter() - t0) * 1000

    # 4. Incremental lifecycle prune (prune 5% archived / stale repos)
    lifecycle_keep_indices = [i for i in range(len(entries)) if (i % 20) != 0]
    t0 = time.perf_counter()
    bm25.prune_indices(lifecycle_keep_indices)
    lifecycle_prune_ms = (time.perf_counter() - t0) * 1000

    # 5. Full rebuild of pruned items
    lifecycle_pruned_entries = [entries[i] for i in lifecycle_keep_indices]
    t0 = time.perf_counter()
    BM25Index.build_from_entries(lifecycle_pruned_entries)
    full_lifecycle_rebuild_ms = (time.perf_counter() - t0) * 1000

    return {
        "corpus_docs": len(entries),
        "append_batch_size": len(appended_entries),
        "append_comparison": {
            "incremental_append_ms": round(incr_append_ms, 3),
            "full_rebuild_ms": round(full_rebuild_ms, 3),
            "speedup": round(full_rebuild_ms / max(incr_append_ms, 0.001), 2),
        },
        "lifecycle_prune_comparison": {
            "retained_docs": len(lifecycle_keep_indices),
            "pruned_docs": len(entries) - len(lifecycle_keep_indices),
            "incremental_prune_ms": round(lifecycle_prune_ms, 3),
            "full_rebuild_ms": round(full_lifecycle_rebuild_ms, 3),
            "speedup": round(full_lifecycle_rebuild_ms / max(lifecycle_prune_ms, 0.001), 2),
        },
    }


def benchmark_facet_filtering(entries: list[dict[str, Any]], matrix: np.ndarray) -> dict[str, Any]:
    """Measure throughput of composite boolean facet queries and candidate acceleration."""
    index_dict = {
        "index_version": 4,
        "embedding_input_version": 3,
        "record_schema_version": 2,
        "embedding_model": "bge-m3",
        "dimension": matrix.shape[1],
        "record_count": len(entries),
        "vectors_normalized": True,
        "_matrix": matrix,
        "vectors": entries,
    }
    prepared = PreparedIndex.from_dict(index_dict, CONFIG)

    test_queries = [
        "(lang:python OR lang:rust) AND stars:>20k AND NOT is:archived",
        "eco:pypi AND type:framework AND license:mit",
        "(lang:go OR lang:typescript) AND stars:5k..50k AND !is:disabled",
        "topic:async AND topic:web AND forks:>100",
        "(lang:c++ OR lang:rust) AND (type:compiler OR type:database)",
    ]

    # Warmup
    for q in test_queries:
        prepared.compute_filter_mask(q)

    # 1. Measure AST parsing and bitmask evaluation throughput
    iterations = 200
    t0 = time.perf_counter()
    total_evals = 0
    for _ in range(iterations):
        for q in test_queries:
            ast = parse_facet_query(q)
            assert ast is not None
            _mask = evaluate_ast_mask(ast, prepared)
            total_evals += 1
    total_time = time.perf_counter() - t0
    ops_per_sec = total_evals / total_time
    avg_latency_us = (total_time / total_evals) * 1_000_000

    # 2. Measure search acceleration with pre-filtering vs unconstrained
    query_vec = matrix[0].tolist()

    # Unconstrained search
    t0 = time.perf_counter()
    for _ in range(50):
        rank("test query", prepared, CONFIG, top_k=10, embed=lambda c, q: query_vec)
    unconstrained_ms = (time.perf_counter() - t0) * 1000 / 50

    # Pre-filtered search (selects ~5% of corpus)
    t0 = time.perf_counter()
    for _ in range(50):
        rank(
            "test query",
            prepared,
            CONFIG,
            top_k=10,
            filters="(lang:python OR lang:rust) AND stars:>60k AND topic:async",
            embed=lambda c, q: query_vec,
        )
    filtered_ms = (time.perf_counter() - t0) * 1000 / 50

    return {
        "candidate_pool_size": len(entries),
        "facet_compilation_and_bitmask": {
            "total_queries_evaluated": total_evals,
            "throughput_queries_per_sec": round(ops_per_sec, 1),
            "avg_latency_microseconds": round(avg_latency_us, 2),
        },
        "search_acceleration": {
            "unconstrained_search_ms": round(unconstrained_ms, 3),
            "prefiltered_search_ms": round(filtered_ms, 3),
            "search_speedup": round(unconstrained_ms / max(filtered_ms, 0.001), 2),
        },
    }


def benchmark_cold_start_and_lazy_retrieval(
    entries: list[dict[str, Any]], matrix: np.ndarray, temp_dir: Path
) -> dict[str, Any]:
    """Measure cold-start loading time and retrieval latency for SQLite sidecar vs legacy monolithic JSON."""
    idx_path = temp_dir / "cold_start_index.json"
    doc = {
        "index_version": 4,
        "embedding_input_version": 3,
        "record_schema_version": 2,
        "embedding_model": "bge-m3",
        "dimension": matrix.shape[1],
        "record_count": len(entries),
        "vectors": entries,
    }
    save_index(idx_path, doc, matrix=matrix, version=4)

    # 1. Legacy cold start: load monolithic JSON + precompute caches in Python from scratch
    # Measure timing without tracemalloc overhead
    t0 = time.perf_counter()
    legacy_doc = load_index(idx_path, mmap=True)
    legacy_doc.pop("_meta_db_path", None)
    _ = PreparedIndex.from_dict(legacy_doc, CONFIG, mmap=True)
    legacy_cold_ms = (time.perf_counter() - t0) * 1000

    # Measure peak RAM in a separate pass
    tracemalloc.start()
    leg_doc = load_index(idx_path, mmap=True)
    leg_doc.pop("_meta_db_path", None)
    _ = PreparedIndex.from_dict(leg_doc, CONFIG, mmap=True)
    _, legacy_peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 2. Optimized cold start: fast SQLite sidecar loading via PreparedIndex.from_path
    # Measure timing without tracemalloc overhead
    t0 = time.perf_counter()
    sidecar_prep = PreparedIndex.from_path(idx_path, CONFIG, mmap=True)
    sidecar_cold_ms = (time.perf_counter() - t0) * 1000

    # Measure peak RAM in a separate pass
    tracemalloc.start()
    _ = PreparedIndex.from_path(idx_path, CONFIG, mmap=True)
    _, sidecar_peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 3. Two-Stage Top-K Hybrid Retrieval latency benchmark
    query_vec = matrix[0].tolist()
    query_text = "web async library in python"

    runs = 50
    t0 = time.perf_counter()
    for _ in range(runs):
        res = rank(query_text, sidecar_prep, CONFIG, top_k=10, embed=lambda c, q: query_vec)
        assert len(res) <= 10
    sidecar_search_ms = (time.perf_counter() - t0) * 1000 / runs

    return {
        "record_count": len(entries),
        "dimension": matrix.shape[1],
        "cold_start": {
            "legacy_json_ms": round(legacy_cold_ms, 3),
            "legacy_peak_kb": round(legacy_peak_ram / 1024, 2),
            "sqlite_sidecar_ms": round(sidecar_cold_ms, 3),
            "sqlite_peak_kb": round(sidecar_peak_ram / 1024, 2),
            "cold_start_speedup": round(legacy_cold_ms / max(sidecar_cold_ms, 0.001), 2),
            "ram_reduction_ratio": round(legacy_peak_ram / max(sidecar_peak_ram, 1), 2),
        },
        "retrieval": {
            "hybrid_search_ms": round(sidecar_search_ms, 3),
            "queries_per_sec": round(1000.0 / max(sidecar_search_ms, 0.001), 1),
        },
    }


def benchmark_query_embedding_cache(temp_dir: Path, dimension: int = 1024) -> dict[str, Any]:
    """Measure persistent SQLite query embedding cache performance vs simulated remote API calls."""
    db_path = temp_dir / "bench_embed_cache.db"
    cache = QueryEmbeddingCache(db_path)

    queries = [f"benchmark search query {i} for high throughput testing" for i in range(100)]
    rng = np.random.RandomState(42)
    raw_vectors = rng.standard_normal((len(queries), dimension)).astype(np.float32)
    norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    vectors = np.divide(
        raw_vectors, norms, out=np.zeros_like(raw_vectors), where=norms != 0
    ).tolist()

    # 1. Warm-up writes (set_batch)
    t0 = time.perf_counter()
    cache.set_batch("bge-m3", queries, vectors)
    batch_write_time_ms = (time.perf_counter() - t0) * 1000

    # 2. Benchmark cache hit reads (get_batch)
    iterations = 50
    t0 = time.perf_counter()
    for _ in range(iterations):
        hits = cache.get_batch("bge-m3", queries)
        assert len(hits) == len(queries)
    batch_read_time = time.perf_counter() - t0
    batch_read_ms = (batch_read_time * 1000) / (iterations * len(queries))
    cache_read_qps = (iterations * len(queries)) / batch_read_time

    # 3. Single query cache get/set micro-benchmark
    t0 = time.perf_counter()
    for _ in range(200):
        v = cache.get("bge-m3", queries[0])
        assert v is not None
    single_get_us = ((time.perf_counter() - t0) / 200) * 1_000_000

    # 4. End-to-end simulated comparison: 200ms remote embedding call vs cache hit
    simulated_remote_ms = 200.0  # typical remote API round trip + inference
    cache_hit_ms = single_get_us / 1000.0
    speedup = simulated_remote_ms / max(cache_hit_ms, 0.001)

    return {
        "cache_entries": len(queries),
        "vector_dimension": dimension,
        "batch_write_ms": round(batch_write_time_ms, 3),
        "single_get_microseconds": round(single_get_us, 2),
        "batch_get_ms_per_query": round(batch_read_ms, 4),
        "throughput_queries_per_sec": round(cache_read_qps, 1),
        "simulated_network_ms": simulated_remote_ms,
        "cache_hit_speedup": round(speedup, 1),
    }


def benchmark_batched_gemm_retrieval(
    entries: list[dict[str, Any]], matrix: np.ndarray, temp_dir: Path, batch_size: int = 32
) -> dict[str, Any]:
    """Measure multi-query evaluation throughput: Sequential single-query vs Batched GEMM + Prefetching."""
    idx_path = temp_dir / "batched_gemm_index.json"
    doc = {
        "index_version": 4,
        "embedding_input_version": 3,
        "record_schema_version": 2,
        "embedding_model": "bge-m3",
        "dimension": matrix.shape[1],
        "record_count": len(entries),
        "vectors": entries,
    }
    save_index(idx_path, doc, matrix=matrix, version=4)
    prepared = PreparedIndex.from_path(idx_path, CONFIG, mmap=True)

    rng = np.random.RandomState(99)
    raw_query_vectors = rng.standard_normal((batch_size, matrix.shape[1])).astype(np.float32)
    norms = np.linalg.norm(raw_query_vectors, axis=1, keepdims=True)
    query_vectors = np.divide(
        raw_query_vectors, norms, out=np.zeros_like(raw_query_vectors), where=norms != 0
    ).tolist()
    queries = [f"benchmark multi query {i} async web framework" for i in range(batch_size)]

    # 1. Sequential single-query rank()
    runs = 3
    t0 = time.perf_counter()
    for _ in range(runs):
        for q, vec in zip(queries, query_vectors):
            _res = rank(q, prepared, CONFIG, top_k=10, embed=lambda c, _q, v=vec: v)
    seq_time = (time.perf_counter() - t0) / runs
    seq_latency_ms = (seq_time * 1000) / batch_size
    seq_qps = batch_size / seq_time

    # 2. Batched rank_many() with BLAS GEMM matrix multiplication & global prefetch
    t0 = time.perf_counter()
    for _ in range(runs):
        _batch_res = rank_many(
            queries, prepared, CONFIG, top_k=10, embed_many=lambda c, _qs: query_vectors
        )
    batch_time = (time.perf_counter() - t0) / runs
    batch_latency_ms = (batch_time * 1000) / batch_size
    batch_qps = batch_size / batch_time

    return {
        "corpus_size": len(entries),
        "batch_size": batch_size,
        "sequential_single_query": {
            "total_time_ms": round(seq_time * 1000, 2),
            "latency_ms_per_query": round(seq_latency_ms, 3),
            "throughput_qps": round(seq_qps, 1),
        },
        "batched_gemm_prefetch": {
            "total_time_ms": round(batch_time * 1000, 2),
            "latency_ms_per_query": round(batch_latency_ms, 3),
            "throughput_qps": round(batch_qps, 1),
        },
        "batch_speedup": round(batch_qps / max(seq_qps, 0.001), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5000, help="Number of synthetic vectors")
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dimension")
    args = parser.parse_args()

    print("=== xists Retrieval Core Optimization Benchmark ===")
    print(f"Generating synthetic dataset: {args.count} records, {args.dim} dimensions...")
    entries, matrix = generate_benchmark_dataset(count=args.count, dimension=args.dim)
    print(f"Dataset generated. Matrix size: {matrix.nbytes / (1024 * 1024):.2f} MB.\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)

        print("1. Benchmarking mmap zero-copy vector loading...")
        mmap_res = benchmark_mmap_loading(entries, matrix, temp_path)
        print(
            f"   - Full RAM Load (mmap=False): {mmap_res['mmap_false']['load_time_ms']} ms, "
            f"{mmap_res['mmap_false']['peak_allocated_kb']:.1f} KB allocated"
        )
        print(
            f"   - mmap Zero-Copy (mmap=True): {mmap_res['mmap_true']['load_time_ms']} ms, "
            f"{mmap_res['mmap_true']['peak_allocated_kb']:.1f} KB allocated"
        )
        print(
            f"   => Loading Speedup: {mmap_res['speedup']}x | "
            f"RAM Allocation Reduction: {mmap_res['ram_reduction_ratio']}x\n"
        )

        print("2. Benchmarking BM25 incremental lifecycle sync...")
        bm25_res = benchmark_bm25_incremental(entries)
        print(
            f"   - Append ({bm25_res['append_batch_size']} docs): Incremental {bm25_res['append_comparison']['incremental_append_ms']} ms "
            f"vs Full Rebuild {bm25_res['append_comparison']['full_rebuild_ms']} ms "
            f"=> Speedup: {bm25_res['append_comparison']['speedup']}x"
        )
        print(
            f"   - Prune ({bm25_res['lifecycle_prune_comparison']['pruned_docs']} docs): Incremental {bm25_res['lifecycle_prune_comparison']['incremental_prune_ms']} ms "
            f"vs Full Rebuild {bm25_res['lifecycle_prune_comparison']['full_rebuild_ms']} ms "
            f"=> Speedup: {bm25_res['lifecycle_prune_comparison']['speedup']}x\n"
        )

        print("3. Benchmarking composite boolean facet filtering...")
        facet_res = benchmark_facet_filtering(entries, matrix)
        print(
            f"   - AST Compilation + Bitmask Throughput: "
            f"{facet_res['facet_compilation_and_bitmask']['throughput_queries_per_sec']:.1f} qps "
            f"({facet_res['facet_compilation_and_bitmask']['avg_latency_microseconds']} µs/query)"
        )
        print(
            f"   - Search Latency: Unconstrained {facet_res['search_acceleration']['unconstrained_search_ms']} ms "
            f"vs Pre-filtered {facet_res['search_acceleration']['prefiltered_search_ms']} ms "
            f"=> Speedup: {facet_res['search_acceleration']['search_speedup']}x\n"
        )

        print("4. Benchmarking Zero-Latency SQLite metadata sidecar and Top-K retrieval...")
        sidecar_res = benchmark_cold_start_and_lazy_retrieval(entries, matrix, temp_path)
        print(
            f"   - Cold-Start: Legacy JSON {sidecar_res['cold_start']['legacy_json_ms']} ms "
            f"vs SQLite Sidecar {sidecar_res['cold_start']['sqlite_sidecar_ms']} ms "
            f"=> Speedup: {sidecar_res['cold_start']['cold_start_speedup']}x | "
            f"RAM Reduction: {sidecar_res['cold_start']['ram_reduction_ratio']}x"
        )
        print(
            f"   - Two-Stage Top-K Hybrid Search: {sidecar_res['retrieval']['hybrid_search_ms']} ms/query "
            f"({sidecar_res['retrieval']['queries_per_sec']} qps)\n"
        )

        print("5. Benchmarking SQLite Query Embedding Cache (Hit vs Miss)...")
        cache_res = benchmark_query_embedding_cache(temp_path, dimension=args.dim)
        print(
            f"   - Single Cache Read: {cache_res['single_get_microseconds']} µs "
            f"({cache_res['throughput_queries_per_sec']} qps)"
        )
        print(
            f"   - Simulated API (200ms) vs Cache Hit ({cache_res['single_get_microseconds'] / 1000:.3f}ms) "
            f"=> End-to-End Speedup: {cache_res['cache_hit_speedup']}x\n"
        )

        print("6. Benchmarking Batched GEMM Matrix Retrieval & Global Prefetching...")
        gemm_res = benchmark_batched_gemm_retrieval(entries, matrix, temp_path, batch_size=32)
        print(
            f"   - Sequential Single Queries: {gemm_res['sequential_single_query']['latency_ms_per_query']} ms/query "
            f"({gemm_res['sequential_single_query']['throughput_qps']} qps)"
        )
        print(
            f"   - Batched GEMM + Prefetch: {gemm_res['batched_gemm_prefetch']['latency_ms_per_query']} ms/query "
            f"({gemm_res['batched_gemm_prefetch']['throughput_qps']} qps) "
            f"=> Throughput Speedup: {gemm_res['batch_speedup']}x\n"
        )

    summary = {
        "status": "success",
        "mmap_loading": mmap_res,
        "bm25_incremental": bm25_res,
        "facet_filtering": facet_res,
        "sqlite_sidecar_and_lazy_retrieval": sidecar_res,
        "query_embedding_cache": cache_res,
        "batched_gemm_retrieval": gemm_res,
    }
    print("Benchmark summary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
