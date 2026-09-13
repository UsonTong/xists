"""Stable, programmatic entry points for searching xists indexes.

This module deliberately does not load ``.env`` files or inspect process
environment variables.  Applications construct :class:`EmbeddingConfig` from
their own configuration and pass it explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from xists.search.compare import compare_projects_prepared
from xists.search.embed import EmbeddingConfig
from xists.search.index import load_index as _load_index
from xists.search.query import EXPLORATORY_THRESHOLD, PreparedIndex, prepare_index, rank
from xists.search.similar import find_similar_prepared
from xists.types import CompareResponse, SearchFilter, SimilarResponse


def load_index(path: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    """Load an index document from *path*.

    The returned value is the JSON-compatible index document, unchanged from
    its on-disk representation. If mmap is True (default) and a version 4
    binary vector sidecar exists, vectors are memory-mapped for zero-copy loading.
    Invalid JSON and filesystem errors are intentionally allowed to reach the
    caller with their original context.
    """

    return _load_index(Path(path), mmap=mmap)


def search(
    query: str,
    index: dict[str, Any] | PreparedIndex,
    *,
    embedding_config: EmbeddingConfig,
    top_k: int = 10,
    ranking_strategy: str = "metadata",
    rerank: Callable[[str, list[str]], list[float]] | None = None,
    rerank_candidate_limit: int = 50,
    exploratory_threshold: float = EXPLORATORY_THRESHOLD,
    rerank_abstain_threshold: float | None = None,
    confidence_calibration: str = "off",
    query_variants: list[str] | None = None,
    rerank_query: str | None = None,
    filters: SearchFilter | dict[str, Any] | str | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    cache: Any = None,
) -> dict[str, Any]:
    """Search an in-memory index using an explicitly configured embedder.

    This is the minimal stable API.  It has no CLI side effects: it does not
    load ``.env``, read environment variables, print, or terminate the
    process.  The result uses the same schema and ranking behavior as the
    default xists search core.  Optional ranking arguments mirror the existing
    core capabilities while keeping every dependency explicit; in particular,
    a reranker is passed as a callable rather than read from configuration.
    Index/model incompatibility and embedding endpoint failures are raised as
    their actionable core exceptions.
    """

    extra_kwargs: dict[str, Any] = {}
    if cache is not None:
        extra_kwargs["cache"] = cache

    return rank(
        query,
        index,
        embedding_config,
        top_k=top_k,
        ranking_strategy=ranking_strategy,
        rerank=rerank,
        rerank_candidate_limit=rerank_candidate_limit,
        exploratory_threshold=exploratory_threshold,
        rerank_abstain_threshold=rerank_abstain_threshold,
        confidence_calibration=confidence_calibration,
        query_variants=query_variants,
        rerank_query=rerank_query,
        filters=filters,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        **extra_kwargs,
    )


def find_similar(
    repo_id: str,
    index: dict[str, Any] | PreparedIndex,
    *,
    top_k: int = 10,
    filters: SearchFilter | dict[str, Any] | str | None = None,
) -> SimilarResponse:
    """Find top-k similar repositories using precomputed embeddings in the index.

    This function does not require an EmbeddingConfig or remote embedding API
    because it operates on the stored unit L2-normalized vector matrix in the index.
    """

    prepared = prepare_index(index)
    return find_similar_prepared(repo_id, prepared, top_k=top_k, filters=filters)


def compare_projects(
    repo_ids: list[str],
    index: dict[str, Any] | PreparedIndex,
) -> CompareResponse:
    """Compare 2 to 5 repositories side by side using indexed metadata and pairwise similarity."""

    prepared = prepare_index(index)
    return compare_projects_prepared(repo_ids, prepared)


__all__ = ["compare_projects", "find_similar", "load_index", "search"]
