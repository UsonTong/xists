"""Stable, programmatic entry points for searching xists indexes.

This module deliberately does not load ``.env`` files or inspect process
environment variables.  Applications construct :class:`EmbeddingConfig` from
their own configuration and pass it explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from xists.search.embed import EmbeddingConfig
from xists.search.index import load_index as _load_index
from xists.search.query import EXPLORATORY_THRESHOLD, rank


def load_index(path: str | Path) -> dict[str, Any]:
    """Load an index document from *path*.

    The returned value is the JSON-compatible index document, unchanged from
    its on-disk representation.  Invalid JSON and filesystem errors are
    intentionally allowed to reach the caller with their original context.
    """

    return _load_index(Path(path))


def search(
    query: str,
    index: dict[str, Any],
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
    )
