"""Stable, programmatic entry points for searching xists indexes.

This module deliberately does not load ``.env`` files or inspect process
environment variables.  Applications construct :class:`EmbeddingConfig` from
their own configuration and pass it explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xists.search.embed import EmbeddingConfig
from xists.search.index import load_index as _load_index
from xists.search.query import rank


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
) -> dict[str, Any]:
    """Search an in-memory index using an explicitly configured embedder.

    This is the minimal stable API.  It has no CLI side effects: it does not
    load ``.env``, read environment variables, print, or terminate the
    process.  The result uses the same schema and ranking behavior as the
    default xists search core.  Index/model incompatibility and embedding
    endpoint failures are raised as their actionable core exceptions.
    """

    return rank(query, index, embedding_config, top_k=top_k)
