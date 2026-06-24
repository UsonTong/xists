"""Query an embedding index and rank records by semantic similarity.

Search keeps xists's principle of not over-recommending: results are bucketed
into high_confidence / exploratory / abstain by cosine similarity, and when
nothing clears the exploratory threshold the search abstains rather than
returning weak guesses.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

from xists.search.embed import EmbeddingConfig, EmbeddingError, call_embeddings, embed_query

# Cosine similarity thresholds. Tunable; conservative by default so weak
# matches abstain instead of being presented as answers.
HIGH_CONFIDENCE_THRESHOLD = 0.55
EXPLORATORY_THRESHOLD = 0.35


class IndexMismatchError(RuntimeError):
    """Raised when the index was built with a different embedding model."""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def confidence_bucket(score: float) -> str:
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high_confidence"
    if score >= EXPLORATORY_THRESHOLD:
        return "exploratory"
    return "abstain"


def ensure_index_matches_model(index: dict[str, Any], config: EmbeddingConfig) -> None:
    """Refuse to search if the index model differs from the configured model.

    Mixing models silently would produce meaningless similarities, so xists
    fails clearly and asks for a rebuild instead.
    """

    index_model = index.get("embedding_model")
    if index_model and index_model != config.model:
        raise IndexMismatchError(
            f"Index was built with embedding model '{index_model}' but the "
            f"configured model is '{config.model}'. Rebuild the index "
            "(xists index build) or set EMBEDDING_MODEL to match."
        )


def _normalized_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise IndexMismatchError("Index vectors must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def _top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    top_count = min(max(top_k, 0), scores.shape[1])
    if top_count == 0:
        return np.empty((scores.shape[0], 0), dtype=np.int64)
    unsorted = np.argpartition(scores, -top_count, axis=1)[:, -top_count:]
    top_scores = np.take_along_axis(scores, unsorted, axis=1)
    order = np.argsort(top_scores, axis=1)[:, ::-1]
    return np.take_along_axis(unsorted, order, axis=1)


def rank_many(
    queries: list[str],
    index: dict[str, Any],
    config: EmbeddingConfig,
    *,
    top_k: int = 10,
    batch_size: int = 64,
    embed_many: Any = call_embeddings,
) -> list[dict[str, Any]]:
    """Rank multiple queries with batched embeddings and matrix similarity."""

    ensure_index_matches_model(index, config)
    if not queries:
        return []

    entries = index.get("vectors", [])
    repo_ids = [entry.get("repo_id") for entry in entries]
    vectors = [entry["vector"] for entry in entries]
    dimension = index.get("dimension")
    if dimension is not None and any(len(vector) != dimension for vector in vectors):
        raise IndexMismatchError("Index contains vectors that do not match its dimension")

    query_vectors: list[list[float]] = []
    for start in range(0, len(queries), batch_size):
        query_vectors.extend(embed_many(config, queries[start : start + batch_size]))
    if len(query_vectors) != len(queries):
        raise EmbeddingError(f"Embedding count mismatch: sent {len(queries)}, received {len(query_vectors)}")
    if dimension is not None and any(len(vector) != dimension for vector in query_vectors):
        raise IndexMismatchError(
            f"One or more query vectors do not match index dimension {dimension}. "
            "Rebuild the index or check the model."
        )

    if not entries:
        return [
            {"query": query, "abstained": True, "results": [], "considered": 0}
            for query in queries
        ]

    index_matrix = _normalized_matrix(vectors)
    query_matrix = _normalized_matrix(query_vectors)
    scores = query_matrix @ index_matrix.T
    top = _top_indices(scores, top_k)

    ranked: list[dict[str, Any]] = []
    for row, query in enumerate(queries):
        results: list[dict[str, Any]] = []
        for column in top[row]:
            score = float(scores[row, column])
            confidence = confidence_bucket(score)
            if confidence == "abstain":
                continue
            results.append(
                {
                    "repo_id": repo_ids[int(column)],
                    "score": score,
                    "confidence": confidence,
                }
            )
        ranked.append(
            {
                "query": query,
                "abstained": len(results) == 0,
                "results": results,
                "considered": len(entries),
            }
        )
    return ranked


def rank(
    query: str,
    index: dict[str, Any],
    config: EmbeddingConfig,
    *,
    top_k: int = 10,
    embed: Any = embed_query,
) -> dict[str, Any]:
    """Rank index entries against the query.

    ``embed`` is injected so tests can supply a mock query vector instead of
    calling the network.
    """

    ensure_index_matches_model(index, config)

    query_vector = embed(config, query)
    dimension = index.get("dimension")
    if dimension is not None and len(query_vector) != dimension:
        raise IndexMismatchError(
            f"Query vector dimension {len(query_vector)} does not match index "
            f"dimension {dimension}. Rebuild the index or check the model."
        )

    top_count = max(top_k, 0)
    scored: list[dict[str, Any]] = []
    for entry in index.get("vectors", []):
        score = cosine_similarity(query_vector, entry["vector"])
        scored.append(
            {
                "repo_id": entry.get("repo_id"),
                "score": score,
                "confidence": confidence_bucket(score),
            }
        )

    if top_count:
        candidates = heapq.nlargest(top_count, scored, key=lambda item: item["score"])
        presented = [item for item in candidates if item["confidence"] != "abstain"]
    else:
        presented = []

    return {
        "query": query,
        "abstained": len(presented) == 0,
        "results": presented,
        "considered": len(scored),
    }
