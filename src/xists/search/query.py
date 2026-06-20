"""Query an embedding index and rank records by semantic similarity.

Search keeps xists's principle of not over-recommending: results are bucketed
into high_confidence / exploratory / abstain by cosine similarity, and when
nothing clears the exploratory threshold the search abstains rather than
returning weak guesses.
"""

from __future__ import annotations

import math
from typing import Any

from xists.search.embed import EmbeddingConfig, EmbeddingError, embed_query

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

    scored.sort(key=lambda item: item["score"], reverse=True)
    presented = [item for item in scored if item["confidence"] != "abstain"][:top_k]

    return {
        "query": query,
        "abstained": len(presented) == 0,
        "results": presented,
        "considered": len(scored),
    }
