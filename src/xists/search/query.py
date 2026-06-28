"""Query an embedding index and rank records by semantic similarity.

Search keeps xists's principle of not over-recommending: results are bucketed
into high_confidence / exploratory / abstain by cosine similarity, and when
nothing clears the exploratory threshold the search abstains rather than
returning weak guesses.
"""

from __future__ import annotations

import heapq
import math
import re
from typing import Any

import numpy as np

from xists.search.embed import EmbeddingConfig, EmbeddingError, call_embeddings, embed_query

# Cosine similarity thresholds. Tunable; conservative by default so weak
# matches abstain instead of being presented as answers.
HIGH_CONFIDENCE_THRESHOLD = 0.55
EXPLORATORY_THRESHOLD = 0.35
GENERIC_TERMS = {
    "and",
    "api",
    "application",
    "applications",
    "alternative",
    "alternatives",
    "automation",
    "backend",
    "build",
    "cli",
    "cloud",
    "compatible",
    "component",
    "components",
    "configuration",
    "data",
    "database",
    "deployment",
    "distributed",
    "engine",
    "framework",
    "frontend",
    "full",
    "integration",
    "integrations",
    "interface",
    "interfaces",
    "library",
    "management",
    "modern",
    "open",
    "platform",
    "replace",
    "replacement",
    "replacing",
    "search",
    "service",
    "software",
    "source",
    "system",
    "tool",
    "tools",
    "ui",
    "web",
    "workflow",
    "with",
    "written",
}
LANGUAGE_TERMS = {"javascript", "typescript", "python", "rust", "go", "java", "php", "ruby", "c++", "c#", "scala"}
ALTERNATIVE_TERMS = {"alternative", "alternatives", "replace", "replacement", "replacing"}
QUERY_JOINERS = {"a", "an", "and", "for", "or", "the", "to", "with"}
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._#-]*")
RERANK_CANDIDATE_MULTIPLIER = 5
MIN_RERANK_CANDIDATES = 50


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


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _keyword_tokens(query: str) -> set[str]:
    return {
        token
        for token in _tokenize(query)
        if len(token) > 2 and token not in GENERIC_TERMS and not token.isdigit()
    }


def _content_keyword_tokens(query: str) -> set[str]:
    return _keyword_tokens(query) - LANGUAGE_TERMS


def _dedupe_adjacent(tokens: list[str]) -> list[str]:
    deduped: list[str] = []
    for token in tokens:
        if not deduped or deduped[-1] != token:
            deduped.append(token)
    return deduped


def _query_text_variants(query: str) -> set[str]:
    tokens = _tokenize(query)
    variants: set[str] = set()

    def add(values: list[str]) -> None:
        if values:
            variants.add(" ".join(values))

    add(tokens)
    add(_dedupe_adjacent(tokens))

    start = 0
    while start < len(tokens) and tokens[start] in LANGUAGE_TERMS:
        start += 1
    if start:
        without_language_prefix = tokens[start:]
        add(without_language_prefix)
        add(_dedupe_adjacent(without_language_prefix))

    return variants


def _alternative_targets(query_tokens: list[str]) -> set[str]:
    targets: set[str] = set()
    for index, token in enumerate(query_tokens):
        if token not in ALTERNATIVE_TERMS:
            continue
        for candidate in reversed(query_tokens[:index]):
            if candidate in QUERY_JOINERS or candidate in LANGUAGE_TERMS or candidate in GENERIC_TERMS:
                continue
            targets.add(candidate)
            break
        if index + 2 < len(query_tokens) and query_tokens[index + 1] in {"for", "to"}:
            candidate = query_tokens[index + 2]
            if candidate not in QUERY_JOINERS and candidate not in LANGUAGE_TERMS and candidate not in GENERIC_TERMS:
                targets.add(candidate)
    return targets


def _variant_in_text(variants: set[str], text: str) -> bool:
    return any(variant and variant in text for variant in variants)


def _has_specific_variant(variants: set[str]) -> bool:
    return any(
        token not in GENERIC_TERMS and token not in LANGUAGE_TERMS and len(token) > 2 and not token.isdigit()
        for variant in variants
        for token in _tokenize(variant)
    )


def _all_metadata_text(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("name", "description", "summary", "language"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    for key in ("topics", "use_cases", "capabilities", "search_phrases"):
        values = metadata.get(key)
        if isinstance(values, list):
            parts.extend(str(value) for value in values if isinstance(value, str) and value.strip())
    return "\n".join(parts)


def _phrase_specificity(phrase: str) -> float:
    tokens = _tokenize(phrase)
    if not tokens:
        return 0.0
    specific = sum(1 for token in tokens if token not in GENERIC_TERMS and len(token) > 2 and token not in LANGUAGE_TERMS)
    if specific == 0:
        return -0.05
    return min(0.08, 0.02 * specific)


def _profile_phrase_match(query: str, metadata: dict[str, Any]) -> dict[str, Any]:
    variants = _query_text_variants(query)
    keyword_tokens = _content_keyword_tokens(query)
    exact_phrases_seen: set[str] = set()
    best_subset_specificity = 0.0
    exact_specificity = 0.0
    exact_token_count = 0
    exact_match = False

    for key in ("use_cases", "capabilities", "search_phrases"):
        for phrase in metadata.get(key) or []:
            phrase_tokens = _tokenize(str(phrase))
            if not phrase_tokens:
                continue
            phrase_text = " ".join(phrase_tokens)
            specificity = _phrase_specificity(str(phrase))
            if phrase_text in variants:
                if phrase_text not in exact_phrases_seen:
                    exact_phrases_seen.add(phrase_text)
                    exact_match = True
                    exact_specificity = max(exact_specificity, specificity)
                    exact_token_count = max(exact_token_count, len(phrase_tokens))
            elif keyword_tokens and keyword_tokens.issubset(set(phrase_tokens)):
                best_subset_specificity = max(best_subset_specificity, specificity)

    return {
        "exact": exact_match,
        "exact_specificity": exact_specificity,
        "exact_token_count": exact_token_count,
        "subset_specificity": best_subset_specificity,
    }


def _query_specificity(query: str) -> float:
    tokens = _tokenize(query)
    if not tokens:
        return 0.0
    content_tokens = _content_keyword_tokens(query)
    if len(content_tokens) >= 5:
        return 1.0
    if len(content_tokens) >= 3:
        return 0.8
    if len(content_tokens) >= 2 and len(tokens) >= 5:
        return 0.65
    if len(content_tokens) >= 2:
        return 0.45
    if len(content_tokens) == 1:
        return 0.25
    return 0.0


def _metadata_multiplier(
    query: str,
    *,
    exact_phrase_match: bool = False,
    unique_exact_phrase_match: bool = False,
    exact_phrase_specificity: float = 0.0,
    exact_phrase_token_count: int = 0,
) -> float:
    specificity = _query_specificity(query)
    if unique_exact_phrase_match:
        if exact_phrase_specificity >= 0.06 or exact_phrase_token_count >= 5:
            return max(0.9, specificity)
        if exact_phrase_specificity >= 0.02 or exact_phrase_token_count >= 3:
            return max(0.65, specificity)
    elif exact_phrase_match and (exact_phrase_specificity >= 0.02 or exact_phrase_token_count >= 4):
        return max(0.35, specificity)
    if specificity >= 0.8:
        return 1.0
    if specificity >= 0.65:
        return 0.8
    if specificity >= 0.45:
        return 0.55
    if specificity >= 0.25:
        return 0.35
    return 0.2


def _metadata_cap(query: str) -> float:
    specificity = _query_specificity(query)
    if specificity >= 0.8:
        return 0.18
    if specificity >= 0.65:
        return 0.14
    if specificity >= 0.45:
        return 0.09
    if specificity >= 0.25:
        return 0.04
    return 0.04


def _metadata_bonus_cap(
    query: str,
    *,
    strong_phrase_match: bool,
    identity_match: bool,
    exact_phrase_match: bool = False,
    unique_exact_phrase_match: bool = False,
    exact_phrase_specificity: float = 0.0,
    exact_phrase_token_count: int = 0,
) -> float:
    cap = _metadata_cap(query)
    if unique_exact_phrase_match:
        if exact_phrase_specificity >= 0.06 or exact_phrase_token_count >= 5:
            cap = max(cap, 0.3)
        elif exact_phrase_specificity >= 0.02 or exact_phrase_token_count >= 3:
            cap = max(cap, 0.14)
    if strong_phrase_match:
        cap += 0.04
    if identity_match:
        cap += 0.025
    return min(cap, 0.32)


def _metadata_score(query: str, entry: dict[str, Any], *, exact_phrase_match_count: int = 0) -> float:
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        return 0.0

    score = 0.0
    query_tokens = _tokenize(query)
    keyword_tokens = _content_keyword_tokens(query)
    language_tokens = _keyword_tokens(query) & LANGUAGE_TERMS
    query_variants = _query_text_variants(query)
    has_specific_variant = _has_specific_variant(query_variants)

    repo_id = str(entry.get("repo_id") or "").lower()
    name = str(metadata.get("name") or "").lower()
    description = str(metadata.get("description") or "").lower()
    summary = str(metadata.get("summary") or "").lower()
    language = str(metadata.get("language") or "").lower()
    topics = {
        token
        for topic in metadata.get("topics") or []
        for token in _tokenize(str(topic))
    }
    metadata_text = _all_metadata_text(metadata).lower()
    metadata_tokens = set(_tokenize(metadata_text)) - LANGUAGE_TERMS
    repo_tokens = set(_tokenize(repo_id.replace("/", " ")))
    name_tokens = set(_tokenize(name))
    identity_match = _variant_in_text(query_variants, repo_id) or _variant_in_text(query_variants, name)
    alternative_targets = _alternative_targets(query_tokens)
    alternative_identity_match = bool(alternative_targets & (repo_tokens | name_tokens))
    phrase_match = _profile_phrase_match(query, metadata)
    exact_phrase_match = bool(phrase_match["exact"])
    unique_exact_phrase_match = exact_phrase_match and exact_phrase_match_count == 1
    exact_phrase_specificity = float(phrase_match["exact_specificity"])
    exact_phrase_token_count = int(phrase_match["exact_token_count"])

    if _variant_in_text(query_variants, repo_id):
        score += 0.2
    if _variant_in_text(query_variants, name):
        score += 0.12
    if has_specific_variant and _variant_in_text(query_variants, description):
        score += 0.06
    if has_specific_variant and _variant_in_text(query_variants, summary):
        score += 0.04

    if keyword_tokens:
        name_overlap = len(keyword_tokens & name_tokens)
        if name_overlap:
            score += min(0.16, 0.08 * name_overlap)
        repo_overlap = len(keyword_tokens & repo_tokens)
        if repo_overlap:
            score += min(0.08, 0.03 * repo_overlap)
        overlap = len(keyword_tokens & metadata_tokens)
        if overlap:
            score += min(0.12, 0.035 * overlap)
        topic_overlap = len(keyword_tokens & topics)
        if topic_overlap:
            score += min(0.06, 0.025 * topic_overlap)

    if language_tokens and language:
        if language in language_tokens:
            language_boost = 0.025 if keyword_tokens else 0.01
            score += language_boost
        else:
            score -= 0.04

    best_phrase_score = 0.0
    strong_phrase_match = exact_phrase_match and exact_phrase_specificity >= 0.02
    if exact_phrase_match:
        exact_base = 0.16 if unique_exact_phrase_match else 0.03
        best_phrase_score = max(best_phrase_score, exact_base + max(0.0, exact_phrase_specificity))
    subset_specificity = float(phrase_match["subset_specificity"])
    if subset_specificity:
        best_phrase_score = max(best_phrase_score, 0.015 + max(0.0, subset_specificity))

    score += best_phrase_score
    if alternative_identity_match:
        score -= 0.2
    return min(
        score
        * _metadata_multiplier(
            query,
            exact_phrase_match=exact_phrase_match,
            unique_exact_phrase_match=unique_exact_phrase_match,
            exact_phrase_specificity=exact_phrase_specificity,
            exact_phrase_token_count=exact_phrase_token_count,
        ),
        _metadata_bonus_cap(
            query,
            strong_phrase_match=strong_phrase_match,
            identity_match=identity_match,
            exact_phrase_match=exact_phrase_match,
            unique_exact_phrase_match=unique_exact_phrase_match,
            exact_phrase_specificity=exact_phrase_specificity,
            exact_phrase_token_count=exact_phrase_token_count,
        ),
    )


def _metadata_match_strength(query: str, item: dict[str, Any]) -> int:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        return 0

    query_variants = _query_text_variants(query)
    if not query_variants:
        return 0

    repo_id = str(item.get("repo_id") or "").lower()
    name = str(metadata.get("name") or "").lower()
    if _variant_in_text(query_variants, repo_id) or _variant_in_text(query_variants, name):
        return 2

    phrase_match = _profile_phrase_match(query, metadata)
    if phrase_match["exact"]:
        if float(phrase_match["exact_specificity"]) >= 0.02 or int(phrase_match["exact_token_count"]) >= 3:
            return 2
        return 1
    return 0


def _rerank_results(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reranked: list[dict[str, Any]] = []
    phrase_match_count = sum(
        1
        for item in results
        if isinstance(item.get("metadata"), dict)
        and _profile_phrase_match(query, item["metadata"])["exact"]
    )
    for item in results:
        semantic_score = float(item["score"])
        metadata_score = _metadata_score(query, item, exact_phrase_match_count=phrase_match_count)
        final_score = semantic_score + metadata_score
        reranked.append(
            {
                **item,
                "semantic_score": semantic_score,
                "metadata_score": metadata_score,
                "score": final_score,
                "confidence": confidence_bucket(final_score),
            }
        )
    reranked.sort(key=lambda candidate: candidate["score"], reverse=True)
    if len(reranked) > 1 and _query_specificity(query) <= 0.45:
        semantic_winner = max(reranked, key=lambda candidate: candidate["semantic_score"])
        rerank_winner = reranked[0]
        semantic_gap = semantic_winner["semantic_score"] - rerank_winner["semantic_score"]
        metadata_advantage = rerank_winner["metadata_score"] - semantic_winner["metadata_score"]
        winner_strength = _metadata_match_strength(query, rerank_winner)
        semantic_strength = _metadata_match_strength(query, semantic_winner)
        if winner_strength >= 2 and semantic_strength < 2:
            required_advantage = 0.005
        elif winner_strength >= 2:
            required_advantage = 0.015
        else:
            required_advantage = 0.04 + max(0.0, semantic_gap)
        if semantic_winner is not rerank_winner and semantic_gap > 0.0 and metadata_advantage < required_advantage:
            reranked.remove(semantic_winner)
            reranked.insert(0, semantic_winner)
    return reranked


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


def _candidate_count(top_k: int, total: int) -> int:
    requested = max(top_k, 0)
    if requested == 0:
        return 0
    return min(total, max(requested, MIN_RERANK_CANDIDATES, requested * RERANK_CANDIDATE_MULTIPLIER))


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
    candidate_count = _candidate_count(top_k, len(entries))
    top = _top_indices(scores, candidate_count)

    ranked: list[dict[str, Any]] = []
    for row, query in enumerate(queries):
        results: list[dict[str, Any]] = []
        for column in top[row]:
            score = float(scores[row, column])
            results.append(
                {
                    "repo_id": repo_ids[int(column)],
                    "score": score,
                    "confidence": confidence_bucket(score),
                    "metadata": entries[int(column)].get("metadata"),
                }
            )
        results = _rerank_results(query, results)
        results = [item for item in results if item["confidence"] != "abstain"]
        results = results[: max(top_k, 0)]
        for item in results:
            item.pop("metadata", None)
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
    candidate_count = _candidate_count(top_k, len(index.get("vectors", [])))
    scored: list[dict[str, Any]] = []
    for entry in index.get("vectors", []):
        score = cosine_similarity(query_vector, entry["vector"])
        scored.append(
            {
                "repo_id": entry.get("repo_id"),
                "score": score,
                "confidence": confidence_bucket(score),
                "metadata": entry.get("metadata"),
            }
        )

    if top_count:
        candidates = heapq.nlargest(candidate_count, scored, key=lambda item: item["score"])
        presented = _rerank_results(query, candidates)
        presented = [item for item in presented if item["confidence"] != "abstain"]
        presented = presented[:top_count]
        for item in presented:
            item.pop("metadata", None)
    else:
        presented = []

    return {
        "query": query,
        "abstained": len(presented) == 0,
        "results": presented,
        "considered": len(scored),
    }
