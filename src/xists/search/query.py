"""Simple, explainable semantic search over an xists embedding index."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

import numpy as np

from xists.records import RECORD_SCHEMA_VERSION
from xists.search.embed import EMBEDDING_INPUT_VERSION, EmbeddingConfig, EmbeddingError, call_embeddings, embed_query
from xists.search.index import decode_vector

HIGH_CONFIDENCE_THRESHOLD = 0.60
EXPLORATORY_THRESHOLD = 0.35
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._#-]*")

GENERIC_TERMS = {
    "a",
    "an",
    "and",
    "app",
    "application",
    "applications",
    "alternative",
    "alternatives",
    "build",
    "building",
    "built",
    "for",
    "from",
    "in",
    "library",
    "of",
    "open",
    "platform",
    "project",
    "repo",
    "repository",
    "service",
    "source",
    "system",
    "the",
    "to",
    "tool",
    "tools",
    "use",
    "with",
}
QUERY_JOINERS = {"a", "an", "and", "for", "in", "of", "or", "the", "to", "with"}
ALTERNATIVE_TERMS = {"alternative", "alternatives", "replace", "replacement", "similar", "like"}
DOMAIN_QUERY_CUES = {"for", "in", "with", "domain", "industry", "pipelines", "infrastructure", "observability"}
EXACT_NAME_QUERY_MAX_TOKENS = 3

LANGUAGE_ALIAS_GROUPS = (
    ("python", ("python", "py")),
    ("javascript", ("javascript", "js")),
    ("typescript", ("typescript", "ts")),
    ("rust", ("rust",)),
    ("go", ("go", "golang")),
    ("java", ("java",)),
    ("php", ("php",)),
    ("ruby", ("ruby",)),
    ("c", ("c",)),
    ("c++", ("c++", "cpp", "cplusplus")),
    ("c#", ("c#", "csharp")),
    ("scala", ("scala",)),
    ("swift", ("swift",)),
    ("kotlin", ("kotlin",)),
    ("dart", ("dart",)),
    ("vue", ("vue",)),
    ("shell", ("shell", "bash", "sh", "zsh")),
    ("jupyter notebook", ("jupyter notebook", "jupyter-notebook", "jupyter", "ipynb")),
)
LANGUAGE_ALIASES = {canonical: set(aliases) for canonical, aliases in LANGUAGE_ALIAS_GROUPS}
LANGUAGE_TERMS = {
    token
    for aliases in LANGUAGE_ALIASES.values()
    for alias in aliases
    for token in TOKEN_RE.findall(alias)
}
LANGUAGE_PREFIXES = sorted(
    {
        tuple(TOKEN_RE.findall(alias))
        for aliases in LANGUAGE_ALIASES.values()
        for alias in aliases
        if TOKEN_RE.findall(alias)
    },
    key=len,
    reverse=True,
)


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


@lru_cache(maxsize=65536)
def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(text.lower()))


@lru_cache(maxsize=65536)
def _expanded_token(token: str) -> frozenset[str]:
    values = {token}
    values.update(part for part in re.split(r"[-._#]+", token) if part)
    for value in list(values):
        if value.endswith("s") and len(value) > 3:
            values.add(value[:-1])
        elif len(value) > 2:
            values.add(f"{value}s")
    return frozenset(values)


def _expanded_token_set(tokens: set[str] | frozenset[str] | tuple[str, ...]) -> set[str]:
    expanded: set[str] = set()
    for token in tokens:
        expanded.update(_expanded_token(token))
    return expanded


def _language_prefix_length(tokens: tuple[str, ...]) -> int:
    for prefix in LANGUAGE_PREFIXES:
        if tokens[: len(prefix)] == prefix:
            return len(prefix)
    return 0


def _language_aliases_from_tokens(tokens: tuple[str, ...] | list[str]) -> set[str]:
    token_set = set(tokens)
    token_text = " ".join(tokens)
    compact_text = "".join(tokens)
    aliases: set[str] = set()
    for canonical, values in LANGUAGE_ALIASES.items():
        for alias in values:
            alias_tokens = _tokenize(alias)
            if not alias_tokens:
                continue
            alias_text = " ".join(alias_tokens)
            if (len(alias_tokens) == 1 and alias_tokens[0] in token_set) or alias_text in {token_text, compact_text}:
                aliases.add(canonical)
                break
    return aliases


@lru_cache(maxsize=8192)
def _query_primary_language_alias(query: str) -> str | None:
    tokens = _tokenize(query)
    prefix_length = _language_prefix_length(tokens)
    if not prefix_length:
        return None
    aliases = _language_aliases_from_tokens(tokens[:prefix_length])
    for canonical, _ in LANGUAGE_ALIAS_GROUPS:
        if canonical in aliases:
            return canonical
    return None


@lru_cache(maxsize=8192)
def _query_language_terms(query: str) -> frozenset[str]:
    tokens = _tokenize(query)
    prefix_length = _language_prefix_length(tokens)
    return frozenset(tokens[:prefix_length]) if prefix_length else frozenset()


@lru_cache(maxsize=8192)
def _keyword_tokens(query: str) -> frozenset[str]:
    language_terms = _query_language_terms(query)
    return frozenset(
        token
        for token in _tokenize(query)
        if len(token) > 1
        and token not in GENERIC_TERMS
        and token not in QUERY_JOINERS
        and token not in language_terms
        and not token.isdigit()
    )


def _query_intent(query: str) -> dict[str, Any]:
    tokens = _tokenize(query)
    keyword_tokens = sorted(_keyword_tokens(query))
    raw_query = query.strip().lower()
    if not tokens:
        intent_type = "empty"
    elif "/" in raw_query or (
        len(tokens) <= EXACT_NAME_QUERY_MAX_TOKENS
        and all(token not in GENERIC_TERMS and token not in QUERY_JOINERS for token in tokens)
    ):
        intent_type = "exact_name"
    elif any(token in ALTERNATIVE_TERMS for token in tokens):
        intent_type = "alternative"
    elif any(token in DOMAIN_QUERY_CUES for token in tokens) and len(keyword_tokens) >= 2:
        intent_type = "domain"
    else:
        intent_type = "functional"
    return {
        "type": intent_type,
        "specificity": min(1.0, len(keyword_tokens) / 5.0),
        "keywords": keyword_tokens,
        "primary_language": _query_primary_language_alias(query),
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def _normalized_text_variants(text: str) -> set[str]:
    tokens = _tokenize(text.replace("/", " "))
    if not tokens:
        return set()
    return {" ".join(tokens), "".join(tokens), "-".join(tokens), "_".join(tokens), ".".join(tokens)}


def _query_variants(query: str) -> set[str]:
    raw = query.strip().lower()
    variants = {raw} if raw else set()
    variants.update(_normalized_text_variants(query))
    return {variant for variant in variants if variant}


def _identity_values(entry: dict[str, Any]) -> list[str]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    repo_id = str(entry.get("repo_id") or "")
    values = [repo_id, str(metadata.get("name") or "")]
    if "/" in repo_id:
        values.extend(part for part in repo_id.split("/") if part)
    values.extend(_string_list(metadata.get("aliases")))
    return [value for value in values if value.strip()]


def _identity_variants(entry: dict[str, Any]) -> set[str]:
    variants: set[str] = set()
    for value in _identity_values(entry):
        variants.add(value.strip().lower())
        variants.update(_normalized_text_variants(value))
    return {variant for variant in variants if variant}


def _exact_identity_match(query: str, entry: dict[str, Any]) -> bool:
    raw_query = query.strip().lower()
    repo_id = str(entry.get("repo_id") or "").strip().lower()
    if repo_id and repo_id in raw_query:
        return True
    if raw_query in {value.strip().lower() for value in _identity_values(entry)}:
        return True
    # ASCII tokenization cannot safely represent mixed CJK natural-language queries.
    if any(0x3400 <= ord(char) <= 0x9FFF for char in query):
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        values = [str(metadata.get("name") or ""), *_string_list(metadata.get("aliases"))]
        for value in values:
            normalized = value.strip().lower()
            if len(normalized) >= 3 and normalized not in LANGUAGE_TERMS and normalized in raw_query:
                return True
        return False
    return bool(_query_variants(query) & _identity_variants(entry))


def _metadata_text(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("name", "description", "summary", "language", "project_type", "search_text"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    for key in (
        "aliases",
        "topics",
        "use_cases",
        "capabilities",
        "ecosystem",
        "replaces",
        "related_projects",
        "search_phrases",
    ):
        parts.extend(_string_list(metadata.get(key)))
    return "\n".join(parts)


def _metadata_language_alias(language: str) -> str | None:
    aliases = _language_aliases_from_tokens(_tokenize(language))
    for canonical, _ in LANGUAGE_ALIAS_GROUPS:
        if canonical in aliases:
            return canonical
    return None


def _numeric_metadata_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


def _popularity_bonus(metadata: dict[str, Any]) -> float:
    stars = _numeric_metadata_value(metadata.get("stars"))
    if stars is None or stars <= 0:
        return 0.0
    return min(0.015, math.log10(stars + 1.0) * 0.0025)


def _repository_state_penalty(metadata: dict[str, Any]) -> tuple[float, list[str]]:
    penalty = 0.0
    states: list[str] = []
    if metadata.get("archived") is True:
        penalty += 0.08
        states.append("archived")
    if metadata.get("disabled") is True:
        penalty += 0.12
        states.append("disabled")
    return penalty, states


def _metadata_adjustment(query: str, entry: dict[str, Any], semantic_score: float) -> dict[str, Any]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    keyword_tokens = set(_keyword_tokens(query))
    keyword_expanded = _expanded_token_set(keyword_tokens)
    text_tokens = _expanded_token_set(_tokenize(_metadata_text(metadata)))
    topic_tokens = _expanded_token_set(
        tuple(token for topic in _string_list(metadata.get("topics")) for token in _tokenize(topic))
    )
    profile_tokens = {
        token
        for key in ("use_cases", "capabilities", "search_phrases")
        for value in _string_list(metadata.get(key))
        for token in _tokenize(value)
    }
    profile_tokens = _expanded_token_set(profile_tokens)

    exact_identity = _exact_identity_match(query, entry)
    matched_terms = sorted(token for token in keyword_tokens if _expanded_token(token) & text_tokens)
    topic_matches = sorted(token for token in keyword_tokens if _expanded_token(token) & topic_tokens)
    profile_matches = sorted(token for token in keyword_tokens if _expanded_token(token) & profile_tokens)

    adjustment = 0.0
    why: list[str] = []
    primary_language = _query_primary_language_alias(query)
    language = str(metadata.get("language") or "")
    language_alias = _metadata_language_alias(language)
    language_match: str | None = None
    language_mismatch: str | None = None

    if exact_identity:
        adjustment += max(0.25, HIGH_CONFIDENCE_THRESHOLD + 0.05 - semantic_score)
        why.append("matched exact repository identity")
    if primary_language and language_alias == primary_language:
        adjustment += 0.03
        language_match = language
        why.append(f"matched language: {language}")
    elif primary_language and language_alias and language_alias != primary_language:
        adjustment -= 0.04
        language_mismatch = language
        why.append(f"language differs: {language}")

    overlap_count = len(matched_terms)
    if overlap_count:
        adjustment += min(0.08, overlap_count * 0.02)
        why.append("matched metadata terms: " + ", ".join(matched_terms[:5]))
    if topic_matches:
        adjustment += min(0.035, len(topic_matches) * 0.015)
        why.append("matched topics: " + ", ".join(topic_matches[:5]))
    if profile_matches:
        adjustment += min(0.045, len(profile_matches) * 0.015)
        why.append("matched profile terms: " + ", ".join(profile_matches[:5]))

    popularity = _popularity_bonus(metadata)
    if popularity:
        adjustment += popularity
        why.append("popular repository")
    state_penalty, repository_state = _repository_state_penalty(metadata)
    if state_penalty:
        adjustment -= state_penalty
        why.append("repository state penalty: " + ", ".join(repository_state))

    if not why:
        why.append("ranked by semantic similarity")

    return {
        "adjustment": adjustment,
        "exact_identity": exact_identity,
        "matched_terms": matched_terms,
        "diagnostics": {
            "identity_match": "exact" if exact_identity else None,
            "language_match": language_match,
            "language_mismatch": language_mismatch,
            "topic_matches": topic_matches,
            "profile_matches": profile_matches,
            "repository_state": repository_state,
            "popularity_bonus": round(popularity, 6) if popularity else 0.0,
        },
        "why": why,
    }


def _score_breakdown(*, semantic_score: float, metadata_score: float, final_score: float) -> dict[str, float]:
    return {
        "semantic": round(semantic_score, 6),
        "metadata": round(metadata_score, 6),
        "final": round(final_score, 6),
    }


def _result_from_score(query: str, entry: dict[str, Any], semantic_score: float) -> dict[str, Any]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    adjustment = _metadata_adjustment(query, entry, semantic_score)
    metadata_score = float(adjustment["adjustment"])
    final_score = semantic_score + metadata_score
    if adjustment["exact_identity"]:
        confidence = "high_confidence"
    elif semantic_score < EXPLORATORY_THRESHOLD:
        confidence = "abstain"
    else:
        confidence = confidence_bucket(final_score)
    return {
        "repo_id": entry.get("repo_id"),
        "url": metadata.get("url"),
        "score": final_score,
        "semantic_score": semantic_score,
        "metadata_score": metadata_score,
        "confidence": confidence,
        "score_breakdown": _score_breakdown(
            semantic_score=semantic_score,
            metadata_score=metadata_score,
            final_score=final_score,
        ),
        "matched_terms": adjustment["matched_terms"],
        "diagnostics": adjustment["diagnostics"],
        "why": adjustment["why"],
        "_identity_pin": bool(adjustment["exact_identity"]),
    }


def _rank_scored_entries(query: str, scored_entries: list[tuple[dict[str, Any], float]], top_k: int) -> list[dict[str, Any]]:
    results = [_result_from_score(query, entry, score) for entry, score in scored_entries]
    results.sort(
        key=lambda item: (
            1 if item.get("_identity_pin") else 0,
            item["score"],
            item["semantic_score"],
            str(item.get("repo_id") or ""),
        ),
        reverse=True,
    )
    presented = [item for item in results if item["confidence"] != "abstain"][: max(top_k, 0)]
    for item in presented:
        item.pop("_identity_pin", None)
        if item.get("url") is None:
            item.pop("url", None)
    return presented


def ensure_index_matches_model(index: dict[str, Any], config: EmbeddingConfig) -> None:
    index_model = index.get("embedding_model")
    if not index_model:
        raise IndexMismatchError(
            f"Index does not record an embedding_model, but the configured "
            f"model is '{config.model}'. Rebuild the index (xists index build) "
            "so compatibility can be verified."
        )
    if index_model != config.model:
        raise IndexMismatchError(
            f"Index was built with embedding model '{index_model}' but the "
            f"configured model is '{config.model}'. Rebuild the index "
            "(xists index build) or set EMBEDDING_MODEL to match."
        )
    input_version = index.get("embedding_input_version")
    if input_version != EMBEDDING_INPUT_VERSION:
        raise IndexMismatchError(
            f"Index embedding_input_version is {input_version!r}, but xists expects "
            f"{EMBEDDING_INPUT_VERSION}. Refresh profiles if needed, then rebuild "
            "the index with xists index build."
        )
    record_schema_version = index.get("record_schema_version")
    if record_schema_version != RECORD_SCHEMA_VERSION:
        raise IndexMismatchError(
            f"Index record_schema_version is {record_schema_version!r}, but xists expects "
            f"{RECORD_SCHEMA_VERSION}. Run xists profile refresh for older records, "
            "then rebuild the index."
        )


def _normalized_matrix(vectors: list[Any]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise IndexMismatchError("Index vectors must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


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

    entries = [entry for entry in index.get("vectors", []) if isinstance(entry, dict)]
    dimension = index.get("dimension")
    vectors = [decode_vector(entry.get("vector"), dimension=dimension) for entry in entries]
    if any(vector is None for vector in vectors):
        raise IndexMismatchError(
            f"Index contains invalid vectors that do not match its dimension {dimension}. "
            "Rebuild the index with xists index build."
        )

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
            {"query": query, "query_intent": _query_intent(query), "abstained": True, "results": [], "considered": 0}
            for query in queries
        ]

    index_matrix = _normalized_matrix([vector for vector in vectors if vector is not None])
    query_matrix = _normalized_matrix(query_vectors)
    scores = query_matrix @ index_matrix.T
    ranked: list[dict[str, Any]] = []
    for row, query in enumerate(queries):
        scored_entries = [(entry, float(scores[row, column])) for column, entry in enumerate(entries)]
        results = _rank_scored_entries(query, scored_entries, top_k)
        ranked.append(
            {
                "query": query,
                "query_intent": _query_intent(query),
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
    """Rank index entries against the query."""

    ensure_index_matches_model(index, config)
    query_vector = embed(config, query)
    dimension = index.get("dimension")
    if dimension is not None and len(query_vector) != dimension:
        raise IndexMismatchError(
            f"Query vector dimension {len(query_vector)} does not match index "
            f"dimension {dimension}. Rebuild the index or check the model."
        )

    entries = [entry for entry in index.get("vectors", []) if isinstance(entry, dict)]
    vectors = [decode_vector(entry.get("vector"), dimension=dimension) for entry in entries]
    if any(vector is None for vector in vectors):
        raise IndexMismatchError(
            f"Index contains invalid vectors that do not match its dimension {dimension}. "
            "Rebuild the index with xists index build."
        )
    scored_entries = [
        (entry, cosine_similarity(query_vector, vector))
        for entry, vector in zip(entries, vectors)
        if vector is not None
    ]
    results = _rank_scored_entries(query, scored_entries, top_k)
    return {
        "query": query,
        "query_intent": _query_intent(query),
        "abstained": len(results) == 0,
        "results": results,
        "considered": len(entries),
    }
