"""Simple, explainable semantic search over an xists embedding index."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from xists.records import RECORD_SCHEMA_VERSION
from xists.search.bm25 import BM25Index
from xists.search.confidence import CONFIDENCE_CALIBRATION_MODES, calibrate_confidence
from xists.search.embed import (
    EMBEDDING_INPUT_VERSION,
    EmbeddingConfig,
    EmbeddingError,
    call_embeddings,
    embed_query,
)
from xists.search.index import INDEX_VERSION, SUPPORTED_INDEX_VERSIONS, decode_vector
from xists.search.rerank import rerank_text_from_entry
from xists.types import SearchFilter

HIGH_CONFIDENCE_THRESHOLD = 0.60
EXPLORATORY_THRESHOLD = 0.35
RANKING_STRATEGIES = ("metadata", "semantic", "rerank", "hybrid")
RERANK_FUSION_RANK_CONSTANT = 60
HYBRID_FUSION_RANK_CONSTANT = 60
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._#-]*")
CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")
CJK_TERM_LENGTHS = (3, 2)
CJK_TERM_LIMIT = 32
TERMINAL_LOOKUP_PUNCTUATION = ".?!\u3002\uff01\uff1f"
LEADING_LOOKUP_PUNCTUATION = ":\uff1a"
EXPLICIT_LOOKUP_PATTERNS = (
    re.compile(r"^\s*(?:查找|搜索|寻找)\s*(.+?)\s*(?:开源)?项目\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*(?:find|search for|look up)\s+(.+?)\s+(?:open[ -]source\s+)?project\s*$",
        re.IGNORECASE,
    ),
)

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
DOMAIN_QUERY_CUES = {
    "for",
    "in",
    "with",
    "domain",
    "industry",
    "pipelines",
    "infrastructure",
    "observability",
}
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
    # Compact v3 indexes decode to NumPy float32 values.  Cast at this public
    # boundary so single-query JSON output never exposes a NumPy scalar.
    return float(dot / (math.sqrt(norm_a) * math.sqrt(norm_b)))


def confidence_bucket(score: float, *, exploratory_threshold: float = EXPLORATORY_THRESHOLD) -> str:
    if not 0.0 <= exploratory_threshold <= 1.0:
        raise ValueError("exploratory threshold must be between 0 and 1")
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high_confidence"
    if score >= exploratory_threshold:
        return "exploratory"
    return "abstain"


@lru_cache(maxsize=65536)
def _tokenize(text: str) -> tuple[str, ...]:
    ascii_tokens = TOKEN_RE.findall(text.lower())
    tokens = list(ascii_tokens)
    seen = set(tokens)
    for run in CJK_RUN_RE.findall(text):
        for width in CJK_TERM_LENGTHS:
            if len(run) < width:
                continue
            for start in range(len(run) - width + 1):
                term = run[start : start + width]
                if term not in seen:
                    tokens.append(term)
                    seen.add(term)
                    if len(tokens) >= CJK_TERM_LIMIT + len(ascii_tokens):
                        return tuple(tokens)
    return tuple(tokens)


@lru_cache(maxsize=65536)
def _expanded_token(token: str) -> frozenset[str]:
    values = {token}
    values.update(part for part in re.split(r"[-._#]+", token) if part)
    for value in tuple(values):
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
            if (len(alias_tokens) == 1 and alias_tokens[0] in token_set) or alias_text in {
                token_text,
                compact_text,
            }:
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
    elif (
        _explicit_lookup_value(query) is not None
        or "/" in raw_query
        or (
            len(tokens) <= EXACT_NAME_QUERY_MAX_TOKENS
            and all(token not in GENERIC_TERMS and token not in QUERY_JOINERS for token in tokens)
        )
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


def _explicit_lookup_value(query: str) -> str | None:
    normalized_query = query.strip().rstrip(TERMINAL_LOOKUP_PUNCTUATION).strip()
    for pattern in EXPLICIT_LOOKUP_PATTERNS:
        match = pattern.fullmatch(normalized_query)
        if match:
            value = match.group(1).lstrip(LEADING_LOOKUP_PUNCTUATION).strip().lower()
            return value or None
    return None


def _identity_values(entry: dict[str, Any]) -> list[str]:
    raw_meta = entry.get("metadata")
    metadata: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
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


def _identity_match_kind(query: str, entry: dict[str, Any]) -> str:
    raw_query = query.strip().lower()
    repo_id = str(entry.get("repo_id") or "").strip().lower()
    if repo_id and repo_id in raw_query:
        return "repo_id"
    if raw_query in {value.strip().lower() for value in _identity_values(entry)}:
        return "exact_value"
    explicit_value = _explicit_lookup_value(query)
    if explicit_value and explicit_value in {
        value.strip().lower() for value in _identity_values(entry)
    }:
        return "exact_value"
    # A project name embedded in a natural-language request is contextual
    # evidence, never an exact lookup. Ecosystem names such as Node.js and
    # React may be reported as context but must never pin their repositories.
    for value in _identity_values(entry):
        normalized = value.strip().lower()
        value_tokens = _tokenize(normalized)
        if (
            len(normalized) >= 3
            and normalized in raw_query
            and not (set(value_tokens) & LANGUAGE_TERMS)
        ):
            return "contextual_name_mention"
    return "none"


def _identity_match_kind_cached(
    raw_query: str,
    explicit_value: str | None,
    cache: dict[str, Any],
) -> str:
    if cache["repo_id_lower"] and cache["repo_id_lower"] in raw_query:
        return "repo_id"
    if raw_query in cache["identity_values_lower"]:
        return "exact_value"
    if explicit_value and explicit_value in cache["identity_values_lower"]:
        return "exact_value"
    for val_lower in cache["id_value_tokens"]:
        if val_lower in raw_query:
            return "contextual_name_mention"
    return "none"


def _exact_identity_match(query: str, entry: dict[str, Any]) -> bool:
    return _identity_match_kind(query, entry) in {"repo_id", "exact_value"}


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


def _build_query_context(query: str) -> dict[str, Any]:
    raw_query = query.strip().lower()
    keyword_tokens = sorted(_keyword_tokens(query))
    expanded_keyword_map = {token: _expanded_token(token) for token in keyword_tokens}
    explicit_val = _explicit_lookup_value(query)
    primary_lang = _query_primary_language_alias(query)
    return {
        "query": query,
        "raw_query": raw_query,
        "keyword_tokens": keyword_tokens,
        "expanded_keyword_map": expanded_keyword_map,
        "explicit_value": explicit_val,
        "primary_language": primary_lang,
    }


def _precompute_entry_cache(entry: dict[str, Any]) -> dict[str, Any]:
    raw_meta = entry.get("metadata")
    metadata: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    repo_id = str(entry.get("repo_id") or "").strip()
    repo_id_lower = repo_id.lower()

    id_values = _identity_values(entry)
    id_values_lower = {v.strip().lower() for v in id_values if v.strip()}

    id_value_tokens: list[str] = []
    for val in id_values:
        val_lower = val.strip().lower()
        if len(val_lower) >= 3:
            val_tokens = _tokenize(val_lower)
            if not (set(val_tokens) & LANGUAGE_TERMS):
                id_value_tokens.append(val_lower)

    text_tokens = _expanded_token_set(_tokenize(_metadata_text(metadata)))
    topic_tokens = _expanded_token_set(
        tuple(token for topic in _string_list(metadata.get("topics")) for token in _tokenize(topic))
    )
    profile_tokens = _expanded_token_set(
        {
            token
            for key in ("use_cases", "capabilities", "search_phrases")
            for value in _string_list(metadata.get(key))
            for token in _tokenize(value)
        }
    )

    language = str(metadata.get("language") or "")
    language_alias = _metadata_language_alias(language)
    language_lower = language.strip().lower()
    popularity = _popularity_bonus(metadata)
    state_penalty, repository_state = _repository_state_penalty(metadata)

    ecosystem_set = frozenset(
        str(e).strip().lower() for e in _string_list(metadata.get("ecosystem")) if str(e).strip()
    )
    topics_set = frozenset(
        str(t).strip().lower() for t in _string_list(metadata.get("topics")) if str(t).strip()
    )
    project_type_norm = (
        str(metadata.get("project_type") or "").strip().lower().replace("-", "_").replace(" ", "_")
    )
    license_lower = str(metadata.get("license") or "").strip().lower()
    stars = int(_numeric_metadata_value(metadata.get("stars")) or 0)
    forks = int(_numeric_metadata_value(metadata.get("forks")) or 0)
    archived = bool(metadata.get("archived") is True)
    disabled = bool(metadata.get("disabled") is True)

    return {
        "repo_id": repo_id,
        "repo_id_lower": repo_id_lower,
        "identity_values_lower": id_values_lower,
        "id_value_tokens": tuple(id_value_tokens),
        "text_tokens": text_tokens,
        "topic_tokens": topic_tokens,
        "profile_tokens": profile_tokens,
        "language": language,
        "language_alias": language_alias,
        "language_lower": language_lower,
        "popularity_bonus": popularity,
        "state_penalty": state_penalty,
        "repository_state": repository_state,
        "url": metadata.get("url"),
        "ecosystem_set": ecosystem_set,
        "topics_set": topics_set,
        "project_type_norm": project_type_norm,
        "license_lower": license_lower,
        "stars": stars,
        "forks": forks,
        "archived": archived,
        "disabled": disabled,
    }


def _metadata_adjustment(
    query: str, entry: dict[str, Any], semantic_score: float
) -> dict[str, Any]:
    raw_meta = entry.get("metadata")
    metadata: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    keyword_tokens = set(_keyword_tokens(query))
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

    identity_kind = _identity_match_kind(query, entry)
    exact_identity = identity_kind in {"repo_id", "exact_value"}
    contextual_identity = identity_kind == "contextual_name_mention"
    matched_terms = sorted(
        token for token in keyword_tokens if _expanded_token(token) & text_tokens
    )
    topic_matches = sorted(
        token for token in keyword_tokens if _expanded_token(token) & topic_tokens
    )
    profile_matches = sorted(
        token for token in keyword_tokens if _expanded_token(token) & profile_tokens
    )

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
    elif contextual_identity:
        adjustment += 0.015
        why.append("mentioned project name in query context")
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
            "identity_match": "exact"
            if exact_identity
            else "contextual"
            if contextual_identity
            else None,
            "identity_evidence": {"kind": identity_kind},
            "language_match": language_match,
            "language_mismatch": language_mismatch,
            "topic_matches": topic_matches,
            "profile_matches": profile_matches,
            "repository_state": repository_state,
            "popularity_bonus": round(popularity, 6) if popularity else 0.0,
        },
        "why": why,
    }


def _metadata_adjustment_cached(
    query_ctx: dict[str, Any],
    entry: dict[str, Any],
    cache: dict[str, Any],
    semantic_score: float,
) -> dict[str, Any]:
    keyword_tokens = query_ctx["keyword_tokens"]
    expanded_keyword_map = query_ctx["expanded_keyword_map"]

    identity_kind = _identity_match_kind_cached(
        query_ctx["raw_query"],
        query_ctx["explicit_value"],
        cache,
    )
    exact_identity = identity_kind in {"repo_id", "exact_value"}
    contextual_identity = identity_kind == "contextual_name_mention"

    matched_terms = sorted(
        token for token in keyword_tokens if expanded_keyword_map[token] & cache["text_tokens"]
    )
    topic_matches = sorted(
        token for token in keyword_tokens if expanded_keyword_map[token] & cache["topic_tokens"]
    )
    profile_matches = sorted(
        token for token in keyword_tokens if expanded_keyword_map[token] & cache["profile_tokens"]
    )

    adjustment = 0.0
    why: list[str] = []
    primary_language = query_ctx["primary_language"]
    language = cache["language"]
    language_alias = cache["language_alias"]
    language_match: str | None = None
    language_mismatch: str | None = None

    if exact_identity:
        adjustment += max(0.25, HIGH_CONFIDENCE_THRESHOLD + 0.05 - semantic_score)
        why.append("matched exact repository identity")
    elif contextual_identity:
        adjustment += 0.015
        why.append("mentioned project name in query context")
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

    popularity = cache["popularity_bonus"]
    if popularity:
        adjustment += popularity
        why.append("popular repository")
    state_penalty = cache["state_penalty"]
    repository_state = cache["repository_state"]
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
            "identity_match": "exact"
            if exact_identity
            else "contextual"
            if contextual_identity
            else None,
            "identity_evidence": {"kind": identity_kind},
            "language_match": language_match,
            "language_mismatch": language_mismatch,
            "topic_matches": topic_matches,
            "profile_matches": profile_matches,
            "repository_state": repository_state,
            "popularity_bonus": round(popularity, 6) if popularity else 0.0,
        },
        "why": why,
    }


def _score_breakdown(
    *,
    semantic_score: float,
    metadata_score: float,
    final_score: float,
    bm25_score: float | None = None,
) -> dict[str, float]:
    breakdown = {
        "semantic": round(semantic_score, 6),
        "metadata": round(metadata_score, 6),
        "final": round(final_score, 6),
    }
    if bm25_score is not None:
        breakdown["bm25"] = round(bm25_score, 6)
    return breakdown


def _result_from_score(
    query: str,
    entry: dict[str, Any],
    semantic_score: float,
    *,
    exploratory_threshold: float = EXPLORATORY_THRESHOLD,
) -> dict[str, Any]:
    raw_meta = entry.get("metadata")
    metadata: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    adjustment = _metadata_adjustment(query, entry, semantic_score)
    metadata_score = float(adjustment["adjustment"])
    final_score = semantic_score + metadata_score
    if adjustment["exact_identity"]:
        confidence = "high_confidence"
    elif semantic_score < exploratory_threshold:
        confidence = "abstain"
    else:
        confidence = confidence_bucket(final_score, exploratory_threshold=exploratory_threshold)
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


def _result_from_score_cached(
    query_ctx: dict[str, Any],
    entry: dict[str, Any],
    cache: dict[str, Any],
    semantic_score: float,
    *,
    exploratory_threshold: float = EXPLORATORY_THRESHOLD,
) -> dict[str, Any]:
    adjustment = _metadata_adjustment_cached(query_ctx, entry, cache, semantic_score)
    metadata_score = float(adjustment["adjustment"])
    final_score = semantic_score + metadata_score
    if adjustment["exact_identity"]:
        confidence = "high_confidence"
    elif semantic_score < exploratory_threshold:
        confidence = "abstain"
    else:
        confidence = confidence_bucket(final_score, exploratory_threshold=exploratory_threshold)
    return {
        "repo_id": entry.get("repo_id"),
        "url": cache["url"],
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


def _find_exact_identity_indices(prepared: PreparedIndex, query_ctx: dict[str, Any]) -> set[int]:
    """Fast lookup of repository indices matching exact query identity."""
    raw_query = query_ctx["raw_query"]
    explicit_value = query_ctx["explicit_value"]
    exact_indices: set[int] = set()

    if hasattr(prepared, "identity_to_indices") and prepared.identity_to_indices:
        if raw_query:
            for idx in prepared.identity_to_indices.get(raw_query, ()):
                exact_indices.add(idx)
            if raw_query in prepared.repo_id_to_index:
                exact_indices.add(prepared.repo_id_to_index[raw_query])
        if explicit_value:
            for idx in prepared.identity_to_indices.get(explicit_value, ()):
                exact_indices.add(idx)
    else:
        for idx, cache in enumerate(prepared.metadata_caches):
            kind = _identity_match_kind_cached(raw_query, explicit_value, cache)
            if kind in {"repo_id", "exact_value"}:
                exact_indices.add(idx)

    if "/" in raw_query:
        for part in raw_query.split():
            if part in prepared.repo_id_to_index:
                exact_indices.add(prepared.repo_id_to_index[part])

    return exact_indices


def _rank_scored_entries(
    query: str,
    scored_entries: list[tuple[dict[str, Any], float]],
    top_k: int,
    *,
    exploratory_threshold: float = EXPLORATORY_THRESHOLD,
) -> list[dict[str, Any]]:
    results = [
        _result_from_score(query, entry, score, exploratory_threshold=exploratory_threshold)
        for entry, score in scored_entries
    ]
    _downgrade_ambiguous_exact_values(results)
    results.sort(
        key=lambda item: (
            1 if item.get("_identity_pin") else 0,
            item["score"],
            item["semantic_score"],
            str(item.get("repo_id") or ""),
        ),
        reverse=True,
    )
    return _present_ranked_results(results, top_k)


def _rank_scored_entries_prepared(
    prepared: PreparedIndex,
    semantic_scores: np.ndarray,
    top_k: int,
    *,
    query_ctx: dict[str, Any],
    exploratory_threshold: float = EXPLORATORY_THRESHOLD,
    filter_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    valid_indices = (
        np.where(filter_mask)[0] if filter_mask is not None else np.arange(len(prepared.entries))
    )
    if len(valid_indices) == 0:
        return []

    # Two-stage candidate truncation if candidate pool is large
    cand_limit = max(top_k * 8, 128)
    if len(valid_indices) > cand_limit:
        exact_doc_ids = _find_exact_identity_indices(prepared, query_ctx)
        valid_set = set(valid_indices)
        valid_exact = exact_doc_ids & valid_set

        if len(prepared.popularity_bonuses_array) == len(prepared.entries):
            est_scores = (
                semantic_scores[valid_indices] + prepared.popularity_bonuses_array[valid_indices]
            )
        else:
            est_scores = semantic_scores[valid_indices]

        top_sub = np.argpartition(-est_scores, cand_limit)[:cand_limit]
        candidate_indices = set(valid_indices[top_sub]) | valid_exact
        eval_indices = list(candidate_indices)
    else:
        eval_indices = [int(i) for i in valid_indices]

    if hasattr(prepared.entries, "prefetch") and eval_indices:
        prepared.entries.prefetch(eval_indices)

    results = [
        _result_from_score_cached(
            query_ctx,
            prepared.entries[i],
            prepared.metadata_caches[i],
            float(semantic_scores[i]),
            exploratory_threshold=exploratory_threshold,
        )
        for i in eval_indices
    ]
    _downgrade_ambiguous_exact_values(results)
    results.sort(
        key=lambda item: (
            1 if item.get("_identity_pin") else 0,
            item["score"],
            item["semantic_score"],
            str(item.get("repo_id") or ""),
        ),
        reverse=True,
    )
    return _present_ranked_results(results, top_k)


def _semantic_result(
    entry: dict[str, Any],
    semantic_score: float,
    *,
    exploratory_threshold: float,
    query: str | None = None,
) -> dict[str, Any]:
    raw_meta = entry.get("metadata")
    metadata: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    confidence = confidence_bucket(semantic_score, exploratory_threshold=exploratory_threshold)
    identity_kind = _identity_match_kind(query, entry) if query is not None else "none"
    identity_match = "contextual" if identity_kind == "contextual_name_mention" else None
    return {
        "repo_id": entry.get("repo_id"),
        "url": metadata.get("url"),
        "score": semantic_score,
        "semantic_score": semantic_score,
        "metadata_score": 0.0,
        "confidence": confidence,
        "score_breakdown": _score_breakdown(
            semantic_score=semantic_score,
            metadata_score=0.0,
            final_score=semantic_score,
        ),
        "matched_terms": [],
        "diagnostics": {
            "identity_match": identity_match,
            "identity_evidence": {"kind": identity_kind},
        },
        "why": ["ranked by semantic similarity"],
        "_identity_pin": False,
    }


def _semantic_result_cached(
    entry: dict[str, Any],
    cache: dict[str, Any],
    semantic_score: float,
    *,
    exploratory_threshold: float,
    query_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    confidence = confidence_bucket(semantic_score, exploratory_threshold=exploratory_threshold)
    if query_ctx is not None:
        identity_kind = _identity_match_kind_cached(
            query_ctx["raw_query"],
            query_ctx["explicit_value"],
            cache,
        )
    else:
        identity_kind = "none"
    identity_match = "contextual" if identity_kind == "contextual_name_mention" else None
    return {
        "repo_id": entry.get("repo_id"),
        "url": cache["url"],
        "score": semantic_score,
        "semantic_score": semantic_score,
        "metadata_score": 0.0,
        "confidence": confidence,
        "score_breakdown": _score_breakdown(
            semantic_score=semantic_score,
            metadata_score=0.0,
            final_score=semantic_score,
        ),
        "matched_terms": [],
        "diagnostics": {
            "identity_match": identity_match,
            "identity_evidence": {"kind": identity_kind},
        },
        "why": ["ranked by semantic similarity"],
        "_identity_pin": False,
    }


def _present_ranked_results(results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    presented = [item for item in results if item["confidence"] != "abstain"][: max(top_k, 0)]
    for item in presented:
        item.pop("_identity_pin", None)
        if item.get("url") is None:
            item.pop("url", None)
    return presented


def _downgrade_ambiguous_exact_values(results: list[dict[str, Any]]) -> None:
    exact_values = [
        item
        for item in results
        if item.get("diagnostics", {}).get("identity_evidence", {}).get("kind") == "exact_value"
    ]
    if len(exact_values) <= 1:
        return
    for item in exact_values:
        item["confidence"] = "exploratory"
        item["diagnostics"]["identity_ambiguity_count"] = len(exact_values)
        item["why"].append("ambiguous exact identity")


def _rank_semantic_entries(
    scored_entries: list[tuple[dict[str, Any], float]],
    top_k: int,
    *,
    exploratory_threshold: float,
) -> list[dict[str, Any]]:
    results = [
        _semantic_result(entry, score, exploratory_threshold=exploratory_threshold)
        for entry, score in scored_entries
    ]
    results.sort(key=lambda item: (item["score"], str(item.get("repo_id") or "")), reverse=True)
    return _present_ranked_results(results, top_k)


def _rank_semantic_entries_prepared(
    prepared: PreparedIndex,
    semantic_scores: np.ndarray,
    top_k: int,
    *,
    exploratory_threshold: float,
    query_ctx: dict[str, Any] | None = None,
    filter_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    valid_indices = (
        np.where(filter_mask)[0] if filter_mask is not None else np.arange(len(prepared.entries))
    )
    if len(valid_indices) == 0:
        return []

    target_count = max(top_k, 0)
    if len(valid_indices) > target_count:
        valid_sem = semantic_scores[valid_indices]
        if target_count > 0:
            top_sub = np.argpartition(-valid_sem, target_count)[:target_count]
            top_sorted = top_sub[np.argsort(-valid_sem[top_sub])]
            eval_indices = [int(valid_indices[sub]) for sub in top_sorted]
        else:
            eval_indices = []
    else:
        sorted_sub = np.argsort(-semantic_scores[valid_indices])
        eval_indices = [int(valid_indices[sub]) for sub in sorted_sub]

    if hasattr(prepared.entries, "prefetch") and eval_indices:
        prepared.entries.prefetch(eval_indices)

    results = [
        _semantic_result_cached(
            prepared.entries[i],
            prepared.metadata_caches[i],
            float(semantic_scores[i]),
            exploratory_threshold=exploratory_threshold,
            query_ctx=query_ctx,
        )
        for i in eval_indices
    ]
    results.sort(key=lambda item: (item["score"], str(item.get("repo_id") or "")), reverse=True)
    return _present_ranked_results(results, top_k)


def _rank_reranked_entries(
    query: str,
    scored_entries: list[tuple[dict[str, Any], float]],
    top_k: int,
    *,
    rerank: Callable[[str, list[str]], list[float]],
    rerank_query: str,
    candidate_limit: int,
    exploratory_threshold: float,
    rerank_abstain_threshold: float | None,
    confidence_calibration: str,
) -> list[dict[str, Any]]:
    if candidate_limit < 1:
        raise ValueError("rerank candidate limit must be at least 1")
    identity_entries = [item for item in scored_entries if _exact_identity_match(query, item[0])]
    identity_ids = {str(entry.get("repo_id") or "") for entry, _ in identity_entries}
    candidates = sorted(scored_entries, key=lambda item: item[1], reverse=True)
    candidates = [
        item for item in candidates if str(item[0].get("repo_id") or "") not in identity_ids
    ][:candidate_limit]
    rerank_scores = rerank(rerank_query, [rerank_text_from_entry(entry) for entry, _ in candidates])
    if len(rerank_scores) != len(candidates):
        raise ValueError(
            f"reranker returned {len(rerank_scores)} scores for {len(candidates)} candidates"
        )
    rerank_order = sorted(
        range(len(candidates)),
        key=lambda position: (rerank_scores[position], -position),
        reverse=True,
    )
    rerank_ranks = {position: rank for rank, position in enumerate(rerank_order, start=1)}
    results = [
        _result_from_score(query, entry, score, exploratory_threshold=exploratory_threshold)
        for entry, score in identity_entries
    ]
    for semantic_rank, ((entry, semantic_score), rerank_score) in enumerate(
        zip(candidates, rerank_scores), start=1
    ):
        rerank_rank = rerank_ranks[semantic_rank - 1]
        fusion_score = 1.0 / (RERANK_FUSION_RANK_CONSTANT + semantic_rank) + 1.0 / (
            RERANK_FUSION_RANK_CONSTANT + rerank_rank
        )
        result = _semantic_result(
            entry, semantic_score, exploratory_threshold=exploratory_threshold, query=query
        )
        result["score"] = fusion_score
        result["rerank_score"] = float(rerank_score)
        result["score_breakdown"] = _score_breakdown(
            semantic_score=semantic_score,
            metadata_score=0.0,
            final_score=fusion_score,
        )
        result["ranking_evidence"] = {
            "semantic_rank": semantic_rank,
            "rerank_rank": rerank_rank,
            "fusion": "reciprocal_rank",
        }
        result["why"] = ["ranked by fused embedding recall and cross-encoder relevance"]
        results.append(result)
    _downgrade_ambiguous_exact_values(results)
    results.sort(
        key=lambda item: (
            1 if item.get("_identity_pin") else 0,
            item["score"],
            item["semantic_score"],
            str(item.get("repo_id") or ""),
        ),
        reverse=True,
    )
    if not identity_entries and rerank_abstain_threshold is not None and results:
        top_rerank_score = results[0].get("rerank_score")
        if (
            isinstance(top_rerank_score, (int, float))
            and top_rerank_score <= rerank_abstain_threshold
        ):
            return []
    return calibrate_confidence(
        _present_ranked_results(results, top_k),
        ranking_strategy="rerank",
        mode=confidence_calibration,
    )


def _rank_reranked_entries_prepared(
    query: str,
    prepared: PreparedIndex,
    semantic_scores: np.ndarray,
    top_k: int,
    *,
    rerank: Callable[[str, list[str]], list[float]],
    rerank_query: str,
    candidate_limit: int,
    exploratory_threshold: float,
    rerank_abstain_threshold: float | None,
    confidence_calibration: str,
    query_ctx: dict[str, Any] | None = None,
    filter_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if candidate_limit < 1:
        raise ValueError("rerank candidate limit must be at least 1")
    if query_ctx is None:
        query_ctx = _build_query_context(query)

    valid_indices = (
        [int(idx) for idx in np.where(filter_mask)[0]]
        if filter_mask is not None
        else list(range(len(prepared.entries)))
    )
    if not valid_indices:
        return []

    exact_doc_ids = _find_exact_identity_indices(prepared, query_ctx)
    valid_set = set(valid_indices)
    identity_indices = sorted(exact_doc_ids & valid_set)

    identity_ids = {prepared.repo_ids[i] for i in identity_indices if prepared.repo_ids[i]}
    candidate_indices = [i for i in valid_indices if prepared.repo_ids[i] not in identity_ids]
    candidate_indices.sort(key=lambda i: float(semantic_scores[i]), reverse=True)
    candidate_indices = candidate_indices[:candidate_limit]

    all_prefetch = identity_indices + candidate_indices
    if hasattr(prepared.entries, "prefetch") and all_prefetch:
        prepared.entries.prefetch(all_prefetch)

    candidates = [prepared.entries[i] for i in candidate_indices]
    rerank_scores = rerank(rerank_query, [rerank_text_from_entry(entry) for entry in candidates])
    if len(rerank_scores) != len(candidates):
        raise ValueError(
            f"reranker returned {len(rerank_scores)} scores for {len(candidates)} candidates"
        )

    rerank_order = sorted(
        range(len(candidates)),
        key=lambda position: (rerank_scores[position], -position),
        reverse=True,
    )
    rerank_ranks = {position: rank for rank, position in enumerate(rerank_order, start=1)}

    results = [
        _result_from_score_cached(
            query_ctx,
            prepared.entries[i],
            prepared.metadata_caches[i],
            float(semantic_scores[i]),
            exploratory_threshold=exploratory_threshold,
        )
        for i in identity_indices
    ]

    for semantic_rank, (idx, rerank_score) in enumerate(
        zip(candidate_indices, rerank_scores), start=1
    ):
        entry = prepared.entries[idx]
        cache = prepared.metadata_caches[idx]
        semantic_score = float(semantic_scores[idx])
        rerank_rank = rerank_ranks[semantic_rank - 1]
        fusion_score = 1.0 / (RERANK_FUSION_RANK_CONSTANT + semantic_rank) + 1.0 / (
            RERANK_FUSION_RANK_CONSTANT + rerank_rank
        )
        result = _semantic_result_cached(
            entry,
            cache,
            semantic_score,
            exploratory_threshold=exploratory_threshold,
            query_ctx=query_ctx,
        )
        result["score"] = fusion_score
        result["rerank_score"] = float(rerank_score)
        result["score_breakdown"] = _score_breakdown(
            semantic_score=semantic_score,
            metadata_score=0.0,
            final_score=fusion_score,
        )
        result["ranking_evidence"] = {
            "semantic_rank": semantic_rank,
            "rerank_rank": rerank_rank,
            "fusion": "reciprocal_rank",
        }
        result["why"] = ["ranked by fused embedding recall and cross-encoder relevance"]
        results.append(result)

    _downgrade_ambiguous_exact_values(results)
    results.sort(
        key=lambda item: (
            1 if item.get("_identity_pin") else 0,
            item["score"],
            item["semantic_score"],
            str(item.get("repo_id") or ""),
        ),
        reverse=True,
    )
    if not identity_indices and rerank_abstain_threshold is not None and results:
        top_rerank_score = results[0].get("rerank_score")
        if (
            isinstance(top_rerank_score, (int, float))
            and top_rerank_score <= rerank_abstain_threshold
        ):
            return []
    return calibrate_confidence(
        _present_ranked_results(results, top_k),
        ranking_strategy="rerank",
        mode=confidence_calibration,
    )


def _rank_hybrid_entries_prepared(
    query: str,
    prepared: PreparedIndex,
    semantic_scores: np.ndarray,
    top_k: int,
    *,
    query_ctx: dict[str, Any] | None = None,
    exploratory_threshold: float = EXPLORATORY_THRESHOLD,
    confidence_calibration: str = "off",
    query_variants: list[str] | None = None,
    filter_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Rank index entries using Reciprocal Rank Fusion of dense embeddings and BM25 sparse index."""
    if query_ctx is None:
        query_ctx = _build_query_context(query)

    valid_indices = (
        np.where(filter_mask)[0] if filter_mask is not None else np.arange(len(prepared.entries))
    )
    if len(valid_indices) == 0:
        return []

    # 1. Compute BM25 sparse scores across documents
    if query_variants:
        bm25_scores = np.maximum.reduce(
            [prepared.bm25_index.score_query(variant) for variant in query_variants]
        )
    else:
        bm25_scores = prepared.bm25_index.score_query(query)

    # 2. Dense ranking (1-based for valid candidates)
    valid_sem_scores = semantic_scores[valid_indices]
    dense_order = np.argsort(-valid_sem_scores, kind="stable")
    dense_ranks = np.empty(len(valid_indices), dtype=np.int32)
    dense_ranks[dense_order] = np.arange(1, len(valid_indices) + 1)

    # 3. BM25 sparse ranking (1-based for valid candidates with score > 0)
    valid_bm25_scores = bm25_scores[valid_indices]
    matching_sub_indices = np.where(valid_bm25_scores > 0)[0]
    bm25_ranks = np.zeros(len(valid_indices), dtype=np.int32)
    if len(matching_sub_indices) > 0:
        sorted_matching = matching_sub_indices[
            np.argsort(-valid_bm25_scores[matching_sub_indices], kind="stable")
        ]
        bm25_ranks[sorted_matching] = np.arange(1, len(sorted_matching) + 1)

    # 4. Reciprocal Rank Fusion calculation with repository prior
    rrf_k = HYBRID_FUSION_RANK_CONSTANT
    rrf_scores = 1.0 / (rrf_k + dense_ranks) + np.where(
        bm25_ranks > 0, 1.0 / (rrf_k + bm25_ranks), 0.0
    )

    if len(prepared.popularity_bonuses_array) == len(prepared.entries):
        pop_bonuses = prepared.popularity_bonuses_array[valid_indices]
    else:
        pop_bonuses = np.array(
            [
                float(prepared.metadata_caches[i].get("popularity_bonus") or 0.0)
                for i in valid_indices
            ],
            dtype=np.float32,
        )

    fusion_scores_arr = rrf_scores + pop_bonuses * 0.7

    # 5. Two-stage candidate selection
    cand_limit = max(top_k * 4, 64)
    if len(valid_indices) > cand_limit:
        exact_doc_ids = _find_exact_identity_indices(prepared, query_ctx)
        valid_set = set(valid_indices)
        valid_exact = exact_doc_ids & valid_set

        top_sub = np.argpartition(-fusion_scores_arr, cand_limit)[:cand_limit]
        candidate_sub_set = set(top_sub)

        # Include dense #1 and BM25 #1
        candidate_sub_set.add(dense_order[0])
        if len(matching_sub_indices) > 0:
            candidate_sub_set.add(sorted_matching[0])

        # Include any exact identity matches
        if valid_exact:
            valid_doc_to_sub = {int(doc_id): sub for sub, doc_id in enumerate(valid_indices)}
            for doc_id in valid_exact:
                if doc_id in valid_doc_to_sub:
                    candidate_sub_set.add(valid_doc_to_sub[doc_id])

        candidate_sub_indices = list(candidate_sub_set)
    else:
        candidate_sub_indices = list(range(len(valid_indices)))

    candidate_doc_ids = [int(valid_indices[sub_idx]) for sub_idx in candidate_sub_indices]
    if hasattr(prepared.entries, "prefetch") and candidate_doc_ids:
        prepared.entries.prefetch(candidate_doc_ids)

    # 6. Build results with diagnostics, confidence, and explanations for candidate pool
    results: list[dict[str, Any]] = []
    for sub_idx in candidate_sub_indices:
        i = valid_indices[sub_idx]
        entry = prepared.entries[i]
        cache = prepared.metadata_caches[i]
        identity_kind = _identity_match_kind_cached(
            query_ctx["raw_query"],
            query_ctx["explicit_value"],
            cache,
        )
        exact_identity = identity_kind in {"repo_id", "exact_value"}
        contextual_identity = identity_kind == "contextual_name_mention"
        sem_score = float(semantic_scores[i])
        fusion_score = float(fusion_scores_arr[sub_idx])
        bm25_score = float(bm25_scores[i])
        d_rank = int(dense_ranks[sub_idx])
        b_rank = int(bm25_ranks[sub_idx]) if bm25_ranks[sub_idx] > 0 else None

        if exact_identity:
            confidence = "high_confidence"
        elif d_rank == 1 and b_rank == 1 and sem_score >= exploratory_threshold:
            confidence = "high_confidence"
        elif sem_score >= HIGH_CONFIDENCE_THRESHOLD:
            confidence = "high_confidence"
        elif sem_score >= exploratory_threshold or bm25_score > 0:
            confidence = "exploratory"
        else:
            confidence = "abstain"

        why: list[str] = []
        if exact_identity:
            why.append("matched exact repository identity")
        elif b_rank is not None and d_rank <= 10:
            why.append(f"ranked by hybrid fusion (semantic rank #{d_rank}, BM25 rank #{b_rank})")
        elif b_rank is not None:
            why.append(f"ranked by BM25 sparse keyword match (BM25 rank #{b_rank})")
        else:
            why.append("ranked by semantic similarity")

        matched_terms = sorted(
            token
            for token in query_ctx["keyword_tokens"]
            if query_ctx["expanded_keyword_map"][token] & cache["text_tokens"]
        )

        identity_match = (
            "exact" if exact_identity else "contextual" if contextual_identity else None
        )

        item = {
            "repo_id": entry.get("repo_id"),
            "url": cache["url"],
            "score": fusion_score,
            "semantic_score": sem_score,
            "bm25_score": bm25_score,
            "metadata_score": 0.0,
            "confidence": confidence,
            "score_breakdown": _score_breakdown(
                semantic_score=sem_score,
                metadata_score=0.0,
                final_score=fusion_score,
                bm25_score=bm25_score,
            ),
            "ranking_evidence": {
                "semantic_rank": d_rank,
                "bm25_rank": b_rank,
                "fusion": "reciprocal_rank",
            },
            "matched_terms": matched_terms,
            "diagnostics": {
                "identity_match": identity_match,
                "identity_evidence": {"kind": identity_kind},
                "bm25_score": round(bm25_score, 6),
            },
            "why": why,
            "_identity_pin": bool(exact_identity),
        }
        results.append(item)

    _downgrade_ambiguous_exact_values(results)
    results.sort(
        key=lambda item: (
            1 if item.get("_identity_pin") else 0,
            item["score"],
            item["semantic_score"],
            str(item.get("repo_id") or ""),
        ),
        reverse=True,
    )
    return calibrate_confidence(
        _present_ranked_results(results, top_k),
        ranking_strategy="hybrid",
        mode=confidence_calibration,
    )


def _validate_index_structure(index: dict[str, Any]) -> None:
    index_version = index.get("index_version")
    if index_version not in SUPPORTED_INDEX_VERSIONS:
        raise IndexMismatchError(
            f"Index index_version is {index_version!r}, but xists expects {INDEX_VERSION}. "
            "Rebuild the index with xists index build."
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
    vectors = index.get("vectors")
    if not isinstance(vectors, list):
        raise IndexMismatchError(
            "Index vectors must be a list. Rebuild the index with xists index build."
        )
    record_count = index.get("record_count")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 0:
        raise IndexMismatchError(
            "Index record_count must be a non-negative integer. Rebuild the index with xists index build."
        )
    vector_count = index.get("vector_count")
    if vector_count is not None and (
        isinstance(vector_count, bool)
        or not isinstance(vector_count, int)
        or vector_count != len(vectors)
    ):
        raise IndexMismatchError(
            "Index vector_count does not match vectors. Rebuild the index with xists index build."
        )
    dimension = index.get("dimension")
    if dimension is None and not vectors:
        if record_count != 0:
            raise IndexMismatchError(
                "Index record_count does not match repositories represented by vectors. "
                "Rebuild the index with xists index build."
            )
        return
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise IndexMismatchError(
            "Index dimension must be a positive integer. Rebuild the index with xists index build."
        )
    for position, entry in enumerate(vectors):
        if not isinstance(entry, dict):
            raise IndexMismatchError(
                f"Index vector entry {position} must be an object. Rebuild the index with xists index build."
            )
        repo_id = entry.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id.strip():
            raise IndexMismatchError(
                f"Index vector entry {position} has an invalid repo_id. Rebuild the index with xists index build."
            )
    if record_count != len(vectors):
        raise IndexMismatchError(
            "Index record_count does not match vectors. Rebuild the index with xists index build."
        )

    if index_version == 4:
        matrix = index.get("_matrix")
        if matrix is not None:
            if (
                not isinstance(matrix, np.ndarray)
                or matrix.ndim != 2
                or matrix.shape != (len(vectors), dimension)
            ):
                raise IndexMismatchError(
                    f"Index binary vector matrix has invalid shape {getattr(matrix, 'shape', None)}, expected ({len(vectors)}, {dimension}). "
                    "Rebuild the index with xists index build."
                )
        elif not index.get("vectors_file"):
            for entry in vectors:
                if decode_vector(entry.get("vector"), dimension=dimension) is None:
                    raise IndexMismatchError(
                        f"Index contains invalid vectors that do not match its dimension {dimension}. "
                        "Rebuild the index with xists index build."
                    )
    else:
        for entry in vectors:
            if decode_vector(entry.get("vector"), dimension=dimension) is None:
                raise IndexMismatchError(
                    f"Index contains invalid vectors that do not match its dimension {dimension}. "
                    "Rebuild the index with xists index build."
                )


def ensure_index_matches_model(
    index: dict[str, Any] | PreparedIndex, config: EmbeddingConfig
) -> None:
    if isinstance(index, PreparedIndex):
        if index.index_version not in SUPPORTED_INDEX_VERSIONS:
            raise IndexMismatchError(
                f"Index index_version is {index.index_version!r}, but xists expects {INDEX_VERSION}. "
                "Rebuild the index with xists index build."
            )
        if not index.embedding_model:
            raise IndexMismatchError(
                f"Index does not record an embedding_model, but the configured "
                f"model is '{config.model}'. Rebuild the index (xists index build) "
                "so compatibility can be verified."
            )
        if index.embedding_model != config.model:
            raise IndexMismatchError(
                f"Index was built with embedding model '{index.embedding_model}' but the "
                f"configured model is '{config.model}'. Rebuild the index "
                "(xists index build) or set EMBEDDING_MODEL to match."
            )
        if index.embedding_input_version != EMBEDDING_INPUT_VERSION:
            raise IndexMismatchError(
                f"Index embedding_input_version is {index.embedding_input_version!r}, but xists expects "
                f"{EMBEDDING_INPUT_VERSION}. Refresh profiles if needed, then rebuild "
                "the index with xists index build."
            )
        if index.record_schema_version != RECORD_SCHEMA_VERSION:
            raise IndexMismatchError(
                f"Index record_schema_version is {index.record_schema_version!r}, but xists expects "
                f"{RECORD_SCHEMA_VERSION}. Run xists profile refresh for older records, "
                "then rebuild the index."
            )
        if index.record_count != len(index.entries):
            raise IndexMismatchError(
                "Index record_count does not match vectors. "
                "Rebuild the index with xists index build."
            )
        return

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
    _validate_index_structure(index)
    if index_model != config.model:
        raise IndexMismatchError(
            f"Index was built with embedding model '{index_model}' but the "
            f"configured model is '{config.model}'. Rebuild the index "
            "(xists index build) or set EMBEDDING_MODEL to match."
        )
    _validate_index_structure(index)


class PreparedIndex:
    """Pre-computed, memory-efficient index for accelerated vector search."""

    def __init__(
        self,
        *,
        raw_index: dict[str, Any],
        matrix: np.ndarray,
        normalized_matrix: np.ndarray,
        entries: Sequence[dict[str, Any]],
        repo_ids: tuple[str, ...],
        repo_id_to_index: dict[str, int],
        metadata_caches: list[dict[str, Any]],
        index_version: int,
        record_schema_version: int,
        embedding_model: str,
        embedding_base_url: str | None,
        embedding_input_version: int,
        dimension: int | None,
        record_count: int,
        bm25_index: BM25Index | None = None,
        meta_db: Any = None,
        stars_array: np.ndarray | None = None,
        forks_array: np.ndarray | None = None,
        archived_array: np.ndarray | None = None,
    ) -> None:
        self.raw_index = raw_index
        self.matrix = matrix
        self.normalized_matrix = normalized_matrix
        self.entries = entries
        self.repo_ids = repo_ids
        self.repo_id_to_index = repo_id_to_index
        self.metadata_caches = metadata_caches
        self.index_version = index_version
        self.record_schema_version = record_schema_version
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.embedding_input_version = embedding_input_version
        self.dimension = dimension
        self.record_count = record_count
        self.bm25_index = (
            bm25_index if bm25_index is not None else BM25Index.build_from_entries(entries)
        )
        self.meta_db = meta_db
        self.stars_array = (
            stars_array
            if stars_array is not None
            else np.array([c["stars"] for c in metadata_caches], dtype=np.int64)
            if metadata_caches
            else np.empty(0, dtype=np.int64)
        )
        self.forks_array = (
            forks_array
            if forks_array is not None
            else np.array([c.get("forks", 0) for c in metadata_caches], dtype=np.int64)
            if metadata_caches
            else np.empty(0, dtype=np.int64)
        )
        self.archived_array = (
            archived_array
            if archived_array is not None
            else np.array([c["archived"] or c["disabled"] for c in metadata_caches], dtype=bool)
            if metadata_caches
            else np.empty(0, dtype=bool)
        )
        self.popularity_bonuses_array = (
            np.array(
                [float(c.get("popularity_bonus") or 0.0) for c in metadata_caches],
                dtype=np.float32,
            )
            if metadata_caches
            else np.empty(0, dtype=np.float32)
        )
        self.identity_to_indices: dict[str, list[int]] = {}
        for idx, c in enumerate(metadata_caches):
            for id_val in c.get("identity_values_lower") or ():
                self.identity_to_indices.setdefault(id_val, []).append(idx)

    def get_entry(self, doc_id: int) -> dict[str, Any]:
        """Retrieve full entry dictionary for doc_id, using SQLite on-demand if available."""
        if 0 <= doc_id < len(self.entries):
            return self.entries[doc_id]
        if self.meta_db is not None:
            entry = self.meta_db.fetch_entry(doc_id)
            if entry is not None:
                return entry
        raise IndexError(f"Doc ID {doc_id} out of range (count {len(self.entries)})")

    def get_entries_batch(self, doc_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        """Retrieve full entry dictionaries for multiple doc_ids in batch."""
        results: dict[int, dict[str, Any]] = {}
        if hasattr(self.entries, "prefetch"):
            self.entries.prefetch(doc_ids)
            for doc_id in doc_ids:
                if 0 <= doc_id < len(self.entries):
                    results[doc_id] = self.entries[doc_id]
        elif self.meta_db is not None:
            results = self.meta_db.fetch_entries_batch(doc_ids)
        else:
            for doc_id in doc_ids:
                if 0 <= doc_id < len(self.entries):
                    results[doc_id] = self.entries[doc_id]
        return results

    def compute_filter_mask(
        self, filters: SearchFilter | dict[str, Any] | str | None
    ) -> np.ndarray | None:
        """Compute 1D boolean mask of valid candidate indices under structured filters or boolean expressions."""
        if not filters or not self.entries:
            return None

        from xists.search.facets import evaluate_ast_mask, parse_filter_criteria

        ast = parse_filter_criteria(filters)
        if ast is None:
            return None

        return evaluate_ast_mask(ast, self)

    @property
    def is_mmap(self) -> bool:
        """Return True if the underlying vector matrix is memory-mapped."""
        return isinstance(self.matrix, np.memmap)

    @classmethod
    def from_dict(
        cls,
        index: dict[str, Any],
        config: EmbeddingConfig | None = None,
        *,
        mmap: bool = True,
    ) -> PreparedIndex:
        if config is not None:
            ensure_index_matches_model(index, config)
        else:
            _validate_index_structure(index)

        dimension = index.get("dimension")
        entries = [entry for entry in index.get("vectors", []) if isinstance(entry, dict)]

        if index.get("_matrix") is not None:
            matrix = index["_matrix"]
            if not isinstance(matrix, np.ndarray):
                matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.shape != (len(entries), dimension or 0):
                raise IndexMismatchError(
                    f"Index vector matrix shape {matrix.shape} does not match entries count {len(entries)} and dimension {dimension}"
                )
        elif index.get("index_version") == 4 and index.get("_vectors_path"):
            matrix = np.load(index["_vectors_path"], mmap_mode="r" if mmap else None)
            if matrix.shape != (len(entries), dimension or 0):
                raise IndexMismatchError(
                    f"Index vector matrix shape {matrix.shape} does not match entries count {len(entries)} and dimension {dimension}"
                )
        else:
            vectors: list[np.ndarray] = []
            for entry in entries:
                vec = decode_vector(entry.get("vector"), dimension=dimension)
                if vec is None:
                    raise IndexMismatchError(
                        f"Index contains invalid vectors that do not match its dimension {dimension}. "
                        "Rebuild the index with xists index build."
                    )
                vectors.append(vec)

            if vectors:
                matrix = np.asarray(vectors, dtype=np.float32)
            else:
                matrix = np.empty((0, dimension or 0), dtype=np.float32)

        if len(matrix) > 0:
            if matrix.ndim != 2:
                raise IndexMismatchError("Index vectors must be a two-dimensional matrix")
            if index.get("vectors_normalized") is True:
                normalized_matrix = matrix
            else:
                sample_size = min(len(matrix), 16)
                sample_norms = np.linalg.norm(matrix[:sample_size], axis=1)
                if np.allclose(sample_norms, 1.0, atol=1e-3) and (
                    len(matrix) <= sample_size
                    or np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3)
                ):
                    normalized_matrix = matrix
                else:
                    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                    normalized_matrix = np.divide(
                        matrix, norms, out=np.zeros_like(matrix), where=norms != 0
                    )
        else:
            normalized_matrix = np.empty((0, dimension or 0), dtype=np.float32)

        meta_db: Any = None
        stars_arr: np.ndarray | None = None
        forks_arr: np.ndarray | None = None
        archived_arr: np.ndarray | None = None
        metadata_caches: list[dict[str, Any]] | None = None
        repo_ids: tuple[str, ...] | None = None
        repo_id_to_index: dict[str, int] | None = None

        if index.get("_meta_db_path"):
            try:
                from xists.search.meta_db import MetaDatabase

                meta_db_path = Path(index["_meta_db_path"])
                if meta_db_path.is_file():
                    meta_db = MetaDatabase(meta_db_path, read_only=True)
                    (
                        repo_ids,
                        metadata_caches,
                        stars_arr,
                        forks_arr,
                        archived_arr,
                    ) = meta_db.load_search_state()
                    if len(repo_ids) == len(entries):
                        repo_id_to_index = {
                            repo_id: idx for idx, repo_id in enumerate(repo_ids) if repo_id
                        }
                    else:
                        metadata_caches = None
                        repo_ids = None
            except Exception:
                meta_db = None
                metadata_caches = None
                repo_ids = None

        if metadata_caches is None or repo_ids is None:
            repo_ids = tuple(str(entry.get("repo_id") or "") for entry in entries)
            repo_id_to_index = {repo_id: idx for idx, repo_id in enumerate(repo_ids) if repo_id}
            metadata_caches = [_precompute_entry_cache(entry) for entry in entries]

        bm25_index: BM25Index | None = None
        if isinstance(index.get("_bm25_index"), BM25Index):
            bm25_index = index["_bm25_index"]
        elif isinstance(index.get("_bm25"), dict):
            try:
                loaded_bm25 = BM25Index.from_dict(index["_bm25"])
                if loaded_bm25.doc_count == len(entries):
                    bm25_index = loaded_bm25
            except Exception:
                bm25_index = None
        elif index.get("_bm25_path"):
            try:
                bm25_path = Path(index["_bm25_path"])
                if bm25_path.is_file():
                    bm25_raw = json.loads(bm25_path.read_text(encoding="utf-8"))
                    if isinstance(bm25_raw, dict):
                        loaded_bm25 = BM25Index.from_dict(bm25_raw)
                        if loaded_bm25.doc_count == len(entries):
                            bm25_index = loaded_bm25
            except Exception:
                bm25_index = None

        if bm25_index is None:
            bm25_index = BM25Index.build_from_entries(entries)

        return cls(
            raw_index=index,
            matrix=matrix,
            normalized_matrix=normalized_matrix,
            entries=entries,
            repo_ids=repo_ids,
            repo_id_to_index=repo_id_to_index or {},
            metadata_caches=metadata_caches,
            index_version=index.get("index_version", INDEX_VERSION),
            record_schema_version=index.get("record_schema_version", RECORD_SCHEMA_VERSION),
            embedding_model=index.get("embedding_model", ""),
            embedding_base_url=index.get("embedding_base_url"),
            embedding_input_version=index.get("embedding_input_version", EMBEDDING_INPUT_VERSION),
            dimension=dimension,
            record_count=index.get("record_count", len(entries)),
            bm25_index=bm25_index,
            meta_db=meta_db,
            stars_array=stars_arr,
            forks_array=forks_arr,
            archived_array=archived_arr,
        )

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        config: EmbeddingConfig | None = None,
        *,
        mmap: bool = True,
    ) -> PreparedIndex:
        """Load and prepare an index directly from file path with maximum cold-start speed."""
        file_path = Path(path)
        meta_db_path = file_path.with_name(f"{file_path.stem}.meta.db")
        vectors_path = file_path.with_name(f"{file_path.stem}.vectors.npy")
        bm25_path = file_path.with_name(f"{file_path.stem}.bm25.json")

        if meta_db_path.is_file() and vectors_path.is_file():
            from xists.search.meta_db import LazyEntriesList, MetaDatabase

            meta_db = MetaDatabase(meta_db_path, read_only=True)
            manifest = meta_db.get_manifest()
            index_version = int(manifest.get("index_version", INDEX_VERSION))
            dimension = manifest.get("dimension")
            if dimension is not None:
                dimension = int(dimension)
            record_schema_version = int(
                manifest.get("record_schema_version", RECORD_SCHEMA_VERSION)
            )
            embedding_model = str(manifest.get("embedding_model") or "")
            embedding_base_url = manifest.get("embedding_base_url")
            embedding_input_version = int(
                manifest.get("embedding_input_version", EMBEDDING_INPUT_VERSION)
            )

            repo_ids, metadata_caches, stars_arr, forks_arr, archived_arr = (
                meta_db.load_search_state()
            )
            repo_id_to_index = {repo_id: idx for idx, repo_id in enumerate(repo_ids) if repo_id}
            entries = LazyEntriesList(meta_db, repo_ids)

            matrix = np.load(vectors_path, mmap_mode="r" if mmap else None)
            if matrix.ndim == 2:
                normalized_matrix = matrix
            else:
                normalized_matrix = np.empty((0, dimension or 0), dtype=np.float32)

            bm25_index: BM25Index | None = None
            if bm25_path.is_file():
                try:
                    bm25_raw = json.loads(bm25_path.read_text(encoding="utf-8"))
                    if isinstance(bm25_raw, dict):
                        loaded_bm25 = BM25Index.from_dict(bm25_raw)
                        if loaded_bm25.doc_count == len(repo_ids):
                            bm25_index = loaded_bm25
                except Exception:
                    bm25_index = None

            if bm25_index is None:
                bm25_index = BM25Index.build_from_entries(entries)

            raw_doc = {
                "index_version": index_version,
                "record_schema_version": record_schema_version,
                "embedding_model": embedding_model,
                "embedding_base_url": embedding_base_url,
                "embedding_input_version": embedding_input_version,
                "dimension": dimension,
                "record_count": len(repo_ids),
                "_meta_db_path": str(meta_db_path),
                "_vectors_path": str(vectors_path),
                "_bm25_path": str(bm25_path) if bm25_path.is_file() else None,
            }

            prepared = cls(
                raw_index=raw_doc,
                matrix=matrix,
                normalized_matrix=normalized_matrix,
                entries=entries,
                repo_ids=repo_ids,
                repo_id_to_index=repo_id_to_index,
                metadata_caches=metadata_caches,
                index_version=index_version,
                record_schema_version=record_schema_version,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url,
                embedding_input_version=embedding_input_version,
                dimension=dimension,
                record_count=len(repo_ids),
                bm25_index=bm25_index,
                meta_db=meta_db,
                stars_array=stars_arr,
                forks_array=forks_arr,
                archived_array=archived_arr,
            )
            if config is not None:
                ensure_index_matches_model(prepared, config)
            return prepared

        # Fallback to load_index
        from xists.search.index import load_index

        doc = load_index(file_path, mmap=mmap)
        return cls.from_dict(doc, config=config, mmap=mmap)

    def __getitem__(self, key: str) -> Any:
        if key == "vectors":
            return self.entries
        if key == "dimension":
            return self.dimension
        if key == "record_count":
            return self.record_count
        if key == "index_version":
            return self.index_version
        if key == "record_schema_version":
            return self.record_schema_version
        if key == "embedding_model":
            return self.embedding_model
        if key == "embedding_base_url":
            return self.embedding_base_url
        if key == "embedding_input_version":
            return self.embedding_input_version
        if isinstance(self.raw_index, dict):
            return self.raw_index[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        if key in {
            "vectors",
            "dimension",
            "record_count",
            "index_version",
            "record_schema_version",
            "embedding_model",
            "embedding_base_url",
            "embedding_input_version",
        }:
            return True
        if isinstance(self.raw_index, dict):
            return key in self.raw_index
        return False

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        if isinstance(self.raw_index, dict):
            return iter(self.raw_index)
        return iter(())


def prepare_index(
    index: dict[str, Any] | PreparedIndex | Path | str,
    config: EmbeddingConfig | None = None,
    *,
    mmap: bool = True,
) -> PreparedIndex:
    """Ensure an index is in the accelerated in-memory PreparedIndex format."""

    if isinstance(index, PreparedIndex):
        if config is not None:
            ensure_index_matches_model(index, config)
        return index
    if isinstance(index, (str, Path)) and Path(index).is_file():
        return PreparedIndex.from_path(index, config, mmap=mmap)
    if isinstance(index, dict):
        cached_prep = index.get("_prepared_index")
        if isinstance(cached_prep, PreparedIndex):
            if config is not None:
                ensure_index_matches_model(cached_prep, config)
            return cached_prep
        prepared = PreparedIndex.from_dict(index, config, mmap=mmap)
        index["_prepared_index"] = prepared
        return prepared
    raise IndexMismatchError(
        f"Index must be a dictionary or PreparedIndex, got {type(index).__name__}"
    )


def _normalized_matrix(vectors: list[Any]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise IndexMismatchError("Index vectors must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def rank_many(
    queries: list[str],
    index: dict[str, Any] | PreparedIndex,
    config: EmbeddingConfig,
    *,
    top_k: int = 10,
    batch_size: int = 64,
    embed_many: Any = call_embeddings,
    ranking_strategy: str = "metadata",
    rerank: Callable[[str, list[str]], list[float]] | None = None,
    rerank_candidate_limit: int = 50,
    exploratory_threshold: float = EXPLORATORY_THRESHOLD,
    rerank_abstain_threshold: float | None = None,
    confidence_calibration: str = "off",
    query_variants: list[list[str]] | None = None,
    rerank_queries: list[str] | None = None,
    filters: SearchFilter | dict[str, Any] | str | None = None,
) -> list[dict[str, Any]]:
    """Rank multiple queries with batched embeddings and matrix similarity."""

    prepared = prepare_index(index, config)
    if ranking_strategy not in RANKING_STRATEGIES:
        raise ValueError(f"Unknown ranking strategy: {ranking_strategy}")
    if ranking_strategy == "rerank" and rerank is None:
        raise ValueError("A reranker is required when ranking_strategy is rerank")
    if not 0.0 <= exploratory_threshold <= 1.0:
        raise ValueError("exploratory threshold must be between 0 and 1")
    if confidence_calibration not in CONFIDENCE_CALIBRATION_MODES:
        raise ValueError(f"Unknown confidence calibration mode: {confidence_calibration}")
    if not queries:
        return []
    if query_variants is None:
        query_variants = [[query] for query in queries]
    if len(query_variants) != len(queries) or any(not variants for variants in query_variants):
        raise ValueError("query variants must contain at least one value for every query")
    if rerank_queries is None:
        rerank_queries = list(queries)
    if len(rerank_queries) != len(queries):
        raise ValueError("rerank queries must contain one value for every query")

    if not prepared.entries:
        return [
            {
                "query": query,
                "query_intent": _query_intent(query),
                "abstained": True,
                "results": [],
                "considered": 0,
            }
            for query in queries
        ]

    filter_mask = prepared.compute_filter_mask(filters)
    total_candidates = (
        int(np.sum(filter_mask)) if filter_mask is not None else len(prepared.entries)
    )

    flattened_variants = [variant for variants in query_variants for variant in variants]
    query_vectors: list[list[float]] = []
    for start in range(0, len(flattened_variants), batch_size):
        batch = flattened_variants[start : start + batch_size]
        if embed_many is call_embeddings:
            query_vectors.extend(embed_many(config, batch, input_type="query"))
        else:
            query_vectors.extend(embed_many(config, batch))
    if len(query_vectors) != len(flattened_variants):
        raise EmbeddingError(
            f"Embedding count mismatch: sent {len(flattened_variants)}, received {len(query_vectors)}"
        )
    dimension = prepared.dimension
    if dimension is not None and any(len(vector) != dimension for vector in query_vectors):
        raise IndexMismatchError(
            f"One or more query vectors do not match index dimension {dimension}. "
            "Rebuild the index or check the model."
        )

    q_mat = np.asarray(query_vectors, dtype=np.float32)
    q_norms = np.linalg.norm(q_mat, axis=1, keepdims=True)
    q_norm = np.divide(q_mat, q_norms, out=np.zeros_like(q_mat), where=q_norms != 0)
    scores = q_norm @ prepared.normalized_matrix.T
    ranked: list[dict[str, Any]] = []
    offset = 0
    for row, query in enumerate(queries):
        started = perf_counter()
        variant_count = len(query_variants[row])
        variant_scores = scores[offset : offset + variant_count]
        offset += variant_count
        semantic_scores = variant_scores.max(axis=0)
        query_ctx = _build_query_context(query)

        if filter_mask is not None and not np.any(filter_mask):
            results = []
        elif ranking_strategy == "semantic":
            results = _rank_semantic_entries_prepared(
                prepared,
                semantic_scores,
                top_k,
                exploratory_threshold=exploratory_threshold,
                query_ctx=query_ctx,
                filter_mask=filter_mask,
            )
        elif ranking_strategy == "rerank":
            assert rerank is not None
            results = _rank_reranked_entries_prepared(
                query,
                prepared,
                semantic_scores,
                top_k,
                rerank=rerank,
                rerank_query=rerank_queries[row],
                candidate_limit=rerank_candidate_limit,
                exploratory_threshold=exploratory_threshold,
                rerank_abstain_threshold=rerank_abstain_threshold,
                confidence_calibration=confidence_calibration,
                query_ctx=query_ctx,
                filter_mask=filter_mask,
            )
        elif ranking_strategy == "hybrid":
            results = _rank_hybrid_entries_prepared(
                query,
                prepared,
                semantic_scores,
                top_k,
                query_ctx=query_ctx,
                exploratory_threshold=exploratory_threshold,
                confidence_calibration=confidence_calibration,
                query_variants=query_variants[row] if query_variants[row] != [query] else None,
                filter_mask=filter_mask,
            )
        else:
            results = _rank_scored_entries_prepared(
                prepared,
                semantic_scores,
                top_k,
                query_ctx=query_ctx,
                exploratory_threshold=exploratory_threshold,
                filter_mask=filter_mask,
            )

        item = {
            "query": query,
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "query_intent": _query_intent(query),
            "abstained": len(results) == 0,
            "results": results,
            "considered": total_candidates,
        }
        if filters:
            item["filters"] = filters
        if query_variants[row] != [query]:
            item["query_variants"] = query_variants[row]
        ranked.append(item)
    return ranked


def rank(
    query: str,
    index: dict[str, Any] | PreparedIndex,
    config: EmbeddingConfig,
    *,
    top_k: int = 10,
    embed: Any = embed_query,
    ranking_strategy: str = "metadata",
    rerank: Callable[[str, list[str]], list[float]] | None = None,
    rerank_candidate_limit: int = 50,
    exploratory_threshold: float = EXPLORATORY_THRESHOLD,
    rerank_abstain_threshold: float | None = None,
    confidence_calibration: str = "off",
    query_variants: list[str] | None = None,
    rerank_query: str | None = None,
    filters: SearchFilter | dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Rank index entries against the query."""

    started = perf_counter()
    prepared = prepare_index(index, config)
    if ranking_strategy not in RANKING_STRATEGIES:
        raise ValueError(f"Unknown ranking strategy: {ranking_strategy}")
    if not 0.0 <= exploratory_threshold <= 1.0:
        raise ValueError("exploratory threshold must be between 0 and 1")
    if confidence_calibration not in CONFIDENCE_CALIBRATION_MODES:
        raise ValueError(f"Unknown confidence calibration mode: {confidence_calibration}")
    if query_variants is None:
        query_variants = [query]
    if not query_variants:
        raise ValueError("query variants must contain at least one value")
    if rerank_query is None:
        rerank_query = query

    query_vectors = [embed(config, variant) for variant in query_variants]
    dimension = prepared.dimension
    if dimension is not None and any(len(vector) != dimension for vector in query_vectors):
        raise IndexMismatchError(
            "Query vector dimension does not match index "
            f"dimension {dimension}. Rebuild the index or check the model."
        )

    if not prepared.entries:
        result = {
            "query": query,
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "query_intent": _query_intent(query),
            "abstained": True,
            "results": [],
            "considered": 0,
        }
        if filters:
            result["filters"] = filters
        if query_variants != [query]:
            result["query_variants"] = query_variants
        return result

    filter_mask = prepared.compute_filter_mask(filters)
    total_candidates = (
        int(np.sum(filter_mask)) if filter_mask is not None else len(prepared.entries)
    )

    q_mat = np.asarray(query_vectors, dtype=np.float32)
    q_norms = np.linalg.norm(q_mat, axis=1, keepdims=True)
    q_norm = np.divide(q_mat, q_norms, out=np.zeros_like(q_mat), where=q_norms != 0)
    sim_matrix = q_norm @ prepared.normalized_matrix.T
    semantic_scores = sim_matrix.max(axis=0)

    query_ctx = _build_query_context(query)
    if filter_mask is not None and not np.any(filter_mask):
        results = []
    elif ranking_strategy == "semantic":
        results = _rank_semantic_entries_prepared(
            prepared,
            semantic_scores,
            top_k,
            exploratory_threshold=exploratory_threshold,
            query_ctx=query_ctx,
            filter_mask=filter_mask,
        )
    elif ranking_strategy == "rerank":
        if rerank is None:
            raise ValueError("A reranker is required when ranking_strategy is rerank")
        results = _rank_reranked_entries_prepared(
            query,
            prepared,
            semantic_scores,
            top_k,
            rerank=rerank,
            rerank_query=rerank_query,
            candidate_limit=rerank_candidate_limit,
            exploratory_threshold=exploratory_threshold,
            rerank_abstain_threshold=rerank_abstain_threshold,
            confidence_calibration=confidence_calibration,
            query_ctx=query_ctx,
            filter_mask=filter_mask,
        )
    elif ranking_strategy == "hybrid":
        results = _rank_hybrid_entries_prepared(
            query,
            prepared,
            semantic_scores,
            top_k,
            query_ctx=query_ctx,
            exploratory_threshold=exploratory_threshold,
            confidence_calibration=confidence_calibration,
            query_variants=query_variants if query_variants != [query] else None,
            filter_mask=filter_mask,
        )
    else:
        results = _rank_scored_entries_prepared(
            prepared,
            semantic_scores,
            top_k,
            query_ctx=query_ctx,
            exploratory_threshold=exploratory_threshold,
            filter_mask=filter_mask,
        )

    result = {
        "query": query,
        "latency_ms": round((perf_counter() - started) * 1000, 3),
        "query_intent": _query_intent(query),
        "abstained": len(results) == 0,
        "results": results,
        "considered": total_candidates,
    }
    if filters:
        result["filters"] = filters
    if query_variants != [query]:
        result["query_variants"] = query_variants
    return result
