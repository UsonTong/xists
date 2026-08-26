"""Bundled starter demo dataset and offline search utilities for xists."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from xists.search.index import load_index

STARTER_DIR = Path(__file__).resolve().parent
STARTER_RECORDS_PATH = STARTER_DIR / "records.json"
STARTER_INDEX_PATH = STARTER_DIR / "index.json"
STARTER_VECTORS_PATH = STARTER_DIR / "index.vectors.npy"


def get_starter_records_path() -> Path:
    """Return the absolute path to the bundled starter records JSON file."""
    return STARTER_RECORDS_PATH


def get_starter_index_path() -> Path:
    """Return the absolute path to the bundled starter index JSON file."""
    return STARTER_INDEX_PATH


def load_starter_records() -> list[dict[str, Any]]:
    """Load the bundled starter records."""
    if not STARTER_RECORDS_PATH.is_file():
        return []
    content = STARTER_RECORDS_PATH.read_text(encoding="utf-8")
    data = json.loads(content)
    return data if isinstance(data, list) else []


def load_starter_index(*, mmap: bool = True) -> dict[str, Any]:
    """Load the bundled starter index."""
    if not STARTER_INDEX_PATH.is_file():
        return {}
    return load_index(STARTER_INDEX_PATH, mmap=mmap)


def _tokenize_terms(text: str) -> list[str]:
    """Extract normalized lowercase tokens from text."""
    if not text:
        return []
    # Split by whitespace, punctuation, hyphens, underscores
    terms = [t for t in re.split(r"[\s,._/\\:;!?'\"()\[\]{}#~`*+=<>@$%^&|]+", text.lower()) if len(t) >= 2]
    return terms


def starter_metadata_search(
    query: str,
    records: list[dict[str, Any]] | None = None,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Offline lexical and metadata search across starter records without API keys."""
    started = perf_counter()
    if records is None:
        records = load_starter_records()

    raw_query = query.strip()
    norm_query = raw_query.lower()
    query_terms = _tokenize_terms(raw_query)

    scored_entries: list[dict[str, Any]] = []

    for record in records:
        repo_id = str(record.get("repo_id") or "")
        name = str(record.get("name") or "")
        url = str(record.get("url") or "")
        github = record.get("github") if isinstance(record.get("github"), dict) else {}
        profile = record.get("llm_profile") if isinstance(record.get("llm_profile"), dict) else {}

        description = str(github.get("description") or "")
        language = str(github.get("language") or "")
        topics = [str(t).lower() for t in (github.get("topics") or []) if isinstance(t, (str, int))]
        stars = int(github.get("stars") or 0)

        summary = str(profile.get("summary") or "")
        use_cases = [str(u) for u in (profile.get("use_cases") or []) if isinstance(u, str)]
        capabilities = [str(c) for c in (profile.get("capabilities") or []) if isinstance(c, str)]
        aliases = [str(a).lower() for a in (profile.get("aliases") or []) if isinstance(a, str)]
        search_phrases = [str(p).lower() for p in (profile.get("search_phrases") or []) if isinstance(p, str)]
        search_text = str(profile.get("search_text") or "").lower()
        ecosystem = [str(e).lower() for e in (profile.get("ecosystem") or []) if isinstance(e, str)]
        replaces = [str(r).lower() for r in (profile.get("replaces") or []) if isinstance(r, str)]

        score = 0.0
        matched_terms: list[str] = []
        why_reasons: list[str] = []

        # 1. Exact or prefix match on name/repo_id/alias
        if norm_query == name.lower() or norm_query == repo_id.lower():
            score += 3.0
            matched_terms.append(norm_query)
            why_reasons.append("exact name match")
        elif norm_query in aliases:
            score += 2.5
            matched_terms.append(norm_query)
            why_reasons.append("exact alias match")
        elif any(norm_query == p for p in search_phrases):
            score += 2.0
            matched_terms.append(norm_query)
            why_reasons.append("exact search phrase match")

        # 2. Term-based matching across fields
        for term in set(query_terms):
            term_matched = False
            if term == name.lower() or term in repo_id.lower():
                score += 1.2
                term_matched = True
            elif term in aliases:
                score += 1.0
                term_matched = True
            elif any(term in t for t in topics):
                score += 0.8
                term_matched = True
            elif any(term in p for p in search_phrases):
                score += 0.7
                term_matched = True
            elif any(term in u.lower() for u in use_cases):
                score += 0.5
                term_matched = True
            elif any(term in c.lower() for c in capabilities):
                score += 0.5
                term_matched = True
            elif any(term in e for e in ecosystem):
                score += 0.5
                term_matched = True
            elif term in description.lower() or term in summary.lower():
                score += 0.3
                term_matched = True
            elif term in search_text:
                score += 0.2
                term_matched = True
            elif any(term in r for r in replaces):
                score += 0.3
                term_matched = True

            if term_matched:
                matched_terms.append(term)

        # 3. Popularity / star factor (small logarithmic boost)
        if score > 0 and stars > 0:
            score += 0.05 * math.log10(stars + 1)

        if score > 0:
            # Determine confidence bucket
            if score >= 2.0:
                confidence = "high"
            elif score >= 1.0:
                confidence = "medium"
            elif score >= 0.3:
                confidence = "exploratory"
            else:
                confidence = "abstain"

            if not why_reasons and matched_terms:
                why_reasons.append(f"matched keywords: {', '.join(sorted(set(matched_terms)))}")

            scored_entries.append({
                "repo_id": repo_id,
                "url": url,
                "score": round(score, 4),
                "semantic_score": 0.0,
                "metadata_score": round(score, 4),
                "confidence": confidence,
                "summary": summary or description,
                "use_cases": use_cases,
                "capabilities": capabilities,
                "topics": topics,
                "stars": stars,
                "language": language,
                "matched_terms": sorted(set(matched_terms)),
                "why": why_reasons if why_reasons else ["offline lexical match"],
                "score_breakdown": {
                    "semantic_score": 0.0,
                    "metadata_score": round(score, 4),
                    "final_score": round(score, 4),
                },
            })

    scored_entries.sort(key=lambda item: (item["score"], item["stars"], item["repo_id"]), reverse=True)
    presented = [item for item in scored_entries if item["confidence"] != "abstain"][:max(top_k, 0)]

    elapsed_ms = round((perf_counter() - started) * 1000, 3)

    return {
        "query": query,
        "latency_ms": elapsed_ms,
        "query_intent": "offline_starter_search",
        "abstained": len(presented) == 0,
        "mode": "offline_starter",
        "results": presented,
        "considered": len(records),
    }
