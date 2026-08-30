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
    terms = [
        t
        for t in re.split(r"[\s,._/\\:;!?'\"()\[\]{}#~`*+=<>@$%^&|]+", text.lower())
        if len(t) >= 2
    ]
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
    query_set = set(query_terms)

    is_framework_query = "framework" in query_set or "frameworks" in query_set
    is_library_query = "library" in query_set or "lib" in query_set
    is_tool_query = any(
        t in query_set
        for t in ("tool", "tools", "cli", "runtime", "engine", "server", "editor", "database", "db")
    )
    is_collection_query = any(
        t in query_set
        for t in (
            "awesome",
            "list",
            "collection",
            "curated",
            "tutorial",
            "primer",
            "guide",
            "learn",
        )
    )

    known_languages = {
        "python",
        "rust",
        "javascript",
        "typescript",
        "go",
        "golang",
        "c++",
        "cpp",
        "c#",
        "java",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "zig",
    }
    query_languages = query_set.intersection(known_languages)
    if "golang" in query_languages:
        query_languages.add("go")

    scored_entries: list[dict[str, Any]] = []

    for record in records:
        repo_id = str(record.get("repo_id") or "")
        name = str(record.get("name") or "")
        url = str(record.get("url") or "")
        raw_github = record.get("github")
        github: dict[str, Any] = raw_github if isinstance(raw_github, dict) else {}
        raw_profile = record.get("llm_profile")
        profile: dict[str, Any] = raw_profile if isinstance(raw_profile, dict) else {}

        description = str(github.get("description") or "").lower()
        language = str(github.get("language") or "").lower()
        topics = [str(t).lower() for t in (github.get("topics") or []) if isinstance(t, (str, int))]
        stars = int(github.get("stars") or 0)

        summary = str(profile.get("summary") or "").lower()
        project_type = str(profile.get("project_type") or "").lower()
        use_cases = [str(u).lower() for u in (profile.get("use_cases") or []) if isinstance(u, str)]
        capabilities = [
            str(c).lower() for c in (profile.get("capabilities") or []) if isinstance(c, str)
        ]
        aliases = [str(a).lower() for a in (profile.get("aliases") or []) if isinstance(a, str)]
        search_phrases = [
            str(p).lower() for p in (profile.get("search_phrases") or []) if isinstance(p, str)
        ]
        search_text = str(profile.get("search_text") or "").lower()
        ecosystem = [str(e).lower() for e in (profile.get("ecosystem") or []) if isinstance(e, str)]
        replaces = [str(r).lower() for r in (profile.get("replaces") or []) if isinstance(r, str)]

        score = 0.0
        matched_terms: set[str] = set()
        why_reasons: list[str] = []

        # 1. Exact or prefix match on name/repo_id/alias
        if norm_query == name.lower() or norm_query == repo_id.lower():
            score += 4.0
            matched_terms.add(norm_query)
            why_reasons.append("exact name match")
        elif norm_query in aliases:
            score += 3.5
            matched_terms.add(norm_query)
            why_reasons.append("exact alias match")
        elif any(norm_query == p for p in search_phrases):
            score += 3.0
            matched_terms.add(norm_query)
            why_reasons.append("exact search phrase match")

        # 2. Phrase matching across summary / search_text / description
        if norm_query in search_text or norm_query in summary or norm_query in description:
            score += 1.5
            why_reasons.append(f'matched exact phrase: "{norm_query}"')
        elif len(query_terms) >= 2:
            for i in range(len(query_terms) - 1):
                subphrase = f"{query_terms[i]} {query_terms[i + 1]}"
                if subphrase in search_text or subphrase in summary or subphrase in description:
                    score += 0.6
                    break

        # 3. Term-based matching across fields
        for term in query_set:
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
            elif (
                any(term in u for u in use_cases)
                or any(term in c for c in capabilities)
                or any(term in e for e in ecosystem)
            ):
                score += 0.5
                term_matched = True
            elif term in description or term in summary:
                score += 0.3
                term_matched = True
            elif term in search_text:
                score += 0.2
                term_matched = True
            elif any(term in r for r in replaces):
                score += 0.3
                term_matched = True

            if term_matched:
                matched_terms.add(term)

        # 4. Coverage Multiplier
        if query_terms:
            coverage = len(matched_terms) / len(query_set)
            if coverage >= 1.0:
                score += 1.0
                why_reasons.append("matched all query terms")
            elif coverage < 0.5 and not why_reasons:
                score *= 0.6

        # 5. Language Intent Alignment
        if query_languages:
            if language in query_languages or any(ql in ecosystem for ql in query_languages):
                score += 1.0
                why_reasons.append(f"matched language: {language.title()}")
            elif (
                language
                and language not in query_languages
                and not any(ql in ecosystem for ql in query_languages)
            ):
                score -= 0.8

        # 6. Project Type Intent Alignment
        if is_framework_query:
            if project_type == "framework":
                score += 1.2
                why_reasons.append("matched project type: framework")
            elif project_type == "library":
                score += 0.4
        elif is_library_query:
            if project_type in ("library", "framework"):
                score += 1.0
        elif is_tool_query:
            if project_type in (
                "tool",
                "cli",
                "runtime",
                "engine",
                "service",
                "platform",
                "database",
            ):
                score += 1.0

        # 7. Collection / Awesome-list / Primer Demotion when not explicitly asked
        if not is_collection_query and project_type in (
            "collection",
            "awesome_list",
            "tutorial",
            "educational",
        ):
            score -= 1.8

        # 8. Popularity / star factor (small logarithmic boost)
        if score > 0 and stars > 0:
            score += 0.05 * math.log10(stars + 1)

        if score > 0:
            # Determine confidence bucket
            if score >= 2.5:
                confidence = "high"
            elif score >= 1.2:
                confidence = "medium"
            elif score >= 0.4:
                confidence = "exploratory"
            else:
                confidence = "abstain"

            if not why_reasons and matched_terms:
                why_reasons.append(f"matched keywords: {', '.join(sorted(matched_terms))}")

            scored_entries.append(
                {
                    "repo_id": repo_id,
                    "url": url,
                    "score": round(score, 4),
                    "semantic_score": 0.0,
                    "metadata_score": round(score, 4),
                    "confidence": confidence,
                    "summary": profile.get("summary") or github.get("description") or "",
                    "use_cases": profile.get("use_cases") or [],
                    "capabilities": profile.get("capabilities") or [],
                    "topics": github.get("topics") or [],
                    "stars": stars,
                    "language": github.get("language") or "",
                    "matched_terms": sorted(matched_terms),
                    "why": why_reasons if why_reasons else ["offline lexical match"],
                    "score_breakdown": {
                        "semantic_score": 0.0,
                        "metadata_score": round(score, 4),
                        "final_score": round(score, 4),
                    },
                }
            )

    scored_entries.sort(
        key=lambda item: (item["score"], item["stars"], item["repo_id"]), reverse=True
    )
    presented = [item for item in scored_entries if item["confidence"] != "abstain"][
        : max(top_k, 0)
    ]

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
