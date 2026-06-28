"""Generate a stratified retrieval eval dataset from full xists records."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._#-]*", re.IGNORECASE)
STAR_TIERS = [
    ("star-lt100", 0, 99),
    ("star-100-999", 100, 999),
    ("star-1k-9k", 1_000, 9_999),
    ("star-10k-49k", 10_000, 49_999),
    ("star-50k-plus", 50_000, 10**12),
]
QUERY_TYPES = ("simple", "complex", "confusable")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [text for value in values if (text := _text(value))]


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value[:80] or "case"


def _dedupe(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        normalized = " ".join(value.split()).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result


def _star_tier(stars: int) -> str:
    for name, low, high in STAR_TIERS:
        if low <= stars <= high:
            return name
    return "star-unknown"


def _repo_record(record: dict[str, Any]) -> dict[str, Any] | None:
    repo_id = _text(record.get("repo_id"))
    github = record.get("github") or {}
    profile = record.get("llm_profile") or {}
    stars = github.get("stars")
    if not repo_id or not isinstance(stars, int):
        return None
    return {
        "repo_id": repo_id,
        "name": _text(record.get("name")) or repo_id.rsplit("/", 1)[-1],
        "description": _text(github.get("description")),
        "topics": _string_list(github.get("topics")),
        "language": _text(github.get("language")),
        "stars": stars,
        "summary": _text(profile.get("summary")),
        "use_cases": _string_list(profile.get("use_cases")),
        "capabilities": _string_list(profile.get("capabilities")),
        "search_phrases": _string_list(profile.get("search_phrases")),
    }


def _topic_phrase(topics: list[str], limit: int) -> str | None:
    selected = [topic for topic in topics if len(topic) > 1][:limit]
    if not selected:
        return None
    return ", ".join(selected)


def _keyword_phrase(text: str, *, limit: int = 6) -> str | None:
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) <= 2 or token.isdigit():
            continue
        if token in {"and", "for", "the", "with", "from", "that", "this", "into", "using"}:
            continue
        tokens.append(token)
    return " ".join(_dedupe(tokens)[:limit]) or None


def _case(
    record: dict[str, Any],
    *,
    query: str,
    kind: str,
    query_type: str,
    star_tier: str,
    index: int,
) -> dict[str, Any]:
    return {
        "id": f"{_slug(record['repo_id'])}-{query_type}-{kind}-{index}",
        "query": query,
        "expected_repo_id": record["repo_id"],
        "tags": [
            "generated",
            query_type,
            kind,
            star_tier,
            f"language-{_slug(record['language'])}" if record.get("language") else "language-unknown",
        ],
        "notes": (
            "Generated from full records.json with stratified query difficulty "
            f"and GitHub star tier; stars={record['stars']}."
        ),
    }


def build_cases_for_record(record: dict[str, Any], *, star_tier: str) -> list[dict[str, Any]]:
    language = record.get("language")
    simple_queries = _dedupe(
        [
            *record["search_phrases"][:3],
            _topic_phrase(record["topics"], 4),
            record["name"],
            _keyword_phrase(record["description"] or ""),
        ]
    )
    complex_queries = _dedupe(
        [
            record["summary"],
            record["description"],
            *record["use_cases"][:3],
            *record["capabilities"][:3],
        ]
    )
    confusable_queries: list[str] = []
    for phrase in [*record["search_phrases"][:3], *record["capabilities"][:2], _topic_phrase(record["topics"], 5)]:
        if phrase and language:
            confusable_queries.append(f"{language} {phrase}")
        elif phrase:
            confusable_queries.append(f"open source {phrase}")
    if language and record["topics"]:
        confusable_queries.append(f"{language} {_topic_phrase(record['topics'], 5)}")
    confusable_queries = _dedupe(confusable_queries)

    cases: list[dict[str, Any]] = []
    for query_type, queries in (
        ("simple", simple_queries),
        ("complex", complex_queries),
        ("confusable", confusable_queries),
    ):
        for index, query in enumerate(queries, start=1):
            if not query:
                continue
            cases.append(
                _case(
                    record,
                    query=query,
                    kind=f"{query_type}-{index}",
                    query_type=query_type,
                    star_tier=star_tier,
                    index=index,
                )
            )
    return cases


def generate_dataset(records: list[dict[str, Any]], *, limit: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {
        (star_tier, query_type): []
        for star_tier, _, _ in STAR_TIERS
        for query_type in QUERY_TYPES
    }
    seen_ids: set[str] = set()
    for raw in records:
        record = _repo_record(raw)
        if record is None:
            continue
        star_tier = _star_tier(record["stars"])
        for case in build_cases_for_record(record, star_tier=star_tier):
            key = (star_tier, next(tag for tag in case["tags"] if tag in QUERY_TYPES))
            if key not in buckets or case["id"] in seen_ids:
                continue
            seen_ids.add(case["id"])
            buckets[key].append(case)

    for cases in buckets.values():
        rng.shuffle(cases)

    target_per_bucket = max(1, limit // len(buckets))
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        selected.extend(buckets[key][:target_per_bucket])

    if len(selected) < limit:
        leftovers = [
            case
            for key in sorted(buckets)
            for case in buckets[key][target_per_bucket:]
        ]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: limit - len(selected)])

    rng.shuffle(selected)
    selected = selected[:limit]

    return {
        "schema_version": 1,
        "dataset_name": "xists-full-stratified-2000",
        "families": {},
        "cases": selected,
    }


def summarize(dataset: dict[str, Any]) -> dict[str, Any]:
    cases = dataset["cases"]
    tags = Counter(tag for case in cases for tag in case["tags"])
    return {
        "case_count": len(cases),
        "query_types": {query_type: tags[query_type] for query_type in QUERY_TYPES},
        "star_tiers": {name: tags[name] for name, _, _ in STAR_TIERS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a stratified retrieval eval dataset")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    dataset = generate_dataset(records, limit=args.limit, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **summarize(dataset)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
