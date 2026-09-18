#!/usr/bin/env python3
"""Generate a >=20k multi-tier evaluation benchmark dataset for the 500k repository corpus.

This script streams from data/index_500k.meta.db without loading the full corpus into memory,
generating 20,000+ high-quality evaluation cases stratified across 4 difficulty tiers:
  Tier 1: Easy / Exact match (~6,000 cases)
  Tier 2: Medium / Functional capability (~8,000 cases)
  Tier 3: Hard / Architectural & Alternative (~4,000 cases)
  Tier 4: Confusable / Constrained (~2,000 cases)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import resource
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# Hard memory safety limit: 2.0 GB virtual memory guard
try:
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
except (ValueError, OSError):
    pass

DB_PATH = Path("data/index_500k.meta.db")
OUTPUT_JSONL_PATH = Path("data/eval_cases_500k_20k.jsonl")
OUTPUT_JSON_PATH = Path("data/eval_cases_500k_20k.json")

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._#-]*", re.IGNORECASE)
STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "for",
    "with",
    "in",
    "of",
    "to",
    "by",
    "from",
    "is",
    "it",
    "this",
    "that",
    "on",
    "as",
    "at",
    "be",
    "are",
    "was",
    "will",
    "tool",
    "library",
    "framework",
    "system",
    "app",
    "application",
    "project",
}


def clean_query(q: str) -> str:
    """Normalize query string and remove excess whitespace."""
    q = re.sub(r"\s+", " ", q).strip()
    return q


def generate_benchmark(
    target_count: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    if not DB_PATH.exists():
        print(f"Error: {DB_PATH} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Generating >= {target_count:,} benchmark cases from {DB_PATH} (seed={seed})...")
    start_time = time.monotonic()
    rng = random.Random(seed)

    conn = sqlite3.connect(f"file:{DB_PATH.resolve().as_posix()}?mode=ro", uri=True)
    cursor = conn.cursor()

    # 1. First pass: scan database and collect candidates by capability and tier
    print("Collecting candidate pools across stars and categories...")

    # We collect candidate descriptors: (repo_id, name, lang, stars, summary, caps, use_cases, replaces, desc)
    # Stream in chunks
    cursor.execute(
        """
        SELECT repo_id, name, language, stars, entry_json
        FROM records
        WHERE disabled = 0 AND stars >= 50
        ORDER BY stars DESC
        """
    )

    t1_candidates: list[dict[str, Any]] = []
    t2_candidates: list[dict[str, Any]] = []
    t3_candidates: list[dict[str, Any]] = []
    t4_candidates: list[dict[str, Any]] = []

    seen_queries: set[str] = set()
    cases: list[dict[str, Any]] = []

    fetched = 0
    while True:
        rows = cursor.fetchmany(10000)
        if not rows:
            break
        for repo_id, name, language, stars, entry_json in rows:
            fetched += 1
            try:
                entry = json.loads(entry_json)
            except Exception:
                continue

            meta = entry.get("metadata") or {}
            desc = (meta.get("description") or "").strip()
            summary = (meta.get("summary") or "").strip()
            caps = meta.get("capabilities") or []
            use_cases = meta.get("use_cases") or []
            replaces = meta.get("replaces") or []
            aliases = meta.get("aliases") or []
            topics = meta.get("topics") or []

            item = {
                "repo_id": repo_id,
                "name": name or repo_id.split("/")[-1],
                "language": language or "",
                "stars": stars or 0,
                "desc": desc,
                "summary": summary,
                "caps": caps,
                "use_cases": use_cases,
                "replaces": replaces,
                "aliases": aliases,
                "topics": topics,
            }

            # T1 pool: good name / aliases and stars >= 100
            if stars >= 100 and len(item["name"]) >= 2:
                t1_candidates.append(item)

            # T2 pool: has capabilities or use_cases
            if (caps or use_cases or desc) and len(summary) > 20:
                t2_candidates.append(item)

            # T3 pool: has replaces or complex summary
            if replaces or (len(caps) >= 2 and len(summary) > 50):
                t3_candidates.append(item)

            # T4 pool: has explicit language and specific topics
            if language and (topics or caps):
                t4_candidates.append(item)

            # Limit memory pool sizes
            if (
                len(t1_candidates) > 60000
                and len(t2_candidates) > 60000
                and len(t3_candidates) > 40000
                and len(t4_candidates) > 40000
            ):
                break

    conn.close()
    print(
        f"Candidate pools gathered from {fetched:,} repos: T1={len(t1_candidates):,}, T2={len(t2_candidates):,}, T3={len(t3_candidates):,}, T4={len(t4_candidates):,}"
    )

    target_t1 = 6000
    target_t2 = 8000
    target_t3 = 4000
    target_t4 = 2000
    case_idx = 0

    # -------------------------------------------------------------
    # TIER 1: Easy / Exact Match (Target: 6,000)
    # -------------------------------------------------------------
    print("Generating Tier 1 (Easy / Exact Match)...")
    rng.shuffle(t1_candidates)
    for item in t1_candidates:
        if len([c for c in cases if c["difficulty"] == "easy"]) >= target_t1:
            break

        repo_id = item["repo_id"]
        name = item["name"]
        aliases = item["aliases"]

        # Decide query form
        dice = rng.random()
        if dice < 0.40:
            query = name
        elif dice < 0.60:
            query = repo_id
        elif dice < 0.75 and aliases:
            query = rng.choice(aliases)
        elif dice < 0.85:
            query = f"github {name}"
        else:
            query = f"{name} repository"

        norm_q = clean_query(query.lower())
        if not norm_q or norm_q in seen_queries or len(norm_q) < 2:
            continue
        seen_queries.add(norm_q)

        case_idx += 1
        cases.append(
            {
                "id": f"eval-500k-{case_idx:06d}",
                "query": query,
                "category": "exact_name",
                "difficulty": "easy",
                "expected_repo_id": repo_id,
                "expected_repo_ids": [repo_id],
                "acceptable_alternatives": [],
                "notes": f"Exact match for {name} ({item['stars']} stars)",
                "tags": [
                    "tier1-easy",
                    "exact_name",
                    item["language"].lower() if item["language"] else "unspecified",
                ],
            }
        )

    # -------------------------------------------------------------
    # TIER 2: Medium / Functional Queries (Target: 8,000)
    # -------------------------------------------------------------
    print("Generating Tier 2 (Medium / Functional)...")
    rng.shuffle(t2_candidates)
    for item in t2_candidates:
        if len([c for c in cases if c["difficulty"] == "medium"]) >= target_t2:
            break

        repo_id = item["repo_id"]
        lang = item["language"]
        caps = item["caps"]
        use_cases = item["use_cases"]
        desc = item["desc"]

        query_candidate = None
        dice = rng.random()
        if dice < 0.45 and caps:
            cap = rng.choice(caps)
            if len(cap.split()) >= 2 and len(cap) < 60:
                query_candidate = f"{cap} in {lang}" if lang and rng.random() < 0.5 else cap
        elif dice < 0.80 and use_cases:
            uc = rng.choice(use_cases)
            if len(uc.split()) >= 2 and len(uc) < 60:
                query_candidate = f"{uc} library" if rng.random() < 0.5 else uc
        elif desc and 10 < len(desc) < 80:
            # Clean description
            cleaned_desc = re.sub(r"^[A-Z0-9_.-]+ is an? ", "", desc, flags=re.IGNORECASE)
            cleaned_desc = re.sub(r"^[A-Z0-9_.-]+: ", "", cleaned_desc)
            if len(cleaned_desc.split()) >= 3:
                query_candidate = cleaned_desc

        if not query_candidate:
            continue

        norm_q = clean_query(query_candidate.lower())
        if not norm_q or norm_q in seen_queries or len(norm_q) < 5:
            continue
        seen_queries.add(norm_q)

        case_idx += 1
        cases.append(
            {
                "id": f"eval-500k-{case_idx:06d}",
                "query": query_candidate,
                "category": "functional",
                "difficulty": "medium",
                "expected_repo_id": repo_id,
                "expected_repo_ids": [repo_id],
                "acceptable_alternatives": [],
                "notes": f"Functional query for {repo_id}",
                "tags": ["tier2-medium", "functional", lang.lower() if lang else "general"],
            }
        )

    # -------------------------------------------------------------
    # TIER 3: Hard / Architectural & Alternative (Target: 4,000)
    # -------------------------------------------------------------
    print("Generating Tier 3 (Hard / Architectural & Alternative)...")
    rng.shuffle(t3_candidates)
    for item in t3_candidates:
        if len([c for c in cases if c["difficulty"] == "hard"]) >= target_t3:
            break

        repo_id = item["repo_id"]
        lang = item["language"]
        replaces = item["replaces"]
        caps = item["caps"]
        summary = item["summary"]

        query_candidate = None
        category = "architectural"
        alt_list: list[str] = []

        if replaces and rng.random() < 0.60:
            target = rng.choice(replaces)
            if isinstance(target, str) and len(target) >= 3:
                category = "alternative"
                query_candidate = rng.choice(
                    [
                        f"open source alternative to {target}",
                        f"self-hosted alternative to {target}",
                        f"open-source replacement for {target}",
                        f"{target} alternative in {lang}" if lang else f"alternative to {target}",
                    ]
                )
                alt_list = [str(r) for r in replaces if str(r) != target]
        elif len(caps) >= 2:
            c1, c2 = rng.sample(caps, 2)
            if len(c1) < 40 and len(c2) < 40:
                query_candidate = f"{c1} with {c2}"
                if lang and rng.random() < 0.5:
                    query_candidate += f" in {lang}"
        elif len(summary) > 40:
            # Extract deep capability sentence
            sentences = [s.strip() for s in summary.split(".") if len(s.strip().split()) >= 5]
            if sentences:
                candidate_s = rng.choice(sentences)
                if len(candidate_s) < 100:
                    query_candidate = candidate_s

        if not query_candidate:
            continue

        norm_q = clean_query(query_candidate.lower())
        if not norm_q or norm_q in seen_queries or len(norm_q) < 8:
            continue
        seen_queries.add(norm_q)

        case_idx += 1
        cases.append(
            {
                "id": f"eval-500k-{case_idx:06d}",
                "query": query_candidate,
                "category": category,
                "difficulty": "hard",
                "expected_repo_id": repo_id,
                "expected_repo_ids": [repo_id],
                "acceptable_alternatives": alt_list,
                "notes": f"Hard/complex query targeting {repo_id}",
                "tags": ["tier3-hard", category, lang.lower() if lang else "general"],
            }
        )

    # -------------------------------------------------------------
    # TIER 4: Confusable / Constrained (Target: 2,000)
    # -------------------------------------------------------------
    print("Generating Tier 4 (Confusable / Constrained)...")
    rng.shuffle(t4_candidates)
    for item in t4_candidates:
        if len([c for c in cases if c["difficulty"] == "confusable"]) >= target_t4:
            break

        repo_id = item["repo_id"]
        lang = item["language"]
        topics = item["topics"]
        caps = item["caps"]
        name = item["name"]

        if not lang:
            continue

        query_candidate = None
        if len(topics) >= 2:
            top_sample = rng.sample(topics, min(2, len(topics)))
            t_str = " ".join(top_sample).replace("-", " ")
            query_candidate = f"{lang} {t_str} tool"
        elif caps:
            cap = rng.choice(caps)
            query_candidate = f"{name} {lang} {cap}"
        elif topics:
            query_candidate = f"{topics[0]} framework written in {lang}"

        if not query_candidate:
            continue

        norm_q = clean_query(query_candidate.lower())
        if not norm_q or norm_q in seen_queries or len(norm_q) < 5:
            continue
        seen_queries.add(norm_q)

        case_idx += 1
        cases.append(
            {
                "id": f"eval-500k-{case_idx:06d}",
                "query": query_candidate,
                "category": "constrained",
                "difficulty": "confusable",
                "expected_repo_id": repo_id,
                "expected_repo_ids": [repo_id],
                "acceptable_alternatives": [],
                "notes": f"Constrained query requiring language={lang}",
                "tags": ["tier4-confusable", "constrained", lang.lower()],
            }
        )

    # Balance up to target_count if needed
    if len(cases) < target_count:
        print(
            f"Supplementing {target_count - len(cases)} cases from high-confidence pool to reach {target_count:,}..."
        )
        for item in t2_candidates:
            if len(cases) >= target_count:
                break
            repo_id = item["repo_id"]
            name = item["name"]
            desc = item["desc"]
            lang = item["language"]
            if not desc:
                continue
            q = f"{name} for {desc[:50].strip()}"
            norm_q = clean_query(q.lower())
            if not norm_q or norm_q in seen_queries:
                continue
            seen_queries.add(norm_q)
            case_idx += 1
            cases.append(
                {
                    "id": f"eval-500k-{case_idx:06d}",
                    "query": q,
                    "category": "functional",
                    "difficulty": "medium",
                    "expected_repo_id": repo_id,
                    "expected_repo_ids": [repo_id],
                    "acceptable_alternatives": [],
                    "notes": f"Supplemented query for {repo_id}",
                    "tags": ["tier2-medium", "functional", lang.lower() if lang else "general"],
                }
            )

    elapsed = time.monotonic() - start_time
    tier_counts = {
        "easy": sum(1 for c in cases if c["difficulty"] == "easy"),
        "medium": sum(1 for c in cases if c["difficulty"] == "medium"),
        "hard": sum(1 for c in cases if c["difficulty"] == "hard"),
        "confusable": sum(1 for c in cases if c["difficulty"] == "confusable"),
    }

    # Write JSONL
    OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Write JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    summary = {
        "total_cases": len(cases),
        "target_count": target_count,
        "elapsed_seconds": round(elapsed, 2),
        "tiers": tier_counts,
        "output_jsonl": str(OUTPUT_JSONL_PATH),
        "output_json": str(OUTPUT_JSON_PATH),
    }

    print(f"\nGenerated {len(cases):,} benchmark cases in {elapsed:.2f}s!")
    print(f"Tier Breakdown: {tier_counts}")
    print(
        f"Files written: {OUTPUT_JSONL_PATH} ({OUTPUT_JSONL_PATH.stat().st_size / 1024 / 1024:.2f} MB)"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 500k benchmark eval dataset.")
    parser.add_argument(
        "--count", type=int, default=20000, help="Target case count (default: 20000)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    summary = generate_benchmark(target_count=args.count, seed=args.seed)
    print("\nBenchmark Generation Summary:", json.dumps(summary, indent=2))
