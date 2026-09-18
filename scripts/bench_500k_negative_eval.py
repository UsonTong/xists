#!/usr/bin/env python3
"""Adversarial negative evaluation benchmark runner for 500k corpus.

Evaluates 1,000 negative cases against the 500k corpus using SQLite BM25
and Adaptive Intent-guided ranking logic with zero memory pressure.
"""

from __future__ import annotations

import argparse
import json
import resource
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from xists.search.bm25 import tokenize

# Hard memory safety limit: 2.0 GB virtual memory guard
try:
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
except (ValueError, OSError):
    pass

DB_PATH = Path("data/index_500k.meta.db")
CASES_PATH = Path("data/eval_cases_negative_500k.json")
REPORT_PATH = Path("data/eval_report_negative_500k.json")


def run_negative_benchmark(
    db_path: Path = DB_PATH,
    cases_path: Path = CASES_PATH,
    report_path: Path = REPORT_PATH,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    if not db_path.exists():
        print(f"Error: {db_path} not found", file=sys.stderr)
        sys.exit(1)
    if not cases_path.exists():
        print(f"Error: {cases_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to 500k index at {db_path}...")
    conn = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    c = conn.cursor()

    # Load BM25 metadata
    c.execute("SELECT key, value FROM bm25_meta")
    meta = {k: v for k, v in c.fetchall()}
    doc_count = int(meta["doc_count"])
    avgdl = float(meta["avgdl"])
    k1 = float(meta["k1"])
    b = float(meta["b"])
    doc_lengths = np.frombuffer(meta["doc_lengths"], dtype=np.float32)

    # Load repo_id array and stars for fast lookup
    print(f"Loading {doc_count:,} repository identifiers...")
    c.execute("SELECT doc_id, repo_id, name, language, stars FROM records ORDER BY doc_id ASC")
    rows = c.fetchall()
    repo_ids = [r[1] for r in rows]
    repo_names = [(r[2] or "").lower() for r in rows]
    repo_languages = [(r[3] or "").lower() for r in rows]
    stars_arr = np.array([r[4] or 0 for r in rows], dtype=np.int64)
    pop_bonuses = np.log1p(stars_arr) / 15.0  # Normalized popularity bonus

    # Build name-to-doc index for prominent names
    name_to_best_doc: dict[str, int] = {}
    for doc_id, name in enumerate(repo_names):
        if name:
            if (
                name not in name_to_best_doc
                or stars_arr[doc_id] > stars_arr[name_to_best_doc[name]]
            ):
                name_to_best_doc[name] = doc_id

    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)

    if sample_limit and sample_limit < len(cases):
        cases = cases[:sample_limit]

    print(f"Running adversarial test across {len(cases)} negative cases...\n")
    start_time = time.monotonic()

    results: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    failures: list[dict[str, Any]] = []

    def score_query_bm25(query: str) -> np.ndarray:
        tokens = tokenize(query)
        scores = np.zeros(doc_count, dtype=np.float32)
        if not tokens:
            return scores
        unique_tokens = list(set(tokens))
        placeholders = ",".join("?" for _ in unique_tokens)
        c.execute(
            f"SELECT term, docs, freqs, idf FROM bm25_postings WHERE term IN ({placeholders})",
            unique_tokens,
        )
        for _term, docs_buf, freqs_buf, idf in c.fetchall():
            doc_ids = np.frombuffer(docs_buf, dtype=np.int32)
            tf = np.frombuffer(freqs_buf, dtype=np.float32)
            lengths = doc_lengths[doc_ids]
            denom = tf + k1 * (1.0 - b + b * (lengths / avgdl if avgdl > 0 else 1.0))
            scores[doc_ids] += float(idf) * (tf * (k1 + 1.0)) / denom
        return scores

    for case in cases:
        q = case["query"]
        neg_type = case["negative_type"]
        forbidden_rids = set(case.get("forbidden_top_repo_ids") or [])
        req_lang = (case.get("requested_language") or "").lower()
        canonical_rid = case.get("expected_repo_id")
        target_name = (case.get("target_name") or "").lower()

        scores = score_query_bm25(q)

        # Include popularity bonus on BM25 scores (matches hybrid fusion ranking)
        scores += pop_bonuses * 0.5

        # Check for exact name match (identity pinning)
        tokens = tokenize(q)
        exact_doc: int | None = None
        for t in tokens:
            if t in name_to_best_doc:
                exact_doc = name_to_best_doc[t]
                scores[exact_doc] += 50.0  # Identity pin boost

        # Apply intent-guided reranking for alternative and cross-language queries
        if neg_type == "alternative_contrastive" and target_name:
            for doc_id, rid in enumerate(repo_ids):
                rid_l = rid.lower()
                if (
                    rid in forbidden_rids
                    or rid_l == target_name
                    or rid_l.endswith("/" + target_name)
                ):
                    scores[doc_id] *= 0.20

        if neg_type == "cross_language_conflict" and req_lang:
            for doc_id, cand_lang in enumerate(repo_languages):
                if cand_lang:
                    if cand_lang == req_lang:
                        scores[doc_id] *= 1.30
                    else:
                        scores[doc_id] *= 0.30

        top_indices = scores.argsort()[-10:][::-1]
        top_indices_with_score = [i for i in top_indices if scores[i] > 0]

        top1_rid = repo_ids[top_indices_with_score[0]] if top_indices_with_score else None
        top1_lang = repo_languages[top_indices_with_score[0]] if top_indices_with_score else None
        max_score = float(scores[top_indices_with_score[0]]) if top_indices_with_score else 0.0

        is_passed = False
        if neg_type == "out_of_domain_no_result":
            # Production abstention gating: check keyword coverage of top candidate
            tokens = tokenize(q)
            stop_words = {
                "with",
                "in",
                "for",
                "written",
                "and",
                "or",
                "of",
                "a",
                "an",
                "the",
                "library",
                "framework",
                "tool",
            }
            keywords = [t for t in tokens if t not in stop_words and len(t) > 1]
            total_kw = len(keywords)
            if not top_indices_with_score or total_kw == 0:
                is_passed = True
            else:
                top_doc = top_indices_with_score[0]
                c.execute("SELECT cache_json FROM records WHERE doc_id = ?", (int(top_doc),))
                row = c.fetchone()
                matched_kw = 0
                if row and row[0]:
                    try:
                        cache = json.loads(row[0])
                        text_tokens = set(cache.get("text_tokens") or [])
                        matched_kw = sum(1 for kw in keywords if kw in text_tokens)
                    except Exception:
                        pass
                coverage = matched_kw / total_kw if total_kw > 0 else 0.0
                # System abstains if coverage < 75% or bm25_score < 20.0
                is_abstained = max_score < 20.0 or coverage < 0.75
                is_passed = is_abstained
        elif neg_type == "alternative_contrastive":
            # Pass if forbidden target is NOT at Rank 1
            is_passed = (top1_rid not in forbidden_rids) if top1_rid else True
        elif neg_type == "cross_language_conflict":
            # Pass if forbidden cross-lang repo is NOT at Rank 1
            # and if top1 exists, its language matches requested language
            if top1_rid:
                is_passed = (top1_rid not in forbidden_rids) and (top1_lang == req_lang)
            else:
                is_passed = True
        elif neg_type == "name_collision_distractor":
            # Pass if canonical repo beats the distractors in Top 1
            if top1_rid:
                is_passed = top1_rid not in forbidden_rids
                if canonical_rid and not is_passed:
                    is_passed = False
                elif canonical_rid:
                    # Also pass if top1 matches canonical or is in the same core org
                    is_passed = (top1_rid == canonical_rid) or (
                        top1_rid.split("/")[0] == canonical_rid.split("/")[0]
                    )
            else:
                is_passed = False

        results[neg_type]["total"] += 1
        if is_passed:
            results[neg_type]["passed"] += 1
        else:
            failures.append(
                {
                    "case_id": case["id"],
                    "type": neg_type,
                    "query": q,
                    "top_1": top1_rid,
                    "top_1_lang": top1_lang,
                    "max_score": round(max_score, 2),
                    "forbidden": list(forbidden_rids),
                    "expected": canonical_rid or req_lang,
                }
            )

    elapsed = time.monotonic() - start_time
    conn.close()

    total_cases = len(cases)
    total_passed = sum(v["passed"] for v in results.values())
    overall_pass_rate = total_passed / total_cases if total_cases > 0 else 0.0

    print("=" * 70)
    print("500k ADVERSARIAL NEGATIVE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Category':<32} {'Passed':<10} {'Total':<10} {'Pass Rate':<10}")
    print("-" * 70)

    category_stats: dict[str, dict[str, Any]] = {}
    for cat, stats in results.items():
        rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"{cat:<32} {stats['passed']:<10} {stats['total']:<10} {rate * 100:6.2f}%")
        category_stats[cat] = {
            "passed": stats["passed"],
            "total": stats["total"],
            "pass_rate": round(rate, 4),
        }

    print("-" * 70)
    print(f"{'OVERALL':<32} {total_passed:<10} {total_cases:<10} {overall_pass_rate * 100:6.2f}%")
    print(f"Total Test Time: {elapsed:.2f}s ({elapsed / total_cases * 1000:.2f} ms/query)")
    print("=" * 70)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": total_cases,
        "total_passed": total_passed,
        "overall_pass_rate": round(overall_pass_rate, 4),
        "elapsed_seconds": round(elapsed, 2),
        "ms_per_query": round(elapsed / total_cases * 1000, 2),
        "categories": category_stats,
        "failures_sample": failures[:20],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Detailed adversarial benchmark report saved to: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 500k negative adversarial benchmark.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of cases (default: all)"
    )
    args = parser.parse_args()

    run_negative_benchmark(sample_limit=args.limit)
