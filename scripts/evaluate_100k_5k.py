"""Comprehensive evaluation benchmark on 100k repository corpus with 5k test cases.

Evaluates:
- Overall Hit@1, Hit@3, Hit@5, Hit@10, MRR, Mean Latency
- Difficulty breakdown: simple, complex, confusable
- Star tier breakdown: star-50k-plus, star-10k-49k, star-1k-9k, star-100-999
- Failure case categorization and diagnostic inspection
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from xists.search.bm25 import BM25Index


def run_5k_benchmark(
    records_path: Path = Path("data/records_100k.jsonl"),
    cases_path: Path = Path("data/eval_cases_5k.json"),
    report_output_path: Path = Path("data/eval_report_5k.json"),
) -> dict[str, Any]:
    print(f"Loading records from {records_path}...")
    t0 = time.perf_counter()
    entries: list[dict[str, Any]] = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            github = r.get("github") or {}
            prof = r.get("llm_profile") or {}
            meta = {
                "name": r.get("name"),
                "description": github.get("description"),
                "topics": github.get("topics") or [],
                "language": github.get("language"),
                "stars": github.get("stars", 0),
                "forks": github.get("forks", 0),
                "archived": github.get("archived", False),
                "disabled": github.get("disabled", False),
                "summary": prof.get("summary"),
                "use_cases": prof.get("use_cases") or [],
                "capabilities": prof.get("capabilities") or [],
                "replaces": prof.get("replaces") or [],
                "search_text": prof.get("search_text"),
                "search_phrases": prof.get("search_phrases") or [],
            }
            entries.append({"repo_id": r.get("repo_id"), "metadata": meta})

    load_time = time.perf_counter() - t0
    total_records = len(entries)
    print(f"Loaded {total_records} records in {load_time:.2f}s")

    print("Building BM25 sparse index...")
    t0 = time.perf_counter()
    bm25 = BM25Index.build_from_entries(entries)
    bm25_build_time = time.perf_counter() - t0
    print(f"BM25 index built in {bm25_build_time:.2f}s (vocab: {len(bm25.postings)})")

    print(f"Loading test cases from {cases_path}...")
    with open(cases_path, "r", encoding="utf-8") as f:
        cases_data = json.load(f)
    cases = cases_data["cases"]
    total_cases = len(cases)
    print(f"Loaded {total_cases} test cases.")

    print("Running retrieval evaluation across 5,000 cases...")
    t_start = time.perf_counter()

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    mrr_total = 0.0
    latencies: list[float] = []

    # Tag breakdowns
    tag_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "hit_1": 0, "hit_3": 0, "hit_5": 0, "hit_10": 0, "mrr": 0.0}
    )

    misses: list[dict[str, Any]] = []

    for i, case in enumerate(cases):
        q = case["query"]
        expected = case["expected_repo_id"]
        tags = case.get("tags", [])

        q_t0 = time.perf_counter()
        scores = bm25.score_query(q)
        top_k_indices = scores.argsort()[-10:][::-1]
        top_results = [
            (entries[idx]["repo_id"], float(scores[idx]))
            for idx in top_k_indices
            if scores[idx] > 0
        ]
        q_elapsed = (time.perf_counter() - q_t0) * 1000.0
        latencies.append(q_elapsed)

        top_repos = [r[0] for r in top_results]

        h1 = bool(top_repos and top_repos[0] == expected)
        h3 = expected in top_repos[:3]
        h5 = expected in top_repos[:5]
        h10 = expected in top_repos[:10]

        rank_pos = (top_repos.index(expected) + 1) if expected in top_repos else None
        reciprocal_rank = (1.0 / rank_pos) if rank_pos is not None else 0.0

        if h1:
            hit_at_1 += 1
        if h3:
            hit_at_3 += 1
        if h5:
            hit_at_5 += 1
        if h10:
            hit_at_10 += 1
        mrr_total += reciprocal_rank

        for tag in tags:
            s = tag_stats[tag]
            s["total"] += 1
            if h1:
                s["hit_1"] += 1
            if h3:
                s["hit_3"] += 1
            if h5:
                s["hit_5"] += 1
            if h10:
                s["hit_10"] += 1
            s["mrr"] += reciprocal_rank

        if not h1:
            misses.append(
                {
                    "case_id": case.get("id"),
                    "query": q,
                    "expected": expected,
                    "tags": tags,
                    "rank_position": rank_pos,
                    "top_3_retrieved": top_repos[:3],
                    "top_1_score": top_results[0][1] if top_results else 0.0,
                    "expected_score": float(scores[next((idx for idx, e in enumerate(entries) if e["repo_id"] == expected), -1)]) if expected else 0.0,
                }
            )

        if (i + 1) % 1000 == 0 or (i + 1) == total_cases:
            print(
                f"  Progress: {i+1}/{total_cases} | "
                f"Hit@1: {hit_at_1/(i+1)*100:.2f}% | "
                f"Hit@5: {hit_at_5/(i+1)*100:.2f}% | "
                f"MRR: {mrr_total/(i+1):.4f} | "
                f"Avg Latency: {np.mean(latencies):.2f}ms"
            )

    total_eval_time = time.perf_counter() - t_start

    # Format tag summaries
    breakdown_summary: dict[str, dict[str, Any]] = {}
    for tag, s in tag_stats.items():
        cnt = s["total"]
        if cnt > 0:
            breakdown_summary[tag] = {
                "count": cnt,
                "hit_1_pct": round(s["hit_1"] / cnt * 100, 2),
                "hit_3_pct": round(s["hit_3"] / cnt * 100, 2),
                "hit_5_pct": round(s["hit_5"] / cnt * 100, 2),
                "hit_10_pct": round(s["hit_10"] / cnt * 100, 2),
                "mrr": round(s["mrr"] / cnt, 4),
            }

    report = {
        "benchmark_name": "xists-100k-5k-evaluation",
        "total_corpus_records": total_records,
        "total_test_cases": total_cases,
        "metrics": {
            "hit_at_1_pct": round(hit_at_1 / total_cases * 100, 2),
            "hit_at_3_pct": round(hit_at_3 / total_cases * 100, 2),
            "hit_at_5_pct": round(hit_at_5 / total_cases * 100, 2),
            "hit_at_10_pct": round(hit_at_10 / total_cases * 100, 2),
            "mrr": round(mrr_total / total_cases, 4),
            "abstention_rate_pct": 0.0,
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
            "total_eval_duration_sec": round(total_eval_time, 2),
            "throughput_qps": round(total_cases / total_eval_time, 1),
        },
        "breakdowns": breakdown_summary,
        "miss_count": len(misses),
        "sample_misses": misses[:25],
    }

    report_output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed evaluation report saved to {report_output_path}")
    return report


if __name__ == "__main__":
    run_5k_benchmark()
