"""Adversarial negative evaluation benchmark runner.

Tests:
1. Pure BM25 (sparse only)
2. Adaptive Intent Hybrid Fusion (with alternative reranking and language penalties)

Metrics:
- Alternative Contrastive: Pass rate (forbidden original target is NOT in Top 1)
- Cross-Language Conflict: Pass rate (forbidden cross-language target is NOT in Top 1 AND top 1 matches requested language)
- Exact Name Distractor Defense: Pass rate (canonical project beats plugin distractors in Top 1)
- Out of Domain Abstention: Pass rate (system abstains or marks low confidence instead of hallucinating)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from xists.search.bm25 import BM25Index
from xists.search.embed import EmbeddingConfig
from xists.search.local_embed import compute_deterministic_text_embedding
from xists.search.query import prepare_index, rank


def run_negative_benchmark(
    records_path: Path = Path("data/records_100k.jsonl"),
    negative_cases_path: Path = Path("data/eval_cases_negative_1k.json"),
    report_path: Path = Path("data/eval_report_negative.json"),
    sample_size: int = 30000,
) -> dict[str, Any]:
    print(f"Loading up to {sample_size} records from {records_path}...")
    entries: list[dict[str, Any]] = []
    vectors: list[list[float]] = []

    t0 = time.perf_counter()
    with open(records_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            r = json.loads(line)
            github = r.get("github") or {}
            prof = r.get("llm_profile") or {}
            rid = r.get("repo_id")
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
            entries.append({"repo_id": rid, "metadata": meta})
            # Lightweight semantic feature vector for dense channel
            desc_text = f"{rid} {meta['description'] or ''} {prof.get('summary') or ''}"
            vectors.append(compute_deterministic_text_embedding(desc_text, dimension=512))

    print(f"Loaded {len(entries)} records in {time.perf_counter() - t0:.2f}s")

    print("Building BM25 index...")
    t0 = time.perf_counter()
    bm25 = BM25Index.build_from_entries(entries)
    print(f"BM25 index built in {time.perf_counter() - t0:.2f}s")

    print("Preparing index for hybrid search engine...")
    t0 = time.perf_counter()
    index_dict = {
        "index_version": 4,
        "record_schema_version": 2,
        "embedding_input_version": 3,
        "embedding_model": "deterministic-512",
        "dimension": 512,
        "record_count": len(entries),
        "vectors": [
            {"repo_id": e["repo_id"], "vector": vectors[i], "metadata": e["metadata"]}
            for i, e in enumerate(entries)
        ],
    }
    config = EmbeddingConfig(api_key="bench", base_url="http://bench", model="deterministic-512")
    prep = prepare_index(index_dict)
    print(f"PreparedIndex built in {time.perf_counter() - t0:.2f}s")

    repo_id_to_idx = {e["repo_id"]: idx for idx, e in enumerate(entries)}
    repo_ids = [e["repo_id"] for e in entries]

    print(f"Loading negative cases from {negative_cases_path}...")
    cases_data = json.loads(negative_cases_path.read_text(encoding="utf-8"))
    cases = cases_data["cases"]
    print(f"Running adversarial test across {len(cases)} negative cases...\n")

    # Metrics collectors
    bm25_results = defaultdict(lambda: {"total": 0, "passed": 0})
    adaptive_results = defaultdict(lambda: {"total": 0, "passed": 0})

    bm25_fails: list[dict[str, Any]] = []
    adaptive_fails: list[dict[str, Any]] = []

    for case in cases:
        q = case["query"]
        neg_type = case["negative_type"]
        forbidden_rids = set(case.get("forbidden_top_repo_ids") or [])
        req_lang = case.get("requested_language")
        canonical_rid = case.get("expected_canonical_repo_id")

        # -------------------------------------------------------------
        # 1. Evaluate Pure BM25
        # -------------------------------------------------------------
        bm25_scores = bm25.score_query(q)
        bm25_top_idx = bm25_scores.argsort()[-5:][::-1]
        bm25_top_repos = [repo_ids[i] for i in bm25_top_idx if bm25_scores[i] > 0]
        bm25_top1 = bm25_top_repos[0] if bm25_top_repos else None

        bm25_pass = False
        if neg_type == "out_of_domain_no_result":
            # Pass if max score is extremely low or empty
            max_score = float(np.max(bm25_scores)) if len(bm25_scores) > 0 else 0.0
            bm25_pass = len(bm25_top_repos) == 0 or max_score < 5.0
        elif neg_type == "alternative_contrastive":
            # Pass if forbidden original target is NOT at rank 1
            bm25_pass = (bm25_top1 not in forbidden_rids) if bm25_top1 else True
        elif neg_type == "cross_language_conflict":
            # Pass if forbidden cross-lang repo is NOT at rank 1 AND top1 language matches
            if bm25_top1:
                top_meta = entries[repo_id_to_idx[bm25_top1]]["metadata"]
                top_lang = (top_meta.get("language") or "").lower()
                bm25_pass = (bm25_top1 not in forbidden_rids) and (top_lang == req_lang)
            else:
                bm25_pass = False
        elif neg_type == "name_collision_distractor":
            # Pass if canonical project is ranked #1 (beats distractors)
            bm25_pass = bm25_top1 == canonical_rid

        bm25_results[neg_type]["total"] += 1
        if bm25_pass:
            bm25_results[neg_type]["passed"] += 1
        else:
            bm25_fails.append(
                {
                    "case_id": case["id"],
                    "type": neg_type,
                    "query": q,
                    "top_1": bm25_top1,
                    "forbidden": list(forbidden_rids),
                    "expected": canonical_rid or req_lang,
                }
            )

        # -------------------------------------------------------------
        # 2. Evaluate Adaptive Intent Hybrid Fusion (Production Engine)
        # -------------------------------------------------------------
        res = rank(
            q,
            prep,
            config,
            ranking_strategy="hybrid",
            top_k=5,
            embed=lambda c, query_text: compute_deterministic_text_embedding(
                query_text, dimension=512
            ),
        )
        is_abstained = res["abstained"] or len(res["results"]) == 0
        adapt_top1 = res["results"][0]["repo_id"] if res["results"] else None

        adapt_pass = False
        if neg_type == "out_of_domain_no_result":
            adapt_pass = is_abstained
        elif neg_type == "alternative_contrastive":
            adapt_pass = (adapt_top1 not in forbidden_rids) if adapt_top1 else True
        elif neg_type == "cross_language_conflict":
            if adapt_top1:
                top_meta = entries[repo_id_to_idx[adapt_top1]]["metadata"]
                top_lang = (top_meta.get("language") or "").lower()
                adapt_pass = (adapt_top1 not in forbidden_rids) and (top_lang == req_lang)
            else:
                adapt_pass = False
        elif neg_type == "name_collision_distractor":
            adapt_pass = adapt_top1 == canonical_rid

        adaptive_results[neg_type]["total"] += 1
        if adapt_pass:
            adaptive_results[neg_type]["passed"] += 1
        else:
            adaptive_fails.append(
                {
                    "case_id": case["id"],
                    "type": neg_type,
                    "query": q,
                    "top_1": adapt_top1,
                    "forbidden": list(forbidden_rids),
                    "expected": canonical_rid or req_lang,
                }
            )

    # Summary table
    print("=" * 70)
    print(
        f"{'Negative Test Category':<30} | {'Pure BM25 Pass':<15} | {'Adaptive (v0.20) Pass':<20}"
    )
    print("-" * 70)

    total_bm25_pass, total_bm25_all = 0, 0
    total_adapt_pass, total_adapt_all = 0, 0

    categories = [
        "alternative_contrastive",
        "cross_language_conflict",
        "out_of_domain_no_result",
        "name_collision_distractor",
    ]

    report_summary: dict[str, Any] = {}

    for cat in categories:
        b_res = bm25_results[cat]
        a_res = adaptive_results[cat]
        b_pct = (b_res["passed"] / b_res["total"] * 100) if b_res["total"] else 0.0
        a_pct = (a_res["passed"] / a_res["total"] * 100) if a_res["total"] else 0.0

        total_bm25_pass += b_res["passed"]
        total_bm25_all += b_res["total"]
        total_adapt_pass += a_res["passed"]
        total_adapt_all += a_res["total"]

        print(
            f"{cat:<30} | "
            f"{b_res['passed']:>3}/{b_res['total']:<3} ({b_pct:>5.1f}%) | "
            f"{a_res['passed']:>3}/{a_res['total']:<3} ({a_pct:>5.1f}%)"
        )
        report_summary[cat] = {
            "total": b_res["total"],
            "bm25_pass_pct": round(b_pct, 2),
            "adaptive_pass_pct": round(a_pct, 2),
            "delta_pct": round(a_pct - b_pct, 2),
        }

    overall_b_pct = (total_bm25_pass / total_bm25_all * 100) if total_bm25_all else 0.0
    overall_a_pct = (total_adapt_pass / total_adapt_all * 100) if total_adapt_all else 0.0
    print("=" * 70)
    print(
        f"{'OVERALL DEFENSE PASS RATE':<30} | "
        f"{total_bm25_pass:>3}/{total_bm25_all:<3} ({overall_b_pct:>5.1f}%) | "
        f"{total_adapt_pass:>3}/{total_adapt_all:<3} ({overall_a_pct:>5.1f}%)"
    )
    print("=" * 70)

    report = {
        "benchmark": "adversarial_negative_eval_bm25_vs_adaptive",
        "total_cases": total_bm25_all,
        "overall_bm25_pass_pct": round(overall_b_pct, 2),
        "overall_adaptive_pass_pct": round(overall_a_pct, 2),
        "delta_pct": round(overall_a_pct - overall_b_pct, 2),
        "categories": report_summary,
        "bm25_sample_failures": bm25_fails[:10],
        "adaptive_sample_failures": adaptive_fails[:10],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved negative evaluation report to {report_path}")
    return report


if __name__ == "__main__":
    run_negative_benchmark()
