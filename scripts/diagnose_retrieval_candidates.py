"""Inspect generic semantic candidates and optional reranker scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xists.cli import load_env_file
from xists.search.embed import embed_query, embedding_config_from_env
from xists.search.index import decode_vector, load_index
from xists.search.rerank import rerank_documents, rerank_text_from_entry, reranker_config_from_env


def candidate_diagnostics(
    index: dict[str, Any],
    query_vector: list[float],
    *,
    candidate_limit: int = 50,
) -> list[dict[str, Any]]:
    """Return a fixed semantic candidate pool for a single-vector index."""

    if candidate_limit < 1:
        raise ValueError("candidate_limit must be at least 1")
    dimension = index.get("dimension")
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("index must declare a positive dimension")
    if len(query_vector) != dimension:
        raise ValueError(f"query vector dimension {len(query_vector)} does not match index dimension {dimension}")

    query = np.asarray(query_vector, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        raise ValueError("query vector must not be all zeros")

    candidates: list[tuple[dict[str, Any], float]] = []
    for entry in index.get("vectors") or []:
        if not isinstance(entry, dict):
            continue
        vector = decode_vector(entry.get("vector"), dimension=dimension)
        if vector is None:
            raise ValueError(f"invalid vector for {entry.get('repo_id')!r}")
        denominator = query_norm * float(np.linalg.norm(vector))
        score = 0.0 if denominator == 0.0 else float(np.dot(query, vector) / denominator)
        candidates.append((entry, score))

    candidates.sort(key=lambda item: (-item[1], str(item[0].get("repo_id") or "")))
    results: list[dict[str, Any]] = []
    for rank, (entry, score) in enumerate(candidates[:candidate_limit], start=1):
        results.append(
            {
                "repo_id": entry.get("repo_id"),
                "semantic_rank": rank,
                "semantic_score": round(score, 8),
            }
        )
    return results


def rerank_diagnostics(
    index: dict[str, Any],
    query_vector: list[float],
    rerank_query: str,
    *,
    candidate_limit: int = 50,
    rerank: Any = rerank_documents,
    reranker_config: Any = None,
) -> list[dict[str, Any]]:
    """Attach reranker scores and ranks to the unchanged semantic candidate pool."""

    candidates = candidate_diagnostics(index, query_vector, candidate_limit=candidate_limit)
    entries = {
        str(entry.get("repo_id") or ""): entry
        for entry in index.get("vectors") or []
        if isinstance(entry, dict)
    }
    documents = [rerank_text_from_entry(entries[str(item["repo_id"] or "")]) for item in candidates]
    if reranker_config is None:
        reranker_config = reranker_config_from_env()
    scores = rerank(reranker_config, rerank_query, documents)
    if len(scores) != len(candidates):
        raise ValueError("reranker returned a score count different from the candidate count")

    rerank_order = sorted(range(len(candidates)), key=lambda position: (scores[position], -position), reverse=True)
    rerank_ranks = {position: rank for rank, position in enumerate(rerank_order, start=1)}
    return [
        {
            **candidate,
            "rerank_score": round(float(score), 8),
            "rerank_rank": rerank_ranks[position],
        }
        for position, (candidate, score) in enumerate(zip(candidates, scores))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Record semantic candidate and optional reranker diagnostics")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--include-rerank", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    load_env_file(args.env_file)
    embedding_config = embedding_config_from_env()
    query_vector = embed_query(embedding_config, args.query)
    index = load_index(args.index)
    if args.include_rerank:
        candidates = rerank_diagnostics(index, query_vector, args.query, candidate_limit=args.candidate_limit)
    else:
        candidates = candidate_diagnostics(index, query_vector, candidate_limit=args.candidate_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "index": str(args.index),
                "query": args.query,
                "candidate_limit": args.candidate_limit,
                "include_rerank": args.include_rerank,
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
