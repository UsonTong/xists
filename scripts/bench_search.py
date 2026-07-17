"""Benchmark core search, index stats, and index verify on a synthetic index.

Offline by design: query vectors are seeded random unit vectors injected via
the ``embed``/``embed_many`` hooks, so no embedding endpoint is needed. Each
measurement runs three times and reports the median.

Usage:
    python scripts/bench_search.py --records /tmp/syn-records.json --index /tmp/syn-index.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from xists.search.embed import EmbeddingConfig
from xists.search.index import load_index
from xists.search.query import rank, rank_many

REPEATS = 3


def _median(fn) -> float:
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    index = load_index(args.index)
    dimension = index["dimension"]
    config = EmbeddingConfig(api_key="bench", base_url="http://bench.invalid/v1", model=index["embedding_model"])
    rng = np.random.RandomState(args.seed)
    query_vector = rng.standard_normal(dimension)
    query_vector = (query_vector / np.linalg.norm(query_vector)).tolist()

    def load_only():
        load_index(args.index)

    def rank_in_memory():
        rank("bench query", index, config, top_k=5, embed=lambda c, q: query_vector)

    def rank_many_in_memory():
        rank_many(["bench query"], index, config, top_k=5, embed_many=lambda c, qs: [query_vector])

    def cli(*cmd: str):
        subprocess.run([sys.executable, "-m", "xists.cli", *cmd], check=True, capture_output=True)

    results = {
        "index": str(args.index),
        "vector_count": len(index.get("vectors") or []),
        "dimension": dimension,
        "seconds": {
            "index_load": _median(load_only),
            "search_core_rank": _median(rank_in_memory),
            "search_core_rank_many": _median(rank_many_in_memory),
            "index_stats_cli": _median(lambda: cli("index", "stats", "--index", str(args.index))),
            "index_verify_cli": _median(
                lambda: cli("index", "verify", "--records", str(args.records), "--index", str(args.index))
            ),
        },
    }
    results["seconds"] = {key: round(value, 4) for key, value in results["seconds"].items()}
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
