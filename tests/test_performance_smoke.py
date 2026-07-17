"""Performance guardrail: catch O(n^2)-class regressions in core search.

The threshold is deliberately loose (20 searches over a 2000-entry index in
5 seconds). It is not a benchmark — real baselines live in
docs/performance.md. Offline by design: query vectors are injected random
unit vectors, no embedding endpoint involved.
"""

import importlib.util
import time
from pathlib import Path

import numpy as np

from xists.search.embed import EmbeddingConfig
from xists.search.query import rank

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_synthetic_index.py"
SPEC = importlib.util.spec_from_file_location("generate_synthetic_index", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

COUNT = 2000
DIMENSION = 64
SEARCHES = 20


def test_twenty_searches_on_2k_index_stay_under_five_seconds():
    records = MODULE.generate_records(COUNT)
    index = MODULE.generate_index(records, dimension=DIMENSION, seed=42)
    config = EmbeddingConfig(api_key="smoke", base_url="http://smoke.invalid/v1", model=MODULE.SYNTHETIC_MODEL)
    rng = np.random.RandomState(0)
    queries = rng.standard_normal((SEARCHES, DIMENSION))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    start = time.perf_counter()
    for row in range(SEARCHES):
        vector = queries[row].tolist()
        result = rank(f"smoke query {row}", index, config, top_k=5, embed=lambda c, q, v=vector: v)
        assert result["results"]
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"20 searches took {elapsed:.2f}s on a {COUNT}x{DIMENSION} index (expected < 5s)"
