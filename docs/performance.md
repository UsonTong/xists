# Performance baseline (v0.5.0)

Measured baselines for local brute-force search at the 1k-10k repo scale the
roadmap targets. This is a measurement record, not a benchmark suite; the
numbers exist to catch accidental complexity regressions, not to be precise.

## Environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 7735H (16 threads) |
| Memory | 14 GiB |
| OS | Linux (Fedora 44, kernel 7.0.11) |
| Python | 3.14.5 |
| numpy | 2.4.6 |
| Date | 2026-07-17 |

## Fixtures

Synthetic records and indexes generated with
`scripts/generate_synthetic_index.py` (seed 42, dimension 1024, unit random
vectors, templated schema-v2 profiles). Measurements taken with
`scripts/bench_search.py`: each item runs 3 times, median reported. Query
embedding is replaced by an injected random unit vector, so the numbers cover
"load index + similarity + ranking" only — no embedding endpoint latency.

```bash
python scripts/generate_synthetic_index.py --count 10000 --dimension 1024 \
  --output-records /tmp/syn-records-10k.json --output-index /tmp/syn-index-10k.json --seed 42
python scripts/bench_search.py --records /tmp/syn-records-10k.json --index /tmp/syn-index-10k.json
```

## Results (median of 3)

| Measurement | 1k × 1024 | 10k × 1024 |
|---|---|---|
| index load (JSON parse) | 0.14 s | 1.57 s |
| search core, `rank()` (single query, in-memory index) | 0.14 s | 1.56 s |
| search core, `rank_many()` (batched/numpy path, in-memory index) | 0.13 s | 0.86 s |
| `index stats` (full CLI subprocess) | 0.36 s | 1.61 s |
| `index verify` (full CLI subprocess) | 0.35 s | 2.06 s |

End-to-end single CLI search at 10k ≈ index load + `rank()` ≈ 3.1 s plus
embedding endpoint latency.

## Conclusions

- **1k: 近似即时 — met.** Everything is around 0.1-0.4 s.
- **10k: 仍可交互 — met with a caveat.** `index stats` / `index verify`
  complete in ~2 s. The numpy brute-force core (`rank_many`) is 0.86 s,
  within the roadmap's 1-second bar for 10k core compute. All costs scale
  linearly with corpus size; no super-linear behavior was observed.
- **Caveat (maintainer decision point):** the single-query `rank()` path —
  what `xists search` uses — takes 1.56 s at 10k, exceeding the 1-second
  core-compute bar. The cause is not O(n²) work: `rank()` scores entries with
  a pure-Python `cosine_similarity` loop (~10M scalar ops at 10k × 1024),
  while `rank_many()` uses a numpy matrix product for the same scores.
  Unifying `rank()` onto the matrix path would fix it but changes scoring
  code (float32 matrix vs float64 scalar arithmetic differs around 1e-4),
  which v0.5.0 forbids touching without a maintainer decision. Left as-is
  and flagged; see the v0.5.0 completion report.
- During measurement one sanctioned repeated-work fix was applied
  (ROADMAP §11 T3 allows fixing per-entry repeated parse/normalize work):
  `_expanded_token` in `query.py` recomputed the same token expansions for
  every entry on every search (~670k calls per 10k-entry search, dominated
  by `re.split`). Adding `lru_cache` — the same idiom already used by
  `_tokenize` and `_keyword_tokens`, with zero scoring change (163 tests
  pass unchanged) — cut the 10k core from 2.28 s → 1.56 s (`rank`) and
  1.62 s → 0.86 s (`rank_many`).

## Guardrail

`tests/test_performance_smoke.py` runs 20 searches against an in-memory
2000 × 64 synthetic index and asserts they finish in under 5 seconds. The
threshold is deliberately loose: it exists to catch O(n²)-class regressions,
not to enforce these baseline numbers.
