# Scale 10k Corpus Manifest

- Selection date: 2026-07-20
- Candidate source: `repos-top10k.txt`, reconciled against `/home/usontong/Downloads/xists-records-final-9912.json`.
- Metadata source: 9,912 reused validated records plus 88 GitHub GraphQL ingests recorded in `data/scale-10k-ingest-report.json`.
- Selection method: all 10,000 candidates from the source list, preserving source order.

## Important Limitation

This is a full-candidate scale and retrieval experiment, not the roadmap's
60/20/20 stratified corpus. The available source pool itself contains 10,000
candidates, so it cannot independently produce a 10,000-record stratified
sample. Its results must not be used as the promotion evidence for the 20k
run. A separately sourced, stratified corpus is required for that decision.

## Run Artifacts

- Ingest report: `data/scale-10k-ingest-report.json` (88 generated, 0 failed).
- Profile refresh reports: `data/scale-10k-refresh-report.json` and
  `data/scale-10k-refresh-retry-report.json` (49 refreshed after retry, 0
  final call failures).
- Profiled records: `data/scale-10k-profiled-records.retry.json`.
- Index: `data/scale-10k-index.json` (9,995 vectors; 5 records lacked any
  embeddable collected text).
- Evaluation cases: `data/scale-10k-eval-cases.json`.
- Evaluation report: `data/scale-10k-eval-report.json`.
