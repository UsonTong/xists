# Scale 10k Diagnosis

Run date: 2026-07-20

## Scope

This run used all candidates in `data/scale-10k-repos.txt`. It is an
operational full-candidate experiment, not the stratified 60/20/20 corpus
specified by the scaling protocol. It therefore cannot promote the project to
the 20k stage.

## Data and Index

- Ingest: 10,000 records; the 88 records not present in the reusable snapshot
  were collected successfully.
- Profile refresh: 49 missing summaries were selected. One empty upstream LLM
  response was retried successfully, so the final LLM call failure count is 0.
- Records validation: passed with warnings. 67 profiles abstained because
  repository evidence was insufficient; 45 of those have no summary and 20
  have no search text. Their raw GitHub metadata remains available to embedding.
- Index verification: passed with 9,995 vectors. Five records had no
  embeddable collected text and were skipped.

## Retrieval Results

| Metric | Result |
| --- | ---: |
| Recall@1 | 42.9% (6/14) |
| Recall@5 | 78.6% (11/14) |
| Exact top-1 | 35.7% (5/14) |
| Wrong high-confidence top-1 | 7 cases |
| No-result abstain rate | 0.0% |

## Attribution

- Semantic crowding: broad AI, database, and frontend queries ranked plausible
  but undeclared alternatives above target projects.
- Identity interpretation: the Chinese FastAPI query was interpreted as an
  exact name and matched `dingo/api` ahead of `fastapi/fastapi`.
- Abstention: both intentional no-result cases returned results, including a
  high-confidence payroll match for the English case.
- Data quality: abstained profiles are low-evidence records rather than failed
  LLM calls; they remain visible as validation warnings.

No retrieval ranking rules were changed for this evaluation. The appropriate
next step is a separately sourced stratified 10k corpus and further measured
investigation of query interpretation, semantic crowding, and no-result
abstention before scheduling 20k.
