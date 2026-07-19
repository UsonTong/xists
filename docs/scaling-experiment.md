# Scaling Experiment Protocol

## Goal and guardrails

This protocol validates that retrieval quality remains stable while the corpus
grows from 2,000 to 10,000 to 20,000 repositories. It is an operational
experiment, not a code acceptance test.

The non-negotiable rule from the roadmap is: `禁止为 eval 分数往 query.py 加规则`.
Investigate regressions through corpus quality, profiles, evaluation cases, and
measured behavior. Do not change search ranking rules to improve an evaluation
score.

## Corpus recipe

Build a separately reviewable repository list for each level. Resolve owner and
repository names from GitHub search or curated organization lists, remove
duplicates, and retain the source and selection date beside each list.

| Segment | Share | Selection method |
| --- | ---: | --- |
| Top-star real projects | 60% | Select well-maintained, non-fork projects from GitHub's top-star results. Split this portion across AI/LLM, web, devtools, infrastructure, and data; include several projects from every domain at every scale. |
| Mid and long tail | 20% | Select active projects with 1k-10k stars from the same five domains, with languages and organizations different from the top-star segment. |
| Deliberate noise | 20% | Include tutorials, awesome-lists, archived repositories, forks of known projects, and small repositories with names similar to popular projects. Label the noise type in the source manifest. |

For 2k, 10k, and 20k respectively, produce `data/scale-2k-repos.txt`,
`data/scale-10k-repos.txt`, and `data/scale-20k-repos.txt`. Each line is one
`owner/repo`. Keep a companion manifest that records the counts in every row of
the table and the domain distribution for the 60% segment.

## Evaluation set

Maintain one versioned case file per corpus level, such as
`data/scale-2k-eval-cases.json`. It must use all of these query categories:

| Category | Required coverage |
| --- | --- |
| exact name | Direct project or organization/project lookup, including at least one Chinese query. |
| functional | Capability-oriented lookup, including at least one Chinese query. |
| ecosystem | Language, framework, or platform lookup, including at least one Chinese query. |
| ambiguous | Broad alternatives or near-neighbor lookup, including at least one Chinese query. |
| no-result | Query that should abstain or have no valid target, including at least one Chinese query. |

Declare legitimate alternatives with the `acceptable` repo-id array. The report's
`recall_at_1` and `recall_at_5` then count either `expected_repo_id` or a
dataset-declared acceptable repository. Do not revise expected answers merely
to improve a score; record any justified dataset correction in the experiment
log.

## Run 2k

Run commands in this order. Preserve `scale-2k-ingest-report.json` and
`scale-2k-refresh-report.json` when failures occur so the retry path and the
failure count remain auditable.

```bash
xists ingest github --repos data/scale-2k-repos.txt --output data/scale-2k-records.json --github-api graphql --github-batch-size 25 --workers 4 --max-rate-limit-wait 3600 --dry-run
xists ingest github --repos data/scale-2k-repos.txt --output data/scale-2k-records.json --report data/scale-2k-ingest-report.json --github-api graphql --github-batch-size 25 --workers 4 --max-rate-limit-wait 3600
xists records validate --records data/scale-2k-records.json
xists records stats --records data/scale-2k-records.json
xists profile refresh --records data/scale-2k-records.json --output data/scale-2k-profiled-records.json --report data/scale-2k-refresh-report.json
xists index build --records data/scale-2k-profiled-records.json --output data/scale-2k-index.json
xists index verify --records data/scale-2k-profiled-records.json --index data/scale-2k-index.json
xists eval run --cases data/scale-2k-eval-cases.json --index data/scale-2k-index.json --output data/scale-2k-eval-report.json --top-k 10
xists eval inspect --report data/scale-2k-eval-report.json --limit 50
```

If ingest or refresh has individual failures, use the corresponding report with
`--retry-failed`; use `--resume` for an interrupted profile refresh:

```bash
xists ingest github --repos data/scale-2k-repos.txt --output data/scale-2k-records.json --report data/scale-2k-ingest-retry-report.json --retry-failed data/scale-2k-ingest-report.json --github-api graphql --github-batch-size 25 --workers 4 --max-rate-limit-wait 3600
xists profile refresh --records data/scale-2k-records.json --output data/scale-2k-profiled-records.json --report data/scale-2k-refresh-retry-report.json --retry-failed data/scale-2k-refresh-report.json
xists profile refresh --records data/scale-2k-records.json --output data/scale-2k-profiled-records.json --resume
```

## Run 10k

Start only after the 2k record is complete and meets the promotion criteria.

```bash
xists ingest github --repos data/scale-10k-repos.txt --output data/scale-10k-records.json --github-api graphql --github-batch-size 25 --workers 4 --max-rate-limit-wait 3600 --dry-run
xists ingest github --repos data/scale-10k-repos.txt --output data/scale-10k-records.json --report data/scale-10k-ingest-report.json --github-api graphql --github-batch-size 25 --workers 4 --max-rate-limit-wait 3600
xists records validate --records data/scale-10k-records.json
xists records stats --records data/scale-10k-records.json
xists profile refresh --records data/scale-10k-records.json --output data/scale-10k-profiled-records.json --report data/scale-10k-refresh-report.json
xists index build --records data/scale-10k-profiled-records.json --output data/scale-10k-index.json
xists index verify --records data/scale-10k-profiled-records.json --index data/scale-10k-index.json
xists eval run --cases data/scale-10k-eval-cases.json --index data/scale-10k-index.json --output data/scale-10k-eval-report.json --top-k 10
xists eval inspect --report data/scale-10k-eval-report.json --limit 50
```

Retry individual failures with the equivalent 2k commands after replacing
`scale-2k` with `scale-10k`; resume interrupted profile refreshes with
`--resume`.

## Run 20k

Start only after the 10k record is complete and meets the promotion criteria.

```bash
xists ingest github --repos data/scale-20k-repos.txt --output data/scale-20k-records.json --github-api graphql --github-batch-size 25 --workers 4 --max-rate-limit-wait 3600 --dry-run
xists ingest github --repos data/scale-20k-repos.txt --output data/scale-20k-records.json --report data/scale-20k-ingest-report.json --github-api graphql --github-batch-size 25 --workers 4 --max-rate-limit-wait 3600
xists records validate --records data/scale-20k-records.json
xists records stats --records data/scale-20k-records.json
xists profile refresh --records data/scale-20k-records.json --output data/scale-20k-profiled-records.json --report data/scale-20k-refresh-report.json
xists index build --records data/scale-20k-profiled-records.json --output data/scale-20k-index.json
xists index verify --records data/scale-20k-profiled-records.json --index data/scale-20k-index.json
xists eval run --cases data/scale-20k-eval-cases.json --index data/scale-20k-index.json --output data/scale-20k-eval-report.json --top-k 10
xists eval inspect --report data/scale-20k-eval-report.json --limit 50
```

Retry individual failures with the equivalent 2k commands after replacing
`scale-2k` with `scale-20k`; resume interrupted profile refreshes with
`--resume`.

## Record template

Record the following after every level in the experiment log:

```text
Scale: 2k | 10k | 20k
Corpus composition: top-star __/__%, mid-long-tail __/__%, noise __/__%
Domain counts: AI/LLM __, web __, devtools __, infra __, data __
Selection source and date:
Elapsed time: ingest __, profile refresh __, index build __, eval __
Failures: ingest __/__, profile refresh __/__ (failure report paths: __)
Metrics: recall@1 __, recall@5 __, exact_hit_at_1 __, abstain_rate __
Issues:
- identity conflict:
- confidence high:
- semantic crowding:
- data quality:
Attribution and action:
Promotion decision:
```

Use `eval inspect` to link every issue to a query id and category. “confidence
high” means a high-confidence result outside the acceptable set, not simply a
low-scoring result.

## Promotion criteria

Move from 2k to 10k, and from 10k to 20k, only when all three conditions hold:

1. The previous level has no systematic evaluation regression, especially in a
   single query category or domain.
2. The combined ingest and profile-refresh failure rate is < 2%.
3. Each discovered issue has an attribution. Data problems become records or
   profile repair work; they must never be attributed to a need to add rules to
   `query.py`.

This experiment requires real GitHub tokens, an LLM endpoint, an embedding
endpoint, and potentially many hours of runtime. Those external resources and
the 20k live execution are the maintainer's scheduled operational work, not
part of this version's code acceptance.
