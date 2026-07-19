# Scale 2k Corpus Manifest

- Selection date: 2026-07-19
- Candidate source: `repos-top10k.txt`, reconciled against `/home/usontong/Downloads/xists-records-final-9912.json`.
- Metadata source: validated stored GitHub snapshot fields (stars, archive state, push time, topics, description) and LLM profiles from the reusable 9,912-record corpus.
- Selection method: deterministic keyword-domain classification over metadata/profile. Segment candidates are sorted by descending stars then case-insensitive repo id; the resulting list is restored to source-record order.
- Candidate reconciliation: 9,912 validated records matched to the source list; 88 source candidates lack reusable records and are excluded.

## Composition

| Segment | Count | Share | Rule |
| --- | ---: | ---: | --- |
| Top-star real projects | 1,200 | 60% | Non-archived, pushed since 2024-01-01, non-tutorial/list, at least 8k stars; 240 per domain. |
| Mid/long-tail active projects | 400 | 20% | Non-archived, pushed since 2024-01-01, 1k-10k stars; 80 per domain. |
| Deliberate noise | 400 | 20% | 100 each: archived, tutorial/learning, awesome-list/resource, near-name. |
| Total | 2,000 | 100% | Unique `owner/repo` identifiers. |

## Domain Distribution

| Segment | AI/LLM | Web | Devtools | Infra | Data |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-star real projects | 240 | 240 | 240 | 240 | 240 |
| Mid/long-tail active projects | 80 | 80 | 80 | 80 | 80 |

## Noise Distribution

| Noise type | Count |
| --- | ---: |
| Archived | 100 |
| Tutorial/learning | 100 |
| Awesome-list/resource | 100 |
| Near-name | 100 |

## Known Limitations

- The reusable corpus reports zero fork records, so this selection cannot include verified forks. This is a source-data limitation.
- Domain labels are deterministic metadata/profile keyword classifications. Multi-domain repositories receive one primary label for quota accounting.
- The stored GitHub state was collected 2026-07-17 through 2026-07-18 and may have changed.
- `xists/no-such-*` evaluation targets are intentional no-result sentinels and must not appear in the corpus.
