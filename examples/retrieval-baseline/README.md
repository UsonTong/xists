# Frozen canonical queries

These maps freeze the English retrieval expression for every committed
cross-domain dev and holdout case. They were generated before the 0.9
configuration freeze and are inputs, not labels: expected repositories remain
only in the evaluation datasets.

Use `canonical-dev.json` only with `data/scale-cross-domain-dev.json` and
`canonical-holdout.json` only with the corresponding holdout dataset. Every case
id must appear exactly once.
