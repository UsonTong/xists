# Public retrieval regression fixture

This tiny, fully offline fixture protects the public search contract; it is not
a claim about production retrieval quality. It uses a fixed 4-dimensional query
embedding map, a current schema-v2 record set, and an index-v3 document.

Run it without credentials or network access:

```bash
python scripts/run_retrieval_regression.py
```

It covers exact identity, functional intent, ecosystem intent, an ambiguous
query, a Chinese query, and a deliberate no-result query. The full 2k/10k
internal corpus remains a separate scale-validation artifact and is documented
in `docs/current-retrieval-baseline.md`.
