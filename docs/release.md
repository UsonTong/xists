# Release Checklist

Use this checklist before preparing or tagging a xists release.

## 1. Confirm the working tree

```bash
git status --short
```

Only intentional source, docs, test, or fixture changes should be present. Do not release with `.env`, token files, generated private data, or local demo artifacts staged.

## 2. Install and test

```bash
python -m pip install -e ".[dev]"
pytest
```

## 3. Verify package version

```bash
xists --version
xists version
```

The CLI version, `src/xists/__init__.py`, and `pyproject.toml` should agree.

## 4. Run local workflow checks

The committed demo artifacts (`demo-records.json`, `demo-index.json`) predate
record schema v2 and embedding input v3. Against them, `records validate`
reports `ok: false`, `index verify` reports `status: invalid`, and `search`
refuses to run — this is the expected, by-design behavior of the compatibility
checks, not a release blocker.

To run the full workflow checks against a passing baseline, first refresh the
demo data (requires the local LLM and embedding endpoints):

```bash
xists profile refresh --records demo-records.json --output demo-records.json
xists index build --records demo-records.json --output demo-index.json
```

Then run:

```bash
xists doctor \
  --records demo-records.json \
  --index demo-index.json \
  --cases examples/eval-cases.json

xists eval cases --cases examples/eval-cases.json
xists eval cases --cases examples/eval-cases-extended.json
xists records validate --records demo-records.json
xists records stats --records demo-records.json
xists index stats --index demo-index.json --limit 5
xists index verify --records demo-records.json --index demo-index.json

xists eval inspect \
  --report demo-eval-report.json \
  --status serious_mismatch \
  --limit 20
```

The expected v0.4.0 local baseline passes the data-quality workflow checks and
has no serious mismatches on the committed smoke evaluation set only after the
demo artifacts have been refreshed and rebuilt as described above. Use the
100-case and extended 112-case files for broader local spot checks.

For a no-network artifact smoke test, run:

```bash
python scripts/smoke_check.py
```

## 5. Commit release-prep changes

```bash
git status --short
git add <intentional files>
git commit -m "chore: prepare v0.4.0 version metadata"
```

Skip this step if there are no pending changes.

## 6. Optional tag and publish

Skip this section when only preparing v0.4.0 locally.

```bash
git tag v0.4.0
```

To publish to the remote repository:

```bash
git push origin main
git push origin v0.4.0
```

## 7. After release

- Confirm GitHub Actions CI passed for the release commit.
- Keep generated records, indexes, reports, `.env`, and token files out of Git.
- If search quality changes after release, run the full eval loop and compare `summary_text` before tagging another version.
