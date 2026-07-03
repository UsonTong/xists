# Release Checklist

Use this checklist before tagging a xists release.

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

If the demo artifacts exist locally, run:

```bash
xists doctor \
  --records demo-records.json \
  --index demo-index.json \
  --cases examples/eval-cases.json

xists index stats --index demo-index.json --limit 5

xists eval inspect \
  --report demo-eval-report.json \
  --status serious_mismatch \
  --limit 20
```

The expected v0.1.0 baseline has no serious mismatches on the committed 100-case evaluation set when run against the current local demo index.

## 5. Commit release-prep changes

```bash
git status --short
git add <intentional files>
git commit -m "chore: prepare v0.1.0 release"
```

Skip this step if there are no pending changes.

## 6. Tag the release

```bash
git tag v0.1.0
```

To publish to the remote repository:

```bash
git push origin main
git push origin v0.1.0
```

## 7. After release

- Confirm GitHub Actions CI passed for the release commit.
- Keep generated records, indexes, reports, `.env`, and token files out of Git.
- If search quality changes after release, run the full eval loop and compare `summary_text` before tagging another version.
