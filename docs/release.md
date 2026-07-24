# Release Guide

This guide prepares a release without silently publishing anything. Creating a
tag, GitHub Release, Release asset, or PyPI upload is an external action and
requires the maintainer's explicit approval for that release.

## Release invariants

- `src/xists/__init__.py` is the single version source. Hatch reads it while
  building; `xists version` reports the same value. The release tag must be
  `v` followed by that version.
- Never stage or publish `.env`, token files, private records/indexes/eval
  reports, checkpoints, or `data/scale-*` artifacts.
- The core wheel has only the runtime dependency declared in `pyproject.toml`;
  release tooling such as `build` belongs in the development environment.
- A public demo asset must be regenerated with the release code and pass both
  `records validate` and `index verify` before it can be uploaded. Do not
  relabel an older schema or embedding-input artifact as current.

## 1. Start from a clean, tested main

```bash
git switch main
git pull --ff-only origin main
git status --short
python -m pip install -e ".[dev]"
python -m pytest tests/ -q
python scripts/smoke_check.py
```

`git status --short` must show no staged secrets or generated data. Confirm the
main CI workflow is green for the exact commit that will be released.

## 2. Set and verify the version

Change only `__version__` in `src/xists/__init__.py`, then commit the version
bump before building. For a v0.7.0 release, the required value is `0.7.0` and
the later tag is `v0.7.0`.

```bash
xists version
python -c 'from xists import __version__; print(__version__)'
git status --short
```

Do not create a tag until the remaining checks pass and publishing has been
explicitly authorized.

## 3. Build and inspect distribution artifacts

Install the build tool if it is not already available, remove only the local
build output, and build both an sdist and wheel:

```bash
python -m pip install build
rm -rf dist build
python -m build
python -m zipfile -l dist/xists-*.whl
tar -tzf dist/xists-*.tar.gz
```

Inspect the two listings. They must contain package code, public documentation,
and `LICENSE`, but not `.env`, token files, `data/scale-*`, generated records,
indexes, reports, checkpoints, test caches, or `dist/` itself.

## 4. Install the wheel in a clean environment

This verifies the built artifact rather than an editable checkout:

```bash
python -m venv /tmp/xists-release-venv
/tmp/xists-release-venv/bin/python -m pip install dist/xists-*.whl
/tmp/xists-release-venv/bin/xists --help
/tmp/xists-release-venv/bin/xists version
/tmp/xists-release-venv/bin/python scripts/smoke_check.py
```

The reported version must match the package version and intended tag. The smoke
script uses committed offline fixtures and does not require credentials or a
model endpoint.

## 5. Prepare a public demo asset, if one is being released

No current-schema demo records/index asset is committed or downloadable yet. To
create one, use a deliberately public repository list and endpoints approved
for the source material. Keep generated files outside git, then run:

```bash
xists records validate --records demo-records.json --format json
xists index verify --records demo-records.json --index demo-index.json --format json
sha256sum demo-records.json demo-index.json
```

Both validation commands must return `0`. Record the file names, SHA-256
hashes, xists version, record schema version, embedding input version,
embedding model, and index dimension in the release notes. Upload only these
validated files as GitHub Release assets after authorization; never commit them
to the repository. Users can instead follow [the demo workflow](demo.md) to
build their own local records and index.

## 6. Authorized publish order

Only after the maintainer explicitly authorizes this release:

```bash
git push origin main
git tag -a v0.7.0 -m "v0.7.0"
git push origin v0.7.0
```

Create the GitHub Release from that tag, attach only the validated public demo
assets (if any), then upload the exact `dist/` artifacts to PyPI using the
maintainer-approved credentials or publish workflow. Do not rebuild after
tagging: publish the artifacts already checked above.

## 7. Verify or recover

After PyPI becomes visible, install the exact version in another clean
environment and repeat the wheel checks with `pip install xists==0.7.0`.
Confirm the GitHub Release links the same tag and hashes documented in its
notes.

If a check fails before upload, stop, fix the release-preparation commit, and
repeat the build from scratch. If PyPI upload has already succeeded, do not
reuse or overwrite the version: publish a new patch version after documenting
the issue. A GitHub Release can be edited or marked as a draft while the
underlying tag and package version are being reconciled.
