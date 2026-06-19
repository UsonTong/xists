# Usage

This page describes the current local workflow for generating xists records from a user-provided GitHub repository list.

## Configure GitHub token

Create a local `.env` file from the example file:

```bash
cp .env.example .env
```

Edit `.env` and set your GitHub token:

```env
GITHUB_TOKEN=your_github_token_here
```

`.env` is ignored by Git and should not be committed.

## Create a repository list

Create `repos.txt` in the project root:

```text
facebook/react
vuejs/core
https://github.com/EbookFoundation/free-programming-books
```

Supported formats:

- `owner/repo`
- `https://github.com/owner/repo`

Blank lines and lines starting with `#` are ignored.

## Generate records

Run:

```bash
python -m xists.cli ingest github
```

This uses the default paths:

```text
repos.txt     -> input repository list
records.json  -> generated records
report.json   -> generation report
```

The same command can be written explicitly as:

```bash
python -m xists.cli ingest github \
  --repos repos.txt \
  --output records.json \
  --report report.json
```

## Custom paths

You can also provide custom paths:

```bash
python -m xists.cli ingest github \
  --repos data/repos.txt \
  --output data/records.json \
  --report data/report.json
```

## Output files

### `records.json`

A JSON array of generated xists records.

Each record includes source metadata, README excerpt, structure signals, evidence, evidence gaps, and snapshot metadata.

See [Record Schema](record-schema.md) for the full record format.

### `report.json`

A JSON report for the ingest run.

It includes:

- input repository count
- generated record count
- failed repository count
- failed repository details
- number of records with README
- number of records without README

## Notes

Current ingestion is repository-list based. xists does not yet automatically discover repository names.

Current records are generated from GitHub API data and rule-based structure analysis. LLM profiling and embedding search are future steps.
