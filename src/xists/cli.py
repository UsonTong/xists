"""Command-line interface for xists."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from xists.ingest.github import GitHubAPIError, collect_record, github_token_from_env, github_token_from_file, parse_github_repo


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or "=" not in value:
            continue
        key, env_value = value.split("=", 1)
        key = key.strip()
        env_value = env_value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = env_value


def load_repo_ids(path: Path) -> list[str]:
    repo_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        repo_ids.append(parse_github_repo(value))
    return repo_ids


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ingest_github(args: argparse.Namespace) -> int:
    repo_ids = load_repo_ids(args.repos)
    token = github_token_from_file(args.token_file) if args.token_file else github_token_from_env()
    records: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for repo_id in repo_ids:
        try:
            records.append(collect_record(repo_id, token=token))
        except GitHubAPIError as error:
            failed.append(
                {
                    "repo_id": repo_id,
                    "reason": str(error),
                    "status": error.status,
                }
            )
        except Exception as error:
            failed.append(
                {
                    "repo_id": repo_id,
                    "reason": str(error),
                    "status": None,
                }
            )

    report = {
        "input_count": len(repo_ids),
        "generated_count": len(records),
        "failed_count": len(failed),
        "failed": failed,
        "records_with_readme": sum(1 for record in records if record.get("readme")),
        "records_without_readme": sum(1 for record in records if not record.get("readme")),
    }

    write_json(args.output, records)
    if args.report:
        write_json(args.report, report)

    print(json.dumps({"records": str(args.output), "report": str(args.report) if args.report else None, **report}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="xists helps developers find what already exists.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Collect repository records")
    ingest_subparsers = ingest.add_subparsers(dest="source", required=True)

    github = ingest_subparsers.add_parser("github", help="Collect records from GitHub repositories")
    github.add_argument("--repos", type=Path, default=Path("repos.txt"), help="Text file with one GitHub owner/repo or URL per line")
    github.add_argument("--output", type=Path, default=Path("records.json"), help="Path to write records JSON")
    github.add_argument("--report", type=Path, default=Path("report.json"), help="Path to write generation report JSON")
    github.add_argument("--token-file", type=Path, default=None, help="Optional file containing a GitHub token")
    github.set_defaults(func=ingest_github)

    return parser


def main() -> int:
    load_env_file(Path(".env"))
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
