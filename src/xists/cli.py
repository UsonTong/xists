"""Command-line interface for xists."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xists.ingest.github import GitHubAPIError, collect_record, github_token_from_env, github_token_from_file, parse_github_repo
from xists.profile.llm import (
    LLMError,
    LLMNotConfiguredError,
    attach_llm_profile,
    generate_llm_profile,
    llm_config_from_env,
)
from xists.search.embed import (
    EmbeddingError,
    EmbeddingNotConfiguredError,
    embedding_config_from_env,
)
from xists.search.index import build_index, load_index
from xists.search.query import IndexMismatchError, rank


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
    try:
        llm_config = llm_config_from_env()
    except LLMNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    repo_ids = load_repo_ids(args.repos)
    token = github_token_from_file(args.token_file) if args.token_file else github_token_from_env()

    # Load existing records for incremental update.
    existing: list[dict[str, Any]] = []
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    existing_ids = {record.get("repo_id") for record in existing}
    skipped = [repo_id for repo_id in repo_ids if repo_id in existing_ids]
    to_ingest = [repo_id for repo_id in repo_ids if repo_id not in existing_ids]

    records: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for repo_id in to_ingest:
        try:
            record = collect_record(repo_id, token=token)
            profile = generate_llm_profile(record, llm_config)
            attach_llm_profile(record, profile)
            records.append(record)
        except GitHubAPIError as error:
            failed.append(
                {
                    "repo_id": repo_id,
                    "reason": str(error),
                    "status": error.status,
                }
            )
        except LLMError as error:
            failed.append(
                {
                    "repo_id": repo_id,
                    "reason": str(error),
                    "status": None,
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

    merged = existing + records

    report = {
        "input_count": len(repo_ids),
        "skipped_count": len(skipped),
        "generated_count": len(records),
        "failed_count": len(failed),
        "failed": failed,
        "records_with_readme": sum(1 for record in records if record.get("readme")),
        "records_without_readme": sum(1 for record in records if not record.get("readme")),
        "records_abstained": sum(
            1 for record in records if (record.get("llm_profile") or {}).get("abstained")
        ),
    }

    write_json(args.output, merged)
    if args.report:
        write_json(args.report, report)

    print(json.dumps(
        {"records": str(args.output), "report": str(args.report) if args.report else None, "total_records": len(merged), **report},
        ensure_ascii=False,
        indent=2,
    ))
    return 1 if failed else 0


def index_build(args: argparse.Namespace) -> int:
    try:
        config = embedding_config_from_env()
    except EmbeddingNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    if not args.records.exists():
        print(f"Records file not found: {args.records}", file=sys.stderr)
        return 2

    records = json.loads(args.records.read_text(encoding="utf-8"))

    # Load existing index for incremental update.
    existing_index: dict[str, Any] | None = None
    if args.output.exists():
        existing_index = json.loads(args.output.read_text(encoding="utf-8"))
        if existing_index.get("embedding_model") and existing_index["embedding_model"] != config.model:
            print(
                f"Index was built with model '{existing_index['embedding_model']}' "
                f"but configured model is '{config.model}'. "
                f"Delete {args.output} and rebuild, or set EMBEDDING_MODEL to match.",
                file=sys.stderr,
            )
            return 1
        existing_ids = {entry.get("repo_id") for entry in existing_index.get("vectors", [])}
        records = [r for r in records if (r.get("repo_id") or r.get("repo_id_requested")) not in existing_ids]

    try:
        new_index = build_index(records, config)
    except EmbeddingError as error:
        print(str(error), file=sys.stderr)
        return 1

    if existing_index and existing_index.get("vectors"):
        merged = {
            **new_index,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "record_count": len(existing_index["vectors"]) + new_index["record_count"],
            "skipped": (existing_index.get("skipped") or []) + new_index["skipped"],
            "vectors": existing_index["vectors"] + new_index["vectors"],
        }
    else:
        merged = new_index

    write_json(args.output, merged)
    print(
        json.dumps(
            {
                "index": str(args.output),
                "embedding_model": merged["embedding_model"],
                "dimension": merged["dimension"],
                "record_count": merged["record_count"],
                "new_vectors": new_index["record_count"],
                "skipped": merged["skipped"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def search(args: argparse.Namespace) -> int:
    try:
        config = embedding_config_from_env()
    except EmbeddingNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    if not args.index.exists():
        print(f"Index file not found: {args.index}. Run 'xists index build' first.", file=sys.stderr)
        return 2

    index = load_index(args.index)
    try:
        result = rank(args.query, index, config, top_k=args.top_k)
    except (IndexMismatchError, EmbeddingError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


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

    index = subparsers.add_parser("index", help="Build the embedding index")
    index_subparsers = index.add_subparsers(dest="index_command", required=True)
    index_build_parser = index_subparsers.add_parser("build", help="Build an embedding index from records")
    index_build_parser.add_argument("--records", type=Path, default=Path("records.json"), help="Records JSON to index")
    index_build_parser.add_argument("--output", type=Path, default=Path("index.json"), help="Path to write the embedding index")
    index_build_parser.set_defaults(func=index_build)

    search_parser = subparsers.add_parser("search", help="Search the embedding index")
    search_parser.add_argument("query", help="Natural-language query")
    search_parser.add_argument("--index", type=Path, default=Path("index.json"), help="Embedding index to search")
    search_parser.add_argument("--top-k", type=int, default=10, help="Maximum number of results to return")
    search_parser.set_defaults(func=search)

    return parser


def main() -> int:
    load_env_file(Path(".env"))
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
