"""Command-line interface for xists."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from xists.search.index import INDEX_VERSION, load_index
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


def _ingest_one(repo_id: str, token: str | None, llm_config: Any) -> dict[str, Any]:
    """Ingest a single repo. Returns a result dict with either 'record' or 'error'."""
    try:
        record = collect_record(repo_id, token=token)
        profile = generate_llm_profile(record, llm_config)
        attach_llm_profile(record, profile)
        return {"repo_id": repo_id, "record": record}
    except GitHubAPIError as error:
        return {"repo_id": repo_id, "error": {"repo_id": repo_id, "reason": str(error), "status": error.status}}
    except LLMError as error:
        return {"repo_id": repo_id, "error": {"repo_id": repo_id, "reason": str(error), "status": None}}
    except Exception as error:
        return {"repo_id": repo_id, "error": {"repo_id": repo_id, "reason": str(error), "status": None}}


def ingest_github(args: argparse.Namespace) -> int:
    try:
        llm_config = llm_config_from_env()
    except LLMNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    repo_ids = load_repo_ids(args.repos)
    token = github_token_from_file(args.token_file) if args.token_file else github_token_from_env()

    # Load existing records for incremental update (skip with --force).
    existing: list[dict[str, Any]] = []
    if not args.force and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    existing_ids = {record.get("repo_id") for record in existing}
    skipped = [repo_id for repo_id in repo_ids if repo_id in existing_ids]
    to_ingest = [repo_id for repo_id in repo_ids if repo_id not in existing_ids]

    merged: list[dict[str, Any]] = list(existing)
    failed: list[dict[str, Any]] = []
    generated = 0
    with_readme = 0
    without_readme = 0
    abstained = 0
    workers = getattr(args, "workers", 1) or 1

    def process_result(result: dict[str, Any]) -> None:
        nonlocal generated, with_readme, without_readme, abstained
        if "error" in result:
            failed.append(result["error"])
            return
        record = result["record"]
        merged.append(record)
        generated += 1
        if record.get("readme"):
            with_readme += 1
        else:
            without_readme += 1
        if (record.get("llm_profile") or {}).get("abstained"):
            abstained += 1

    if workers > 1 and to_ingest:
        # Multi-threaded: process repos concurrently, write checkpoint after all complete.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_ingest_one, repo_id, token, llm_config): repo_id
                for repo_id in to_ingest
            }
            for future in as_completed(futures):
                process_result(future.result())
                # Checkpoint after each thread completes.
                write_json(args.output, merged)
    else:
        # Single-threaded: process one by one with checkpoint after each.
        for repo_id in to_ingest:
            process_result(_ingest_one(repo_id, token, llm_config))
            write_json(args.output, merged)

    report = {
        "input_count": len(repo_ids),
        "skipped_count": len(skipped),
        "generated_count": generated,
        "failed_count": len(failed),
        "failed": failed,
        "records_with_readme": with_readme,
        "records_without_readme": without_readme,
        "records_abstained": abstained,
    }

    if args.report:
        write_json(args.report, report)

    print(json.dumps(
        {"records": str(args.output), "report": str(args.report) if args.report else None, "total_records": len(merged), **report},
        ensure_ascii=False,
        indent=2,
    ))
    return 1 if failed else 0


def _index_write_checkpoint(
    output: Path,
    *,
    index_version: int,
    embedding_model: str,
    embedding_base_url: str,
    dimension: int | None,
    record_count: int,
    skipped: list[str],
    vectors: list[dict[str, Any]],
) -> None:
    write_json(
        output,
        {
            "index_version": index_version,
            "embedding_model": embedding_model,
            "embedding_base_url": embedding_base_url,
            "dimension": dimension,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "record_count": record_count,
            "skipped": skipped,
            "vectors": vectors,
        },
    )


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
    batch_size = 64

    # Load existing index for incremental update (skip with --force).
    vectors: list[dict[str, Any]] = []
    skipped: list[str] = []
    dimension: int | None = None

    if not args.force and args.output.exists():
        existing_index = json.loads(args.output.read_text(encoding="utf-8"))
        if existing_index.get("embedding_model") and existing_index["embedding_model"] != config.model:
            print(
                f"Index was built with model '{existing_index['embedding_model']}' "
                f"but configured model is '{config.model}'. "
                f"Delete {args.output} and rebuild, or set EMBEDDING_MODEL to match.",
                file=sys.stderr,
            )
            return 1
        vectors = existing_index.get("vectors", [])
        skipped = existing_index.get("skipped", [])
        dimension = existing_index.get("dimension")
        existing_ids = {entry.get("repo_id") for entry in vectors}
        records = [r for r in records if (r.get("repo_id") or r.get("repo_id_requested")) not in existing_ids]

    # Prepare embeddable records.
    from xists.search.embed import call_embeddings, embedding_text_from_record

    embeddable: list[dict[str, Any]] = []
    for record in records:
        text = embedding_text_from_record(record)
        repo_id = record.get("repo_id") or record.get("repo_id_requested")
        if not text:
            skipped.append(repo_id or "<unknown>")
            continue
        embeddable.append({"repo_id": repo_id, "text": text})

    new_count = 0
    for start in range(0, len(embeddable), batch_size):
        batch = embeddable[start : start + batch_size]
        try:
            results = call_embeddings(config, [item["text"] for item in batch])
        except EmbeddingError as error:
            print(str(error), file=sys.stderr)
            return 1
        if len(results) != len(batch):
            print(
                f"Embedding count mismatch: sent {len(batch)}, received {len(results)}",
                file=sys.stderr,
            )
            return 1
        for item, vector in zip(batch, results):
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                print(
                    f"Inconsistent embedding dimension: {len(vector)} vs {dimension}",
                    file=sys.stderr,
                )
                return 1
            vectors.append({"repo_id": item["repo_id"], "vector": vector})
            new_count += 1

        # Checkpoint: write after each batch.
        _index_write_checkpoint(
            args.output,
            index_version=INDEX_VERSION,
            embedding_model=config.model,
            embedding_base_url=config.base_url,
            dimension=dimension,
            record_count=len(vectors),
            skipped=skipped,
            vectors=vectors,
        )

    # Write final checkpoint if no batches were processed (empty input).
    if not embeddable:
        _index_write_checkpoint(
            args.output,
            index_version=INDEX_VERSION,
            embedding_model=config.model,
            embedding_base_url=config.base_url,
            dimension=dimension,
            record_count=len(vectors),
            skipped=skipped,
            vectors=vectors,
        )

    print(
        json.dumps(
            {
                "index": str(args.output),
                "embedding_model": config.model,
                "dimension": dimension,
                "record_count": len(vectors),
                "new_vectors": new_count,
                "skipped": skipped,
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
    github.add_argument("--force", action="store_true", help="Ignore existing records.json and reprocess all repos")
    github.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (default: 1)")
    github.set_defaults(func=ingest_github)

    index = subparsers.add_parser("index", help="Build the embedding index")
    index_subparsers = index.add_subparsers(dest="index_command", required=True)
    index_build_parser = index_subparsers.add_parser("build", help="Build an embedding index from records")
    index_build_parser.add_argument("--records", type=Path, default=Path("records.json"), help="Records JSON to index")
    index_build_parser.add_argument("--output", type=Path, default=Path("index.json"), help="Path to write the embedding index")
    index_build_parser.add_argument("--force", action="store_true", help="Ignore existing index.json and rebuild from scratch")
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
