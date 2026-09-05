"""CLI package and entry points for xists."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

from xists import __version__
from xists.api import (
    compare_projects as public_compare_projects,
)
from xists.api import (
    find_similar as public_find_similar,
)
from xists.api import (
    search as public_search,
)
from xists.cli.common import (
    _check_payload,
    _counter_items,
    _failed_repo_ids_from_report,
    _failure_entry,
    _format_command_summary,
    _format_dry_run_text,
    _format_top_items,
    _load_canonical_queries,
    _prepare_query_transforms,
    _print_embedding_error,
    _read_index_file,
    _read_records_file,
    _safe_divide,
    load_env_file,
    load_repo_ids,
    load_workspace_environment,
    write_json,
    write_json_atomic,
)
from xists.cli.compare import (
    _format_compare_text,
    compare,
)
from xists.cli.doctor import (
    _format_doctor_text,
    doctor,
)
from xists.cli.eval import (
    eval_cases,
    eval_inspect,
    eval_run,
)
from xists.cli.index import (
    _compute_checkpoint_checksum,
    _format_index_stats_text,
    _format_index_verify_text,
    _index_checkpoint_path,
    _index_stats_report,
    _index_verify_report,
    _index_write_checkpoint,
    _load_checkpoint_resilient,
    index_append,
    index_build,
    index_merge,
    index_migrate,
    index_prune,
    index_pull,
    index_stats,
    index_verify,
)
from xists.cli.ingest import (
    _append_ingest_checkpoint,
    _chunks,
    _collect_with_fallback,
    _collect_with_rate_limit,
    _ingest_checkpoint_path,
    _ingest_graphql_batch,
    _ingest_one,
    _load_ingest_checkpoint,
    _print_ingest_progress,
    _summarize_error,
    ingest_github,
)
from xists.cli.profile import (
    _append_profile_refresh_checkpoint,
    _load_profile_refresh_checkpoint,
    _profile_refresh_checkpoint_path,
    _profile_refresh_report_payload,
    profile_refresh,
)
from xists.cli.records import (
    _format_records_stats_text,
    _format_records_validation_text,
    _records_next_steps,
    _records_stats_report,
    records_inspect,
    records_stats,
    records_validate,
)
from xists.cli.search import (
    _append_search_detail,
    _format_search_text,
    _index_metadata_by_repo_id,
    _index_summaries_by_repo_id,
    _search_confidence_text,
    search,
)
from xists.cli.similar import (
    _format_project_badge,
    _format_similar_text,
    similar,
)
from xists.cli.workspace import (
    mcp,
    version,
    workspace_init,
)
from xists.eval.inspect import inspect_report, load_report
from xists.eval.run import evaluate_dataset
from xists.eval.schema import load_dataset
from xists.ingest.github import (
    collect_record,
    collect_record_graphql,
    collect_records_graphql,
    github_token_from_env,
    github_token_from_file,
)
from xists.profile.llm import (
    generate_llm_profile,
    llm_config_from_env,
)
from xists.search.embed import (
    call_embeddings,
    embedding_config_from_env,
    probe_embedding_endpoint,
)
from xists.search.query import (
    CONFIDENCE_CALIBRATION_MODES,
    RANKING_STRATEGIES,
)
from xists.search.transform import QUERY_TRANSFORM_MODES
from xists.workspace import resolve_workspace


def _prioritize_root_command_help(parser: argparse.ArgumentParser, subparsers: Any) -> None:
    """Order root command help by the user-facing search workflow."""

    preferred_order = (
        "init",
        "search",
        "similar",
        "compare",
        "index",
        "ingest",
        "profile",
        "doctor",
        "records",
        "eval",
        "version",
    )
    actions = {action.dest: action for action in subparsers._choices_actions}
    subparsers._choices_actions[:] = [
        *(actions[name] for name in preferred_order if name in actions),
        *(action for action in subparsers._choices_actions if action.dest not in preferred_order),
    ]
    commands_group = subparsers.container
    parser._action_groups.remove(commands_group)
    parser._action_groups.insert(1, commands_group)
    parser._optionals.title = "Options"


class _XistsHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep root command help focused on the command names themselves."""

    def _format_action(self, action: argparse.Action) -> str:
        if isinstance(action, argparse._SubParsersAction) and not action.help:
            return self._join_parts(
                self._format_action(subaction)
                for subaction in self._iter_indented_subactions(action)
            )
        return super()._format_action(action)


def build_parser() -> argparse.ArgumentParser:
    workspace = resolve_workspace()
    parser_options: dict[str, Any] = {
        "prog": "xists",
        "usage": "%(prog)s search <QUERY> [OPTIONS]\n       %(prog)s <COMMAND> [ARGS]",
        "description": (
            'Start here:\n  xists init\n  xists search "self-hosted photo gallery"\n  xists doctor'
        ),
        "formatter_class": _XistsHelpFormatter,
    }
    if "color" in inspect.signature(argparse.ArgumentParser).parameters:
        parser_options["color"] = False
    parser = argparse.ArgumentParser(**parser_options)
    parser.add_argument("--version", action="version", version=f"xists {__version__}")
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print the xists version")
    version_parser.set_defaults(func=version)

    mcp_parser = subparsers.add_parser("mcp", help="Start the optional MCP server over stdio")
    mcp_parser.add_argument(
        "--index", type=Path, default=workspace.index, help="Embedding index to serve"
    )
    mcp_parser.set_defaults(func=mcp)

    init_parser = subparsers.add_parser("init", help="Create the default local workspace")
    init_parser.add_argument(
        "--demo",
        action="store_true",
        help="Populate workspace with bundled starter demo records and index",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files in workspace"
    )
    init_parser.set_defaults(func=workspace_init)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check local configuration and expected data files"
    )
    doctor_parser.add_argument(
        "--records", type=Path, default=workspace.records, help="Records JSON to check"
    )
    doctor_parser.add_argument(
        "--index", type=Path, default=workspace.index, help="Embedding index to check"
    )
    doctor_parser.add_argument(
        "--cases", type=Path, default=workspace.eval_cases, help="Evaluation cases JSON to check"
    )
    doctor_parser.add_argument(
        "--token-file", type=Path, default=None, help="Optional file containing GitHub tokens"
    )
    doctor_parser.add_argument(
        "--check-endpoints",
        action="store_true",
        help="Probe the configured embedding endpoint with a small real request",
    )
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when endpoint probes fail; implies --check-endpoints",
    )
    doctor_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    doctor_parser.set_defaults(func=doctor, workspace=workspace)

    ingest_p = subparsers.add_parser("ingest", help="Collect repository records")
    ingest_subparsers = ingest_p.add_subparsers(dest="source", required=True)

    github = ingest_subparsers.add_parser("github", help="Collect records from GitHub repositories")
    github.add_argument(
        "--repos",
        type=Path,
        default=workspace.repos,
        help="Text file with one GitHub owner/repo or URL per line",
    )
    github.add_argument(
        "--output", type=Path, default=workspace.records, help="Path to write records JSON"
    )
    github.add_argument(
        "--report",
        type=Path,
        default=workspace.ingest_report,
        help="Path to write generation report JSON",
    )
    github.add_argument(
        "--token-file", type=Path, default=None, help="Optional file containing a GitHub token"
    )
    github.add_argument(
        "--force", action="store_true", help="Ignore existing records.json and reprocess all repos"
    )
    github.add_argument(
        "--resume", action="store_true", help="Resume from an existing partial JSONL checkpoint"
    )
    github.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate ingest work without calling GitHub or writing files",
    )
    github.add_argument(
        "--workers", type=int, default=1, help="Number of concurrent workers (default: 1)"
    )
    github.add_argument(
        "--github-api",
        choices=("rest", "graphql"),
        default="rest",
        help="GitHub API backend: rest (default) or graphql (lower quota usage)",
    )
    github.add_argument(
        "--github-batch-size",
        type=int,
        default=1,
        help="Repos per GraphQL request when --github-api=graphql (default: 1, recommended: 25-50)",
    )
    github.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    github.add_argument(
        "--retry-failed",
        type=Path,
        default=None,
        help="Only process repos listed in a failure report JSON",
    )
    github.add_argument(
        "--max-rate-limit-wait",
        type=float,
        default=3600,
        help="Maximum seconds to wait for exhausted GitHub rate limits (default: 3600)",
    )
    github.set_defaults(func=ingest_github)

    index_p = subparsers.add_parser("index", help="Build the embedding index")
    index_subparsers = index_p.add_subparsers(dest="index_command", required=True)
    index_build_parser = index_subparsers.add_parser(
        "build", help="Build an embedding index from records"
    )
    index_build_parser.add_argument(
        "--records", type=Path, default=workspace.records, help="Records JSON to index"
    )
    index_build_parser.add_argument(
        "--output", type=Path, default=workspace.index, help="Path to write the embedding index"
    )
    index_build_parser.add_argument(
        "--force", action="store_true", help="Ignore existing index.json and rebuild from scratch"
    )
    index_build_parser.add_argument(
        "--resume", action="store_true", help="Resume from an existing partial index checkpoint"
    )
    index_build_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding requests (default: 64)",
    )
    index_build_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent embedding requests (default: 1)",
    )
    index_build_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    index_build_parser.set_defaults(func=index_build)
    index_stats_parser = index_subparsers.add_parser(
        "stats", help="Summarize an embedding index without printing vectors"
    )
    index_stats_parser.add_argument(
        "--index", type=Path, default=workspace.index, help="Embedding index to inspect"
    )
    index_stats_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum languages/topics to print"
    )
    index_stats_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    index_stats_parser.set_defaults(func=index_stats)
    index_verify_parser = index_subparsers.add_parser(
        "verify", help="Check that records and index are in sync"
    )
    index_verify_parser.add_argument(
        "--records", type=Path, default=workspace.records, help="Records JSON to compare"
    )
    index_verify_parser.add_argument(
        "--index", type=Path, default=workspace.index, help="Embedding index to verify"
    )
    index_verify_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    index_verify_parser.set_defaults(func=index_verify)
    index_migrate_parser = index_subparsers.add_parser(
        "migrate", help="Migrate an index from v3 Base64 to v4 dual-file binary format"
    )
    index_migrate_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input index file (e.g. index-v3.json)",
    )
    index_migrate_parser.add_argument(
        "--output", type=Path, default=None, help="Path to the output index file (e.g. index.json)"
    )
    index_migrate_parser.add_argument(
        "--output-dir", type=Path, default=None, help="Directory to save the migrated index"
    )
    index_migrate_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    index_migrate_parser.set_defaults(func=index_migrate)
    index_pull_parser = index_subparsers.add_parser(
        "pull", help="Pull a pre-built index and records from preset, URL, or bundled demo"
    )
    index_pull_parser.add_argument(
        "source",
        nargs="?",
        default="demo",
        help="Preset name (demo, starter, curated-1k) or URL to index (default: demo)",
    )
    index_pull_parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace.root,
        help="Directory to save downloaded index and records (default: workspace root)",
    )
    index_pull_parser.add_argument(
        "--sha256",
        default=None,
        help="Expected SHA-256 hex checksum to verify download integrity",
    )
    index_pull_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing index and records files in target directory",
    )
    index_pull_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    index_pull_parser.set_defaults(func=index_pull)

    index_append_parser = index_subparsers.add_parser(
        "append", help="Incrementally append a repository or extra records to an index"
    )
    index_append_parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="GitHub repository identifier (e.g. owner/repo) to collect and append",
    )
    index_append_parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to extra records JSON file to append",
    )
    index_append_parser.add_argument(
        "--index",
        type=Path,
        default=workspace.index,
        help="Target embedding index to update (default: workspace index)",
    )
    index_append_parser.add_argument(
        "--records",
        type=Path,
        default=workspace.records,
        help="Target records JSON file to update (default: workspace records)",
    )
    index_append_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-embedding even if fingerprint has not changed",
    )
    index_append_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    index_append_parser.set_defaults(func=index_append)

    index_merge_parser = index_subparsers.add_parser(
        "merge", help="Merge multiple embedding indexes and vector matrices into a unified index"
    )
    index_merge_parser.add_argument(
        "--indices",
        type=Path,
        nargs="+",
        required=True,
        help="Input index JSON files to merge",
    )
    index_merge_parser.add_argument(
        "--output",
        type=Path,
        default=workspace.index,
        help="Target path for merged index JSON (default: workspace index)",
    )
    index_merge_parser.add_argument(
        "--records",
        type=Path,
        nargs="*",
        default=None,
        help="Optional corresponding input records JSON files to merge",
    )
    index_merge_parser.add_argument(
        "--output-records",
        type=Path,
        default=workspace.records,
        help="Target path for merged records JSON (default: workspace records)",
    )
    index_merge_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    index_merge_parser.set_defaults(func=index_merge)

    index_prune_parser = index_subparsers.add_parser(
        "prune", help="Prune archived, disabled, or blacklisted repositories from index and records"
    )
    index_prune_parser.add_argument(
        "--index",
        type=Path,
        default=workspace.index,
        help="Embedding index to prune (default: workspace index)",
    )
    index_prune_parser.add_argument(
        "--records",
        type=Path,
        default=workspace.records,
        help="Records JSON to prune (default: workspace records)",
    )
    index_prune_parser.add_argument(
        "--archived",
        action="store_true",
        default=True,
        help="Prune repositories archived on GitHub (default: True)",
    )
    index_prune_parser.add_argument(
        "--no-archived",
        dest="archived",
        action="store_false",
        help="Do not prune archived repositories",
    )
    index_prune_parser.add_argument(
        "--disabled",
        action="store_true",
        default=True,
        help="Prune repositories disabled on GitHub (default: True)",
    )
    index_prune_parser.add_argument(
        "--no-disabled",
        dest="disabled",
        action="store_false",
        help="Do not prune disabled repositories",
    )
    index_prune_parser.add_argument(
        "--remove-repo",
        type=str,
        action="append",
        default=[],
        help="Specific repo_id to prune (can be repeated)",
    )
    index_prune_parser.add_argument(
        "--blocklist",
        type=Path,
        default=None,
        help="File containing repo_ids to prune (one per line)",
    )
    index_prune_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report repositories to be pruned without modifying files",
    )
    index_prune_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    index_prune_parser.set_defaults(func=index_prune)

    records_p = subparsers.add_parser("records", help="Inspect generated repository records")
    records_subparsers = records_p.add_subparsers(dest="records_command", required=True)
    records_inspect_parser = records_subparsers.add_parser(
        "inspect", help="Print a compact summary of records"
    )
    records_inspect_parser.add_argument(
        "--records", type=Path, default=workspace.records, help="Records JSON to inspect"
    )
    records_inspect_parser.add_argument(
        "--repo", default=None, help="Only show records whose owner/repo contains this text"
    )
    records_inspect_parser.add_argument(
        "--limit", type=int, default=20, help="Maximum records to print"
    )
    records_inspect_parser.set_defaults(func=records_inspect)
    records_stats_parser = records_subparsers.add_parser(
        "stats", help="Summarize records quality and metadata"
    )
    records_stats_parser.add_argument(
        "--records", type=Path, default=workspace.records, help="Records JSON to summarize"
    )
    records_stats_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum languages/topics/project types to print"
    )
    records_stats_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    records_stats_parser.set_defaults(func=records_stats)
    records_validate_parser = records_subparsers.add_parser(
        "validate", help="Validate record schema and profile quality"
    )
    records_validate_parser.add_argument(
        "--records", type=Path, default=workspace.records, help="Records JSON to validate"
    )
    records_validate_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    records_validate_parser.set_defaults(func=records_validate)

    profile_p = subparsers.add_parser("profile", help="Refresh LLM profiles for records")
    profile_subparsers = profile_p.add_subparsers(dest="profile_command", required=True)
    profile_refresh_parser = profile_subparsers.add_parser(
        "refresh", help="Regenerate LLM profiles and schema v2 fields"
    )
    profile_refresh_parser.add_argument(
        "--records", type=Path, default=workspace.records, help="Records JSON to refresh"
    )
    profile_refresh_parser.add_argument(
        "--output",
        type=Path,
        default=workspace.refreshed_records,
        help="Path to write refreshed records JSON",
    )
    profile_refresh_parser.add_argument(
        "--force", action="store_true", help="Refresh every record instead of only outdated ones"
    )
    profile_refresh_parser.add_argument(
        "--workers", type=int, default=1, help="Concurrent LLM refresh workers (default: 1)"
    )
    profile_refresh_parser.add_argument(
        "--resume", action="store_true", help="Resume from an existing partial JSONL checkpoint"
    )
    profile_refresh_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate refresh work without calling the LLM or writing files",
    )
    profile_refresh_parser.add_argument(
        "--report", type=Path, default=None, help="Path to write a refresh failure report JSON"
    )
    profile_refresh_parser.add_argument(
        "--retry-failed",
        type=Path,
        default=None,
        help="Only process repos listed in a failure report JSON",
    )
    profile_refresh_parser.add_argument(
        "--only-missing-search-text",
        action="store_true",
        help="Only refresh records whose profile is missing search_text",
    )
    profile_refresh_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    profile_refresh_parser.set_defaults(func=profile_refresh)

    search_parser = subparsers.add_parser("search", help="Search the embedding index")
    search_parser.add_argument("query", help="Natural-language query")
    search_parser.add_argument(
        "--index", type=Path, default=workspace.index, help="Embedding index to search"
    )
    search_parser.add_argument(
        "--records",
        type=Path,
        default=workspace.records,
        help="Records JSON for offline metadata search",
    )
    search_parser.add_argument(
        "--demo", action="store_true", help="Search the bundled starter demo dataset"
    )
    search_parser.add_argument(
        "--offline",
        action="store_true",
        help="Perform offline lexical/metadata search without embedding APIs",
    )
    search_parser.add_argument(
        "--top-k", type=int, default=10, help="Maximum number of results to return"
    )
    search_parser.add_argument(
        "--ranking-strategy",
        choices=RANKING_STRATEGIES,
        default="metadata",
        help="Ranking mode: metadata, semantic, rerank, or hybrid",
    )
    search_parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=50,
        help="Embedding candidates to send to a reranker (default: 50)",
    )
    search_parser.add_argument(
        "--exploratory-threshold",
        type=float,
        default=0.35,
        help="Minimum embedding similarity for a non-identity result (default: 0.35)",
    )
    search_parser.add_argument(
        "--rerank-abstain-threshold",
        type=float,
        default=None,
        help="Optional minimum cross-encoder score required for the fused top result",
    )
    search_parser.add_argument(
        "--confidence-calibration",
        choices=CONFIDENCE_CALIBRATION_MODES,
        default="off",
        help="Post-ranking confidence calibration mode (default: off)",
    )
    search_parser.add_argument(
        "--query-transform-mode",
        choices=QUERY_TRANSFORM_MODES,
        default="off",
        help="Optional English canonical query mode: off, canonical, or merge (default: off)",
    )
    search_parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Filter by primary language (e.g. python, rust, go)",
    )
    search_parser.add_argument(
        "--ecosystem",
        type=str,
        default=None,
        help="Filter by package ecosystem (e.g. pypi, npm, cargo)",
    )
    search_parser.add_argument(
        "--project-type",
        type=str,
        default=None,
        help="Filter by project type (e.g. framework, library, cli_tool)",
    )
    search_parser.add_argument(
        "--min-stars", type=int, default=None, help="Minimum GitHub star count"
    )
    search_parser.add_argument(
        "--max-stars", type=int, default=None, help="Maximum GitHub star count"
    )
    search_parser.add_argument(
        "--license",
        type=str,
        default=None,
        help="Filter by license SPDX ID or name (e.g. mit, apache-2.0)",
    )
    search_parser.add_argument(
        "--topic",
        action="append",
        default=None,
        dest="topics",
        help="Filter by topic tag (can be repeated)",
    )
    search_parser.add_argument(
        "--include-archived",
        action="store_true",
        default=False,
        help="Include archived or disabled repositories",
    )
    search_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    search_parser.set_defaults(func=search)

    similar_parser = subparsers.add_parser(
        "similar", help="Find similar repositories to an indexed project"
    )
    similar_parser.add_argument("repo_id", help="Repository identifier (e.g. owner/repo)")
    similar_parser.add_argument(
        "--index", type=Path, default=workspace.index, help="Embedding index to search"
    )
    similar_parser.add_argument(
        "--demo", action="store_true", help="Search the bundled starter demo dataset"
    )
    similar_parser.add_argument(
        "--top-k", type=int, default=10, help="Maximum number of similar results to return"
    )
    similar_parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Filter by primary language (e.g. python, rust, go)",
    )
    similar_parser.add_argument(
        "--ecosystem",
        type=str,
        default=None,
        help="Filter by package ecosystem (e.g. pypi, npm, cargo)",
    )
    similar_parser.add_argument(
        "--project-type",
        type=str,
        default=None,
        help="Filter by project type (e.g. framework, library, cli_tool)",
    )
    similar_parser.add_argument(
        "--min-stars", type=int, default=None, help="Minimum GitHub star count"
    )
    similar_parser.add_argument(
        "--max-stars", type=int, default=None, help="Maximum GitHub star count"
    )
    similar_parser.add_argument(
        "--license",
        type=str,
        default=None,
        help="Filter by license SPDX ID or name (e.g. mit, apache-2.0)",
    )
    similar_parser.add_argument(
        "--topic",
        action="append",
        default=None,
        dest="topics",
        help="Filter by topic tag (can be repeated)",
    )
    similar_parser.add_argument(
        "--include-archived",
        action="store_true",
        default=False,
        help="Include archived or disabled repositories",
    )
    similar_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    similar_parser.set_defaults(func=similar)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare 2 to 5 repositories side-by-side"
    )
    compare_parser.add_argument(
        "repo_ids",
        nargs="+",
        help="2 to 5 repository identifiers to compare (e.g. repo_a repo_b)",
    )
    compare_parser.add_argument(
        "--index", type=Path, default=workspace.index, help="Embedding index to inspect"
    )
    compare_parser.add_argument(
        "--demo", action="store_true", help="Compare projects in bundled starter demo dataset"
    )
    compare_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json for scripts and agents",
    )
    compare_parser.set_defaults(func=compare)

    eval_p = subparsers.add_parser("eval", help="Evaluate retrieval quality")
    eval_subparsers = eval_p.add_subparsers(dest="eval_command", required=True)
    eval_run_parser = eval_subparsers.add_parser(
        "run", help="Run retrieval evaluation against an index"
    )
    eval_run_parser.add_argument(
        "--cases", type=Path, default=workspace.eval_cases, help="Evaluation dataset JSON"
    )
    eval_run_parser.add_argument(
        "--index", type=Path, default=workspace.index, help="Embedding index to evaluate"
    )
    eval_run_parser.add_argument(
        "--output",
        type=Path,
        default=workspace.eval_report,
        help="Path to write evaluation report JSON",
    )
    eval_run_parser.add_argument(
        "--top-k", type=int, default=10, help="Maximum results to score per query"
    )
    eval_run_parser.add_argument(
        "--batch-size", type=int, default=64, help="Number of queries to embed per batch"
    )
    eval_run_parser.add_argument(
        "--ranking-strategy",
        choices=RANKING_STRATEGIES,
        default="metadata",
        help="Ranking mode: metadata, semantic, rerank, or hybrid",
    )
    eval_run_parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=50,
        help="Embedding candidates to send to a reranker (default: 50)",
    )
    eval_run_parser.add_argument(
        "--exploratory-threshold",
        type=float,
        default=0.35,
        help="Minimum embedding similarity for a non-identity result (default: 0.35)",
    )
    eval_run_parser.add_argument(
        "--rerank-abstain-threshold",
        type=float,
        default=None,
        help="Optional minimum cross-encoder score required for the fused top result",
    )
    eval_run_parser.add_argument(
        "--confidence-calibration",
        choices=CONFIDENCE_CALIBRATION_MODES,
        default="off",
        help="Post-ranking confidence calibration mode (default: off)",
    )
    eval_run_parser.add_argument(
        "--query-transform-mode",
        choices=QUERY_TRANSFORM_MODES,
        default="off",
        help="Optional English canonical query mode: off, canonical, or merge (default: off)",
    )
    eval_run_parser.add_argument(
        "--canonical-queries",
        type=Path,
        default=None,
        help="Optional JSON object mapping every eval case id to a fixed canonical query",
    )
    eval_run_parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="Records JSON used for optional LLM judge comparisons",
    )
    eval_run_parser.add_argument(
        "--llm-judge", action="store_true", help="Run an LLM pairwise judge on top-1 mismatches"
    )
    eval_run_parser.set_defaults(func=eval_run)

    eval_inspect_parser = eval_subparsers.add_parser(
        "inspect", help="Inspect misses and summary from an evaluation report"
    )
    eval_inspect_parser.add_argument(
        "--report",
        type=Path,
        default=workspace.eval_report,
        help="Evaluation report JSON to inspect",
    )
    eval_inspect_parser.add_argument(
        "--status",
        choices=("exact", "acceptable", "serious_mismatch", "insufficient_evidence"),
        default=None,
        help="Only show cases with this top-1 status",
    )
    eval_inspect_parser.add_argument("--limit", type=int, default=20, help="Maximum cases to print")
    eval_inspect_parser.add_argument(
        "--include-exact",
        action="store_true",
        help="Include exact top-1 cases in the inspection output",
    )
    eval_inspect_parser.add_argument(
        "--tag", default=None, help="Only show cases carrying this tag"
    )
    eval_inspect_parser.add_argument(
        "--query-intent",
        dest="query_intent",
        choices=("empty", "exact_name", "alternative", "domain", "functional"),
        default=None,
        help="Only show cases with this query intent",
    )
    eval_inspect_parser.set_defaults(func=eval_inspect)

    eval_cases_parser = eval_subparsers.add_parser(
        "cases", help="Validate and summarize an evaluation dataset"
    )
    eval_cases_parser.add_argument(
        "--cases", type=Path, default=workspace.eval_cases, help="Evaluation dataset JSON"
    )
    eval_cases_parser.add_argument("--tag", default=None, help="Only show cases carrying this tag")
    eval_cases_parser.add_argument(
        "--query-intent",
        dest="query_intent",
        choices=("empty", "exact_name", "alternative", "domain", "functional"),
        default=None,
        help="Only show cases with this query intent",
    )
    eval_cases_parser.add_argument("--limit", type=int, default=20, help="Maximum cases to print")
    eval_cases_parser.set_defaults(func=eval_cases)

    _prioritize_root_command_help(parser, subparsers)
    return parser


def main() -> int:
    load_workspace_environment(resolve_workspace())
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIDENCE_CALIBRATION_MODES",
    "QUERY_TRANSFORM_MODES",
    "RANKING_STRATEGIES",
    "_XistsHelpFormatter",
    "_append_ingest_checkpoint",
    "_append_profile_refresh_checkpoint",
    "_append_search_detail",
    "_check_payload",
    "_chunks",
    "_collect_with_fallback",
    "_collect_with_rate_limit",
    "_compute_checkpoint_checksum",
    "_counter_items",
    "_failed_repo_ids_from_report",
    "_failure_entry",
    "_format_command_summary",
    "_format_compare_text",
    "_format_doctor_text",
    "_format_dry_run_text",
    "_format_index_stats_text",
    "_format_index_verify_text",
    "_format_project_badge",
    "_format_records_stats_text",
    "_format_records_validation_text",
    "_format_search_number",
    "_format_search_text",
    "_format_similar_text",
    "_format_top_items",
    "_index_checkpoint_path",
    "_index_metadata_by_repo_id",
    "_index_stats_report",
    "_index_summaries_by_repo_id",
    "_index_verify_report",
    "_index_write_checkpoint",
    "_ingest_checkpoint_path",
    "_ingest_graphql_batch",
    "_ingest_one",
    "_load_canonical_queries",
    "_load_checkpoint_resilient",
    "_load_ingest_checkpoint",
    "_load_profile_refresh_checkpoint",
    "_prepare_query_transforms",
    "_print_embedding_error",
    "_print_ingest_progress",
    "_prioritize_root_command_help",
    "_profile_refresh_checkpoint_path",
    "_profile_refresh_report_payload",
    "_read_index_file",
    "_read_records_file",
    "_records_next_steps",
    "_records_stats_report",
    "_safe_divide",
    "_search_confidence_text",
    "_summarize_error",
    "_terminal_width",
    "_wrap_terminal_text",
    "build_parser",
    "call_embeddings",
    "collect_record",
    "collect_record_graphql",
    "collect_records_graphql",
    "compare",
    "doctor",
    "embedding_config_from_env",
    "eval_cases",
    "eval_inspect",
    "eval_run",
    "evaluate_dataset",
    "generate_llm_profile",
    "github_token_from_env",
    "github_token_from_file",
    "index_append",
    "index_build",
    "index_merge",
    "index_migrate",
    "index_prune",
    "index_pull",
    "index_stats",
    "index_verify",
    "ingest_github",
    "inspect_report",
    "load_dataset",
    "load_env_file",
    "load_records_file",
    "load_repo_ids",
    "load_report",
    "load_workspace_environment",
    "llm_config_from_env",
    "main",
    "mcp",
    "probe_embedding_endpoint",
    "profile_refresh",
    "public_compare_projects",
    "public_find_similar",
    "public_search",
    "records_inspect",
    "records_stats",
    "records_validate",
    "resolve_workspace",
    "search",
    "similar",
    "version",
    "workspace_init",
    "write_json",
    "write_json_atomic",
]
