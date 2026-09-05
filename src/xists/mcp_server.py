"""MCP server assembly for xists.

The MCP SDK is intentionally imported only when this integration is used so
the core package remains usable without the optional dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xists.api import compare_projects as public_compare_projects
from xists.api import find_similar as public_find_similar
from xists.api import load_index
from xists.api import search as public_search
from xists.search.embed import EmbeddingConfig, embedding_config_from_env


class MCPNotInstalledError(RuntimeError):
    """Raised when the optional MCP SDK is unavailable."""


class MCPStartupError(RuntimeError):
    """Raised when the MCP server cannot load its local search state."""


MAX_TOP_K = 20
_PROFILE_FIELDS = (
    "url",
    "aliases",
    "description",
    "topics",
    "language",
    "license",
    "stars",
    "forks",
    "summary",
    "use_cases",
    "capabilities",
    "project_type",
    "ecosystem",
    "replaces",
    "related_projects",
)


def _fastmcp_class() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as error:
        if error.name == "mcp" or (error.name is not None and error.name.startswith("mcp.")):
            raise MCPNotInstalledError(
                'MCP support is not installed. Install it with: pip install "xists[mcp]"'
            ) from error
        raise
    return FastMCP


def create_server(index: dict[str, Any], embedding_config: EmbeddingConfig) -> Any:
    """Create the MCP server with state loaded once at process startup."""

    FastMCP = _fastmcp_class()
    server = FastMCP("xists")
    metadata_by_repo = _metadata_by_repo_id(index)

    @server.tool(description="Search the local xists project index.")
    def search_projects(
        query: str,
        top_k: int = 10,
        ranking_strategy: str = "hybrid",
        language: str | None = None,
        ecosystem: str | None = None,
        project_type: str | None = None,
        min_stars: int | None = None,
        max_stars: int | None = None,
        license: str | None = None,
        topics: list[str] | None = None,
        include_archived: bool = False,
    ) -> dict[str, Any]:
        """Return ranked project candidates for a natural-language query."""

        _validate_query(query)
        _validate_top_k(top_k)
        filters: dict[str, Any] = {}
        if language is not None:
            filters["language"] = language
        if ecosystem is not None:
            filters["ecosystem"] = ecosystem
        if project_type is not None:
            filters["project_type"] = project_type
        if min_stars is not None:
            filters["min_stars"] = min_stars
        if max_stars is not None:
            filters["max_stars"] = max_stars
        if license is not None:
            filters["license"] = license
        if topics is not None:
            filters["topics"] = topics
        if include_archived:
            filters["include_archived"] = True

        result = public_search(
            query,
            index,
            embedding_config=embedding_config,
            top_k=top_k,
            ranking_strategy=ranking_strategy,
            filters=filters or None,
        )
        return _enrich_search_result(result, metadata_by_repo)

    @server.tool(
        description="Find projects similar to an existing repository in the local xists index."
    )
    def find_similar_projects(
        repo_id: str,
        top_k: int = 10,
        language: str | None = None,
        ecosystem: str | None = None,
        project_type: str | None = None,
        min_stars: int | None = None,
        max_stars: int | None = None,
        license: str | None = None,
        topics: list[str] | None = None,
        include_archived: bool = False,
    ) -> dict[str, Any]:
        """Return top-k similar repositories using precomputed embeddings and structured metadata."""

        _validate_repo_id(repo_id)
        _validate_top_k(top_k)
        filters: dict[str, Any] = {}
        if language is not None:
            filters["language"] = language
        if ecosystem is not None:
            filters["ecosystem"] = ecosystem
        if project_type is not None:
            filters["project_type"] = project_type
        if min_stars is not None:
            filters["min_stars"] = min_stars
        if max_stars is not None:
            filters["max_stars"] = max_stars
        if license is not None:
            filters["license"] = license
        if topics is not None:
            filters["topics"] = topics
        if include_archived:
            filters["include_archived"] = True

        result = public_find_similar(
            repo_id,
            index,
            top_k=top_k,
            filters=filters or None,
        )
        return dict(result)

    @server.tool(description="Compare 2 to 5 repositories side by side from the local xists index.")
    def compare_projects(
        repo_ids: list[str],
    ) -> dict[str, Any]:
        """Return side-by-side attributes, pairwise similarity matrix, commonalities, and differentiators."""

        if not isinstance(repo_ids, (list, tuple)) or len(repo_ids) < 2 or len(repo_ids) > 5:
            raise ValueError("repo_ids must be a list of 2 to 5 repository identifiers")
        for r in repo_ids:
            if not isinstance(r, str) or not r.strip():
                raise ValueError(
                    "Each repository identifier in repo_ids must be a non-empty string"
                )

        result = public_compare_projects(
            list(repo_ids),
            index,
        )
        return dict(result)

    @server.tool(description="Inspect the indexed profile for one repository.")
    def inspect_project(repo_id: str) -> dict[str, Any]:
        """Return the stored, non-vector profile for an exact repository id."""

        normalized = repo_id.strip()
        if not normalized:
            raise ValueError("repo_id must be a non-empty repository id")
        metadata = metadata_by_repo.get(normalized)
        if metadata is None:
            raise ValueError(f"Repository not found in the current index: {normalized}")
        return {"repo_id": normalized, **_public_profile(metadata)}

    @server.tool(description="Show compatibility and size information for the local xists index.")
    def index_stats() -> dict[str, Any]:
        """Return public index metadata without embedding vectors."""

        return {
            "index_version": index.get("index_version"),
            "record_schema_version": index.get("record_schema_version"),
            "embedding_input_version": index.get("embedding_input_version"),
            "embedding_model": index.get("embedding_model"),
            "dimension": index.get("dimension"),
            "record_count": index.get("record_count"),
            "indexed_project_count": len(metadata_by_repo),
        }

    return server


def _metadata_by_repo_id(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata_by_repo: dict[str, dict[str, Any]] = {}
    for entry in index.get("vectors") or []:
        if not isinstance(entry, dict):
            continue
        repo_id = entry.get("repo_id")
        metadata = entry.get("metadata")
        if isinstance(repo_id, str) and isinstance(metadata, dict):
            metadata_by_repo[repo_id] = metadata
    return metadata_by_repo


def _public_profile(metadata: dict[str, Any]) -> dict[str, Any]:
    return {field: metadata[field] for field in _PROFILE_FIELDS if field in metadata}


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}")


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")


def _validate_repo_id(repo_id: str) -> None:
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty string")


def _enrich_search_result(
    result: dict[str, Any], metadata_by_repo: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    enriched_results: list[dict[str, Any]] = []
    for item in result.get("results") or []:
        if not isinstance(item, dict):
            continue
        enriched = dict(item)
        repo_id = item.get("repo_id")
        metadata = metadata_by_repo.get(repo_id) if isinstance(repo_id, str) else None
        if metadata:
            for field, value in _public_profile(metadata).items():
                if field not in enriched:
                    enriched[field] = value
        enriched_results.append(enriched)
    return {**result, "results": enriched_results}


def run_server(index_path: Path) -> None:
    """Load local search state and run the MCP stdio transport."""

    # Check the optional integration before local configuration so a core-only
    # installation always receives its actionable installation instruction.
    _fastmcp_class()
    try:
        config = embedding_config_from_env()
    except Exception as error:
        raise MCPStartupError(str(error)) from error
    if not index_path.exists():
        raise MCPStartupError(f"Index file not found: {index_path}. Run 'xists index build' first.")
    try:
        index = load_index(index_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise MCPStartupError(f"Could not load index {index_path}: {error}") from error
    if not isinstance(index, dict):
        raise MCPStartupError(f"Could not load index {index_path}: index JSON must be an object")

    create_server(index, config).run(transport="stdio")
