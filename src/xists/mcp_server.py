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
from xists.search.query import PreparedIndex, prepare_index
from xists.search.similar import _resolve_repo_index


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


def create_server(
    index: dict[str, Any] | PreparedIndex,
    embedding_config: EmbeddingConfig,
    *,
    cache: Any = None,
) -> Any:
    """Create the MCP server with state loaded once at process startup."""

    FastMCP = _fastmcp_class()
    server = FastMCP("xists")
    prepared = prepare_index(index)

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
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
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
            prepared,
            embedding_config=embedding_config,
            top_k=top_k,
            ranking_strategy=ranking_strategy,
            filters=filters or None,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            cache=cache,
        )
        return _enrich_search_result(result, prepared)

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
            prepared,
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
            prepared,
        )
        return dict(result)

    @server.tool(description="Inspect the indexed profile for one repository.")
    def inspect_project(repo_id: str) -> dict[str, Any]:
        """Return the stored, non-vector profile for an exact repository id."""

        normalized = repo_id.strip()
        if not normalized:
            raise ValueError("repo_id must be a non-empty repository id")
        idx = _resolve_repo_index(normalized, prepared)
        if idx is None:
            raise ValueError(f"Repository not found in the current index: {normalized}")
        entry = prepared.entries[idx]
        metadata = entry.get("metadata") if isinstance(entry, dict) else None
        if not isinstance(metadata, dict):
            metadata = {}
        return {"repo_id": str(entry.get("repo_id") or normalized), **_public_profile(metadata)}

    @server.tool(description="Show compatibility and size information for the local xists index.")
    def index_stats() -> dict[str, Any]:
        """Return public index metadata without embedding vectors."""

        return {
            "index_version": prepared.index_version,
            "record_schema_version": prepared.record_schema_version,
            "embedding_input_version": prepared.embedding_input_version,
            "embedding_model": prepared.embedding_model,
            "dimension": prepared.dimension,
            "record_count": prepared.record_count,
            "indexed_project_count": len(prepared.entries),
        }

    return server


def _metadata_by_repo_id(index: dict[str, Any] | PreparedIndex) -> dict[str, dict[str, Any]]:
    metadata_by_repo: dict[str, dict[str, Any]] = {}
    vectors = index.get("vectors") if isinstance(index, dict) else getattr(index, "entries", None)
    for entry in vectors or []:
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
    result: dict[str, Any], prepared: PreparedIndex | dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(prepared, PreparedIndex):
        prepared = prepare_index(prepared)

    raw_results = result.get("results") or []
    top_indices: list[int] = []
    items_to_enrich: list[tuple[dict[str, Any], int | None]] = []

    for item in raw_results:
        if not isinstance(item, dict):
            continue
        repo_id = item.get("repo_id")
        idx = prepared.repo_id_to_index.get(repo_id) if isinstance(repo_id, str) else None
        if idx is not None:
            top_indices.append(idx)
        items_to_enrich.append((item, idx))

    if hasattr(prepared.entries, "prefetch") and top_indices:
        prepared.entries.prefetch(top_indices)

    enriched_results: list[dict[str, Any]] = []
    for item, idx in items_to_enrich:
        enriched = dict(item)
        if idx is not None:
            entry = prepared.entries[idx]
            metadata = entry.get("metadata") if isinstance(entry, dict) else None
            if isinstance(metadata, dict):
                for field, value in _public_profile(metadata).items():
                    if field not in enriched:
                        enriched[field] = value
        enriched_results.append(enriched)
    return {**result, "results": enriched_results}


def run_server(index_path: Path, *, cache: Any = None) -> None:
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
        prepared = PreparedIndex.from_path(index_path, config=config)
    except Exception:
        try:
            index = load_index(index_path)
            if not isinstance(index, dict):
                raise MCPStartupError(
                    f"Could not load index {index_path}: index JSON must be an object"
                )
            prepared = prepare_index(index, config=config)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise MCPStartupError(f"Could not load index {index_path}: {error}") from error
        except Exception as error:
            raise MCPStartupError(f"Could not load index {index_path}: {error}") from error

    if cache is None:
        try:
            from xists.workspace import get_default_embedding_cache

            cache = get_default_embedding_cache()
        except Exception:
            cache = None

    create_server(prepared, config, cache=cache).run(transport="stdio")
