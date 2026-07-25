"""MCP server assembly for xists.

The MCP SDK is intentionally imported only when this integration is used so
the core package remains usable without the optional dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xists.api import load_index
from xists.search.embed import EmbeddingConfig, embedding_config_from_env


class MCPNotInstalledError(RuntimeError):
    """Raised when the optional MCP SDK is unavailable."""


class MCPStartupError(RuntimeError):
    """Raised when the MCP server cannot load its local search state."""


def _fastmcp_class() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as error:
        if error.name == "mcp" or error.name.startswith("mcp."):
            raise MCPNotInstalledError(
                'MCP support is not installed. Install it with: pip install "xists[mcp]"'
            ) from error
        raise
    return FastMCP


def create_server(index: dict[str, Any], embedding_config: EmbeddingConfig) -> Any:
    """Create the MCP server with state loaded once at process startup."""

    del index, embedding_config
    FastMCP = _fastmcp_class()
    return FastMCP("xists")


def run_server(index_path: Path) -> None:
    """Load local search state and run the MCP stdio transport."""

    try:
        config = embedding_config_from_env()
    except Exception as error:
        raise MCPStartupError(str(error)) from error
    if not index_path.exists():
        raise MCPStartupError(
            f"Index file not found: {index_path}. Run 'xists index build' first."
        )
    try:
        index = load_index(index_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise MCPStartupError(f"Could not load index {index_path}: {error}") from error

    create_server(index, config).run(transport="stdio")
