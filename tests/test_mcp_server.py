import asyncio
import json
from pathlib import Path

import pytest

from xists.mcp_server import MCPNotInstalledError, MCPStartupError, run_server


def _index() -> dict:
    return {
        "index_version": 3,
        "record_schema_version": 2,
        "embedding_input_version": 3,
        "embedding_model": "example-model",
        "dimension": 2,
        "record_count": 1,
        "vectors": [
            {
                "repo_id": "owner/project",
                "metadata": {
                    "url": "https://github.com/owner/project",
                    "summary": "A project summary.",
                    "use_cases": ["automation"],
                    "capabilities": ["search"],
                    "not_public": "excluded",
                },
                "vector": [1.0, 0.0],
            }
        ],
    }


def _tool_payload(server, name: str, arguments: dict) -> dict:
    content, structured = asyncio.run(server.call_tool(name, arguments))
    assert len(content) == 1
    assert json.loads(content[0].text) == structured
    return structured


def test_run_server_requires_an_existing_index(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "example-model")

    with pytest.raises(MCPStartupError, match="Index file not found"):
        run_server(tmp_path / "index.json")


def test_missing_optional_sdk_has_actionable_error(monkeypatch):
    import xists.mcp_server as server

    def missing_sdk():
        raise MCPNotInstalledError('MCP support is not installed. Install it with: pip install "xists[mcp]"')

    monkeypatch.setattr(server, "_fastmcp_class", missing_sdk)

    with pytest.raises(MCPNotInstalledError, match='pip install "xists\\[mcp\\]"'):
        server.create_server({}, object())


def test_server_registers_search_inspect_and_index_tools(monkeypatch):
    import xists.mcp_server as server_module

    monkeypatch.setattr(
        server_module,
        "public_search",
        lambda query, index, **kwargs: {
            "query": query,
            "query_intent": {"type": "functional"},
            "abstained": False,
            "results": [{"repo_id": "owner/project", "score": 0.8, "confidence": "high"}],
        },
    )
    server = server_module.create_server(_index(), object())

    assert {tool.name for tool in asyncio.run(server.list_tools())} == {
        "search_projects",
        "inspect_project",
        "index_stats",
    }
    search = _tool_payload(server, "search_projects", {"query": "project", "top_k": 5})
    assert search["results"] == [
        {
            "repo_id": "owner/project",
            "score": 0.8,
            "confidence": "high",
            "url": "https://github.com/owner/project",
            "summary": "A project summary.",
            "use_cases": ["automation"],
            "capabilities": ["search"],
        }
    ]
    inspect = _tool_payload(server, "inspect_project", {"repo_id": "owner/project"})
    assert inspect["summary"] == "A project summary."
    assert "not_public" not in inspect
    stats = _tool_payload(server, "index_stats", {})
    assert stats["embedding_model"] == "example-model"
    assert "vectors" not in stats


def test_search_projects_rejects_invalid_top_k(monkeypatch):
    import xists.mcp_server as server_module
    from mcp.server.fastmcp.exceptions import ToolError

    server = server_module.create_server(_index(), object())

    with pytest.raises(ToolError, match="top_k must be an integer between 1 and 20"):
        asyncio.run(server.call_tool("search_projects", {"query": "project", "top_k": 21}))
