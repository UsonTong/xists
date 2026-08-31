import asyncio
import json
import os
import sys

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


def test_run_server_rejects_non_object_index(tmp_path, monkeypatch):
    index_path = tmp_path / "index.json"
    index_path.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("EMBEDDING_API_KEY", "key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "example-model")

    with pytest.raises(MCPStartupError, match="index JSON must be an object"):
        run_server(index_path)


def test_missing_optional_sdk_has_actionable_error(monkeypatch):
    import xists.mcp_server as server

    def missing_sdk():
        raise MCPNotInstalledError(
            'MCP support is not installed. Install it with: pip install "xists[mcp]"'
        )

    monkeypatch.setattr(server, "_fastmcp_class", missing_sdk)

    with pytest.raises(MCPNotInstalledError, match='pip install "xists\\[mcp\\]"'):
        server.create_server({}, object())


def test_run_server_checks_optional_sdk_before_embedding_config(tmp_path, monkeypatch):
    import xists.mcp_server as server

    def missing_sdk():
        raise MCPNotInstalledError(
            'MCP support is not installed. Install it with: pip install "xists[mcp]"'
        )

    monkeypatch.setattr(server, "_fastmcp_class", missing_sdk)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    with pytest.raises(MCPNotInstalledError, match='pip install "xists\\[mcp\\]"'):
        server.run_server(tmp_path / "index.json")


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
    from mcp.server.fastmcp.exceptions import ToolError

    import xists.mcp_server as server_module

    server = server_module.create_server(_index(), object())

    with pytest.raises(ToolError, match="top_k must be an integer between 1 and 20"):
        asyncio.run(server.call_tool("search_projects", {"query": "project", "top_k": 21}))


def test_search_projects_rejects_an_empty_query():
    from mcp.server.fastmcp.exceptions import ToolError

    server = __import__("xists.mcp_server", fromlist=["create_server"]).create_server(
        _index(), object()
    )

    with pytest.raises(ToolError, match="query must be a non-empty string"):
        asyncio.run(server.call_tool("search_projects", {"query": "  "}))


def test_search_projects_forwards_ranking_strategy(monkeypatch):
    import xists.mcp_server as server_module

    captured_kwargs = {}

    def fake_search(query, index, **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "query": query,
            "query_intent": {"type": "functional"},
            "abstained": False,
            "results": [{"repo_id": "owner/project", "score": 0.8, "confidence": "high"}],
        }

    monkeypatch.setattr(server_module, "public_search", fake_search)
    server = server_module.create_server(_index(), object())

    _tool_payload(
        server,
        "search_projects",
        {"query": "project", "top_k": 3, "ranking_strategy": "hybrid"},
    )
    assert captured_kwargs["ranking_strategy"] == "hybrid"
    assert captured_kwargs["top_k"] == 3


def test_search_projects_forwards_filters(monkeypatch):
    import xists.mcp_server as server_module

    captured_kwargs = {}

    def fake_search(query, index, **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "query": query,
            "query_intent": {"type": "functional"},
            "abstained": False,
            "results": [{"repo_id": "owner/project", "score": 0.8, "confidence": "high"}],
        }

    monkeypatch.setattr(server_module, "public_search", fake_search)
    server = server_module.create_server(_index(), object())

    _tool_payload(
        server,
        "search_projects",
        {
            "query": "project",
            "language": "python",
            "min_stars": 500,
            "license": "mit",
            "include_archived": True,
        },
    )
    assert captured_kwargs["filters"] == {
        "language": "python",
        "min_stars": 500,
        "license": "mit",
        "include_archived": True,
    }


def test_stdio_server_runs_tools_without_corrupting_protocol(tmp_path):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(_index()), encoding="utf-8")
    environment = {
        **os.environ,
        "EMBEDDING_API_KEY": "key",
        "EMBEDDING_BASE_URL": "https://example.test/v1",
        "EMBEDDING_MODEL": "example-model",
        "MCP_LOG_LEVEL": "CRITICAL",
    }
    stderr_path = tmp_path / "server.stderr"

    async def exercise_server(stderr):
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "xists.cli", "mcp", "--index", str(index_path)],
            env=environment,
        )
        async with stdio_client(parameters, errlog=stderr) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                stats = await session.call_tool("index_stats")
                inspected = await session.call_tool("inspect_project", {"repo_id": "owner/project"})
        return tools, stats, inspected

    with stderr_path.open("w+", encoding="utf-8") as stderr:
        tools, stats, inspected = asyncio.run(exercise_server(stderr))

    assert {tool.name for tool in tools.tools} == {
        "search_projects",
        "inspect_project",
        "index_stats",
    }
    assert stats.isError is False
    assert stats.structuredContent["indexed_project_count"] == 1
    assert inspected.isError is False
    assert inspected.structuredContent["repo_id"] == "owner/project"
    assert "\x1b" not in stderr_path.read_text(encoding="utf-8")
