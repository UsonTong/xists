from pathlib import Path

import pytest

from xists.mcp_server import MCPNotInstalledError, MCPStartupError, run_server


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
