import json
from pathlib import Path
from unittest.mock import patch

from xists.cli import build_parser, index_pull, search, workspace_init
from xists.search.pull import compute_file_sha256


def test_cli_workspace_init_demo(tmp_path, monkeypatch, capsys):
    workspace_root_path = tmp_path / "demo_workspace"
    monkeypatch.setenv("XISTS_HOME", str(workspace_root_path))

    args = build_parser().parse_args(["init", "--demo"])
    assert args.func is workspace_init
    assert args.demo is True

    ret = workspace_init(args)
    assert ret == 0
    assert (workspace_root_path / "records.json").is_file()
    assert (workspace_root_path / "index.json").is_file()
    assert (workspace_root_path / "index.vectors.npy").is_file()
    assert (workspace_root_path / ".env").is_file()

    out = capsys.readouterr().out
    assert "Workspace" in out
    assert "(demo mode)" in out
    assert "20 starter repos" in out
    assert "Next steps" in out


def test_cli_index_pull_demo_text_and_json(tmp_path, monkeypatch, capsys):
    target_dir = tmp_path / "pulled_dir"
    args_text = build_parser().parse_args(["index", "pull", "demo", "--output-dir", str(target_dir)])
    assert args_text.func is index_pull

    ret = index_pull(args_text)
    assert ret == 0
    assert (target_dir / "index.json").is_file()
    assert (target_dir / "records.json").is_file()

    out_text = capsys.readouterr().out
    assert "Index pulled successfully" in out_text
    assert "20" in out_text

    # JSON format with --force
    args_json = build_parser().parse_args([
        "index", "pull", "demo",
        "--output-dir", str(target_dir),
        "--force",
        "--format", "json",
    ])
    ret_json = index_pull(args_json)
    assert ret_json == 0
    out_json = capsys.readouterr().out
    data = json.loads(out_json)
    assert data["preset"] == "demo"
    assert data["records_count"] >= 20
    assert data["index_version"] == 4


def test_cli_index_pull_fails_when_exists_without_force(tmp_path, capsys):
    target_dir = tmp_path / "conflict_dir"
    target_dir.mkdir()
    (target_dir / "index.json").write_text("{}", encoding="utf-8")

    args = build_parser().parse_args(["index", "pull", "demo", "--output-dir", str(target_dir)])
    ret = index_pull(args)
    assert ret == 1
    err = capsys.readouterr().err
    assert "Target files already exist" in err


def test_cli_search_demo_without_api_keys(monkeypatch, capsys):
    # Ensure no embedding keys in environment
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    args_json = build_parser().parse_args([
        "search", "python web framework",
        "--demo",
        "--format", "json",
    ])
    assert args_json.func is search
    assert args_json.demo is True

    ret = search(args_json)
    assert ret == 0
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["query"] == "python web framework"
    assert not result["abstained"]
    assert len(result["results"]) > 0
    top_repos = [r["repo_id"] for r in result["results"]]
    assert "fastapi/fastapi" in top_repos or "flask/flask" in top_repos

    # Text mode
    args_text = build_parser().parse_args([
        "search", "vector database",
        "--demo",
        "--format", "text",
    ])
    ret_text = search(args_text)
    assert ret_text == 0
    out_text = capsys.readouterr().out
    assert "Search" in out_text
    assert "qdrant/qdrant" in out_text


def test_cli_search_offline_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    args = build_parser().parse_args([
        "search", "code editor",
        "--offline",
        "--format", "json",
    ])
    assert args.offline is True
    ret = search(args)
    assert ret == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["abstained"]
    top_repos = [r["repo_id"] for r in result["results"]]
    assert "microsoft/vscode" in top_repos
