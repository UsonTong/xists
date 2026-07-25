import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

from xists.cli import build_parser, doctor, load_workspace_environment, workspace_init
from xists.workspace import initialize_workspace, resolve_workspace, workspace_root


def test_workspace_root_uses_xists_home_and_expands_user(monkeypatch):
    monkeypatch.setenv("XISTS_HOME", "~/.xists-test")

    assert workspace_root() == (Path.home() / ".xists-test").resolve()


def test_resolve_workspace_uses_configured_root_in_empty_directory(tmp_path, monkeypatch):
    workspace_root_path = tmp_path / "workspace"
    empty_directory = tmp_path / "empty"
    empty_directory.mkdir()
    monkeypatch.setenv("XISTS_HOME", str(workspace_root_path))

    workspace = resolve_workspace(cwd=empty_directory)

    assert workspace.mode == "workspace"
    assert workspace.root == workspace_root_path
    assert workspace.repos == workspace_root_path / "repos.txt"
    assert workspace.records == workspace_root_path / "records.json"
    assert workspace.index == workspace_root_path / "index.json"
    assert workspace.eval_cases == workspace_root_path / "eval-cases.json"
    assert workspace.eval_report == workspace_root_path / "eval-report.json"


def test_resolve_workspace_uses_legacy_directory_for_any_legacy_file(tmp_path, monkeypatch):
    workspace_root_path = tmp_path / "workspace"
    legacy_directory = tmp_path / "legacy"
    legacy_directory.mkdir()
    (legacy_directory / "index.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("XISTS_HOME", str(workspace_root_path))

    workspace = resolve_workspace(cwd=legacy_directory)

    assert workspace.mode == "legacy"
    assert workspace.root == legacy_directory
    assert workspace.records == legacy_directory / "records.json"
    assert workspace.index == legacy_directory / "index.json"
    assert workspace.eval_cases == legacy_directory / "eval-cases.json"


def test_parser_uses_workspace_defaults_for_every_default_data_path(tmp_path, monkeypatch):
    workspace_root_path = tmp_path / "workspace"
    empty_directory = tmp_path / "empty"
    empty_directory.mkdir()
    monkeypatch.chdir(empty_directory)
    monkeypatch.setenv("XISTS_HOME", str(workspace_root_path))

    parser = build_parser()
    ingest = parser.parse_args(["ingest", "github"])
    doctor = parser.parse_args(["doctor"])
    index_build = parser.parse_args(["index", "build"])
    profile = parser.parse_args(["profile", "refresh"])
    search = parser.parse_args(["search", "query"])
    evaluation = parser.parse_args(["eval", "run"])
    inspection = parser.parse_args(["eval", "inspect"])

    assert ingest.repos == workspace_root_path / "repos.txt"
    assert ingest.output == workspace_root_path / "records.json"
    assert ingest.report == workspace_root_path / "report.json"
    assert doctor.records == workspace_root_path / "records.json"
    assert doctor.index == workspace_root_path / "index.json"
    assert doctor.cases == workspace_root_path / "eval-cases.json"
    assert index_build.records == workspace_root_path / "records.json"
    assert index_build.output == workspace_root_path / "index.json"
    assert profile.records == workspace_root_path / "records.json"
    assert profile.output == workspace_root_path / "records-v2.json"
    assert search.index == workspace_root_path / "index.json"
    assert evaluation.cases == workspace_root_path / "eval-cases.json"
    assert evaluation.index == workspace_root_path / "index.json"
    assert evaluation.output == workspace_root_path / "eval-report.json"
    assert inspection.report == workspace_root_path / "eval-report.json"


def test_parser_keeps_legacy_defaults_together_and_explicit_paths_win(tmp_path, monkeypatch):
    workspace_root_path = tmp_path / "workspace"
    legacy_directory = tmp_path / "legacy"
    legacy_directory.mkdir()
    (legacy_directory / "repos.txt").write_text("owner/repo\n", encoding="utf-8")
    monkeypatch.chdir(legacy_directory)
    monkeypatch.setenv("XISTS_HOME", str(workspace_root_path))

    parser = build_parser()
    defaults = parser.parse_args(["index", "verify"])
    explicit = parser.parse_args(
        ["index", "verify", "--records", "other-records.json", "--index", "other-index.json"]
    )

    assert defaults.records == legacy_directory / "records.json"
    assert defaults.index == legacy_directory / "index.json"
    assert explicit.records == Path("other-records.json")
    assert explicit.index == Path("other-index.json")


def test_parser_explicit_paths_override_workspace_defaults_for_all_workflows(tmp_path, monkeypatch):
    workspace_root_path = tmp_path / "workspace"
    current_directory = tmp_path / "empty"
    current_directory.mkdir()
    monkeypatch.chdir(current_directory)
    monkeypatch.setenv("XISTS_HOME", str(workspace_root_path))

    parser = build_parser()
    ingest = parser.parse_args(
        ["ingest", "github", "--repos", "input.txt", "--output", "output.json", "--report", "report.json"]
    )
    index = parser.parse_args(
        ["index", "build", "--records", "input.json", "--output", "index-output.json"]
    )
    profile = parser.parse_args(
        ["profile", "refresh", "--records", "input.json", "--output", "profile-output.json"]
    )
    search = parser.parse_args(["search", "query", "--index", "search-index.json"])
    evaluation = parser.parse_args(
        ["eval", "run", "--cases", "cases.json", "--index", "eval-index.json", "--output", "evaluation.json"]
    )

    assert (ingest.repos, ingest.output, ingest.report) == (
        Path("input.txt"),
        Path("output.json"),
        Path("report.json"),
    )
    assert (index.records, index.output) == (Path("input.json"), Path("index-output.json"))
    assert (profile.records, profile.output) == (Path("input.json"), Path("profile-output.json"))
    assert search.index == Path("search-index.json")
    assert (evaluation.cases, evaluation.index, evaluation.output) == (
        Path("cases.json"),
        Path("eval-index.json"),
        Path("evaluation.json"),
    )


def test_workspace_environment_priority_is_shell_then_current_directory_then_workspace(tmp_path, monkeypatch):
    workspace_root_path = tmp_path / "workspace"
    current_directory = tmp_path / "current"
    workspace_root_path.mkdir()
    current_directory.mkdir()
    (workspace_root_path / ".env").write_text(
        "XISTS_TEST_FROM_WORKSPACE=workspace\n"
        "XISTS_TEST_CURRENT_WINS=workspace\n"
        "XISTS_TEST_SHELL_WINS=workspace\n",
        encoding="utf-8",
    )
    (current_directory / ".env").write_text(
        "XISTS_TEST_CURRENT_WINS=current\n"
        "XISTS_TEST_SHELL_WINS=current\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XISTS_TEST_SHELL_WINS", "shell")
    monkeypatch.delenv("XISTS_TEST_FROM_WORKSPACE", raising=False)
    monkeypatch.delenv("XISTS_TEST_CURRENT_WINS", raising=False)

    load_workspace_environment(resolve_workspace(cwd=current_directory, environ={"XISTS_HOME": str(workspace_root_path)}), cwd=current_directory)

    assert os.environ["XISTS_TEST_FROM_WORKSPACE"] == "workspace"
    assert os.environ["XISTS_TEST_CURRENT_WINS"] == "current"
    assert os.environ["XISTS_TEST_SHELL_WINS"] == "shell"


def test_initialize_workspace_is_idempotent_and_preserves_existing_data(tmp_path):
    root = tmp_path / "workspace"

    assert initialize_workspace(root) == (True, True)
    (root / "records.json").write_text("[]\n", encoding="utf-8")
    original_env = (root / ".env").read_text(encoding="utf-8")

    assert initialize_workspace(root) == (False, False)
    assert (root / "records.json").read_text(encoding="utf-8") == "[]\n"
    assert (root / ".env").read_text(encoding="utf-8") == original_env

    if os.name == "posix":
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert stat.S_IMODE((root / ".env").stat().st_mode) == 0o600


def test_init_command_creates_only_the_configured_workspace(tmp_path, monkeypatch, capsys):
    workspace_root_path = tmp_path / "workspace"
    current_directory = tmp_path / "current"
    current_directory.mkdir()
    monkeypatch.chdir(current_directory)
    monkeypatch.setenv("XISTS_HOME", str(workspace_root_path))

    args = build_parser().parse_args(["init"])

    assert args.func is workspace_init
    with patch("urllib.request.urlopen", side_effect=AssertionError("init must not access the network")):
        assert workspace_init(args) == 0
    assert workspace_root_path.is_dir()
    assert (workspace_root_path / ".env").is_file()
    assert not (current_directory / ".env").exists()
    output = capsys.readouterr().out
    assert str(workspace_root_path) in output
    assert "Next steps" in output


def test_doctor_reports_workspace_paths_in_text_and_json(tmp_path, monkeypatch, capsys):
    workspace_root_path = tmp_path / "workspace"
    current_directory = tmp_path / "current"
    current_directory.mkdir()
    monkeypatch.chdir(current_directory)
    monkeypatch.setenv("XISTS_HOME", str(workspace_root_path))
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://embedding.invalid/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "fixture/embed")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "fixture/llm")

    args = build_parser().parse_args(["doctor"])

    assert doctor(args) == 0
    text = capsys.readouterr().out
    assert "Workspace" in text
    assert "default workspace" in text
    assert str(workspace_root_path) in text
    assert str(workspace_root_path / "records.json") in text
    assert str(workspace_root_path / "index.json") in text

    args = build_parser().parse_args(["doctor", "--format", "json"])
    assert doctor(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workspace"]["mode"] == "workspace"
    assert payload["workspace"]["root"] == str(workspace_root_path)
    assert payload["workspace"]["paths"] == {
        "records": str(workspace_root_path / "records.json"),
        "index": str(workspace_root_path / "index.json"),
        "cases": str(workspace_root_path / "eval-cases.json"),
    }
