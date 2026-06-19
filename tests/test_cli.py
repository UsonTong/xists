from pathlib import Path

from xists.cli import build_parser, load_env_file, load_repo_ids


def test_load_env_file_loads_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# local secrets",
                "GITHUB_TOKEN=from-file",
                "QUOTED=\"quoted value\"",
                "SINGLE_QUOTED='single quoted value'",
                "INVALID_LINE",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("SINGLE_QUOTED", raising=False)

    load_env_file(env_file)

    assert __import__("os").environ["GITHUB_TOKEN"] == "from-file"
    assert __import__("os").environ["QUOTED"] == "quoted value"
    assert __import__("os").environ["SINGLE_QUOTED"] == "single quoted value"


def test_load_env_file_does_not_override_existing_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GITHUB_TOKEN=from-file\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")

    load_env_file(env_file)

    assert __import__("os").environ["GITHUB_TOKEN"] == "from-env"


def test_load_env_file_ignores_missing_file(tmp_path):
    load_env_file(tmp_path / ".env")


def test_load_repo_ids_skips_blank_lines_and_comments(tmp_path):
    repos_file = tmp_path / "repos.txt"
    repos_file.write_text(
        "\n".join(
            [
                "# popular repos",
                "facebook/react",
                "",
                "https://github.com/vuejs/core",
            ]
        ),
        encoding="utf-8",
    )
    assert load_repo_ids(repos_file) == ["facebook/react", "vuejs/core"]


def test_ingest_github_parser_uses_default_paths():
    args = build_parser().parse_args(["ingest", "github"])

    assert args.repos == Path("repos.txt")
    assert args.output == Path("records.json")
    assert args.report == Path("report.json")


def test_ingest_github_parser_accepts_custom_paths():
    args = build_parser().parse_args(
        [
            "ingest",
            "github",
            "--repos",
            "data/repos.txt",
            "--output",
            "data/records.json",
            "--report",
            "data/report.json",
        ]
    )

    assert args.repos == Path("data/repos.txt")
    assert args.output == Path("data/records.json")
    assert args.report == Path("data/report.json")
