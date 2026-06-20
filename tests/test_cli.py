import json
from pathlib import Path
from unittest.mock import patch

from xists.cli import build_parser, index_build, ingest_github, load_env_file, load_repo_ids


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


def _make_record(repo_id: str) -> dict:
    return {
        "repo_id": repo_id,
        "url": f"https://github.com/{repo_id}",
        "github": {"description": f"{repo_id} description", "topics": [], "language": "Python"},
        "llm_profile": {"summary": f"{repo_id} summary", "confidence": "high", "abstained": False},
    }


def test_ingest_github_skips_existing_records(tmp_path, monkeypatch):
    repos_file = tmp_path / "repos.txt"
    repos_file.write_text("a/b\nc/d\n", encoding="utf-8")

    output_file = tmp_path / "records.json"
    output_file.write_text(json.dumps([_make_record("a/b")]), encoding="utf-8")

    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "http://test/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    collected = []

    def fake_collect(repo_id, *, token=None):
        collected.append(repo_id)
        return _make_record(repo_id)

    def fake_generate(record, config, *, caller=None):
        return record.get("llm_profile", {})

    args = build_parser().parse_args(
        ["ingest", "github", "--repos", str(repos_file), "--output", str(output_file), "--report", str(tmp_path / "report.json")]
    )

    with patch("xists.cli.collect_record", side_effect=fake_collect), \
         patch("xists.cli.generate_llm_profile", side_effect=fake_generate):
        code = ingest_github(args)

    assert code == 0
    assert collected == ["c/d"]

    merged = json.loads(output_file.read_text())
    assert len(merged) == 2
    assert [r["repo_id"] for r in merged] == ["a/b", "c/d"]


def test_ingest_github_creates_new_file_when_no_existing(tmp_path, monkeypatch):
    repos_file = tmp_path / "repos.txt"
    repos_file.write_text("x/y\n", encoding="utf-8")

    output_file = tmp_path / "records.json"

    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "http://test/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    args = build_parser().parse_args(
        ["ingest", "github", "--repos", str(repos_file), "--output", str(output_file), "--report", str(tmp_path / "report.json")]
    )

    with patch("xists.cli.collect_record", side_effect=lambda rid, **kw: _make_record(rid)), \
         patch("xists.cli.generate_llm_profile", side_effect=lambda r, c, **kw: r.get("llm_profile", {})):
        code = ingest_github(args)

    assert code == 0
    merged = json.loads(output_file.read_text())
    assert len(merged) == 1
    assert merged[0]["repo_id"] == "x/y"


def test_index_build_skips_existing_vectors(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "local")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:6597/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    records_file = tmp_path / "records.json"
    records_file.write_text(json.dumps([
        {"repo_id": "a/b", "github": {"description": "A", "topics": []}, "llm_profile": {"summary": "A summary", "search_phrases": []}},
        {"repo_id": "c/d", "github": {"description": "C", "topics": []}, "llm_profile": {"summary": "C summary", "search_phrases": []}},
    ]), encoding="utf-8")

    output_file = tmp_path / "index.json"
    output_file.write_text(json.dumps({
        "index_version": 1,
        "embedding_model": "BAAI/bge-m3",
        "embedding_base_url": "http://localhost:6597/v1",
        "dimension": 4,
        "built_at": "2026-01-01T00:00:00+00:00",
        "record_count": 1,
        "skipped": [],
        "vectors": [{"repo_id": "a/b", "vector": [1.0, 0.0, 0.0, 0.0]}],
    }), encoding="utf-8")

    def fake_call_embeddings(config, inputs, *, timeout=60):
        return [[0.0, 1.0, 0.0, 0.0] for _ in inputs]

    args = build_parser().parse_args(
        ["index", "build", "--records", str(records_file), "--output", str(output_file)]
    )

    with patch("xists.search.index.call_embeddings", side_effect=fake_call_embeddings):
        code = index_build(args)

    assert code == 0
    index = json.loads(output_file.read_text())
    assert index["record_count"] == 2
    assert len(index["vectors"]) == 2
    assert index["vectors"][0]["repo_id"] == "a/b"
    assert index["vectors"][0]["vector"] == [1.0, 0.0, 0.0, 0.0]
    assert index["vectors"][1]["repo_id"] == "c/d"


def test_index_build_rejects_model_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "local")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:6597/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    records_file = tmp_path / "records.json"
    records_file.write_text(json.dumps([
        {"repo_id": "a/b", "github": {"description": "A", "topics": []}, "llm_profile": {"summary": "A", "search_phrases": []}},
    ]), encoding="utf-8")

    output_file = tmp_path / "index.json"
    output_file.write_text(json.dumps({
        "index_version": 1,
        "embedding_model": "different-model",
        "dimension": 4,
        "record_count": 1,
        "skipped": [],
        "vectors": [{"repo_id": "a/b", "vector": [1.0, 0.0, 0.0, 0.0]}],
    }), encoding="utf-8")

    args = build_parser().parse_args(
        ["index", "build", "--records", str(records_file), "--output", str(output_file)]
    )

    code = index_build(args)
    assert code == 1
