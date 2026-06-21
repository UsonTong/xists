import json
from pathlib import Path
from unittest.mock import patch

from xists.cli import build_parser, index_build, ingest_github, load_env_file, load_repo_ids
from xists.search.embed import EmbeddingError


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

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["started_at"].endswith("+00:00")
    assert report["finished_at"].endswith("+00:00")
    assert report["duration_seconds"] >= 0
    assert report["workers"] == 1
    assert report["force"] is False
    assert report["xists_version"]
    assert report["llm"] == {
        "provider": "openai_compatible",
        "model": "m",
        "prompt_version": 1,
    }
    serialized_report = json.dumps(report)
    assert "key" not in serialized_report
    assert "http://test/v1" not in serialized_report


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


def test_index_build_rebuilds_legacy_vectors_without_fingerprints(tmp_path, monkeypatch):
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

    with patch("xists.cli.call_embeddings", side_effect=fake_call_embeddings):
        code = index_build(args)

    assert code == 0
    index = json.loads(output_file.read_text())
    assert index["record_count"] == 2
    assert len(index["vectors"]) == 2
    assert index["vectors"][0]["repo_id"] == "a/b"
    assert index["vectors"][0]["vector"] == [0.0, 1.0, 0.0, 0.0]
    assert index["vectors"][1]["repo_id"] == "c/d"
    assert index["vectors"][1]["embedding_input_fingerprint"]


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


def test_ingest_github_force_reprocesses_existing(tmp_path, monkeypatch):
    repos_file = tmp_path / "repos.txt"
    repos_file.write_text("a/b\n", encoding="utf-8")

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
        ["ingest", "github", "--repos", str(repos_file), "--output", str(output_file),
         "--report", str(tmp_path / "report.json"), "--force"]
    )

    with patch("xists.cli.collect_record", side_effect=fake_collect), \
         patch("xists.cli.generate_llm_profile", side_effect=fake_generate):
        code = ingest_github(args)

    assert code == 0
    assert collected == ["a/b"]

    merged = json.loads(output_file.read_text())
    assert len(merged) == 1


def test_ingest_github_checkpoint_writes_after_each_record(tmp_path, monkeypatch):
    repos_file = tmp_path / "repos.txt"
    repos_file.write_text("a/b\nc/d\ne/f\n", encoding="utf-8")

    output_file = tmp_path / "records.json"

    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "http://test/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    call_count = 0

    def fake_collect(repo_id, *, token=None):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise Exception("simulated crash")
        return _make_record(repo_id)

    def fake_generate(record, config, *, caller=None):
        return record.get("llm_profile", {})

    args = build_parser().parse_args(
        ["ingest", "github", "--repos", str(repos_file), "--output", str(output_file),
         "--report", str(tmp_path / "report.json")]
    )

    with patch("xists.cli.collect_record", side_effect=fake_collect), \
         patch("xists.cli.generate_llm_profile", side_effect=fake_generate):
        code = ingest_github(args)

    # Third repo crashed, but first two should be saved.
    assert code == 1
    saved = json.loads(output_file.read_text())
    assert len(saved) == 2
    assert [r["repo_id"] for r in saved] == ["a/b", "c/d"]


def test_ingest_github_force_ignores_existing(tmp_path, monkeypatch):
    repos_file = tmp_path / "repos.txt"
    repos_file.write_text("a/b\n", encoding="utf-8")

    output_file = tmp_path / "records.json"
    old_record = _make_record("a/b")
    old_record["llm_profile"]["summary"] = "old summary"
    output_file.write_text(json.dumps([old_record]), encoding="utf-8")

    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "http://test/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    new_record = _make_record("a/b")
    new_record["llm_profile"]["summary"] = "new summary"

    def fake_collect(repo_id, *, token=None):
        return new_record

    def fake_generate(record, config, *, caller=None):
        return record.get("llm_profile", {})

    args = build_parser().parse_args(
        ["ingest", "github", "--repos", str(repos_file), "--output", str(output_file),
         "--report", str(tmp_path / "report.json"), "--force"]
    )

    with patch("xists.cli.collect_record", side_effect=fake_collect), \
         patch("xists.cli.generate_llm_profile", side_effect=fake_generate):
        code = ingest_github(args)

    assert code == 0
    saved = json.loads(output_file.read_text())
    assert len(saved) == 1
    assert saved[0]["llm_profile"]["summary"] == "new summary"


def test_index_build_force_rebuilds_from_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "local")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:6597/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    records_file = tmp_path / "records.json"
    records_file.write_text(json.dumps([
        {"repo_id": "a/b", "github": {"description": "A", "topics": []}, "llm_profile": {"summary": "A summary", "search_phrases": []}},
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
        return [[0.0, 0.0, 1.0, 0.0] for _ in inputs]

    args = build_parser().parse_args(
        ["index", "build", "--records", str(records_file), "--output", str(output_file), "--force"]
    )

    with patch("xists.cli.call_embeddings", side_effect=fake_call_embeddings):
        code = index_build(args)

    assert code == 0
    index = json.loads(output_file.read_text())
    assert index["record_count"] == 1
    assert index["vectors"][0]["vector"] == [0.0, 0.0, 1.0, 0.0]


def test_index_build_checkpoint_writes_after_each_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "local")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:6597/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    records = []
    for i in range(3):
        records.append({
            "repo_id": f"r{i}/repo",
            "github": {"description": f"Repo {i}", "topics": []},
            "llm_profile": {"summary": f"Summary {i}", "search_phrases": []},
        })

    records_file = tmp_path / "records.json"
    records_file.write_text(json.dumps(records), encoding="utf-8")

    output_file = tmp_path / "index.json"

    def fake_call_embeddings(config, inputs, *, timeout=60):
        return [[float(i)] for i in range(len(inputs))]

    args = build_parser().parse_args(
        ["index", "build", "--records", str(records_file), "--output", str(output_file)]
    )

    with patch("xists.cli.call_embeddings", side_effect=fake_call_embeddings):
        code = index_build(args)

    assert code == 0
    index = json.loads(output_file.read_text())
    assert index["record_count"] == 3
    assert len(index["vectors"]) == 3


def test_index_build_checkpoint_saves_partial_on_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "local")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:6597/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    # Use 65 records to force 2 batches (batch_size=64).
    records = []
    for i in range(65):
        records.append({
            "repo_id": f"r{i}/repo",
            "github": {"description": f"Repo {i}", "topics": []},
            "llm_profile": {"summary": f"Summary {i}", "search_phrases": []},
        })

    records_file = tmp_path / "records.json"
    records_file.write_text(json.dumps(records), encoding="utf-8")

    output_file = tmp_path / "index.json"

    call_count = 0

    def fake_call_embeddings(config, inputs, *, timeout=60):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise EmbeddingError("simulated crash on second batch")
        return [[float(i)] for i in range(len(inputs))]

    args = build_parser().parse_args(
        ["index", "build", "--records", str(records_file), "--output", str(output_file)]
    )

    with patch("xists.cli.call_embeddings", side_effect=fake_call_embeddings):
        code = index_build(args)

    assert code == 1
    # First batch (64 records) should be saved even though second batch crashed.
    index = json.loads(output_file.read_text())
    assert index["record_count"] == 64
    assert len(index["vectors"]) == 64
