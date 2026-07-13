import json
from pathlib import Path
from unittest.mock import patch

from xists import __version__
from xists.cli import (
    build_parser,
    doctor,
    eval_cases,
    eval_inspect,
    eval_run,
    index_build,
    index_stats,
    ingest_github,
    load_env_file,
    load_repo_ids,
    records_inspect,
    version,
)
from xists.ingest.github import GitHubAPIError
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


def test_version_parser_accepts_version_command():
    args = build_parser().parse_args(["version"])

    assert args.func is version


def test_version_prints_json(capsys):
    args = build_parser().parse_args(["version"])

    code = version(args)

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"version": __version__}


def test_global_version_flag_prints_plain_version(capsys):
    try:
        build_parser().parse_args(["--version"])
    except SystemExit as error:
        assert error.code == 0

    assert capsys.readouterr().out.strip() == f"xists {__version__}"

def test_ingest_github_parser_uses_default_paths():
    args = build_parser().parse_args(["ingest", "github"])

    assert args.repos == Path("repos.txt")
    assert args.output == Path("records.json")
    assert args.report == Path("report.json")
    assert args.github_api == "rest"
    assert args.github_batch_size == 1


def test_ingest_github_parser_accepts_graphql_backend():
    args = build_parser().parse_args(["ingest", "github", "--github-api", "graphql", "--github-batch-size", "25"])

    assert args.github_api == "graphql"
    assert args.github_batch_size == 25


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


def test_doctor_parser_uses_default_paths():
    args = build_parser().parse_args(["doctor"])

    assert args.records == Path("records.json")
    assert args.index == Path("index.json")
    assert args.cases == Path("eval-cases.json")
    assert args.check_endpoints is False
    assert args.strict is False


def test_doctor_parser_accepts_strict_flag():
    args = build_parser().parse_args(["doctor", "--strict"])

    assert args.strict is True
    assert args.check_endpoints is False


def test_index_stats_parser_uses_default_path():
    args = build_parser().parse_args(["index", "stats"])

    assert args.index == Path("index.json")
    assert args.limit == 10


def test_records_inspect_parser_uses_default_path():
    args = build_parser().parse_args(["records", "inspect"])

    assert args.records == Path("records.json")
    assert args.repo is None
    assert args.limit == 20


def test_eval_run_parser_uses_default_paths():
    args = build_parser().parse_args(["eval", "run"])

    assert args.cases == Path("eval-cases.json")
    assert args.index == Path("index.json")
    assert args.output == Path("eval-report.json")
    assert args.top_k == 10
    assert args.batch_size == 64


def test_eval_run_writes_report(tmp_path, monkeypatch):
    cases_file = tmp_path / "eval-cases.json"
    cases_file.write_text("{}", encoding="utf-8")
    index_file = tmp_path / "index.json"
    index_file.write_text("{}", encoding="utf-8")
    output_file = tmp_path / "eval-report.json"

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")

    args = build_parser().parse_args(
        [
            "eval",
            "run",
            "--cases",
            str(cases_file),
            "--index",
            str(index_file),
            "--output",
            str(output_file),
            "--top-k",
            "5",
            "--batch-size",
            "8",
        ]
    )

    def fake_evaluate_dataset(cases, index, config, *, top_k=10, batch_size=64, llm_judge_config=None, records_path=None, judge_caller=None):
        assert cases == cases_file
        assert index == index_file
        assert config.model == "bge-m3"
        assert top_k == 5
        assert batch_size == 8
        assert llm_judge_config is None
        assert records_path is None
        return {
            "dataset_name": "smoke",
            "case_count": 1,
            "metrics": {"exact_hit_at_1": 1.0},
            "confidence": {"top_1_high_confidence_count": 1},
            "judge_summary": {"enabled": False, "total_ran": 0},
            "results": [],
        }

    with patch("xists.cli.evaluate_dataset", side_effect=fake_evaluate_dataset):
        code = eval_run(args)

    assert code == 0
    report = json.loads(output_file.read_text(encoding="utf-8"))
    assert report["dataset_name"] == "smoke"
    assert report["metrics"]["exact_hit_at_1"] == 1.0


def test_eval_cases_prints_dataset_summary(tmp_path, capsys):
    cases_file = tmp_path / "eval-cases.json"
    cases_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "smoke",
                "cases": [
                    {
                        "id": "api",
                        "query": "python api framework",
                        "expected_repo_id": "fastapi/fastapi",
                        "tags": ["api", "python"],
                    },
                    {
                        "id": "exact",
                        "query": "react",
                        "expected_repo_id": "react/react",
                        "tags": ["frontend", "name"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(["eval", "cases", "--cases", str(cases_file), "--tag", "api"])

    code = eval_cases(args)

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset_name"] == "smoke"
    assert payload["case_count"] == 2
    assert payload["matching_count"] == 1
    assert payload["cases"][0]["id"] == "api"
    assert {item["tag"] for item in payload["tag_counts"]} >= {"api", "python"}


def test_eval_run_parser_supports_judge_flags():
    args = build_parser().parse_args(["eval", "run", "--llm-judge", "--records", "records.json"])

    assert args.llm_judge is True
    assert args.records == Path("records.json")


def test_eval_inspect_parser_uses_default_report_path():
    args = build_parser().parse_args(["eval", "inspect"])

    assert args.report == Path("eval-report.json")
    assert args.status is None
    assert args.limit == 20
    assert args.include_exact is False
    assert args.tag is None
    assert args.query_intent is None


def test_eval_cases_parser_uses_default_path():
    args = build_parser().parse_args(["eval", "cases"])

    assert args.cases == Path("eval-cases.json")
    assert args.tag is None
    assert args.query_intent is None
    assert args.limit == 20


def test_eval_inspect_filters_by_tag_and_query_intent(tmp_path, capsys):
    report_file = tmp_path / "eval-report.json"
    report_file.write_text(
        json.dumps(
            {
                "dataset_name": "smoke",
                "case_count": 2,
                "metrics": {"exact_top1_rate": 0.5, "serious_top1_error_rate": 0.5},
                "confidence": {"wrong_high_confidence_top_1_count": 1},
                "top1_summary": {
                    "top1_miss_count": 1,
                    "top1_miss_acceptable_count": 0,
                    "top1_miss_serious_count": 1,
                    "top1_miss_insufficient_evidence_count": 0,
                },
                "results": [
                    {
                        "id": "ok",
                        "query": "react",
                        "query_intent": {"type": "exact_name"},
                        "tags": ["frontend", "name"],
                        "top1_status": "exact",
                        "expected_repo_id": "react/react",
                        "top_result_repo_id": "react/react",
                    },
                    {
                        "id": "bad",
                        "query": "api framework",
                        "query_intent": {"type": "functional"},
                        "tags": ["api", "backend"],
                        "top1_status": "serious_mismatch",
                        "expected_repo_id": "fastapi/fastapi",
                        "top_result_repo_id": "react/react",
                        "top_result_confidence": "high_confidence",
                        "top_result_why": ["ranked by semantic similarity"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["eval", "inspect", "--report", str(report_file), "--tag", "api", "--query-intent", "functional"])

    code = eval_inspect(args)

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matching_count"] == 1
    assert payload["cases"][0]["id"] == "bad"
    assert payload["filter"]["tag"] == "api"
    assert payload["filter"]["intent"] == "functional"


def test_eval_inspect_prints_filtered_cases(tmp_path, capsys):
    report_file = tmp_path / "eval-report.json"
    report_file.write_text(
        json.dumps(
            {
                "dataset_name": "smoke",
                "case_count": 2,
                "metrics": {"exact_top1_rate": 0.5, "serious_top1_error_rate": 0.5},
                "confidence": {"wrong_high_confidence_top_1_count": 1},
                "top1_summary": {
                    "top1_miss_count": 1,
                    "top1_miss_acceptable_count": 0,
                    "top1_miss_serious_count": 1,
                    "top1_miss_insufficient_evidence_count": 0,
                },
                "results": [
                    {
                        "id": "ok",
                        "query": "react",
                        "top1_status": "exact",
                        "expected_repo_id": "react/react",
                        "top_result_repo_id": "react/react",
                    },
                    {
                        "id": "bad",
                        "query": "api framework",
                        "top1_status": "serious_mismatch",
                        "expected_repo_id": "fastapi/fastapi",
                        "top_result_repo_id": "react/react",
                        "top_result_confidence": "high_confidence",
                        "top_result_why": ["ranked by semantic similarity"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["eval", "inspect", "--report", str(report_file), "--status", "serious_mismatch"])

    code = eval_inspect(args)

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matching_count"] == 1
    assert payload["cases"][0]["id"] == "bad"
    assert payload["cases"][0]["top_result_why"] == ["ranked by semantic similarity"]
    assert "wrong high-confidence: 1 cases" in payload["summary_text"]


def test_doctor_reports_config_and_files_without_secrets(tmp_path, monkeypatch, capsys):
    records_file = tmp_path / "records.json"
    records_file.write_text("[]", encoding="utf-8")
    index_file = tmp_path / "index.json"
    index_file.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:6597/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")

    args = build_parser().parse_args(
        [
            "doctor",
            "--records", str(records_file),
            "--index", str(index_file),
            "--cases", str(tmp_path / "missing-eval-cases.json"),
        ]
    )

    code = doctor(args)

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    statuses = {check["name"]: check["status"] for check in payload["checks"]}
    assert statuses["embedding_config"] == "ok"
    assert statuses["llm_config"] == "ok"
    assert statuses["github_token"] == "ok"
    assert statuses["eval_cases_file"] == "warn"
    serialized = json.dumps(payload)
    assert "embed-secret" not in serialized
    assert "llm-secret" not in serialized
    assert "github-secret" not in serialized



def test_doctor_reports_actionable_next_steps_for_missing_config(tmp_path, monkeypatch, capsys):
    for name in (
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "GITHUB_TOKEN",
        "GITHUB_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    args = build_parser().parse_args(
        [
            "doctor",
            "--records", str(tmp_path / "records.json"),
            "--index", str(tmp_path / "index.json"),
            "--cases", str(tmp_path / "eval-cases.json"),
        ]
    )

    code = doctor(args)

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    checks = {check["name"]: check for check in payload["checks"]}
    assert payload["ok"] is False
    assert checks["embedding_config"]["status"] == "error"
    assert any("EMBEDDING_API_KEY" in step for step in checks["embedding_config"]["next_steps"])
    assert checks["llm_config"]["status"] == "error"
    assert any("LLM_API_KEY" in step for step in checks["llm_config"]["next_steps"])
    assert checks["github_token"]["status"] == "warn"
    assert any("GITHUB_TOKEN" in step for step in checks["github_token"]["next_steps"])
    assert checks["records_file"]["status"] == "warn"
    assert "xists ingest github" in checks["records_file"]["next_steps"][0]
    assert "xists index build" in checks["index_file"]["next_steps"][0]
    assert "examples/eval-cases.json" in checks["eval_cases_file"]["next_steps"][0]

def test_doctor_check_endpoints_reports_embedding_probe(tmp_path, monkeypatch, capsys):
    records_file = tmp_path / "records.json"
    records_file.write_text("[]", encoding="utf-8")
    index_file = tmp_path / "index.json"
    index_file.write_text("{}", encoding="utf-8")
    cases_file = tmp_path / "eval-cases.json"
    cases_file.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:6597/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")

    args = build_parser().parse_args(
        [
            "doctor",
            "--check-endpoints",
            "--records", str(records_file),
            "--index", str(index_file),
            "--cases", str(cases_file),
        ]
    )

    with patch(
        "xists.cli.probe_embedding_endpoint",
        return_value={
            "model": "BAAI/bge-m3",
            "dimension": 1024,
            "resolved_url": "http://localhost:6597/v1/embeddings",
            "response_kind": "openai",
        },
    ) as probe:
        code = doctor(args)

    assert code == 0
    probe.assert_called_once()
    payload = json.loads(capsys.readouterr().out)
    statuses = {check["name"]: check["status"] for check in payload["checks"]}
    assert payload["ok"] is True
    assert statuses["embedding_endpoint"] == "ok"
    endpoint_check = next(check for check in payload["checks"] if check["name"] == "embedding_endpoint")
    assert endpoint_check["dimension"] == 1024
    assert endpoint_check["resolved_url"] == "http://localhost:6597/v1/embeddings"


def test_doctor_strict_fails_when_embedding_probe_fails(tmp_path, monkeypatch, capsys):
    records_file = tmp_path / "records.json"
    records_file.write_text("[]", encoding="utf-8")
    index_file = tmp_path / "index.json"
    index_file.write_text("{}", encoding="utf-8")
    cases_file = tmp_path / "eval-cases.json"
    cases_file.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:6597/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")

    args = build_parser().parse_args(
        [
            "doctor",
            "--strict",
            "--records", str(records_file),
            "--index", str(index_file),
            "--cases", str(cases_file),
        ]
    )

    with patch("xists.cli.probe_embedding_endpoint", side_effect=EmbeddingError("connection refused")):
        code = doctor(args)

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    endpoint_check = next(check for check in payload["checks"] if check["name"] == "embedding_endpoint")
    assert payload["ok"] is False
    assert endpoint_check["status"] == "error"
    assert endpoint_check["message"] == "connection refused"
    assert "EMBEDDING_BASE_URL" in endpoint_check["hint"]
    assert any("EMBEDDING_BASE_URL" in step for step in endpoint_check["next_steps"])
    assert any("doctor --check-endpoints --strict" in step for step in endpoint_check["next_steps"])


def test_index_stats_prints_compact_summary(tmp_path, capsys):
    index_file = tmp_path / "index.json"
    index_file.write_text(
        json.dumps(
            {
                "index_version": 1,
                "embedding_model": "BAAI/bge-m3",
                "embedding_base_url": "http://localhost:6597/v1",
                "embedding_input_version": 1,
                "dimension": 2,
                "built_at": "2026-01-01T00:00:00+00:00",
                "record_count": 2,
                "skipped": ["empty/repo"],
                "vectors": [
                    {
                        "repo_id": "react/react",
                        "embedding_input_fingerprint": "abc",
                        "metadata": {"language": "JavaScript", "topics": ["frontend", "ui"]},
                        "vector": [1.0, 0.0],
                    },
                    {
                        "repo_id": "fastapi/fastapi",
                        "metadata": {"language": "Python", "topics": ["api", "framework"]},
                        "vector": [0.0, 1.0],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(["index", "stats", "--index", str(index_file), "--limit", "1"])

    code = index_stats(args)

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["vector_count"] == 2
    assert payload["skipped_count"] == 1
    assert payload["missing_fingerprint_count"] == 1
    assert payload["top_languages"] == [{"language": "JavaScript", "count": 1}]
    assert "vectors" not in payload


def test_records_inspect_filters_and_summarizes_records(tmp_path, capsys):
    records_file = tmp_path / "records.json"
    records_file.write_text(
        json.dumps(
            [
                _make_record("react/react"),
                _make_record("fastapi/fastapi"),
            ]
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        ["records", "inspect", "--records", str(records_file), "--repo", "fastapi", "--limit", "5"]
    )

    code = records_inspect(args)

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["record_count"] == 2
    assert payload["matching_count"] == 1
    assert payload["items"][0]["repo_id"] == "fastapi/fastapi"
    assert payload["items"][0]["summary"] == "fastapi/fastapi summary"


def _make_record(repo_id: str) -> dict:
    return {
        "repo_id": repo_id,
        "url": f"https://github.com/{repo_id}",
        "github": {"description": f"{repo_id} description", "topics": [], "language": "Python"},
        "llm_profile": {"summary": f"{repo_id} summary", "confidence": "high", "abstained": False},
    }


def test_ingest_github_uses_graphql_batches(tmp_path, monkeypatch):
    repos_file = tmp_path / "repos.txt"
    repos_file.write_text("a/b\nc/d\ne/f\n", encoding="utf-8")

    output_file = tmp_path / "records.json"

    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "http://test/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    batches = []

    def fake_collect(repo_ids, *, token=None):
        batches.append((list(repo_ids), token))
        return [_make_record(repo_id) for repo_id in repo_ids]

    def fake_generate(record, config, *, caller=None):
        return record.get("llm_profile", {})

    args = build_parser().parse_args(
        [
            "ingest", "github",
            "--repos", str(repos_file),
            "--output", str(output_file),
            "--report", str(tmp_path / "report.json"),
            "--github-api", "graphql",
            "--github-batch-size", "2",
        ]
    )

    with patch("xists.cli.collect_records_graphql", side_effect=fake_collect), \
         patch("xists.cli.generate_llm_profile", side_effect=fake_generate):
        code = ingest_github(args)

    assert code == 0
    assert batches == [(["a/b", "c/d"], "tok"), (["e/f"], "tok")]
    saved = json.loads(output_file.read_text())
    assert [r["repo_id"] for r in saved] == ["a/b", "c/d", "e/f"]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["github_api"] == "graphql"
    assert report["github_batch_size"] == 2


def test_ingest_github_graphql_batch_reports_errors(tmp_path, monkeypatch):
    repos_file = tmp_path / "repos.txt"
    repos_file.write_text("a/b\nc/d\n", encoding="utf-8")

    output_file = tmp_path / "records.json"

    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "http://test/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    def fake_collect(repo_ids, *, token=None):
        raise Exception("GraphQL batch failed")

    def fake_collect_one(repo_id, *, token=None):
        raise GitHubAPIError("single repo failed", status=502)

    def fake_generate(record, config, *, caller=None):
        return record.get("llm_profile", {})

    args = build_parser().parse_args(
        [
            "ingest", "github",
            "--repos", str(repos_file),
            "--output", str(output_file),
            "--report", str(tmp_path / "report.json"),
            "--github-api", "graphql",
            "--github-batch-size", "2",
        ]
    )

    with patch("xists.cli.collect_records_graphql", side_effect=fake_collect), \
         patch("xists.cli.collect_record_graphql", side_effect=fake_collect_one), \
         patch("xists.cli.collect_record", side_effect=fake_collect_one), \
         patch("xists.cli.generate_llm_profile", side_effect=fake_generate):
        code = ingest_github(args)

    assert code == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["failed_count"] == 2
    assert {e["repo_id"] for e in report["failed"]} == {"a/b", "c/d"}
    assert all("single repo failed" in e["reason"] for e in report["failed"])


def test_ingest_github_graphql_batch_falls_back_to_single_repo(tmp_path, monkeypatch):
    repos_file = tmp_path / "repos.txt"
    repos_file.write_text("a/b\nc/d\n", encoding="utf-8")

    output_file = tmp_path / "records.json"

    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "http://test/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    def fake_collect_batch(repo_ids, *, token=None):
        raise GitHubAPIError("batch bad gateway", status=502)

    def fake_collect_one(repo_id, *, token=None):
        return _make_record(repo_id)

    def fake_generate(record, config, *, caller=None):
        return record.get("llm_profile", {})

    args = build_parser().parse_args(
        [
            "ingest", "github",
            "--repos", str(repos_file),
            "--output", str(output_file),
            "--report", str(tmp_path / "report.json"),
            "--github-api", "graphql",
            "--github-batch-size", "2",
        ]
    )

    with patch("xists.cli.collect_records_graphql", side_effect=fake_collect_batch), \
         patch("xists.cli.collect_record_graphql", side_effect=fake_collect_one), \
         patch("xists.cli.generate_llm_profile", side_effect=fake_generate):
        code = ingest_github(args)

    assert code == 0
    saved = json.loads(output_file.read_text())
    assert [record["repo_id"] for record in saved] == ["a/b", "c/d"]


def test_ingest_github_rest_falls_back_to_graphql_for_single_repo(tmp_path, monkeypatch):
    repos_file = tmp_path / "repos.txt"
    repos_file.write_text("a/b\n", encoding="utf-8")

    output_file = tmp_path / "records.json"

    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "http://test/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    def fake_collect_rest(repo_id, *, token=None):
        raise GitHubAPIError("rest bad gateway", status=502)

    def fake_collect_graphql(repo_id, *, token=None):
        return _make_record(repo_id)

    def fake_generate(record, config, *, caller=None):
        return record.get("llm_profile", {})

    args = build_parser().parse_args(
        [
            "ingest", "github",
            "--repos", str(repos_file),
            "--output", str(output_file),
            "--report", str(tmp_path / "report.json"),
            "--github-api", "rest",
        ]
    )

    with patch("xists.cli.collect_record", side_effect=fake_collect_rest), \
         patch("xists.cli.collect_record_graphql", side_effect=fake_collect_graphql), \
         patch("xists.cli.generate_llm_profile", side_effect=fake_generate):
        code = ingest_github(args)

    assert code == 0
    saved = json.loads(output_file.read_text())
    assert saved[0]["repo_id"] == "a/b"


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
    assert report["github_api"] == "rest"
    assert report["github_batch_size"] == 1
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


def test_index_build_refreshes_metadata_when_reusing_vector(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "local")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:6597/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    record = {
        "repo_id": "vuejs/core",
        "name": "core",
        "github": {
            "description": "Progressive JavaScript framework.",
            "topics": ["frontend", "vue"],
            "language": "JavaScript",
        },
        "llm_profile": {
            "summary": "Vue builds modern web interfaces.",
            "use_cases": ["building web interfaces"],
            "capabilities": ["reactive components"],
            "search_phrases": ["progressive framework for building modern web interfaces"],
        },
    }
    from xists.search.embed import embedding_input_fingerprint

    records_file = tmp_path / "records.json"
    records_file.write_text(json.dumps([record]), encoding="utf-8")

    output_file = tmp_path / "index.json"
    output_file.write_text(json.dumps({
        "index_version": 1,
        "embedding_model": "BAAI/bge-m3",
        "embedding_base_url": "http://localhost:6597/v1",
        "dimension": 2,
        "built_at": "2026-01-01T00:00:00+00:00",
        "record_count": 1,
        "skipped": [],
        "vectors": [
            {
                "repo_id": "vuejs/core",
                "embedding_input_fingerprint": embedding_input_fingerprint(record),
                "vector": [1.0, 0.0],
            }
        ],
    }), encoding="utf-8")

    def fake_call_embeddings(config, inputs, *, timeout=60):
        raise AssertionError("unchanged vectors should be reused without embedding calls")

    args = build_parser().parse_args(
        ["index", "build", "--records", str(records_file), "--output", str(output_file)]
    )

    with patch("xists.cli.call_embeddings", side_effect=fake_call_embeddings):
        code = index_build(args)

    assert code == 0
    index = json.loads(output_file.read_text())
    assert index["record_count"] == 1
    assert index["vectors"][0]["vector"] == [1.0, 0.0]
    assert index["vectors"][0]["metadata"]["language"] == "JavaScript"
    assert index["vectors"][0]["metadata"]["topics"] == ["frontend", "vue"]
    assert index["vectors"][0]["metadata"]["search_phrases"] == [
        "progressive framework for building modern web interfaces"
    ]


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
