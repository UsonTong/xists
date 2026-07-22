import json

import pytest

from xists.eval.inspect import inspect_report
from xists.eval.judge import parse_judge_response
from xists.eval.run import evaluate_dataset
from xists.eval.schema import EvaluationDatasetError, normalize_dataset
from xists.profile.llm import LLMConfig, LLMResponse
from xists.search.embed import EmbeddingConfig

CONFIG = EmbeddingConfig(api_key="k", base_url="http://localhost/v1", model="bge-m3")
LLM_CONFIG = LLMConfig(api_key="lk", base_url="http://localhost/v1", model="judge-model")


def test_normalize_dataset_expands_acceptable_sets():
    dataset = normalize_dataset(
        {
            "schema_version": 1,
            "dataset_name": "smoke",
            "families": {"react-family": ["react/react", "preactjs/preact"]},
            "cases": [
                {
                    "id": "case-1",
                    "query": "frontend ui library",
                    "expected_repo_id": "react/react",
                    "acceptable_repo_ids": ["facebook/react"],
                    "acceptable_families": ["react-family"],
                    "tags": ["frontend"],
                }
            ],
        }
    )

    assert dataset["cases"][0]["acceptable_set"] == ["facebook/react", "preactjs/preact", "react/react"]


def test_normalize_dataset_accepts_acceptable_alias_and_keeps_old_cases_compatible():
    dataset = normalize_dataset(
        {
            "schema_version": 1,
            "dataset_name": "acceptable-alias",
            "cases": [
                {
                    "id": "alias",
                    "query": "组件库",
                    "expected_repo_id": "react/react",
                    "acceptable": ["preactjs/preact"],
                },
                {
                    "id": "legacy",
                    "query": "frontend ui library",
                    "expected_repo_id": "vuejs/core",
                },
            ],
        }
    )

    assert dataset["cases"][0]["acceptable"] == ["preactjs/preact"]
    assert dataset["cases"][0]["acceptable_set"] == ["preactjs/preact", "react/react"]
    assert dataset["cases"][1]["acceptable"] == []
    assert dataset["cases"][1]["acceptable_set"] == ["vuejs/core"]


def test_normalize_dataset_rejects_unknown_family():
    with pytest.raises(EvaluationDatasetError):
        normalize_dataset(
            {
                "schema_version": 1,
                "dataset_name": "smoke",
                "cases": [
                    {
                        "id": "case-1",
                        "query": "frontend ui library",
                        "expected_repo_id": "react/react",
                        "acceptable_families": ["missing-family"],
                    }
                ],
            }
        )


def test_parse_judge_response_normalizes_fields():
    parsed = parse_judge_response(
        json.dumps(
            {
                "verdict": "acceptable_substitute",
                "difference_size": "small",
                "query_specificity": "underspecified",
                "language_ecosystem_material": False,
                "reason_short": "Query is broad and both repos fit.",
                "expected_advantages": ["Python ecosystem"],
                "top1_advantages": ["API-first platform"],
                "confidence": "high",
            }
        )
    )

    assert parsed["verdict"] == "acceptable_substitute"
    assert parsed["difference_size"] == "small"
    assert parsed["query_specificity"] == "underspecified"
    assert parsed["reason_short"] == "Query is broad and both repos fit."


def test_evaluate_dataset_reports_exact_and_top1_status_metrics(tmp_path):
    cases_file = tmp_path / "eval-cases.json"
    cases_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "smoke",
                "families": {"react-family": ["preactjs/preact"]},
                "cases": [
                    {
                        "id": "exact-top-1",
                        "query": "frontend ui library",
                        "expected_repo_id": "react/react",
                        "tags": ["frontend"],
                    },
                    {
                        "id": "acceptable-family",
                        "query": "react-like ui library",
                        "expected_repo_id": "react/react",
                        "acceptable_families": ["react-family"],
                    },
                    {
                        "id": "abstain",
                        "query": "unknown niche thing",
                        "expected_repo_id": "unknown/repo",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    index_file = tmp_path / "index.json"
    index_file.write_text(
        json.dumps(
            {
                "embedding_model": "bge-m3",
                "dimension": 2,
                "vectors": [],
            }
        ),
        encoding="utf-8",
    )

    def fake_rank_many(queries, index, config, *, top_k=10, batch_size=64):
        assert queries == ["frontend ui library", "react-like ui library", "unknown niche thing"]
        return [
            {
                "query": queries[0],
                "abstained": False,
                "query_intent": {"type": "functional"},
                "results": [
                    {
                        "repo_id": "react/react",
                        "score": 0.8,
                        "confidence": "high_confidence",
                        "why": ["matched topic: frontend"],
                        "matched_terms": ["frontend"],
                        "score_breakdown": {"semantic": 0.7, "metadata": 0.1, "final": 0.8},
                        "rerank_score": 2.4,
                        "ranking_evidence": {"semantic_rank": 1, "rerank_rank": 2},
                        "confidence_evidence": {
                            "version": "evidence-v1",
                            "mode": "evidence-v1",
                            "downgrade_reasons": [],
                        },
                        "diagnostics": {
                            "query_terms": ["frontend"],
                            "matched_terms": ["frontend"],
                            "topic_matches": ["frontend"],
                            "capability_terms": [],
                            "type_cue_matches": [],
                            "entity_match": None,
                            "language_match": None,
                            "phrase_match": None,
                        },
                    }
                ],
                "considered": 3,
            },
            {
                "query": queries[1],
                "abstained": False,
                "results": [
                    {"repo_id": "preactjs/preact", "score": 0.7, "confidence": "high_confidence"},
                    {"repo_id": "react/react", "score": 0.6, "confidence": "exploratory"},
                ],
                "considered": 3,
            },
            {
                "query": queries[2],
                "abstained": True,
                "results": [],
                "considered": 3,
            },
        ]

    report = evaluate_dataset(cases_file, index_file, CONFIG, embed_many=fake_rank_many)

    assert report["case_count"] == 3
    assert report["metrics"] == {
        "exact_hit_at_1": 0.333333,
        "exact_hit_at_k": 0.666667,
        "mrr_exact": 0.5,
        "acceptable_hit_at_1": 0.666667,
        "acceptable_hit_at_k": 0.666667,
        "recall_at_1": 0.666667,
        "recall_at_5": 0.666667,
        "mrr_acceptable": 0.666667,
        "abstain_rate": 0.333333,
        "acceptable_minus_exact_hit_at_1": 0.333333,
        "acceptable_minus_exact_hit_at_k": 0.0,
        "mrr_acceptable_minus_exact": 0.166667,
        "exact_top1_rate": 0.333333,
        "acceptable_top1_rate": 0.333333,
        "serious_top1_error_rate": 0.333333,
        "insufficient_evidence_top1_rate": 0.0,
        "effective_top1_rate": 0.666667,
    }
    assert report["confidence"] == {
        "top_1_high_confidence_count": 2,
        "top_1_exploratory_count": 0,
        "top_1_missing_count": 1,
        "wrong_high_confidence_top_1_count": 0,
    }
    assert report["judge_summary"] == {
        "enabled": False,
        "model": None,
        "prompt_version": None,
        "total_ran": 0,
        "acceptable_substitute_count": 0,
        "serious_mismatch_count": 0,
        "insufficient_evidence_count": 0,
        "small_difference_count": 0,
        "moderate_difference_count": 0,
        "large_difference_count": 0,
    }
    assert report["top1_summary"] == {
        "top1_miss_count": 2,
        "top1_miss_acceptable_count": 1,
        "top1_miss_serious_count": 1,
        "top1_miss_insufficient_evidence_count": 0,
        "top1_miss_acceptable_rate": 0.5,
        "top1_miss_serious_rate": 0.5,
        "top1_miss_insufficient_evidence_rate": 0.0,
    }
    assert report["summary"]["exact_top_1"] == {"count": 1, "rate": 0.333333}
    assert report["summary"]["acceptable_top_1"] == {
        "count": 2,
        "rate": 0.666667,
        "description": "top-1 was exact or an acceptable repo/family/judge substitute",
    }
    assert report["summary"]["acceptable_substitute_top_1"]["count"] == 1
    assert report["summary"]["effective_top_1"]["count"] == 2
    assert "exact top-1: 33.3% (1/3)" in report["summary_text"]
    assert "recall@1: 66.7% (2/3)" in report["summary_text"]
    assert "recall@5: 66.7% (2/3)" in report["summary_text"]
    assert "acceptable top-1: 66.7% (2/3)" in report["summary_text"]
    assert report["top_misses"][0]["id"] == "abstain"
    assert report["top_misses"][0]["top1_status"] == "serious_mismatch"
    assert report["results"][0]["query_intent"] == {"type": "functional"}
    assert report["results"][0]["latency_ms"] is None
    assert report["results"][0]["top_result_why"] == ["matched topic: frontend"]
    assert report["results"][0]["top_result_matched_terms"] == ["frontend"]
    assert report["results"][0]["top_result_diagnostics"]["topic_matches"] == ["frontend"]
    assert report["results"][0]["top_result_score_breakdown"]["final"] == 0.8
    assert report["results"][0]["top_result_rerank_score"] == 2.4
    assert report["results"][0]["top_result_ranking_evidence"] == {
        "semantic_rank": 1,
        "rerank_rank": 2,
    }
    assert report["results"][0]["top_result_confidence_evidence"] == {
        "version": "evidence-v1",
        "mode": "evidence-v1",
        "downgrade_reasons": [],
    }
    assert report["confidence_calibration"] == "off"
    assert report["results"][0]["top1_status"] == "exact"
    assert report["results"][1]["top1_status"] == "acceptable"
    assert report["results"][1]["exact_match"] is False
    assert report["results"][1]["acceptable_match"] is True
    assert report["results"][1]["exact_rank"] == 2
    assert report["results"][1]["acceptable_rank"] == 1

    inspection = inspect_report(report, status="serious_mismatch", limit=5)
    assert inspection["summary"]["case_count"] == 3
    assert inspection["matching_count"] == 1
    assert inspection["cases"][0]["id"] == "abstain"

    tagged = inspect_report(report, tag="frontend", include_exact=True, limit=5)
    assert tagged["filter"]["tag"] == "frontend"
    assert tagged["matching_count"] == 1
    assert tagged["cases"][0]["id"] == "exact-top-1"
    assert tagged["cases"][0]["top_result_diagnostics"]["topic_matches"] == ["frontend"]
    assert tagged["cases"][0]["top_result_matched_terms"] == ["frontend"]

    functional = inspect_report(report, intent="functional", include_exact=True, limit=5)
    assert functional["filter"]["intent"] == "functional"
    assert functional["matching_count"] == 1
    assert functional["cases"][0]["id"] == "exact-top-1"


def test_evaluate_dataset_reports_recall_at_k_for_acceptable_and_chinese_cases(tmp_path):
    cases_file = tmp_path / "eval-cases.json"
    cases_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "recall-smoke",
                "cases": [
                    {"id": "exact", "query": "react", "expected_repo_id": "react/react"},
                    {
                        "id": "acceptable",
                        "query": "轻量级 React 替代品",
                        "expected_repo_id": "react/react",
                        "acceptable": ["preactjs/preact"],
                    },
                    {"id": "rank-five", "query": "Python API framework", "expected_repo_id": "tiangolo/fastapi"},
                    {"id": "miss", "query": "database", "expected_repo_id": "postgres/postgres"},
                ],
            }
        ),
        encoding="utf-8",
    )
    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps({"embedding_model": "bge-m3", "dimension": 2, "vectors": []}), encoding="utf-8")

    def fake_rank_many(queries, index, config, *, top_k=10, batch_size=64):
        assert queries[1] == "轻量级 React 替代品"
        return [
            {"results": [{"repo_id": "react/react"}]},
            {"results": [{"repo_id": "preactjs/preact"}]},
            {"results": [{"repo_id": "a/a"}, {"repo_id": "b/b"}, {"repo_id": "c/c"}, {"repo_id": "d/d"}, {"repo_id": "tiangolo/fastapi"}]},
            {"results": [{"repo_id": "mysql/mysql-server"}]},
        ]

    report = evaluate_dataset(cases_file, index_file, CONFIG, embed_many=fake_rank_many)

    assert report["metrics"]["recall_at_1"] == 0.5
    assert report["metrics"]["recall_at_5"] == 0.75
    assert report["results"][1]["acceptable_match"] is True
    assert report["results"][2]["acceptable_rank"] == 5
    assert report["summary"]["recall_at_1"]["count"] == 2
    assert report["summary"]["recall_at_5"]["count"] == 3
    assert "recall@5: 75.0% (3/4)" in report["summary_text"]


def test_evaluate_dataset_with_llm_judge_keeps_hard_metrics_and_adds_analysis(tmp_path):
    cases_file = tmp_path / "eval-cases.json"
    cases_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "judge-smoke",
                "cases": [
                    {
                        "id": "mismatch",
                        "query": "web framework for building APIs",
                        "expected_repo_id": "fastapi/fastapi",
                    },
                    {
                        "id": "exact",
                        "query": "python micro framework for building web applications",
                        "expected_repo_id": "pallets/flask",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps({"embedding_model": "bge-m3", "dimension": 2, "vectors": []}), encoding="utf-8")

    records_file = tmp_path / "records.json"
    records_file.write_text(
        json.dumps(
            [
                {
                    "repo_id": "fastapi/fastapi",
                    "url": "https://github.com/fastapi/fastapi",
                    "github": {"description": "Python web framework for building APIs", "topics": [], "language": "Python"},
                    "readme": {"excerpt": "FastAPI framework"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
                {
                    "repo_id": "api-platform/api-platform",
                    "url": "https://github.com/api-platform/api-platform",
                    "github": {"description": "API platform for web APIs", "topics": [], "language": "PHP"},
                    "readme": {"excerpt": "API Platform framework"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
                {
                    "repo_id": "pallets/flask",
                    "url": "https://github.com/pallets/flask",
                    "github": {"description": "Python micro framework for building web applications", "topics": [], "language": "Python"},
                    "readme": {"excerpt": "Flask framework"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    def fake_rank_many(queries, index, config, *, top_k=10, batch_size=64):
        return [
            {
                "query": queries[0],
                "abstained": False,
                "results": [
                    {"repo_id": "api-platform/api-platform", "score": 0.8, "confidence": "high_confidence"},
                    {"repo_id": "fastapi/fastapi", "score": 0.7, "confidence": "high_confidence"},
                ],
                "considered": 2,
            },
            {
                "query": queries[1],
                "abstained": False,
                "results": [{"repo_id": "pallets/flask", "score": 0.9, "confidence": "high_confidence"}],
                "considered": 2,
            },
        ]

    def fake_judge_caller(config, messages):
        return LLMResponse(
            content=json.dumps(
                {
                    "verdict": "acceptable_substitute",
                    "difference_size": "small",
                    "query_specificity": "underspecified",
                    "language_ecosystem_material": False,
                    "reason_short": "The query is broad and both repos satisfy the API framework need.",
                    "expected_advantages": ["Python ecosystem"],
                    "top1_advantages": ["API-first platform"],
                    "confidence": "high",
                }
            ),
            token_usage={"total_tokens": 123},
        )

    report = evaluate_dataset(
        cases_file,
        index_file,
        CONFIG,
        embed_many=fake_rank_many,
        llm_judge_config=LLM_CONFIG,
        records_path=records_file,
        judge_caller=fake_judge_caller,
    )

    assert report["metrics"]["exact_hit_at_1"] == 0.5
    assert report["metrics"]["exact_hit_at_k"] == 1.0
    assert report["metrics"]["acceptable_top1_rate"] == 0.5
    assert report["metrics"]["serious_top1_error_rate"] == 0.0
    assert report["metrics"]["insufficient_evidence_top1_rate"] == 0.0
    assert report["metrics"]["effective_top1_rate"] == 1.0
    assert report["judge_summary"]["enabled"] is True
    assert report["judge_summary"]["prompt_version"] == 3
    assert report["judge_summary"]["total_ran"] == 1
    assert report["judge_summary"]["acceptable_substitute_count"] == 1
    assert report["judge_summary"]["small_difference_count"] == 1
    assert report["top1_summary"] == {
        "top1_miss_count": 1,
        "top1_miss_acceptable_count": 1,
        "top1_miss_serious_count": 0,
        "top1_miss_insufficient_evidence_count": 0,
        "top1_miss_acceptable_rate": 1.0,
        "top1_miss_serious_rate": 0.0,
        "top1_miss_insufficient_evidence_rate": 0.0,
    }


def test_evaluate_dataset_tracks_insufficient_evidence_top1_bucket(tmp_path):
    cases_file = tmp_path / "eval-cases.json"
    cases_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "judge-insufficient",
                "cases": [
                    {
                        "id": "mismatch",
                        "query": "lightweight api server",
                        "expected_repo_id": "tiangolo/fastapi",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps({"embedding_model": "bge-m3", "dimension": 2, "vectors": []}), encoding="utf-8")

    records_file = tmp_path / "records.json"
    records_file.write_text(
        json.dumps(
            [
                {
                    "repo_id": "tiangolo/fastapi",
                    "url": "https://github.com/tiangolo/fastapi",
                    "github": {"description": "Fast API framework", "topics": [], "language": "Python"},
                    "readme": {"excerpt": "FastAPI framework"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
                {
                    "repo_id": "some/other-api",
                    "url": "https://github.com/some/other-api",
                    "github": {"description": "Another API server", "topics": [], "language": "Python"},
                    "readme": {"excerpt": "Another API framework"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    def fake_rank_many(queries, index, config, *, top_k=10, batch_size=64):
        return [
            {
                "query": queries[0],
                "abstained": False,
                "results": [{"repo_id": "some/other-api", "score": 0.8, "confidence": "high_confidence"}],
                "considered": 2,
            }
        ]

    def fake_judge_caller(config, messages):
        return LLMResponse(
            content=json.dumps(
                {
                    "verdict": "insufficient_evidence",
                    "difference_size": "moderate",
                    "query_specificity": "underspecified",
                    "language_ecosystem_material": False,
                    "reason_short": "Not enough evidence to compare fairly.",
                    "expected_advantages": [],
                    "top1_advantages": [],
                    "confidence": "medium",
                }
            ),
            token_usage={"total_tokens": 42},
        )

    report = evaluate_dataset(
        cases_file,
        index_file,
        CONFIG,
        embed_many=fake_rank_many,
        llm_judge_config=LLM_CONFIG,
        records_path=records_file,
        judge_caller=fake_judge_caller,
    )

    assert report["metrics"]["exact_top1_rate"] == 0.0
    assert report["metrics"]["acceptable_top1_rate"] == 0.0
    assert report["metrics"]["serious_top1_error_rate"] == 0.0
    assert report["metrics"]["insufficient_evidence_top1_rate"] == 1.0
    assert report["metrics"]["effective_top1_rate"] == 0.0
    assert report["judge_summary"]["insufficient_evidence_count"] == 1
    assert report["top1_summary"] == {
        "top1_miss_count": 1,
        "top1_miss_acceptable_count": 0,
        "top1_miss_serious_count": 0,
        "top1_miss_insufficient_evidence_count": 1,
        "top1_miss_acceptable_rate": 0.0,
        "top1_miss_serious_rate": 0.0,
        "top1_miss_insufficient_evidence_rate": 1.0,
    }
    assert report["results"][0]["top1_status"] == "insufficient_evidence"
    assert report["results"][0]["judge_ran"] is True
    assert report["results"][0]["judge_verdict"] == "insufficient_evidence"


def test_evaluate_dataset_partitions_top1_miss_buckets(tmp_path):
    cases_file = tmp_path / "eval-cases.json"
    cases_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "mixed-top1-statuses",
                "families": {"react-family": ["preactjs/preact"]},
                "cases": [
                    {
                        "id": "exact",
                        "query": "react ui library",
                        "expected_repo_id": "react/react",
                        "tags": ["frontend"],
                    },
                    {
                        "id": "dataset-acceptable",
                        "query": "react-like ui library",
                        "expected_repo_id": "react/react",
                        "acceptable_families": ["react-family"],
                    },
                    {
                        "id": "judge-acceptable",
                        "query": "broad api framework",
                        "expected_repo_id": "fastapi/fastapi",
                    },
                    {
                        "id": "judge-serious",
                        "query": "static site generator",
                        "expected_repo_id": "vercel/next.js",
                    },
                    {
                        "id": "judge-insufficient",
                        "query": "data workflow orchestration",
                        "expected_repo_id": "apache/airflow",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps({"embedding_model": "bge-m3", "dimension": 2, "vectors": []}), encoding="utf-8")

    records_file = tmp_path / "records.json"
    records_file.write_text(
        json.dumps(
            [
                {
                    "repo_id": "react/react",
                    "url": "https://github.com/react/react",
                    "github": {"description": "React UI library", "topics": [], "language": "JavaScript"},
                    "readme": {"excerpt": "React"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
                {
                    "repo_id": "preactjs/preact",
                    "url": "https://github.com/preactjs/preact",
                    "github": {"description": "Preact UI library", "topics": [], "language": "JavaScript"},
                    "readme": {"excerpt": "Preact"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
                {
                    "repo_id": "fastapi/fastapi",
                    "url": "https://github.com/fastapi/fastapi",
                    "github": {"description": "Python API framework", "topics": [], "language": "Python"},
                    "readme": {"excerpt": "FastAPI"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
                {
                    "repo_id": "api-platform/api-platform",
                    "url": "https://github.com/api-platform/api-platform",
                    "github": {"description": "API platform", "topics": [], "language": "PHP"},
                    "readme": {"excerpt": "API Platform"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
                {
                    "repo_id": "vercel/next.js",
                    "url": "https://github.com/vercel/next.js",
                    "github": {"description": "React framework", "topics": [], "language": "JavaScript"},
                    "readme": {"excerpt": "Next.js"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
                {
                    "repo_id": "some/random-repo",
                    "url": "https://github.com/some/random-repo",
                    "github": {"description": "Unrelated repo", "topics": [], "language": "Rust"},
                    "readme": {"excerpt": "Random"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
                {
                    "repo_id": "apache/airflow",
                    "url": "https://github.com/apache/airflow",
                    "github": {"description": "Workflow orchestration", "topics": [], "language": "Python"},
                    "readme": {"excerpt": "Airflow"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
                {
                    "repo_id": "unknown/orchestrator",
                    "url": "https://github.com/unknown/orchestrator",
                    "github": {"description": "Another orchestrator", "topics": [], "language": "Go"},
                    "readme": {"excerpt": "Unknown orchestrator"},
                    "structure": {"signals": ["has_readme"]},
                    "evidence": [{"kind": "github_description"}],
                    "evidence_gaps": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    def fake_rank_many(queries, index, config, *, top_k=10, batch_size=64):
        return [
            {
                "query": queries[0],
                "abstained": False,
                "results": [{"repo_id": "react/react", "score": 0.9, "confidence": "high_confidence"}],
                "considered": 3,
            },
            {
                "query": queries[1],
                "abstained": False,
                "results": [{"repo_id": "preactjs/preact", "score": 0.8, "confidence": "high_confidence"}],
                "considered": 3,
            },
            {
                "query": queries[2],
                "abstained": False,
                "results": [{"repo_id": "api-platform/api-platform", "score": 0.8, "confidence": "high_confidence"}],
                "considered": 3,
            },
            {
                "query": queries[3],
                "abstained": False,
                "results": [{"repo_id": "some/random-repo", "score": 0.8, "confidence": "high_confidence"}],
                "considered": 3,
            },
            {
                "query": queries[4],
                "abstained": False,
                "results": [{"repo_id": "unknown/orchestrator", "score": 0.8, "confidence": "high_confidence"}],
                "considered": 3,
            },
        ]

    verdicts = iter(
        [
            {
                "verdict": "acceptable_substitute",
                "difference_size": "small",
                "query_specificity": "underspecified",
                "language_ecosystem_material": False,
                "reason_short": "Broad query, acceptable substitute.",
                "expected_advantages": [],
                "top1_advantages": [],
                "confidence": "high",
            },
            {
                "verdict": "serious_mismatch",
                "difference_size": "large",
                "query_specificity": "specified",
                "language_ecosystem_material": True,
                "reason_short": "Misses important constraints.",
                "expected_advantages": [],
                "top1_advantages": [],
                "confidence": "high",
            },
            {
                "verdict": "insufficient_evidence",
                "difference_size": "moderate",
                "query_specificity": "underspecified",
                "language_ecosystem_material": False,
                "reason_short": "Not enough evidence.",
                "expected_advantages": [],
                "top1_advantages": [],
                "confidence": "medium",
            },
        ]
    )

    def fake_judge_caller(config, messages):
        return LLMResponse(content=json.dumps(next(verdicts)), token_usage={"total_tokens": 64})

    report = evaluate_dataset(
        cases_file,
        index_file,
        CONFIG,
        embed_many=fake_rank_many,
        llm_judge_config=LLM_CONFIG,
        records_path=records_file,
        judge_caller=fake_judge_caller,
    )

    assert report["judge_summary"]["total_ran"] == 3
    assert report["judge_summary"]["acceptable_substitute_count"] == 1
    assert report["judge_summary"]["serious_mismatch_count"] == 1
    assert report["judge_summary"]["insufficient_evidence_count"] == 1
    assert report["metrics"]["exact_top1_rate"] == 0.2
    assert report["metrics"]["acceptable_top1_rate"] == 0.4
    assert report["metrics"]["serious_top1_error_rate"] == 0.2
    assert report["metrics"]["insufficient_evidence_top1_rate"] == 0.2
    assert report["metrics"]["effective_top1_rate"] == 0.6
    assert report["top1_summary"] == {
        "top1_miss_count": 4,
        "top1_miss_acceptable_count": 2,
        "top1_miss_serious_count": 1,
        "top1_miss_insufficient_evidence_count": 1,
        "top1_miss_acceptable_rate": 0.5,
        "top1_miss_serious_rate": 0.25,
        "top1_miss_insufficient_evidence_rate": 0.25,
    }
    assert [result["top1_status"] for result in report["results"]] == [
        "exact",
        "acceptable",
        "acceptable",
        "serious_mismatch",
        "insufficient_evidence",
    ]
