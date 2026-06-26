import json

import pytest

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
                "results": [{"repo_id": "react/react", "score": 0.8, "confidence": "high_confidence"}],
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
        "mrr_acceptable": 0.666667,
        "abstain_rate": 0.333333,
        "acceptable_minus_exact_hit_at_1": 0.333333,
        "acceptable_minus_exact_hit_at_k": 0.0,
        "mrr_acceptable_minus_exact": 0.166667,
        "exact_top1_rate": 0.333333,
        "acceptable_top1_rate": 0.0,
        "serious_top1_error_rate": 0.666667,
        "effective_top1_rate": 0.333333,
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
        "top1_miss_acceptable_count": 0,
        "top1_miss_serious_count": 2,
        "top1_miss_acceptable_rate": 0.0,
        "top1_miss_serious_rate": 1.0,
    }
    assert report["results"][0]["top1_status"] == "exact"
    assert report["results"][1]["top1_status"] == "serious_mismatch"
    assert report["results"][1]["exact_match"] is False
    assert report["results"][1]["acceptable_match"] is True
    assert report["results"][1]["exact_rank"] == 2
    assert report["results"][1]["acceptable_rank"] == 1


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
        "top1_miss_acceptable_rate": 1.0,
        "top1_miss_serious_rate": 0.0,
    }
    assert report["results"][0]["top1_status"] == "acceptable"
    assert report["results"][0]["judge_ran"] is True
    assert report["results"][0]["judge_verdict"] == "acceptable_substitute"
    assert report["results"][1]["top1_status"] == "exact"
    assert report["results"][1]["judge_ran"] is False
