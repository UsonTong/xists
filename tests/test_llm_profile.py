import json

import pytest

from xists.profile.llm import (
    LLMConfig,
    LLMError,
    LLMNotConfiguredError,
    LLMResponse,
    PROFILE_PROMPT_VERSION,
    attach_llm_profile,
    build_profile_messages,
    generate_llm_profile,
    input_evidence_kinds,
    llm_config_from_env,
    parse_llm_profile_response,
    profile_input_from_record,
    profile_prompt_hash,
)


def make_record():
    return {
        "repo_id": "react/react",
        "url": "https://github.com/react/react",
        "github": {
            "description": "The library for web and native user interfaces.",
            "topics": ["javascript", "ui"],
            "language": "JavaScript",
        },
        "readme": {"excerpt": "React is a JavaScript library for building user interfaces."},
        "structure": {"signals": ["has_package_json", "has_tests"]},
        "evidence": [
            {"kind": "github_description"},
            {"kind": "github_topics"},
            {"kind": "readme_excerpt"},
            {"kind": "structure_signals"},
        ],
        "evidence_gaps": [],
    }


def test_llm_config_from_env_requires_all_values(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(LLMNotConfiguredError):
        llm_config_from_env()


def test_llm_config_from_env_builds_config(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    config = llm_config_from_env()

    assert config.api_key == "key"
    assert config.base_url == "https://api.example.com/v1"
    assert config.model == "test-model"
    assert config.chat_completions_url == "https://api.example.com/v1/chat/completions"


def test_profile_input_only_includes_collected_evidence():
    profile_input = profile_input_from_record(make_record())

    assert profile_input["repo_id"] == "react/react"
    assert profile_input["github_description"].startswith("The library")
    assert profile_input["github_topics"] == ["javascript", "ui"]
    assert profile_input["readme_excerpt"].startswith("React is")
    assert profile_input["structure_signals"] == ["has_package_json", "has_tests"]
    assert "stars" not in profile_input


def test_input_evidence_kinds():
    assert input_evidence_kinds(make_record()) == [
        "github_description",
        "github_topics",
        "readme_excerpt",
        "structure_signals",
    ]


def test_build_profile_messages_has_system_and_user():
    messages = build_profile_messages(profile_input_from_record(make_record()))
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "react/react" in messages[1]["content"]


def test_parse_llm_profile_response_normalizes():
    content = json.dumps(
        {
            "summary": "  React is a UI library.  ",
            "use_cases": ["building UIs", "", "  components  "],
            "capabilities": "not a list",
            "not_for": ["backend services"],
            "search_phrases": ["frontend library"],
            "confidence": "HIGH",
            "abstained": False,
        }
    )

    profile = parse_llm_profile_response(content)

    assert profile["summary"] == "React is a UI library."
    assert profile["use_cases"] == ["building UIs", "components"]
    assert profile["capabilities"] == []
    assert profile["confidence"] == "high"
    assert profile["abstained"] is False


def test_parse_llm_profile_response_strips_code_fence():
    content = "```json\n{\"summary\": \"x\", \"confidence\": \"low\", \"abstained\": true}\n```"
    profile = parse_llm_profile_response(content)
    assert profile["summary"] == "x"
    assert profile["abstained"] is True


def test_parse_llm_profile_response_invalid_confidence_defaults_low():
    content = json.dumps({"summary": "x", "confidence": "great"})
    assert parse_llm_profile_response(content)["confidence"] == "low"


def test_parse_llm_profile_response_rejects_non_json():
    with pytest.raises(LLMError):
        parse_llm_profile_response("not json")


def test_parse_llm_profile_response_rejects_non_object():
    with pytest.raises(LLMError):
        parse_llm_profile_response("[1, 2, 3]")


def test_generate_llm_profile_adds_provenance():
    config = LLMConfig(api_key="key", base_url="https://api.example.com/v1", model="test-model")

    def fake_caller(cfg, messages):
        return json.dumps(
            {
                "summary": "React is a UI library.",
                "use_cases": ["building UIs"],
                "capabilities": ["declarative rendering"],
                "not_for": ["backend services"],
                "search_phrases": ["frontend library"],
                "confidence": "high",
                "abstained": False,
            }
        )

    profile = generate_llm_profile(make_record(), config, caller=fake_caller)

    assert profile["provider"] == "openai_compatible"
    assert profile["model"] == "test-model"
    assert "base_url" not in profile
    assert profile["generated_at"].endswith("+00:00")
    assert profile["prompt_version"] == PROFILE_PROMPT_VERSION
    assert profile["prompt_hash"] == profile_prompt_hash()
    assert profile["duration_seconds"] >= 0
    assert profile["token_usage"] is None
    assert profile["input_evidence_kinds"] == [
        "github_description",
        "github_topics",
        "readme_excerpt",
        "structure_signals",
    ]
    assert profile["summary"] == "React is a UI library."


def test_generate_llm_profile_records_token_usage():
    config = LLMConfig(api_key="key", base_url="https://api.example.com/v1", model="test-model")

    def fake_caller(cfg, messages):
        return LLMResponse(
            content=json.dumps(
                {
                    "summary": "React is a UI library.",
                    "confidence": "high",
                    "abstained": False,
                }
            ),
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    profile = generate_llm_profile(make_record(), config, caller=fake_caller)

    assert profile["token_usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_attach_llm_profile():
    record = make_record()
    attach_llm_profile(record, {"summary": "x"})
    assert record["llm_profile"] == {"summary": "x"}
