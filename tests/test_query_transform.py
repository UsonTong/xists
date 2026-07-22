import json

import pytest

from xists.profile.llm import LLMResponse
from xists.search.transform import (
    QueryTransformConfig,
    QueryTransformError,
    QueryTransformNotConfiguredError,
    query_transform_config_from_env,
    query_variants,
    transform_queries,
)


def test_query_transform_config_requires_all_values(monkeypatch):
    monkeypatch.delenv("QUERY_TRANSFORM_API_KEY", raising=False)
    monkeypatch.delenv("QUERY_TRANSFORM_BASE_URL", raising=False)
    monkeypatch.delenv("QUERY_TRANSFORM_MODEL", raising=False)

    with pytest.raises(QueryTransformNotConfiguredError, match="QUERY_TRANSFORM_API_KEY"):
        query_transform_config_from_env()


def test_transform_queries_preserves_input_order_and_uses_json_contract():
    config = QueryTransformConfig(api_key="key", base_url="https://example.test/v1", model="test-model")
    observed = {}

    def fake_caller(llm_config, messages, *, timeout):
        observed["config"] = llm_config
        observed["messages"] = messages
        observed["timeout"] = timeout
        return LLMResponse(content=json.dumps({"queries": ["Vue open-source project", "Kafka streaming"]}))

    transformed = transform_queries(config, ["查找 Vue 开源项目", "Kafka 流处理"], caller=fake_caller)

    assert transformed == ["Vue open-source project", "Kafka streaming"]
    assert observed["config"].model == "test-model"
    assert observed["timeout"] == 60
    assert "Preserve repository names" in observed["messages"][0]["content"]
    assert json.loads(observed["messages"][1]["content"])["queries"][0] == "查找 Vue 开源项目"


def test_transform_queries_rejects_invalid_response_shape():
    config = QueryTransformConfig(api_key="key", base_url="https://example.test/v1", model="test-model")

    with pytest.raises(QueryTransformError, match="one query"):
        transform_queries(
            config,
            ["一个查询"],
            caller=lambda *args, **kwargs: LLMResponse(content='{"queries": []}'),
        )


def test_query_variants_keeps_original_and_canonical_expressions_distinct():
    assert query_variants("中文查询", "English query", "off") == ["中文查询"]
    assert query_variants("中文查询", "English query", "canonical") == ["English query"]
    assert query_variants("中文查询", "English query", "merge") == ["中文查询", "English query"]
    assert query_variants("English query", "English query", "merge") == ["English query"]
