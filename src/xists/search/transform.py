"""Optional query canonicalization through a configured chat endpoint."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

from xists.profile.llm import LLMConfig, LLMError, LLMResponse, call_llm

QUERY_TRANSFORM_MODES = ("off", "canonical", "merge")

QUERY_TRANSFORM_SYSTEM_PROMPT = (
    "Convert each search query into a concise English retrieval expression. "
    "Preserve repository names, aliases, programming-language names, versions, "
    "and other technical identifiers exactly. Do not add constraints, examples, "
    "or domain knowledge that are not present in the query. If a query is already "
    "English, preserve its meaning without unnecessary rewriting. Respond with one "
    "JSON object containing exactly a `queries` array of strings in the input order."
)


class QueryTransformError(RuntimeError):
    """Raised when a query transformation request cannot be used."""


class QueryTransformNotConfiguredError(QueryTransformError):
    """Raised when query transformation is requested without configuration."""


@dataclass(frozen=True)
class QueryTransformConfig:
    api_key: str
    base_url: str
    model: str

    @property
    def llm_config(self) -> LLMConfig:
        return LLMConfig(api_key=self.api_key, base_url=self.base_url, model=self.model)


def query_transform_config_from_env() -> QueryTransformConfig:
    """Load the optional query transformation endpoint configuration."""

    api_key = os.environ.get("QUERY_TRANSFORM_API_KEY")
    base_url = os.environ.get("QUERY_TRANSFORM_BASE_URL")
    model = os.environ.get("QUERY_TRANSFORM_MODEL")
    missing = [
        name
        for name, value in (
            ("QUERY_TRANSFORM_API_KEY", api_key),
            ("QUERY_TRANSFORM_BASE_URL", base_url),
            ("QUERY_TRANSFORM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise QueryTransformNotConfiguredError(
            "Query transformation requires a compatible chat endpoint. "
            f"Missing environment variables: {', '.join(missing)}."
        )
    return QueryTransformConfig(api_key=api_key, base_url=base_url, model=model)


def _parse_transformed_queries(content: str, expected_count: int) -> list[str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise QueryTransformError(f"Query transformation response was not valid JSON: {error}") from error
    values = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(values, list) or len(values) != expected_count:
        raise QueryTransformError("Query transformation response must contain one query for every input")
    transformed = [value.strip() if isinstance(value, str) else "" for value in values]
    if any(not value for value in transformed):
        raise QueryTransformError("Query transformation response contains an empty query")
    return transformed


def transform_queries(
    config: QueryTransformConfig,
    queries: list[str],
    *,
    caller: Callable[..., LLMResponse] = call_llm,
) -> list[str]:
    """Return one English canonical retrieval expression for each input query."""

    if not queries:
        return []
    messages = [
        {"role": "system", "content": QUERY_TRANSFORM_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"queries": queries}, ensure_ascii=False)},
    ]
    try:
        response = caller(config.llm_config, messages, timeout=60)
    except LLMError as error:
        raise QueryTransformError(str(error)) from error
    return _parse_transformed_queries(response.content, len(queries))


def query_variants(query: str, canonical_query: str, mode: str) -> list[str]:
    """Build embedding inputs for an optional canonical query representation."""

    if mode not in QUERY_TRANSFORM_MODES:
        raise ValueError(f"Unknown query transform mode: {mode}")
    if mode == "off" or canonical_query == query:
        return [query]
    if mode == "canonical":
        return [canonical_query]
    return [query, canonical_query]
