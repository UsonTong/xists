"""Cross-encoder reranking through a configured HTTP endpoint."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


USER_AGENT = "xists-reranker"


class RerankerError(RuntimeError):
    """Raised when a reranking request cannot be used."""


class RerankerNotConfiguredError(RerankerError):
    """Raised when no reranker endpoint is configured."""


@dataclass(frozen=True)
class RerankerConfig:
    api_key: str | None
    base_url: str
    model: str | None = None
    protocol: str = "tei"

    @property
    def rerank_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if self.protocol == "passages":
            return base_url
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return f"{base_url}/rerank"


def reranker_config_from_env() -> RerankerConfig:
    base_url = os.environ.get("RERANKER_BASE_URL")
    if not base_url:
        raise RerankerNotConfiguredError(
            "Reranking requires RERANKER_BASE_URL. Configure a compatible reranking endpoint."
        )
    return RerankerConfig(
        api_key=os.environ.get("RERANKER_API_KEY"),
        base_url=base_url,
        model=os.environ.get("RERANKER_MODEL"),
        protocol=os.environ.get("RERANKER_PROTOCOL", "tei"),
    )


def rerank_text_from_entry(entry: dict[str, Any]) -> str:
    """Build a generic evidence document for one indexed repository."""

    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    parts: list[str] = []
    repo_id = entry.get("repo_id")
    if isinstance(repo_id, str) and repo_id.strip():
        parts.append(repo_id.strip())
    for key in ("name", "description", "summary", "search_text", "language", "project_type"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for key in ("topics", "use_cases", "capabilities", "ecosystem", "search_phrases"):
        values = metadata.get(key)
        if isinstance(values, list):
            parts.extend(str(value).strip() for value in values if isinstance(value, str) and value.strip())
    return "\n".join(parts)


def _request_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_tei_scores(data: Any, expected_count: int) -> list[float]:
    if not isinstance(data, list) or len(data) != expected_count:
        raise RerankerError(f"Unexpected reranker response shape: {data!r}")
    scores: list[float | None] = [None] * expected_count
    for item in data:
        if not isinstance(item, dict):
            raise RerankerError(f"Unexpected reranker result: {item!r}")
        index = item.get("index")
        score = item.get("score")
        if not isinstance(index, int) or not 0 <= index < expected_count or not isinstance(score, (int, float)):
            raise RerankerError(f"Unexpected reranker result: {item!r}")
        if scores[index] is not None:
            raise RerankerError(f"Duplicate reranker result index: {index}")
        scores[index] = float(score)
    if any(score is None for score in scores):
        raise RerankerError("Reranker response omitted one or more candidate scores")
    return [float(score) for score in scores]


def _parse_passage_scores(data: Any, expected_count: int) -> list[float]:
    if not isinstance(data, dict) or not isinstance(data.get("rankings"), list):
        raise RerankerError(f"Unexpected reranker response shape: {data!r}")
    rankings = data["rankings"]
    if len(rankings) != expected_count:
        raise RerankerError(f"Unexpected reranker response shape: {data!r}")
    scores: list[float | None] = [None] * expected_count
    for item in rankings:
        if not isinstance(item, dict):
            raise RerankerError(f"Unexpected reranker result: {item!r}")
        index = item.get("index")
        score = item.get("logit")
        if not isinstance(index, int) or not 0 <= index < expected_count or not isinstance(score, (int, float)):
            raise RerankerError(f"Unexpected reranker result: {item!r}")
        if scores[index] is not None:
            raise RerankerError(f"Duplicate reranker result index: {index}")
        scores[index] = float(score)
    if any(score is None for score in scores):
        raise RerankerError("Reranker response omitted one or more candidate scores")
    return [float(score) for score in scores]


def rerank_documents(
    config: RerankerConfig,
    query: str,
    documents: list[str],
    *,
    timeout: int = 60,
    request_json: Any = _request_json,
) -> list[float]:
    """Return reranker scores in the same order as ``documents``."""

    if not documents:
        return []
    if config.protocol not in {"tei", "passages"}:
        raise RerankerError("RERANKER_PROTOCOL must be 'tei' or 'passages'")
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    if config.protocol == "passages":
        if not config.model:
            raise RerankerError("RERANKER_MODEL is required for the passages protocol")
        payload: dict[str, Any] = {
            "model": config.model,
            "query": {"text": query},
            "passages": [{"text": document} for document in documents],
        }
    else:
        payload = {"query": query, "texts": documents, "raw_scores": True}
        if config.model:
            payload["model"] = config.model
    try:
        data = request_json(config.rerank_url, payload, headers, timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RerankerError(f"Reranker HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RerankerError(f"Reranker request failed: {error}") from error
    if config.protocol == "passages":
        return _parse_passage_scores(data, len(documents))
    return _parse_tei_scores(data, len(documents))
