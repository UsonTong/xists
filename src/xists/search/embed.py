"""Embedding against an OpenAI-compatible embeddings endpoint.

The endpoint can be OpenAI itself or any compatible server, including a local
bge-m3 served through vLLM, Infinity, Text Embeddings Inference, Xinference, or
LocalAI. xists treats them all the same: ``POST {base_url}/embeddings``.

The embedding text is built only from collected facts and the evidence-based
llm_profile. ``not_for`` is intentionally excluded so it does not pull in
queries the repository is a poor fit for.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

USER_AGENT = "xists-embedding"
EMBEDDING_INPUT_VERSION = 2


class EmbeddingError(RuntimeError):
    """Raised when the embedding call or its response cannot be used."""


class EmbeddingNotConfiguredError(EmbeddingError):
    """Raised when no embedding configuration is available in the environment."""


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str
    base_url: str
    model: str

    @property
    def embeddings_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/embeddings"

    @property
    def tei_embed_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return f"{base_url}/embed"


def embedding_config_from_env() -> EmbeddingConfig:
    """Build an EmbeddingConfig from environment variables.

    Raises EmbeddingNotConfiguredError if required values are missing so callers
    can fail fast with a clear message.
    """

    api_key = os.environ.get("EMBEDDING_API_KEY")
    base_url = os.environ.get("EMBEDDING_BASE_URL")
    model = os.environ.get("EMBEDDING_MODEL")

    missing = [
        name
        for name, value in (
            ("EMBEDDING_API_KEY", api_key),
            ("EMBEDDING_BASE_URL", base_url),
            ("EMBEDDING_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise EmbeddingNotConfiguredError(
            "Embedding is required for indexing and search but is not configured. "
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them in your .env (see .env.example)."
        )

    return EmbeddingConfig(api_key=api_key, base_url=base_url, model=model)


def embedding_text_from_record(record: dict[str, Any]) -> str:
    """Build the text used to embed a record.

    Uses collected facts plus the evidence-based llm_profile. ``not_for`` is
    excluded on purpose so a repository is not matched to queries it is a poor
    fit for.
    """

    github = record.get("github") or {}
    profile = record.get("llm_profile") or {}

    # Content parts carry the actual semantic signal. The repository name alone
    # is not enough to embed, so a record with no content here is skipped.
    content: list[str] = []
    if github.get("description"):
        content.append(str(github["description"]))
    topics = github.get("topics") or []
    if topics:
        content.append(", ".join(str(t) for t in topics))
    if github.get("language"):
        content.append(str(github["language"]))
    if profile.get("summary"):
        content.append(str(profile["summary"]))
    for key in ("use_cases", "capabilities", "search_phrases"):
        values = profile.get(key) or []
        if values:
            content.append(", ".join(str(v) for v in values))
    if not content:
        return ""

    name = record.get("repo_id") or record.get("name")
    parts = ([str(name)] if name else []) + content
    return "\n".join(parts).strip()


def embedding_input_fingerprint(record: dict[str, Any]) -> str | None:
    text = embedding_text_from_record(record)
    if not text:
        return None
    payload = json.dumps(
        {
            "embedding_input_version": EMBEDDING_INPUT_VERSION,
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_openai_response(data: Any, count: int) -> list[list[float]] | None:
    """Try to parse an OpenAI-style response. Returns None if the shape
    does not match so callers can try the next format."""
    if not isinstance(data, dict) or "data" not in data:
        return None
    items = data["data"]
    if not isinstance(items, list):
        return None
    try:
        items_sorted = sorted(items, key=lambda item: item["index"])
        vectors = [item["embedding"] for item in items_sorted]
        if len(vectors) != count:
            return None
        return vectors
    except (KeyError, IndexError, TypeError):
        return None


def _parse_tei_response(data: Any, count: int) -> list[list[float]] | None:
    """Try to parse a TEI-style response (bare array of arrays)."""
    if not isinstance(data, list):
        return None
    if len(data) != count:
        return None
    if not all(isinstance(v, list) for v in data):
        return None
    return data


def _request_json(url: str, body: bytes, headers: dict[str, str], timeout: int) -> Any:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_embeddings(
    config: EmbeddingConfig,
    inputs: list[str],
    *,
    timeout: int = 60,
) -> list[list[float]]:
    """Call an OpenAI-compatible embeddings endpoint, return vectors in order.

    Tries the standard ``/embeddings`` path first.  If the server rejects it
    (e.g. TEI which exposes ``/embed`` instead), falls back to the TEI path.
    Supports both the OpenAI response shape ``{"data": [...]}`` and the TEI
    bare-array shape ``[[...], ...]``.
    """

    if not inputs:
        return []

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    openai_headers = dict(headers)
    tei_headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}

    last_error: Exception | None = None
    for url, payload, hdrs in [
        (config.embeddings_url, {"model": config.model, "input": inputs}, openai_headers),
        (config.tei_embed_url, {"inputs": inputs}, tei_headers),
    ]:
        body = json.dumps(payload).encode("utf-8")
        try:
            data = _request_json(url, body, hdrs, timeout)
        except urllib.error.HTTPError as error:
            try:
                detail = error.read().decode("utf-8")
            except Exception:
                detail = str(error)
            last_error = EmbeddingError(
                f"Embedding request failed at {url} (HTTP {error.code}): {detail}"
            )
            continue
        except urllib.error.URLError as error:
            last_error = EmbeddingError(f"Embedding request failed at {url}: {error}")
            continue

        vectors = _parse_openai_response(data, len(inputs))
        if vectors is not None:
            return vectors
        vectors = _parse_tei_response(data, len(inputs))
        if vectors is not None:
            return vectors

        last_error = EmbeddingError(
            f"Unexpected embedding response shape from {url}: {json.dumps(data)[:300]}"
        )

    if last_error:
        raise last_error
    raise EmbeddingError("No embedding endpoint responded")


def embed_query(config: EmbeddingConfig, query: str) -> list[float]:
    vectors = call_embeddings(config, [query])
    if not vectors:
        raise EmbeddingError("Embedding endpoint returned no vector for the query")
    return vectors[0]
