"""Build and load the embedding index for xists records.

The index is derived data: vectors computed from records. It is stored
separately from records.json because changing the embedding model invalidates
all vectors and requires a rebuild. The index records which model and dimension
were used so search can refuse to run against a mismatched model.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from xists.search.embed import (
    EMBEDDING_INPUT_VERSION,
    EmbeddingConfig,
    EmbeddingError,
    call_embeddings,
    embedding_input_fingerprint,
    embedding_text_from_record,
)
from xists.records import RECORD_SCHEMA_VERSION

INDEX_VERSION = 3
VECTOR_ENCODING_FLOAT32_BASE64 = "float32_base64"


def encode_vector(vector: list[float]) -> str:
    """Encode an embedding as compact, portable float32 data."""

    values = np.asarray(vector, dtype="<f4")
    if values.ndim != 1:
        raise ValueError("Embedding vectors must be one-dimensional")
    return base64.b64encode(values.tobytes()).decode("ascii")


def decode_vector(value: Any, *, dimension: int | None = None) -> np.ndarray | None:
    """Decode compact vectors while accepting legacy JSON number arrays."""

    if isinstance(value, str):
        try:
            vector = np.frombuffer(base64.b64decode(value.encode("ascii"), validate=True), dtype="<f4")
        except (ValueError, TypeError):
            return None
    elif isinstance(value, list):
        try:
            vector = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if vector.ndim != 1 or (dimension is not None and vector.size != dimension):
        return None
    return vector


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if isinstance(value, str) and value.strip()]


def entry_metadata(record: dict[str, Any]) -> dict[str, Any]:
    github = record.get("github") or {}
    profile = record.get("llm_profile") or {}
    return {
        "schema_version": record.get("schema_version"),
        "name": record.get("name"),
        "url": record.get("url"),
        "aliases": _string_list(profile.get("aliases")),
        "description": github.get("description"),
        "topics": _string_list(github.get("topics")),
        "language": github.get("language"),
        "stars": github.get("stars"),
        "forks": github.get("forks"),
        "archived": github.get("archived"),
        "disabled": github.get("disabled"),
        "pushed_at": github.get("pushed_at"),
        "summary": profile.get("summary"),
        "use_cases": _string_list(profile.get("use_cases")),
        "capabilities": _string_list(profile.get("capabilities")),
        "project_type": profile.get("project_type"),
        "ecosystem": _string_list(profile.get("ecosystem")),
        "replaces": _string_list(profile.get("replaces")),
        "related_projects": _string_list(profile.get("related_projects")),
        "search_text": profile.get("search_text"),
        "search_phrases": _string_list(profile.get("search_phrases")),
    }


def build_index(
    records: list[dict[str, Any]],
    config: EmbeddingConfig,
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Embed every record and return an index document.

    Records without any embeddable text are skipped and reported in
    ``skipped`` so the caller can surface them.
    """

    embeddable: list[dict[str, Any]] = []
    skipped: list[str] = []
    for record in records:
        text = embedding_text_from_record(record)
        repo_id = record.get("repo_id") or record.get("repo_id_requested")
        if not text:
            skipped.append(repo_id or "<unknown>")
            continue
        embeddable.append(
            {
                "repo_id": repo_id,
                "text": text,
                "fingerprint": embedding_input_fingerprint(record),
                "metadata": entry_metadata(record),
            }
        )

    vectors: list[dict[str, Any]] = []
    dimension: int | None = None
    for start in range(0, len(embeddable), batch_size):
        batch = embeddable[start : start + batch_size]
        results = call_embeddings(
            config, [item["text"] for item in batch], input_type="passage"
        )
        if len(results) != len(batch):
            raise EmbeddingError(
                f"Embedding count mismatch: sent {len(batch)}, received {len(results)}"
            )
        for item, vector in zip(batch, results):
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise EmbeddingError(
                    f"Inconsistent embedding dimension: {len(vector)} vs {dimension}"
                )
            vectors.append(
                {
                    "repo_id": item["repo_id"],
                    "embedding_input_fingerprint": item["fingerprint"],
                    "metadata": item["metadata"],
                    "vector": encode_vector(vector),
                }
            )

    return {
        "index_version": INDEX_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_model": config.model,
        "embedding_base_url": config.base_url,
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
        "dimension": dimension,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(vectors),
        "skipped": skipped,
        "vectors": vectors,
    }


def load_index(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
