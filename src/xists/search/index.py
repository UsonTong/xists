"""Build and load the embedding index for xists records.

The index is derived data: vectors computed from records. It is stored
separately from records.json because changing the embedding model invalidates
all vectors and requires a rebuild. The index records which model and dimension
were used so search can refuse to run against a mismatched model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xists.search.embed import (
    EMBEDDING_INPUT_VERSION,
    EmbeddingConfig,
    EmbeddingError,
    call_embeddings,
    embedding_input_fingerprint,
    embedding_text_from_record,
)

INDEX_VERSION = 1


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if isinstance(value, str) and value.strip()]


def entry_metadata(record: dict[str, Any]) -> dict[str, Any]:
    github = record.get("github") or {}
    profile = record.get("llm_profile") or {}
    return {
        "name": record.get("name"),
        "description": github.get("description"),
        "topics": _string_list(github.get("topics")),
        "language": github.get("language"),
        "summary": profile.get("summary"),
        "use_cases": _string_list(profile.get("use_cases")),
        "capabilities": _string_list(profile.get("capabilities")),
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
        results = call_embeddings(config, [item["text"] for item in batch])
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
                    "vector": vector,
                }
            )

    return {
        "index_version": INDEX_VERSION,
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
