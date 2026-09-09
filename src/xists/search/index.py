"""Build and load the embedding index for xists records.

The index is derived data: vectors computed from records. It is stored
separately from records.json because changing the embedding model invalidates
all vectors and requires a rebuild. The index records which model and dimension
were used so search can refuse to run against a mismatched model.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xists.records import RECORD_SCHEMA_VERSION
from xists.search.embed import (
    EMBEDDING_INPUT_VERSION,
    EmbeddingConfig,
    EmbeddingError,
    call_embeddings,
    embedding_input_fingerprint,
    embedding_text_from_record,
)

INDEX_VERSION = 4
LEGACY_INDEX_VERSION = 3
SUPPORTED_INDEX_VERSIONS = (3, 4)
VECTOR_ENCODING_FLOAT32_BASE64 = "float32_base64"


def encode_vector(vector: list[float]) -> str:
    """Encode an embedding as compact, portable float32 data."""

    values = np.asarray(vector, dtype="<f4")
    if values.ndim != 1:
        raise ValueError("Embedding vectors must be one-dimensional")
    return base64.b64encode(values.tobytes()).decode("ascii")


def decode_vector(value: Any, *, dimension: int | None = None) -> np.ndarray | None:
    """Decode compact vectors while accepting legacy JSON number arrays and NumPy arrays."""

    if isinstance(value, str):
        try:
            vector = np.frombuffer(
                base64.b64decode(value.encode("ascii"), validate=True), dtype="<f4"
            )
        except (ValueError, TypeError):
            return None
    elif isinstance(value, np.ndarray):
        vector = np.asarray(value, dtype=np.float32)
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
        "license": github.get("license"),
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

    raw_vectors: list[list[float]] = []
    vectors: list[dict[str, Any]] = []
    dimension: int | None = None
    for start in range(0, len(embeddable), batch_size):
        batch = embeddable[start : start + batch_size]
        results = call_embeddings(config, [item["text"] for item in batch], input_type="passage")
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
            raw_vectors.append(vector)
            vectors.append(
                {
                    "repo_id": item["repo_id"],
                    "embedding_input_fingerprint": item["fingerprint"],
                    "metadata": item["metadata"],
                    "vector": encode_vector(vector),
                }
            )

    if raw_vectors:
        matrix = np.asarray(raw_vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
    else:
        matrix = np.empty((0, dimension or 0), dtype=np.float32)

    return {
        "index_version": INDEX_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_model": config.model,
        "embedding_base_url": config.base_url,
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
        "dimension": dimension,
        "built_at": datetime.now(UTC).isoformat(),
        "record_count": len(vectors),
        "skipped": skipped,
        "vectors_normalized": True,
        "vectors": vectors,
        "_matrix": matrix,
    }


def save_index(
    path: Path | str,
    index: dict[str, Any],
    *,
    matrix: np.ndarray | None = None,
    bm25_index: Any = None,
    version: int = INDEX_VERSION,
) -> None:
    """Save an index document to disk.

    For INDEX_VERSION = 4 (default), metadata is saved to JSON and the
    vector matrix is saved as a sidecar binary file (<stem>.vectors.npy) for
    fast mmap loading. A BM25 sparse index is saved as <stem>.bm25.json.
    For INDEX_VERSION = 3, a single JSON document with Base64-encoded vectors
    is written.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if version == 4:
        vectors_filename = f"{file_path.stem}.vectors.npy"
        vectors_path = file_path.with_name(vectors_filename)

        if matrix is not None:
            vec_matrix = np.asarray(matrix, dtype=np.float32)
        elif "_matrix" in index and index["_matrix"] is not None:
            vec_matrix = np.asarray(index["_matrix"], dtype=np.float32)
        else:
            vectors_list = []
            dimension = index.get("dimension")
            for entry in index.get("vectors", []):
                vec = decode_vector(entry.get("vector"), dimension=dimension)
                if vec is None:
                    raise ValueError(f"Entry {entry.get('repo_id')} has invalid or missing vector")
                vectors_list.append(vec)
            if vectors_list:
                vec_matrix = np.asarray(vectors_list, dtype=np.float32)
            else:
                vec_matrix = np.empty((0, dimension or 0), dtype=np.float32)

        if len(vec_matrix) > 0 and vec_matrix.ndim == 2:
            norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True)
            if not np.allclose(norms, 1.0, atol=1e-3):
                vec_matrix = np.divide(
                    vec_matrix, norms, out=np.zeros_like(vec_matrix), where=norms != 0
                )

        # Atomic write of sidecar .npy
        temp_vec_path = vectors_path.with_name(
            f".{vectors_path.name}.tmp.{datetime.now().timestamp()}"
        )
        with temp_vec_path.open("wb") as f:
            np.save(f, vec_matrix)
        temp_vec_path.replace(vectors_path)

        clean_vectors = []
        for entry in index.get("vectors", []):
            clean_entry = {
                "repo_id": entry.get("repo_id"),
                "embedding_input_fingerprint": entry.get("embedding_input_fingerprint"),
                "metadata": entry.get("metadata", {}),
            }
            clean_vectors.append(clean_entry)

        # BM25 sidecar serialization
        bm25_data: dict[str, Any] | None = None
        if bm25_index is not None:
            bm25_data = bm25_index.to_dict() if hasattr(bm25_index, "to_dict") else bm25_index
        elif "_bm25_index" in index and hasattr(index["_bm25_index"], "to_dict"):
            bm25_data = index["_bm25_index"].to_dict()
        elif "_bm25" in index and isinstance(index["_bm25"], dict):
            bm25_data = index["_bm25"]
        elif clean_vectors:
            from xists.search.bm25 import BM25Index

            bm25_data = BM25Index.build_from_entries(clean_vectors).to_dict()

        bm25_filename: str | None = None
        if bm25_data is not None:
            bm25_filename = f"{file_path.stem}.bm25.json"
            bm25_path = file_path.with_name(bm25_filename)
            temp_bm25_path = bm25_path.with_name(
                f".{bm25_path.name}.tmp.{datetime.now().timestamp()}"
            )
            temp_bm25_path.write_text(json.dumps(bm25_data, ensure_ascii=False), encoding="utf-8")
            temp_bm25_path.replace(bm25_path)

        # MetaDatabase SQLite sidecar serialization
        from xists.search.meta_db import MetaDatabase
        from xists.search.query import _precompute_entry_cache

        caches = index.get("_metadata_caches")
        if caches is None:
            caches = [_precompute_entry_cache(entry) for entry in clean_vectors]

        meta_db_filename = f"{file_path.stem}.meta.db"
        meta_db_path = file_path.with_name(meta_db_filename)
        meta_manifest = {
            "index_version": 4,
            "record_schema_version": index.get("record_schema_version", RECORD_SCHEMA_VERSION),
            "embedding_model": index.get("embedding_model", ""),
            "embedding_base_url": index.get("embedding_base_url"),
            "embedding_input_version": index.get(
                "embedding_input_version", EMBEDDING_INPUT_VERSION
            ),
            "dimension": int(vec_matrix.shape[1])
            if vec_matrix.ndim == 2 and vec_matrix.shape[1] > 0
            else index.get("dimension"),
            "built_at": index.get("built_at") or datetime.now(UTC).isoformat(),
            "record_count": len(clean_vectors),
            "vectors_file": vectors_filename,
            "bm25_file": bm25_filename,
            "meta_db_file": meta_db_filename,
        }
        MetaDatabase.create_from_entries(
            meta_db_path,
            manifest=meta_manifest,
            entries=clean_vectors,
            caches=caches,
        )

        clean_document = {
            "index_version": 4,
            "record_schema_version": index.get("record_schema_version", RECORD_SCHEMA_VERSION),
            "embedding_model": index.get("embedding_model", ""),
            "embedding_base_url": index.get("embedding_base_url"),
            "embedding_input_version": index.get(
                "embedding_input_version", EMBEDDING_INPUT_VERSION
            ),
            "dimension": int(vec_matrix.shape[1])
            if vec_matrix.ndim == 2 and vec_matrix.shape[1] > 0
            else index.get("dimension"),
            "built_at": index.get("built_at") or datetime.now(UTC).isoformat(),
            "record_count": len(clean_vectors),
            "skipped": index.get("skipped", []),
            "vectors_normalized": True,
            "vectors_file": vectors_filename,
            "bm25_file": bm25_filename,
            "meta_db_file": meta_db_filename,
            "vectors": clean_vectors,
        }
        temp_json_path = file_path.with_name(f".{file_path.name}.tmp.{datetime.now().timestamp()}")
        temp_json_path.write_text(
            json.dumps(clean_document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp_json_path.replace(file_path)
    else:
        # Legacy v3 single-file Base64
        clean_vectors = []
        dimension = index.get("dimension")
        raw_mat = matrix if matrix is not None else index.get("_matrix")
        for i, entry in enumerate(index.get("vectors", [])):
            if "vector" in entry and isinstance(entry["vector"], str):
                vec_str = entry["vector"]
            elif raw_mat is not None and i < len(raw_mat):
                vec_str = encode_vector(raw_mat[i].tolist())
            else:
                vec_str = encode_vector([0.0] * (dimension or 0))
            clean_entry = {
                "repo_id": entry.get("repo_id"),
                "embedding_input_fingerprint": entry.get("embedding_input_fingerprint"),
                "metadata": entry.get("metadata", {}),
                "vector": vec_str,
            }
            clean_vectors.append(clean_entry)

        doc = {
            "index_version": 3,
            "record_schema_version": index.get("record_schema_version", RECORD_SCHEMA_VERSION),
            "embedding_model": index.get("embedding_model", ""),
            "embedding_base_url": index.get("embedding_base_url"),
            "embedding_input_version": index.get(
                "embedding_input_version", EMBEDDING_INPUT_VERSION
            ),
            "dimension": dimension,
            "built_at": index.get("built_at") or datetime.now(UTC).isoformat(),
            "record_count": len(clean_vectors),
            "skipped": index.get("skipped", []),
            "vectors": clean_vectors,
        }
        temp_json_path = file_path.with_name(f".{file_path.name}.tmp.{datetime.now().timestamp()}")
        temp_json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_json_path.replace(file_path)


def load_index(path: Path | str, *, mmap: bool = True) -> dict[str, Any]:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    doc = json.loads(content)
    if not isinstance(doc, dict):
        return doc

    index_version = doc.get("index_version")
    if index_version == 4:
        if "vectors_file" in doc:
            vectors_path = file_path.parent / doc["vectors_file"]
            if vectors_path.is_file():
                try:
                    matrix = np.load(vectors_path, mmap_mode="r" if mmap else None)
                    doc["_matrix"] = matrix
                    doc["_vectors_path"] = str(vectors_path)
                    doc["_mmap"] = bool(mmap)
                except Exception:
                    pass
        if "bm25_file" in doc and doc["bm25_file"]:
            bm25_path = file_path.parent / doc["bm25_file"]
            if bm25_path.is_file():
                doc["_bm25_path"] = str(bm25_path)
        if "meta_db_file" in doc and doc["meta_db_file"]:
            meta_db_path = file_path.parent / doc["meta_db_file"]
            if meta_db_path.is_file():
                doc["_meta_db_path"] = str(meta_db_path)
        else:
            default_meta_db = file_path.with_name(f"{file_path.stem}.meta.db")
            if default_meta_db.is_file():
                doc["_meta_db_path"] = str(default_meta_db)
    return doc
