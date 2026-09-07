"""Incremental record and vector append for xists index."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from xists.ingest.github import collect_record, parse_github_repo
from xists.profile.llm import (
    LLMConfig,
    generate_llm_profile,
    llm_config_from_env,
)
from xists.records import RECORD_SCHEMA_VERSION, normalize_llm_profile
from xists.search.bm25 import BM25Index
from xists.search.embed import (
    EmbeddingConfig,
    call_embeddings,
    embedding_config_from_env,
    embedding_input_fingerprint,
    embedding_text_from_record,
)
from xists.search.index import (
    INDEX_VERSION,
    decode_vector,
    entry_metadata,
    load_index,
    save_index,
)


def append_records_to_index(
    records: list[dict[str, Any]],
    index_path: Path | str,
    records_path: Path | str | None = None,
    *,
    config: EmbeddingConfig | None = None,
    force: bool = False,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Incrementally append or update records and their vector embeddings into an index.

    Parameters:
        records: List of new/updated record dictionaries.
        index_path: Path to the target index.json file.
        records_path: Optional path to records.json file to update alongside index.
        config: Optional EmbeddingConfig (resolved from env if not provided).
        force: Force re-embedding even if fingerprint has not changed.
        batch_size: Batch size for embedding calls.
    """
    started = perf_counter()
    idx_path = Path(index_path).resolve()
    if not idx_path.is_file():
        raise FileNotFoundError(
            f"Target index file not found: {idx_path}. Use 'xists index build' to initialize."
        )

    # 1. Load existing index and matrix
    index_doc = load_index(idx_path, mmap=False)
    index_version = index_doc.get("index_version", INDEX_VERSION)
    dimension = index_doc.get("dimension")
    existing_vectors_entries = list(index_doc.get("vectors") or [])

    # Load matrix
    if "_matrix" in index_doc and index_doc["_matrix"] is not None:
        matrix = np.asarray(index_doc["_matrix"], dtype=np.float32)
    else:
        # Reconstruct matrix from decoded vectors
        matrix_rows = []
        for v in existing_vectors_entries:
            vec = decode_vector(v.get("vector"), dimension=dimension)
            if vec is None:
                raise ValueError(f"Corrupt vector in index entry: {v.get('repo_id')}")
            matrix_rows.append(vec)
        if matrix_rows:
            matrix = np.asarray(matrix_rows, dtype=np.float32)
        else:
            matrix = np.empty((0, dimension or 0), dtype=np.float32)

    if dimension is None and matrix.ndim == 2 and matrix.shape[1] > 0:
        dimension = int(matrix.shape[1])

    # 2. Load existing records if path provided
    existing_records: list[dict[str, Any]] = []
    rec_path = Path(records_path).resolve() if records_path else None
    if rec_path and rec_path.is_file():
        try:
            content = rec_path.read_text(encoding="utf-8")
            existing_records = json.loads(content)
            if not isinstance(existing_records, list):
                existing_records = []
        except Exception:
            existing_records = []

    # Map existing items by lowercase repo_id
    existing_vec_map: dict[str, int] = {
        str(v.get("repo_id")).lower(): idx
        for idx, v in enumerate(existing_vectors_entries)
        if "repo_id" in v
    }
    existing_rec_map: dict[str, int] = {
        str(r.get("repo_id") or r.get("repo_id_requested")).lower(): idx
        for idx, r in enumerate(existing_records)
        if r.get("repo_id") or r.get("repo_id_requested")
    }

    # 3. Resolve EmbeddingConfig
    if config is None:
        try:
            config = embedding_config_from_env()
        except Exception as error:
            raise ValueError(
                f"Embedding configuration required to compute vectors for appended records: {error}"
            ) from error

    # 4. Classify candidates into (skip, update, add)
    to_embed: list[
        tuple[dict[str, Any], str, str, str | None]
    ] = []  # (record, repo_id, text, action)
    added_count = 0
    updated_count = 0
    skipped_count = 0
    skipped_details: list[dict[str, str]] = []

    for r in records:
        repo_id = str(r.get("repo_id") or r.get("repo_id_requested") or "").strip()
        if not repo_id:
            skipped_count += 1
            skipped_details.append({"repo_id": "<unknown>", "reason": "missing repo_id"})
            continue

        text = embedding_text_from_record(r)
        if not text:
            skipped_count += 1
            skipped_details.append({"repo_id": repo_id, "reason": "empty embeddable text"})
            continue

        fingerprint = embedding_input_fingerprint(r)
        rid_low = repo_id.lower()

        if rid_low in existing_vec_map:
            existing_entry = existing_vectors_entries[existing_vec_map[rid_low]]
            old_fp = existing_entry.get("embedding_input_fingerprint")
            if not force and old_fp == fingerprint:
                skipped_count += 1
                skipped_details.append({"repo_id": repo_id, "reason": "fingerprint unchanged"})
                # Still update record in records.json if needed
                if rec_path and rid_low in existing_rec_map:
                    existing_records[existing_rec_map[rid_low]] = r
                continue
            # Mark for update
            to_embed.append((r, repo_id, text, "update"))
        else:
            # Mark for add
            to_embed.append((r, repo_id, text, "add"))

    # 5. Compute embeddings in batches
    if to_embed:
        texts_to_call = [item[2] for item in to_embed]
        new_vectors: list[list[float]] = []
        for i in range(0, len(texts_to_call), batch_size):
            chunk = texts_to_call[i : i + batch_size]
            vectors_chunk = call_embeddings(config, chunk, input_type="passage")
            new_vectors.extend(vectors_chunk)

        for (r, repo_id, _, action), float_vec in zip(to_embed, new_vectors):
            rid_low = repo_id.lower()
            norm_vec = np.asarray(float_vec, dtype=np.float32)
            if dimension is not None and norm_vec.size != dimension:
                raise ValueError(
                    f"Vector dimension mismatch for {repo_id}: expected {dimension}, got {norm_vec.size}"
                )
            if dimension is None:
                dimension = int(norm_vec.size)

            entry = {
                "repo_id": repo_id,
                "embedding_input_fingerprint": embedding_input_fingerprint(r),
                "metadata": entry_metadata(r),
            }

            if action == "update":
                pos = existing_vec_map[rid_low]
                existing_vectors_entries[pos] = entry
                if matrix.shape[0] > pos:
                    matrix[pos] = norm_vec
                updated_count += 1
                if rec_path:
                    if rid_low in existing_rec_map:
                        existing_records[existing_rec_map[rid_low]] = r
                    else:
                        existing_records.append(r)
                        existing_rec_map[rid_low] = len(existing_records) - 1
            else:  # add
                pos = len(existing_vectors_entries)
                existing_vectors_entries.append(entry)
                existing_vec_map[rid_low] = pos
                if matrix.shape[0] == 0:
                    matrix = norm_vec.reshape(1, -1)
                else:
                    matrix = np.vstack([matrix, norm_vec.reshape(1, -1)])
                added_count += 1
                if rec_path:
                    if rid_low in existing_rec_map:
                        existing_records[existing_rec_map[rid_low]] = r
                    else:
                        existing_records.append(r)
                        existing_rec_map[rid_low] = len(existing_records) - 1

    # 6. Save updated index, BM25 index, and records atomically
    index_doc["dimension"] = dimension
    index_doc["record_count"] = len(existing_vectors_entries)
    index_doc["vectors"] = existing_vectors_entries
    index_doc["built_at"] = datetime.now(UTC).isoformat()
    if "_matrix" in index_doc:
        index_doc["_matrix"] = matrix

    bm25_index: BM25Index | None = None
    if index_doc.get("_bm25_path"):
        try:
            bm25_raw = json.loads(Path(index_doc["_bm25_path"]).read_text(encoding="utf-8"))
            bm25_index = BM25Index.from_dict(bm25_raw)
        except Exception:
            bm25_index = None

    if bm25_index is not None and updated_count == 0 and added_count > 0:
        added_entries = existing_vectors_entries[-added_count:]
        bm25_index.append_entries(added_entries)
    elif added_count > 0 or updated_count > 0:
        bm25_index = BM25Index.build_from_entries(existing_vectors_entries)

    save_index(idx_path, index_doc, matrix=matrix, bm25_index=bm25_index, version=index_version)

    if rec_path:
        temp_rec_path = rec_path.with_name(f".{rec_path.name}.tmp.{datetime.now().timestamp()}")
        temp_rec_path.write_text(
            json.dumps(existing_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp_rec_path.replace(rec_path)

    elapsed_ms = round((perf_counter() - started) * 1000, 3)

    return {
        "status": "success",
        "added": added_count,
        "updated": updated_count,
        "skipped": skipped_count,
        "total_records": len(existing_records) if rec_path else len(existing_vectors_entries),
        "total_vectors": len(existing_vectors_entries),
        "dimension": dimension,
        "elapsed_ms": elapsed_ms,
        "index_path": str(idx_path),
        "records_path": str(rec_path) if rec_path else None,
        "skipped_details": skipped_details,
    }


def append_repo_to_index(
    repo: str,
    index_path: Path | str,
    records_path: Path | str | None = None,
    *,
    github_token: str | None = None,
    llm_config: LLMConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Ingest, profile, embed, and incrementally append a single GitHub repo to index and records."""
    repo_id = parse_github_repo(repo)

    # 1. Collect GitHub record
    record = collect_record(repo_id, token=github_token)

    # 2. Generate LLM profile if config present
    if llm_config is None:
        try:
            llm_config = llm_config_from_env()
        except Exception:
            llm_config = None

    if llm_config is not None:
        profile_res = generate_llm_profile(record, llm_config)
        record["llm_profile"] = profile_res.get("llm_profile") or {}
    record["llm_profile"] = normalize_llm_profile(record.get("llm_profile"))
    record["schema_version"] = RECORD_SCHEMA_VERSION

    # 3. Append to index
    return append_records_to_index(
        [record],
        index_path=index_path,
        records_path=records_path,
        config=embedding_config,
        force=force,
    )
