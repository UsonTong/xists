"""Multi-index merge and fusion utilities for xists."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from xists.records import RECORD_SCHEMA_VERSION
from xists.search.embed import EMBEDDING_INPUT_VERSION
from xists.search.index import (
    INDEX_VERSION,
    decode_vector,
    load_index,
    save_index,
)


def _profile_quality_score(record_or_meta: dict[str, Any]) -> float:
    """Calculate heuristic quality score to resolve duplicate conflicts."""
    score = 0.0
    profile = record_or_meta.get("llm_profile") if "llm_profile" in record_or_meta else record_or_meta
    confidence = str(profile.get("confidence") or "").lower()
    if confidence == "high":
        score += 10.0
    elif confidence == "medium":
        score += 5.0
    elif confidence == "low":
        score += 1.0

    if profile.get("summary"):
        score += 3.0
    if profile.get("use_cases"):
        score += 2.0
    if profile.get("capabilities"):
        score += 2.0
    if profile.get("search_text"):
        score += 2.0
    if profile.get("aliases"):
        score += 1.0
    if not profile.get("abstained", False):
        score += 5.0

    return score


def merge_indices(
    index_paths: list[Path | str],
    output_index_path: Path | str,
    records_paths: list[Path | str] | None = None,
    output_records_path: Path | str | None = None,
) -> dict[str, Any]:
    """Merge multiple index documents and vector matrices into a single unified index.

    Parameters:
        index_paths: List of paths to input index.json files.
        output_index_path: Path where merged index.json will be written.
        records_paths: Optional list of corresponding records.json files.
        output_records_path: Optional path where merged records.json will be written.
    """
    started = perf_counter()
    if len(index_paths) < 2:
        raise ValueError("At least two index paths are required to perform a merge.")

    resolved_index_paths = [Path(p).resolve() for p in index_paths]
    for p in resolved_index_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Input index file not found: {p}")

    # 1. Load and validate compatibility of all input indices
    loaded_indices: list[dict[str, Any]] = []
    base_model: str | None = None
    base_dimension: int | None = None

    for p in resolved_index_paths:
        idx_doc = load_index(p, mmap=False)
        model = idx_doc.get("embedding_model")
        dim = idx_doc.get("dimension")

        if base_model is None:
            base_model = model
        elif model != base_model:
            raise ValueError(
                f"Cannot merge indices with different embedding models: '{base_model}' vs '{model}' in {p}"
            )

        if base_dimension is None:
            base_dimension = dim
        elif dim != base_dimension:
            raise ValueError(
                f"Cannot merge indices with different vector dimensions: {base_dimension} vs {dim} in {p}"
            )

        loaded_indices.append(idx_doc)

    # 2. Load records if provided
    loaded_records_by_index: list[list[dict[str, Any]]] = []
    if records_paths:
        for rp in records_paths:
            rp_resolved = Path(rp).resolve()
            if rp_resolved.is_file():
                try:
                    recs = json.loads(rp_resolved.read_text(encoding="utf-8"))
                    loaded_records_by_index.append(recs if isinstance(recs, list) else [])
                except Exception:
                    loaded_records_by_index.append([])
            else:
                loaded_records_by_index.append([])

    # 3. Deduplicate across indices
    # Map repo_id.lower() -> (chosen_entry, chosen_vector, chosen_record, quality_score, source_idx)
    merged_entries_map: dict[str, dict[str, Any]] = {}
    duplicates_found = 0

    for idx_i, idx_doc in enumerate(loaded_indices):
        entries = idx_doc.get("vectors") or []
        dim = idx_doc.get("dimension", base_dimension)

        # Retrieve matrix
        if "_matrix" in idx_doc and idx_doc["_matrix"] is not None:
            mat = np.asarray(idx_doc["_matrix"], dtype=np.float32)
        else:
            rows = []
            for entry in entries:
                vec = decode_vector(entry.get("vector"), dimension=dim)
                rows.append(vec if vec is not None else np.zeros(dim or 0, dtype=np.float32))
            mat = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, dim or 0), dtype=np.float32)

        records_list = loaded_records_by_index[idx_i] if idx_i < len(loaded_records_by_index) else []
        records_by_id = {
            str(r.get("repo_id") or r.get("repo_id_requested")).lower(): r
            for r in records_list
            if r.get("repo_id") or r.get("repo_id_requested")
        }

        for row_i, entry in enumerate(entries):
            repo_id = str(entry.get("repo_id") or "").strip()
            if not repo_id:
                continue
            rid_low = repo_id.lower()
            vec_row = mat[row_i] if row_i < mat.shape[0] else None
            record_item = records_by_id.get(rid_low)

            meta = entry.get("metadata") or {}
            quality = _profile_quality_score(record_item if record_item else meta)

            if rid_low in merged_entries_map:
                duplicates_found += 1
                prev_quality = merged_entries_map[rid_low]["quality"]
                if quality > prev_quality:
                    # Replace with higher quality
                    merged_entries_map[rid_low] = {
                        "entry": entry,
                        "vector": vec_row,
                        "record": record_item,
                        "quality": quality,
                        "source": str(resolved_index_paths[idx_i]),
                    }
            else:
                merged_entries_map[rid_low] = {
                    "entry": entry,
                    "vector": vec_row,
                    "record": record_item,
                    "quality": quality,
                    "source": str(resolved_index_paths[idx_i]),
                }

    # 4. Construct unified output matrix and vectors metadata
    final_entries: list[dict[str, Any]] = []
    final_records: list[dict[str, Any]] = []
    final_vector_rows: list[np.ndarray] = []

    for item in merged_entries_map.values():
        entry = item["entry"]
        final_entries.append({
            "repo_id": entry.get("repo_id"),
            "embedding_input_fingerprint": entry.get("embedding_input_fingerprint"),
            "metadata": entry.get("metadata", {}),
        })
        if item["record"]:
            final_records.append(item["record"])
        if item["vector"] is not None:
            final_vector_rows.append(np.asarray(item["vector"], dtype=np.float32))

    if final_vector_rows:
        merged_matrix = np.vstack(final_vector_rows)
    else:
        merged_matrix = np.empty((0, base_dimension or 0), dtype=np.float32)

    # 5. Save output index and records
    out_idx = Path(output_index_path).resolve()
    merged_doc = {
        "index_version": INDEX_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_model": base_model,
        "embedding_base_url": loaded_indices[0].get("embedding_base_url"),
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
        "dimension": base_dimension,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(final_entries),
        "skipped": [],
        "vectors": final_entries,
    }

    save_index(out_idx, merged_doc, matrix=merged_matrix, version=INDEX_VERSION)

    out_rec = Path(output_records_path).resolve() if output_records_path else None
    if out_rec and final_records:
        out_rec.parent.mkdir(parents=True, exist_ok=True)
        temp_rec = out_rec.with_name(f".{out_rec.name}.tmp.{datetime.now().timestamp()}")
        temp_rec.write_text(json.dumps(final_records, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_rec.replace(out_rec)

    elapsed_ms = round((perf_counter() - started) * 1000, 3)

    return {
        "status": "success",
        "input_indices_count": len(resolved_index_paths),
        "merged_records_count": len(final_entries),
        "duplicates_resolved": duplicates_found,
        "embedding_model": base_model,
        "dimension": base_dimension,
        "elapsed_ms": elapsed_ms,
        "output_index": str(out_idx),
        "output_records": str(out_rec) if out_rec else None,
    }
