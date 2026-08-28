"""Index health pruning utilities for xists."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from xists.search.index import (
    INDEX_VERSION,
    decode_vector,
    load_index,
    save_index,
)


def prune_index(
    index_path: Path | str,
    records_path: Path | str | None = None,
    *,
    prune_archived: bool = True,
    prune_disabled: bool = True,
    blocklist_repos: set[str] | list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prune unhealthy, archived, disabled, or blacklisted repositories from index and records.

    Parameters:
        index_path: Path to the target index.json.
        records_path: Optional path to corresponding records.json.
        prune_archived: Whether to remove repositories marked archived by GitHub.
        prune_disabled: Whether to remove repositories marked disabled.
        blocklist_repos: Set/list of repo_ids to explicitly prune.
        dry_run: If True, only reports which repositories would be pruned without modifying files.
    """
    started = perf_counter()
    idx_path = Path(index_path).resolve()
    if not idx_path.is_file():
        raise FileNotFoundError(f"Index file not found: {idx_path}")

    index_doc = load_index(idx_path, mmap=False)
    entries = list(index_doc.get("vectors") or [])
    dimension = index_doc.get("dimension")
    index_version = index_doc.get("index_version", INDEX_VERSION)

    # Load matrix
    if "_matrix" in index_doc and index_doc["_matrix"] is not None:
        matrix = np.asarray(index_doc["_matrix"], dtype=np.float32)
    else:
        rows = []
        for entry in entries:
            vec = decode_vector(entry.get("vector"), dimension=dimension)
            rows.append(vec if vec is not None else np.zeros(dimension or 0, dtype=np.float32))
        matrix = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, dimension or 0), dtype=np.float32)

    # Load records
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

    blocklist_norm = {str(r).strip().lower() for r in (blocklist_repos or []) if str(r).strip()}

    records_by_id = {
        str(r.get("repo_id") or r.get("repo_id_requested")).lower(): r
        for r in existing_records
        if r.get("repo_id") or r.get("repo_id_requested")
    }

    keep_indices: list[int] = []
    pruned_entries: list[dict[str, Any]] = []
    prune_reasons_breakdown: dict[str, int] = {"archived": 0, "disabled": 0, "blocklist": 0}

    for i, entry in enumerate(entries):
        repo_id = str(entry.get("repo_id") or "")
        rid_low = repo_id.lower()
        meta = entry.get("metadata") or {}
        rec = records_by_id.get(rid_low) or {}
        github = rec.get("github") if isinstance(rec.get("github"), dict) else {}

        is_archived = bool(meta.get("archived") or github.get("archived", False))
        is_disabled = bool(meta.get("disabled") or github.get("disabled", False))
        is_blocked = rid_low in blocklist_norm or repo_id in blocklist_norm

        reasons = []
        if prune_archived and is_archived:
            reasons.append("archived")
            prune_reasons_breakdown["archived"] += 1
        if prune_disabled and is_disabled:
            reasons.append("disabled")
            prune_reasons_breakdown["disabled"] += 1
        if is_blocked:
            reasons.append("blocklist")
            prune_reasons_breakdown["blocklist"] += 1

        if reasons:
            pruned_entries.append({
                "repo_id": repo_id,
                "reasons": reasons,
            })
        else:
            keep_indices.append(i)

    retained_count = len(keep_indices)
    pruned_count = len(pruned_entries)

    if not dry_run and pruned_count > 0:
        # Reconstruct matrix
        if matrix.shape[0] > 0 and len(keep_indices) > 0:
            retained_matrix = matrix[keep_indices]
        else:
            retained_matrix = np.empty((0, dimension or 0), dtype=np.float32)

        retained_entries = [entries[idx] for idx in keep_indices]
        retained_rids = {str(e.get("repo_id")).lower() for e in retained_entries if e.get("repo_id")}

        index_doc["vectors"] = retained_entries
        index_doc["record_count"] = len(retained_entries)
        index_doc["built_at"] = datetime.now(timezone.utc).isoformat()
        if "_matrix" in index_doc:
            index_doc["_matrix"] = retained_matrix

        save_index(idx_path, index_doc, matrix=retained_matrix, version=index_version)

        if rec_path and existing_records:
            retained_records = [
                r
                for r in existing_records
                if str(r.get("repo_id") or r.get("repo_id_requested")).lower() in retained_rids
            ]
            temp_rec = rec_path.with_name(f".{rec_path.name}.tmp.{datetime.now().timestamp()}")
            temp_rec.write_text(json.dumps(retained_records, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_rec.replace(rec_path)

    elapsed_ms = round((perf_counter() - started) * 1000, 3)

    return {
        "status": "success",
        "dry_run": dry_run,
        "total_before": len(entries),
        "retained_count": retained_count,
        "pruned_count": pruned_count,
        "reasons": prune_reasons_breakdown,
        "pruned_repositories": pruned_entries,
        "index_path": str(idx_path),
        "records_path": str(rec_path) if rec_path else None,
        "elapsed_ms": elapsed_ms,
    }
