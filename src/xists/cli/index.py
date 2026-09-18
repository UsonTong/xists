"""Index CLI commands (build, stats, verify, migrate, pull, append, merge, prune)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xists.cli.common import (
    _counter_items,
    _format_command_summary,
    _format_top_items,
    _print_embedding_error,
    _read_index_file,
    _read_records_file,
    write_json_atomic,
)
from xists.ingest.github import GitHubAPIError
from xists.profile.llm import PROFILE_PROMPT_VERSION
from xists.records import (
    RECORD_SCHEMA_VERSION,
    record_repo_id,
    records_validation_report,
)
from xists.search.append import (
    append_records_to_index,
    append_repo_to_index,
)
from xists.search.embed import (
    EMBEDDING_INPUT_VERSION,
    EmbeddingError,
    EmbeddingNotConfiguredError,
    call_embeddings,
    embedding_config_from_env,
    embedding_input_fingerprint,
    embedding_text_from_record,
)
from xists.search.index import (
    INDEX_VERSION,
    SUPPORTED_INDEX_VERSIONS,
    decode_vector,
    encode_vector,
    entry_metadata,
    save_index,
)
from xists.search.merge import merge_indices
from xists.search.prune import prune_index
from xists.search.pull import pull_index


def _compute_checkpoint_checksum(vectors: list[dict[str, Any]]) -> str:
    """Compute lightweight CRC32 checksum for vector entries in checkpoint."""
    signatures = [
        f"{entry.get('repo_id')}:{entry.get('embedding_input_fingerprint')}:{len(str(entry.get('vector', '')))}"
        for entry in vectors
        if isinstance(entry, dict)
    ]
    data = "\n".join(signatures).encode("utf-8")
    return f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"


def _index_write_checkpoint(
    output: Path,
    *,
    index_version: int,
    record_schema_version: int,
    embedding_model: str,
    embedding_base_url: str,
    embedding_input_version: int,
    dimension: int | None,
    record_count: int,
    skipped: list[str],
    vectors: list[dict[str, Any]],
) -> None:
    # Index files can be large enough that a reader may otherwise observe a
    # partially truncated JSON document while a checkpoint is being rewritten.
    # Replacing a completed sibling file keeps every visible checkpoint valid.
    checksum = _compute_checkpoint_checksum(vectors)
    write_json_atomic(
        output,
        {
            "index_version": index_version,
            "record_schema_version": record_schema_version,
            "embedding_model": embedding_model,
            "embedding_base_url": embedding_base_url,
            "embedding_input_version": embedding_input_version,
            "dimension": dimension,
            "built_at": datetime.now(UTC).isoformat(),
            "record_count": record_count,
            "skipped": skipped,
            "checkpoint_checksum": checksum,
            "vectors": vectors,
        },
    )


def _load_checkpoint_resilient(path: Path, dimension: int | None = None) -> dict[str, Any]:
    """Load a checkpoint with self-healing recovery for truncated or corrupted files."""
    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            expected_checksum = data.get("checkpoint_checksum")
            vectors = [e for e in data.get("vectors", []) if isinstance(e, dict)]
            if expected_checksum:
                actual_checksum = _compute_checkpoint_checksum(vectors)
                if actual_checksum != expected_checksum:
                    valid_vectors = [
                        e
                        for e in vectors
                        if e.get("repo_id")
                        and decode_vector(e.get("vector"), dimension=dimension) is not None
                    ]
                    print(
                        f"Warning: Checkpoint checksum mismatch at {path}. Auto-healed: retained {len(valid_vectors)}/{len(vectors)} valid vectors.",
                        file=sys.stderr,
                    )
                    data["vectors"] = valid_vectors
                    data["record_count"] = len(valid_vectors)
                    data["checkpoint_checksum"] = _compute_checkpoint_checksum(valid_vectors)
                    write_json_atomic(path, data)
            return data
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        print(
            f"Warning: Checkpoint at {path} is truncated/corrupted ({error}). Attempting self-healing recovery...",
            file=sys.stderr,
        )

    repaired_vectors: list[dict[str, Any]] = []
    # Resilient balanced-brace scanner for vector entries in JSON array
    start_pos = 0
    vectors_pos = raw_text.find('"vectors"')
    if vectors_pos != -1:
        bracket_pos = raw_text.find("[", vectors_pos)
        if bracket_pos != -1:
            start_pos = bracket_pos

    in_string = False
    escape = False
    depth = 0
    obj_start = -1

    for i in range(start_pos, len(raw_text)):
        char = raw_text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start != -1:
                    chunk = raw_text[obj_start : i + 1]
                    try:
                        entry = json.loads(chunk)
                        if isinstance(entry, dict) and entry.get("repo_id"):
                            repaired_vectors.append(entry)
                    except Exception:
                        pass
                    obj_start = -1

    def _extract_field(field_name: str, default: Any, is_int: bool = False) -> Any:
        pattern = rf'"{field_name}"\s*:\s*(\d+)' if is_int else rf'"{field_name}"\s*:\s*"([^"]*)"'
        found = re.search(pattern, raw_text)
        if found:
            return int(found.group(1)) if is_int else found.group(1)
        return default

    model = _extract_field("embedding_model", "")
    dim = _extract_field("dimension", dimension, is_int=True)
    base_url = _extract_field("embedding_base_url", "")
    input_ver = _extract_field("embedding_input_version", EMBEDDING_INPUT_VERSION, is_int=True)
    schema_ver = _extract_field("record_schema_version", RECORD_SCHEMA_VERSION, is_int=True)

    repaired_doc = {
        "index_version": INDEX_VERSION,
        "record_schema_version": schema_ver,
        "embedding_model": model,
        "embedding_base_url": base_url,
        "embedding_input_version": input_ver,
        "dimension": dim,
        "built_at": datetime.now(UTC).isoformat(),
        "record_count": len(repaired_vectors),
        "skipped": [],
        "checkpoint_checksum": _compute_checkpoint_checksum(repaired_vectors),
        "vectors": repaired_vectors,
    }

    write_json_atomic(path, repaired_doc)
    print(
        f"Self-healing complete: Recovered {len(repaired_vectors)} vectors from {path}.",
        file=sys.stderr,
    )
    return repaired_doc


def _index_checkpoint_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.partial.json")


def index_build(args: argparse.Namespace) -> int:
    try:
        config = embedding_config_from_env()
    except EmbeddingNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    if not args.records.exists():
        print(f"Records file not found: {args.records}", file=sys.stderr)
        return 2

    records = json.loads(args.records.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print(f"Records file must contain a JSON list: {args.records}", file=sys.stderr)
        return 1
    validation = records_validation_report(
        records, expected_profile_prompt_version=PROFILE_PROMPT_VERSION
    )
    if not validation["ok"]:
        print(
            f"Records schema/quality validation failed for {args.records}.\n"
            f"Expected record schema_version {RECORD_SCHEMA_VERSION}; "
            f"errors: {validation['errors']}.\n"
            "Next steps:\n"
            f"  1. Refresh profiles: xists profile refresh --records {args.records} --output records-v2.json\n"
            "  2. Rebuild index: xists index build --records records-v2.json --output index.json",
            file=sys.stderr,
        )
        return 1
    batch_size = getattr(args, "batch_size", 64) or 64

    checkpoint_path = _index_checkpoint_path(args.output)
    if args.resume and not checkpoint_path.exists():
        print(f"Checkpoint file not found: {checkpoint_path}", file=sys.stderr)
        return 1
    if checkpoint_path.exists() and not args.resume:
        print(
            f"Checkpoint file already exists: {checkpoint_path}\n"
            "Next steps:\n"
            f"  1. Re-run with --resume to continue from the checkpoint\n"
            f"  2. Delete {checkpoint_path} if you want to restart from scratch",
            file=sys.stderr,
        )
        return 1

    # Load existing index for fingerprint-aware incremental update (skip with --force).
    vectors: list[dict[str, Any]] = []
    skipped: list[str] = []
    dimension: int | None = None
    reusable_vectors: dict[str, dict[str, Any]] = {}

    source_index_path: Path | None = None
    if args.resume:
        source_index_path = checkpoint_path
    elif not args.force and args.output.exists():
        source_index_path = args.output
    if source_index_path is not None:
        if args.resume:
            existing_index = _load_checkpoint_resilient(source_index_path, dimension=dimension)
        else:
            try:
                existing_index = _read_index_file(source_index_path)
            except ValueError:
                existing_index = {}
        if (
            existing_index.get("embedding_model")
            and existing_index["embedding_model"] != config.model
        ):
            print(
                f"Index was built with model '{existing_index['embedding_model']}' "
                f"but configured model is '{config.model}'. "
                f"Delete {args.output} and rebuild, or set EMBEDDING_MODEL to match.",
                file=sys.stderr,
            )
            return 1
        reusable = (
            existing_index.get("embedding_input_version") == EMBEDDING_INPUT_VERSION
            and existing_index.get("record_schema_version") == RECORD_SCHEMA_VERSION
        )
        if reusable:
            dimension = existing_index.get("dimension")
            raw_matrix = existing_index.get("_matrix")
            reusable_vectors = {}
            for i, entry in enumerate(existing_index.get("vectors", [])):
                repo_id = entry.get("repo_id")
                if not repo_id:
                    continue
                if raw_matrix is not None and i < len(raw_matrix):
                    reusable_vectors[repo_id] = {**entry, "vector": raw_matrix[i]}
                else:
                    reusable_vectors[repo_id] = entry

    # Prepare embeddable records and reuse unchanged vectors.
    embeddable: list[dict[str, Any]] = []
    for record in records:
        text = embedding_text_from_record(record)
        repo_id = record.get("repo_id") or record.get("repo_id_requested")
        if not text:
            skipped.append(repo_id or "<unknown>")
            continue
        fingerprint = embedding_input_fingerprint(record)
        metadata = entry_metadata(record)
        existing = reusable_vectors.get(repo_id)
        if existing and existing.get("embedding_input_fingerprint") == fingerprint:
            vector = decode_vector(existing.get("vector"), dimension=dimension)
            if vector is None:
                embeddable.append(
                    {
                        "repo_id": repo_id,
                        "text": text,
                        "fingerprint": fingerprint,
                        "metadata": metadata,
                    }
                )
                continue
            if dimension is None:
                dimension = int(vector.size)
            if vector.size == dimension:
                existing = {**existing, "metadata": metadata}
                vectors.append(existing)
                continue
        embeddable.append(
            {"repo_id": repo_id, "text": text, "fingerprint": fingerprint, "metadata": metadata}
        )

    def write_partial_checkpoint() -> None:
        _write_cp = getattr(
            sys.modules.get("xists.cli"), "_index_write_checkpoint", _index_write_checkpoint
        )
        _write_cp(
            checkpoint_path,
            index_version=INDEX_VERSION,
            record_schema_version=RECORD_SCHEMA_VERSION,
            embedding_model=config.model,
            embedding_base_url=config.base_url,
            embedding_input_version=EMBEDDING_INPUT_VERSION,
            dimension=dimension,
            record_count=len(vectors),
            skipped=skipped,
            vectors=vectors,
        )

    new_count = 0
    checkpoint_every_batches = 16
    concurrency = getattr(args, "concurrency", 1) or 1
    keys_pool = (
        config.api_keys if config.api_keys else ((config.api_key,) if config.api_key else ())
    )

    batches = [
        embeddable[start : start + batch_size] for start in range(0, len(embeddable), batch_size)
    ]
    total_batches = len(batches)

    _call_embeddings = getattr(sys.modules.get("xists.cli"), "call_embeddings", call_embeddings)

    if concurrency > 1 and total_batches > 1:

        def process_batch(
            idx: int, batch: list[dict[str, Any]]
        ) -> tuple[int, list[dict[str, Any]], list[list[float]]]:
            key = keys_pool[idx % len(keys_pool)] if keys_pool else ""
            worker_cfg = config.with_api_key(key) if key else config
            res = _call_embeddings(
                worker_cfg, [item["text"] for item in batch], input_type="passage"
            )
            return idx, batch, res

        batch_results: dict[int, list[dict[str, Any]]] = {}
        batches_done = 0
        error_occurred: Exception | None = None

        with ThreadPoolExecutor(max_workers=min(concurrency, total_batches)) as executor:
            future_to_idx = {
                executor.submit(process_batch, idx, batch): idx for idx, batch in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                try:
                    b_idx, batch, results = future.result()
                except EmbeddingError as error:
                    error_occurred = error
                    break
                except Exception as error:
                    error_occurred = error
                    break

                if len(results) != len(batch):
                    error_occurred = RuntimeError(
                        f"Embedding count mismatch: sent {len(batch)}, received {len(results)}"
                    )
                    break

                batch_vectors = []
                for item, emb_vec in zip(batch, results):
                    if dimension is None:
                        dimension = len(emb_vec)
                    elif len(emb_vec) != dimension:
                        error_occurred = RuntimeError(
                            f"Inconsistent embedding dimension: {len(emb_vec)} vs {dimension}"
                        )
                        break
                    batch_vectors.append(
                        {
                            "repo_id": item["repo_id"],
                            "embedding_input_fingerprint": item["fingerprint"],
                            "metadata": item["metadata"],
                            "vector": encode_vector(emb_vec),
                        }
                    )
                if error_occurred:
                    break

                batch_results[b_idx] = batch_vectors
                batches_done += 1
                new_count += len(batch_vectors)

                if batches_done % checkpoint_every_batches == 0 or batches_done == total_batches:
                    current_vectors = list(vectors)
                    for i in sorted(batch_results.keys()):
                        current_vectors.extend(batch_results[i])
                    _write_cp = getattr(
                        sys.modules.get("xists.cli"),
                        "_index_write_checkpoint",
                        _index_write_checkpoint,
                    )
                    _write_cp(
                        checkpoint_path,
                        index_version=INDEX_VERSION,
                        record_schema_version=RECORD_SCHEMA_VERSION,
                        embedding_model=config.model,
                        embedding_base_url=config.base_url,
                        embedding_input_version=EMBEDDING_INPUT_VERSION,
                        dimension=dimension,
                        record_count=len(current_vectors),
                        skipped=skipped,
                        vectors=current_vectors,
                    )
                    print(
                        f"index progress: {len(current_vectors)}/{len(embeddable) + len(vectors)} embedded",
                        file=sys.stderr,
                        flush=True,
                    )

        if error_occurred:
            current_vectors = list(vectors)
            for i in sorted(batch_results.keys()):
                current_vectors.extend(batch_results[i])
            _write_cp = getattr(
                sys.modules.get("xists.cli"), "_index_write_checkpoint", _index_write_checkpoint
            )
            _write_cp(
                checkpoint_path,
                index_version=INDEX_VERSION,
                record_schema_version=RECORD_SCHEMA_VERSION,
                embedding_model=config.model,
                embedding_base_url=config.base_url,
                embedding_input_version=EMBEDDING_INPUT_VERSION,
                dimension=dimension,
                record_count=len(current_vectors),
                skipped=skipped,
                vectors=current_vectors,
            )
            if isinstance(error_occurred, EmbeddingError):
                _print_embedding_error(error_occurred, command="index build")
            else:
                print(str(error_occurred), file=sys.stderr)
            return 1

        for i in range(total_batches):
            vectors.extend(batch_results[i])
    else:
        for batch_number, batch in enumerate(batches, start=1):
            try:
                results = _call_embeddings(
                    config, [item["text"] for item in batch], input_type="passage"
                )
            except EmbeddingError as error:
                write_partial_checkpoint()
                _print_embedding_error(error, command="index build")
                return 1
            if len(results) != len(batch):
                write_partial_checkpoint()
                print(
                    f"Embedding count mismatch: sent {len(batch)}, received {len(results)}",
                    file=sys.stderr,
                )
                return 1
            for item, vector in zip(batch, results):
                if dimension is None:
                    dimension = len(vector)
                elif len(vector) != dimension:
                    write_partial_checkpoint()
                    print(
                        f"Inconsistent embedding dimension: {len(vector)} vs {dimension}",
                        file=sys.stderr,
                    )
                    return 1
                vectors.append(
                    {
                        "repo_id": item["repo_id"],
                        "embedding_input_fingerprint": item["fingerprint"],
                        "metadata": item["metadata"],
                        "vector": encode_vector(vector),
                    }
                )
                new_count += 1

            if batch_number % checkpoint_every_batches == 0:
                write_partial_checkpoint()
                print(
                    f"index progress: {len(vectors)}/{len(embeddable) + len(vectors) - new_count} embedded",
                    file=sys.stderr,
                    flush=True,
                )

    save_index(
        args.output,
        {
            "index_version": INDEX_VERSION,
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "embedding_model": config.model,
            "embedding_base_url": config.base_url,
            "embedding_input_version": EMBEDDING_INPUT_VERSION,
            "dimension": dimension,
            "built_at": datetime.now(UTC).isoformat(),
            "record_count": len(vectors),
            "skipped": skipped,
            "vectors": vectors,
        },
        version=INDEX_VERSION,
        quantize=getattr(args, "quantize", "none"),
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint_npy = checkpoint_path.with_name(f"{checkpoint_path.stem}.vectors.npy")
    if checkpoint_npy.exists():
        checkpoint_npy.unlink()

    payload = {
        "index": str(args.output),
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_model": config.model,
        "dimension": dimension,
        "record_count": len(vectors),
        "new_vectors": new_count,
        "skipped": skipped,
    }
    if getattr(args, "format", "text") == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            _format_command_summary(
                "Index built",
                [
                    ("File", args.output),
                    ("Projects", len(vectors)),
                    ("New vectors", new_count),
                    ("Skipped", len(skipped)),
                    ("Model", config.model),
                    ("Dimensions", dimension),
                ],
                stream=sys.stdout,
            )
        )
    return 0


def _index_stats_report(index: dict[str, Any], *, index_path: Path, limit: int) -> dict[str, Any]:
    vectors = index.get("vectors") or []
    languages: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    missing_metadata = 0
    missing_fingerprints = 0
    for entry in vectors:
        if not isinstance(entry, dict):
            continue
        if not entry.get("embedding_input_fingerprint"):
            missing_fingerprints += 1
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            missing_metadata += 1
            continue
        language = metadata.get("language")
        if isinstance(language, str) and language.strip():
            languages[language] += 1
        for topic in metadata.get("topics") or []:
            if isinstance(topic, str) and topic.strip():
                topics[topic] += 1

    dimension = index.get("dimension")
    vector_dtype = index.get("vector_dtype") or (
        str(index["_matrix"].dtype) if index.get("_matrix") is not None else "float32"
    )
    vector_quantization = index.get("vector_quantization") or (
        "sq8" if vector_dtype == "int8" else "none"
    )
    bytes_per_elem = 1 if vector_dtype == "int8" else (2 if vector_dtype == "float16" else 4)
    estimated_memory_mb: float | None = None
    if isinstance(dimension, int) and dimension > 0:
        estimated_memory_mb = round(len(vectors) * dimension * bytes_per_elem / 1024 / 1024, 1)

    payload = {
        "index": str(index_path),
        "index_version": index.get("index_version"),
        "record_schema_version": index.get("record_schema_version"),
        "embedding_model": index.get("embedding_model"),
        "embedding_base_url": index.get("embedding_base_url"),
        "embedding_input_version": index.get("embedding_input_version"),
        "dimension": index.get("dimension"),
        "vector_dtype": vector_dtype,
        "vector_quantization": vector_quantization,
        "built_at": index.get("built_at"),
        "record_count": index.get("record_count"),
        "vector_count": len(vectors),
        "estimated_memory_mb": estimated_memory_mb,
        "skipped_count": len(index.get("skipped") or []),
        "missing_metadata_count": missing_metadata,
        "missing_fingerprint_count": missing_fingerprints,
        "top_languages": _counter_items(languages, "language", limit),
        "top_topics": _counter_items(topics, "topic", limit),
    }
    return payload


def _format_index_stats_text(report: dict[str, Any]) -> str:
    quant_info = report.get("vector_quantization") or "none"
    dtype_info = report.get("vector_dtype") or "float32"
    format_label = f"{dtype_info} ({quant_info})" if quant_info != "none" else dtype_info
    return _format_command_summary(
        "Index",
        [
            ("File", report["index"]),
            ("Projects", report.get("record_count")),
            ("Vectors", report.get("vector_count")),
            ("Format", format_label),
            ("Model", report.get("embedding_model")),
            ("Dimensions", report.get("dimension")),
            ("Built", report.get("built_at")),
            (
                "Memory",
                f"{report['estimated_memory_mb']} MB"
                if report.get("estimated_memory_mb") is not None
                else "unknown",
            ),
            ("Skipped", report.get("skipped_count")),
            ("Languages", _format_top_items(report.get("top_languages") or [], "language")),
            ("Topics", _format_top_items(report.get("top_topics") or [], "topic")),
        ],
    )


def index_stats(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(
            f"Index file not found: {args.index}. Run 'xists index build' first.", file=sys.stderr
        )
        return 2

    try:
        index = _read_index_file(args.index)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    payload = _index_stats_report(index, index_path=args.index, limit=args.limit)
    if getattr(args, "format", "text") == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_format_index_stats_text(payload))
    return 0


def _index_verify_report(records: list[dict[str, Any]], index: dict[str, Any]) -> dict[str, Any]:
    record_validation = records_validation_report(
        records, expected_profile_prompt_version=PROFILE_PROMPT_VERSION
    )
    errors: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    if not record_validation["ok"]:
        errors["records_validation_failed"] = sum(record_validation["errors"].values())
    if index.get("index_version") not in SUPPORTED_INDEX_VERSIONS:
        errors["index_version_mismatch"] += 1
    if index.get("record_schema_version") != RECORD_SCHEMA_VERSION:
        errors["record_schema_version_mismatch"] += 1
    if index.get("embedding_input_version") != EMBEDDING_INPUT_VERSION:
        errors["embedding_input_version_mismatch"] += 1

    vectors = [entry for entry in index.get("vectors") or [] if isinstance(entry, dict)]
    vector_by_id = {entry.get("repo_id"): entry for entry in vectors if entry.get("repo_id")}
    dimension = index.get("dimension")
    if not isinstance(dimension, int) or dimension <= 0:
        errors["dimension_missing"] += 1
    missing_fingerprints = [
        entry.get("repo_id") for entry in vectors if not entry.get("embedding_input_fingerprint")
    ]
    if missing_fingerprints:
        errors["missing_fingerprints"] = len(missing_fingerprints)

    dimension_mismatches: list[str] = []
    invalid_vectors: list[str] = []
    if index.get("index_version") == 4 and index.get("_matrix") is not None:
        mat = index["_matrix"]
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            invalid_vectors = [
                str(entry.get("repo_id")) for entry in vectors if entry.get("repo_id")
            ]
        elif isinstance(dimension, int) and mat.shape[1] != dimension:
            dimension_mismatches = [
                str(entry.get("repo_id")) for entry in vectors if entry.get("repo_id")
            ]
        elif mat.shape[0] != len(vectors):
            invalid_vectors = [f"matrix_rows_{mat.shape[0]}_vs_vectors_{len(vectors)}"]
    else:
        dimension_mismatches = [
            str(entry.get("repo_id"))
            for entry in vectors
            if isinstance(dimension, int)
            and (decoded := decode_vector(entry.get("vector"))) is not None
            and decoded.size != dimension
            and entry.get("repo_id")
        ]
        invalid_vectors = [
            str(entry.get("repo_id"))
            for entry in vectors
            if decode_vector(entry.get("vector")) is None and entry.get("repo_id")
        ]

    if dimension_mismatches:
        errors["dimension_mismatch"] = len(dimension_mismatches)
    if invalid_vectors:
        errors["invalid_vectors"] = len(invalid_vectors)
    if isinstance(index.get("record_count"), int) and index.get("record_count") != len(vectors):
        warnings["record_count_mismatch"] += 1

    record_ids = {record_repo_id(record) for record in records if record_repo_id(record)}
    missing_vectors: list[str] = []
    stale_vectors: list[str] = []
    skipped_expected: list[str] = []
    for record in records:
        repo_id = record_repo_id(record)
        if not repo_id:
            continue
        fingerprint = embedding_input_fingerprint(record)
        if fingerprint is None or not embedding_text_from_record(record):
            skipped_expected.append(repo_id)
            continue
        entry = vector_by_id.get(repo_id)
        if entry is None:
            missing_vectors.append(repo_id)
        elif entry.get("embedding_input_fingerprint") != fingerprint:
            stale_vectors.append(repo_id)
    extra_vectors = sorted(str(repo_id) for repo_id in vector_by_id if repo_id not in record_ids)
    if missing_vectors:
        errors["missing_vectors"] = len(missing_vectors)
    if stale_vectors:
        errors["stale_vectors"] = len(stale_vectors)
    if extra_vectors:
        warnings["extra_vectors"] = len(extra_vectors)

    ok = not errors
    stale_only = set(errors).issubset({"missing_vectors", "stale_vectors"}) and (
        missing_vectors or stale_vectors or extra_vectors
    )
    return {
        "ok": ok,
        "status": "ok" if ok else ("stale" if stale_only else "invalid"),
        "index_version": index.get("index_version"),
        "expected_index_version": INDEX_VERSION,
        "record_schema_version": index.get("record_schema_version"),
        "expected_record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_input_version": index.get("embedding_input_version"),
        "expected_embedding_input_version": EMBEDDING_INPUT_VERSION,
        "record_count": len(records),
        "vector_count": len(vectors),
        "errors": dict(errors),
        "warnings": dict(warnings),
        "missing_fingerprints": missing_fingerprints,
        "dimension_mismatches": dimension_mismatches,
        "invalid_vectors": invalid_vectors,
        "missing_vectors": missing_vectors,
        "stale_vectors": stale_vectors,
        "extra_vectors": extra_vectors,
        "skipped_expected": skipped_expected,
        "records_validation": record_validation,
        "next_steps": []
        if ok
        else [
            "Refresh profiles if records are old: xists profile refresh --records records.json --output records-v2.json",
            "Rebuild the index: xists index build --records records-v2.json --output index.json",
        ],
    }


def _format_index_verify_text(report: dict[str, Any], records_path: Path, index_path: Path) -> str:
    lines = [
        f"records: {records_path}",
        f"index: {index_path}",
        f"status: {report['status']}",
        f"ok: {str(report['ok']).lower()}",
        f"records: {report['record_count']}",
        f"vectors: {report['vector_count']}",
    ]
    for label in ("errors", "warnings"):
        lines.append(f"{label}:")
        items = report.get(label) or {}
        if items:
            for key, value in sorted(items.items()):
                lines.append(f"  {key}: {value}")
        else:
            lines.append("  none")
    if report.get("next_steps"):
        lines.append("next steps:")
        for step in report["next_steps"]:
            lines.append(f"  - {step}")
    return "\n".join(lines)


def index_verify(args: argparse.Namespace) -> int:
    if not args.records.exists():
        print(
            f"Records file not found: {args.records}. Run 'xists ingest github' first.",
            file=sys.stderr,
        )
        return 2
    if not args.index.exists():
        print(
            f"Index file not found: {args.index}. Run 'xists index build' first.", file=sys.stderr
        )
        return 2
    try:
        records = _read_records_file(args.records)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    try:
        index = _read_index_file(args.index)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    report = _index_verify_report(records, index)
    report["records"] = str(args.records)
    report["index"] = str(args.index)
    if getattr(args, "format", "text") == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format_index_verify_text(report, args.records, args.index))
    return 0 if report["ok"] else 1


def index_migrate(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"Input index file not found: {args.input}", file=sys.stderr)
        return 2

    if not args.output and not args.output_dir:
        print(
            "Must specify either --output or --output-dir for migration destination.",
            file=sys.stderr,
        )
        return 2

    try:
        index = _read_index_file(args.input)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    source_version = index.get("index_version")
    if source_version not in SUPPORTED_INDEX_VERSIONS:
        print(
            f"Unsupported index_version {source_version!r} in {args.input}. "
            f"Expected one of {SUPPORTED_INDEX_VERSIONS}.",
            file=sys.stderr,
        )
        return 1

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "index.json"
    else:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

    dimension = index.get("dimension")
    vectors = index.get("vectors") or []
    if not isinstance(vectors, list):
        print(f"Invalid index vectors in {args.input}", file=sys.stderr)
        return 1

    if index.get("_matrix") is not None:
        matrix = np.asarray(index["_matrix"], dtype=np.float32)
    else:
        vectors_list: list[np.ndarray] = []
        for entry in vectors:
            vec = decode_vector(entry.get("vector"), dimension=dimension)
            if vec is None:
                print(f"Failed to decode vector for repo {entry.get('repo_id')}", file=sys.stderr)
                return 1
            vectors_list.append(vec)
        matrix = (
            np.asarray(vectors_list, dtype=np.float32)
            if vectors_list
            else np.empty((0, dimension or 0), dtype=np.float32)
        )

    quantize_mode = getattr(args, "quantize", "none")
    save_index(output_path, index, matrix=matrix, version=4, quantize=quantize_mode)

    input_size = args.input.stat().st_size
    output_json_size = output_path.stat().st_size
    vectors_file = output_path.with_name(f"{output_path.stem}.vectors.npy")
    output_npy_size = vectors_file.stat().st_size if vectors_file.exists() else 0
    scales_file = output_path.with_name(f"{output_path.stem}.scales.npy")
    scales_size = scales_file.stat().st_size if scales_file.exists() else 0
    total_output_size = output_json_size + output_npy_size + scales_size
    reduction_pct = ((input_size - total_output_size) / input_size * 100) if input_size > 0 else 0.0

    payload = {
        "input": str(args.input),
        "output": str(output_path),
        "vectors_file": str(vectors_file),
        "scales_file": str(scales_file) if scales_size > 0 else None,
        "vector_quantization": quantize_mode,
        "source_version": source_version,
        "target_version": 4,
        "record_count": len(vectors),
        "dimension": dimension,
        "source_size_bytes": input_size,
        "target_size_bytes": total_output_size,
        "target_json_bytes": output_json_size,
        "target_vectors_bytes": output_npy_size + scales_size,
        "size_reduction_percent": round(reduction_pct, 2),
    }

    if getattr(args, "format", "text") == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:

        def _format_bytes(num: float) -> str:
            for unit in ("B", "KB", "MB", "GB"):
                if num < 1024.0:
                    return f"{num:.1f} {unit}" if unit != "B" else f"{num} B"
                num /= 1024.0
            return f"{num:.1f} TB"

        print(
            _format_command_summary(
                "Index migrated",
                [
                    ("Source", args.input),
                    ("Output", output_path),
                    ("Vectors file", vectors_file),
                    ("Records", len(vectors)),
                    ("Dimensions", dimension),
                    ("Source size", _format_bytes(input_size)),
                    (
                        "Target size",
                        f"{_format_bytes(total_output_size)} ({reduction_pct:.1f}% reduction)",
                    ),
                    ("Format", "INDEX_VERSION 4 (dual-file binary)"),
                ],
                stream=sys.stdout,
            )
        )
    return 0


def index_pull(args: argparse.Namespace) -> int:
    try:
        result = pull_index(
            args.source,
            args.output_dir,
            sha256=args.sha256,
            force=args.force,
        )
    except FileExistsError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as error:
        print(f"Failed to pull index: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Failed to pull index: {error}", file=sys.stderr)
        return 1

    if getattr(args, "format", "text") == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    rows = [
        ("Source", result.get("source")),
        ("Output directory", result.get("output_dir")),
        ("Records", result.get("records_path")),
        ("Index", result.get("index_path")),
        ("Vectors", result.get("vectors_path") or "n/a"),
        ("Records count", result.get("records_count")),
        (
            "Index version",
            f"v{result.get('index_version')} (binary mmap)"
            if result.get("index_version") == 4
            else result.get("index_version"),
        ),
        ("Dimension", result.get("dimension") or "n/a"),
        ("SHA-256", result.get("sha256")),
    ]
    print(_format_command_summary("Index pulled successfully", rows, stream=sys.stdout))
    print("\nNext steps")
    print('  1. Try search: xists search "fast web framework"')
    print("  2. Inspect index: xists index stats")
    return 0


def index_append(args: argparse.Namespace) -> int:
    if not args.repo and not args.input:
        print(
            "Error: Either --repo owner/repo or --input records.json must be specified.",
            file=sys.stderr,
        )
        return 2

    if not args.index.exists():
        print(
            f"Target index file not found: {args.index}. Run 'xists index build' first.",
            file=sys.stderr,
        )
        return 2

    try:
        if args.repo:
            result = append_repo_to_index(
                args.repo,
                index_path=args.index,
                records_path=args.records if args.records and args.records.exists() else None,
                force=args.force,
            )
        else:
            if not args.input.is_file():
                print(f"Input records file not found: {args.input}", file=sys.stderr)
                return 2
            records = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                print(
                    f"Input records must be a JSON array of records: {args.input}", file=sys.stderr
                )
                return 2
            result = append_records_to_index(
                records,
                index_path=args.index,
                records_path=args.records if args.records and args.records.exists() else None,
                force=args.force,
            )
    except EmbeddingNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError, GitHubAPIError) as error:
        print(f"Failed to append to index: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Failed to append to index: {error}", file=sys.stderr)
        return 1

    if getattr(args, "format", "text") == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    rows = [
        ("Target index", result.get("index_path")),
        ("Target records", result.get("records_path") or "n/a"),
        ("Added records", result.get("added")),
        ("Updated records", result.get("updated")),
        ("Skipped records", result.get("skipped")),
        ("Total vectors", result.get("total_vectors")),
        ("Dimension", result.get("dimension")),
        ("Elapsed time", f"{result.get('elapsed_ms')} ms"),
    ]
    print(_format_command_summary("Incremental append complete", rows, stream=sys.stdout))
    return 0


def index_merge(args: argparse.Namespace) -> int:
    if len(args.indices) < 2:
        print("Error: At least two index files are required to merge.", file=sys.stderr)
        return 2

    try:
        result = merge_indices(
            index_paths=args.indices,
            output_index_path=args.output,
            records_paths=args.records,
            output_records_path=args.output_records,
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"Failed to merge indices: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Failed to merge indices: {error}", file=sys.stderr)
        return 1

    if getattr(args, "format", "text") == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    rows = [
        ("Input indices count", result.get("input_indices_count")),
        ("Merged records count", result.get("merged_records_count")),
        ("Duplicate conflicts resolved", result.get("duplicates_resolved")),
        ("Embedding model", result.get("embedding_model")),
        ("Dimension", result.get("dimension")),
        ("Output index", result.get("output_index")),
        ("Output records", result.get("output_records") or "n/a"),
        ("Elapsed time", f"{result.get('elapsed_ms')} ms"),
    ]
    print(_format_command_summary("Index merge complete", rows, stream=sys.stdout))
    return 0


def index_prune(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Target index file not found: {args.index}", file=sys.stderr)
        return 2

    blocklist: list[str] = list(args.remove_repo or [])
    if args.blocklist and args.blocklist.is_file():
        try:
            lines = [
                line.strip()
                for line in args.blocklist.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
            blocklist.extend(lines)
        except Exception as error:
            print(f"Failed to read blocklist file: {error}", file=sys.stderr)
            return 2

    try:
        result = prune_index(
            index_path=args.index,
            records_path=args.records if args.records and args.records.exists() else None,
            prune_archived=args.archived,
            prune_disabled=args.disabled,
            blocklist_repos=blocklist,
            dry_run=args.dry_run,
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"Failed to prune index: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Failed to prune index: {error}", file=sys.stderr)
        return 1

    if getattr(args, "format", "text") == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    header = "Index health pruning (dry run)" if args.dry_run else "Index health pruning complete"
    reasons = result.get("reasons", {})
    rows = [
        ("Target index", result.get("index_path")),
        ("Target records", result.get("records_path") or "n/a"),
        ("Total records before", result.get("total_before")),
        ("Pruned count", result.get("pruned_count")),
        ("Retained count", result.get("retained_count")),
        (
            "Prune reasons",
            f"archived={reasons.get('archived', 0)}, disabled={reasons.get('disabled', 0)}, blocklist={reasons.get('blocklist', 0)}",
        ),
        ("Elapsed time", f"{result.get('elapsed_ms')} ms"),
    ]
    print(_format_command_summary(header, rows, stream=sys.stdout))
    return 0


def index_quantize(args: argparse.Namespace) -> int:
    """Quantize an existing vector index to float16 or sq8 (int8) to reduce storage and memory."""
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Index file not found: {index_path}", file=sys.stderr)
        return 2

    mode = getattr(args, "mode", "float16")
    output_path = Path(args.output) if getattr(args, "output", None) else index_path

    try:
        from xists.search.index import load_index

        index = load_index(index_path, mmap=True)
    except Exception as error:
        print(f"Failed to load index {index_path}: {error}", file=sys.stderr)
        return 1

    matrix = index.get("_matrix")
    if matrix is None or not isinstance(matrix, np.ndarray):
        print(f"No vector matrix found for index {index_path}", file=sys.stderr)
        return 1

    old_size = 0
    if index.get("_vectors_path") and Path(index["_vectors_path"]).exists():
        old_size = Path(index["_vectors_path"]).stat().st_size

    save_index(output_path, index, matrix=matrix, version=4, quantize=mode)

    new_vectors_path = output_path.with_name(f"{output_path.stem}.vectors.npy")
    new_size = new_vectors_path.stat().st_size if new_vectors_path.exists() else 0
    scales_path = output_path.with_name(f"{output_path.stem}.scales.npy")
    scales_size = scales_path.stat().st_size if scales_path.exists() else 0
    total_new_size = new_size + scales_size

    reduction = ((old_size - total_new_size) / old_size * 100) if old_size > 0 else 0.0

    payload = {
        "index": str(output_path),
        "mode": mode,
        "record_count": index.get("record_count", len(matrix)),
        "dimension": matrix.shape[1] if matrix.ndim == 2 else None,
        "original_vector_bytes": old_size,
        "quantized_vector_bytes": total_new_size,
        "size_reduction_percent": round(reduction, 2),
    }

    if getattr(args, "format", "text") == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Successfully quantized index {index_path} -> {output_path}")
        print(f"  Mode: {mode}")
        print(f"  Original size: {old_size / (1024 * 1024):.2f} MB")
        print(f"  New size: {total_new_size / (1024 * 1024):.2f} MB ({reduction:.1f}% reduction)")
    return 0


__all__ = [
    "_compute_checkpoint_checksum",
    "_format_index_stats_text",
    "_format_index_verify_text",
    "_index_checkpoint_path",
    "_index_stats_report",
    "_index_verify_report",
    "_index_write_checkpoint",
    "_load_checkpoint_resilient",
    "index_append",
    "index_build",
    "index_merge",
    "index_migrate",
    "index_prune",
    "index_pull",
    "index_quantize",
    "index_stats",
    "index_verify",
]
