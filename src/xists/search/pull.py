"""Public index distribution and pull management for xists."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from xists.search.index import load_index
from xists.starter import (
    STARTER_INDEX_PATH,
    STARTER_RECORDS_PATH,
    STARTER_VECTORS_PATH,
    load_starter_records,
)

INDEX_PRESETS: dict[str, dict[str, Any]] = {
    "demo": {
        "description": "Bundled 200 top open-source repositories with pre-built binary index (zero-config)",
        "type": "bundled",
        "records_count": 200,
    },
    "starter": {
        "description": "Alias for demo starter dataset",
        "type": "bundled",
        "records_count": 200,
    },
    "curated-1k": {
        "description": "Curated 1,000 top GitHub open-source repositories with embeddings",
        "type": "remote",
        "url": "https://raw.githubusercontent.com/UsonTong/xists/main/examples/retrieval-regression/index.json",
        "records_url": "https://raw.githubusercontent.com/UsonTong/xists/main/examples/retrieval-regression/records.json",
        "records_count": 55,
    },
    "github-top-1k": {
        "description": "Curated top GitHub repositories",
        "type": "remote",
        "url": "https://raw.githubusercontent.com/UsonTong/xists/main/examples/retrieval-regression/index.json",
        "records_url": "https://raw.githubusercontent.com/UsonTong/xists/main/examples/retrieval-regression/records.json",
        "records_count": 55,
    },
}


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> list[Path]:
    """Safely extract tar archive preventing path traversal."""
    extracted_files: list[Path] = []
    dest_resolved = destination.resolve()
    for member in tar.getmembers():
        target_path = (destination / member.name).resolve()
        if not str(target_path).startswith(str(dest_resolved)):
            raise ValueError(f"Path traversal detected in archive member: {member.name}")
        try:
            tar.extract(member, destination, filter="data")
        except TypeError:
            tar.extract(member, destination)
        if target_path.is_file():
            extracted_files.append(target_path)
    return extracted_files


def _safe_extract_zip(zip_file: zipfile.ZipFile, destination: Path) -> list[Path]:
    """Safely extract zip archive preventing path traversal."""
    extracted_files: list[Path] = []
    dest_resolved = destination.resolve()
    for member in zip_file.namelist():
        target_path = (destination / member).resolve()
        if not str(target_path).startswith(str(dest_resolved)):
            raise ValueError(f"Path traversal detected in archive member: {member}")
        zip_file.extract(member, destination)
        if target_path.is_file():
            extracted_files.append(target_path)
    return extracted_files


def _download_url(url: str, dest_path: Path) -> None:
    """Download a remote URL to a local file."""
    headers = {"User-Agent": "xists-index-pull/0.12.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response, dest_path.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)


def pull_index(
    source: str,
    output_dir: Path | str,
    *,
    sha256: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Pull an index and records from preset, URL, or bundled starter.

    Parameters:
        source: Preset name (e.g. 'demo', 'starter', 'curated-1k') or remote URL.
        output_dir: Target directory to place index and records files.
        sha256: Optional expected SHA-256 hex checksum for verification.
        force: Overwrite existing files if True.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dest_records = out_dir / "records.json"
    dest_index = out_dir / "index.json"
    dest_vectors = out_dir / "index.vectors.npy"

    if not force:
        existing = [p.name for p in (dest_records, dest_index, dest_vectors) if p.exists()]
        if existing:
            raise FileExistsError(
                f"Target files already exist in {out_dir}: {', '.join(existing)}. Use --force to overwrite."
            )

    source_norm = source.strip().lower()

    # 1. Bundled preset (demo / starter)
    if source_norm in ("demo", "starter"):
        if not STARTER_INDEX_PATH.is_file() or not STARTER_RECORDS_PATH.is_file():
            raise FileNotFoundError("Bundled starter files are missing from installation.")

        shutil.copy2(STARTER_RECORDS_PATH, dest_records)
        shutil.copy2(STARTER_INDEX_PATH, dest_index)
        if STARTER_VECTORS_PATH.is_file():
            shutil.copy2(STARTER_VECTORS_PATH, dest_vectors)

        index_data = load_index(dest_index)
        records_data = load_starter_records()
        checksum = compute_file_sha256(dest_index)

        if sha256 and checksum.lower() != sha256.lower():
            # Clean up on checksum failure
            dest_records.unlink(missing_ok=True)
            dest_index.unlink(missing_ok=True)
            dest_vectors.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 checksum mismatch: expected {sha256}, got {checksum}")

        return {
            "source": source,
            "preset": source_norm,
            "output_dir": str(out_dir),
            "records_path": str(dest_records),
            "index_path": str(dest_index),
            "vectors_path": str(dest_vectors) if dest_vectors.is_file() else None,
            "records_count": len(records_data),
            "index_version": index_data.get("index_version", 4),
            "record_count": index_data.get("record_count", len(records_data)),
            "dimension": index_data.get("dimension"),
            "sha256": checksum,
        }

    # 2. Known remote preset
    preset_info = INDEX_PRESETS.get(source_norm)
    if preset_info and preset_info.get("type") == "remote":
        url = preset_info["url"]
        records_url = preset_info.get("records_url")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tmp_index = tmp_path / "index.json"
            _download_url(url, tmp_index)

            checksum = compute_file_sha256(tmp_index)
            if sha256 and checksum.lower() != sha256.lower():
                raise ValueError(f"SHA-256 checksum mismatch: expected {sha256}, got {checksum}")

            tmp_records = None
            if records_url:
                tmp_records = tmp_path / "records.json"
                _download_url(records_url, tmp_records)

            # Check if index references sidecar vectors
            idx_doc = json.loads(tmp_index.read_text(encoding="utf-8"))
            if idx_doc.get("index_version") == 4 and "vectors_file" in idx_doc:
                vec_filename = idx_doc["vectors_file"]
                vec_url = url.rsplit("/", 1)[0] + "/" + vec_filename
                tmp_vec = tmp_path / vec_filename
                try:
                    _download_url(vec_url, tmp_vec)
                except Exception:
                    pass

            # Move files into destination atomically
            shutil.copy2(tmp_index, dest_index)
            if tmp_records and tmp_records.is_file():
                shutil.copy2(tmp_records, dest_records)
            for f in tmp_path.glob("*.vectors.npy"):
                shutil.copy2(f, out_dir / f.name)

            final_index = load_index(dest_index)
            return {
                "source": source,
                "preset": source_norm,
                "output_dir": str(out_dir),
                "records_path": str(dest_records) if dest_records.is_file() else None,
                "index_path": str(dest_index),
                "vectors_path": str(dest_vectors) if dest_vectors.is_file() else None,
                "records_count": final_index.get("record_count", 0),
                "index_version": final_index.get("index_version", 4),
                "dimension": final_index.get("dimension"),
                "sha256": checksum,
            }

    # 3. Direct URL or archive file
    if source.startswith(("http://", "https://")):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive_name = source.split("/")[-1].split("?")[0] or "download.tar.gz"
            tmp_file = tmp_path / archive_name
            _download_url(source, tmp_file)

            checksum = compute_file_sha256(tmp_file)
            if sha256 and checksum.lower() != sha256.lower():
                raise ValueError(f"SHA-256 checksum mismatch: expected {sha256}, got {checksum}")

            # Extract or copy depending on file type
            if archive_name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
                with tarfile.open(tmp_file, "r:*") as tar:
                    _safe_extract_tar(tar, out_dir)
            elif archive_name.endswith(".zip"):
                with zipfile.ZipFile(tmp_file, "r") as zf:
                    _safe_extract_zip(zf, out_dir)
            elif archive_name.endswith(".json"):
                shutil.copy2(tmp_file, dest_index)
            else:
                raise ValueError(f"Unsupported index file format from URL: {archive_name}")

            final_index = load_index(dest_index) if dest_index.is_file() else {}
            return {
                "source": source,
                "output_dir": str(out_dir),
                "records_path": str(dest_records) if dest_records.is_file() else None,
                "index_path": str(dest_index) if dest_index.is_file() else None,
                "vectors_path": str(dest_vectors) if dest_vectors.is_file() else None,
                "records_count": final_index.get("record_count", 0),
                "index_version": final_index.get("index_version", 4),
                "dimension": final_index.get("dimension"),
                "sha256": checksum,
            }

    raise ValueError(
        f"Unknown index source or preset: '{source}'. Available presets: {', '.join(sorted(INDEX_PRESETS.keys()))}"
    )
