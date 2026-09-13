"""SQLite-backed metadata sidecar database for zero-latency index cold-start and on-demand entry retrieval."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

META_DB_SUFFIX = ".meta.db"


def cache_to_serializable(cache: dict[str, Any]) -> dict[str, Any]:
    """Convert a precomputed entry cache into a JSON-serializable dictionary."""
    data = dict(cache)
    if "identity_values_lower" in data and isinstance(
        data["identity_values_lower"], (set, frozenset)
    ):
        data["identity_values_lower"] = list(data["identity_values_lower"])
    if "text_tokens" in data and isinstance(data["text_tokens"], (set, frozenset)):
        data["text_tokens"] = list(data["text_tokens"])
    if "topic_tokens" in data and isinstance(data["topic_tokens"], (set, frozenset)):
        data["topic_tokens"] = list(data["topic_tokens"])
    if "profile_tokens" in data and isinstance(data["profile_tokens"], (set, frozenset)):
        data["profile_tokens"] = list(data["profile_tokens"])
    if "ecosystem_set" in data and isinstance(data["ecosystem_set"], (set, frozenset)):
        data["ecosystem_set"] = list(data["ecosystem_set"])
    if "topics_set" in data and isinstance(data["topics_set"], (set, frozenset)):
        data["topics_set"] = list(data["topics_set"])
    if "replaces_set" in data and isinstance(data["replaces_set"], (set, frozenset)):
        data["replaces_set"] = list(data["replaces_set"])
    if "id_value_tokens" in data and isinstance(data["id_value_tokens"], (tuple, list)):
        data["id_value_tokens"] = list(data["id_value_tokens"])
    return data


def serializable_to_cache(data: dict[str, Any]) -> dict[str, Any]:
    """Reconstitute a precomputed entry cache from a deserialized dictionary with set types."""
    cache = dict(data)
    id_lower = data.get("identity_values_lower")
    cache["identity_values_lower"] = set(id_lower) if id_lower else set()
    text_tok = data.get("text_tokens")
    cache["text_tokens"] = set(text_tok) if text_tok else set()
    topic_tok = data.get("topic_tokens")
    cache["topic_tokens"] = set(topic_tok) if topic_tok else set()
    prof_tok = data.get("profile_tokens")
    cache["profile_tokens"] = set(prof_tok) if prof_tok else set()
    eco_set = data.get("ecosystem_set")
    cache["ecosystem_set"] = frozenset(eco_set) if eco_set else frozenset()
    top_set = data.get("topics_set")
    cache["topics_set"] = frozenset(top_set) if top_set else frozenset()
    rep_set = data.get("replaces_set")
    cache["replaces_set"] = frozenset(rep_set) if rep_set else frozenset()
    id_tok = data.get("id_value_tokens")
    cache["id_value_tokens"] = tuple(id_tok) if id_tok else ()
    return cache


def init_meta_db_schema(conn: sqlite3.Connection) -> None:
    """Initialize metadata sidecar tables and indexes."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS manifest (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS records (
            doc_id INTEGER PRIMARY KEY,
            repo_id TEXT UNIQUE NOT NULL,
            name TEXT,
            language TEXT,
            stars INTEGER DEFAULT 0,
            forks INTEGER DEFAULT 0,
            archived INTEGER DEFAULT 0,
            disabled INTEGER DEFAULT 0,
            cache_json TEXT,
            entry_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_records_repo_id ON records (repo_id);
        """
    )


class MetaDatabase:
    """Read/write wrapper around the SQLite metadata sidecar database."""

    def __init__(self, db_path: Path | str, *, read_only: bool = True) -> None:
        self.db_path = Path(db_path).resolve()
        self.read_only = read_only
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.read_only:
                uri = f"file:{self.db_path.as_posix()}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
                conn.execute("PRAGMA query_only = ON;")
                conn.execute("PRAGMA temp_store = MEMORY;")
            else:
                conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                init_meta_db_schema(conn)
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> MetaDatabase:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def get_manifest(self) -> dict[str, Any]:
        """Fetch all key-value entries from manifest table."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM manifest")
        manifest: dict[str, Any] = {}
        for key, value_str in cursor.fetchall():
            try:
                manifest[key] = json.loads(value_str)
            except (json.JSONDecodeError, TypeError):
                manifest[key] = value_str
        return manifest

    def get_doc_count(self) -> int:
        """Return total number of records in database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM records")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def get_repo_ids(self) -> list[str]:
        """Return all repository IDs ordered by doc_id."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT repo_id FROM records ORDER BY doc_id ASC")
        return [str(row[0]) for row in cursor.fetchall()]

    def load_search_state(
        self,
    ) -> tuple[tuple[str, ...], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
        """Bulk load precomputed caches and filtering arrays in a single fast query."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT doc_id, repo_id, stars, forks, archived, disabled, cache_json
            FROM records
            ORDER BY doc_id ASC
            """
        )
        rows = cursor.fetchall()
        count = len(rows)

        repo_ids: list[str] = []
        caches: list[dict[str, Any]] = []
        stars_arr = np.empty(count, dtype=np.int64)
        forks_arr = np.empty(count, dtype=np.int64)
        archived_arr = np.empty(count, dtype=bool)

        for i, (_doc_id, repo_id, stars, forks, archived, disabled, cache_json) in enumerate(rows):
            repo_ids.append(repo_id)
            stars_arr[i] = stars or 0
            forks_arr[i] = forks or 0
            archived_arr[i] = bool(archived or disabled)
            if cache_json:
                try:
                    raw_cache = json.loads(cache_json)
                    caches.append(serializable_to_cache(raw_cache))
                except Exception:
                    caches.append({})
            else:
                caches.append({})

        return tuple(repo_ids), caches, stars_arr, forks_arr, archived_arr

    def fetch_entry(self, doc_id: int) -> dict[str, Any] | None:
        """Fetch full entry JSON for a single doc_id."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT entry_json FROM records WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if not row or not row[0]:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def fetch_entries_batch(self, doc_ids: Sequence[int | np.integer]) -> dict[int, dict[str, Any]]:
        """Fetch full entry JSON for multiple doc_ids in a single query."""
        if not doc_ids:
            return {}
        unique_ids = list({int(i) for i in doc_ids})
        placeholders = ",".join("?" for _ in unique_ids)
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT doc_id, entry_json FROM records WHERE doc_id IN ({placeholders})",
            unique_ids,
        )
        results: dict[int, dict[str, Any]] = {}
        for row in cursor.fetchall():
            doc_id, entry_json = row
            try:
                results[doc_id] = json.loads(entry_json)
            except Exception:
                pass
        return results

    def fetch_entry_by_repo_id(self, repo_id: str) -> dict[str, Any] | None:
        """Fetch full entry JSON by repo_id."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT entry_json FROM records WHERE repo_id = ?", (repo_id,))
        row = cursor.fetchone()
        if not row or not row[0]:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    @classmethod
    def create_from_entries(
        cls,
        db_path: Path | str,
        *,
        manifest: dict[str, Any],
        entries: list[dict[str, Any]],
        caches: list[dict[str, Any]] | None = None,
    ) -> MetaDatabase:
        """Create and populate a new SQLite metadata database atomically."""
        final_path = Path(db_path).resolve()
        temp_path = final_path.with_name(f".{final_path.name}.tmp")

        if temp_path.exists():
            temp_path.unlink()

        conn = sqlite3.connect(str(temp_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode = OFF;")
        conn.execute("PRAGMA synchronous = OFF;")
        init_meta_db_schema(conn)

        # 1. Insert manifest
        manifest_rows = [
            (k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v)
            for k, v in manifest.items()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO manifest (key, value) VALUES (?, ?)", manifest_rows
        )

        # 2. Insert records
        record_rows = []
        for i, entry in enumerate(entries):
            repo_id = str(entry.get("repo_id") or f"doc_{i}")
            meta = entry.get("metadata") or {}
            name = str(meta.get("name") or repo_id.split("/")[-1])
            language = str(meta.get("language") or "")
            stars = int(meta.get("stars") or 0)
            forks = int(meta.get("forks") or 0)
            archived = 1 if meta.get("archived") is True else 0
            disabled = 1 if meta.get("disabled") is True else 0

            cache_data = caches[i] if caches and i < len(caches) else None
            cache_json = (
                json.dumps(cache_to_serializable(cache_data), ensure_ascii=False)
                if cache_data
                else None
            )
            entry_json = json.dumps(entry, ensure_ascii=False)

            record_rows.append(
                (
                    i,
                    repo_id,
                    name,
                    language,
                    stars,
                    forks,
                    archived,
                    disabled,
                    cache_json,
                    entry_json,
                )
            )

        conn.executemany(
            """
            INSERT INTO records (
                doc_id, repo_id, name, language, stars, forks, archived, disabled, cache_json, entry_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            record_rows,
        )
        conn.commit()
        conn.close()

        temp_path.replace(final_path)
        return cls(final_path, read_only=True)


class LazyEntriesList(Sequence[dict[str, Any]]):
    """A memory-efficient lazy sequence that fetches full entry JSON on demand from MetaDatabase."""

    def __init__(self, meta_db: MetaDatabase, repo_ids: Sequence[str]) -> None:
        self.meta_db = meta_db
        self.repo_ids = tuple(repo_ids)
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.repo_ids)

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"Index must be integer, not {type(idx).__name__}")
        int_idx = int(idx)
        if int_idx < 0:
            int_idx += len(self.repo_ids)
        if int_idx < 0 or int_idx >= len(self.repo_ids):
            raise IndexError(f"Index {int_idx} out of range (length {len(self.repo_ids)})")
        if int_idx not in self._cache:
            entry = self.meta_db.fetch_entry(int_idx)
            if entry is None:
                entry = {"repo_id": self.repo_ids[int_idx], "metadata": {}}
            self._cache[int_idx] = entry
        return self._cache[int_idx]

    def prefetch(self, indices: Sequence[int | np.integer]) -> None:
        """Batch prefetch multiple entries in a single query."""
        int_indices = [int(i) for i in indices]
        missing = [i for i in int_indices if i not in self._cache and 0 <= i < len(self.repo_ids)]
        if missing:
            fetched = self.meta_db.fetch_entries_batch(missing)
            for i in missing:
                self._cache[i] = fetched.get(i) or {"repo_id": self.repo_ids[i], "metadata": {}}
