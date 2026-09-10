"""SQLite-backed persistent cache for query and text embeddings.

Provides fast, zero-dependency, cross-session vector caching to eliminate
redundant remote API calls and reduce search latency to sub-millisecond levels.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MAX_ENTRIES = 50_000
DEFAULT_CACHE_DIR_NAME = "cache"
DEFAULT_CACHE_FILE_NAME = "embeddings.db"


def compute_cache_key(
    model: str,
    query: str,
    *,
    dimension: int | None = None,
    input_type: str = "query",
) -> str:
    """Compute a deterministic SHA-256 fingerprint for an embedding request."""
    normalized_query = query.strip()
    dim_str = str(dimension) if dimension is not None else "0"
    payload = f"{model.strip()}:{dim_str}:{input_type.strip()}:{normalized_query}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QueryEmbeddingCache:
    """Persistent SQLite-backed embedding cache with LRU eviction and WAL mode."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        enabled: bool = True,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve() if db_path is not None else None
        self.max_entries = max(100, max_entries)
        self.enabled = enabled and self.db_path is not None
        self._lock = threading.Lock()
        self._initialized = False

        if self.enabled and self.db_path is not None:
            self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        if self._initialized or not self.enabled or self.db_path is None:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    conn.execute("PRAGMA busy_timeout=5000;")
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS query_cache (
                            key TEXT PRIMARY KEY,
                            model TEXT NOT NULL,
                            dimension INTEGER NOT NULL,
                            input_type TEXT NOT NULL,
                            query TEXT NOT NULL,
                            vector_blob BLOB NOT NULL,
                            created_at REAL NOT NULL,
                            last_accessed REAL NOT NULL,
                            access_count INTEGER NOT NULL DEFAULT 1
                        );
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_query_cache_accessed ON query_cache(last_accessed);"
                    )
                    conn.commit()
                self._initialized = True
            except (sqlite3.Error, OSError):
                # Fallback to disabled if directory or database is non-writable
                self.enabled = False

    def _get_connection(self) -> sqlite3.Connection | None:
        if not self.enabled or self.db_path is None:
            return None
        self._ensure_initialized()
        if not self._initialized:
            return None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000;")
            return conn
        except (sqlite3.Error, OSError):
            return None

    def get(
        self,
        model: str,
        query: str,
        *,
        dimension: int | None = None,
        input_type: str = "query",
    ) -> list[float] | None:
        """Look up a cached embedding vector. Returns None on cache miss or error."""
        if not self.enabled:
            return None

        key = compute_cache_key(model, query, dimension=dimension, input_type=input_type)
        conn = self._get_connection()
        if conn is None:
            return None

        try:
            with conn:
                cursor = conn.execute(
                    "SELECT vector_blob, dimension FROM query_cache WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None

                blob, stored_dim = row
                now = time.time()
                conn.execute(
                    "UPDATE query_cache SET last_accessed = ?, access_count = access_count + 1 WHERE key = ?",
                    (now, key),
                )
                vec = np.frombuffer(blob, dtype=np.float32).tolist()
                return vec
        except (sqlite3.Error, OSError, ValueError):
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def get_batch(
        self,
        model: str,
        queries: Sequence[str],
        *,
        dimension: int | None = None,
        input_type: str = "query",
    ) -> list[list[float] | None]:
        """Look up multiple queries in a single transaction."""
        if not self.enabled or not queries:
            return [None] * len(queries)

        keys = [
            compute_cache_key(model, q, dimension=dimension, input_type=input_type) for q in queries
        ]
        results: list[list[float] | None] = [None] * len(queries)

        conn = self._get_connection()
        if conn is None:
            return results

        try:
            with conn:
                hit_keys: list[str] = []
                now = time.time()
                for idx, key in enumerate(keys):
                    cursor = conn.execute(
                        "SELECT vector_blob FROM query_cache WHERE key = ?",
                        (key,),
                    )
                    row = cursor.fetchone()
                    if row is not None:
                        blob = row[0]
                        results[idx] = np.frombuffer(blob, dtype=np.float32).tolist()
                        hit_keys.append(key)

                if hit_keys:
                    conn.executemany(
                        "UPDATE query_cache SET last_accessed = ?, access_count = access_count + 1 WHERE key = ?",
                        [(now, k) for k in hit_keys],
                    )
            return results
        except (sqlite3.Error, OSError, ValueError):
            return results
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def set(
        self,
        model: str,
        query: str,
        vector: Sequence[float],
        *,
        dimension: int | None = None,
        input_type: str = "query",
    ) -> None:
        """Store an embedding vector in the persistent cache."""
        if not self.enabled or not vector:
            return

        key = compute_cache_key(model, query, dimension=dimension, input_type=input_type)
        dim = len(vector) if dimension is None else dimension
        blob = np.array(vector, dtype=np.float32).tobytes()
        now = time.time()

        conn = self._get_connection()
        if conn is None:
            return

        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO query_cache (key, model, dimension, input_type, query, vector_blob, created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(key) DO UPDATE SET
                        vector_blob = excluded.vector_blob,
                        last_accessed = excluded.last_accessed,
                        access_count = query_cache.access_count + 1;
                    """,
                    (key, model, dim, input_type, query.strip(), blob, now, now),
                )
                self._maybe_evict_lru(conn)
        except (sqlite3.Error, OSError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def set_batch(
        self,
        model: str,
        queries: Sequence[str],
        vectors: Sequence[Sequence[float]],
        *,
        dimension: int | None = None,
        input_type: str = "query",
    ) -> None:
        """Store multiple embedding vectors in a single transaction."""
        if not self.enabled or not queries or len(queries) != len(vectors):
            return

        now = time.time()
        rows = []
        for q, vec in zip(queries, vectors):
            if not vec:
                continue
            k = compute_cache_key(model, q, dimension=dimension, input_type=input_type)
            dim = len(vec) if dimension is None else dimension
            blob = np.array(vec, dtype=np.float32).tobytes()
            rows.append((k, model, dim, input_type, q.strip(), blob, now, now))

        if not rows:
            return

        conn = self._get_connection()
        if conn is None:
            return

        try:
            with conn:
                conn.executemany(
                    """
                    INSERT INTO query_cache (key, model, dimension, input_type, query, vector_blob, created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(key) DO UPDATE SET
                        vector_blob = excluded.vector_blob,
                        last_accessed = excluded.last_accessed,
                        access_count = query_cache.access_count + 1;
                    """,
                    rows,
                )
                self._maybe_evict_lru(conn)
        except (sqlite3.Error, OSError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _maybe_evict_lru(self, conn: sqlite3.Connection) -> None:
        """Evict oldest entries if table size exceeds max_entries."""
        cursor = conn.execute("SELECT COUNT(*) FROM query_cache")
        row = cursor.fetchone()
        count = row[0] if row else 0
        if count > self.max_entries:
            excess = count - int(self.max_entries * 0.9)
            conn.execute(
                f"DELETE FROM query_cache WHERE key IN (SELECT key FROM query_cache ORDER BY last_accessed ASC LIMIT {excess});"
            )

    def evict_lru(self, target_count: int | None = None) -> int:
        """Explicitly evict oldest entries down to target_count."""
        if not self.enabled:
            return 0
        conn = self._get_connection()
        if conn is None:
            return 0
        try:
            with conn:
                cursor = conn.execute("SELECT COUNT(*) FROM query_cache")
                count = cursor.fetchone()[0]
                limit_target = (
                    target_count if target_count is not None else int(self.max_entries * 0.9)
                )
                if count > limit_target:
                    excess = count - limit_target
                    conn.execute(
                        f"DELETE FROM query_cache WHERE key IN (SELECT key FROM query_cache ORDER BY last_accessed ASC LIMIT {excess});"
                    )
                    return excess
                return 0
        except (sqlite3.Error, OSError):
            return 0
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def clear(self) -> None:
        """Remove all entries from the cache."""
        if not self.enabled:
            return
        conn = self._get_connection()
        if conn is None:
            return
        try:
            with conn:
                conn.execute("DELETE FROM query_cache;")
        except (sqlite3.Error, OSError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def stats(self) -> dict[str, Any]:
        """Return cache statistics including entry count and file size."""
        if not self.enabled or self.db_path is None:
            return {"enabled": False, "entry_count": 0, "size_bytes": 0, "db_path": None}

        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        entry_count = 0
        conn = self._get_connection()
        if conn is not None:
            try:
                with conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM query_cache")
                    entry_count = cursor.fetchone()[0]
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        return {
            "enabled": True,
            "entry_count": entry_count,
            "max_entries": self.max_entries,
            "size_bytes": size_bytes,
            "db_path": str(self.db_path),
        }
