"""Unit tests for SQLite metadata sidecar database."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from xists.search.meta_db import (
    MetaDatabase,
    cache_to_serializable,
    serializable_to_cache,
)


def test_cache_serialization_roundtrip():
    original = {
        "repo_id": "test/repo",
        "identity_values_lower": {"test/repo", "repo"},
        "text_tokens": {"test", "repo", "library"},
        "topic_tokens": {"async", "web"},
        "profile_tokens": {"fast", "api"},
        "ecosystem_set": frozenset(["python", "pypi"]),
        "topics_set": frozenset(["web", "async"]),
        "id_value_tokens": ("test", "repo"),
        "stars": 1234,
        "archived": False,
    }

    serialized = cache_to_serializable(original)
    assert isinstance(serialized["identity_values_lower"], list)
    assert isinstance(serialized["ecosystem_set"], list)

    reconstituted = serializable_to_cache(serialized)
    assert reconstituted["identity_values_lower"] == original["identity_values_lower"]
    assert reconstituted["ecosystem_set"] == original["ecosystem_set"]
    assert reconstituted["topics_set"] == original["topics_set"]
    assert reconstituted["id_value_tokens"] == original["id_value_tokens"]


def test_meta_database_lifecycle_and_queries():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.meta.db"
        manifest = {
            "index_version": 4,
            "dimension": 1024,
            "embedding_model": "bge-m3",
            "record_count": 3,
        }
        entries = [
            {
                "repo_id": "org/repo-0",
                "metadata": {
                    "name": "repo-0",
                    "language": "Python",
                    "stars": 100,
                    "forks": 10,
                    "archived": False,
                    "summary": "First repo summary",
                },
            },
            {
                "repo_id": "org/repo-1",
                "metadata": {
                    "name": "repo-1",
                    "language": "Rust",
                    "stars": 500,
                    "forks": 50,
                    "archived": True,
                    "summary": "Second repo summary",
                },
            },
            {
                "repo_id": "org/repo-2",
                "metadata": {
                    "name": "repo-2",
                    "language": "Go",
                    "stars": 1000,
                    "forks": 100,
                    "archived": False,
                    "summary": "Third repo summary",
                },
            },
        ]
        caches = [
            {"repo_id": "org/repo-0", "stars": 100, "text_tokens": {"python"}},
            {"repo_id": "org/repo-1", "stars": 500, "text_tokens": {"rust"}},
            {"repo_id": "org/repo-2", "stars": 1000, "text_tokens": {"go"}},
        ]

        # 1. Create database
        MetaDatabase.create_from_entries(
            db_path,
            manifest=manifest,
            entries=entries,
            caches=caches,
        )
        assert db_path.is_file()

        # 2. Read manifest & counts
        with MetaDatabase(db_path, read_only=True) as ro_db:
            loaded_manifest = ro_db.get_manifest()
            assert loaded_manifest["index_version"] == 4
            assert loaded_manifest["dimension"] == 1024
            assert loaded_manifest["embedding_model"] == "bge-m3"
            assert ro_db.get_doc_count() == 3

            # 3. Load search state
            repo_ids, loaded_caches, stars_arr, forks_arr, archived_arr = ro_db.load_search_state()
            assert repo_ids == ("org/repo-0", "org/repo-1", "org/repo-2")
            assert len(loaded_caches) == 3
            assert np.array_equal(stars_arr, np.array([100, 500, 1000], dtype=np.int64))
            assert np.array_equal(forks_arr, np.array([10, 50, 100], dtype=np.int64))
            assert np.array_equal(archived_arr, np.array([False, True, False], dtype=bool))

            # 4. Fetch single entry
            e1 = ro_db.fetch_entry(1)
            assert e1 is not None
            assert e1["repo_id"] == "org/repo-1"
            assert e1["metadata"]["summary"] == "Second repo summary"

            # 5. Fetch batch entries
            batch = ro_db.fetch_entries_batch([0, 2])
            assert len(batch) == 2
            assert 0 in batch and 2 in batch
            assert batch[0]["repo_id"] == "org/repo-0"
            assert batch[2]["repo_id"] == "org/repo-2"

            # 6. Fetch by repo_id
            by_id = ro_db.fetch_entry_by_repo_id("org/repo-0")
            assert by_id is not None
            assert by_id["repo_id"] == "org/repo-0"
