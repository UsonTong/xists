import io
import json
import tarfile
from unittest.mock import patch

import pytest

from xists.search.pull import (
    compute_file_sha256,
    pull_index,
)


def test_pull_index_demo_preset(tmp_path):
    out_dir = tmp_path / "pulled_demo"
    res = pull_index("demo", out_dir)

    assert res["preset"] == "demo"
    assert (out_dir / "records.json").is_file()
    assert (out_dir / "index.json").is_file()
    assert (out_dir / "index.vectors.npy").is_file()
    assert res["records_count"] >= 20
    assert res["index_version"] == 4
    assert res["sha256"] == compute_file_sha256(out_dir / "index.json")


def test_pull_index_starter_preset(tmp_path):
    out_dir = tmp_path / "pulled_starter"
    res = pull_index("starter", out_dir)

    assert res["preset"] == "starter"
    assert (out_dir / "records.json").is_file()
    assert (out_dir / "index.json").is_file()
    assert (out_dir / "index.vectors.npy").is_file()


def test_pull_index_fails_without_force_when_files_exist(tmp_path):
    out_dir = tmp_path / "conflict"
    out_dir.mkdir()
    (out_dir / "index.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Target files already exist"):
        pull_index("demo", out_dir, force=False)

    # With force=True, should overwrite and succeed
    res = pull_index("demo", out_dir, force=True)
    assert res["preset"] == "demo"
    assert (out_dir / "records.json").is_file()


def test_pull_index_sha256_checksum_verification(tmp_path):
    out_dir = tmp_path / "checksum_test"

    # Compute genuine sha256
    temp_dir = tmp_path / "temp"
    genuine_res = pull_index("demo", temp_dir)
    genuine_sha = genuine_res["sha256"]

    # Pulling with valid sha succeeds
    res = pull_index("demo", out_dir, sha256=genuine_sha)
    assert res["sha256"] == genuine_sha

    # Pulling with invalid sha raises ValueError
    bad_dir = tmp_path / "bad_checksum"
    with pytest.raises(ValueError, match="SHA-256 checksum mismatch"):
        pull_index(
            "demo",
            bad_dir,
            sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )


def test_pull_index_unknown_preset_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown index source or preset"):
        pull_index("nonexistent_preset_xyz", tmp_path)


def test_pull_index_remote_tar_archive(tmp_path):
    # Create a dummy tar.gz containing index.json and records.json
    archive_dir = tmp_path / "mock_archive"
    archive_dir.mkdir()
    sample_index = {"index_version": 4, "record_count": 1, "dimension": 64, "vectors": []}
    sample_records = [{"repo_id": "test/repo", "name": "repo"}]

    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
        for name, data in [("index.json", sample_index), ("records.json", sample_records)]:
            raw = json.dumps(data).encode("utf-8")
            ti = tarfile.TarInfo(name=name)
            ti.size = len(raw)
            tar.addfile(ti, io.BytesIO(raw))
    tar_bytes.seek(0)

    class MockResponse:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, amt=None):
            return self.content.read(amt)

    out_dir = tmp_path / "extracted_tar"
    with patch("urllib.request.urlopen", return_value=MockResponse(tar_bytes)):
        res = pull_index("https://example.com/custom-index.tar.gz", out_dir)

    assert (out_dir / "index.json").is_file()
    assert (out_dir / "records.json").is_file()
    assert res["index_version"] == 4


def test_pull_index_safe_extraction_prevents_path_traversal(tmp_path):
    # Create malicious tar with ../../../evil.txt
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
        raw = b"malicious content"
        ti = tarfile.TarInfo(name="../../evil.txt")
        ti.size = len(raw)
        tar.addfile(ti, io.BytesIO(raw))
    tar_bytes.seek(0)

    class MockResponse:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, amt=None):
            return self.content.read(amt)

    out_dir = tmp_path / "dest"
    with patch("urllib.request.urlopen", return_value=MockResponse(tar_bytes)):
        with pytest.raises(ValueError, match="Path traversal detected"):
            pull_index("https://example.com/malicious.tar.gz", out_dir)
