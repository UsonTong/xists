import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from xists.cli import build_parser, index_append, index_merge, index_prune
from xists.records import RECORD_SCHEMA_VERSION
from xists.search.append import append_records_to_index, append_repo_to_index
from xists.search.embed import EmbeddingConfig
from xists.search.index import INDEX_VERSION, load_index, save_index
from xists.search.merge import merge_indices
from xists.search.prune import prune_index


def _mock_call_embeddings(config, texts, *, timeout=120, input_type=None):
    vectors = []
    for t in texts:
        # Generate deterministic vector of dimension 4
        val = float(len(t) % 10 + 1)
        vec = [val, val * 0.5, val * 0.25, 1.0]
        norm = float(np.linalg.norm(vec))
        vectors.append((np.array(vec) / norm).tolist())
    return vectors


@pytest.fixture
def mock_embedding_config():
    return EmbeddingConfig(
        api_key="test-key",
        base_url="https://api.test/v1",
        model="test-model",
    )


@pytest.fixture
def sample_workspace(tmp_path, mock_embedding_config):
    records = [
        {
            "schema_version": RECORD_SCHEMA_VERSION,
            "repo_id": "owner/repo1",
            "name": "repo1",
            "url": "https://github.com/owner/repo1",
            "github": {"description": "First test repository", "stars": 100, "archived": False, "disabled": False},
            "llm_profile": {"summary": "First summary", "search_text": "first search text", "confidence": "high"},
        },
        {
            "schema_version": RECORD_SCHEMA_VERSION,
            "repo_id": "owner/repo2",
            "name": "repo2",
            "url": "https://github.com/owner/repo2",
            "github": {"description": "Second test repository (archived)", "stars": 50, "archived": True, "disabled": False},
            "llm_profile": {"summary": "Second summary", "search_text": "second search text", "confidence": "medium"},
        },
    ]
    records_file = tmp_path / "records.json"
    records_file.write_text(json.dumps(records, indent=2), encoding="utf-8")

    index_file = tmp_path / "index.json"
    matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)

    index_doc = {
        "index_version": 4,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_model": "test-model",
        "dimension": 4,
        "record_count": 2,
        "vectors": [
            {
                "repo_id": "owner/repo1",
                "embedding_input_fingerprint": "fp1",
                "metadata": {"name": "repo1", "archived": False, "disabled": False},
            },
            {
                "repo_id": "owner/repo2",
                "embedding_input_fingerprint": "fp2",
                "metadata": {"name": "repo2", "archived": True, "disabled": False},
            },
        ],
    }
    save_index(index_file, index_doc, matrix=matrix, version=4)

    return tmp_path, records_file, index_file


def test_append_records_to_index_adds_new_repo(sample_workspace, mock_embedding_config):
    tmp_path, records_file, index_file = sample_workspace

    new_record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "repo_id": "owner/repo3",
        "name": "repo3",
        "url": "https://github.com/owner/repo3",
        "github": {"description": "Third test repository", "stars": 200, "archived": False, "disabled": False},
        "llm_profile": {"summary": "Third summary", "search_text": "third search text", "confidence": "high"},
    }

    with patch("xists.search.append.call_embeddings", side_effect=_mock_call_embeddings):
        result = append_records_to_index(
            [new_record],
            index_path=index_file,
            records_path=records_file,
            config=mock_embedding_config,
        )

    assert result["status"] == "success"
    assert result["added"] == 1
    assert result["updated"] == 0
    assert result["skipped"] == 0
    assert result["total_vectors"] == 3
    assert result["total_records"] == 3

    # Verify index and matrix updated
    loaded = load_index(index_file, mmap=False)
    assert loaded["record_count"] == 3
    assert loaded["_matrix"].shape == (3, 4)
    assert loaded["vectors"][2]["repo_id"] == "owner/repo3"

    # Verify records file updated
    saved_records = json.loads(records_file.read_text(encoding="utf-8"))
    assert len(saved_records) == 3
    assert saved_records[2]["repo_id"] == "owner/repo3"


def test_append_records_idempotent_skip_when_fingerprint_matches(sample_workspace, mock_embedding_config):
    tmp_path, records_file, index_file = sample_workspace
    existing_records = json.loads(records_file.read_text(encoding="utf-8"))

    # Append same records without change -> should skip re-embedding
    with patch("xists.search.append.call_embeddings", side_effect=_mock_call_embeddings) as mock_embed:
        result = append_records_to_index(
            existing_records,
            index_path=index_file,
            records_path=records_file,
            config=mock_embedding_config,
            force=False,
        )
        assert mock_embed.call_count == 1  # only repo1 was fp1 vs calculated

    # Running a second time with exact same records -> 0 embed calls
    with patch("xists.search.append.call_embeddings", side_effect=_mock_call_embeddings) as mock_embed:
        result2 = append_records_to_index(
            existing_records,
            index_path=index_file,
            records_path=records_file,
            config=mock_embedding_config,
            force=False,
        )
        assert result2["skipped"] == 2
        assert mock_embed.call_count == 0


def test_merge_indices_combines_matrices_and_deduplicates(tmp_path):
    idx1_file = tmp_path / "idx1.json"
    idx2_file = tmp_path / "idx2.json"
    rec1_file = tmp_path / "rec1.json"
    rec2_file = tmp_path / "rec2.json"

    # Index 1: repoA, repoB (low quality)
    doc1 = {
        "index_version": 4,
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {"repo_id": "org/repoA", "metadata": {"name": "repoA"}},
            {"repo_id": "org/repoB", "metadata": {"name": "repoB", "confidence": "low"}},
        ],
    }
    mat1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    save_index(idx1_file, doc1, matrix=mat1, version=4)
    rec1_file.write_text(json.dumps([{"repo_id": "org/repoA"}, {"repo_id": "org/repoB", "llm_profile": {"confidence": "low"}}]))

    # Index 2: repoB (high quality), repoC
    doc2 = {
        "index_version": 4,
        "embedding_model": "bge-m3",
        "dimension": 2,
        "vectors": [
            {"repo_id": "org/repoB", "metadata": {"name": "repoB", "confidence": "high", "summary": "Better summary"}},
            {"repo_id": "org/repoC", "metadata": {"name": "repoC"}},
        ],
    }
    mat2 = np.array([[0.5, 0.5], [0.0, 0.5]], dtype=np.float32)
    save_index(idx2_file, doc2, matrix=mat2, version=4)
    rec2_file.write_text(json.dumps([{"repo_id": "org/repoB", "llm_profile": {"confidence": "high", "summary": "Better summary"}}, {"repo_id": "org/repoC"}]))

    out_idx = tmp_path / "merged_idx.json"
    out_rec = tmp_path / "merged_rec.json"

    result = merge_indices(
        [idx1_file, idx2_file],
        output_index_path=out_idx,
        records_paths=[rec1_file, rec2_file],
        output_records_path=out_rec,
    )

    assert result["status"] == "success"
    assert result["input_indices_count"] == 2
    assert result["merged_records_count"] == 3
    assert result["duplicates_resolved"] == 1

    loaded_merged = load_index(out_idx, mmap=False)
    assert loaded_merged["record_count"] == 3
    assert loaded_merged["_matrix"].shape == (3, 2)
    repos = [v["repo_id"] for v in loaded_merged["vectors"]]
    assert repos == ["org/repoA", "org/repoB", "org/repoC"]


def test_merge_indices_rejects_model_mismatch(tmp_path):
    idx1 = tmp_path / "i1.json"
    idx2 = tmp_path / "i2.json"
    save_index(idx1, {"embedding_model": "model-A", "dimension": 4, "vectors": []}, matrix=np.empty((0, 4), dtype=np.float32), version=4)
    save_index(idx2, {"embedding_model": "model-B", "dimension": 4, "vectors": []}, matrix=np.empty((0, 4), dtype=np.float32), version=4)

    with pytest.raises(ValueError, match="different embedding models"):
        merge_indices([idx1, idx2], output_index_path=tmp_path / "merged.json")


def test_prune_index_removes_archived_and_disabled(sample_workspace):
    tmp_path, records_file, index_file = sample_workspace

    result = prune_index(
        index_path=index_file,
        records_path=records_file,
        prune_archived=True,
        prune_disabled=True,
        dry_run=False,
    )

    assert result["status"] == "success"
    assert result["total_before"] == 2
    assert result["pruned_count"] == 1
    assert result["retained_count"] == 1
    assert result["reasons"]["archived"] == 1

    # Verify index updated
    loaded = load_index(index_file, mmap=False)
    assert loaded["record_count"] == 1
    assert loaded["_matrix"].shape == (1, 4)
    assert loaded["vectors"][0]["repo_id"] == "owner/repo1"

    # Verify records updated
    recs = json.loads(records_file.read_text(encoding="utf-8"))
    assert len(recs) == 1
    assert recs[0]["repo_id"] == "owner/repo1"


def test_prune_index_dry_run_does_not_modify_files(sample_workspace):
    tmp_path, records_file, index_file = sample_workspace

    result = prune_index(
        index_path=index_file,
        records_path=records_file,
        prune_archived=True,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["pruned_count"] == 1

    # Files should still have 2 items
    loaded = load_index(index_file, mmap=False)
    assert loaded["record_count"] == 2


def test_cli_index_append_and_prune_and_merge(sample_workspace, monkeypatch, capsys):
    tmp_path, records_file, index_file = sample_workspace
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.test/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-model")

    # 1. Test CLI index prune
    args_prune = build_parser().parse_args([
        "index", "prune",
        "--index", str(index_file),
        "--records", str(records_file),
        "--format", "json",
    ])
    assert args_prune.func is index_prune
    ret_prune = index_prune(args_prune)
    assert ret_prune == 0
    out_prune = json.loads(capsys.readouterr().out)
    assert out_prune["pruned_count"] == 1
    assert out_prune["retained_count"] == 1

    # 2. Test CLI index append with input file
    extra_records_file = tmp_path / "extra.json"
    extra_record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "repo_id": "owner/extra_repo",
        "name": "extra_repo",
        "url": "https://github.com/owner/extra_repo",
        "github": {"description": "Extra test repo", "stars": 300, "archived": False, "disabled": False},
        "llm_profile": {"summary": "Extra summary", "search_text": "extra search text", "confidence": "high"},
    }
    extra_records_file.write_text(json.dumps([extra_record]), encoding="utf-8")

    args_append = build_parser().parse_args([
        "index", "append",
        "--input", str(extra_records_file),
        "--index", str(index_file),
        "--records", str(records_file),
        "--format", "json",
    ])
    assert args_append.func is index_append

    with patch("xists.search.append.call_embeddings", side_effect=_mock_call_embeddings):
        ret_append = index_append(args_append)
        assert ret_append == 0

    out_append = json.loads(capsys.readouterr().out)
    assert out_append["status"] == "success"
    assert out_append["added"] == 1
    assert out_append["total_vectors"] == 2  # 1 retained + 1 added

    # 3. Test CLI index append with --repo
    mock_collected = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "repo_id": "owner/collected_repo",
        "name": "collected_repo",
        "url": "https://github.com/owner/collected_repo",
        "github": {"description": "Collected repo", "stars": 400, "archived": False, "disabled": False},
        "llm_profile": {"summary": "Collected summary", "search_text": "collected search text", "confidence": "high"},
    }
    args_repo_append = build_parser().parse_args([
        "index", "append",
        "--repo", "owner/collected_repo",
        "--index", str(index_file),
        "--records", str(records_file),
        "--format", "text",
    ])
    with patch("xists.search.append.collect_record", return_value=mock_collected), \
         patch("xists.search.append.call_embeddings", side_effect=_mock_call_embeddings):
        ret_repo_append = index_append(args_repo_append)
        assert ret_repo_append == 0

    out_text = capsys.readouterr().out
    assert "Incremental append complete" in out_text

    # 4. Test CLI index merge
    idx2_file = tmp_path / "idx2.json"
    doc2 = {
        "index_version": 4,
        "embedding_model": "test-model",
        "dimension": 4,
        "vectors": [
            {"repo_id": "owner/another_repo", "metadata": {"name": "another_repo"}},
        ],
    }
    save_index(idx2_file, doc2, matrix=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32), version=4)
    out_merged = tmp_path / "final_merged.json"

    args_merge = build_parser().parse_args([
        "index", "merge",
        "--indices", str(index_file), str(idx2_file),
        "--output", str(out_merged),
        "--format", "json",
    ])
    assert args_merge.func is index_merge
    ret_merge = index_merge(args_merge)
    assert ret_merge == 0
    out_merge_json = json.loads(capsys.readouterr().out)
    assert out_merge_json["status"] == "success"
    assert out_merge_json["input_indices_count"] == 2
    assert out_merge_json["merged_records_count"] == 4  # 3 from index_file + 1 from idx2_file


def test_cli_index_append_and_merge_error_handling(sample_workspace, capsys):
    tmp_path, records_file, index_file = sample_workspace

    # Error when neither --repo nor --input is given
    args_no_input = build_parser().parse_args([
        "index", "append",
        "--index", str(index_file),
    ])
    assert index_append(args_no_input) == 2
    err = capsys.readouterr().err
    assert "Either --repo owner/repo or --input" in err

    # Error when merge given < 2 indices
    args_single_merge = build_parser().parse_args([
        "index", "merge",
        "--indices", str(index_file),
        "--output", str(tmp_path / "out.json"),
    ])
    assert index_merge(args_single_merge) == 2
    err_merge = capsys.readouterr().err
    assert "At least two index files are required" in err_merge

