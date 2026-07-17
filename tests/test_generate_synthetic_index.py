import importlib.util
import json
from pathlib import Path

from xists.records import RECORD_SCHEMA_VERSION, records_validation_report
from xists.search.embed import EMBEDDING_INPUT_VERSION, embedding_input_fingerprint
from xists.search.index import INDEX_VERSION

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_synthetic_index.py"
SPEC = importlib.util.spec_from_file_location("generate_synthetic_index", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_records_pass_validation_core_checks():
    records = MODULE.generate_records(50)
    report = records_validation_report(records)
    assert report["ok"] is True
    assert not report["errors"]


def test_index_matches_build_index_fields():
    records = MODULE.generate_records(50)
    index = MODULE.generate_index(records, dimension=8, seed=42)
    assert set(index) == {
        "index_version",
        "record_schema_version",
        "embedding_model",
        "embedding_base_url",
        "embedding_input_version",
        "dimension",
        "built_at",
        "record_count",
        "skipped",
        "vectors",
    }
    assert index["index_version"] == INDEX_VERSION
    assert index["record_schema_version"] == RECORD_SCHEMA_VERSION
    assert index["embedding_model"] == "synthetic-test-model"
    assert index["embedding_input_version"] == EMBEDDING_INPUT_VERSION
    assert index["dimension"] == 8
    assert index["record_count"] == 50
    assert index["skipped"] == []
    assert len(index["vectors"]) == 50
    entry = index["vectors"][0]
    assert set(entry) == {"repo_id", "embedding_input_fingerprint", "metadata", "vector"}
    assert entry["repo_id"] == "synthetic/repo-000000"
    assert entry["embedding_input_fingerprint"] == embedding_input_fingerprint(records[0])
    assert len(entry["vector"]) == 8


def test_same_seed_is_deterministic():
    records = MODULE.generate_records(50)
    first = MODULE.generate_index(records, dimension=8, seed=7)
    second = MODULE.generate_index(records, dimension=8, seed=7)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_different_seed_changes_vectors():
    records = MODULE.generate_records(5)
    first = MODULE.generate_index(records, dimension=8, seed=1)
    second = MODULE.generate_index(records, dimension=8, seed=2)
    assert first["vectors"][0]["vector"] != second["vectors"][0]["vector"]
