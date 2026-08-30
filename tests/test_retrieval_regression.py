import json
import subprocess
import sys
from pathlib import Path

from xists.eval.schema import load_dataset

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "retrieval-regression"


def test_public_retrieval_regression_fixture_is_valid_and_current():
    dataset = load_dataset(FIXTURE / "cases.json")
    index = json.loads((FIXTURE / "index.json").read_text(encoding="utf-8"))
    records = json.loads((FIXTURE / "records.json").read_text(encoding="utf-8"))

    assert index["index_version"] == 3
    assert index["record_schema_version"] == 2
    assert index["record_count"] == len(index["vectors"]) == len(records)
    assert {"exact", "functional", "ecosystem", "ambiguous", "chinese", "no-result"}.issubset(
        {tag for case in dataset["cases"] for tag in case["tags"]}
    )


def test_public_retrieval_regression_runs_offline():
    completed = subprocess.run(
        [sys.executable, "scripts/run_retrieval_regression.py"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert report["case_count"] == 6
