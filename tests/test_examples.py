import json
from pathlib import Path

from xists.cli import load_repo_ids
from xists.eval.schema import load_dataset


ROOT = Path(__file__).resolve().parent.parent


def test_example_repos_file_loads_cleanly():
    repo_ids = load_repo_ids(ROOT / "repos.txt")

    assert len(repo_ids) == 200
    assert "react/react" in repo_ids
    assert "fastapi/fastapi" in repo_ids
    assert len(repo_ids) == len(set(repo_ids))


def test_example_eval_dataset_is_valid_and_matches_example_repos():
    dataset = load_dataset(ROOT / "examples" / "eval-cases.json")
    example_repo_ids = set(load_repo_ids(ROOT / "repos.txt"))

    assert dataset["dataset_name"] == "xists-baseline-112"
    assert len(dataset["cases"]) == 112
    assert set(dataset["families"]["frontend-ui"]).issubset(example_repo_ids)

    for case in dataset["cases"]:
        assert case["expected_repo_id"] in example_repo_ids
        assert set(case["acceptable_set"]).issubset(example_repo_ids)

    raw = json.loads((ROOT / "examples" / "eval-cases.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["dataset_name"] == "xists-baseline-112"
