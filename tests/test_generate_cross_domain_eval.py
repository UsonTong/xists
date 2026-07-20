import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_cross_domain_eval.py"
SPEC = importlib.util.spec_from_file_location("generate_cross_domain_eval", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _records() -> list[dict]:
    repo_ids = {
        project["repo_id"]
        for projects in MODULE.PROJECTS.values()
        for project in projects
    }
    repo_ids.add("MariaDB/server")
    return [{"repo_id": repo_id} for repo_id in repo_ids]


def test_cross_domain_datasets_are_complete_and_valid_for_each_corpus():
    datasets = {"dev": MODULE.build_dataset("dev"), "holdout": MODULE.build_dataset("holdout")}

    MODULE.validate_datasets(datasets, [_records(), _records()])

    assert len(datasets["dev"]["cases"]) == 60
    assert len(datasets["holdout"]["cases"]) == 60
    assert {tag for case in datasets["dev"]["cases"] for tag in case["tags"] if tag.startswith("domain-")} == {
        "domain-ai-llm",
        "domain-web",
        "domain-devtools",
        "domain-infra",
        "domain-data",
    }
