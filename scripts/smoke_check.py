"""Run no-network smoke checks for committed xists demo artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from xists.cli import load_repo_ids
from xists.eval.inspect import inspect_report, load_report
from xists.eval.schema import load_dataset
from xists.search.index import load_index


ROOT = Path(__file__).resolve().parent.parent


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _check_dataset(path: Path, repo_ids: set[str]) -> dict[str, Any]:
    dataset = load_dataset(path)
    missing_expected = [case["expected_repo_id"] for case in dataset["cases"] if case["expected_repo_id"] not in repo_ids]
    missing_acceptable = sorted(
        {
            repo_id
            for case in dataset["cases"]
            for repo_id in case["acceptable_set"]
            if repo_id not in repo_ids
        }
    )
    return {
        "path": _relative(path),
        "dataset_name": dataset["dataset_name"],
        "case_count": len(dataset["cases"]),
        "family_count": len(dataset.get("families") or {}),
        "ok": not missing_expected and not missing_acceptable,
        "missing_expected_repo_ids": missing_expected,
        "missing_acceptable_repo_ids": missing_acceptable,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    repos_path = ROOT / args.repos
    records_path = ROOT / args.records
    index_path = ROOT / args.index
    report_path = ROOT / args.report

    repo_ids = set(load_repo_ids(repos_path))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    index = load_index(index_path)
    report = load_report(report_path)
    report_inspection = inspect_report(report, status="serious_mismatch", limit=1)

    datasets = [_check_dataset(ROOT / path, repo_ids) for path in args.cases]
    index_vector_count = len(index.get("vectors") or [])
    checks = [
        {"name": "repos", "ok": len(repo_ids) > 0, "count": len(repo_ids), "path": _relative(repos_path)},
        {
            "name": "records",
            "ok": isinstance(records, list) and len(records) > 0,
            "count": len(records) if isinstance(records, list) else 0,
            "path": _relative(records_path),
        },
        {
            "name": "index",
            "ok": index_vector_count == index.get("record_count") and index_vector_count > 0,
            "record_count": index.get("record_count"),
            "vector_count": index_vector_count,
            "path": _relative(index_path),
        },
        {
            "name": "eval_report",
            "ok": report_inspection["summary"]["serious_mismatch"]["count"] == 0,
            "dataset_name": report.get("dataset_name"),
            "case_count": report.get("case_count"),
            "serious_mismatch_count": report_inspection["summary"]["serious_mismatch"]["count"],
            "path": _relative(report_path),
        },
        *[{"name": "eval_dataset", **dataset} for dataset in datasets],
    ]
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run no-network smoke checks for xists demo artifacts.")
    parser.add_argument("--repos", type=Path, default=Path("repos.txt"), help="Repository list to validate")
    parser.add_argument("--records", type=Path, default=Path("demo-records.json"), help="Demo records JSON")
    parser.add_argument("--index", type=Path, default=Path("demo-index.json"), help="Demo index JSON")
    parser.add_argument("--report", type=Path, default=Path("demo-eval-report.json"), help="Demo eval report JSON")
    parser.add_argument(
        "--cases",
        type=Path,
        nargs="+",
        default=[Path("examples/eval-cases.json"), Path("examples/eval-cases-extended.json")],
        help="Evaluation datasets to validate against repos.txt",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_smoke(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
