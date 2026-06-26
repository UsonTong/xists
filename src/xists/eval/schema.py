"""Load and validate retrieval evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation dataset is invalid."""


SUPPORTED_SCHEMA_VERSION = 1


def _expect_string(value: Any, *, field: str, case_id: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        prefix = f"case {case_id}: " if case_id else ""
        raise EvaluationDatasetError(f"{prefix}{field} must be a non-empty string")
    return value.strip()


def _expect_string_list(value: Any, *, field: str, case_id: str | None = None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        prefix = f"case {case_id}: " if case_id else ""
        raise EvaluationDatasetError(f"{prefix}{field} must be a list of strings")
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            prefix = f"case {case_id}: " if case_id else ""
            raise EvaluationDatasetError(f"{prefix}{field} must contain only non-empty strings")
        values.append(item.strip())
    return values


def load_dataset(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise EvaluationDatasetError(f"Evaluation cases file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise EvaluationDatasetError(f"Evaluation cases file is not valid JSON: {error}") from error
    return normalize_dataset(raw)


def normalize_dataset(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise EvaluationDatasetError("Evaluation dataset must be a JSON object")

    schema_version = raw.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise EvaluationDatasetError(
            f"Unsupported evaluation schema_version {schema_version!r}; expected {SUPPORTED_SCHEMA_VERSION}"
        )

    dataset_name = _expect_string(raw.get("dataset_name"), field="dataset_name")
    raw_families = raw.get("families") or {}
    if not isinstance(raw_families, dict):
        raise EvaluationDatasetError("families must be an object mapping family name to repo ids")

    families: dict[str, list[str]] = {}
    for family_name, repo_ids in raw_families.items():
        normalized_name = _expect_string(family_name, field="families key")
        families[normalized_name] = _expect_string_list(repo_ids, field=f"families.{normalized_name}")

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvaluationDatasetError("cases must be a non-empty list")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise EvaluationDatasetError(f"case at index {index} must be an object")
        case_id = _expect_string(raw_case.get("id"), field="id")
        if case_id in seen_ids:
            raise EvaluationDatasetError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)

        query = _expect_string(raw_case.get("query"), field="query", case_id=case_id)
        expected_repo_id = _expect_string(
            raw_case.get("expected_repo_id"), field="expected_repo_id", case_id=case_id
        )
        acceptable_repo_ids = _expect_string_list(
            raw_case.get("acceptable_repo_ids"), field="acceptable_repo_ids", case_id=case_id
        )
        acceptable_families = _expect_string_list(
            raw_case.get("acceptable_families"), field="acceptable_families", case_id=case_id
        )
        tags = _expect_string_list(raw_case.get("tags"), field="tags", case_id=case_id)
        notes = raw_case.get("notes")
        if notes is not None and (not isinstance(notes, str) or not notes.strip()):
            raise EvaluationDatasetError(f"case {case_id}: notes must be a non-empty string when provided")

        unknown_families = [name for name in acceptable_families if name not in families]
        if unknown_families:
            raise EvaluationDatasetError(
                f"case {case_id}: unknown acceptable_families: {', '.join(unknown_families)}"
            )

        acceptable_set = {expected_repo_id, *acceptable_repo_ids}
        for family_name in acceptable_families:
            acceptable_set.update(families[family_name])

        cases.append(
            {
                "id": case_id,
                "query": query,
                "expected_repo_id": expected_repo_id,
                "acceptable_repo_ids": acceptable_repo_ids,
                "acceptable_families": acceptable_families,
                "acceptable_set": sorted(acceptable_set),
                "tags": tags,
                "notes": notes.strip() if isinstance(notes, str) else None,
            }
        )

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "families": families,
        "cases": cases,
    }
