"""Shared helpers for xists records and record-level validation."""

from __future__ import annotations

from collections import Counter
from typing import Any

RECORD_SCHEMA_VERSION = 2
CONFIDENCE_VALUES = {"high", "medium", "low"}
MIN_SEARCH_TEXT_CHARS = 24
RETRIEVAL_PROFILE_FIELDS = (
    "summary",
    "use_cases",
    "capabilities",
    "aliases",
    "project_type",
    "ecosystem",
    "replaces",
    "related_projects",
    "search_text",
    "search_phrases",
)


def _clean_str(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _clean_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def normalize_llm_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    profile = profile or {}
    confidence = str(profile.get("confidence", "low")).lower()
    if confidence not in CONFIDENCE_VALUES:
        confidence = "low"

    normalized = {
        "summary": _clean_str(profile.get("summary")),
        "use_cases": _clean_str_list(profile.get("use_cases")),
        "capabilities": _clean_str_list(profile.get("capabilities")),
        "not_for": _clean_str_list(profile.get("not_for")),
        "aliases": _clean_str_list(profile.get("aliases")),
        "project_type": _clean_str(profile.get("project_type")),
        "ecosystem": _clean_str_list(profile.get("ecosystem")),
        "replaces": _clean_str_list(profile.get("replaces")),
        "related_projects": _clean_str_list(profile.get("related_projects")),
        "search_text": _clean_str(profile.get("search_text")),
        "search_phrases": _clean_str_list(profile.get("search_phrases")),
        "confidence": confidence,
        "abstained": bool(profile.get("abstained", False)),
        "provider": profile.get("provider"),
        "model": profile.get("model"),
        "generated_at": profile.get("generated_at"),
        "prompt_version": profile.get("prompt_version"),
        "prompt_hash": profile.get("prompt_hash"),
    }
    return normalized


def record_schema_version(record: dict[str, Any]) -> int | None:
    value = record.get("schema_version")
    return value if isinstance(value, int) else None


def record_repo_id(record: dict[str, Any]) -> str | None:
    value = record.get("repo_id") or record.get("repo_id_requested")
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def record_profile(record: dict[str, Any]) -> dict[str, Any]:
    profile = record.get("llm_profile")
    return normalize_llm_profile(profile if isinstance(profile, dict) else None)


def preserve_retrieval_profile_fields(
    existing: dict[str, Any] | None, generated: dict[str, Any]
) -> dict[str, Any]:
    """Keep existing retrieval evidence when a refreshed profile omits it."""

    preserved = dict(generated)
    existing_profile = normalize_llm_profile(existing)
    generated_profile = normalize_llm_profile(generated)
    for field in RETRIEVAL_PROFILE_FIELDS:
        if not generated_profile[field] and existing_profile[field]:
            preserved[field] = existing_profile[field]
    if generated_profile["abstained"] and existing_profile["search_text"]:
        preserved["abstained"] = existing_profile["abstained"]
        preserved["confidence"] = existing_profile["confidence"]
    return preserved


def profile_refresh_reason(
    record: dict[str, Any],
    *,
    only_missing_search_text: bool = False,
    only_missing_summary: bool = False,
    expected_prompt_version: int | None = None,
) -> str | None:
    profile = record_profile(record)
    schema_version = record_schema_version(record)
    if only_missing_summary:
        return None if profile.get("summary") else "missing_summary"
    if only_missing_search_text:
        return None if profile.get("search_text") else "missing_search_text"
    if schema_version != RECORD_SCHEMA_VERSION:
        return "schema_version_mismatch"
    if expected_prompt_version is not None and profile.get("prompt_version") != expected_prompt_version:
        return "profile_prompt_version_mismatch"
    if profile.get("search_text") is None:
        return "missing_search_text"
    if profile.get("aliases") == []:
        return "missing_aliases"
    if profile.get("project_type") is None:
        return "missing_project_type"
    if profile.get("ecosystem") == []:
        return "missing_ecosystem"
    return None


def records_validation_report(
    records: list[dict[str, Any]],
    *,
    expected_schema_version: int = RECORD_SCHEMA_VERSION,
    expected_profile_prompt_version: int | None = None,
) -> dict[str, Any]:
    issues: dict[str, Counter[str]] = {
        "errors": Counter(),
        "warnings": Counter(),
    }
    duplicates: list[str] = []
    seen_ids: set[str] = set()
    schema_versions: Counter[int | None] = Counter()
    low_confidence: list[str] = []
    abstained: list[str] = []
    prompt_versions: Counter[int | None] = Counter()
    archived = 0
    disabled = 0
    missing_readme = 0
    short_search_text: list[str] = []
    valid_records = 0

    for record in records:
        record_has_error = False
        repo_id = record_repo_id(record)
        schema_versions[record_schema_version(record)] += 1
        profile = record_profile(record)
        prompt_versions[profile.get("prompt_version") if isinstance(profile.get("prompt_version"), int) else None] += 1
        github = record.get("github") if isinstance(record.get("github"), dict) else {}
        if github.get("archived") is True:
            archived += 1
        if github.get("disabled") is True:
            disabled += 1
        if not record.get("readme"):
            missing_readme += 1
        if repo_id is None:
            issues["errors"]["missing_repo_id"] += 1
            continue

        if repo_id in seen_ids:
            duplicates.append(repo_id)
        else:
            seen_ids.add(repo_id)

        if record_schema_version(record) != expected_schema_version:
            issues["errors"]["schema_version_mismatch"] += 1
            record_has_error = True

        if not record.get("url"):
            issues["errors"]["missing_url"] += 1
            record_has_error = True
        if not record.get("name"):
            issues["errors"]["missing_name"] += 1
            record_has_error = True

        if not record.get("llm_profile"):
            issues["errors"]["missing_llm_profile"] += 1
            record_has_error = True
            continue

        profile_abstained = bool(profile.get("abstained"))
        if profile.get("summary") is None:
            if profile_abstained:
                issues["warnings"]["abstained_missing_summary"] += 1
            else:
                issues["errors"]["missing_summary"] += 1
                record_has_error = True
        search_text = profile.get("search_text")
        if not search_text:
            if profile_abstained:
                issues["warnings"]["abstained_missing_search_text"] += 1
            else:
                issues["errors"]["missing_search_text"] += 1
                record_has_error = True
        elif len(search_text) < MIN_SEARCH_TEXT_CHARS:
            issues["warnings"]["search_text_too_short"] += 1
            short_search_text.append(repo_id)
        if not profile.get("aliases"):
            issues["warnings"]["missing_aliases"] += 1
        if profile.get("project_type") is None:
            issues["warnings"]["missing_project_type"] += 1
        if not profile.get("ecosystem"):
            issues["warnings"]["missing_ecosystem"] += 1
        if expected_profile_prompt_version is not None and profile.get("prompt_version") != expected_profile_prompt_version:
            issues["warnings"]["profile_prompt_version_mismatch"] += 1
        if profile.get("confidence") == "low":
            low_confidence.append(repo_id)
            issues["warnings"]["low_confidence_profile"] += 1
        if profile_abstained:
            abstained.append(repo_id)
            issues["warnings"]["profile_abstained"] += 1
        if not record_has_error:
            valid_records += 1

    if duplicates:
        issues["errors"]["duplicate_repo_id"] = len(duplicates)
        valid_records = max(valid_records - len(duplicates), 0)

    quality = {
        "ok": valid_records,
        "missing_search_text": issues["errors"].get("missing_search_text", 0),
        "missing_aliases": issues["warnings"].get("missing_aliases", 0),
        "search_text_too_short": issues["warnings"].get("search_text_too_short", 0),
        "profile_abstained": len(abstained),
        "low_confidence": len(low_confidence),
        "archived": archived,
        "disabled": disabled,
        "missing_readme": missing_readme,
        "duplicates": len(duplicates),
    }

    return {
        "schema_version": expected_schema_version,
        "record_count": len(records),
        "schema_versions": {str(key): value for key, value in schema_versions.items()},
        "prompt_versions": {str(key): value for key, value in prompt_versions.items()},
        "quality": quality,
        "errors": dict(issues["errors"]),
        "warnings": dict(issues["warnings"]),
        "duplicates": duplicates,
        "low_confidence": low_confidence,
        "abstained": abstained,
        "short_search_text": short_search_text,
        "ok": not issues["errors"],
    }
