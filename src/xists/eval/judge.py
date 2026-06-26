"""LLM-assisted pairwise judgement for retrieval mismatches."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

from xists.profile.llm import LLMConfig, LLMError, LLMResponse, call_llm, profile_input_from_record

JUDGE_PROMPT_VERSION = 3
JUDGE_CONFIDENCE_VALUES = {"high", "medium", "low"}
JUDGE_VERDICTS = {"acceptable_substitute", "serious_mismatch", "insufficient_evidence"}
JUDGE_DIFFERENCE_SIZES = {"small", "moderate", "large"}
JUDGE_QUERY_SPECIFICITY = {"underspecified", "specified"}

JUDGE_SYSTEM_PROMPT = (
    "You are judging one evaluation case for a repository search system.\n"
    "You compare two repositories for the same query:\n"
    "- expected repository: the reference repository stored in the evaluation dataset\n"
    "- top1 repository: the repository currently ranked first by the search system\n"
    "The expected repository is not always the only acceptable answer. Your job is to decide whether the top1 repository is still acceptable for the query, even when it differs from the expected repository.\n"
    "You are given ONLY the query and structured evidence extracted from the two repositories. Use nothing else.\n"
    "Definitions:\n"
    "- query: the user's search request in natural language.\n"
    "- expected repository: the dataset's reference answer for this query. It is a reference answer, not automatically the only valid answer.\n"
    "- top1 repository: the repository currently ranked first by the search system for this query.\n"
    "- acceptable_substitute: the top1 repository is different from the expected repository, but still satisfies the query well enough that a developer would reasonably accept it.\n"
    "- serious_mismatch: the top1 repository fails a material requirement expressed in the query.\n"
    "- query_specificity=specified means the query explicitly names a material constraint such as language, runtime, ecosystem, protocol, framework family, or implementation style.\n"
    "- query_specificity=underspecified means the query is broad and leaves those constraints open.\n"
    "- language_ecosystem_material=true only when language or ecosystem differences change whether the top1 repository actually satisfies the query.\n"
    "Decision procedure:\n"
    "1. Extract the explicit constraints from the query.\n"
    "2. Decide whether the query is specified or underspecified. A short query can still be specified if it explicitly names a language, ecosystem, framework family, protocol, or implementation style.\n"
    "3. Check whether the top1 repository satisfies all explicit, material constraints.\n"
    "4. If top1 violates any explicit, material constraint, return serious_mismatch.\n"
    "5. If top1 satisfies the query but is simply a different reasonable option, return acceptable_substitute.\n"
    "6. If the evidence is too thin to judge fairly, return insufficient_evidence.\n"
    "Calibration examples:\n"
    "Example A:\n"
    "query='Python web framework for building APIs'\n"
    "expected repository=FastAPI\n"
    "top1 repository=API Platform (PHP)\n"
    "output: verdict=serious_mismatch, query_specificity=specified, language_ecosystem_material=true\n"
    "because Python is an explicit material constraint and top1 violates it.\n"
    "Example B:\n"
    "query='Go web framework'\n"
    "expected repository=Echo\n"
    "top1 repository=Fiber\n"
    "output: verdict=acceptable_substitute, query_specificity=specified, language_ecosystem_material=false\n"
    "because the query specifies Go, but does not specify which Go framework; both satisfy the query.\n"
    "Example C:\n"
    "query='collaborative filtering recommender system'\n"
    "expected repository=one recommender implementation\n"
    "top1 repository=another recommender implementation\n"
    "output: verdict=acceptable_substitute, query_specificity=underspecified, language_ecosystem_material=false\n"
    "because the query is broad and both satisfy the same need.\n"
    "Rules:\n"
    "- Never invent facts beyond the provided evidence.\n"
    "- Do not treat 'expected repository' as automatically better or automatically the only valid answer.\n"
    "- Keep reason_short concrete and reference the query constraint or lack of constraint.\n"
    "Respond with a single JSON object and nothing else, using exactly these keys: verdict, difference_size, query_specificity, language_ecosystem_material, reason_short, expected_advantages, top1_advantages, confidence."
)


def build_judge_messages(
    query: str,
    *,
    expected_record: dict[str, Any],
    top1_record: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "query": query,
        "expected_repo": profile_input_from_record(expected_record),
        "top1_repo": profile_input_from_record(top1_record),
    }
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Pairwise evaluation evidence (JSON):\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\nProduce the judgement JSON now.",
        },
    ]


def judge_prompt_hash() -> str:
    payload = json.dumps(
        {
            "prompt_version": JUDGE_PROMPT_VERSION,
            "system_prompt": JUDGE_SYSTEM_PROMPT,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_judge_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMError(f"Judge response was not valid JSON: {error}") from error

    if not isinstance(data, dict):
        raise LLMError("Judge response JSON must be an object")

    verdict = str(data.get("verdict", "insufficient_evidence")).strip().lower()
    if verdict not in JUDGE_VERDICTS:
        verdict = "insufficient_evidence"

    difference_size = str(data.get("difference_size", "moderate")).strip().lower()
    if difference_size not in JUDGE_DIFFERENCE_SIZES:
        difference_size = "moderate"

    query_specificity = str(data.get("query_specificity", "underspecified")).strip().lower()
    if query_specificity not in JUDGE_QUERY_SPECIFICITY:
        query_specificity = "underspecified"

    confidence = str(data.get("confidence", "low")).strip().lower()
    if confidence not in JUDGE_CONFIDENCE_VALUES:
        confidence = "low"

    reason_short = data.get("reason_short")
    reason_short = reason_short.strip() if isinstance(reason_short, str) and reason_short.strip() else None

    return {
        "verdict": verdict,
        "difference_size": difference_size,
        "query_specificity": query_specificity,
        "language_ecosystem_material": bool(data.get("language_ecosystem_material", False)),
        "reason_short": reason_short,
        "expected_advantages": _coerce_str_list(data.get("expected_advantages")),
        "top1_advantages": _coerce_str_list(data.get("top1_advantages")),
        "confidence": confidence,
    }


def judge_top1_vs_expected(
    query: str,
    *,
    expected_record: dict[str, Any],
    top1_record: dict[str, Any],
    config: LLMConfig,
    caller: Any | None = None,
) -> dict[str, Any]:
    messages = build_judge_messages(query, expected_record=expected_record, top1_record=top1_record)
    started = time.perf_counter()
    caller_fn = caller or call_llm
    response = caller_fn(config, messages)
    duration_seconds = time.perf_counter() - started
    if isinstance(response, LLMResponse):
        content = response.content
        token_usage = response.token_usage
    else:
        content = response
        token_usage = None
    judgement = parse_judge_response(content)
    judgement.update(
        {
            "provider": "openai_compatible",
            "model": config.model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "prompt_version": JUDGE_PROMPT_VERSION,
            "prompt_hash": judge_prompt_hash(),
            "duration_seconds": duration_seconds,
            "token_usage": token_usage,
        }
    )
    return judgement
