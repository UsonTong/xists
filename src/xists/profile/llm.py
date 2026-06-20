"""LLM profile generation against an OpenAI-compatible chat completions API.

This module turns a collected xists record into an ``llm_profile``. It only
feeds the LLM the evidence that was actually collected from the source, and it
instructs the model to abstain rather than invent details when evidence is
missing. xists fills the provenance fields (provider, model, base_url,
generated_at, input_evidence_kinds) itself so they stay trustworthy.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

USER_AGENT = "xists-llm-profile"
CONFIDENCE_VALUES = {"high", "medium", "low"}

PROFILE_SYSTEM_PROMPT = (
    "You analyze a single open-source repository and produce a structured "
    "profile that helps developers decide whether it already solves their "
    "problem.\n"
    "You are given ONLY facts collected from the repository (description, "
    "topics, README excerpt, structure signals). Use nothing else.\n"
    "Rules:\n"
    "- Never invent facts. If the evidence does not support a field, leave it "
    "empty (empty array) or null (summary).\n"
    "- If the evidence is too thin to describe the repository, set "
    "\"abstained\" to true, set \"confidence\" to \"low\", and keep the other "
    "fields empty.\n"
    "- Do not copy marketing language. Be concrete and neutral.\n"
    "- search_phrases are natural-language queries a developer might type to "
    "look for this kind of tool.\n"
    "Respond with a single JSON object and nothing else, using exactly these "
    "keys: summary, use_cases, capabilities, not_for, search_phrases, "
    "confidence, abstained."
)


class LLMError(RuntimeError):
    """Raised when the LLM call or its response cannot be used."""


class LLMNotConfiguredError(LLMError):
    """Raised when no LLM configuration is available in the environment."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


def llm_config_from_env() -> LLMConfig:
    """Build an LLMConfig from environment variables.

    Raises LLMNotConfiguredError if the required values are missing so callers
    can fail fast with a clear message instead of generating records without a
    profile.
    """

    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")

    missing = [
        name
        for name, value in (
            ("LLM_API_KEY", api_key),
            ("LLM_BASE_URL", base_url),
            ("LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise LLMNotConfiguredError(
            "LLM is required to generate records but is not configured. "
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them in your .env (see .env.example)."
        )

    return LLMConfig(api_key=api_key, base_url=base_url, model=model)


def profile_input_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Extract only the collected evidence the LLM is allowed to use."""

    github = record.get("github") or {}
    readme = record.get("readme") or {}
    structure = record.get("structure") or {}

    return {
        "repo_id": record.get("repo_id"),
        "url": record.get("url"),
        "github_description": github.get("description"),
        "github_topics": github.get("topics") or [],
        "primary_language": github.get("language"),
        "readme_excerpt": readme.get("excerpt"),
        "structure_signals": structure.get("signals") or [],
        "evidence_gaps": record.get("evidence_gaps") or [],
    }


def input_evidence_kinds(record: dict[str, Any]) -> list[str]:
    """List which evidence kinds were available as LLM input."""

    return [item.get("kind") for item in record.get("evidence") or [] if item.get("kind")]


def build_profile_messages(profile_input: dict[str, Any]) -> list[dict[str, str]]:
    user_content = (
        "Repository evidence (JSON):\n"
        + json.dumps(profile_input, ensure_ascii=False, indent=2)
        + "\n\nProduce the profile JSON now."
    )
    return [
        {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_llm_profile_response(content: str) -> dict[str, Any]:
    """Parse and normalize the model's JSON response into profile fields."""

    text = content.strip()
    if text.startswith("```"):
        # Strip a fenced code block if the model wrapped the JSON.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMError(f"LLM response was not valid JSON: {error}") from error

    if not isinstance(data, dict):
        raise LLMError("LLM response JSON must be an object")

    confidence = str(data.get("confidence", "low")).lower()
    if confidence not in CONFIDENCE_VALUES:
        confidence = "low"

    summary = data.get("summary")
    summary = summary.strip() if isinstance(summary, str) and summary.strip() else None

    return {
        "summary": summary,
        "use_cases": _coerce_str_list(data.get("use_cases")),
        "capabilities": _coerce_str_list(data.get("capabilities")),
        "not_for": _coerce_str_list(data.get("not_for")),
        "search_phrases": _coerce_str_list(data.get("search_phrases")),
        "confidence": confidence,
        "abstained": bool(data.get("abstained", False)),
    }


def call_llm(config: LLMConfig, messages: list[dict[str, str]], *, timeout: int = 60) -> str:
    """Call an OpenAI-compatible chat completions endpoint, return the content."""

    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    request = urllib.request.Request(config.chat_completions_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8")
        except Exception:
            detail = str(error)
        raise LLMError(f"LLM request failed (HTTP {error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise LLMError(f"LLM request failed: {error}") from error

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise LLMError(f"Unexpected LLM response shape: {data}") from error


def generate_llm_profile(
    record: dict[str, Any],
    config: LLMConfig,
    *,
    caller: Any = call_llm,
) -> dict[str, Any]:
    """Generate an llm_profile for a record.

    ``caller`` is injected so tests can supply a mock instead of hitting the
    network. It must accept (config, messages) and return the message content.
    """

    profile_input = profile_input_from_record(record)
    messages = build_profile_messages(profile_input)
    content = caller(config, messages)
    profile = parse_llm_profile_response(content)

    profile.update(
        {
            "provider": "openai_compatible",
            "model": config.model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_evidence_kinds": input_evidence_kinds(record),
        }
    )
    return profile


def attach_llm_profile(record: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    record["llm_profile"] = profile
    return record
