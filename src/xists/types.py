"""Core domain types and schema definitions for xists."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class LLMProfile(TypedDict, total=False):
    """Schema v2 LLM Profile generated from repository metadata."""

    summary: str | None
    use_cases: list[str]
    capabilities: list[str]
    not_for: list[str]
    aliases: list[str]
    project_type: str | None
    ecosystem: list[str]
    replaces: list[str]
    related_projects: list[str]
    search_text: str | None
    search_phrases: list[str]
    confidence: Literal["high", "medium", "low"] | str
    abstained: bool
    provider: str | None
    model: str | None
    generated_at: str | None
    prompt_version: int | None
    prompt_hash: str | None


class GitHubMetadata(TypedDict, total=False):
    """Raw GitHub repository metadata captured during ingestion."""

    repo_id: str
    name: str
    description: str | None
    stars: int
    forks: int
    primary_language: str | None
    languages: list[str]
    topics: list[str]
    archived: bool
    disabled: bool
    license: str | None
    default_branch: str
    created_at: str | None
    updated_at: str | None
    pushed_at: str | None
    readme_content: str | None
    readme_size: int
    has_readme: bool
    status: str


class StructureSignals(TypedDict, total=False):
    """Signals derived from repo structure (e.g. entry points, file tree)."""

    has_package_json: bool
    has_pyproject_toml: bool
    has_cargo_toml: bool
    has_go_mod: bool
    has_dockerfile: bool
    has_license: bool


class Record(TypedDict, total=False):
    """Schema v2 repository record."""

    schema_version: int
    repo_id: str
    name: str
    url: str
    github: GitHubMetadata | dict[str, Any]
    structure: StructureSignals | dict[str, Any] | None
    llm_profile: LLMProfile | dict[str, Any]
    status: str


class QueryIntent(TypedDict, total=False):
    """Extracted query intent and classification."""

    type: (
        Literal[
            "exact",
            "lookup",
            "functional",
            "ecosystem",
            "ambiguous",
            "conversational",
            "exploratory",
        ]
        | str
    )
    primary_terms: list[str]
    cjk_terms: list[str]
    explicit_repo: str | None
    language: str | None
    ecosystem: str | None
    project_type: str | None


class SearchResultItem(TypedDict, total=False):
    """Ranked search result item."""

    repo_id: str
    name: str
    url: str
    score: float
    semantic_score: float
    bm25_score: float
    metadata_score: float
    confidence: Literal["high_confidence", "medium_confidence", "exploratory", "abstain"] | str
    summary: str | None
    why: list[str]
    metadata: dict[str, Any]


class SearchResponse(TypedDict, total=False):
    """Public search response format."""

    query: str
    query_intent: QueryIntent | dict[str, Any]
    abstained: bool
    total_candidates: int
    results: list[SearchResultItem]
    elapsed_ms: float


class IndexVectorEntry(TypedDict, total=False):
    """Vector metadata entry within an index document."""

    repo_id: str
    embedding_input_fingerprint: str | None
    vector: list[float] | str | None  # Inline list or Base64 string in v3
    metadata: dict[str, Any]


class IndexManifest(TypedDict, total=False):
    """Full schema for index.json manifest."""

    index_version: int
    record_schema_version: int
    embedding_model: str
    embedding_base_url: str | None
    embedding_input_version: int
    dimension: int
    built_at: str
    record_count: int
    vectors_file: str | None
    skipped: list[dict[str, Any]]
    vectors: list[IndexVectorEntry]


class EvalCase(TypedDict, total=False):
    """Single test case for retrieval evaluation."""

    id: str
    query: str
    category: str
    expected_repo_id: str
    acceptable_alternatives: list[str]
    notes: str | None


class EvalResult(TypedDict, total=False):
    """Evaluation output for a single case."""

    id: str
    query: str
    category: str
    expected_repo_id: str
    top_result_repo_id: str | None
    top_result_score: float | None
    hit_rank: int | None
    is_exact_match: bool
    is_acceptable_alternative: bool
    is_serious_mismatch: bool
    abstained: bool
    status: (
        Literal[
            "exact_match",
            "acceptable_alternative",
            "serious_mismatch",
            "insufficient_evidence",
            "correct_abstain",
            "false_positive_abstain",
        ]
        | str
    )


class EvalSummary(TypedDict, total=False):
    """Summary metrics of an evaluation run."""

    total_cases: int
    exact_matches: int
    acceptable_alternatives: int
    serious_mismatches: int
    recall_at_1: float
    recall_at_5: float
    mrr: float


class EvalReport(TypedDict, total=False):
    """Full schema for evaluation report JSON."""

    xists_version: str
    schema_version: int
    evaluated_at: str
    elapsed_ms: float
    retrieval_configuration: dict[str, Any]
    summary: EvalSummary
    cases: list[EvalResult]


__all__ = [
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "EvalSummary",
    "GitHubMetadata",
    "IndexManifest",
    "IndexVectorEntry",
    "LLMProfile",
    "QueryIntent",
    "Record",
    "SearchResponse",
    "SearchResultItem",
    "StructureSignals",
]
