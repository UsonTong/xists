import threading

import pytest

from xists import __version__
from xists.ingest.github import (
    GITHUB_API_VERSION,
    GitHubSnapshot,
    TokenPool,
    build_graphql_batch_query,
    build_record,
    clean_readme_excerpt,
    fetch_snapshot_graphql,
    fetch_snapshots_graphql,
    evidence_gaps,
    github_token_from_env,
    github_token_from_file,
    parse_github_repo,
    structure_signals,
    tree_paths,
)


def test_parse_github_repo_accepts_owner_repo():
    assert parse_github_repo("facebook/react") == "facebook/react"
    assert parse_github_repo(" facebook/react ") == "facebook/react"


def test_parse_github_repo_accepts_github_urls():
    assert parse_github_repo("https://github.com/facebook/react") == "facebook/react"
    assert parse_github_repo("https://github.com/facebook/react/") == "facebook/react"
    assert parse_github_repo("http://github.com/vuejs/core/tree/main") == "vuejs/core"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "react",
        "facebook/react/extra",
        "https://gitlab.com/foo/bar",
        "https://github.com/foo",
    ],
)
def test_parse_github_repo_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_github_repo(value)


def test_clean_readme_excerpt_removes_badges_and_empty_lines():
    text = """
![Build](https://img.shields.io/badge/build-passing-green)
[![CI](https://example.com/badge.svg)](https://example.com)

# Project

Project does something useful.
---
"""

    assert clean_readme_excerpt(text) == "# Project\nProject does something useful."


def test_clean_readme_excerpt_returns_none_for_empty_content():
    assert clean_readme_excerpt(None) is None
    assert clean_readme_excerpt("\n\n") is None


def test_clean_readme_excerpt_truncates_to_max_chars():
    assert clean_readme_excerpt("abcdef", max_chars=3) == "abc"


def test_tree_paths_extracts_paths():
    tree = {"tree": [{"path": "README.md"}, {"path": "src/main.py"}, {"type": "tree"}]}

    assert tree_paths(tree) == ["README.md", "src/main.py"]
    assert tree_paths(None) == []


def test_structure_signals_detects_common_repository_features():
    paths = [
        "README.md",
        "package.json",
        "pnpm-lock.yaml",
        "src/index.ts",
        "docs/index.md",
        "examples/basic.js",
        "tests/test_main.py",
        "scripts/build.sh",
    ]

    assert structure_signals(paths, readme_present=False) == [
        "has_readme",
        "has_package_json",
        "has_js_lockfile",
        "has_src_directory",
        "has_docs_directory",
        "has_examples_or_fixtures",
        "has_tests",
        "has_scripts_directory",
    ]


def test_evidence_gaps_records_missing_information():
    gaps = evidence_gaps({}, None, [], None)

    assert gaps == [
        "missing_github_description",
        "missing_github_topics",
        "missing_readme_excerpt",
        "missing_repository_tree",
        "missing_tree_paths",
    ]


def test_evidence_gaps_records_truncated_tree():
    gaps = evidence_gaps(
        {"description": "A project", "topics": ["python"]},
        "README excerpt",
        ["README.md"],
        {"truncated": True, "tree": [{"path": "README.md"}]},
    )

    assert gaps == ["truncated_repository_tree"]


def test_github_token_from_file_reads_trimmed_token(tmp_path):
    token_file = tmp_path / "github-token"
    token_file.write_text(" token-value\n", encoding="utf-8")

    assert github_token_from_file(token_file) == ["token-value"]


def test_github_token_from_file_returns_empty_for_missing_or_empty_file(tmp_path):
    assert github_token_from_file(tmp_path / "missing") == []

    token_file = tmp_path / "empty"
    token_file.write_text("\n", encoding="utf-8")
    assert github_token_from_file(token_file) == []


def test_build_record_creates_traceable_record():
    snapshot = GitHubSnapshot(
        requested_repo_id="facebook/react",
        metadata={
            "full_name": "react/react",
            "owner": {"login": "react"},
            "name": "react",
            "html_url": "https://github.com/react/react",
            "description": "The library for web and native user interfaces.",
            "topics": ["javascript", "react", "ui"],
            "stargazers_count": 245995,
            "forks_count": 49277,
            "language": "JavaScript",
            "license": {"spdx_id": "MIT"},
            "archived": False,
            "disabled": False,
            "homepage": "https://react.dev",
            "default_branch": "main",
            "created_at": "2013-05-24T16:15:54Z",
            "updated_at": "2026-06-19T00:00:00Z",
            "pushed_at": "2026-06-19T00:00:00Z",
            "open_issues_count": 1900,
        },
        readme={
            "path": "README.md",
            "html_url": "https://github.com/react/react/blob/main/README.md",
            "download_url": "https://raw.githubusercontent.com/react/react/main/README.md",
        },
        readme_text="React is a JavaScript library for building user interfaces.",
        tree={
            "truncated": False,
            "tree": [
                {"path": "README.md"},
                {"path": "package.json"},
                {"path": "packages/react/index.js"},
                {"path": "scripts/build.js"},
            ],
        },
    )

    record = build_record(snapshot)

    assert record["schema_version"] == 1
    assert record["xists_version"] == __version__
    assert record["github_api_version"] == GITHUB_API_VERSION
    assert record["repo_id_requested"] == "facebook/react"
    assert record["repo_id"] == "react/react"
    assert record["platform"] == "github"
    assert record["owner"] == "react"
    assert record["name"] == "react"
    assert record["url"] == "https://github.com/react/react"
    assert record["github"]["topics"] == ["javascript", "react", "ui"]
    assert record["readme"]["excerpt"] == "React is a JavaScript library for building user interfaces."
    assert record["structure"] == {
        "signals": ["has_readme", "has_package_json", "has_packages_directory", "has_scripts_directory"],
        "tree_file_count": 4,
        "tree_truncated": False,
    }
    assert [item["kind"] for item in record["evidence"]] == [
        "github_description",
        "github_topics",
        "readme_excerpt",
        "structure_signals",
    ]
    assert record["evidence_gaps"] == []
    assert record["lifecycle_state"] == "candidate"
    assert record["snapshot_source"] == "github_api"
    assert record["snapshot_time"].endswith("+00:00")


# --- TokenPool tests ---


def test_token_pool_round_robin():
    pool = TokenPool(["tok_a", "tok_b", "tok_c"])
    assert pool.next_token() == "tok_a"
    assert pool.next_token() == "tok_b"
    assert pool.next_token() == "tok_c"
    assert pool.next_token() == "tok_a"  # wraps around


def test_token_pool_empty_returns_none():
    pool = TokenPool([])
    assert pool.next_token() is None


def test_token_pool_single_token():
    pool = TokenPool(["only_tok"])
    assert pool.next_token() == "only_tok"
    assert pool.next_token() == "only_tok"


def test_token_pool_thread_safety():
    pool = TokenPool(["tok_a", "tok_b", "tok_c"])
    results: list[str] = []
    lock = threading.Lock()

    def collect():
        for _ in range(100):
            token = pool.next_token()
            with lock:
                results.append(token)

    threads = [threading.Thread(target=collect) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 400
    assert all(t in ("tok_a", "tok_b", "tok_c") for t in results)


# --- github_token_from_env tests ---


def test_github_token_from_env_returns_single_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "single_tok")
    monkeypatch.delenv("GITHUB_TOKENS", raising=False)
    assert github_token_from_env() == ["single_tok"]


def test_github_token_from_env_returns_multiple_tokens(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKENS", "tok1, tok2, tok3")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert github_token_from_env() == ["tok1", "tok2", "tok3"]


def test_github_token_from_env_tokens_overrides_single(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKENS", "tok1,tok2")
    monkeypatch.setenv("GITHUB_TOKEN", "single_tok")
    assert github_token_from_env() == ["tok1", "tok2"]


def test_github_token_from_env_returns_empty_when_unset(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKENS", raising=False)
    assert github_token_from_env() == []


# --- github_token_from_file with multiple lines ---


def test_github_token_from_file_reads_multiple_tokens(tmp_path):
    token_file = tmp_path / "tokens"
    token_file.write_text("tok_a\ntok_b\ntok_c\n", encoding="utf-8")
    assert github_token_from_file(token_file) == ["tok_a", "tok_b", "tok_c"]


def test_github_token_from_file_skips_blank_lines(tmp_path):
    token_file = tmp_path / "tokens"
    token_file.write_text("tok_a\n\ntok_b\n  \ntok_c\n", encoding="utf-8")
    assert github_token_from_file(token_file) == ["tok_a", "tok_b", "tok_c"]


# --- GraphQL snapshot tests ---


def test_build_graphql_batch_query_aliases_multiple_repositories():
    query, variables, aliases = build_graphql_batch_query(["facebook/react", "vuejs/core"])

    assert "r0: repository(owner: $owner0, name: $name0)" in query
    assert "r1: repository(owner: $owner1, name: $name1)" in query
    assert "fragment RepoSnapshotFields on Repository" in query
    assert variables == {"owner0": "facebook", "name0": "react", "owner1": "vuejs", "name1": "core"}
    assert aliases == {"r0": ("facebook/react", "facebook"), "r1": ("vuejs/core", "vuejs")}


def _fake_graphql_repository():
    """Return a realistic GraphQL repository payload for testing."""
    return {
        "nameWithOwner": "facebook/react",
        "name": "react",
        "url": "https://github.com/facebook/react",
        "description": "The library for web and native user interfaces.",
        "stargazerCount": 245995,
        "forkCount": 49277,
        "primaryLanguage": {"name": "JavaScript"},
        "licenseInfo": {"spdxId": "MIT"},
        "isArchived": False,
        "isDisabled": False,
        "homepageUrl": "https://react.dev",
        "createdAt": "2013-05-24T16:15:54Z",
        "updatedAt": "2026-06-19T00:00:00Z",
        "pushedAt": "2026-06-19T00:00:00Z",
        "issues": {"totalCount": 1900},
        "pullRequests": {"totalCount": 100},
        "defaultBranchRef": {"name": "main"},
        "repositoryTopics": {
            "nodes": [
                {"topic": {"name": "javascript"}},
                {"topic": {"name": "react"}},
            ]
        },
        "readmeMd": {"text": "# React"},
        "readmeMarkdown": None,
        "readmeRst": None,
        "readmeTxt": None,
        "readmePlain": None,
        "readmemd": None,
        "readmeMarkdownLower": None,
        "readmeRstLower": None,
        "readmeTxtLower": None,
        "readmePlainLower": None,
        "readmeMdMixed": None,
        "readmeMarkdownMixed": None,
        "readmeMixed": None,
        "tree": {
            "entries": [
                {"name": "README.md", "type": "blob",
                 "object": {"entries": []}},
                {"name": "packages", "type": "tree",
                 "object": {"entries": [
                     {"name": "react", "type": "tree",
                      "object": {"entries": [
                          {"name": "__tests__", "type": "tree"}
                      ]}}
                 ]}},
            ]
        },
    }


def test_fetch_snapshot_graphql_maps_repository_payload(monkeypatch):
    def fake_request(query, variables, *, token=None):
        assert variables == {"owner": "facebook", "name": "react"}
        assert token == "tok"
        return {"data": {"repository": _fake_graphql_repository()}}

    monkeypatch.setattr("xists.ingest.github.request_graphql", fake_request)

    snapshot = fetch_snapshot_graphql("facebook/react", token="tok")

    assert snapshot.requested_repo_id == "facebook/react"
    assert snapshot.metadata["full_name"] == "facebook/react"
    assert snapshot.metadata["topics"] == ["javascript", "react"]
    assert snapshot.metadata["language"] == "JavaScript"
    assert snapshot.metadata["open_issues_count"] == 2000
    assert snapshot.readme["path"] == "README.md"
    assert snapshot.readme_text == "# React"
    paths = tree_paths(snapshot.tree)
    assert "README.md" in paths
    assert "packages/react" in paths
    assert "packages/react/__tests__" in paths


def test_collect_record_graphql_sets_snapshot_source(monkeypatch):
    from xists.ingest.github import collect_record_graphql

    def fake_request(query, variables, *, token=None):
        return {"data": {"repository": _fake_graphql_repository()}}

    monkeypatch.setattr("xists.ingest.github.request_graphql", fake_request)

    record = collect_record_graphql("facebook/react", token="tok")

    assert record["snapshot_source"] == "github_graphql"
    assert record["repo_id"] == "facebook/react"


def test_collect_records_graphql_sets_snapshot_source(monkeypatch):
    from xists.ingest.github import collect_records_graphql

    repo2 = {**_fake_graphql_repository(), "nameWithOwner": "vuejs/core", "name": "core", "url": "https://github.com/vuejs/core"}

    def fake_request(query, variables, *, token=None):
        return {"data": {"r0": _fake_graphql_repository(), "r1": repo2}}

    monkeypatch.setattr("xists.ingest.github.request_graphql", fake_request)

    records = collect_records_graphql(["facebook/react", "vuejs/core"], token="tok")

    assert len(records) == 2
    assert all(r["snapshot_source"] == "github_graphql" for r in records)
    assert [r["repo_id"] for r in records] == ["facebook/react", "vuejs/core"]


def test_fetch_snapshots_graphql_maps_multiple_repositories(monkeypatch):
    def fake_request(query, variables, *, token=None):
        assert token == "tok"
        return {
            "data": {
                "r0": {
                    "nameWithOwner": "facebook/react",
                    "name": "react",
                    "url": "https://github.com/facebook/react",
                    "description": "React",
                    "stargazerCount": 1,
                    "forkCount": 2,
                    "primaryLanguage": {"name": "JavaScript"},
                    "licenseInfo": {"spdxId": "MIT"},
                    "isArchived": False,
                    "isDisabled": False,
                    "homepageUrl": None,
                    "createdAt": "2020-01-01T00:00:00Z",
                    "updatedAt": "2020-01-02T00:00:00Z",
                    "pushedAt": "2020-01-03T00:00:00Z",
                    "issues": {"totalCount": 3},
                    "pullRequests": {"totalCount": 7},
                    "defaultBranchRef": {"name": "main"},
                    "repositoryTopics": {"nodes": []},
                    "readmeMd": {"text": "# React"},
                    "readmeMarkdown": None, "readmeRst": None, "readmeTxt": None, "readmePlain": None,
                    "readmemd": None, "readmeMarkdownLower": None, "readmeRstLower": None,
                    "readmeTxtLower": None, "readmePlainLower": None, "readmeMdMixed": None,
                    "readmeMarkdownMixed": None, "readmeMixed": None,
                    "tree": {"entries": []},
                },
                "r1": {
                    "nameWithOwner": "vuejs/core",
                    "name": "core",
                    "url": "https://github.com/vuejs/core",
                    "description": "Vue",
                    "stargazerCount": 4,
                    "forkCount": 5,
                    "primaryLanguage": {"name": "TypeScript"},
                    "licenseInfo": {"spdxId": "MIT"},
                    "isArchived": False,
                    "isDisabled": False,
                    "homepageUrl": None,
                    "createdAt": "2020-01-01T00:00:00Z",
                    "updatedAt": "2020-01-02T00:00:00Z",
                    "pushedAt": "2020-01-03T00:00:00Z",
                    "issues": {"totalCount": 6},
                    "pullRequests": {"totalCount": 8},
                    "defaultBranchRef": {"name": "main"},
                    "repositoryTopics": {"nodes": []},
                    "readmeMd": {"text": "# Vue"},
                    "readmeMarkdown": None, "readmeRst": None, "readmeTxt": None, "readmePlain": None,
                    "readmemd": None, "readmeMarkdownLower": None, "readmeRstLower": None,
                    "readmeTxtLower": None, "readmePlainLower": None, "readmeMdMixed": None,
                    "readmeMarkdownMixed": None, "readmeMixed": None,
                    "tree": {"entries": []},
                },
            }
        }

    monkeypatch.setattr("xists.ingest.github.request_graphql", fake_request)

    snapshots = fetch_snapshots_graphql(["facebook/react", "vuejs/core"], token="tok")

    assert [s.metadata["full_name"] for s in snapshots] == ["facebook/react", "vuejs/core"]
    assert [s.readme_text for s in snapshots] == ["# React", "# Vue"]
    assert [s.metadata["open_issues_count"] for s in snapshots] == [10, 14]


def test_github_token_from_file_returns_empty_for_missing_file(tmp_path):
    assert github_token_from_file(tmp_path / "missing") == []
