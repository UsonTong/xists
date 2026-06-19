import pytest

from xists.ingest.github import (
    GitHubSnapshot,
    build_record,
    clean_readme_excerpt,
    evidence_gaps,
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

    assert github_token_from_file(token_file) == "token-value"


def test_github_token_from_file_returns_none_for_missing_or_empty_file(tmp_path):
    assert github_token_from_file(tmp_path / "missing") is None

    token_file = tmp_path / "empty"
    token_file.write_text("\n", encoding="utf-8")
    assert github_token_from_file(token_file) is None


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
