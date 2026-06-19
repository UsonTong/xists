"""GitHub API collection for xists records."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GITHUB_API_BASE = "https://api.github.com"
USER_AGENT = "xists-record-ingest"


class GitHubAPIError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class GitHubSnapshot:
    requested_repo_id: str
    metadata: dict[str, Any]
    readme: dict[str, Any] | None
    readme_text: str | None
    tree: dict[str, Any] | None


def parse_github_repo(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("empty GitHub repository identifier")

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urllib.parse.urlparse(value)
        if parsed.netloc.lower() != "github.com":
            raise ValueError(f"not a GitHub URL: {value}")
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"GitHub URL must include owner and repo: {value}")
        return f"{parts[0]}/{parts[1]}"

    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"repo must use owner/repo format: {value}")
    return value


def request_json(path: str, token: str | None = None) -> dict[str, Any]:
    url = path if path.startswith("https://") else f"{GITHUB_API_BASE}/{path.lstrip('/')}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
            message = payload.get("message") or str(error)
        except Exception:
            message = str(error)
        raise GitHubAPIError(message, status=error.code) from error


def fetch_snapshot(repo_id: str, token: str | None = None) -> GitHubSnapshot:
    requested = parse_github_repo(repo_id)
    metadata = request_json(f"repos/{requested}", token=token)

    readme = None
    readme_text = None
    try:
        readme = request_json(f"repos/{requested}/readme", token=token)
        content = readme.get("content")
        if content and readme.get("encoding") == "base64":
            readme_text = base64.b64decode(content).decode("utf-8", errors="replace")
    except GitHubAPIError as error:
        if error.status != 404:
            raise

    tree = None
    default_branch = metadata.get("default_branch")
    if default_branch:
        try:
            tree = request_json(
                f"repos/{requested}/git/trees/{default_branch}?recursive=1",
                token=token,
            )
        except GitHubAPIError:
            tree = None

    return GitHubSnapshot(
        requested_repo_id=requested,
        metadata=metadata,
        readme=readme,
        readme_text=readme_text,
        tree=tree,
    )


def clean_readme_excerpt(text: str | None, max_chars: int = 1200) -> str | None:
    if not text:
        return None

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[!") or line.startswith("!["):
            continue
        if line.count("img.shields.io") or line.count("badge.svg"):
            continue
        if re.fullmatch(r"[#*`_\-\s]+", line):
            continue
        lines.append(line)

    excerpt = "\n".join(lines)
    return excerpt[:max_chars] if excerpt else None


def tree_paths(tree: dict[str, Any] | None) -> list[str]:
    if not tree:
        return []
    return [item.get("path", "") for item in tree.get("tree", []) if item.get("path")]


def structure_signals(paths: list[str], readme_present: bool) -> list[str]:
    path_set = set(paths)

    def any_exact(*names: str) -> bool:
        return any(name in path_set for name in names)

    def any_prefix(prefix: str) -> bool:
        prefix = prefix.rstrip("/") + "/"
        return any(path.startswith(prefix) for path in paths)

    def any_contains(fragment: str) -> bool:
        return any(fragment in path for path in paths)

    signals: list[str] = []
    if readme_present or any(path.lower().startswith("readme") for path in paths):
        signals.append("has_readme")
    if any_exact("package.json"):
        signals.append("has_package_json")
    if any_exact("pyproject.toml", "setup.py", "requirements.txt"):
        signals.append("has_python_manifest")
    if any_exact("go.mod"):
        signals.append("has_go_mod")
    if any_exact("Cargo.toml"):
        signals.append("has_cargo_manifest")
    if any_exact("yarn.lock", "package-lock.json", "pnpm-lock.yaml"):
        signals.append("has_js_lockfile")
    if any_prefix("src"):
        signals.append("has_src_directory")
    if any_prefix("packages"):
        signals.append("has_packages_directory")
    if any_prefix("docs") or any_prefix("documentation") or any_prefix("website"):
        signals.append("has_docs_directory")
    if any_prefix("examples") or any_prefix("fixtures"):
        signals.append("has_examples_or_fixtures")
    if any_prefix("tests") or any_contains("__tests__") or any(path.endswith((".test.js", ".test.ts", "_test.go")) for path in paths):
        signals.append("has_tests")
    if any_prefix("scripts"):
        signals.append("has_scripts_directory")
    return signals


def evidence_gaps(metadata: dict[str, Any], readme_excerpt: str | None, paths: list[str], tree: dict[str, Any] | None) -> list[str]:
    gaps: list[str] = []
    if not metadata.get("description"):
        gaps.append("missing_github_description")
    if not metadata.get("topics"):
        gaps.append("missing_github_topics")
    if not readme_excerpt:
        gaps.append("missing_readme_excerpt")
    if tree is None:
        gaps.append("missing_repository_tree")
    elif tree.get("truncated"):
        gaps.append("truncated_repository_tree")
    if not paths:
        gaps.append("missing_tree_paths")
    return gaps


def build_record(snapshot: GitHubSnapshot) -> dict[str, Any]:
    metadata = snapshot.metadata
    readme_excerpt = clean_readme_excerpt(snapshot.readme_text)
    paths = tree_paths(snapshot.tree)
    signals = structure_signals(paths, bool(snapshot.readme))

    repo_url = metadata.get("html_url")
    topics = metadata.get("topics") or []
    evidence: list[dict[str, Any]] = []
    if metadata.get("description"):
        evidence.append(
            {
                "kind": "github_description",
                "text": metadata.get("description"),
                "source_url": repo_url,
            }
        )
    if topics:
        evidence.append(
            {
                "kind": "github_topics",
                "text": ", ".join(topics),
                "source_url": f"{repo_url}/topics" if repo_url else None,
            }
        )
    if readme_excerpt and snapshot.readme:
        evidence.append(
            {
                "kind": "readme_excerpt",
                "text": readme_excerpt,
                "source_url": snapshot.readme.get("html_url"),
            }
        )
    if signals:
        evidence.append(
            {
                "kind": "structure_signals",
                "text": ", ".join(signals),
                "source_url": repo_url,
            }
        )

    return {
        "record_version": "xists-record-v1",
        "repo_id_requested": snapshot.requested_repo_id,
        "repo_id": metadata.get("full_name"),
        "platform": "github",
        "owner": (metadata.get("owner") or {}).get("login"),
        "name": metadata.get("name"),
        "url": repo_url,
        "github": {
            "description": metadata.get("description"),
            "topics": topics,
            "stars": metadata.get("stargazers_count"),
            "forks": metadata.get("forks_count"),
            "language": metadata.get("language"),
            "license": (metadata.get("license") or {}).get("spdx_id"),
            "archived": metadata.get("archived"),
            "disabled": metadata.get("disabled"),
            "homepage": metadata.get("homepage"),
            "default_branch": metadata.get("default_branch"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "pushed_at": metadata.get("pushed_at"),
            "open_issues": metadata.get("open_issues_count"),
        },
        "readme": None
        if not snapshot.readme
        else {
            "path": snapshot.readme.get("path"),
            "source_url": snapshot.readme.get("html_url"),
            "download_url": snapshot.readme.get("download_url"),
            "excerpt": readme_excerpt,
        },
        "structure": {
            "signals": signals,
            "tree_file_count": len(paths),
            "tree_truncated": None if snapshot.tree is None else bool(snapshot.tree.get("truncated")),
        },
        "evidence": evidence,
        "evidence_gaps": evidence_gaps(metadata, readme_excerpt, paths, snapshot.tree),
        "lifecycle_state": "candidate",
        "snapshot_source": "github_api",
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
    }


def collect_record(repo_id: str, token: str | None = None) -> dict[str, Any]:
    return build_record(fetch_snapshot(repo_id, token=token))


def github_token_from_file(path: Path) -> str | None:
    try:
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return token or None


def github_token_from_env() -> str | None:
    return os.environ.get("GITHUB_TOKEN")
