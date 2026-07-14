"""GitHub API collection for xists records."""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xists import __version__
from xists.records import RECORD_SCHEMA_VERSION

GITHUB_API_BASE = "https://api.github.com"
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
GITHUB_API_VERSION = "2022-11-28"
USER_AGENT = "xists-record-ingest"
README_CANDIDATES = (
    "README.md", "README.markdown", "README.rst", "README.txt", "README",
    "readme.md", "readme.markdown", "readme.rst", "readme.txt", "readme",
    "Readme.md", "Readme.markdown", "Readme",
)
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}

GRAPHQL_REPOSITORY_FRAGMENT = """
fragment RepoSnapshotFields on Repository {
  nameWithOwner
  name
  url
  description
  stargazerCount
  forkCount
  primaryLanguage { name }
  licenseInfo { spdxId }
  isArchived
  isDisabled
  homepageUrl
  createdAt
  updatedAt
  pushedAt
  issues(states: OPEN) { totalCount }
  pullRequests(states: OPEN) { totalCount }
  defaultBranchRef { name }
  repositoryTopics(first: 100) { nodes { topic { name } } }
  readmeMd: object(expression: "HEAD:README.md") { ... on Blob { text } }
  readmeMarkdown: object(expression: "HEAD:README.markdown") { ... on Blob { text } }
  readmeRst: object(expression: "HEAD:README.rst") { ... on Blob { text } }
  readmeTxt: object(expression: "HEAD:README.txt") { ... on Blob { text } }
  readmePlain: object(expression: "HEAD:README") { ... on Blob { text } }
  readmemd: object(expression: "HEAD:readme.md") { ... on Blob { text } }
  readmeMarkdownLower: object(expression: "HEAD:readme.markdown") { ... on Blob { text } }
  readmeRstLower: object(expression: "HEAD:readme.rst") { ... on Blob { text } }
  readmeTxtLower: object(expression: "HEAD:readme.txt") { ... on Blob { text } }
  readmePlainLower: object(expression: "HEAD:readme") { ... on Blob { text } }
  readmeMdMixed: object(expression: "HEAD:Readme.md") { ... on Blob { text } }
  readmeMarkdownMixed: object(expression: "HEAD:Readme.markdown") { ... on Blob { text } }
  readmeMixed: object(expression: "HEAD:Readme") { ... on Blob { text } }
  tree: object(expression: "HEAD:") {
    ... on Tree {
      entries {
        name
        type
        ... on TreeEntry {
          object {
            ... on Tree {
              entries {
                name
                type
                ... on TreeEntry {
                  object {
                    ... on Tree {
                      entries {
                        name
                        type
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

GRAPHQL_REPO_SNAPSHOT_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    ...RepoSnapshotFields
  }
  rateLimit { cost remaining limit resetAt }
}
""" + GRAPHQL_REPOSITORY_FRAGMENT


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


class TokenPool:
    """Round-robin token pool for distributing GitHub API requests across multiple tokens."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._index = 0
        self._lock = threading.Lock()

    def next_token(self) -> str | None:
        """Return the next token in round-robin order, or None if the pool is empty."""
        if not self._tokens:
            return None
        with self._lock:
            token = self._tokens[self._index % len(self._tokens)]
            self._index += 1
            return token


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
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                payload = json.loads(error.read().decode("utf-8"))
                message = payload.get("message") or str(error)
            except Exception:
                message = str(error)
            last_error = GitHubAPIError(message, status=error.code)
            if error.code not in RETRYABLE_HTTP_STATUSES or attempt == 2:
                raise last_error from error
            time.sleep(2**attempt)
        except urllib.error.URLError as error:
            last_error = GitHubAPIError(f"{error}", status=None)
            if attempt == 2:
                raise last_error from error
            time.sleep(2**attempt)
    raise last_error or GitHubAPIError("GitHub request failed")


def request_graphql(query: str, variables: dict[str, Any], token: str | None = None) -> dict[str, Any]:
    if not token:
        raise GitHubAPIError("GitHub GraphQL API requires GITHUB_TOKEN")

    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = urllib.request.Request(
        GITHUB_GRAPHQL_URL,
        data=body,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            try:
                payload = json.loads(error.read().decode("utf-8"))
                message = payload.get("message") or str(error)
            except Exception:
                message = str(error)
            last_error = GitHubAPIError(message, status=error.code)
            if error.code not in RETRYABLE_HTTP_STATUSES or attempt == 2:
                raise last_error from error
            time.sleep(2**attempt)
        except urllib.error.URLError as error:
            last_error = GitHubAPIError(f"{error}", status=None)
            if attempt == 2:
                raise last_error from error
            time.sleep(2**attempt)
    else:
        raise last_error or GitHubAPIError("GitHub GraphQL request failed")

    errors = payload.get("errors") or []
    if errors:
        message = "; ".join(err.get("message", "GraphQL error") for err in errors)
        raise GitHubAPIError(message)
    return payload


def _snapshot_from_graphql_repository(requested: str, owner: str, repository: dict[str, Any]) -> GitHubSnapshot:
    repo_url = repository.get("url")
    topics = [
        node["topic"]["name"]
        for node in (repository.get("repositoryTopics") or {}).get("nodes", [])
        if node.get("topic", {}).get("name")
    ]
    open_issues = (repository.get("issues") or {}).get("totalCount", 0)
    open_prs = (repository.get("pullRequests") or {}).get("totalCount", 0)
    metadata = {
        "full_name": repository.get("nameWithOwner"),
        "owner": {"login": owner},
        "name": repository.get("name"),
        "html_url": repo_url,
        "description": repository.get("description"),
        "topics": topics,
        "stargazers_count": repository.get("stargazerCount"),
        "forks_count": repository.get("forkCount"),
        "language": (repository.get("primaryLanguage") or {}).get("name"),
        "license": {"spdx_id": (repository.get("licenseInfo") or {}).get("spdxId")},
        "archived": repository.get("isArchived"),
        "disabled": repository.get("isDisabled"),
        "homepage": repository.get("homepageUrl"),
        "default_branch": (repository.get("defaultBranchRef") or {}).get("name"),
        "created_at": repository.get("createdAt"),
        "updated_at": repository.get("updatedAt"),
        "pushed_at": repository.get("pushedAt"),
        "open_issues_count": open_issues + open_prs,
    }

    readme = None
    readme_text = None
    readme_keys = (
        "readmeMd", "readmeMarkdown", "readmeRst", "readmeTxt", "readmePlain",
        "readmemd", "readmeMarkdownLower", "readmeRstLower", "readmeTxtLower", "readmePlainLower",
        "readmeMdMixed", "readmeMarkdownMixed", "readmeMixed",
    )
    for key, path in zip(readme_keys, README_CANDIDATES):
        blob = repository.get(key)
        if blob and blob.get("text"):
            readme_text = blob["text"]
            readme = {
                "path": path,
                "html_url": f"{repo_url}/blob/HEAD/{path}" if repo_url else None,
                "download_url": f"{repo_url}/raw/HEAD/{path}" if repo_url else None,
            }
            break

    # Flatten the 3-level nested tree into a flat path list.
    all_paths: list[dict[str, Any]] = []
    root = repository.get("tree") or {}
    for e1 in root.get("entries") or []:
        n1 = e1["name"]
        all_paths.append({"path": n1, "type": e1.get("type", "blob")})
        for e2 in (e1.get("object") or {}).get("entries") or []:
            n2 = f"{n1}/{e2['name']}"
            all_paths.append({"path": n2, "type": e2.get("type", "blob")})
            for e3 in (e2.get("object") or {}).get("entries") or []:
                all_paths.append({"path": f"{n2}/{e3['name']}", "type": e3.get("type", "blob")})

    tree = {"truncated": False, "tree": all_paths}

    return GitHubSnapshot(
        requested_repo_id=requested,
        metadata=metadata,
        readme=readme,
        readme_text=readme_text,
        tree=tree,
    )


def fetch_snapshot_graphql(repo_id: str, token: str | None = None) -> GitHubSnapshot:
    requested = parse_github_repo(repo_id)
    owner, name = requested.split("/", 1)
    payload = request_graphql(GRAPHQL_REPO_SNAPSHOT_QUERY, {"owner": owner, "name": name}, token=token)
    repository = (payload.get("data") or {}).get("repository")
    if not repository:
        raise GitHubAPIError(f"GitHub repository not found: {requested}", status=404)
    return _snapshot_from_graphql_repository(requested, owner, repository)


def build_graphql_batch_query(repo_ids: list[str]) -> tuple[str, dict[str, Any], dict[str, tuple[str, str]]]:
    if not repo_ids:
        raise ValueError("repo_ids must not be empty")

    variables: dict[str, Any] = {}
    aliases: dict[str, tuple[str, str]] = {}
    variable_defs: list[str] = []
    repository_fields: list[str] = []
    for index, repo_id in enumerate(repo_ids):
        requested = parse_github_repo(repo_id)
        owner, name = requested.split("/", 1)
        owner_var = f"owner{index}"
        name_var = f"name{index}"
        alias = f"r{index}"
        variables[owner_var] = owner
        variables[name_var] = name
        aliases[alias] = (requested, owner)
        variable_defs.extend([f"${owner_var}: String!", f"${name_var}: String!"])
        repository_fields.append(
            f"{alias}: repository(owner: ${owner_var}, name: ${name_var}) {{ ...RepoSnapshotFields }}"
        )

    query = (
        f"query({', '.join(variable_defs)}) {{\n"
        + "\n".join(repository_fields)
        + "\nrateLimit { cost remaining limit resetAt }\n}\n"
        + GRAPHQL_REPOSITORY_FRAGMENT
    )
    return query, variables, aliases


def fetch_snapshots_graphql(repo_ids: list[str], token: str | None = None) -> list[GitHubSnapshot]:
    query, variables, aliases = build_graphql_batch_query(repo_ids)
    payload = request_graphql(query, variables, token=token)
    data = payload.get("data") or {}
    snapshots: list[GitHubSnapshot] = []
    for alias, (requested, owner) in aliases.items():
        repository = data.get(alias)
        if not repository:
            raise GitHubAPIError(f"GitHub repository not found: {requested}", status=404)
        snapshots.append(_snapshot_from_graphql_repository(requested, owner, repository))
    return snapshots


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
    except GitHubAPIError:
        # README access is optional. When GitHub intermittently fails on the
        # README endpoint, keep the rest of the repository snapshot so ingest
        # can continue.
        readme = None
        readme_text = None

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
        if "img.shields.io" in line or "badge.svg" in line:
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
        "schema_version": RECORD_SCHEMA_VERSION,
        "xists_version": __version__,
        "github_api_version": GITHUB_API_VERSION,
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


def collect_record_graphql(repo_id: str, token: str | None = None) -> dict[str, Any]:
    record = build_record(fetch_snapshot_graphql(repo_id, token=token))
    record["snapshot_source"] = "github_graphql"
    return record


def collect_records_graphql(repo_ids: list[str], token: str | None = None) -> list[dict[str, Any]]:
    records = [build_record(snapshot) for snapshot in fetch_snapshots_graphql(repo_ids, token=token)]
    for record in records:
        record["snapshot_source"] = "github_graphql"
    return records


def github_token_from_file(path: Path) -> list[str]:
    """Read tokens from a file. One token per line. Returns list of non-empty tokens."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    tokens = [line.strip() for line in lines if line.strip()]
    return tokens


def github_token_from_env() -> list[str]:
    """Resolve tokens from environment.

    Priority: GITHUB_TOKENS (comma-separated) > GITHUB_TOKEN (single).
    """
    tokens_raw = os.environ.get("GITHUB_TOKENS", "").strip()
    if tokens_raw:
        return [t.strip() for t in tokens_raw.split(",") if t.strip()]
    single = os.environ.get("GITHUB_TOKEN", "").strip()
    return [single] if single else []
