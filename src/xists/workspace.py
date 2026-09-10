"""Resolve the local files used by the xists command-line workflow."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEGACY_FILENAMES = (
    "repos.txt",
    "records.json",
    "index.json",
    "eval-cases.json",
)

ENV_TEMPLATE = """# xists local configuration
# Shell environment variables override this file. A .env in the current
# directory overrides this workspace file for that command.

# GITHUB_TOKEN=
# LLM_API_KEY=
# LLM_BASE_URL=
# LLM_MODEL=
# EMBEDDING_API_KEY=
# EMBEDDING_BASE_URL=
# EMBEDDING_MODEL=
"""


@dataclass(frozen=True)
class WorkspacePaths:
    """The default file locations for one xists workspace."""

    root: Path
    mode: str

    @property
    def env_file(self) -> Path:
        return self.root / ".env"

    @property
    def repos(self) -> Path:
        return self.root / "repos.txt"

    @property
    def records(self) -> Path:
        return self.root / "records.json"

    @property
    def ingest_report(self) -> Path:
        return self.root / "report.json"

    @property
    def index(self) -> Path:
        return self.root / "index.json"

    @property
    def refreshed_records(self) -> Path:
        return self.root / "records-v2.json"

    @property
    def eval_cases(self) -> Path:
        return self.root / "eval-cases.json"

    @property
    def eval_report(self) -> Path:
        return self.root / "eval-report.json"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def embedding_cache_db(self) -> Path:
        return self.cache_dir / "embeddings.db"


def workspace_root(environ: Mapping[str, str] | None = None) -> Path:
    """Return the configured workspace root without creating it."""

    environment = os.environ if environ is None else environ
    configured = environment.get("XISTS_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".xists").resolve()


def resolve_workspace(
    *,
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> WorkspacePaths:
    """Choose either an existing legacy directory or the default workspace."""

    current_directory = (Path.cwd() if cwd is None else cwd).expanduser().resolve()
    if any((current_directory / filename).exists() for filename in LEGACY_FILENAMES):
        return WorkspacePaths(root=current_directory, mode="legacy")
    return WorkspacePaths(root=workspace_root(environ), mode="workspace")


def initialize_workspace(root: Path) -> tuple[bool, bool]:
    """Create an empty workspace and configuration template without overwriting data."""

    root = root.expanduser().resolve()
    created_root = not root.exists()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if created_root and os.name == "posix":
        root.chmod(0o700)

    env_file = root / ".env"
    created_env_file = not env_file.exists()
    if created_env_file:
        env_file.write_text(ENV_TEMPLATE, encoding="utf-8")
        if os.name == "posix":
            env_file.chmod(0o600)

    return created_root, created_env_file


def populate_demo_workspace(root: Path, *, force: bool = False) -> dict[str, Any]:
    """Populate workspace with bundled starter demo index and records."""
    from xists.search.pull import pull_index

    root = root.expanduser().resolve()
    return pull_index("demo", root, force=force)


def get_default_embedding_cache(
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Resolve and instantiate the default QueryEmbeddingCache for the environment."""
    from xists.search.cache import QueryEmbeddingCache

    env = os.environ if environ is None else environ
    if env.get("XISTS_DISABLE_EMBED_CACHE", "").lower() in {"1", "true", "yes"}:
        return QueryEmbeddingCache(None, enabled=False)

    custom_db = env.get("XISTS_EMBEDDING_CACHE_DB", "").strip()
    if custom_db:
        return QueryEmbeddingCache(Path(custom_db))

    paths = resolve_workspace(environ=env)
    return QueryEmbeddingCache(paths.embedding_cache_db)
