"""Resolve the local files used by the xists command-line workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


LEGACY_FILENAMES = (
    "repos.txt",
    "records.json",
    "index.json",
    "eval-cases.json",
)


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
