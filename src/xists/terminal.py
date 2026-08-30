"""Small, dependency-free helpers for readable terminal output."""

from __future__ import annotations

from typing import Literal

TerminalRole = Literal["title", "body", "link", "success", "warning", "error", "muted"]


def style(text: str, role: TerminalRole, *, stream: object) -> str:
    """Preserve terminal text without emitting ANSI control sequences."""

    return text
