"""Small, dependency-free helpers for readable terminal output."""

from __future__ import annotations

import os
from typing import Literal, Protocol


class _TerminalStream(Protocol):
    def isatty(self) -> bool: ...


TerminalRole = Literal["title", "body", "link", "success", "warning", "error", "muted"]

_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_WARM_WHITE = "\x1b[38;2;243;240;232m"
_GRAY_BLUE = "\x1b[38;2;115;144;183m"


def color_enabled(stream: _TerminalStream) -> bool:
    """Return whether it is appropriate to emit ANSI color for *stream*."""

    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    try:
        return stream.isatty()
    except (AttributeError, OSError):
        return False


def style(text: str, role: TerminalRole, *, stream: _TerminalStream) -> str:
    """Apply a restrained brand role, or return plain text when color is disabled."""

    if not color_enabled(stream) or not text:
        return text

    prefix = {
        "title": _BOLD + _WARM_WHITE,
        "body": _WARM_WHITE,
        "link": _GRAY_BLUE,
        "success": _GRAY_BLUE,
        "warning": _BOLD + _WARM_WHITE,
        "error": _BOLD + _WARM_WHITE,
        "muted": _DIM + _WARM_WHITE,
    }[role]
    return f"{prefix}{text}{_RESET}"
