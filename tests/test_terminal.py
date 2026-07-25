from __future__ import annotations

import pytest

from xists.terminal import style


class _Stream:
    def __init__(self, isatty: bool) -> None:
        self._isatty = isatty

    def isatty(self) -> bool:
        return self._isatty


@pytest.mark.parametrize("role", ("title", "body", "link", "success", "warning", "error", "muted"))
def test_style_never_emits_ansi_sequences(role):
    rendered = style("xists", role, stream=_Stream(True))

    assert rendered == "xists"
    assert "\x1b[" not in rendered
