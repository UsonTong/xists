from __future__ import annotations

from xists.terminal import color_enabled, style


class _Stream:
    def __init__(self, isatty: bool) -> None:
        self._isatty = isatty

    def isatty(self) -> bool:
        return self._isatty


def test_style_uses_brand_colors_for_interactive_terminal(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)

    rendered = style("xists", "link", stream=_Stream(True))

    assert rendered == "\x1b[38;2;115;144;183mxists\x1b[0m"


def test_style_is_plain_when_output_is_not_a_terminal(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)

    assert color_enabled(_Stream(False)) is False
    assert style("xists", "title", stream=_Stream(False)) == "xists"


def test_style_respects_no_color_for_interactive_terminal(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    assert color_enabled(_Stream(True)) is False
    assert style("xists", "success", stream=_Stream(True)) == "xists"


def test_style_disables_color_for_dumb_terminal(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")

    assert style("xists", "error", stream=_Stream(True)) == "xists"
