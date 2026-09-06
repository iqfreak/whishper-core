"""ConsoleDisplay must never crash on non-Latin transcripts, and must render
numbered partials/finals per the shared Segment{id} contract.

Windows consoles default to cp1252, which cannot encode Arabic/CJK text.
Regression: UnicodeEncodeError('charmap') in show_partial/show fired the
moment the transcript was Arabic. ConsoleDisplay reconfigures stdout to
UTF-8 + errors="replace" at construction, so all of these survive.
"""
import io
import sys

from voicelang_core.adapters.console import ConsoleDisplay
from voicelang_core.types import Segment

ARABIC = "مرحبا بالعالم — اختبار"


def _cp1252_stdout(monkeypatch):
    raw = io.BytesIO()
    wrapped = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", wrapped)
    return raw, wrapped


def test_partial_arabic_survives_cp1252(monkeypatch):
    raw, wrapped = _cp1252_stdout(monkeypatch)
    ConsoleDisplay().show_partial(Segment(status="partial", source_text=ARABIC))
    wrapped.flush()
    assert ARABIC.encode("utf-8") in raw.getvalue()


def test_final_arabic_survives_cp1252(monkeypatch):
    raw, wrapped = _cp1252_stdout(monkeypatch)
    ConsoleDisplay().show(Segment(id=1, status="final", source_text=ARABIC,
                                  translated_text=ARABIC))
    wrapped.flush()
    assert ARABIC.encode("utf-8") in raw.getvalue()


def test_partial_writes_carriage_return_with_shared_id(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    ConsoleDisplay().show_partial(
        Segment(id=5, status="partial", source_text="hello")
    )
    out = buf.getvalue()
    assert out.startswith("\r[5] hello")
    assert "\n" not in out  # in-place rewrite, never a new line


def test_final_writes_numbered_line_after_partial(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    d = ConsoleDisplay()
    d.show_partial(Segment(id=7, status="partial", source_text="hola"))
    d.show(Segment(id=7, status="final", source_text="hola",
                   translated_text="salut"))
    out = buf.getvalue()
    # Same id in partial and final: partial rewrites in place, final lands on
    # its own numbered line.
    assert "\r[7] hola" in out
    assert "\n[7] salut\n" in out


def test_works_under_plain_stringio(monkeypatch):
    # Streams without reconfigure (StringIO) must not blow up in __init__.
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    ConsoleDisplay().show_partial(
        Segment(status="partial", source_text="plain ascii")
    )
    assert "plain ascii" in buf.getvalue()


def test_works_with_none_stdout(monkeypatch):
    # Frozen --windowed child where redirection somehow did not run.
    monkeypatch.setattr(sys, "stdout", None)
    ConsoleDisplay().show_partial(Segment(status="partial", source_text="still fine"))