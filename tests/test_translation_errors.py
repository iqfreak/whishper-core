"""Translation failure degrades gracefully (spec §9) — the worker never crashes.

A dead translation endpoint must keep the transcription block flowing, mark
the translation block with an inline error state, and let the next segment
proceed. Segment.translated_text stays None on failure.
"""
import io
import sys

from voicelang_core import Pipeline
from voicelang_core.adapters.fake import (
    FakeAudioSource, FakeDisplay, FakeStreamingTranscriber, FakeTranslator,
)
from voicelang_core.adapters.console import ConsoleDisplay
from voicelang_core.adapters.overlay import OverlayDisplay
from voicelang_core.types import AudioChunk, Segment


class _FailingTranslator(FakeTranslator):
    def translate(self, text, source, target):
        raise RuntimeError("translate endpoint down")


def _chunks(n=3):
    return [AudioChunk(pcm=b"\x00\x00" * 16000, sample_rate=16000) for _ in range(n)]


def test_pipeline_survives_translation_failure():
    disp = FakeDisplay()
    p = Pipeline(
        source=FakeAudioSource(_chunks(3)),
        transcriber=FakeStreamingTranscriber(final_text="bonjour", language="fr"),
        translator=_FailingTranslator(), display=disp, target_language="en",
    )
    p.run_streaming()  # must NOT raise
    assert len(disp.shown) == 1
    assert disp.shown[0].translated_text is None
    assert disp.shown[0].source_text == "bonjour"  # transcription keeps flowing


def test_overlay_marks_translation_block_error():
    disp = OverlayDisplay(headless=True)
    p = Pipeline(
        source=FakeAudioSource(_chunks(3)),
        transcriber=FakeStreamingTranscriber(final_text="bonjour", language="fr"),
        translator=_FailingTranslator(), display=disp, target_language="en",
    )
    p.run_streaming()
    rendered = "\n".join(disp.lines)
    assert "⚠" in rendered  # inline error marker on the translation block
    assert "bonjour" in rendered  # Block A (source) intact


def test_console_marks_translation_block_error(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    ConsoleDisplay().show(Segment(id=1, status="final", source_text="bonjour",
                                  translated_text=None))
    out = buf.getvalue()
    assert "⚠" in out
    assert "bonjour" in out