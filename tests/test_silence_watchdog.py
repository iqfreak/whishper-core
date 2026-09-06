"""No-audio watchdog: silent capture must warn visibly, not fail silently.

Regression driver: a mute mic / blocked mic / nothing playing on loopback
used to produce literally nothing — which users read as "the mod is broken".
The pipeline watches chunk energy and warns via display.warn().
"""
import io
import sys

from voicelang_core import Pipeline
from voicelang_core.adapters.fake import (
    FakeAudioSource, FakeDisplay, FakeStreamingTranscriber, FakeTranslator,
)
from voicelang_core.adapters.console import ConsoleDisplay
from voicelang_core.adapters.overlay import OverlayDisplay
from voicelang_core.types import AudioChunk


class _WarnCapture(FakeDisplay):
    def __init__(self):
        super().__init__()
        self.warns = []

    def warn(self, text: str) -> None:
        self.warns.append(text)


def _silent_chunks(n):
    # zero-energy 0.25 s chunks
    return [AudioChunk(pcm=b"\x00\x00" * 4000, sample_rate=16000) for _ in range(n)]


def test_silent_mic_warns_with_privacy_hint():
    class _NamedMic(FakeAudioSource):
        """Same behavior, class named like the real MicSource."""

    _NamedMic.__name__ = "MicSource"

    disp = _WarnCapture()
    p = Pipeline(
        source=_NamedMic(_silent_chunks(8)),
        transcriber=FakeStreamingTranscriber(), translator=FakeTranslator(),
        display=disp, target_language="en",
    )
    p._silence_warn_seconds = 0.01
    p._silence_warn_chunks = 3
    p._silence_cooldown = 0.01
    p.run_streaming()
    assert len(disp.warns) == 1  # warned once, not per chunk
    assert "MicSource" in disp.warns[0]
    assert "Privacy" in disp.warns[0]  # actionable Windows hint


def test_loud_chunk_resets_silence_timer():
    disp = _WarnCapture()
    chunks = _silent_chunks(6)
    # a loud chunk in the middle resets the silent streak
    chunks[3] = AudioChunk(pcm=b"\xff\x7f" * 4000, sample_rate=16000)
    p = Pipeline(
        source=FakeAudioSource(chunks),
        transcriber=FakeStreamingTranscriber(), translator=FakeTranslator(),
        display=disp, target_language="en",
    )
    p._silence_warn_seconds = 0.01
    p._silence_warn_chunks = 3
    p._silence_cooldown = 0.01
    p.run_streaming()
    # chunks 1-3 silent (warns once at 3), 4 loud (reset), 5-6 silent —
    # the second streak is too short to warn again
    assert len(disp.warns) == 1


def test_console_warn_renders():
    import pytest
    from pytest import MonkeyPatch

    mp = MonkeyPatch()
    buf = io.StringIO()
    mp.setattr(sys, "stdout", buf)
    try:
        ConsoleDisplay().warn("mic silent")
    finally:
        mp.undo()
    assert "⚠ mic silent" in buf.getvalue()


def test_overlay_headless_warn_surfaces_in_lines():
    disp = OverlayDisplay(headless=True)
    disp.warn("no audio detected")
    assert any("⚠" in l for l in disp.lines)