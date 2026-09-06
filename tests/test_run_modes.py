"""Integration test: actually drive the pipeline run modes end-to-end.

This is the regression guard the skill calls out as most valuable for
multi-engine code. It exercises partial/final separation, the Display contract,
and run-mode dispatch with REAL port implementations (fakes) -- not just
construction. A green suite that only *builds* pipelines never catches a missing
``show_partial`` or a partial leaking into the translator.

These tests need no heavy deps: the fakes are stdlib-only, so they run in the
numpy-only suite.
"""
import os

from voicelang_core.pipeline import Pipeline
from voicelang_core.types import AudioChunk, Segment
from voicelang_core.adapters.fake import (
    FakeCaptionSource, FakeStreamingTranscriber, FakeTranscriber,
    FakeAudioSource, FakeTranslator, FakeDisplay,
)


def _chunks(n: int = 6) -> list[AudioChunk]:
    return [AudioChunk(pcm=b"\x00" * 16000, sample_rate=16000) for _ in range(n)]


def test_run_captions_partials_and_finals():
    disp = FakeDisplay()
    trans = FakeTranslator()
    p = Pipeline(source=FakeCaptionSource(rounds=2), translator=trans,
                 display=disp, target_language="es")
    p.run_captions()
    # 2 rounds -> 2 finals (translated + shown) and 2 partials (preview only).
    assert len(disp.shown) == 2
    assert len(disp.partials) == 2
    # No partial leaks into the translator or the final display.
    assert all("partial" not in t.source_text for t in disp.shown)
    assert len(trans.calls) == 2  # translate called once per final, not per partial
    # Shared monotonic ids: finals numbered 1, 2; partials reused them.
    assert [s.id for s in disp.shown] == [1, 2]
    assert [s.id for s in disp.partials] == [1, 2]


def test_run_streaming_partials_and_finals():
    disp = FakeDisplay()
    p = Pipeline(source=FakeAudioSource(chunks=_chunks(6)),
                 transcriber=FakeStreamingTranscriber(), translator=FakeTranslator(),
                 display=disp, target_language="es")
    p.run_streaming()
    # finals on every 3rd chunk (3, 6) -> 2; one partial per chunk -> 6.
    assert len(disp.shown) == 2
    assert len(disp.partials) == 6
    assert all("partial" not in t.source_text for t in disp.shown)
    # A partial never allocates: first partial already carries id 1.
    assert disp.partials[0].id == 1


def test_run_batch_finalizes():
    disp = FakeDisplay()
    p = Pipeline(source=FakeAudioSource(chunks=_chunks(6)),
                 transcriber=FakeTranscriber(), translator=FakeTranslator(),
                 display=disp, target_language="es")
    p.run()
    assert len(disp.shown) == 2  # finalize every 3rd chunk
    assert [s.id for s in disp.shown] == [1, 2]


def test_overlay_display_satisfies_contract_headless():
    from voicelang_core.adapters.overlay import OverlayDisplay

    d = OverlayDisplay(headless=True)
    d.show(Segment(id=1, status="final", source_text="hi", translated_text="hola"))
    d.show_partial(Segment(status="partial", source_text="hola (partial)"))
    assert d.lines  # buffers without Qt, no crash
    # Same id visible in both blocks.
    assert "1. hi" in d.transcribed_lines[-1]
    assert "1. hola" in d.translated_lines[-1]


def test_source_wav_requires_input():
    import pytest

    from voicelang_core.run import _build_source

    with pytest.raises(SystemExit):
        _build_source("wav", None)


def test_source_builders_construct():
    from voicelang_core.run import _build_source

    assert _build_source("mic", None) is not None
    loop = _build_source("loopback", None)
    assert loop is not None
    assert loop._rate == 16000  # noqa: SLF001 -- lazy construct, cheap check
    named = _build_source("loopback", None, "Headphone (Realtek(R) Audio) [Loopback]")
    assert named._device == "Headphone (Realtek(R) Audio) [Loopback]"  # noqa: SLF001


def test_mic_accepts_source_device_selection():
    """--source-device must select the MIC too (name or index)."""
    from voicelang_core.run import _build_source
    from voicelang_core.adapters.mic import MicSource

    m = _build_source("mic", None, source_device="Microphone (Realtek HD Audio Mic input)")
    assert isinstance(m, MicSource)
    assert m._device == "Microphone (Realtek HD Audio Mic input)"  # noqa: SLF001
    mi = _build_source("mic", None, source_device="28")
    assert mi._device == "28"


def test_list_input_devices_shape():
    """Mic picker data shape (empty list degrades gracefully without sounddevice)."""
    from voicelang_core.adapters.mic import list_input_devices

    devs = list_input_devices()
    assert isinstance(devs, list)
    for d in devs:
        assert "index" in d and "name" in d and "rate" in d


def test_loopback_decimation_factors():
    import numpy as np

    from voicelang_core.adapters.wasapi_loopback import _decimate_to_16k

    # 48k -> 16k is an exact 3:1 mean decimation.
    a = np.arange(48_000, dtype=np.int16)
    out = _decimate_to_16k(a, 48_000)
    assert len(out) == 16_000
    # 44.1k -> 16k uses interpolation; length scales accordingly.
    b = np.arange(44_100, dtype=np.int16)
    out2 = _decimate_to_16k(b, 44_100)
    assert abs(len(out2) - 16_000) <= 1
    # Identity fast-path.
    assert _decimate_to_16k(a, 16_000) is a


def test_finish_tail_flushes_final_segment():
    """WAV replay tail: the transcriber's finish() flush must reach the display
    with a monotonic id (regression: the old Translation path dropped it)."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from voicelang_core.ports import StreamingTranscriber
    from voicelang_core.run import _finish_tail
    from voicelang_core.types import Segment as Seg

    class _TailST(StreamingTranscriber):
        def transcribe_stream(self, chunk):
            return iter(())

        def transcribe(self, chunk):
            return None

        def finish(self):
            yield Seg(status="final", source_text="tail utterance",
                      source_language="en")

    disp = FakeDisplay()
    p = Pipeline(source=FakeAudioSource(chunks=[]), transcriber=_TailST(),
                 translator=FakeTranslator(), display=disp, target_language="en")
    p.run_streaming()
    _finish_tail(p)
    assert len(disp.shown) == 1
    assert disp.shown[0].id == 1          # pipeline stamped it, monotonic
    assert disp.shown[0].source_text == "tail utterance"
    assert disp.shown[0].translated_text == "[en] tail utterance"  # translated


def test_app_wants_cli_routing():
    """Packaged exe opens the GUI by default; any headless flag routes to CLI."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from voicelang_app import _wants_cli

    assert _wants_cli(["voicelang.exe"]) is False                    # GUI default
    assert _wants_cli(["voicelang.exe", "--run", "--config", "x"]) is True
    assert _wants_cli(["voicelang.exe", "--headless", "--source", "mic"]) is True
    assert _wants_cli(["voicelang.exe", "--list-engines"]) is True
    assert _wants_cli(["voicelang.exe", "--display=overlay"]) is True
    assert _wants_cli(["voicelang.exe", "--gui"]) is False           # GUI explicit


def test_cli_list_engines_flag():
    """--list-engines exits 0 and enumerates the engine catalog (spec §8)."""
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-m", "voicelang_core.run", "--list-engines"],
        capture_output=True, text=True, timeout=120, cwd=root,
    )
    assert r.returncode == 0, r.stderr
    assert "moonshine" in r.stdout
    assert "libretranslate" in r.stdout