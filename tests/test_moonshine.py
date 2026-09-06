"""Moonshine Voice adapter: real tiny-streaming model + real TTS WAV.

The model downloads on first use into the moonshine cache, then runs offline.
Skip-if: package missing (extra not installed).
"""
import os

import pytest

from voicelang_core.adapters.moonshine import MoonshineTranscriber
from voicelang_core.adapters.wav_file import load_wav
from voicelang_core.types import AudioChunk

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV = os.path.join(_ROOT, "e2e_speech.wav")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(WAV),
    reason="e2e_speech.wav not generated",
)


def _moonshine_available() -> bool:
    try:
        import moonshine_voice  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _moonshine_available(),
                                reason="moonshine-voice extra not installed")


def test_moonshine_streams_partials_and_finals():
    t = MoonshineTranscriber(language="en", model_arch="TINY_STREAMING")
    audio = load_wav(WAV)
    step = int(0.25 * 16000)
    partials, finals = [], []
    for i in range(0, len(audio), step):
        c = AudioChunk(
            pcm=(audio[i : i + step] * 32767).astype("int16").tobytes(),
            sample_rate=16000,
        )
        for seg in t.transcribe_stream(c):
            (finals if seg.status == "final" else partials).append(seg.source_text)
    finals += [s.source_text for s in t.finish()]
    joined = " ".join(finals + partials).lower()
    assert any(w in joined for w in ("mission", "dawn", "extraction", "position")), joined
    assert finals  # line-final segments were emitted