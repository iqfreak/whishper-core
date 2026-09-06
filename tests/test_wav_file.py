"""WavFileSource: tolerant decode + block replay (no external deps)."""
import math
import wave

import numpy as np

from voicelang_core.adapters.wav_file import WavFileSource, load_wav
from voicelang_core.types import AudioChunk


def _write_wav(path, rate, width, chans, frames: bytes):
    with wave.open(path, "wb") as w:
        w.setnchannels(chans)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(frames)


def _sine(rate, seconds, amp=0.5):
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)


def test_load_wav_16bit_mono_16k(tmp_path):
    audio = _sine(16000, 1.0)
    pcm = (audio * 32767).astype(np.int16).tobytes()
    p = tmp_path / "a.wav"
    _write_wav(str(p), 16000, 2, 1, pcm)
    out = load_wav(str(p))
    assert out.dtype == np.float32
    assert len(out) == 16000
    assert np.abs(out).max() <= 1.0


def test_load_wav_8bit_stereo_44k_resampled(tmp_path):
    # Music sample format: 8-bit unsigned, stereo, 44100 -- the SAPI default.
    rate, seconds = 44100, 0.5
    mono = _sine(rate, seconds)
    stereo = np.stack([mono, mono], axis=1)
    raw = ((stereo * 127.0) + 128.0).astype(np.uint8).tobytes()
    p = tmp_path / "b.wav"
    _write_wav(str(p), rate, 1, 2, raw)
    out = load_wav(str(p))
    assert out.dtype == np.float32
    assert len(out) == 16000 * seconds  # resampled to 16 kHz
    assert np.abs(out).max() <= 1.0


def test_wav_file_source_chunking(tmp_path):
    audio = _sine(16000, 2.0)
    pcm = (audio * 32767).astype(np.int16).tobytes()
    p = tmp_path / "c.wav"
    _write_wav(str(p), 16000, 2, 1, pcm)
    src = WavFileSource(str(p), block_seconds=0.5)
    assert src.duration_seconds == 2.0
    chunks = list(src.stream())
    assert len(chunks) == 4  # 2.0s / 0.5s blocks
    for c in chunks:
        assert isinstance(c, AudioChunk)
        assert c.sample_rate == 16000
        assert len(c.pcm) == 8000 * 2  # 0.5s of int16 mono