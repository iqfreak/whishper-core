"""WavFileSource: replay a WAV file as an AudioSource (no microphone needed).

Lets the shipped CLI (``python -m voicelang_core.run --input file.wav``) drive
the full capture -> transcribe -> translate -> display pipeline from a real
audio file: deterministic, reproducible, and the exact same code path the mic
uses downstream of the source.

Decoding is deliberately tolerant (8-bit unsigned PCM, 16-bit PCM, stereo ->
mono mean, arbitrary sample rate resampled to 16 kHz) so any WAV plays.
"""

from __future__ import annotations

import wave
from typing import Iterator

import numpy as np

from ..ports import AudioSource
from ..types import AudioChunk

RATE = 16000  # faster-whisper native rate


def load_wav(path: str) -> np.ndarray:
    """Decode any wave.Wave_read-openable WAV to mono float32 [-1, 1] at RATE."""
    with wave.open(path, "rb") as w:
        rate, width, chans = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width == 1:  # 8-bit unsigned PCM
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:  # 16-bit signed PCM
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if chans == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    if rate != RATE:
        try:
            import librosa  # exact-resample quality when the e2e extra is present

            audio = librosa.resample(audio, orig_sr=rate, target_sr=RATE)
        except ModuleNotFoundError:  # pragma: no cover - linear fallback
            n_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
            n_new = np.linspace(0.0, 1.0, int(len(audio) * RATE / rate), endpoint=False)
            audio = np.interp(n_new, n_old, audio).astype(np.float32)
    return audio


class WavFileSource(AudioSource):
    """Replays a WAV file in fixed-length blocks, one AudioChunk per block."""

    def __init__(self, path: str, block_seconds: float = 1.0):
        self._path = path
        self._audio = load_wav(path)
        self._rate = RATE
        self._step = int(RATE * block_seconds)

    @property
    def duration_seconds(self) -> float:
        return len(self._audio) / self._rate

    def stream(self) -> Iterator[AudioChunk]:
        for i in range(0, len(self._audio), self._step):
            block = self._audio[i : i + self._step]
            yield AudioChunk(
                pcm=(block * 32767).astype(np.int16).tobytes(),
                sample_rate=self._rate,
            )