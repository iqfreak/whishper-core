"""Domain types shared across the pipeline. Stdlib + numpy only.

numpy is allowed here deliberately (README: "Core deps: numpy only in
pipeline/types/ports") so the shared PCM->float32 conversion lives in one
place instead of 5 adapters with dtype drift.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional

import time
import numpy as np

def pcm_bytes_to_float32(pcm: bytes) -> "np.ndarray":
    """Raw 16-bit signed little-endian PCM bytes -> float32 in [-1, 1].

    Canonical conversion for the 16 kHz mono pipeline path. Every adapter
    feeds its model float32; this keeps the 5 copies identical (the old
    duplication drifted on dtype spelling: ``<i2`` vs ``int16``).
    """
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0

def pcm_energy(pcm: bytes) -> float:
    """Root-mean-square of an int16 PCM buffer (0..32768).

    Used by the pipeline's no-audio watchdog: near-zero energy for a few
    seconds means the capture device is silent (muted / wrong device /
    Windows mic privacy), which deserves a visible warning — silent input
    otherwise reads as "the mod is broken".
    """
    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a * a)))

@dataclass
class AudioChunk:
    """One buffer of captured audio.

    Carried as raw 16-bit signed little-endian PCM so the port has no numpy
    dependency; adapters convert to float32 internally.
    P0: capture_ts records time.monotonic() when chunk was captured for
    end-to-end latency measurement (capture -> VAD -> ASR -> display).
    """
    pcm: bytes
    sample_rate: int = 16000
    channels: int = 1
    capture_ts: float = field(default_factory=time.monotonic)

@dataclass
class Segment:
    """One transcript unit, numbered per the shared monotonic id contract.

    The `id` is OWNED by the pipeline worker: adapters yield `id=0` as a
    sentinel and the pipeline stamps the real id before the segment reaches
    the display. It increments ONLY on `status="final"`; partial updates
    reuse the same id (mutated in place), so the transcription and
    translation blocks can never drift out of sync. The translator fills
    `translated_text` on the SAME id.
    """
    id: int = 0
    status: Literal["partial", "final"] = "final"
    source_text: str = ""
    translated_text: Optional[str] = None
    timestamp: float = 0.0
    source_language: Optional[str] = None

@dataclass
class Translation:
    source_text: str
    target_text: str
    source_language: str
    target_language: str
