"""Microphone capture source (sounddevice). Streams 16kHz mono PCM chunks."""
from .base import AudioSource  # noqa: F401
from ..types import AudioChunk

import numpy as np


def list_input_devices() -> list[dict]:
    """Enumerate capture (input) devices for the mic picker.

    Returns [{"index", "name", "rate", "channels"}] — empty when sounddevice
    is missing. Mirrors list_loopback_devices() in wasapi_loopback.py.
    """
    try:
        import sounddevice as sd  # noqa: PLC0415
    except ImportError:
        return []
    try:
        raw = sd.query_devices(device=None, kind="input")
    except Exception:  # noqa: BLE001 -- device enumeration is best-effort
        return []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    return [{"index": int(d["index"]), "name": str(d["name"]),
             "rate": int(d.get("default_samplerate") or 16000),
             "channels": int(d.get("max_input_channels") or 1)}
            for d in items if isinstance(d, dict)]


class MicSource(AudioSource):
    def __init__(self, block_seconds: float = 0.02, sample_rate: int = 16000,
                 device: int | str | None = None):
        self._block = block_seconds
        self._rate = sample_rate
        self._device = device

    def stream(self):
        # Lazy: sounddevice is a 'real' extra; import here so that importing
        # this module (and run.py) never hard-requires the mic backend.
        import sounddevice as sd  # noqa: PLC0415

        blocksize = int(self._block * self._rate)
        with sd.RawInputStream(
            samplerate=self._rate, channels=1, dtype="int16",
            blocksize=blocksize, device=self._device,
        ) as stream:
            while True:
                pcm, _ = stream.read(blocksize)
                # RawInputStream.read() returns a _cffi_backend.buffer (no
                # .tobytes()). bytes() works for cffi buffers, numpy and
                # memoryview alike.
                yield AudioChunk(pcm=bytes(pcm), sample_rate=self._rate)
