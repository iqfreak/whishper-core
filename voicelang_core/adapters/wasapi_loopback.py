"""WASAPI loopback capture for the game-overlay surface.

Captures whatever an OUTPUT device is PLAYING -- e.g. teammates through
Discord/voice chat -- so you caption them without a mic. Which device to
capture is user-selectable (the Volume-Mixer view: each app routes through
one of the loopback-capable devices). This is the capture seam only;
rendering lives in adapters/overlay.py (OverlayDisplay).

Implementation: pyaudiowpatch (patched PortAudio with WASAPI loopback, the
only portable way to get the output mix). sounddevice's own PortAudio build
exposes NO loopback devices and WasapiSettings has no loopback flag, so
there is no sounddevice fallback -- just a clear error.

NOTE: device-level loopback captures ALL apps on that device, not a single
app. Per-APP isolation is NOT a public Windows API (session control exists
via pycaw; per-session audio capture does not).
"""
from typing import Iterator

import numpy as np

from .base import AudioSource  # noqa: F401
from ..types import AudioChunk


def _decimate_to_16k(pcm: np.ndarray, dev_rate: int, target: int = 16000) -> np.ndarray:
    """Downmix a mono int16 buffer to ``target`` Hz without extra deps.

    Exact integer factors (48k -> 16k) use mean-decimation: cheap, no
    aliasing filter needed for speech. Non-integer rates (44.1k) fall back
    to linear interpolation. Never up-samples.
    """
    if dev_rate == target:
        return pcm
    n_in = len(pcm)
    if dev_rate % target == 0:
        f = dev_rate // target
        n_out = n_in // f
        if n_out == 0:
            return np.zeros(0, dtype=np.int16)
        return pcm[: n_out * f].reshape(-1, f).mean(axis=1).astype(np.int16)
    out_n = int(n_in * target / dev_rate)
    xs = np.linspace(0.0, n_in, out_n, endpoint=False)
    return np.interp(xs, np.arange(n_in), pcm.astype(np.float32)).astype(np.int16)


def list_loopback_devices() -> list[dict]:
    """Enumerate WASAPI loopback-capable devices (Volume-Mixer device view).

    Returns [{"index", "name", "rate"}] for every device whose name marks it
    as a loopback endpoint. Empty when pyaudiowpatch is missing.
    """
    try:
        import pyaudiowpatch as pyaudio  # noqa: PLC0415 -- lazy, optional dep
    except ImportError:
        return []
    pa = pyaudio.PyAudio()
    try:
        out = []
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except OSError:
                continue
            if not info["maxInputChannels"]:
                continue
            name = str(info["name"])
            if "loopback" not in name.lower():
                continue
            out.append({"index": int(info["index"]),
                        "name": name,
                        "rate": int(info["defaultSampleRate"])})
        return out
    finally:
        pa.terminate()


class WASAPILoopbackSource(AudioSource):
    def __init__(self, device: int | str | None = None, block_seconds: float = 0.02,
                 sample_rate: int = 16000):
        self._device = device  # int index, or a device NAME containing "[loopback]"
        self._block = block_seconds
        self._rate = sample_rate

    def _resolve_info(self, pa):
        if self._device is None:
            info = pa.get_default_wasapi_loopback()
            if info is None:
                raise RuntimeError(
                    "no WASAPI loopback device found; pick one (GUI/app: "
                    "Capture source = Game/audio, then choose a device)")
            return info
        if isinstance(self._device, int):
            return pa.get_device_info_by_index(self._device)
        # Name-based lookup (survives index churn between reboots).
        needle = str(self._device).lower()
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except OSError:
                continue
            if needle in str(info["name"]).lower():
                return info
        raise RuntimeError(f"no loopback device matching name {self._device!r}")

    def stream(self) -> Iterator[AudioChunk]:
        try:
            import pyaudiowpatch as pyaudio  # noqa: PLC0415 -- lazy, optional dep
        except ImportError as exc:  # pragma: no cover -- env-specific
            raise RuntimeError(
                "WASAPI loopback needs pyaudiowpatch:  uv sync --extra real\n"
                "(sounddevice cannot loopback -- its PortAudio build exposes "
                "no loopback devices)") from exc

        pa = pyaudio.PyAudio()
        try:
            info = self._resolve_info(pa)
            dev_rate = int(info["defaultSampleRate"])
            blocksize = int(self._block * dev_rate)  # frames at DEVICE rate
            channels = min(2, int(info["maxInputChannels"]) or 2)
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=dev_rate,
                frames_per_buffer=blocksize,
                input=True,
                input_device_index=int(info["index"]),
            )
            try:
                while True:
                    raw = stream.read(blocksize, exception_on_overflow=False)
                    arr = np.frombuffer(raw, dtype=np.int16)
                    if channels > 1:
                        arr = arr.reshape(-1, channels).mean(axis=1)
                    mono = _decimate_to_16k(arr, dev_rate, self._rate)
                    yield AudioChunk(pcm=mono.astype(np.int16).tobytes(),
                                     sample_rate=self._rate)
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            pa.terminate()