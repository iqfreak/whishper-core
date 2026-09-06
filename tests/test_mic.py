"""MicSource must serialize sounddevice's raw capture buffer to PCM bytes.

Regression: sd.RawInputStream.read() returns a ``_cffi_backend.buffer`` for
raw streams (dtype="int16"), which has NO ``.tobytes()``. MicSource called
``pcm.tobytes()`` and died with AttributeError on the very first block.
``bytes()`` works across cffi buffers, numpy arrays and memoryviews alike.
The existing factory stubs returned numpy (which has .tobytes), so this
shape never reached the tests until it crashed the real app.
"""
import ctypes
import sys
import types

from voicelang_core.adapters.mic import MicSource


def test_mic_stream_converts_cffi_like_buffer(monkeypatch):
    rate, block = 16000, 1.0
    n = int(block * rate)  # samples per block
    payload = bytes(range(256)) * (n * 2 // 256)  # exactly n*2 bytes (int16 LE)
    assert len(payload) == n * 2

    fake = types.ModuleType("sounddevice")

    class RawInputStream:
        def __init__(self, **kw):
            self.kw = kw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self, size):
            assert size == n
            # cffi buffer analogue: buffer protocol WITHOUT .tobytes()
            assert not hasattr(ctypes.create_string_buffer(payload), "tobytes")
            return ctypes.create_string_buffer(payload, len(payload)), None

    fake.RawInputStream = RawInputStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    gen = MicSource(block_seconds=block, sample_rate=rate).stream()
    chunk = next(gen)
    assert chunk.sample_rate == rate
    assert chunk.pcm == payload