"""Smoke test for engine selection.

Stubs the heavy deps (faster-whisper, sounddevice, requests) so this stays
numpy-only, but still proves ``build_pipeline`` wires the correct ports and
``run_mode`` for every engine. This locks the regression class that the
original 9 tests missed: the factory must construct all four engines without
``ImportError``/``TypeError`` and hand the ``Pipeline`` the right ports.
"""
import sys
import types

import pytest

from voicelang_core.engines import EngineName, EngineConfig, build_pipeline
from voicelang_core.ports import CaptionSource, AudioSource, StreamingTranscriber


@pytest.fixture
def stub_heavy(monkeypatch):
    """Inject fake heavy modules so adapter imports don't need real deps."""
    fake_fw = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return ([], None)

    fake_fw.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    fake_sd = types.ModuleType("sounddevice")

    class RawInputStream:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self, n):
            import numpy as np

            return np.zeros(n, dtype="int16"), None

    fake_sd.RawInputStream = RawInputStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    fake_req = types.ModuleType("requests")

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"text": ""}

    fake_req.post = lambda *a, **k: _Resp()
    monkeypatch.setitem(sys.modules, "requests", fake_req)


def test_openasr_builds_captions():
    p, mode = build_pipeline(EngineConfig(name=EngineName.openasr, capture="mic"))
    assert mode == "captions"
    assert isinstance(p.source, CaptionSource)
    assert p.translator.is_passthrough is True


def test_passthrough_builds_batch():
    p, mode = build_pipeline(EngineConfig(name=EngineName.passthrough))
    assert mode == "batch"
    assert p.transcriber is not None


def test_faster_whisper_builds_streaming(stub_heavy):
    p, mode = build_pipeline(EngineConfig(
        name=EngineName.faster_whisper, model_size="small", device="cpu"))
    assert mode == "streaming"
    assert isinstance(p.source, AudioSource)
    assert isinstance(p.transcriber, StreamingTranscriber)


def test_whisper_http_builds_streaming(stub_heavy):
    p, mode = build_pipeline(EngineConfig(
        name=EngineName.whisper_http, asr_endpoint="http://localhost:5000"))
    assert mode == "streaming"
    assert isinstance(p.source, AudioSource)
    assert isinstance(p.transcriber, StreamingTranscriber)


def test_libretranslate_wired_when_configured(stub_heavy):
    p, mode = build_pipeline(EngineConfig(
        name=EngineName.faster_whisper, translate="libretranslate",
        libretranslate_url="http://lt:5000"))
    assert p.translator.is_passthrough is False
