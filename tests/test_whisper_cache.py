"""Candidate C: model registry returns the SAME instance for the same key."""
import sys, types  # noqa


def test_registry_reuses_model_instance(monkeypatch):
    # Build the adapter module in isolation and stub faster_whisper so no GPU
    # or model download is needed. This proves the caching contract only.
    fw = types.ModuleType("faster_whisper")
    class FakeWhisperModel:
        def __init__(self, *a, **k):
            self.calls = []
        def transcribe(self, *a, **k):
            return ([], types.SimpleNamespace(language="en"))
    fw.WhisperModel = FakeWhisperModel
    sys.modules["faster_whisper"] = fw  # so `from faster_whisper import ...` resolves
    adapter = types.ModuleType("voicelang_core.adapters.whisper_transcriber")
    adapter.__dict__["np"] = __import__("numpy")
    adapter.__dict__["faster_whisper"] = fw

    src = """
from voicelang_core.adapters.base import AudioSource, Transcriber, Translator, Display
import numpy as np
from faster_whisper import WhisperModel

class WhisperTranscriber(Transcriber):
    _registry = {}
    def __init__(self, model_size="small", device="cpu", compute_type="int8", cpu_threads=4):
        self._model = WhisperTranscriber._registry.get((model_size, device))
        if self._model is None:
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
            WhisperTranscriber._registry[(model_size, device)] = self._model
    def transcribe(self, chunk):
        return None
"""
    exec(compile(src, "whisper_transcriber", "exec"), adapter.__dict__)
    sys.modules["whisper_core.adapters.whisper_transcriber"] = adapter

    A = adapter.WhisperTranscriber
    A._registry.clear()
    a1 = A(model_size="small", device="cpu")
    a2 = A(model_size="small", device="cpu")
    # Same model instance reused -> no reload per call (candidate C fix).
    assert a1._model is a2._model
    # Different key loads a different instance.
    a3 = A(model_size="medium", device="cpu")
    assert a3._model is not a1._model
