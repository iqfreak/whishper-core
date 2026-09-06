"""Moonshine language guard: "auto" (Whisper's config default) must map to "en".

Regression: a saved config with language="auto" + "--transcriber moonshine"
crashed with "Language not found: auto" — the frozen smoke test caught it.
Stubs moonshine_voice so no model lookup/download happens.
"""
import sys
import types


def _install_stub_monkeypatch(monkeypatch):
    fake_mv = types.ModuleType("moonshine_voice")
    fake_mv.ModelArch = types.SimpleNamespace(TINY_STREAMING="TINY_STREAMING")
    fake_mv.TranscriptEventListener = type("TranscriptEventListener", (), {})

    def get_model_for_language(language, model_arch, on_progress=None):
        return ("MOCK_MODEL_PATH", model_arch)

    class _FakeTranscriber:
        def __init__(self, model_path=None, model_arch=None, update_interval=0.2):
            pass

        def add_listener(self, listener):
            pass

    fake_mv.get_model_for_language = get_model_for_language
    fake_mv.Transcriber = _FakeTranscriber
    monkeypatch.setitem(sys.modules, "moonshine_voice", fake_mv)


def test_moonshine_auto_language_maps_to_en(monkeypatch):
    _install_stub_monkeypatch(monkeypatch)
    from voicelang_core.adapters.moonshine import MoonshineTranscriber

    t = MoonshineTranscriber(language="auto", model_arch="TINY_STREAMING")
    assert t._language == "en"  # never reaches get_model_for_language as "auto"


def test_moonshine_empty_language_maps_to_en(monkeypatch):
    _install_stub_monkeypatch(monkeypatch)
    from voicelang_core.adapters.moonshine import MoonshineTranscriber

    t = MoonshineTranscriber(language="", model_arch="TINY_STREAMING")
    assert t._language == "en"


def test_moonshine_concrete_language_preserved(monkeypatch):
    _install_stub_monkeypatch(monkeypatch)
    from voicelang_core.adapters.moonshine import MoonshineTranscriber

    t = MoonshineTranscriber(language="ar", model_arch="TINY_STREAMING")
    assert t._language == "ar"