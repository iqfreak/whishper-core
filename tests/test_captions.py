"""Verify the CaptionSource pipeline path + the OpenASR event parser contract."""
from voicelang_core import Pipeline
from voicelang_core.adapters import (
    FakeCaptionSource, FakeDisplay, PassthroughTranslator, FakeTranslator,
)
from voicelang_core.adapters.openasr_source import OpenASRSource


def test_run_captions_emits_partials_and_finals():
    src = FakeCaptionSource(final_text="hi there", language="es", rounds=2)
    disp = FakeDisplay()
    p = Pipeline(source=src, translator=PassthroughTranslator(),
                 display=disp, target_language="en")
    p.run_captions()

    # 2 rounds * (1 partial + 1 final) = 2 partials shown, 2 finals displayed.
    assert len(disp.partials) == 2
    assert len(disp.shown) == 2
    # Passthrough filled translated_text with the source on the SAME id.
    assert disp.shown[0].translated_text == "hi there"
    assert disp.shown[0].source_language == "es"
    assert [s.id for s in disp.shown] == [1, 2]


def test_run_captions_via_translator_when_not_passthrough():
    src = FakeCaptionSource(final_text="bonjour", language="fr", rounds=1)
    translator = FakeTranslator()
    disp = FakeDisplay()
    p = Pipeline(source=src, translator=translator, display=disp,
                 target_language="en")
    p.run_captions()
    # Partial is shown raw; only the FINAL caption is translated.
    assert translator.calls == [("bonjour", "fr", "en")]
    assert len(disp.shown) == 1


def test_openasr_parse_event_distinguishes_partial_and_final():
    # The one schema-coupled function; contract read from OpenASR Rust source.
    partial = OpenASRSource._parse_event('{"type":"transcript.partial","text":"I think","is_final":false}')
    final = OpenASRSource._parse_event('{"type":"transcript.final","text":"I think so","is_final":true}')
    none = OpenASRSource._parse_event('{"type":"session.created"}')
    junk = OpenASRSource._parse_event("not json at all")

    assert partial is not None and partial.status == "partial"
    assert final is not None and final.status == "final"
    assert none is None
    assert junk is None


def test_openasr_source_builds_expected_command(monkeypatch):
    launched = {}

    class FakeProc:
        def __init__(self, stdout, stderr): self.stdout = stdout; self.stderr = stderr
        def terminate(self): pass
        def wait(self, timeout=0): return 0
        def kill(self): pass

    def fake_popen(cmd, **kw):
        launched["cmd"] = cmd
        return FakeProc(iter([]), iter([]))   # empty stream -> no captions

    monkeypatch.setattr("voicelang_core.adapters.openasr_source.subprocess.Popen", fake_popen)
    src = OpenASRSource(binary="openasr", model="whisper-small", capture="mic")
    list(src.captions())  # drain (empty)
    assert "openasr" in launched["cmd"]
    assert "live" in launched["cmd"]
    assert "whisper-small" in launched["cmd"]