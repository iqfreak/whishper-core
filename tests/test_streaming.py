"""Verify the streaming pipeline + English-only passthrough path."""
from voicelang_core import Pipeline
from voicelang_core.adapters import (
    FakeAudioSource, FakeStreamingTranscriber, FakeTranslator, FakeDisplay,
    PassthroughTranslator,
)
from voicelang_core.types import AudioChunk


def _chunks(n):
    return [AudioChunk(pcm=b"\x00\x00" * 16000, sample_rate=16000) for _ in range(n)]


def test_streaming_runs_partials_and_finals():
    src = FakeAudioSource(_chunks(6))          # 6 chunks
    trans = FakeStreamingTranscriber(final_text="hi there", language="es")
    translator = FakeTranslator()              # not passthrough -> translate called
    disp = FakeDisplay()

    p = Pipeline(source=src, transcriber=trans, translator=translator, display=disp,
                 target_language="en")
    p.run_streaming()

    # 6 partials shown live, 2 finals (every 3rd chunk) sent to translator.
    assert len(disp.partials) == 6
    assert len(translator.calls) == 2
    assert translator.calls[0] == ("hi there", "es", "en")
    assert [s.id for s in disp.shown] == [1, 2]


def test_passthrough_skips_translation_network_hop():
    src = FakeAudioSource(_chunks(3))
    trans = FakeStreamingTranscriber(final_text="bonjour", language="fr")
    disp = FakeDisplay()

    p = Pipeline(source=src, transcriber=trans, translator=PassthroughTranslator(),
                 display=disp, target_language="en")
    p.run_streaming()

    # Passthrough: display got the same text Whisper produced, on the shared id.
    assert len(disp.shown) == 1
    assert disp.shown[0].source_text == "bonjour"
    assert disp.shown[0].translated_text == "bonjour"
    assert disp.shown[0].source_language == "fr"