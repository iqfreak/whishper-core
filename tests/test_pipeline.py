"""Verify the pipeline wiring and the model-cache contract without any heavy deps."""
from voicelang_core import Pipeline
from voicelang_core.adapters import (
    FakeAudioSource, FakeTranscriber, FakeTranslator, FakeDisplay,
)
from voicelang_core.types import AudioChunk


def _chunks(n):
    return [AudioChunk(pcm=b"\x00\x00" * 16000, sample_rate=16000) for _ in range(n)]


def test_pipeline_stamps_monotonic_ids_only_on_finals():
    # 6 chunks -> 2 finals (every 3rd), each preceded by partials.
    from voicelang_core.adapters.fake import FakeStreamingTranscriber as FST

    disp = FakeDisplay()
    p = Pipeline(
        source=FakeAudioSource(_chunks(6)),
        transcriber=FST(final_text="hi there", language="es"),
        translator=FakeTranslator(), display=disp,
        target_language="en", source_language="auto",
    )
    p.run_streaming()
    # Finals are numbered 1, 2 — sequential, monotonic, starting at 1.
    assert [s.id for s in disp.shown] == [1, 2]
    # Partials mutated ids in place — they never allocated new ids.
    partial_ids = [s.id for s in disp.partials]
    assert partial_ids[0] == 1 and partial_ids[3] == 2


def test_pipeline_finalizes_only_on_flush_and_displays():
    src = FakeAudioSource(_chunks(6))          # 6 chunks -> 2 finalizations (every 3rd)
    trans = FakeTranscriber(final_text="hi there", language="es")
    trans2 = FakeTranslator()
    disp = FakeDisplay()

    p = Pipeline(source=src, transcriber=trans, translator=trans2, display=disp,
                 target_language="en", source_language="auto")
    p.run()

    # Two segments finalized (chunk 3 and chunk 6).
    assert len(disp.shown) == 2
    # Translator was called for each finalization with detected language.
    assert trans2.calls[0] == ("hi there", "es", "en")
    # Display received the translated caption on the SAME segment id.
    assert disp.shown[0].translated_text == "[en] hi there"
    assert [s.id for s in disp.shown] == [1, 2]


def test_transcriber_returns_none_for_non_final_chunks():
    trans = FakeTranscriber()
    # First 2 chunks should not finalize.
    assert trans.transcribe(_chunks(1)[0]) is None
    assert trans.transcribe(_chunks(1)[0]) is None
    assert trans.transcribe(_chunks(1)[0]) is not None  # 3rd finalizes