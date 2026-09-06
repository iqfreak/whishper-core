"""Test fakes. No external deps. Used to verify the pipeline wiring."""
from .base import (AudioSource, Transcriber, Translator, Display,
                   StreamingTranscriber, CaptionSource)
from ..types import AudioChunk, Segment, Translation


class FakeAudioSource(AudioSource):
    def __init__(self, chunks, block_seconds: float | None = None):
        self._chunks = list(chunks)
        self.block_seconds = block_seconds  # accepted for bench compat

    def stream(self):
        for c in self._chunks:
            yield c


class FakeCaptionSource(CaptionSource):
    """Emits a partial then a final on each iteration (live-caption cadence)."""

    def __init__(self, final_text="hello world", language="en", rounds=2):
        self._text = final_text
        self._language = language
        self._rounds = rounds

    def captions(self):
        for _ in range(self._rounds):
            yield Segment(status="partial",
                          source_text=f"{self._text} (partial)",
                          source_language=self._language)
            yield Segment(status="final", source_text=self._text,
                          source_language=self._language)


class FakeTranscriber(Transcriber):
    """Yields a finalized Segment only on the last chunk; None otherwise."""

    def __init__(self, final_text="hello world", language="en"):
        self._text = final_text
        self._language = language
        self._seen = 0

    def transcribe(self, chunk: AudioChunk):
        self._seen += 1
        # Finalize on every 3rd chunk to mimic VAD flush cadence.
        if self._seen % 3 == 0:
            return Segment(status="final", source_text=self._text,
                           source_language=self._language)
        return None


class FakeStreamingTranscriber(StreamingTranscriber):
    """Emits one partial then a final on every 3rd chunk (live-caption cadence)."""

    def __init__(self, final_text="hello world", language="en"):
        self._text = final_text
        self._language = language
        self._seen = 0

    def transcribe_stream(self, chunk: AudioChunk):
        self._seen += 1
        # Partial hypothesis on every chunk; lock in a final on every 3rd.
        yield Segment(status="partial",
                      source_text=f"{self._text} (partial)",
                      source_language=self._language)
        if self._seen % 3 == 0:
            yield Segment(status="final", source_text=self._text,
                          source_language=self._language)

    def transcribe(self, chunk: AudioChunk):
        # Not used by run_streaming(); required by the ABC.
        return None


class FakeTranslator(Translator):
    is_passthrough = False

    def __init__(self):
        self.calls = []

    def translate(self, text, source, target):
        self.calls.append((text, source, target))
        return Translation(
            source_text=text,
            target_text=f"[{target}] {text}",
            source_language=source,
            target_language=target,
        )


class FakeDisplay(Display):
    def __init__(self):
        self.shown = []
        self.partials = []

    def show(self, segment: Segment):
        self.shown.append(segment)

    def show_partial(self, segment: Segment):
        self.partials.append(segment)
