"""Ports (interfaces) for the streaming pipeline.

Everything in the core depends only on these. Concrete adapters live in
voicelang_core.adapters.* and are NOT imported by the pipeline, so the core
is testable with fakes and free of any heavy dependency.

Two engine shapes are supported:

- CaptionSource  -- a self-contained live caption stream (capture + ASR +
  translate fused behind one port). This is what OpenASR / whisper-streaming
  expose, and it is the recommended seam for a real-time product.
- Transcriber / StreamingTranscriber -- a chunk-in, segment-out transcriber
  when you own the audio capture yourself.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .types import AudioChunk, Segment, Translation


class CaptionSource(ABC):
    """Live caption stream: emits partial AND final Segments over time.

    The implementation owns capture + ASR (+ optional translate). The pipeline
    consumes the stream; it never touches raw audio or the model.
    """

    @abstractmethod
    def captions(self) -> Iterator[Segment]:
        ...


class AudioSource(ABC):
    """Capture seam. Produces a stream of audio chunks."""

    @abstractmethod
    def stream(self) -> Iterator[AudioChunk]:
        ...


class Transcriber(ABC):
    """Batch transcribe seam. Turns audio into finalized transcript segments.

    Implementations handle VAD chunking and model caching internally.
    """

    @abstractmethod
    def transcribe(self, chunk: AudioChunk) -> Optional[Segment]:
        """Return a finalized Segment, or None if the chunk is not final."""


class StreamingTranscriber(Transcriber):
    """Streaming variant: yields partial AND final segments per chunk.

    Implementations keep an in-process model and feed it continuous audio
    (no HTTP round-trip, no per-call model reload).
    """

    @abstractmethod
    def transcribe_stream(self, chunk: AudioChunk) -> Iterator[Segment]:
        ...


class Translator(ABC):
    """Translate seam. For English-only OpenASR paths, use PassthroughTranslator."""

    @property
    def is_passthrough(self) -> bool:
        return False

    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> Translation:
        ...


class Display(ABC):
    """Display seam. Where captions land: console, Discord, overlay."""

    @abstractmethod
    def show(self, segment: Segment) -> None:
        """Render a finalized segment (id + source_text + translated_text).

        The segment carries the shared monotonic `id`; both transcription
        and translation blocks render the same id so they never drift.
        """

    @abstractmethod
    def show_partial(self, segment: Segment) -> None:
        """Live partial preview (raw, untranslated hypothesis). Contract-required.

        Mutates the SAME segment id in place — never allocates a new id.
        """

    def warn(self, text: str) -> None:
        """Non-fatal advisory (e.g. the no-audio watchdog). Default: ignore."""

    def close(self, timeout: float = 1.0) -> None:
        """Flush and release resources. Default no-op; file sink overrides."""

    @property
    def stats(self) -> dict:
        return {}


__all__ = ["CaptionSource", "AudioSource", "Transcriber", "StreamingTranscriber",
           "Translator", "Display"]
