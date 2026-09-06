"""faster-whisper Transcriber adapter.

Fixes candidate C from the voicelang review: the original transcription-api
built and loaded a fresh Whisper model on every request. Streaming fires
many short chunks, so we keep a registry keyed by (model_size, device) and
reuse the loaded model across the session. A loaded model is a deep module:
one instance, N transcription calls.
"""
from .base import AudioSource, Transcriber, Translator, Display  # noqa: F401
from ..types import AudioChunk, Segment, pcm_bytes_to_float32

import threading


class WhisperTranscriber(Transcriber):
    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8", cpu_threads: int = 4):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        key = (model_size, device)
        with WhisperTranscriber._registry_lock:
            self._model = WhisperTranscriber._registry.get(key)
            if self._model is None:
                from faster_whisper import WhisperModel  # lazy: no import cost when using moonshine/nemotron

                self._model = WhisperModel(
                    model_size, device=device,
                    compute_type=compute_type, cpu_threads=cpu_threads,
                )
                WhisperTranscriber._registry[key] = self._model

    _registry: dict = {}  # (model_size, device) -> WhisperModel
    _registry_lock = threading.Lock()

    @classmethod
    def prewarm(cls, model_size="small", device="cpu"):
        cls(model_size=model_size, device=device)

    def transcribe(self, chunk: AudioChunk) -> Segment | None:
        audio = pcm_bytes_to_float32(chunk.pcm)
        segments, info = self._model.transcribe(
            audio, beam_size=5, word_timestamps=False, language=None,
        )
        # Simple flush: return the joined text of this chunk's segments.
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            return None
        return Segment(status="final", source_text=text,
                       source_language=info.language)