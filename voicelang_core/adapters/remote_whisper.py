"""Remote Whisper ASR as a StreamingTranscriber (the original voicelang seam).

This is the "use whisper over HTTP" path: local capture (MicSource /
WASAPILoopbackSource) feeds audio chunks to a remote whisper transcription
service. It is exactly the original voicelang architecture we reviewed
(Candidate A: ``ASR_ENDPOINT`` was a hardcoded HTTP call baked into
``utils.SendTranscriptionRequest``). Here it is a first-class, injectable port
instead of a hardcoded call site.

Latency is one network round-trip per chunk, so this is the higher-latency
option vs OpenASR / local faster-whisper. Use it when the model must live on a
separate box (e.g. a GPU server) and the client is thin.

The request/response contract mirrors the original voicelang transcription-api
(``POST {asr_endpoint}/transcribe`` with raw PCM, JSON ``{"text": ...}`` back).
Adjust the field/URL names if your endpoint differs -- this is the only
schema-coupled code in the adapter.
"""
from .base import StreamingTranscriber  # noqa: F401
from ..types import AudioChunk, Segment


class RemoteWhisperTranscriber(StreamingTranscriber):
    def __init__(self, asr_endpoint: str, timeout: float = 30.0,
                 language: str | None = None):
        self._endpoint = asr_endpoint.rstrip("/")
        self._timeout = timeout
        self._language = language

    def transcribe_stream(self, chunk: AudioChunk):
        import requests  # noqa: PLC0415 -- lazy so import voicelang_core never requires requests
        # Local PCM bytes -> remote whisper. One round-trip per chunk.
        files = {"audio": ("chunk.pcm", chunk.pcm, "application/octet-stream")}
        data = {}
        if self._language:
            data["language"] = self._language
        resp = requests.post(
            f"{self._endpoint}/transcribe", files=files, data=data,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        text = payload.get("text") or "".join(
            s.get("text", "") for s in payload.get("segments", [])
        ).strip()
        if text:
                    yield Segment(status="final", source_text=text,
                                  source_language=self._language)

    def transcribe(self, chunk: AudioChunk) -> "Segment | None":
        # Batch fallback used by Pipeline.run().
        for seg in self.transcribe_stream(chunk):
            return seg
        return None
