"""Moonshine Voice: per-language streaming ASR behind the Transcriber port.

The user's model strategy: tiny (34M-param) STREAMING models specialized per
language, downloaded ON DEMAND from a catalog the first time a language is
used, then cached locally and reused offline.

Supported languages (moonshine_voice.supported_languages()):
    ar de en es ja ko tl uk vi zh
Each maps to a streaming model (ar/en/es/... -> Tiny Streaming by default;
ko/uk fall back to their legacy Community models). `get_model_for_language()`
downloads into %LOCALAPPDATA%/moonshine_voice/... on first use and returns
the cached path afterwards.

Runtime (`moonshine-voice` extra): memory-mappable .ort ONNX models, no
torch/transformers. Requires:  uv sync --extra moonshine

Implementation notes:
- Events arrive on the library's processing thread; the adapter bridges them
  through a queue, drained per-fed-chunk (partials = in-progress hypothesis,
  lines = finished segments).
- `finish()` finalizes the tail of a finite stream (file replay): the last
  partial becomes a final line when the audio ends without a pause.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Iterator, Optional

import numpy as np

from .base import StreamingTranscriber  # noqa: F401
from ..types import AudioChunk, Segment, pcm_bytes_to_float32

_INSTANCES: dict = {}  # (language, arch_name) -> (Transcriber, model_path)
_REFCOUNTS: dict = {}  # (language, arch_name) -> int
_LOCK = threading.Lock()
_MAX_INSTANCES = 2


def _arch(name: str):
    import moonshine_voice as mv  # lazy: heavy dep, not part of the core

    return getattr(mv.ModelArch, name.upper(), mv.ModelArch.TINY_STREAMING)


class MoonshineTranscriber(StreamingTranscriber):
    """Moonshine Voice per-language streaming ASR.

    Model + language are fixed per instance; the underlying transcriber is
    cached per (language, arch) so switching languages mid-session only pays
    for a download, never a double load.
    """

    def __init__(self, language: str = "en", model_arch: str = "TINY_STREAMING",
                 update_interval: float = 0.12):
        # Moonshine's catalog is per-language (no auto-detect). "auto" — the
        # Whisper/Config default — must never reach get_model_for_language:
        # a saved config with language="auto" + --transcriber moonshine used
        # to crash with "Language not found: auto". Map it to English.
        language = language if language and language != "auto" else "en"
        self._language = language
        self._arch_name = model_arch.upper()
        self._update_interval = float(update_interval)
        self._queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._last_partial = ""
        self._last_emit_t = 0.0
        self._started = False

        key = (language, self._arch_name)
        self._key = key  # for refcounted close
        with _LOCK:
            hit = _INSTANCES.get(key)
            if hit is not None:
                # LRU: mark as recently used
                _INSTANCES.pop(key)
                _INSTANCES[key] = hit
                _REFCOUNTS[key] = _REFCOUNTS.get(key, 0) + 1
            else:
                # Evict LRU only if not still referenced
                if len(_INSTANCES) >= _MAX_INSTANCES:
                    # Find oldest with refcount 0 or 1 (evictable)
                    evict_key = None
                    for k in list(_INSTANCES.keys()):
                        if _REFCOUNTS.get(k, 1) <= 1:
                            evict_key = k
                            break
                    if evict_key is None:
                        evict_key = next(iter(_INSTANCES))
                    # Only close if refcount is 0/1 (no active live references beyond the registry)
                    if _REFCOUNTS.get(evict_key, 0) <= 1:
                        old_t, _ = _INSTANCES.pop(evict_key)
                        _REFCOUNTS.pop(evict_key, None)
                        closer = getattr(old_t, "close", None)
                        if callable(closer):
                            try:
                                closer()
                            except Exception:  # noqa: BLE001 -- eviction is best-effort
                                pass
                    else:
                        # Still referenced: skip eviction, just don't cache new entry beyond limit (or drop oldest anyway without closing)
                        # Drop registry entry but keep refcount tracking: remove from _INSTANCES but don't close
                        _INSTANCES.pop(evict_key, None)
                import moonshine_voice as mv  # lazy
                import moonshine_voice as mv  # lazy

                model_path, arch = mv.get_model_for_language(
                    language, _arch(self._arch_name), on_progress=None,
                )
                transcriber = mv.Transcriber(
                    model_path=model_path, model_arch=arch,
                    update_interval=self._update_interval,
                )
                _INSTANCES[key] = (transcriber, model_path)
                _REFCOUNTS[key] = 1
            self._t, self._model_path = _INSTANCES[key]
        # Every instance wires its OWN listener to its OWN queue; the
        # underlying model/transcriber is shared, the event routing is not.
        import moonshine_voice as mv  # noqa: PLC0415

        class _Listener(mv.TranscriptEventListener):
            def on_line_text_changed(self, event):  # partial hypothesis
                self._push("text", event.line.text or "")

            def on_line_completed(self, event):  # finished segment
                self._push("line", event.line.text or "")

        listener = _Listener()
        listener._push = self._push
        self._listener = listener
        self._t.add_listener(listener)

    def close(self) -> None:
        """Detach this instance's listener from the SHARED transcriber.

        WP-3.5 FIX: refcounted — closing a shared transcriber still referenced
        by another wrapper would kill it. We detach the listener always, but
        only close the underlying model when refcount reaches 0.
        """
        rem = getattr(self._t, "remove_listener", None)
        if rem is not None:
            try:
                rem(self._listener)
            except Exception:  # noqa: BLE001 -- detach is best-effort
                pass
        # Decrement refcount
        try:
            with _LOCK:
                k = getattr(self, "_key", None)
                if k is not None and k in _REFCOUNTS:
                    _REFCOUNTS[k] -= 1
                    if _REFCOUNTS[k] <= 0:
                        _REFCOUNTS.pop(k, None)
                        # Keep in _INSTANCES for LRU reuse; only closed on eviction or when explicitly evicted
                        pass
        except Exception:
            pass

    def _push(self, kind: str, text: str) -> None:
        self._queue.put((kind, text.strip()))

    def warmup(self, duration_s: float = 0.6):
        """Prime moonshine pipeline with silence so first real chunk is ~80ms not 500ms."""
        try:
            zeros = np.zeros(int(16000 * duration_s), dtype=np.float32)
            self._feed(zeros, 16000)
            # drain any warmup artifacts but don't emit
            time.sleep(0.05)
            _ = self._drain()
        except Exception:
            pass

    def _drain(self) -> list:
        items = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _feed(self, audio: np.ndarray, sample_rate: int) -> None:
        if not self._started:
            self._t.start()
            self._started = True
        self._t.add_audio(audio, sample_rate)

    def transcribe_stream(self, chunk: AudioChunk) -> Iterator[Segment]:
        # Handle stereo -> mono correctly (vosk-style). pcm_bytes_to_float32 assumes mono.
        if getattr(chunk, "channels", 1) != 1:
            try:
                import numpy as np
                pcm_i16 = np.frombuffer(chunk.pcm, dtype=np.int16)
                if chunk.channels > 1 and pcm_i16.size % chunk.channels == 0:
                    pcm_i16 = pcm_i16.reshape(-1, chunk.channels).mean(axis=1).astype(np.int16)
                    audio = pcm_i16.astype(np.float32) / 32768.0
                else:
                    audio = pcm_bytes_to_float32(chunk.pcm)
            except Exception:
                audio = pcm_bytes_to_float32(chunk.pcm)
        else:
            audio = pcm_bytes_to_float32(chunk.pcm)
        self._feed(audio, chunk.sample_rate)
        now = time.monotonic()
        for kind, text in self._drain():
            if not text:
                continue
            if kind == "line":
                self._last_partial = ""
                self._last_emit_t = now
                yield Segment(status="final", source_text=text,
                              source_language=None)
            elif text != self._last_partial:
                dt = now - getattr(self, '_last_emit_t', 0)
                last = self._last_partial
                grow = len(text) - len(last)
                is_extension = text.startswith(last) or last.startswith(text)
                should_emit = (dt >= self._update_interval) or (abs(grow) >= 3) or (not is_extension)
                if not should_emit and len(text) < 4:
                    continue
                if should_emit:
                    self._last_partial = text
                    self._last_emit_t = now
                    yield Segment(status="partial", source_text=text,
                                  source_language=None)

    def transcribe(self, chunk: AudioChunk) -> Optional[Segment]:
        for seg in self.transcribe_stream(chunk):
            return seg
        return None

    def finish(self) -> Iterator[Segment]:
        """Finalize the tail of a finite stream (file replay)."""
        if self._started:
            self._t.stop()
            self._started = False
            for kind, text in self._drain():
                if kind == "line" and text:
                    yield Segment(status="final", source_text=text,
                                  source_language=None)
