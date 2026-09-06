"""The pipeline: consume a caption stream, translate, display.

Pure orchestration. Depends only on ports. No OpenASR, faster-whisper, or
LibreTranslate imported here.

Three run modes:
  run_captions()      -- CaptionSource: best for real-time (OpenASR). Each
                         Segment already carries (optionally translated) text;
                         English-only OpenASR sets is_passthrough on the
                         translator, so no second network hop.
  run_streaming()     -- StreamingTranscriber + AudioSource.
  run()               -- batch Transcriber + AudioSource.

Segment ids are owned HERE. Adapters yield `id=0` sentinels; the pipeline
stamps monotonic ids that increment ONLY on `final`, so partial updates
mutate the in-flight segment in place and the transcription/translation
blocks can never drift out of sync.
"""
from typing import Optional

import time

from .ports import (
    AudioSource, Transcriber, StreamingTranscriber, Translator, Display,
    CaptionSource,
)
from .types import Segment


class Pipeline:
    def __init__(
        self,
        *,
        source: Optional["CaptionSource | AudioSource"] = None,
        transcriber: Optional["Transcriber | StreamingTranscriber"] = None,
        translator: Optional[Translator] = None,
        display: Optional[Display] = None,
        target_language: str = "en",
        source_language: str = "auto",
    ) -> None:
        self.source = source
        self.transcriber = transcriber
        self.translator = translator
        self.display = display
        self.target_language = target_language
        self.source_language = source_language
        # id contract: next fresh id, and the id of the utterance in flight.
        self._next_id = 1
        self._open_id: Optional[int] = None
        # no-audio watchdog state
        self._silence_start: Optional[float] = None
        self._silence_warned: float = 0.0
        self._silent_chunks: int = 0
        self._silence_warn_seconds: float = 5.0
        self._silence_warn_chunks: int = 20
        self._silence_cooldown: float = 30.0

    # ---- id stamping --------------------------------------------------------

    def _stamp_id(self, seg: "Segment") -> None:
        """Assign the monotonic id to an adapter-emitted segment.

        Adapters yield `id=0` as a sentinel. A non-zero id (adapter already
        knows) is trusted and absorbed into the counters.
        """
        if seg.id == 0:
            if self._open_id is None:
                self._open_id = self._next_id
            seg.id = self._open_id
        else:
            self._open_id = seg.id

    # ---- CaptionSource path (OpenASR-shaped) --------------------------------

    def _handle_segment(self, seg: "Segment", translator: "Translator") -> None:
        self._stamp_id(seg)
        if seg.status == "final":
            self._render(seg, translator)
            # Only a final advances the counter: the next utterance gets the
            # next id. Partials never allocate.
            self._next_id = max(self._next_id, seg.id + 1)
            self._open_id = None
            self.display.show(seg)
        else:
            # Live partial preview: mutates the SAME segment id in place.
            self.display.show_partial(seg)

    def run_captions(self) -> None:
        if not isinstance(self.source, CaptionSource):
            raise TypeError("run_captions requires a CaptionSource")
        translator = self._require_translator()
        for seg in self.source.captions():
            self._handle_segment(seg, translator)

    # ---- Transcriber paths (own capture) ------------------------------------

    def run_streaming(self) -> None:
        if not isinstance(self.transcriber, StreamingTranscriber):
            raise TypeError("run_streaming requires a StreamingTranscriber")
        src = self._require_audio_source()
        translator = self._require_translator()
        for chunk in src.stream():
            self._watch_silence(chunk)
            for seg in self.transcriber.transcribe_stream(chunk):
                self._handle_segment(seg, translator)
            # Pump Qt loop so overlay coalesce timer + queued renders fire
            # even though this thread IS the GUI thread (blocking capture).
            try:
                pump = getattr(self.display, "_pump", None)
                if callable(pump):
                    pump()
                elif getattr(self.display, "_app", None) is not None:
                    self.display._app.processEvents()  # type: ignore
            except Exception:
                pass

    def run(self) -> None:
        src = self._require_audio_source()
        translator = self._require_translator()
        for chunk in src.stream():
            seg = self.transcriber.transcribe(chunk)
            if seg is None:
                continue
            self._handle_segment(seg, translator)

    # ---- helpers ------------------------------------------------------------

    def _watch_silence(self, chunk: "AudioChunk") -> None:
        """Announce silent capture so "nothing appears" becomes actionable.

        A live source producing near-zero energy for a few seconds means the
        input is muted / the wrong device / Windows blocks mic access. Warn
        once per cooldown via ``display.warn`` instead of failing silently.
        """
        from .types import pcm_energy

        now = time.monotonic()
        if pcm_energy(chunk.pcm) < 20:
            self._silent_chunks += 1
            if self._silence_start is None:
                self._silence_start = now
            if ((self._silent_chunks >= self._silence_warn_chunks
                 or now - self._silence_start > self._silence_warn_seconds)
                    and now - self._silence_warned > self._silence_cooldown):
                self._silence_warned = now
                name = type(self.source).__name__ if self.source is not None else "source"
                hint = (
                    "silent input — check the mic (level/mute) and Windows "
                    "Settings → Privacy → Microphone → allow desktop apps"
                    if "Mic" in name else
                    "no audio playing on the captured output device"
                )
                self.display.warn(f"{name}: no audio detected for "
                                  f"{int(self._silence_warn_seconds)}s — {hint}")
        else:
            self._silence_start = None
            self._silent_chunks = 0

    def _render(self, seg: "Segment", translator: "Translator") -> None:
        """Translate segment text and write it back onto the SAME segment.

        `translated_text` fills in on the existing id — late translation
        arrival never creates a new row in the display blocks. A translation
        failure degrades gracefully (spec §9): `translated_text` stays None,
        the transcription block keeps showing source text, and the worker
        does NOT crash.
        """
        src = seg.source_language or self.source_language
        if translator.is_passthrough:
            seg.translated_text = seg.source_text
            return
        try:
            seg.translated_text = translator.translate(
                seg.source_text, src, self.target_language
            ).target_text
        except Exception as exc:  # noqa: BLE001 -- degrade, never crash the worker
            seg.translated_text = None
            # Agent4: surface runtime translation failure with clear mode-specific error (not silent)
            try:
                mode_name = type(translator).__name__
                _friendly = {"LibreTranslateTranslator": "libretranslate", "DeepLTranslator": "deepl", "DeepLXTranslator": "deeplx", "GoogleTranslator": "google"}.get(mode_name, mode_name)
                # include mode and hint so misconfigured URL/key is actionable
                hint = f"Translation is set to {_friendly} but failed: {exc} — check translate_url/translate_key"
                # display may be None in some tests; guard
                if self.display is not None and hasattr(self.display, "warn"):
                    self.display.warn(hint)
            except Exception:
                pass

    def _require_translator(self) -> "Translator":
        if self.translator is None:
            raise RuntimeError("Pipeline.translator is required for this run mode")
        return self.translator

    def _require_audio_source(self) -> "AudioSource":
        if not isinstance(self.source, AudioSource):
            raise TypeError("this run mode requires an AudioSource")
        return self.source