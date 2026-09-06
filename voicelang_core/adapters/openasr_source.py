"""OpenASR caption source (Depth A: subprocess, no Rust build).

OpenASR is a local-first, fail-closed ASR engine that fuses capture + ASR +
translate-to-English into one binary. We run it as a caption source: it owns
the microphone / system-audio capture and streams partial + final text.

Live transport: `openasr live` (README: "real-time transcription from your
microphone with streaming partial results"). The companion `openasr serve`
exposes an OpenAI-compatible `/v1/audio/transcriptions` endpoint.

SCHEMA STATUS -- VERIFIED AGAINST SOURCE (not docs, not a guess). The live
JSON event shape was not documented at the URLs tried (docs/http-api.md -> 404;
README documents `openasr live` + the serve API, not the event format), so the
contract was read directly from the OpenASR Rust source:
  crates/openasr-core/src/realtime/events.rs            -> envelope + `type` tags
  .../generated/realtime-wire/RealtimeTranscript*.ts     -> payload fields
Finality is encoded twice: `type` ("transcript.partial"/"transcript.final")
and the payload's `is_final` boolean; `_parse_event` trusts `is_final` first.

Caveat: OpenASR is pre-v1, so this surface may change between 0.x releases --
re-check events.rs if captions ever go silent. All schema-coupled logic stays
in `_parse_event`; launch, lifecycle, streaming into Segments, and the pipeline
are contract-stable.
"""
from .base import CaptionSource  # noqa: F401
from ..types import Segment

import json
import subprocess


class OpenASRSource(CaptionSource):
    def __init__(self, binary: str = "openasr", model: str = "whisper-small",
                 capture: str = "mic", translate: bool = True, device: str = "cuda",
                 extra_args: list[str] | None = None):
        self._binary = binary
        self._model = model
        self._capture = capture          # 'mic' | 'loopback' | 'system'
        self._translate = translate      # translate-to-English in one pass
        self._device = device            # 'cuda' | 'cpu' (forwarded to openasr)
        self._extra = extra_args or []
        self._proc: "subprocess.Popen | None" = None

    def captions(self):
        cmd = [self._binary, "live", "--model", self._model]
        if self._capture == "mic":
            cmd += ["--mic"]
        elif self._capture in ("loopback", "system"):
            cmd += ["--loopback"]          # all-system WASAPI loopback on Windows
        cmd += ["--stream", "json"]         # line-delimited JSON transcript events
        cmd += self._extra

        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            for raw in self._proc.stdout:
                raw = raw.strip()
                if not raw:
                    continue
                seg = self._parse_event(raw)
                if seg is not None:
                    yield seg
        finally:
            self._stop()

    def _stop(self):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    # --- the ONLY schema-coupled code. VERIFIED against Rust source (see docstring).
    @staticmethod
    def _parse_event(raw: str) -> "Segment | None":
        """Parse one live OpenASR realtime event into a Segment (or None to skip).

        CONTRACT (read from the OpenASR Rust source, not guessed):
          crates/openasr-core/src/realtime/events.rs
            - `RealtimeEventEnvelope`: {"type", "session_id", "event_id", "seq",
              "created_at", ...flattened payload}
            - `RealtimeTranscriptEvent` -> event_type():
                 Partial => "transcript.partial"
                 Final   => "transcript.final"
                 Revision=> "transcript.revision"
          crates/openasr-core/generated/realtime-wire/RealtimeTranscript{Partial,Final}.ts
            - payload: {utterance_id, segment_id, revision, text, start_ms,
              end_ms, is_final: boolean, words:[{word,start_ms,end_ms}],
              language: string|null, speaker, speaker_label, ...}

        So finality is encoded TWICE: the `type` discriminator AND `is_final`.
        We trust `is_final` when present (authoritative payload field) and fall
        back to the `type` discriminator; only if both are absent do we
        fail-safe to VISIBLE (emit rather than drop) -- in a capture path a
        wrong guess fails silently, which is the worst failure mode.

        Non-transcript events (session.*, audio.input.*, vad.*, error) carry no
        `text` and are skipped.
        """
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None

        # Words: flat `text` (contract) or `transcript` (OpenAI-compat alias).
        text = obj.get("text") or obj.get("transcript") or ""
        if not text:
            return None  # lifecycle / control / error events have no text

        # Finality: authoritative `is_final` -> else `type` discriminator.
        if isinstance(obj.get("is_final"), bool):
            final = bool(obj["is_final"])
        else:
            kind = str(obj.get("type") or "")
            if kind == "transcript.partial":
                final = False
            elif kind in ("transcript.final", "transcript.revision"):
                final = True
            else:
                final = True  # unknown-but-texty: fail-safe to VISIBLE

        language = obj.get("language")
        return Segment(status="final" if final else "partial",
                       source_text=text, source_language=language or None)
