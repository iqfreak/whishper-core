"""Fun-ASR-Nano transcriber adapter (Hugging Face Transformers).

Wraps ``FunAudioLLM/Fun-ASR-Nano-2512-hf`` — the HF-Transformers build of
Fun-ASR-Nano (0.8B, zh/en/ja, 7 Chinese dialect groups + 26 regional
accents) — behind the ``StreamingTranscriber`` port.

Fun-ASR-Nano is a seq2seq ASR model, NOT a true streaming model: each
`generate()` call transcribes the whole utterance it is given. For live
captions this adapter therefore buffers incoming audio and runs inference
once per *energy-detected utterance* (voice onset -> finalize after a
silence gap, or a hard length cap), emitting hypothesis previews at a
coarser cadence while the utterance is still open. That mirrors how the
whisper adapter batches, but keeps `generate()` calls utterance-scoped
instead of per-chunk so the 0.8B model stays affordable.

Timing is accumulated from SAMPLES (seconds = samples / 16000), not wall
clock: deterministic under tests, and in live use it tracks the audio the
model actually hears rather than scheduler jitter.

Heavy deps (torch, transformers, accelerate, librosa) stay lazy — importing
this module never requires them. The Transformers integration is not yet in
a released transformers (PR huggingface/transformers#46180); the adapter is
version-tolerant: it prefers the new ``processor.apply_transcription_request``
chat-template API and falls back to the classic feature-extractor path.
"""
from __future__ import annotations

import threading
from typing import Iterator, Optional

import numpy as np

from ..ports import StreamingTranscriber
from ..types import AudioChunk, Segment, pcm_bytes_to_float32

# Model id, resolved by huggingface_hub on first use (cached under
# ~/.cache/huggingface). 0.8B bf16.
FUNASR_NANO_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512-hf"

# Nano recognizes Chinese (incl. zh-cn / dialect tags), English, Japanese.
NANO_LANGS = ("zh", "zh-cn", "en", "ja")

# SenseVoice-style 'auto' / dialect tags: Nano has no auto-detect UI tag;
# route 'auto'/'yue'/'cantonese' to Nano's Chinese model and reject the rest
# loudly (a silent language mismatch reads as "broken transcriber").
_LANG_ALIASES = {
    "auto": "zh",
    "yue": "zh",
    "cantonese": "zh",
}

# ---- capture-side defaults (seconds, at the adapter's 16 kHz input) -------
_VOICE_ENERGY = 300.0       # int16 RMS above which a chunk counts as speech
_SILENCE_GAP = 0.7          # trailing quiet (in samples) closes an utterance
_MAX_UTTERANCE = 12.0       # hard cap keeps memory/latency bounded
_MIN_UTTERANCE = 0.4        # discard accidental blips shorter than this
_PARTIAL_INTERVAL = 2.5     # hypothesis preview cadence while speaking

_RATE = 16000


def _resolve_device() -> str:
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _normalize_language(language: Optional[str]) -> Optional[str]:
    """Map GUI/config tags onto Nano's zh/en/ja; None = let the model decide."""
    if not language:
        return None
    lang = str(language).strip().lower()
    if lang in _LANG_ALIASES:
        return _LANG_ALIASES[lang]
    if lang in NANO_LANGS:
        return lang  # zh-cn is passed through as-is (model accepts it)
    raise ValueError(
        f"Fun-ASR-Nano supports zh/zh-cn/en/ja only (got {language!r}); "
        "for 31-language recognition use Fun-ASR-MLT-Nano-2512."
    )


class FunAsrNanoTranscriber(StreamingTranscriber):
    """Fun-ASR-Nano (HF Transformers) behind the StreamingTranscriber port.

    Args:
        language: zh / zh-cn / en / ja (or "auto" -> zh). Anything else raises
                  at construction time so a bad tag never silently mangles
                  output.
        model_id: override the default Fun-ASR-Nano checkpoint.
        device: "auto" | "cuda:0" | "cpu". Default "auto".
        silence_gap: seconds of quiet that closes a buffered utterance.
        max_utterance: hard cap on a single buffered utterance (seconds).
        partial_interval: hypothesis preview cadence while speaking (seconds).
    """

    def __init__(
        self,
        language: str | None = None,
        model_id: str = FUNASR_NANO_MODEL_ID,
        device: str = "auto",
        silence_gap: float = _SILENCE_GAP,
        max_utterance: float = _MAX_UTTERANCE,
        partial_interval: float = _PARTIAL_INTERVAL,
    ):
        self._language = _normalize_language(language)
        self._model_id = model_id
        self._device = _resolve_device() if device == "auto" else device
        self._silence_gap = silence_gap
        self._max_utterance = max_utterance
        self._partial_interval = partial_interval

        self._model = None
        self._processor = None

        # utterance buffer state (sample-accumulated timing)
        self._reset()

    # ---- model lifecycle ---------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Fun-ASR-Nano needs torch + the transformers PR build:\n"
                "  uv sync --extra funasr-nano\n"
                "(see https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf)"
            ) from exc
        try:
            from transformers import (  # noqa: PLC0415
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Fun-ASR-Nano needs transformers with the Fun-ASR integration "
                "(PR huggingface/transformers#46180):\n"
                "  uv sync --extra funasr-nano"
            ) from exc

        key = (self._model_id, self._device)
        with FunAsrNanoTranscriber._registry_lock:
            entry = FunAsrNanoTranscriber._registry.get(key)
            if entry is not None:
                self._model, self._processor = entry
                return
            dtype = torch.bfloat16 if self._device.startswith("cuda") else torch.float32
            # HF mirror for CN users: keep env HF_ENDPOINT if set
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self._model_id, dtype=dtype, device_map="auto", trust_remote_code=True,
                    local_files_only=False,
                )
            except Exception:  # noqa: BLE001 -- device_map needs accelerate; fall back
                try:
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        self._model_id, dtype=dtype, trust_remote_code=True,
                    ).to(self._device)
                except Exception:
                    # final fallback: local cache only (offline)
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        self._model_id, dtype=dtype, trust_remote_code=True, local_files_only=True,
                    ).to(self._device)
            processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
            FunAsrNanoTranscriber._registry[key] = (model, processor)
            self._model, self._processor = model, processor

    _registry: dict = {}
    _registry_lock = threading.Lock()

    # ---- input preparation (version-tolerant) ------------------------------

    def _prepare_inputs(self, audio: np.ndarray):
        """Processor inputs for the current chunk buffer."""
        proc = self._processor
        try:
            # PR #46180 chat-template API: the checkpoint builds the full
            # instruction (language, prompt, hotwords) itself.
            inputs = proc.apply_transcription_request(
                audio=audio, language=self._language, return_tensors="pt",
            )
        except (AttributeError, TypeError, NotImplementedError):
            # Classic seq2seq feature-extractor path (any transformers build).
            inputs = proc(
                audio=audio, sampling_rate=_RATE, return_tensors="pt",
            )
        return inputs.to(self._device)

    def _transcribe_buffer(self) -> str:
        """Run one generate() on the current buffer; returns decoded text."""
        audio = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
        inputs = self._prepare_inputs(audio)
        generated = self._model.generate(**inputs, max_new_tokens=200)
        # chat-template path prepends instruction tokens — slice them off.
        # BatchFeature may expose input_ids as attribute or dict key.
        input_ids = getattr(inputs, "input_ids", None)
        if input_ids is None and isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
        if input_ids is not None:
            try:
                # handle both torch.Tensor / np.ndarray (shape) and list
                if hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
                    prompt_len = int(input_ids.shape[1])
                elif hasattr(input_ids, "shape"):
                    prompt_len = int(input_ids.shape[0])
                else:
                    prompt_len = len(input_ids[0]) if len(input_ids) else 0
                generated = generated[:, prompt_len:]
            except Exception:
                pass
        return str(self._processor.decode(generated[0], skip_special_tokens=True)).strip()

    # ---- utterance state machine (sample-accumulated timing) ---------------

    def _flush(self, status: str) -> Optional[Segment]:
        if self._voice_seconds < _MIN_UTTERANCE:
            self._reset()
            return None
        text = self._transcribe_buffer()
        self._reset()
        if not text:
            return None
        return Segment(status=status, source_text=text, source_language=self._language)

    def _reset(self) -> None:
        self._buf: list[np.ndarray] = []
        self._buf_samples = 0
        self._buf_seconds = 0.0
        self._voice_seconds = 0.0  # voiced time only (blips stay blips)
        self._silent_seconds = 0.0
        self._partial_at = 0.0  # buffered-seconds offset of the last preview

    # ---- port ---------------------------------------------------------------

    def transcribe_stream(self, chunk: AudioChunk) -> Iterator[Segment]:
        self._ensure_model()
        audio = pcm_bytes_to_float32(chunk.pcm)
        if audio.size == 0:
            return
        seconds = audio.size / _RATE
        rms = float(np.sqrt(np.mean(audio * audio))) * 32768.0  # int16 scale

        if rms >= _VOICE_ENERGY:
            # voice: open/resume the utterance
            self._buf.append(audio)
            self._buf_samples += audio.size
            self._buf_seconds += seconds
            self._voice_seconds += seconds
            self._silent_seconds = 0.0
        elif self._buf_samples:
            # quiet while an utterance is open: keep the tail, count silence
            self._buf.append(audio)
            self._buf_samples += audio.size
            self._buf_seconds += seconds
            self._silent_seconds += seconds
            if self._silent_seconds >= self._silence_gap:
                seg = self._flush("final")
                if seg is not None:
                    yield seg
                return
        else:
            return  # pre-voice silence: drop, don't buffer

        # hard cap — a single utterance never grows unbounded
        if self._buf_seconds >= self._max_utterance:
            seg = self._flush("final")
            if seg is not None:
                yield seg
            return

        # coarse hypothesis preview while still speaking
        if (self._buf_seconds >= _PARTIAL_INTERVAL
                and self._buf_seconds - self._partial_at >= self._partial_interval):
            self._partial_at = self._buf_seconds
            try:
                text = self._transcribe_buffer()
            except Exception:  # noqa: BLE001 -- preview is best-effort
                text = ""
            if text:
                yield Segment(status="partial", source_text=text,
                              source_language=self._language)

    def transcribe(self, chunk: AudioChunk) -> Optional[Segment]:
        # batch compat: the last final (mirrors the funasr adapter behavior)
        last = None
        for seg in self.transcribe_stream(chunk):
            if seg.status == "final":
                last = seg
        return last

    def finish(self) -> Iterator[Segment]:
        """Flush the trailing utterance (file replay / stop)."""
        if not self._buf_samples:
            return
        seg = self._flush("final")
        if seg is not None:
            yield seg