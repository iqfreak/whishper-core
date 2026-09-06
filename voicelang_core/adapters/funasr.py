"""FunASR StreamingTranscriber adapter (modelscope.github.io FunASR).

Wraps DAMO's FunASR toolkit (Paraformer / SenseVoice) behind the
StreamingTranscriber port. Heavy deps (torch, funasr, modelscope) stay
lazy — importing this module never requires them.

Language routing:
  zh / yue / zh-cn  -> paraformer-zh-streaming (true streaming, 220M, cache+is_final)
  other / auto / en -> SenseVoiceSmall (multilingual, batch-per-chunk; good quality, no cache)

Both paths expose the same StreamingTranscriber API (transcribe_stream + finish).
For the streaming Paraformer, we use the cache+chunk_size protocol from the
FunASR docs (chunk_size=[0,10,5], 600 ms stride, encoder/decoder look-back).

Model downloads are handled by FunASR/ModelScope on first use (cached under
~/.cache/modelscope). Device is auto-probed: \"cuda\" when torch.cuda.is_available()
else \"cpu\".
"""
from typing import Iterator, Optional

import numpy as np

from ..ports import StreamingTranscriber
from ..types import AudioChunk, Segment, pcm_bytes_to_float32


# Model ids
_ZH_STREAMING = "paraformer-zh-streaming"  # iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online on HF
_SENSE_VOICE = "iic/SenseVoiceSmall"       # multilingual ASR (zh/en/yue/ja/ko)


def _resolve_device() -> str:
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _pcm_bytes_to_float32(pcm: bytes) -> np.ndarray:
    # Canonical conversion lives in types.py (Phase 5 unification) — the old
    # local copy drifted on dtype spelling (<i2 vs int16).
    return pcm_bytes_to_float32(pcm)


class FunASRTranscriber(StreamingTranscriber):
    """FunASR-backed StreamingTranscriber.

    Args:
        language: language tag (en, zh, auto, ...). Selects the underlying
                  FunASR model. Defaults to \"auto\" (SenseVoice auto-detect).
        model: optional explicit ModelScope model id to override routing.
        device: \"auto\" | \"cuda:0\" | \"cpu\". Default \"auto\".
    """

    def __init__(self, language: str = "auto", model: str | None = None, device: str = "auto"):
        self._language = (language or "auto").lower()
        # Normalize bare SenseVoiceSmall (GUI sends "SenseVoiceSmall" without org) to full ModelScope id
        if model == "SenseVoiceSmall":
            model = _SENSE_VOICE
        self._model_id = model or ( _ZH_STREAMING if self._language in ("zh", "zh-cn", "yue", "cantonese") else _SENSE_VOICE)
        self._device = _resolve_device() if device == "auto" else device
        self._model = None
        self._cache: dict = {}
        # Streaming Paraformer chunk config (600 ms)
        self._chunk_size = [0, 10, 5]
        self._enc_look_back = 4
        self._dec_look_back = 1
        self._is_streaming_model = "streaming" in self._model_id

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from funasr import AutoModel  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "FunASR not installed. Install with:\n"
                "  uv sync --extra funasr   # or: pip install funasr modelscope torch torchaudio\n"
                "See https://github.com/modelscope/FunASR") from exc
        # AutoModel handles download+caching
        # For streaming Paraformer we keep VAD/punc off and let the pipeline drive chunking
        # For SenseVoice we add minimal VAD for better segmentation
        # robust download: try ModelScope alias, then HF mirror id, then local cache
        kwargs = {"model": self._model_id, "device": self._device, "hub": "ms"}
        if not self._is_streaming_model:
            kwargs["disable_update"] = True
        try:
            self._model = AutoModel(**kwargs)
        except Exception as e1:
            # retry without hub pin, then with HF id for streaming paraformer
            try:
                alt_kwargs = {"model": self._model_id, "device": self._device}
                if not self._is_streaming_model:
                    alt_kwargs["disable_update"] = True
                self._model = AutoModel(**alt_kwargs)
            except Exception as e2:
                # final: explicit HF mapping for zh streaming fallback
                if self._model_id == "paraformer-zh-streaming":
                    hf_id = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
                    try:
                        from modelscope import snapshot_download as ms_dl
                        path = ms_dl(hf_id)
                        self._model = AutoModel(model=path, device=self._device)
                    except Exception as e3:
                        raise RuntimeError(f"FunASR download failed ms={e1} alt={e2} hf={e3}. Try: uv sync --extra funasr and check HF_ENDPOINT/hub connectivity") from e3
                else:
                    raise RuntimeError(f"FunASR download failed ms={e1} alt={e2}. Model={self._model_id}") from e2

    def transcribe_stream(self, chunk: AudioChunk) -> Iterator[Segment]:
        self._ensure_model()
        pcm_f32 = _pcm_bytes_to_float32(chunk.pcm)
        if pcm_f32.size == 0:
            return
        if self._is_streaming_model:
            # Streaming Paraformer: feed 16 kHz mono float32, cache protocol
            try:
                result = self._model.generate(
                    input=pcm_f32,
                    cache=self._cache,
                    is_final=False,
                    chunk_size=self._chunk_size,
                    encoder_chunk_look_back=self._enc_look_back,
                    decoder_chunk_look_back=self._dec_look_back,
                )
            except TypeError:
                # Older funasr may not have chunk_size kwargs at top level
                result = self._model.generate(input=pcm_f32, cache=self._cache, is_final=False)
            text = self._extract_text(result)
            if text:
                # Paraformer streaming emits incremental finals; mark partial until finish()
                # We emit as partial hypothesis so UI can preview; actual final flushed in finish().
                yield Segment(status="partial", source_text=text,
                              source_language=self._language)
        else:
            # SenseVoice (batch per chunk) - VAD flush style: emit final when non-empty
            # SenseVoice supports language auto-detect; we pass raw chunk
            try:
                result = self._model.generate(input=pcm_f32, language=self._language if self._language != "auto" else "auto",
                                              use_itn=True)
            except TypeError:
                result = self._model.generate(input=pcm_f32)
            text = self._extract_text(result)
            if text and text.strip():
                yield Segment(status="final", source_text=text.strip(),
                              source_language=self._language)

    def transcribe(self, chunk: AudioChunk) -> Optional[Segment]:
        # Batch compat: route through transcribe_stream and pick last final
        last = None
        for seg in self.transcribe_stream(chunk):
            if seg.status == "final":
                last = seg
        return last

    def finish(self) -> Iterator[Segment]:
        """Flush streaming cache tail (Paraformer) ; no-op for SenseVoice."""
        if not self._is_streaming_model or self._model is None or not self._cache:
            return
        try:
            result = self._model.generate(
                input=np.zeros(0, dtype=np.float32),
                cache=self._cache,
                is_final=True,
                chunk_size=self._chunk_size,
                encoder_chunk_look_back=self._enc_look_back,
                decoder_chunk_look_back=self._dec_look_back,
            )
        except TypeError:
            try:
                result = self._model.generate(input=np.zeros(0, dtype=np.float32), cache=self._cache, is_final=True)
            except Exception:
                return
        except Exception:
            return
        text = self._extract_text(result)
        if text and text.strip():
            yield Segment(status="final", source_text=text.strip(),
                          source_language=self._language)
        self._cache = {}

    @staticmethod
    def _extract_text(result) -> str:
        if not result:
            return ""
        # FunASR returns list[dict] where dict has "text" or "sentence"
        try:
            if isinstance(result, list) and len(result) > 0:
                first = result[0]
                if isinstance(first, dict):
                    return str(first.get("text") or first.get("sentence") or first.get("transcript") or "")
                if isinstance(first, str):
                    return first
            if isinstance(result, dict):
                return str(result.get("text") or result.get("sentence") or "")
            if isinstance(result, str):
                return result
        except Exception:
            return ""
        return ""
