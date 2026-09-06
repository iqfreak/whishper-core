"""Nemotron-3.5-ASR-Streaming-0.6B (sherpa-onnx int8 ONNX) behind the port.

A cache-aware streaming RNNT ASR (FastConformer encoder) by NVIDIA: 40
language-locales from one model, punctuation + capitalization built in,
true incremental partials. Runs on CPU (int8) -- CUDA needs an
onnxruntime-gpu provider, out of scope here.

Model files (encoder/decoder/joiner .int8.onnx + tokens.txt) are NOT
pip-installable; download and extract them once (the GUI's Nemotron download
worker tries Hugging Face first, then this GitHub release tarball):

    curl -L -o /tmp/n.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2
    mkdir -p models && tar xjf /tmp/n.tar.bz2 -C models

Runtime:  pip install sherpa-onnx   (also: uv sync --extra nemotron)

The recognizer (model weights) is cached per (model_dir, chunk, threads);
streams are cheap and per-utterance. This mirrors the Whisper registry
pattern (no per-request reload). Registry access is guarded by a lock so
concurrent pipeline starts cannot double-load (check-then-load race).
"""

from __future__ import annotations

import os
import threading
from typing import Iterator, Optional

import numpy as np

from .base import StreamingTranscriber  # noqa: F401
from ..types import AudioChunk, Segment, pcm_bytes_to_float32

_RECOGNIZER_REGISTRY: dict = {}
_REGISTRY_LOCK = threading.Lock()


def _register_pip_onnxruntime() -> None:
    """Force sherpa-onnx to use the pip onnxruntime (newer than the bundled one).

    sherpa-onnx-core bundles onnxruntime 1.17.1, which cannot load ONNX
    models exported mid-2026 (IR/API 27). The pip onnxruntime (>=1.20) lives
    in site-packages/onnxruntime/capi; registering that dir FIRST wins DLL
    resolution. Mirrors _add_cuda_dll_dirs() in whisper_streaming.py.
    """
    if os.name != "nt":
        return
    try:
        import onnxruntime

        capi = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
        if os.path.isdir(capi):
            os.add_dll_directory(capi)
    except (ImportError, OSError):
        pass


def _build_recognizer(model_dir: str, num_threads: int):
    key = (os.path.abspath(model_dir), num_threads)
    recognizer = _RECOGNIZER_REGISTRY.get(key)
    if recognizer is None:
        # Double-checked locking: builds are slow (model load); only one
        # thread should ever pay it.
        with _REGISTRY_LOCK:
            recognizer = _RECOGNIZER_REGISTRY.get(key)
            if recognizer is None:
                _register_pip_onnxruntime()
                import sherpa_onnx  # lazy: heavy dep, not part of the core

                enc = os.path.join(model_dir, "encoder.int8.onnx")
                dec = os.path.join(model_dir, "decoder.int8.onnx")
                joi = os.path.join(model_dir, "joiner.int8.onnx")
                tok = os.path.join(model_dir, "tokens.txt")
                for f in (enc, dec, joi, tok):
                    if not os.path.isfile(f):
                        raise FileNotFoundError(
                            f"missing Nemotron model file: {f}\n"
                            "Download it (see adapter docstring) and pass --nemotron-dir."
                        )
                recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=enc,
                    decoder=dec,
                    joiner=joi,
                    tokens=tok,
                    num_threads=num_threads,
                    sample_rate=16000,
                    feature_dim=80,
                    enable_endpoint_detection=True,
                    decoding_method="greedy_search",
                )
                _RECOGNIZER_REGISTRY[key] = recognizer
    return recognizer


class NemotronStreamingTranscriber(StreamingTranscriber):
    """NVIDIA Nemotron-3.5 streaming ASR behind the StreamingTranscriber port.

    Yields a partial Segment per decode pass (changing hypothesis) and a
    final Segment whenever the endpoint detector fires (utterance end).
    """

    def __init__(self, model_dir: str, language: str = "auto",
                 num_threads: int = 4):
        self._model_dir = model_dir
        self._language = language
        self._recognizer = _build_recognizer(model_dir, num_threads)
        self._stream = self._recognizer.create_stream()
        if language and language != "auto":
            self._stream.set_option("language", language)
        self._last_partial = ""

    def _feed(self, audio: np.ndarray) -> None:
        self._stream.accept_waveform(16000, audio)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    def _partial_segment(self, text: str) -> bool:
        """Yield a partial Segment (raw hypothesis) when the text moved."""
        if text and text != self._last_partial:
            self._last_partial = text
            return True
        return False

    def transcribe_stream(self, chunk: AudioChunk) -> Iterator[Segment]:
        audio = pcm_bytes_to_float32(chunk.pcm)
        self._feed(audio)
        if self._recognizer.is_endpoint(self._stream):
            text = (self._recognizer.get_result(self._stream) or "").strip()
            if text:
                yield Segment(status="final", source_text=text,
                              source_language=None)
            # Start a fresh utterance; the hypothesis cache resets too.
            self._recognizer.reset(self._stream)
            self._last_partial = ""
        else:
            partial = (self._recognizer.get_result(self._stream) or "").strip()
            if partial and self._partial_segment(partial):
                yield Segment(status="partial", source_text=partial,
                              source_language=None)

    def transcribe(self, chunk: AudioChunk) -> Optional[Segment]:
        # Batch flush (Pipeline.run): feed everything, take the best hypothesis.
        audio = pcm_bytes_to_float32(chunk.pcm)
        self._feed(audio)
        text = (self._recognizer.get_result(self._stream) or "").strip()
        if text:
            return Segment(status="final", source_text=text,
                           source_language=None)
        return None