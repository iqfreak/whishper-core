"""In-process streaming Whisper transcriber.

Two modes, both GPU-friendly with `small`:
  - task=\"translate\": source speech -> English in ONE pass (our chosen path).
  - backend=\"whisper_streaming\": partial + final captions via sliding-window
    inference (best live UX). Falls back to batch faster-whisper if the
    streaming lib is not installed.

Model is cached in the registry (candidate C fix): loaded once per
(model_size, device), reused across the session. No HTTP, no per-call reload.

P0: capture blocks are 20 ms, but Whisper still needs >=1.0 s inference
window. Small 20 ms blocks are buffered into 1.0 s non-overlapping chunks
so capture stays at 10 ms avg fill while Whisper keeps quality.
"""
from .base import StreamingTranscriber  # noqa: F401
from ..types import AudioChunk, Segment, pcm_bytes_to_float32

import os
import sys
import threading
import numpy as np

_DLL_DIRS_REGISTERED = False


def _add_cuda_dll_dirs() -> None:
    """Make pip-wheel CUDA DLLs findable -- no system CUDA toolkit needed.

    nvidia-cublas-cu12 / nvidia-cuda-runtime-cu12 / nvidia-cudnn-cu12 install
    cuBLAS/cuDNN/cudart DLLs into site-packages/nvidia/*/bin, but Windows only
    finds them via PATH or os.add_dll_directory. Registering them here turns a
    bare `uv run python -m voicelang_core.run` into a CUDA-capable run.
    """
    global _DLL_DIRS_REGISTERED
    if os.name != "nt":
        return
    if _DLL_DIRS_REGISTERED:
        return
    candidates: list[str] = []
    if getattr(sys, "frozen", False):
        # PyInstaller one-dir: _MEIPASS is the _internal dir, but be robust --
        # probe _MEIPASS, exe dir, and exe/_internal sibling (all observed layouts).
        meipass = getattr(sys, "_MEIPASS", "")
        exe_dir = os.path.dirname(sys.executable)
        if meipass:
            candidates.append(meipass)
            # In some layouts _MEIPASS == exe_dir; also check sibling _internal
            candidates.append(os.path.join(meipass, "_internal"))
        candidates.append(exe_dir)
        candidates.append(os.path.join(exe_dir, "_internal"))
        # Also check parent of exe_dir (when _MEIPASS is nested)
        candidates.append(os.path.dirname(exe_dir))
    else:
        import site

        try:
            candidates.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            candidates.append(site.getusersitepackages())
        except Exception:
            pass
        # Also probe the venv's site-packages directly when site module is odd
        for p in sys.path:
            if "site-packages" in p and p not in candidates:
                candidates.append(p)
    seen: set[str] = set()
    for sp in candidates:
        if not sp or sp in seen:
            continue
        seen.add(sp)
        nv = os.path.join(sp, "nvidia")
        if not os.path.isdir(nv):
            continue
        try:
            pkgs = os.listdir(nv)
        except OSError:
            continue
        for pkg in pkgs:
            bin_dir = os.path.join(nv, pkg, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except OSError:
                    pass
                # Also prepend to PATH as fallback for delay-load resolution
                try:
                    if bin_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                except Exception:
                    pass
    _DLL_DIRS_REGISTERED = True


def _resolve_device(device: str) -> str:
    """Resolve \"auto\" to cuda/cpu by actually PROBING CUDA, not counting it.

    get_cuda_device_count() can report >=1 while CUDA is unusable -- on this
    box it once did, and the real failure (``cublas64_12.dll is not found``)
    only surfaced mid-transcription. Counting devices is a false positive;
    loading a throwaway tensor exercises the same code path the encoder uses.
    """
    if device != "auto":
        return device
    _add_cuda_dll_dirs()
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            # Real probe: a tiny storage+copy touches cuBLAS/cuDNN for real.
            # Enum member is lowercase (\"cuda\") on ct2 >= 4.8, uppercase
            # (\"CUDA\") before -- accept both so the probe never dies on the
            # constant itself (that would silently force CPU forever).
            cuda_enum = getattr(ctranslate2.Device, "cuda", None) or getattr(
                ctranslate2.Device, "CUDA", None
            )
            if cuda_enum is not None:
                _ = ctranslate2.StorageView.from_array(
                    np.zeros((1, 1), dtype=np.float32)
                ).to_device(cuda_enum)
                return "cuda"
    except Exception:  # noqa: BLE001 -- any CUDA failure -> CPU
        pass
    return "cpu"


class WhisperStreamingTranscriber(StreamingTranscriber):
    # P0: 20 ms capture blocks -> Whisper still needs >=1.0 s window
    # Buffer small blocks into 1.0 s non-overlapping chunks.
    _WHISPER_MIN_BYTES = 32000  # 16000 samples *2 @16k mono = 1.0 s

    def __init__(self, model_size: str = "small", device: str = "auto",
                 compute_type: str = "auto", cpu_threads: int = 4,
                 translate: bool = True):
        """Streaming Whisper behind the Transcriber port.

        device=\"auto\" (default) picks CUDA when a working device is present and
        falls back to CPU; compute_type=\"auto\" picks a type the device actually
        supports. The old defaults (device=\"cuda\", compute_type=\"fp16\") crashed
        on machines with no usable CUDA *and* on int8-converted model repos --
        and run.py inherits these defaults, so a bad default means the shipped
        command fails out of the box.
        """
        self._translate_task = translate
        self._requested_device = device
        self._model_size = model_size
        self._cpu_threads = cpu_threads
        self._requested_compute = compute_type
        _add_cuda_dll_dirs()  # explicit device=\"cuda\" needs the DLLs too
        device = _resolve_device(device)
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        key = (model_size, device, compute_type)
        # Try CUDA first; if that fails due to missing DLL, fall back to CPU once.
        try:
            with WhisperStreamingTranscriber._registry_lock:
                model = WhisperStreamingTranscriber._registry.get(key)
                if model is None:
                    from faster_whisper import WhisperModel
                    model = WhisperModel(model_size, device=device,
                                         compute_type=compute_type, cpu_threads=cpu_threads)
                    WhisperStreamingTranscriber._registry[key] = model
            self._model = model
            self._device = device
            self._compute_type = compute_type
        except Exception as exc:
            msg = str(exc).lower()
            # WP-1.2 FIX: extend to OOM patterns
            is_cuda_err = any(k in msg for k in ("cublas", "cudart", "cudnn", "cuda", "not found or cannot be loaded", "out of memory", "cudnn_status_alloc_failed", "cudaerrormemoryallocation", "memory allocation", "oom when allocating", "alloc"))
            if device == "cuda" and is_cuda_err:
                # Re-probe may have been stale; ensure DLL dirs are registered then retry on CPU
                _add_cuda_dll_dirs()
                fallback_compute = "int8"
                fallback_key = (model_size, "cpu", fallback_compute)
                with WhisperStreamingTranscriber._registry_lock:
                    model = WhisperStreamingTranscriber._registry.get(fallback_key)
                    if model is None:
                        from faster_whisper import WhisperModel
                        model = WhisperModel(model_size, device="cpu",
                                             compute_type=fallback_compute, cpu_threads=cpu_threads)
                        WhisperStreamingTranscriber._registry[fallback_key] = model
                self._model = model
                self._device = "cpu"
                self._compute_type = fallback_compute
            else:
                raise
        # P0: buffer for 20 ms -> 1.0 s aggregation
        self._buf = bytearray()
        self._buf_lock = threading.Lock()
        self._window_start_ts: float | None = None  # P1: speech_onset vs window_complete anchoring
        self._speech_onset_ts: float | None = None
        self._last_speech_ts: float | None = None

    _registry: dict = {}
    _registry_lock = threading.Lock()

    def transcribe(self, chunk: AudioChunk) -> "Segment | None":
        # Batch flush used by Pipeline.run(); returns the first finalized segment.
        for seg in self.transcribe_stream(chunk):
            return seg
        return None

    def warmup(self, duration_s: float = 0.6):
        """Prime CUDA context + model kernels so first real chunk is ~130ms not 600ms."""
        try:
            audio = np.zeros(int(16000 * duration_s), dtype=np.float32)
            task = "translate" if self._translate_task else "transcribe"
            segs, _ = self._model.transcribe(audio, beam_size=1, task=task,
                                             language=None, vad_filter=False,
                                             condition_on_previous_text=False,
                                             without_timestamps=True)
            list(segs)
        except Exception:
            pass

    def warmup_all_shapes(self, durations=(0.5, 1.0, 2.0)):
        """P1/P3: per-shape warmup — cuBLAS heuristics cache per input length."""
        for d in durations:
            self.warmup(float(d))

    def _transcribe_audio(self, audio: np.ndarray):
        task = "translate" if self._translate_task else "transcribe"
        try:
            segments, info = self._model.transcribe(
                audio, beam_size=1, task=task, language=None,
                vad_filter=False, condition_on_previous_text=False,
                without_timestamps=True,
            )
        except Exception as exc:
            msg = str(exc).lower()
            is_cuda_err = any(k in msg for k in ("cublas", "cudart", "cudnn", "cannot be loaded"))
            if getattr(self, "_device", "") == "cuda" and is_cuda_err:
                # Hot fallback: rebuild on CPU and retry once
                _add_cuda_dll_dirs()
                fallback_compute = "int8"
                fallback_key = (self._model_size, "cpu", fallback_compute)
                with WhisperStreamingTranscriber._registry_lock:
                    model = WhisperStreamingTranscriber._registry.get(fallback_key)
                    if model is None:
                        from faster_whisper import WhisperModel
                        model = WhisperModel(self._model_size, device="cpu",
                                             compute_type=fallback_compute, cpu_threads=self._cpu_threads)
                        WhisperStreamingTranscriber._registry[fallback_key] = model
                self._model = model
                self._device = "cpu"
                self._compute_type = fallback_compute
                segments, info = self._model.transcribe(
                    audio, beam_size=1, task=task, language=None,
                    vad_filter=False, condition_on_previous_text=False,
                    without_timestamps=True,
                )
            else:
                raise
        for s in segments:
            text = s.text.strip()
            if text:
                yield Segment(status="final", source_text=text,
                              timestamp=s.end, source_language=info.language)

    def transcribe_stream(self, chunk: AudioChunk, *, beam_size: int = 1,
                          without_timestamps: bool = True):
        """Fastest-latency streaming path for 4050 + gaming.

        P0: 20 ms capture blocks are buffered into >=1.0 s inference chunks
        for Whisper quality. Streaming engines (moonshine/nemotron) should
        bypass Whisper entirely — this buffer keeps Whisper viable when
        selected but not the gaming hot path.

        P1 anchoring: each 1.0 s window records window_start_ts (first block
        in buffer) and a VAD-derived speech_onset_ts is kept on the pipeline
        side; the yielded Segment is annotated with _window_start_ts and
        _capture_ts so metrics can report both window_complete→partial and
        speech_onset→partial separately.
        """
        # Accumulate small capture blocks; only transcribe when we have >=1.0 s.
        # This keeps avg capture fill at ~10 ms (20 ms block) while preserving
        # Whisper's >=1.0 s quality window.
        import time as _time
        with self._buf_lock:
            if self._window_start_ts is None and hasattr(chunk, "capture_ts"):
                self._window_start_ts = float(chunk.capture_ts)
            elif self._window_start_ts is None:
                self._window_start_ts = _time.monotonic()
            self._buf.extend(chunk.pcm)
            if len(self._buf) < self._WHISPER_MIN_BYTES:
                return
            # Take exactly 1.0 s (non-overlapping) for efficient batch
            pcm_bytes = bytes(self._buf[:self._WHISPER_MIN_BYTES])
            del self._buf[:self._WHISPER_MIN_BYTES]
            window_start = self._window_start_ts
            # Next window starts at this chunk's ts (or now if missing)
            self._window_start_ts = float(getattr(chunk, "capture_ts", _time.monotonic()))
        window_complete_ts = _time.monotonic()
        audio = pcm_bytes_to_float32(pcm_bytes)
        # Use shared helper for actual inference (keeps fallback logic single-source)
        # Note: beam_size/without_timestamps forwarded from caller but we pin to
        # known-good gaming defaults for the quality tier.
        for seg in self._transcribe_audio(audio):
            # Annotate for P1 latency split
            try:
                seg._window_start_ts = window_start  # type: ignore[attr-defined]
                seg._window_complete_ts = window_complete_ts  # type: ignore[attr-defined]
                seg._capture_ts = float(getattr(chunk, "capture_ts", window_complete_ts))  # type: ignore[attr-defined]
            except Exception:
                pass
            yield seg

    def finish(self):
        """Flush any buffered tail (file replay). Zero-pad 0.16–1.0 s instead of dropping."""
        with self._buf_lock:
            if not self._buf:
                return
            pcm_bytes = bytes(self._buf)
            window_start = self._window_start_ts
            self._buf.clear()
            self._window_start_ts = None
        # 0.16 s = 5120 bytes keeps short callouts ("behind you")
        if len(pcm_bytes) < 5120:
            return
        if len(pcm_bytes) < self._WHISPER_MIN_BYTES:
            # pad with silence to 1.0 s to keep model happy
            pcm_bytes = pcm_bytes + b"\x00\x00" * ((self._WHISPER_MIN_BYTES - len(pcm_bytes)) // 2)
        audio = pcm_bytes_to_float32(pcm_bytes)
        import time as _time
        window_complete_ts = _time.monotonic()
        for seg in self._transcribe_audio(audio):
            try:
                seg._window_start_ts = window_start  # type: ignore[attr-defined]
                seg._window_complete_ts = window_complete_ts  # type: ignore[attr-defined]
                seg._capture_ts = window_complete_ts  # type: ignore[attr-defined]
            except Exception:
                pass
            yield seg
