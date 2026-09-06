"""CPU VAD gate — Silero via sherpa-onnx with energy fallback.

Prevents silence from entering expensive ASR inference. Must be:
- CPU-only (never GPU)
- Low-latency (~0.6ms per 30ms frame)
- Authoritative: silence never reaches ASR
- Instrumented: exposes decision + overhead

Uses sherpa-onnx VoiceActivityDetector when model available,
falls back to RMS energy threshold when unavailable (keeps pipeline functional
without sherpa extra). Model path probing:
  1) packaged silero_vad.onnx (downloaded via tools)
  2) faster_whisper assets silero_vad_v6.onnx (already in .venv)
  3) energy fallback
"""
from __future__ import annotations
import os
import time
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import pcm_bytes_to_float32, pcm_energy

# Probe for sherpa VAD model
def _probe_silero_model() -> Optional[str]:
    candidates = []
    # 1. repo models/silero_vad.onnx (if user downloaded)
    try:
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[1]
        candidates.append(str(repo / "models" / "silero_vad.onnx"))
        candidates.append(str(repo / "models" / "silero-vad" / "silero_vad.onnx"))
    except Exception:
        pass
    # 2. faster-whisper bundled v6
    try:
        import faster_whisper  # type: ignore
        import pathlib as _pl
        fw = _pl.Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"
        candidates.append(str(fw))
    except Exception:
        pass
    # 3. site-packages alternative name
    try:
        import site, os as _os
        for sp in site.getsitepackages():
            p = os.path.join(sp, "faster_whisper", "assets", "silero_vad_v6.onnx")
            candidates.append(p)
    except Exception:
        pass
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None

@dataclass
class VadDecision:
    is_speech: bool
    energy: float
    vad_score: float | None = None  # sherpa score 0..1 when available
    overhead_ms: float = 0.0

class VadGate:
    """CPU VAD gate before every ASR engine.

    Thread-safe for single producer (capture) -> gate check.
    When sherpa model available, uses Silero window_size 512 @16k (32ms)
    with threshold/min_speech_duration tuning for low-latency onset.
    Otherwise falls back to RMS energy (threshold 20 + higher 300 for speech).

    The gate is intentionally slightly permissive to preserve speech onset:
    min_speech_duration 0.1s, min_silence 0.3s, hangover 0.2s.

    Instrument: call_count, speech_count, skipped_count, total_overhead_ms
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 threshold: float = 0.5,
                 min_silence_duration: float = 0.3,
                 min_speech_duration: float = 0.1,
                 window_size: int = 512,
                 energy_threshold: float = 80.0,
                 enabled: bool = True):
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.energy_threshold = energy_threshold
        self._lock = threading.Lock()
        self._vad: Optional[object] = None
        self._use_sherpa = False
        self._hangover_frames = 0
        self._hangover_max = max(1, int(0.2 * sample_rate / window_size))  # 0.2s hangover ~6 frames @512
        # metrics
        self.call_count = 0
        self.speech_count = 0
        self.skipped_count = 0
        self.total_overhead_ms = 0.0
        self._model_path = _probe_silero_model()
        # TEMP: force energy fallback — sherpa_onnx 1.13.6 bundles ORT 1.17.1 and
        # segfaults on faster_whisper silero_vad_v6.onnx (API 27). Pip ORT is 1.29
        # but sherpa ignores it. Keep VAD functional via RMS energy until sherpa
        # upgrades bundled ORT. All pipeline paths use VadGate so bench can't crash.
        if enabled and self._model_path:
            self._vad = None
            self._use_sherpa = False
        elif enabled:
            # no model -> energy fallback will be used
            self._use_sherpa = False

    def is_speech(self, pcm: bytes) -> VadDecision:
        t0 = time.perf_counter()
        energy = pcm_energy(pcm)
        decision = False
        score = None
        if not self.enabled:
            decision = True
        elif self._use_sherpa and self._vad is not None:
            try:
                import numpy as np
                samples = pcm_bytes_to_float32(pcm)
                # sherpa VAD is stateful: accept_waveform tracks segments
                # For per-chunk gate we use is_speech_detected after accept
                self._vad.accept_waveform(samples)
                # is_speech_detected True when currently in speech
                # Also check empty() -> if we have queued speech segments, count as speech
                if self._vad.is_speech_detected():
                    decision = True
                    self._hangover_frames = self._hangover_max
                else:
                    # hangover: keep speech for a few frames after detection ends
                    if not self._vad.empty():
                        decision = True
                        self._hangover_frames = self._hangover_max
                    elif self._hangover_frames > 0:
                        decision = True
                        self._hangover_frames -= 1
                    else:
                        decision = False
                # energy override: if very loud, force speech even if VAD says silence
                # prevents missing quiet onset
                if energy > 400:
                    decision = True
                # if very quiet, force silence even if VAD hallucinates
                if energy < 15 and not decision:
                    decision = False
            except Exception:
                # fallback to energy on any VAD error
                decision = energy >= self.energy_threshold
        else:
            # WP-1.5 FIX: apply same 0.2s hangover to energy fallback (was dead code in sherpa branch only)
            raw_speech = energy >= self.energy_threshold
            if raw_speech:
                self._hangover_frames = self._hangover_max
                decision = True
            elif self._hangover_frames > 0:
                decision = True
                self._hangover_frames -= 1
            else:
                decision = False

        overhead = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self.call_count += 1
            self.total_overhead_ms += overhead
            if decision:
                self.speech_count += 1
            else:
                self.skipped_count += 1
        return VadDecision(is_speech=decision, energy=energy, vad_score=score, overhead_ms=overhead)

    def stats(self) -> dict:
        with self._lock:
            avg = self.total_overhead_ms / max(1, self.call_count)
            return {
                "calls": self.call_count,
                "speech": self.speech_count,
                "skipped": self.skipped_count,
                "avg_overhead_ms": avg,
                "total_overhead_ms": self.total_overhead_ms,
                "use_sherpa": self._use_sherpa,
                "model": self._model_path or "energy-fallback",
            }

    def reset(self):
        with self._lock:
            if self._vad is not None:
                try:
                    self._vad.reset()
                except Exception:
                    pass
            self._hangover_frames = 0
