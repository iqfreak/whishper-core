"""Pipeline metrics — capture/VAD/ASR/translation/overlay stage timings.

Every dropped block counted and logged. No silent loss.
Provides p50/p95/p99 and queue depth tracking.
"""
from __future__ import annotations
import time
import threading
import statistics
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class StageMetrics:
    latencies_ms: List[float] = field(default_factory=list)
    count: int = 0
    drops: int = 0
    queue_depth_max: int = 0
    queue_depth_samples: List[int] = field(default_factory=list)

    def add_latency(self, ms: float):
        self.latencies_ms.append(ms)
        self.count += 1

    def record_drop(self, n: int = 1):
        self.drops += n

    def record_queue_depth(self, depth: int):
        self.queue_depth_samples.append(depth)
        if depth > self.queue_depth_max:
            self.queue_depth_max = depth

    def percentiles(self) -> Dict[str, float]:
        if not self.latencies_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        s = sorted(self.latencies_ms)
        n = len(s)
        def pct(p: float) -> float:
            k = int((p/100.0) * n)
            k = min(max(k, 0), n-1)
            return s[k]
        return {
            "p50": pct(50),
            "p95": pct(95),
            "p99": pct(99),
            "avg": sum(s)/n,
            "min": s[0],
            "max": s[-1],
            "count": n,
        }

@dataclass
class PipelineMetrics:
    """All stage metrics with thread-safe recording."""
    capture: StageMetrics = field(default_factory=StageMetrics)
    vad: StageMetrics = field(default_factory=StageMetrics)
    asr_enqueue: StageMetrics = field(default_factory=StageMetrics)
    asr_start: StageMetrics = field(default_factory=StageMetrics)
    asr_first_partial: StageMetrics = field(default_factory=StageMetrics)
    asr_final: StageMetrics = field(default_factory=StageMetrics)
    translation: StageMetrics = field(default_factory=StageMetrics)
    overlay: StageMetrics = field(default_factory=StageMetrics)
    e2e_first_partial: StageMetrics = field(default_factory=StageMetrics)
    e2e_final: StageMetrics = field(default_factory=StageMetrics)
    # P1: onset-anchored vs window-complete split; P2: streaming lag not drops
    window_complete_to_partial: StageMetrics = field(default_factory=StageMetrics)
    speech_onset_to_partial: StageMetrics = field(default_factory=StageMetrics)
    speech_onset_to_final: StageMetrics = field(default_factory=StageMetrics)
    consumer_lag: StageMetrics = field(default_factory=StageMetrics)
    dropped_blocks: int = 0
    skipped_silent: int = 0
    engine: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_drop(self, stage: str = "capture", n: int = 1):
        with self._lock:
            self.dropped_blocks += n
            # also stage
            m = getattr(self, stage, None)
            if isinstance(m, StageMetrics):
                m.record_drop(n)

    def record_skipped_silent(self, n: int = 1):
        with self._lock:
            self.skipped_silent += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "engine": self.engine,
                "dropped_blocks": self.dropped_blocks,
                "skipped_silent": self.skipped_silent,
                "capture": self.capture.percentiles(),
                "vad": self.vad.percentiles(),
                "asr_first_partial": self.asr_first_partial.percentiles(),
                "asr_final": self.asr_final.percentiles(),
                "e2e_first_partial": self.e2e_first_partial.percentiles(),
                "e2e_final": self.e2e_final.percentiles(),
                "window_complete_to_partial": self.window_complete_to_partial.percentiles(),
                "speech_onset_to_partial": self.speech_onset_to_partial.percentiles(),
                "speech_onset_to_final": self.speech_onset_to_final.percentiles(),
                "consumer_lag": self.consumer_lag.percentiles(),
                "translation": self.translation.percentiles(),
                "overlay": self.overlay.percentiles(),
                "asr_queue_max": self.asr_enqueue.queue_depth_max,
                "translate_queue_max": self.translation.queue_depth_max,
            }

    def log_summary(self):
        s = self.snapshot()
        lines = [f"Engine: {s['engine']}", f"Dropped blocks: {s['dropped_blocks']}  Skipped silent: {s['skipped_silent']}"]
        for k in ("e2e_first_partial", "e2e_final", "asr_first_partial", "asr_final", "speech_onset_to_partial", "speech_onset_to_final", "window_complete_to_partial"):
            p = s[k]
            if p["count"]:
                lines.append(f"{k}: p50={p['p50']:.1f}ms p95={p['p95']:.1f}ms p99={p['p99']:.1f}ms avg={p['avg']:.1f} n={p['count']}")
        if s["consumer_lag"]["count"]:
            p = s["consumer_lag"]
            lines.append(f"consumer_lag: p50={p['p50']:.1f}ms p95={p['p95']:.1f}ms n={p['count']} (streaming, never-drop)")
        lines.append(f"ASR queue max depth {s['asr_queue_max']}  Translate queue max {s['translate_queue_max']}")
        return "\n".join(lines)
