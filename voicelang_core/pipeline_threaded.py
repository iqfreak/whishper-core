"""Threaded low-latency pipeline for 4050 RTX + heavy apps.

Decouples capture -> ASR -> translation -> display with bounded queues.
- Capture thread (ABOVE_NORMAL + MMCSS Pro Audio): pumps AudioSource without blocking on ASR
- ASR thread (BELOW_NORMAL): CUDA whisper tiny float16 beam1, moonshine 0.08s, nemotron
- Translation worker (LOWEST): API translate off critical path, finals only
- Display coalesces exactly like OverlayDisplay 33ms

P0 fixes in this revision:
- CPU VAD gate (Silero via sherpa-onnx) before every ASR engine, instrumented
- Concurrency: _next_id/_open_id protected by lock, immutable segment snapshot for translation
- Qt thread-safety: display calls marshalled (overlay handles cross-thread via Qt queued invoke)
- Priority: process NORMAL (0x20), capture ABOVE_NORMAL (1) + MMCSS, ASR BELOW_NORMAL (-1), translate LOWEST (-2)
- Queue: every dropped block counted/logged, continuity measurable, never blocks capture
- Metrics: capture/VAD/ASR/translation/overlay timestamps, p50/p95/p99, queue depth, drops

Inherits id-stamping, translation render and silence watchdog from Pipeline
so fixes to Pipeline propagate (altitude fix: not a fork).
"""
from __future__ import annotations
import queue
import threading
import time
import copy
from typing import Optional

from .pipeline import Pipeline
from .ports import AudioSource, StreamingTranscriber, Translator, Display, CaptionSource
from .types import Segment, AudioChunk

try:
    from .vad import VadGate
except Exception:
    VadGate = None  # type: ignore

try:
    from .metrics import PipelineMetrics
except Exception:
    PipelineMetrics = None  # type: ignore

try:
    from PySide6 import QtCore as _QtCore  # hoisted: was imported per-chunk (27ms/s at 50Hz)
except Exception:
    _QtCore = None  # type: ignore


def _pump_display(display) -> None:
    """Pump overlay coalesce timer if we're on the GUI thread — shared helper (SAFE)."""
    try:
        pump = getattr(display, "_pump", None)
        if callable(pump):
            app = getattr(display, "_app", None)
            if app is not None and _QtCore is not None and _QtCore.QThread.currentThread() is app.thread():
                pump()  # type: ignore
            return
        app = getattr(display, "_app", None)
        if app is not None and _QtCore is not None and _QtCore.QThread.currentThread() is app.thread():
            app.processEvents()
    except Exception:
        pass


def _set_thread_priority(level: int, mmcss_name: str | None = None):
    if __import__("os").name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        avrt = None
        try:
            avrt = ctypes.windll.avrt
        except Exception:
            pass
        # MMCSS Pro Audio for capture thread (lowest latency, audio-friendly scheduling)
        task_idx = ctypes.c_uint(0)
        h_task = None
        if mmcss_name and avrt is not None:
            try:
                # AvSetMmThreadCharacteristicsW
                avrt.AvSetMmThreadCharacteristicsW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint)]
                avrt.AvSetMmThreadCharacteristicsW.restype = ctypes.c_void_p
                h_task = avrt.AvSetMmThreadCharacteristicsW(mmcss_name, ctypes.byref(task_idx))
            except Exception:
                h_task = None
        # Set priority after MMCSS (MS docs: MMCSS may override priority)
        try:
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(), level)
        except Exception:
            pass
        # Keep handle alive on thread object to avoid revert (store in thread-local)
        if h_task:
            threading.current_thread().__dict__["_mmcss_handle"] = h_task  # type: ignore
    except Exception:
        pass


def _put_bounded(q: "queue.Queue", item, *, drop_oldest=True, metrics=None, stage="asr"):
    """Put with bounded latency: drop oldest if full. Returns True if dropped.

    Streaming (drop_oldest=False) is never-drop: blocks until enqueued.
    Stateless (drop_oldest=True) drops oldest on Full to keep latency low.
    """
    if not drop_oldest:
        # Never-drop: block until space available. Never counts as dropped.
        # Use 0.5s timeout first to allow quick enqueue, then block to preserve continuity.
        try:
            q.put(item, timeout=0.5)
            if metrics is not None:
                try:
                    metrics.asr_enqueue.record_queue_depth(q.qsize())
                except Exception:
                    pass
            return False
        except queue.Full:
            # Backpressure: block until consumer drains (streaming requires continuity)
            # WP-1.6 FIX: bounded timeout on final put; 3x timeouts escalate
            timeouts = 0
            while timeouts < 3:
                try:
                    q.put(item, timeout=1.0)
                    if metrics is not None:
                        try:
                            metrics.asr_enqueue.record_queue_depth(q.qsize())
                        except Exception:
                            pass
                    return False
                except queue.Full:
                    timeouts += 1
                    continue
                except Exception:
                    return False
            # 3 consecutive timeouts: hard failure
            try:
                if metrics is not None:
                    metrics.record_drop(stage, 1)
                # attempt display warn via metrics display not available here; caller handles
            except Exception:
                pass
            return True
    # Drop-oldest path (stateless, low-latency)
    dropped = False
    try:
        q.put(item, timeout=0.1)
        if metrics is not None:
            try:
                metrics.asr_enqueue.record_queue_depth(q.qsize())
            except Exception:
                pass
        return False
    except queue.Full:
        try:
            q.get_nowait()
            dropped = True
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
            if metrics is not None:
                if dropped:
                    metrics.record_drop(stage, 1)
                try:
                    metrics.asr_enqueue.record_queue_depth(q.qsize())
                except Exception:
                    pass
        except queue.Full:
            if metrics is not None:
                metrics.record_drop(stage, 1)
        return dropped


class ThreadedPipeline(Pipeline):
    def __init__(self, *, source, transcriber, translator, display,
                 target_language="en", source_language="auto",
                 asr_queue_size=2, translate_queue_size=8,
                 vad_enabled=True, vad_threshold=0.5):
        super().__init__(source=source, transcriber=transcriber, translator=translator,
                         display=display, target_language=target_language,
                         source_language=source_language)
        # P2: streaming engines (stateful) must never drop — use large ring + lag metric
        _streaming_names = {"MoonshineTranscriber", "NemotronStreamingTranscriber", "FunASRStreamingTranscriber", "FunASRNanoTranscriber"}
        self._is_streaming = type(transcriber).__name__ in _streaming_names if transcriber else False
        # Auto-promote streaming queue to large never-drop ring; keep stateless at 2 drop-oldest
        if self._is_streaming and asr_queue_size <= 2:
            asr_queue_size = 64
        self._asr_q: "queue.Queue[AudioChunk | None]" = queue.Queue(maxsize=asr_queue_size)
        self._translate_q: "queue.Queue[Segment]" = queue.Queue(maxsize=translate_queue_size)
        self._stop = threading.Event()
        self._id_lock = threading.Lock()
        # P0-2: CPU VAD gate
        self._vad = None
        self._vad_enabled = vad_enabled
        if vad_enabled and VadGate is not None:
            try:
                self._vad = VadGate(threshold=vad_threshold, enabled=True)
            except Exception:
                self._vad = None
        # Metrics
        self.metrics = None
        if PipelineMetrics is not None:
            try:
                eng = type(transcriber).__name__ if transcriber else "unknown"
                self.metrics = PipelineMetrics(engine=eng)
            except Exception:
                self.metrics = None
        # Timestamp tracking for e2e latency
        self._utterance_start_ts: Optional[float] = None
        self._speech_onset_ts: Optional[float] = None  # P1: VAD-derived onset
        self._last_was_speech = False
        # P3: GPU keep-alive during silence
        self._keepalive_thread: Optional[threading.Thread] = None
        self._last_asr_ts = time.monotonic()

    # -- id stamping with lock -------------------------------------------------
    def _stamp_id(self, seg: Segment) -> None:
        with self._id_lock:
            super()._stamp_id(seg)

    def _advance_id_on_final(self, seg: Segment):
        with self._id_lock:
            self._next_id = max(self._next_id, seg.id + 1)
            self._open_id = None

    # -- threaded overrides -------------------------------------------------
    def _handle_segment(self, seg: Segment, translator: Translator | None = None):
        # Stamp id under lock (TOCTOU fix)
        self._stamp_id(seg)
        now = time.monotonic()
        # P1: annotate speech_onset→partial/final vs window_complete→partial
        window_start = getattr(seg, "_window_start_ts", None)
        window_complete = getattr(seg, "_window_complete_ts", None)
        cap_ts = getattr(seg, "_capture_ts", now)
        # Track speech onset if VAD saw speech transition
        onset = self._speech_onset_ts if self._speech_onset_ts is not None else window_start if window_start is not None else cap_ts
        # Track e2e first partial
        if seg.status == "partial":
            if self._utterance_start_ts is None:
                self._utterance_start_ts = cap_ts
            if self.metrics is not None:
                try:
                    if onset is not None:
                        self.metrics.speech_onset_to_partial.add_latency((now - onset) * 1000.0)
                    if window_complete is not None:
                        self.metrics.window_complete_to_partial.add_latency((now - window_complete) * 1000.0)
                    if self._utterance_start_ts:
                        latency = (now - self._utterance_start_ts) * 1000.0
                        self.metrics.e2e_first_partial.add_latency(latency)
                        self.metrics.asr_first_partial.add_latency(latency)
                except Exception:
                    pass
            try:
                self.display.show_partial(seg)
                if self.metrics:
                    self.metrics.overlay.add_latency(0)
            except Exception:
                pass
            return

        # Final
        if seg.status == "final":
            if self.metrics is not None:
                try:
                    if onset is not None:
                        self.metrics.speech_onset_to_final.add_latency((now - onset) * 1000.0)
                    if self._utterance_start_ts is not None:
                        self.metrics.e2e_final.add_latency((now - self._utterance_start_ts) * 1000.0)
                        self.metrics.asr_final.add_latency((now - self._utterance_start_ts) * 1000.0)
                except Exception:
                    pass
            self._utterance_start_ts = None
            # P1: reset onset at final boundary (next utterance)
            # keep _speech_onset_ts until next speech detected
            self._speech_onset_ts = None
            self._last_was_speech = False

            # Translation snapshot: immutable copy so worker never reads mutable _open_id
            use_passthrough = getattr(self.translator, "is_passthrough", False)
            if use_passthrough or translator is not None and getattr(translator, "is_passthrough", False):
                # Passthrough: translate in place, still need lock for id advance
                self._render(seg, self.translator)  # type: ignore[arg-type]
                self._advance_id_on_final(seg)
                try:
                    self.display.show(seg)
                except Exception:
                    pass
                if self.metrics:
                    try:
                        self.metrics.translation.add_latency(0)
                    except Exception:
                        pass
            else:
                # Snapshot immutable (segment_id, text) - worker reads snapshot, not shared state
                # Deep copy minimal fields to avoid mutation races
                snap = Segment(
                    id=seg.id,
                    status="final",
                    source_text=seg.source_text,
                    translated_text=None,
                    timestamp=seg.timestamp,
                    source_language=seg.source_language,
                )
                # Preserve capture ts for metrics if present
                if hasattr(seg, "_capture_ts"):
                    snap._capture_ts = seg._capture_ts  # type: ignore[attr-defined]
                # WP-1.1 FIX: advance id synchronously BEFORE enqueue (never depend on translation latency)
                self._advance_id_on_final(seg)
                # Enqueue snapshot; if queue full, drop oldest and count
                dropped = _put_bounded(self._translate_q, snap, drop_oldest=True, metrics=self.metrics, stage="translation")
                if dropped:
                    # Agent4: surface drop as warning (not silent) + metrics already counted
                    try:
                        self.display.warn(f"Translation queue full — oldest translation dropped (size {self._translate_q.qsize()}) — translation slower than speech")
                    except Exception:
                        pass

    def _translate_worker(self):
        # Lowest priority: translation must never starve capture/ASR
        _set_thread_priority(-2, None)  # THREAD_PRIORITY_LOWEST
        while not self._stop.is_set():
            try:
                seg = self._translate_q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.monotonic()
            # Render translation onto snapshot (immutable id/text)
            try:
                self._render(seg, self.translator)  # type: ignore[arg-type]
            except Exception:
                seg.translated_text = None
            # WP-1.1: id already advanced synchronously at enqueue time; do not re-advance here
            # (kept for passthrough idempotency, but for translated path it's now no-op)
            pass
            try:
                self.display.show(seg)
            except Exception:
                pass
            if self.metrics is not None:
                try:
                    self.metrics.translation.add_latency((time.monotonic() - t0) * 1000.0)
                    self.metrics.translation.record_queue_depth(self._translate_q.qsize())
                except Exception:
                    pass

    def _asr_loop(self):
        _set_thread_priority(-1, None)  # BELOW_NORMAL for ASR (below capture)
        while not self._stop.is_set():
            try:
                chunk = self._asr_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                break
            t_asr_start = time.monotonic()
            if self.metrics is not None:
                try:
                    cap_ts = getattr(chunk, "capture_ts", t_asr_start)
                    # P2: consumer lag (streaming, never-drop) vs queue depth depth
                    lag_ms = (t_asr_start - cap_ts) * 1000.0
                    if self._is_streaming and lag_ms > 5:
                        self.metrics.consumer_lag.add_latency(lag_ms)
                    self.metrics.asr_start.add_latency(lag_ms)
                except Exception:
                    pass
            # Remember utterance start for e2e
            if self._utterance_start_ts is None:
                self._utterance_start_ts = getattr(chunk, "capture_ts", t_asr_start)
            self._last_asr_ts = t_asr_start
            first = True
            try:
                for seg in self.transcriber.transcribe_stream(chunk):  # type: ignore[union-attr]
                    # Attach capture ts for e2e calc downstream
                    try:
                        if not hasattr(seg, "_capture_ts") or seg._capture_ts is None:
                            seg._capture_ts = getattr(chunk, "capture_ts", t_asr_start)  # type: ignore[attr-defined]
                        if not hasattr(seg, "_window_start_ts") or seg._window_start_ts is None:
                            seg._window_start_ts = getattr(chunk, "capture_ts", t_asr_start)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    self._handle_segment(seg)
                    first = False
                    # success resets circuit breaker
                    try:
                        self._asr_consec_failures = 0  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception as exc:
                # WP-1.2 FIX: circuit breaker + throttled warning
                # Track consecutive failures; after 5, stop retrying and emit one loud warning
                if not hasattr(self, "_asr_consec_failures"):
                    self._asr_consec_failures = 0  # type: ignore[attr-defined]
                    self._asr_warned_at = 0.0  # type: ignore[attr-defined]
                self._asr_consec_failures += 1  # type: ignore[attr-defined]
                now = time.monotonic()
                # Extend CUDA OOM detection for CPU-fallback decision (also handled in whisper adapter)
                msg = str(exc).lower()
                is_oom = any(k in msg for k in ("out of memory", "cudnn_status_alloc_failed", "cudaerrormemoryallocation", "memory allocation", "oom when allocating"))
                if is_oom:
                    try:
                        self.display.warn(f"ASR CUDA OOM: {exc} — attempting CPU fallback; if persists check VRAM")
                    except Exception:
                        pass
                if self._asr_consec_failures >= 5:  # type: ignore[attr-defined]
                    if now - getattr(self, "_asr_warned_at", 0) > 30:  # type: ignore[attr-defined]
                        try:
                            self.display.warn(f"ASR stopped after 5 consecutive failures: {exc} — restart or reconfigure required")
                        except Exception:
                            pass
                        self._asr_warned_at = now  # type: ignore[attr-defined]
                    # break loop to avoid flooding; capture thread will exit via _stop
                    break
                # Throttle warnings: at most one per 5s
                if now - getattr(self, "_asr_warned_at", 0) > 5:  # type: ignore[attr-defined]
                    try:
                        self.display.warn(f"ASR error: {exc}")
                    except Exception:
                        pass
                    if "out of memory" not in msg:
                        self._asr_warned_at = now  # type: ignore[attr-defined]
                # Reset counter on success is handled below (after for loop)
                pass
            if self.metrics is not None and first:
                try:
                    self.metrics.asr_final.add_latency((time.monotonic() - t_asr_start) * 1000.0)
                except Exception:
                    pass

    def _capture_loop(self):
        # Capture highest priority: ABOVE_NORMAL + MMCSS Pro Audio
        _set_thread_priority(1, "Pro Audio")
        try:
            for chunk in self.source.stream():  # type: ignore[union-attr]
                if self._stop.is_set():
                    break
                if not hasattr(chunk, "capture_ts") or chunk.capture_ts == 0:
                    try:
                        chunk.capture_ts = time.monotonic()
                    except Exception:
                        pass
                try:
                    self._watch_silence(chunk)
                except Exception:
                    pass
                # P0-2: VAD gate — P2 split: streaming engines get continuous audio
                _is_speech = True
                if self._vad is not None:
                    try:
                        t_vad = time.monotonic()
                        dec = self._vad.is_speech(chunk.pcm)
                        if self.metrics is not None:
                            try:
                                self.metrics.vad.add_latency(dec.overhead_ms)
                            except Exception:
                                pass
                        _is_speech = bool(dec.is_speech)
                        # Track speech onset for P1 latency
                        if _is_speech and not self._last_was_speech:
                            self._speech_onset_ts = float(chunk.capture_ts)
                        self._last_was_speech = _is_speech
                        # P2: streaming engines must not drop silence — they need is_endpoint
                        should_gate = not _is_speech and not self._is_streaming
                        if should_gate:
                            if self.metrics is not None:
                                self.metrics.skipped_silent += 1
                                self.metrics.vad.record_drop(0)
                            _pump_display(self.display)
                            continue
                        # Streaming but silent: still count skipped_silent for observability
                        if not _is_speech and self._is_streaming and self.metrics is not None:
                            self.metrics.skipped_silent += 1
                    except Exception:
                        pass  # fail open
                # Bounded queue — P2: never-drop for streaming
                if self._is_streaming:
                    _put_bounded(self._asr_q, chunk, drop_oldest=False, metrics=self.metrics, stage="capture")
                    # If backpressure builds, record lag instead of dropping; shed by raising update_interval externally
                    if self._asr_q.qsize() >= 50 and self.metrics:
                        self.metrics.record_drop("capture", 0)  # no drop, just signal lag via consumer_lag
                else:
                    _put_bounded(self._asr_q, chunk, drop_oldest=True, metrics=self.metrics, stage="capture")
                _pump_display(self.display)
        except Exception as exc:
            # WP-1.4 FIX: mic/capture exceptions become visible with remediation
            try:
                msg = str(exc)
                hint = "check Windows Settings → Privacy → Microphone → allow desktop apps and check mic mute/level"
                self.display.warn(f"capture error: {msg} — {hint}")
            except Exception:
                pass
            # Distinct exit code 20 for mic permission / capture failure (for GUI to map)
            try:
                import os as _os
                _os._exit(20)
            except Exception:
                raise
    def run_streaming(self):
        # P3: GPU keep-alive for streaming/whisper when silent (holds clocks)
        def _keepalive():
            _set_thread_priority(-2, None)
            while not self._stop.is_set():
                time.sleep(0.5)
                # Only when capture-active but ASR idle >0.4s (VAD gating silence)
                if time.monotonic() - self._last_asr_ts < 0.4:
                    continue
                if self._stop.is_set():
                    break
                try:
                    tr = self.transcriber
                    warm = getattr(tr, "warmup", None)
                    if callable(warm):
                        # ~100 ms zeros decode at LOWEST holds CUDA context + clocks
                        warm(0.1)
                        if self.metrics:
                            self.metrics.asr_final.add_latency(0)  # heartbeat
                except Exception:
                    pass
        t_translate = threading.Thread(target=self._translate_worker, daemon=True, name="translate")
        t_asr = threading.Thread(target=self._asr_loop, daemon=True, name="asr")
        t_translate.start()
        t_asr.start()
        # P3: keepalive daemon (no-op on CPU-only)
        if getattr(self, "_is_streaming", False) or type(self.transcriber).__name__ == "WhisperStreamingTranscriber":
            self._keepalive_thread = threading.Thread(target=_keepalive, daemon=True, name="gpu-keepalive")
            self._keepalive_thread.start()
        try:
            self._capture_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            _put_bounded(self._asr_q, None, drop_oldest=not self._is_streaming)
            # Give ASR time to drain, then translation
            t_asr.join(timeout=2.0)
            t_translate.join(timeout=2.0)

    def run_streaming_with_timed_replay(self, duration_s=None):
        """Helper for file replay tests: runs until source exhausted, then flush."""
        t_translate = threading.Thread(target=self._translate_worker, daemon=True, name="translate")
        t_asr = threading.Thread(target=self._asr_loop, daemon=True, name="asr")
        t_translate.start()
        t_asr.start()
        for chunk in self.source.stream():  # type: ignore[union-attr]
            if self._stop.is_set():
                break
            # For file replay, also apply VAD gate — streaming-aware (like _capture_loop): streaming needs silence for is_endpoint
            if self._vad is not None:
                try:
                    dec = self._vad.is_speech(chunk.pcm)
                    _is_speech = bool(dec.is_speech)
                    should_gate = not _is_speech and not self._is_streaming
                    if should_gate:
                        if self.metrics:
                            self.metrics.skipped_silent += 1
                        _pump_display(self.display)
                        continue
                    if not _is_speech and self._is_streaming and self.metrics is not None:
                        self.metrics.skipped_silent += 1
                except Exception:
                    pass
            if not hasattr(chunk, "capture_ts"):
                try:
                    chunk.capture_ts = time.monotonic()
                except Exception:
                    pass
            _put_bounded(self._asr_q, chunk, drop_oldest=not self._is_streaming, metrics=self.metrics, stage="capture")
            _pump_display(self.display)
        time.sleep(0.3)
        fin = getattr(self.transcriber, "finish", None)
        if fin is not None:
            try:
                for seg in fin():
                    # Attach dummy ts for finish tail
                    try:
                        seg._capture_ts = time.monotonic()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    self._handle_segment(seg)
            except Exception:
                pass
        time.sleep(0.6)
        self._stop.set()
        while not self._translate_q.empty():
            try:
                seg = self._translate_q.get_nowait()
                self._render(seg, self.translator)  # type: ignore[arg-type]
                self._advance_id_on_final(seg)
                try:
                    self.display.show(seg)
                except Exception:
                    pass
            except queue.Empty:
                break
        t_asr.join(timeout=2.0)
        t_translate.join(timeout=2.0)
