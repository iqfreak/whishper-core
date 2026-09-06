"""Full P0 benchmark — capture 20ms, VAD, ASR engines, threaded pipeline, overlay.
Run: PYTHONPATH=C:/Users/ahmed/whishper-core .venv/Scripts/python tools/bench_all_p0.py
Outputs: stdout + bench_all_results.json + bench_all_results.md
"""
from voicelang_core.adapters.whisper_streaming import _add_cuda_dll_dirs
_add_cuda_dll_dirs()
import time, json, platform, sys, os, statistics, pathlib

import numpy as np

results = {"meta": {}, "stages": {}}
def log(s):
    print(s, flush=True)
    results.setdefault("_log", []).append(s)

# meta
import subprocess, textwrap
try:
    import faster_whisper, ctranslate2, torch
    log(f"python {platform.python_version()} {platform.system()} {platform.machine()}")
    log(f"faster_whisper {faster_whisper.__version__} ctranslate2 {ctranslate2.__version__} torch {torch.__version__} cuda={torch.cuda.is_available()}")
    try:
        log(f"ct2 cuda devices {ctranslate2.get_cuda_device_count()}")
    except Exception as e:
        log(f"ct2 cuda probe err {e}")
except Exception as e:
    log(f"meta probe err {e}")

try:
    out = subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"], timeout=5, text=True).strip()
    log(f"nvidia-smi {out}")
    results["meta"]["nvidia_smi"] = out
except Exception as e:
    log(f"nvidia-smi err {e}")

wav_path = "C:/Users/ahmed/whishper-core/e2e_speech.wav"
if not pathlib.Path(wav_path).exists():
    wav_path = "e2e_speech.wav"
try:
    from voicelang_core.adapters.wav_file import WavFileSource as _WavLoader
    from voicelang_core.types import pcm_bytes_to_float32
    _src = _WavLoader(wav_path, block_seconds=1.0)
    chunks_all = list(_src.stream())
    pcm_all = b"".join(c.pcm for c in chunks_all)
    data = pcm_bytes_to_float32(pcm_all)
    sr = 16000
    log(f"wav {wav_path}: {len(data)/16000:.3f}s mean={np.abs(data).mean():.4f} max={np.abs(data).max():.4f} chunks={len(chunks_all)}")
    results["meta"]["wav"] = {"path": wav_path, "dur": len(data)/16000, "mean": float(np.abs(data).mean())}
except Exception as e:
    log(f"wav load err {e}")
    data = np.zeros(16000*2, dtype=np.float32)
    sr = 16000

# 1. Capture block latency theory + WavFileSource block test
log("\n=== 1. Capture block latency (theory + WavFileSource) ===")
for bs in [1.0, 0.5, 0.25, 0.02, 0.01]:
    try:
        from voicelang_core.adapters.wav_file import WavFileSource
        src = WavFileSource(wav_path, block_seconds=bs)
        chunks = list(src.stream())
        avg_fill = bs/2*1000
        log(f"block {bs*1000:.0f}ms -> chunks {len(chunks)} avg_fill {avg_fill:.1f}ms (P0 target 10ms @20ms)")
        results["stages"].setdefault("capture_blocks", {})[str(bs)] = {"chunks": len(chunks), "avg_fill_ms": avg_fill}
    except Exception as e:
        log(f"block {bs} err {e}")

# also test MicSource blocksize calculation (no hardware read)
try:
    from voicelang_core.adapters.mic import MicSource
    import inspect
    sig = inspect.signature(MicSource.__init__)
    log(f"MicSource default block_seconds {sig.parameters['block_seconds'].default}")
    results["stages"]["mic_default"] = str(sig.parameters['block_seconds'].default)
    from voicelang_core.adapters.wasapi_loopback import WASAPILoopbackSource
    sig2 = inspect.signature(WASAPILoopbackSource.__init__)
    log(f"WASAPILoopback default {sig2.parameters['block_seconds'].default}")
except Exception as e:
    log(f"capture defaults err {e}")

# 2. VAD gate bench
log("\n=== 2. VAD gate (CPU Silero / energy fallback) ===")
try:
    from voicelang_core.vad import VadGate
    from voicelang_core.types import AudioChunk
    from voicelang_core.adapters.wav_file import WavFileSource
    # probe model path
    try:
        from voicelang_core.vad import _probe_silero_model
        mp = _probe_silero_model()
        log(f"VAD model probe: {mp}")
    except Exception as e:
        log(f"VAD probe err {e}")
        mp = None
    for label, thr, en_thr in [("default(0.5/80)",0.5,80), ("permissive(0.4/60)",0.4,60)]:
        # Use energy fallback only — sherpa_onnx bundled ORT 1.17.1 segfaults on API 27 models
        vg = VadGate(threshold=thr, energy_threshold=en_thr, enabled=True)
        # force energy fallback to avoid segfault: monkey-patch _use_sherpa=False if ORT <1.20
        try:
            import onnxruntime as ort
            # parse 1.17.1 < 1.20 => force fallback
            ver = tuple(int(x) for x in ort.__version__.split(".")[:2])
            if ver < (1, 20):
                vg._use_sherpa = False
                log(f"  ORT {ort.__version__} <1.20 -> force energy fallback (avoid segfault)")
        except Exception:
            pass
        vg._use_sherpa = False  # always fallback for bench stability
        src = WavFileSource(wav_path, block_seconds=0.02)
        t0=time.perf_counter()
        n=0
        speech=0
        for ch in src.stream():
            d=vg.is_speech(ch.pcm)
            n+=1
            if d.is_speech: speech+=1
            if n>=336: break
        elapsed=(time.perf_counter()-t0)*1000
        st=vg.stats()
        log(f" VAD {label}: calls={st['calls']} speech={st['speech']} skipped={st['skipped']} avg_overhead={st['avg_overhead_ms']:.3f}ms total={st['total_overhead_ms']:.1f}ms use_sherpa={st['use_sherpa']}")
        log(f"  filespeech ratio {speech}/{n}={speech/max(1,n):.2%}")
        results["stages"].setdefault("vad", {})[label] = st
    # vad disabled path
    vg_off = VadGate(enabled=False)
    log(f"VAD disabled stats {vg_off.stats()}")
except Exception as e:
    import traceback; log(f"VAD bench err {e}\n{traceback.format_exc()}")

# 3. Whisper bench (tiny/small) — real GPU
log("\n=== 3. Whisper bench (CUDA float16 vs CPU fallback) ===")
try:
    import ctranslate2
    from faster_whisper import WhisperModel
    from voicelang_core.adapters.whisper_streaming import WhisperStreamingTranscriber
    # use 1s speech chunk from data offset 0.5s (has speech)
    speech_chunk = data[int(0.5*16000): int(1.5*16000)] if len(data)>16000*1.5 else data[:16000]
    log(f"speech_chunk len {len(speech_chunk)} mean {np.abs(speech_chunk).mean():.4f}")
    for spec in [("tiny","float16"), ("small","float16"), ("small","int8_float16")]:
        size, ct = spec
        try:
            t0=time.perf_counter()
            w = WhisperStreamingTranscriber(model_size=size, device="cuda", compute_type=ct, cpu_threads=2, translate=True)
            load_ms=(time.perf_counter()-t0)*1000
            log(f"[{size} {ct} cuda via Transcriber] init {load_ms:.0f}ms device={w._device} compute={w._compute_type}")
            # warmup
            t0=time.perf_counter()
            w.warmup(0.6)
            warm_ms=(time.perf_counter()-t0)*1000
            log(f"  warmup 0.6s {warm_ms:.0f}ms")
            times=[]
            outs=[]
            for i in range(5):
                t0=time.perf_counter()
                # use internal model directly for apples-apples with old bench
                segs,info = w._model.transcribe(speech_chunk, beam_size=1, task="translate", language=None, vad_filter=False, condition_on_previous_text=False, without_timestamps=True)
                out=list(segs)
                dt=(time.perf_counter()-t0)*1000
                times.append(dt)
                outs.append(out[0].text.strip() if out else "")
                log(f"  run {i} 1s beam1: {dt:.0f}ms -> {outs[-1][:40]!r}")
            p50=statistics.median(times)
            p95=sorted(times)[int(len(times)*0.95)] if len(times)>1 else times[0]
            log(f"  p50 {p50:.0f}ms p95 {p95:.0f}ms RTF {p50/1000:.3f}")
            results["stages"].setdefault("whisper", {})[f"{size}_{ct}"] = {"load_ms": load_ms, "warmup_ms": warm_ms, "p50": p50, "p95": p95, "texts": outs[:1], "device": w._device}
            # also show 0.5/1/2s RTF via direct model
            for secs in [0.5,1.0,2.0]:
                ch = data[int(0.5*16000): int((0.5+secs)*16000)] if len(data)>(0.5+secs)*16000 else data[:int(secs*16000)]
                t0=time.perf_counter()
                segs,info = w._model.transcribe(ch, beam_size=1, task="translate", language=None, vad_filter=False, condition_on_previous_text=False, without_timestamps=True)
                out=list(segs)
                dt=(time.perf_counter()-t0)*1000
                log(f"    {secs}s RTF {dt/(secs*1000):.2f} {dt:.0f}ms")
        except Exception as e:
            import traceback; log(f" [{size} {ct}] err {e}\n{traceback.format_exc()}")
            results["stages"].setdefault("whisper", {})[f"{size}_{ct}"] = {"error": str(e)}
except Exception as e:
    import traceback; log(f"whisper bench outer err {e}\n{traceback.format_exc()}")

# 4. Moonshine 0.08 bench
log("\n=== 4. Moonshine Tiny Streaming 0.08 ===")
try:
    from voicelang_core.adapters.moonshine import MoonshineTranscriber
    from voicelang_core.types import AudioChunk
    mt = MoonshineTranscriber(language="en", model_arch="TINY_STREAMING", update_interval=0.08)
    log("moonshine tiny loaded")
    t0=time.perf_counter()
    mt.warmup(0.6)
    log(f"  warmup 0.6s done {(time.perf_counter()-t0)*1000:.0f}ms")
    t0=time.perf_counter()
    first_partial=None
    partials=[]
    finals=[]
    for i in range(int(len(data)/(16000*0.02))):
        chunk = data[i*int(16000*0.02): (i+1)*int(16000*0.02)]
        if len(chunk)<320: break
        pcm=(chunk*32767).astype(np.int16).tobytes()
        ac=AudioChunk(pcm=pcm, sample_rate=16000)
        for seg in mt.transcribe_stream(ac):
            now=(time.perf_counter()-t0)*1000
            if seg.status=="partial" and first_partial is None:
                first_partial=now
            if seg.status=="partial": partials.append((now, seg.source_text))
            else: finals.append((now, seg.source_text))
    # flush
    for seg in mt.finish():
        finals.append(((time.perf_counter()-t0)*1000, seg.source_text))
    log(f"  first partial {first_partial:.0f}ms" if first_partial else "  no partial")
    if partials:
        for ms, txt in partials[:6]:
            log(f"    {ms:.0f}ms partial {txt!r}")
    for ms, txt in finals[:6]:
        log(f"    {ms:.0f}ms final {txt!r}")
    mt.close()
    results["stages"]["moonshine"] = {"first_partial_ms": first_partial, "partials": len(partials), "finals": len(finals)}
except Exception as e:
    import traceback; log(f"moonshine bench err {e}\n{traceback.format_exc()}"); results["stages"]["moonshine"]={"error": str(e)}

# 5. Threaded pipeline bench
log("\n=== 5. Threaded pipeline (capture->ASR->translate->overlay) ===")
try:
    from voicelang_core.adapters.wav_file import WavFileSource
    from voicelang_core.ports import Translator
    from voicelang_core.types import Translation, Segment
    from voicelang_core.adapters.overlay import OverlayDisplay
    from voicelang_core.pipeline_threaded import ThreadedPipeline
    from voicelang_core.adapters.fake import FakeStreamingTranscriber, FakeAudioSource
    from voicelang_core.types import AudioChunk

    class PT(Translator):
        is_passthrough=True
        def translate(self, text, target_language='en', source_language=None):
            return Translation(source_text=text, target_text=text, source_language=source_language or 'en', target_language=target_language)

    # 5a fake transducer (measures pipeline overhead without GPU)
    log(" 5a Fake transducer (pipeline overhead only, 0.02 block, vad off)")
    def pcm20(): return (np.random.randint(-500,500, size=320, dtype=np.int16)).tobytes()
    src = FakeAudioSource([AudioChunk(pcm=pcm20(), sample_rate=16000) for _ in range(336)], block_seconds=0.02)
    # give each chunk a capture_ts
    trans = FakeStreamingTranscriber(final_text="hello world", language="en")
    disp = OverlayDisplay(headless=True, max_lines=10)
    pipe = ThreadedPipeline(source=src, transcriber=trans, translator=PT(), display=disp, vad_enabled=False)
    t0=time.perf_counter()
    pipe.run_streaming_with_timed_replay()
    dt=(time.perf_counter()-t0)*1000
    log(f"   fake done {dt:.0f}ms lines {len(disp.lines)} transcribed {disp.transcribed_lines[:2]}")
    if pipe.metrics:
        log(f"   metrics {pipe.metrics.log_summary()}")
        results["stages"].setdefault("threaded", {})["fake"] = pipe.metrics.snapshot()

    # 5b whisper tiny threaded with vad on
    log(" 5b Whisper tiny threaded (0.02 capture -> 1.0 buffered, vad on, passthrough)")
    try:
        from voicelang_core.adapters.whisper_streaming import WhisperStreamingTranscriber
        w = WhisperStreamingTranscriber(model_size="tiny", device="cuda", compute_type="float16", cpu_threads=2, translate=True)
        w.warmup(0.6)
        src2 = WavFileSource(wav_path, block_seconds=0.02)
        disp2 = OverlayDisplay(headless=True, max_lines=20)
        pipe2 = ThreadedPipeline(source=src2, transcriber=w, translator=PT(), display=disp2, vad_enabled=True)
        t0=time.perf_counter()
        pipe2.run_streaming_with_timed_replay()
        dt=(time.perf_counter()-t0)*1000
        log(f"   whisper tiny threaded done {dt:.0f}ms transcribed {disp2.transcribed_lines[:3]} lines {disp2.lines[:3]}")
        if pipe2.metrics:
            log(f"   metrics {pipe2.metrics.log_summary()}")
            results["stages"].setdefault("threaded", {})["whisper_tiny"] = pipe2.metrics.snapshot()
            results["stages"]["threaded"]["whisper_tiny"]["transcribed_lines"] = disp2.transcribed_lines[:5]
    except Exception as e:
        import traceback; log(f"  5b err {e}\n{traceback.format_exc()}")

    # 5c moonshine threaded
    log(" 5c Moonshine threaded (0.02 capture, 0.08 update, vad on)")
    try:
        from voicelang_core.adapters.moonshine import MoonshineTranscriber
        mt2 = MoonshineTranscriber(language="en", model_arch="TINY_STREAMING", update_interval=0.08)
        mt2.warmup(0.6)
        src3 = WavFileSource(wav_path, block_seconds=0.02)
        disp3 = OverlayDisplay(headless=True, max_lines=20)
        pipe3 = ThreadedPipeline(source=src3, transcriber=mt2, translator=PT(), display=disp3, vad_enabled=True)
        t0=time.perf_counter()
        pipe3.run_streaming_with_timed_replay()
        dt=(time.perf_counter()-t0)*1000
        log(f"   moonshine threaded done {dt:.0f}ms transcribed {disp3.transcribed_lines[:3]}")
        if pipe3.metrics:
            log(f"   metrics {pipe3.metrics.log_summary()}")
            results["stages"].setdefault("threaded", {})["moonshine"] = pipe3.metrics.snapshot()
            results["stages"]["threaded"]["moonshine"]["transcribed_lines"] = disp3.transcribed_lines[:5]
        mt2.close()
    except Exception as e:
        import traceback; log(f"  5c err {e}\n{traceback.format_exc()}")

    # 5d queue drop test (force small queue with bursty source vs slow ASR)
    log(" 5d Queue drop / bounded latency (asr_queue 2, fast source vs slow ASR)")
    class SlowTrans:
        def transcribe_stream(self, chunk):
            time.sleep(0.05) # 50ms per chunk, source pumps 20ms => backpressure
            from voicelang_core.types import Segment
            yield Segment(status="final", source_text="slow")
        def finish(self):
            from voicelang_core.types import Segment
            yield Segment(status="final", source_text="slow tail")
            return
            yield
    src4 = FakeAudioSource([AudioChunk(pcm=pcm20(), sample_rate=16000) for _ in range(50)])
    disp4 = OverlayDisplay(headless=True, max_lines=100)
    pipe4 = ThreadedPipeline(source=src4, transcriber=SlowTrans(), translator=PT(), display=disp4, asr_queue_size=2, translate_queue_size=8, vad_enabled=False)
    pipe4.run_streaming_with_timed_replay()
    log(f"   drops {pipe4.metrics.dropped_blocks if pipe4.metrics else '?'} transcribed {len(disp4.transcribed_lines)} vs chunks 50 (should drop)")
    results["stages"].setdefault("threaded", {})["drops_test"] = pipe4.metrics.snapshot() if pipe4.metrics else {}

except Exception as e:
    import traceback; log(f"threaded bench err {e}\n{traceback.format_exc()}")

# 6. Overlay coalesce
log("\n=== 6. Overlay coalesce (33ms) ===")
try:
    from voicelang_core.adapters.overlay import OverlayDisplay
    from voicelang_core.types import Segment
    d = OverlayDisplay(headless=True, max_lines=5)
    t0=time.perf_counter()
    for i in range(20):
        d.show_partial(Segment(status="partial", source_text=f"partial {i}"))
        time.sleep(0.005)
    d.show(Segment(id=1, status="final", source_text="final 1", translated_text="final 1"))
    log(f" overlay lines {d.lines} hist left {d._hist_left}")
    results["stages"]["overlay"] = {"lines": d.lines, "hist_len": len(d._hist_left)}
except Exception as e:
    log(f"overlay bench err {e}")

# save
try:
    with open("tools/bench_all_results.json","w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log(f"\nsaved tools/bench_all_results.json")
    # md summary
    md = ["# P0 Bench Results", "", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    md.append("## Capture")
    for k,v in results.get("stages",{}).get("capture_blocks",{}).items():
        md.append(f"- {k}s: {v}")
    md.append("\n## Whisper")
    for k,v in results.get("stages",{}).get("whisper",{}).items():
        md.append(f"- {k}: {v}")
    md.append("\n## VAD")
    for k,v in results.get("stages",{}).get("vad",{}).items():
        md.append(f"- {k}: {v}")
    md.append("\n## Threaded")
    for k,v in results.get("stages",{}).get("threaded",{}).items():
        md.append(f"- {k}: {str(v)[:500]}")
    with open("tools/bench_all_results.md","w", encoding="utf-8") as f:
        f.write("\\n".join(md))
    log("saved tools/bench_all_results.md")
except Exception as e:
    log(f"save err {e}")

log("\n=== BENCH DONE ===")
