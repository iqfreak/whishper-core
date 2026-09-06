"""Benchmark fastest latency + best inference on RTX 4050 for all 5 engines.

Measures: warmup, per-chunk decode (115ms small float16), partial emission
for Moonshine (0.08 update), and threaded pipeline overall with pipelined
translation (no block). Run this to prove the 4050 config.
"""
from voicelang_core.adapters.whisper_streaming import _add_cuda_dll_dirs
_add_cuda_dll_dirs()
import time, numpy as np, librosa, os
import statistics

wav = "e2e_speech.wav"
data, sr = librosa.load(wav, sr=16000, mono=True)
print(f"wav {wav}: {len(data)/16000:.2f}s, sr 16000, mean {np.abs(data).mean():.4f}")

# --- whisper small float16 beam1 without_timestamps ---
from faster_whisper import WhisperModel
for spec in [("small","float16"), ("tiny","float16"), ("small","int8_float16")]:
    size, ct = spec
    t0=time.perf_counter()
    m=WhisperModel(size, device="cuda", compute_type=ct, cpu_threads=2)
    load_ms=(time.perf_counter()-t0)*1000
    print(f"\n[{size} {ct} cuda] load {load_ms:.0f}ms")
    # warmup
    t0=time.perf_counter()
    from voicelang_core.adapters.whisper_streaming import WhisperStreamingTranscriber
    w=WhisperStreamingTranscriber(model_size=size, device="cuda", compute_type=ct, cpu_threads=2)
    w.warmup(1.0)
    warm_ms=(time.perf_counter()-t0)*1000
    print(f"  warmup 1s zeros {warm_ms:.0f}ms (should be ~120-180ms after cached, 600 cold)")
    chunk=data[int(2.0*16000):int(3.0*16000)]
    times=[]
    for i in range(5):
        t0=time.perf_counter()
        segs,info=w._model.transcribe(chunk, beam_size=1, task="translate", language="en",
                                      vad_filter=False, condition_on_previous_text=False,
                                      without_timestamps=True)
        out=list(segs)
        dt=(time.perf_counter()-t0)*1000
        times.append(dt)
        print(f"  run {i} 1s chunk beam1 no-ts: {dt:.0f}ms -> {[s.text.strip() for s in out][:1]}")
    print(f"  p50 {statistics.median(times):.0f}ms p95 {sorted(times)[int(len(times)*0.95) if len(times)>1 else 0]:.0f}ms (post-warmup)")
    # also test chunk sweep
    for secs in [0.5,1.0,2.0]:
        ch=data[:int(16000*secs)]
        t0=time.perf_counter()
        segs,info=w._model.transcribe(ch, beam_size=1, task="translate", language="en",
                                      vad_filter=False, condition_on_previous_text=False,
                                      without_timestamps=True)
        out=list(segs)
        dt=(time.perf_counter()-t0)*1000
        print(f"    {secs}s chunk: {dt:.0f}ms RTF {dt/(secs*1000):.2f} text len {len(out[0].text.strip()) if out else 0}")

# --- moonshine tiny 0.08 ---
print("\n--- Moonshine Tiny Streaming (update_interval 0.08) ---")
try:
    from voicelang_core.adapters.moonshine import MoonshineTranscriber
    from voicelang_core.types import AudioChunk
    mt=MoonshineTranscriber(language="en", model_arch="TINY_STREAMING", update_interval=0.08)
    print("moonshine tiny loaded (first load may download/cache)")
    t0=time.perf_counter()
    # feed like run.py does: 0.25s blocks
    first_partial=None
    for i in range(int(len(data)/(16000*0.25))):
        chunk=data[i*int(16000*0.25):(i+1)*int(16000*0.25)]
        pcm=(chunk*32767).astype(np.int16).tobytes()
        ac=AudioChunk(pcm=pcm, sample_rate=16000)
        for seg in mt.transcribe_stream(ac):
            if seg.status=="partial" and first_partial is None:
                first_partial=(time.perf_counter()-t0)*1000
                print(f"  first partial at {first_partial:.0f}ms streaming clock: {seg.source_text!r}")
            print(f"  {(time.perf_counter()-t0)*1000:.0f}ms {seg.status} {seg.source_text!r}")
    for seg in mt.finish():
        print(f"  finish {seg.status} {seg.source_text!r}")
    if first_partial is not None and first_partial < 650:
        print(f"  OK first partial {first_partial:.0f}ms < 650ms")
    mt.close()
except Exception as e:
    import traceback; traceback.print_exc()

# --- threaded pipeline with pipelined translation ---
print("\n--- Threaded pipeline (capture+ASR decoupled, translate pipelined) ---")
from voicelang_core.adapters.wav_file import WavFileSource
from voicelang_core.adapters.passthrough import PassthroughTranslator
from voicelang_core.adapters.console import ConsoleDisplay
from voicelang_core.pipeline_threaded import ThreadedPipeline

class TimedDisplay(ConsoleDisplay):
    def __init__(self):
        self.t0=time.perf_counter()
        self.events=[]
    def show_partial(self, seg):
        self.events.append(((time.perf_counter()-self.t0)*1000, "partial", seg.id, seg.source_text))
        super().show_partial(seg)
    def show(self, seg):
        self.events.append(((time.perf_counter()-self.t0)*1000, seg.status, seg.id, seg.source_text, seg.translated_text))
        super().show(seg)

src=WavFileSource(wav, block_seconds=0.25)  # streaming cadence for moonshine
transcriber=MoonshineTranscriber(language="en", model_arch="TINY_STREAMING", update_interval=0.08)
translator=PassthroughTranslator()
display=TimedDisplay()
pipe=ThreadedPipeline(source=src, transcriber=transcriber, translator=translator, display=display)
# run like e2e but with threaded pipeline
pipe.run_streaming_with_timed_replay()
print(f"threaded moonshine events: {len(display.events)}")
for e in display.events[:12]:
    print(f"  {e[0]:.0f}ms {e[1]} {e[2:]}")
transcriber.close()

# whisper threaded too
print("\n--- Threaded pipeline whisper small float16 ---")
src2=WavFileSource(wav, block_seconds=0.5)  # 0.5 for whisper to avoid 0.5s truncated hallucination
from voicelang_core.adapters.whisper_streaming import WhisperStreamingTranscriber
w2=WhisperStreamingTranscriber(model_size="small", device="cuda", compute_type="float16", cpu_threads=2)
w2.warmup(1.0)
display2=TimedDisplay()
pipe2=ThreadedPipeline(source=src2, transcriber=w2, translator=PassthroughTranslator(), display=display2)
pipe2.run_streaming_with_timed_replay()
print(f"threaded whisper small events: {len(display2.events)}")
for e in display2.events[:8]:
    print(f"  {e[0]:.0f}ms {e[1]} {e[2:]}")
print("\nbenchmark done")
