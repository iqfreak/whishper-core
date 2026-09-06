"""Benchmark fastest latency + best inference on RTX 4050 for all 5 engines."""
from voicelang_core.adapters.whisper_streaming import _add_cuda_dll_dirs
_add_cuda_dll_dirs()
import time, numpy as np, librosa, os
import statistics

wav = "e2e_speech.wav"
data, sr = librosa.load(wav, sr=16000, mono=True)
print(f"wav {wav}: {len(data)/16000:.2f}s")

from faster_whisper import WhisperModel
# Pick per earlier result: tiny fastest (41ms/1s), small quality+115ms. Report both but recommendation tiny for <100ms.
for spec in [("small","float16"), ("tiny","float16"), ("small","int8_float16")]:
    size, ct = spec
    t0=time.perf_counter()
    m=WhisperModel(size, device="cuda", compute_type=ct, cpu_threads=2)
    load_ms=(time.perf_counter()-t0)*1000
    print(f"\n[{size} {ct} cuda] load {load_ms:.0f}ms")
    from voicelang_core.adapters.whisper_streaming import WhisperStreamingTranscriber
    w=WhisperStreamingTranscriber(model_size=size, device="cuda", compute_type=ct, cpu_threads=2)
    # warmup already done inside Transcriber's first transcribe; call explicit warmup
    t0=time.perf_counter()
    w.warmup(1.0)
    warm_ms=(time.perf_counter()-t0)*1000
    print(f"  warmup 1s zeros {warm_ms:.0f}ms")
    chunk=data[int(2.0*16000):int(3.0*16000)]
    # slice that's actually speech (skip leading silence zeros which were skewing toward [] earlier)
    # find speech region: chunk from 0..1s of wav has speech
    speech_chunk=data[0:16000]
    times=[]
    for i in range(5):
        t0=time.perf_counter()
        segs,info=w._model.transcribe(speech_chunk, beam_size=1, task="translate", language="en",
                                      vad_filter=False, condition_on_previous_text=False,
                                      without_timestamps=True)
        out=list(segs)
        dt=(time.perf_counter()-t0)*1000
        times.append(dt)
        print(f"  run {i} 1s speech beam1: {dt:.0f}ms -> {[s.text.strip() for s in out][:1]}")
    print(f"  p50 {statistics.median(times):.0f}ms")

print("\n--- Moonshine Tiny Streaming (update_interval 0.08) ---")
try:
    from voicelang_core.adapters.moonshine import MoonshineTranscriber
    from voicelang_core.types import AudioChunk
    mt=MoonshineTranscriber(language="en", model_arch="TINY_STREAMING", update_interval=0.08)
    print("moonshine tiny loaded")
    t0=time.perf_counter()
    first_partial=None
    for i in range(int(len(data)/(16000*0.25))):
        chunk=data[i*int(16000*0.25):(i+1)*int(16000*0.25)]
        pcm=(chunk*32767).astype(np.int16).tobytes()
        ac=AudioChunk(pcm=pcm, sample_rate=16000)
        for seg in mt.transcribe_stream(ac):
            if seg.status=="partial" and first_partial is None:
                first_partial=(time.perf_counter()-t0)*1000
                print(f"  first partial at {first_partial:.0f}ms: {seg.source_text!r}")
            print(f"  {(time.perf_counter()-t0)*1000:.0f}ms {seg.status} {seg.source_text!r}")
    for seg in mt.finish():
        print(f"  finish {seg.status} {seg.source_text!r}")
    print(f"  OK first partial {first_partial:.0f}ms" if first_partial else "  no partial")
    mt.close()
except Exception as e:
    import traceback; traceback.print_exc()

print("\n--- Threaded pipeline (capture+ASR decoupled, translate pipelined) ---")
from voicelang_core.adapters.wav_file import WavFileSource
from voicelang_core.adapters.passthrough import PassthroughTranslator
from voicelang_core.adapters.console import ConsoleDisplay
from voicelang_core.pipeline_threaded import ThreadedPipeline
from voicelang_core.types import Segment

class TimedDisplay(ConsoleDisplay):
    def __init__(self):
        super().__init__()
        self.t0=time.perf_counter()
        self.events=[]
    def show_partial(self, seg: Segment):
        self.events.append(((time.perf_counter()-self.t0)*1000, "partial", seg.id, seg.source_text))
        super().show_partial(seg)
    def show(self, seg: Segment):
        self.events.append(((time.perf_counter()-self.t0)*1000, seg.status, seg.id, seg.source_text, seg.translated_text))
        super().show(seg)

src=WavFileSource(wav, block_seconds=0.25)
transcriber=MoonshineTranscriber(language="en", model_arch="TINY_STREAMING", update_interval=0.08)
translator=PassthroughTranslator()
display=TimedDisplay()
pipe=ThreadedPipeline(source=src, transcriber=transcriber, translator=translator, display=display)
pipe.run_streaming_with_timed_replay()
print(f"threaded moonshine events: {len(display.events)}")
for e in display.events[:10]:
    print(f"  {e[0]:.0f}ms {e[1]} {e[2:]}")
transcriber.close()

print("\n--- Threaded pipeline whisper small float16 ---")
src2=WavFileSource(wav, block_seconds=0.5)
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
