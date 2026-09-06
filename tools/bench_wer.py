"""WER harness — jiwer, per-tier KPIs. Run: PYTHONPATH=. .venv/Scripts/python tools/bench_wer.py
Requires: pip install jiwer  (optional — skips if missing)
Uses: LibriSpeech dev-clean slice or C:/Users/ahmed/whishper-core/e2e_speech.wav hand-label
"""
import pathlib, sys
try:
    import jiwer
except ImportError:
    print("jiwer not installed — pip install jiwer to run WER; skipping")
    sys.exit(0)

# Hand-labeled e2e_speech.wav reference (approx — replace with verified transcript)
REFS = {
    "e2e_speech.wav": "the mission starts at dawn move to the extraction point and hold position",
}

def wer(hyp: str, ref: str) -> float:
    return jiwer.wer(ref.lower(), hyp.lower())

if __name__ == "__main__":
    # Demo on fixture via whisper tiny
    from voicelang_core.adapters.wav_file import WavFileSource
    from voicelang_core.adapters.whisper_streaming import WhisperStreamingTranscriber
    from voicelang_core.types import AudioChunk
    wav = "C:/Users/ahmed/whishper-core/e2e_speech.wav"
    if not pathlib.Path(wav).exists():
        wav = "e2e_speech.wav"
    ref = REFS["e2e_speech.wav"]
    print(f"ref: {ref}")
    for model_size in ("tiny",):
        tr = WhisperStreamingTranscriber(model_size=model_size, device="auto")
        tr.warmup(0.6)
        src = WavFileSource(wav, block_seconds=1.0)
        hyps = []
        for chunk in src.stream():
            for seg in tr.transcribe_stream(chunk):
                hyps.append(seg.source_text)
        for seg in tr.finish():
            hyps.append(seg.source_text)
        hyp = " ".join(hyps).strip().lower()
        print(f"[{model_size}] hyp: {hyp}")
        print(f"[{model_size}] WER: {wer(hyp, ref):.2%}  (streaming tier target: windowed final WER <10% clean, <20% gaming)")
        # Per-tier gate: streaming tier first-partial p50 checked in bench_all_p0; windowed tier final WER here
    print("Tier KPIs: streaming first-partial p50 ≤300ms (bench_all_p0 speech_onset_to_partial); windowed first-final ≤1.5s + WER above")
