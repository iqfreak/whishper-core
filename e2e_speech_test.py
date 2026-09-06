"""End-to-end pipeline test with REAL SPEECH audio (no microphone needed).

Generates speech via Windows SAPI TTS, then runs it through the exact same
ports run.py uses: AudioSource -> WhisperStreamingTranscriber -> Translator
-> Display. Proves the glue works with real audio, not silence.

Run:  uv run python e2e_speech_test.py
"""
import sys
import wave

import numpy as np

try:
    import librosa
except ModuleNotFoundError:  # pragma: no cover
    librosa = None

from voicelang_core.adapters.overlay import OverlayDisplay
from voicelang_core.adapters.passthrough import PassthroughTranslator
from voicelang_core.adapters.wav_file import WavFileSource
from voicelang_core.adapters.whisper_streaming import WhisperStreamingTranscriber
from voicelang_core.pipeline import Pipeline

TEXT = "The mission starts at dawn. Move to the extraction point and hold position."

WAV = "e2e_speech.wav"
RATE = 16000


def make_speech():
    """Synthesize speech to a 16k mono WAV using Windows SAPI."""
    import win32com.client  # pywin32 -- only present on Windows

    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    # 16 kHz mono 16-bit PCM: SAFT16kHz16BitMono == 32
    stream.Format.Type = 32
    stream.Open(WAV, 3, False)  # SSFMCreateForWrite
    speaker.AudioOutputStream = stream
    speaker.Speak(TEXT)
    stream.Close()

    with wave.open(WAV, "rb") as w:
        rate, width, chans = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    print(f"tts wrote: {rate}Hz {width*8}-bit {chans}ch", flush=True)

    if width == 1:  # 8-bit unsigned PCM
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:  # 16-bit signed
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if chans == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    if rate != RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=RATE)
    print(f"resampled -> {RATE}Hz, {len(audio)/RATE:.2f}s", flush=True)
    return audio


def main():
    print(f"TTS text: {TEXT!r}", flush=True)
    audio = make_speech()
    print(f"speech: {len(audio) / RATE:.2f}s @ {RATE}Hz", flush=True)

    captured = []
    display = OverlayDisplay()
    print(f"overlay headless: {display.headless}", flush=True)

    class Capture:
        """Wraps the real overlay and records what it was asked to show."""

        def show(self, segment):
            captured.append(("final", segment.translated_text or segment.source_text))
            print(f"  FINAL: {segment.translated_text or segment.source_text!r}", flush=True)
            display.show(segment)

        def show_partial(self, segment):
            captured.append(("partial", segment.source_text))
            print(f"  PARTIAL: {segment.source_text!r}", flush=True)
            display.show_partial(segment)

    pipeline = Pipeline(
        source=WavFileSource(WAV, block_seconds=1.0),
        transcriber=WhisperStreamingTranscriber(model_size="small"),
        translator=PassthroughTranslator(),
        display=Capture(),
        target_language="en",
    )
    print("--- running pipeline ---", flush=True)
    pipeline.run_streaming()

    finals = [t for k, t in captured if k == "final"]
    partials = [t for k, t in captured if k == "partial"]
    print(f"\npartials: {len(partials)}  finals: {len(finals)}", flush=True)
    print(f"overlay lines: {display.lines}", flush=True)
    if not finals:
        print("RESULT: FAIL -- no final caption produced", flush=True)
        return 1
    print(f"RESULT: PASS -- best final: {finals[-1]!r}", flush=True)
    print(f"expected roughly: {TEXT!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())