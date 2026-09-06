"""Nemotron adapter tests. Real int8 model + real TTS WAV when present.

The real-inference test runs in a SUBPROCESS: moonshine-voice loads its own
bundled onnxruntime, which collides with sherpa's pip onnxruntime in one
process (access violation). Engines are used one at a time in production; the
subprocess mirrors that isolation.
"""
import os
import subprocess
import sys

import pytest

from voicelang_core.adapters.nemotron import NemotronStreamingTranscriber

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(
    _ROOT, "models",
    "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
)
WAV = os.path.join(_ROOT, "e2e_speech.wav")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(MODEL_DIR, "encoder.int8.onnx"))
    or not os.path.isfile(WAV),
    reason="nemotron int8 model or e2e_speech.wav not downloaded",
)

_INFERENCE_SCRIPT = r"""
import sys
import numpy as np
sys.path.insert(0, {root!r})
from voicelang_core.adapters.nemotron import NemotronStreamingTranscriber
from voicelang_core.adapters.wav_file import load_wav
from voicelang_core.types import AudioChunk

t = NemotronStreamingTranscriber(model_dir={model!r})
audio = np.concatenate([load_wav({wav!r}), np.zeros(3 * 16000, dtype=np.float32)])
segs = []
for i in range(0, len(audio), 8000):
    c = AudioChunk(pcm=(audio[i:i+8000] * 32767).astype("int16").tobytes(),
                   sample_rate=16000)
    segs.extend(t.transcribe_stream(c))
finals = " ".join(s.source_text for s in segs if s.status == "final").lower()
assert finals, "no final segment produced"
assert any(w in finals for w in ("mission", "dawn", "extraction", "position")), finals
print("INFERENCE PASS:", finals)
""".format(root=_ROOT, model=MODEL_DIR, wav=WAV)


def test_nemotron_streaming_transcribes_real_speech():
    r = subprocess.run([sys.executable, "-c", _INFERENCE_SCRIPT],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "INFERENCE PASS" in r.stdout


def test_nemotron_missing_model_dir_raises():
    with pytest.raises(FileNotFoundError):
        NemotronStreamingTranscriber(model_dir=os.path.join(_ROOT, "models", "nope"))