"""Fun-ASR-Nano adapter: utterance state machine + lazy deps + factory wiring.

We stub torch + transformers (PR-#46180 API and the classic path) so the
adapter's code paths are exercised without downloading a 0.8B model.
"""
import builtins
import sys
import types

import numpy as np
import pytest

from voicelang_core.types import AudioChunk


@pytest.fixture(autouse=True)
def _clear_registry():
    """Each test installs its own fake processor/model — drop cached entries."""
    from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber

    FunAsrNanoTranscriber._registry.clear()
    yield
    FunAsrNanoTranscriber._registry.clear()


def _install_fakes(chat_api: bool = False):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    sys.modules["torch"] = fake_torch

    class FakeModel:
        def __init__(self, *a, **kw):
            self.kw = kw
            self.calls = 0

        def generate(self, **kw):
            self.calls += 1
            return np.array([[1, 2, 3, 4, 5]])  # 5 token ids

        def to(self, dev):
            return self

    class FakeInputs(dict):
        # transformers returns BatchFeature (a dict) — generate(**inputs)
        # requires a mapping, so the fake must be one too.
        def __init__(self, input_ids=None):
            super().__init__()
            self.input_ids = input_ids  # None = classic path (no slicing)

        def to(self, dev):
            return self

    class FakeProcessor:
        def __init__(self, chat: bool):
            self.chat = chat
            self.decodes = []

        def apply_transcription_request(self, audio=None, language=None,
                                        return_tensors=None):
            # PR #46180 chat-template API — present only in the PR build
            if not self.chat:
                raise AttributeError("apply_transcription_request requires PR #46180")
            return FakeInputs(input_ids=np.ones((1, 3), dtype=np.int64))

        def __call__(self, audio=None, sampling_rate=None, return_tensors=None):
            return FakeInputs()  # classic feature-extractor path

        def decode(self, ids, skip_special_tokens=False):
            self.decodes.append(list(ids))
            # classic: full 5 tokens -> "hello"; chat: sliced 2 -> "world"
            return "hello" if len(ids) == 5 else ("world" if len(ids) == 2 else f"len{len(ids)}")

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModelForSpeechSeq2Seq = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeModel())
    fake_tf.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeProcessor(chat_api))
    sys.modules["transformers"] = fake_tf
    return fake_tf


def _tone(seconds: float, amp: int = 8000) -> AudioChunk:
    n = int(16000 * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    pcm = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    return AudioChunk(pcm=pcm.tobytes(), sample_rate=16000)


def _quiet(seconds: float) -> AudioChunk:
    return AudioChunk(pcm=b"\x00\x00" * int(16000 * seconds), sample_rate=16000)


# ---- language routing -------------------------------------------------------

def test_language_normalization():
    from voicelang_core.adapters.funasr_nano import (
        FunAsrNanoTranscriber, _normalize_language,
    )
    assert _normalize_language(None) is None
    assert _normalize_language("") is None
    assert _normalize_language("auto") == "zh"
    assert _normalize_language("zh") == "zh"
    assert _normalize_language("zh-cn") == "zh-cn"
    assert _normalize_language("ZH-CN") == "zh-cn"
    assert _normalize_language("yue") == "zh"
    assert _normalize_language("cantonese") == "zh"
    assert _normalize_language("en") == "en"
    assert _normalize_language("ja") == "ja"
    with pytest.raises(ValueError):
        _normalize_language("fr")
    with pytest.raises(ValueError):
        FunAsrNanoTranscriber(language="de", device="cpu")


# ---- utterance state machine ------------------------------------------------

def test_pre_voice_silence_is_dropped():
    _install_fakes()
    from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber

    tr = FunAsrNanoTranscriber(language="en", device="cpu")
    for _ in range(8):
        assert list(tr.transcribe_stream(_quiet(0.25))) == []
    assert tr._model.calls == 0  # no inference on silence
    assert tr._buf_samples == 0


def test_blip_is_discarded():
    _install_fakes()
    from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber

    tr = FunAsrNanoTranscriber(language="en", device="cpu")
    list(tr.transcribe_stream(_tone(0.1)))       # 0.1 s blip
    segs = list(tr.transcribe_stream(_quiet(0.8)))  # long silence closes it
    assert segs == []
    assert tr._model.calls == 0  # never inferred a blip


def test_utterance_final_after_silence_gap():
    _install_fakes()
    from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber

    tr = FunAsrNanoTranscriber(language="zh", device="cpu", silence_gap=0.7)
    for _ in range(2):
        list(tr.transcribe_stream(_tone(0.25)))      # 0.5 s voice
    segs = []
    for _ in range(4):                               # 1.0 s quiet (>= gap)
        segs += list(tr.transcribe_stream(_quiet(0.25)))
    finals = [s for s in segs if s.status == "final"]
    assert len(finals) == 1
    assert finals[0].source_text == "hello"
    assert tr._model.calls == 1  # one generate per utterance


def test_max_utterance_caps_without_silence():
    _install_fakes()
    from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber

    tr = FunAsrNanoTranscriber(language="zh", device="cpu",
                               max_utterance=0.6, silence_gap=9.0)
    segs = []
    for _ in range(3):                               # 0.75 s of voice
        segs += list(tr.transcribe_stream(_tone(0.25)))
    finals = [s for s in segs if s.status == "final"]
    assert len(finals) == 1 and finals[0].source_text == "hello"


def test_partial_preview_then_final():
    _install_fakes()
    from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber

    tr = FunAsrNanoTranscriber(language="zh", device="cpu",
                               partial_interval=0.5, silence_gap=0.7)
    partials = []
    for _ in range(11):                              # 2.75 s of voice
        partials += [s for s in tr.transcribe_stream(_tone(0.25))
                     if s.status == "partial"]
    assert len(partials) >= 1
    assert all(p.source_text == "hello" for p in partials)
    finals = [s for s in tr.transcribe_stream(_quiet(0.8))
              if s.status == "final"]
    assert len(finals) == 1


def test_finish_flushes_trailing_utterance():
    _install_fakes()
    from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber

    tr = FunAsrNanoTranscriber(language="en", device="cpu")
    for _ in range(2):
        list(tr.transcribe_stream(_tone(0.25)))
    fins = list(tr.finish())
    assert len(fins) == 1 and fins[0].status == "final"
    assert list(tr.finish()) == []  # second call is a no-op


# ---- processor paths --------------------------------------------------------

def test_classic_processor_path_no_slice():
    _install_fakes(chat_api=False)
    from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber

    tr = FunAsrNanoTranscriber(language="en", device="cpu")
    for _ in range(2):
        list(tr.transcribe_stream(_tone(0.25)))
    fins = list(tr.finish())
    assert fins[0].source_text == "hello"  # full 5-token decode


def test_chat_template_path_slices_instruction_tokens():
    _install_fakes(chat_api=True)
    from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber

    tr = FunAsrNanoTranscriber(language="zh", device="cpu")
    for _ in range(2):
        list(tr.transcribe_stream(_tone(0.25)))
    fins = list(tr.finish())
    assert fins[0].source_text == "world"  # instruction tokens sliced off


# ---- missing deps -----------------------------------------------------------

def test_missing_torch_raises_actionable_error(monkeypatch):
    _install_fakes()

    def _no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *a, **k)

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _no_torch)

    from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber

    tr = FunAsrNanoTranscriber(language="en", device="cpu")
    with pytest.raises(RuntimeError) as ei:
        list(tr.transcribe_stream(_tone(0.25)))
    assert "uv sync --extra funasr-nano" in str(ei.value)


# ---- factory wiring ---------------------------------------------------------

def test_run_factory_builds_funasr_nano():
    _install_fakes()
    from voicelang_core.config import Config
    from voicelang_core.run import _build_transcriber

    tr = _build_transcriber("funasr-nano", Config(language="zh-cn"))
    assert tr.__class__.__name__ == "FunAsrNanoTranscriber"
    assert tr._language == "zh-cn"
    tr2 = _build_transcriber("funasr-nano", Config(language="auto"))
    assert tr2._language == "zh"


def test_engines_build_pipeline_funasr_nano():
    _install_fakes()
    from voicelang_core.engines import EngineConfig, EngineName, build_pipeline

    pipeline, run_mode = build_pipeline(EngineConfig(
        name=EngineName.funasr_nano, language="en"))
    assert run_mode == "streaming"
    assert pipeline.transcriber.__class__.__name__ == "FunAsrNanoTranscriber"