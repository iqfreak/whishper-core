"""FunASR adapter: routing + lazy deps + pipeline integration (no real torch/model needed).

We stub funasr.AutoModel and torch.cuda so the adapter's code paths are exercised
without downloading a ~220MB model or needing CUDA.
"""
import sys
import types
import numpy as np
import pytest

from voicelang_core.types import AudioChunk


def _install_fakes():
    # fake torch
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules.setdefault("torch", fake_torch)

    # fake funasr
    fake_funasr = types.ModuleType("funasr")
    class FakeModel:
        def __init__(self, model, device, **kw):
            self.model = model
            self.device = device
            self._calls = []
            # populate cache on first call if streaming model
            self._is_streaming = "streaming" in model
        def generate(self, input=None, cache=None, is_final=False, language=None, **kw):
            if cache is not None and not is_final and self._is_streaming:
                cache["enc"] = 1  # mark cache populated
            if is_final:
                if cache and cache.get("enc"):
                    return [{"text": " world"}]
                return [{"text": ""}]
            # non-stream batch vs streaming partial
            if self._is_streaming:
                return [{"text": "hello"}]
            return [{"text": "hello world"}]
    fake_funasr.AutoModel = FakeModel
    sys.modules["funasr"] = fake_funasr
    return fake_funasr


def test_language_routing():
    _install_fakes()
    from voicelang_core.adapters.funasr import FunASRTranscriber
    assert FunASRTranscriber(language="zh")._model_id == "paraformer-zh-streaming"
    assert FunASRTranscriber(language="zh-cn")._model_id == "paraformer-zh-streaming"
    assert FunASRTranscriber(language="yue")._model_id == "paraformer-zh-streaming"
    assert FunASRTranscriber(language="en")._model_id == "iic/SenseVoiceSmall"
    assert FunASRTranscriber(language="auto")._model_id == "iic/SenseVoiceSmall"
    assert FunASRTranscriber(language="en", model="custom/model")._model_id == "custom/model"
    # explicit model overrides
    assert FunASRTranscriber(language="zh", model="iic/SenseVoiceSmall")._is_streaming_model is False


def test_streaming_emits_partial_and_finish_tail():
    _install_fakes()
    from voicelang_core.adapters.funasr import FunASRTranscriber
    tr = FunASRTranscriber(language="zh")
    chunk = AudioChunk(pcm=np.zeros(1600, dtype=np.int16).tobytes(), sample_rate=16000)
    segs = list(tr.transcribe_stream(chunk))
    assert len(segs) == 1 and segs[0].source_text == "hello" and segs[0].status == "partial"
    fins = list(tr.finish())
    assert len(fins) == 1 and fins[0].source_text == "world" and fins[0].status == "final"


def test_batch_sensevoice_emits_final():
    _install_fakes()
    from voicelang_core.adapters.funasr import FunASRTranscriber
    tr = FunASRTranscriber(language="en")
    chunk = AudioChunk(pcm=np.zeros(1600, dtype=np.int16).tobytes(), sample_rate=16000)
    segs = list(tr.transcribe_stream(chunk))
    assert len(segs) == 1 and segs[0].source_text.strip() == "hello world" and segs[0].status == "final"
    # finish is no-op for batch
    assert list(tr.finish()) == []


def test_extract_text_variants():
    from voicelang_core.adapters.funasr import FunASRTranscriber
    assert FunASRTranscriber._extract_text([{"text": "hi"}]) == "hi"
    assert FunASRTranscriber._extract_text([{"sentence": "hi2"}]) == "hi2"
    assert FunASRTranscriber._extract_text([{"transcript": "hi3"}]) == "hi3"
    assert FunASRTranscriber._extract_text([{"text": ""}]) == ""
    assert FunASRTranscriber._extract_text([]) == ""
    assert FunASRTranscriber._extract_text(None) == ""
    assert FunASRTranscriber._extract_text([ "raw string"]) == "raw string"
    assert FunASRTranscriber._extract_text({"text": "dict"}) == "dict"


def test_transcriber_via_run_factory():
    _install_fakes()
    from voicelang_core.config import Config
    from voicelang_core.run import _build_transcriber
    tr = _build_transcriber("funasr", Config(language="zh"))
    assert tr.__class__.__name__ == "FunASRTranscriber"
    assert tr._model_id == "paraformer-zh-streaming"
    tr2 = _build_transcriber("funasr", Config(language="en"))
    assert tr2._model_id == "iic/SenseVoiceSmall"