"""WP-6 QA: GUI provider/language cross-validation."""
import pytest
from voicelang_core.engines import get_registry

def test_funasr_nano_rejects_invalid_language():
    from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber
    with pytest.raises(ValueError):
        FunAsrNanoTranscriber(language="xx")

def test_vibeasr_rejects_invalid_language():
    from voicelang_core.adapters.vibeasr import VibeAsrTranscriber
    with pytest.raises(ValueError):
        VibeAsrTranscriber(language="xx")

def test_audio8_rejects_invalid_language():
    from voicelang_core.adapters.audio8 import Audio8Transcriber
    with pytest.raises(ValueError):
        Audio8Transcriber(language="xx")

def test_registry_consistency():
    reg = get_registry()
    assert "vibeasr-bitnet" in reg or "vibeasr" in reg
    assert "audio8" in reg
    for name, entry in reg.items():
        assert entry["languages"], f"{name} missing language catalog"
        assert "streaming" in entry["capabilities"]
