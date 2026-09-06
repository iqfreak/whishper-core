"""WP-6 QA: unknown transcriber does not silently resolve to Whisper."""
from voicelang_core.config import Config

def test_unknown_transcriber_fallback():
    cfg = Config.from_dict({"transcriber": "unknown_engine_xyz"})
    assert cfg.transcriber == "moonshine"

def test_known_transcriber_passes():
    cfg = Config.from_dict({"transcriber": "vibeasr-bitnet"})
    # vibeasr-bitnet is in known set per our config patch
    assert cfg.transcriber in ("vibeasr-bitnet", "vibeasr")
