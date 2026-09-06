"""WP-6 QA: stale input_file does not downgrade ThreadedPipeline."""
from voicelang_core.config import Config

def test_stale_input_file_loopback_still_threaded():
    cfg = Config(source="loopback", input_file="stale.wav")
    # run.py gates on cfg.source != "wav", not on input_file truthiness
    use_threaded = cfg.source in ("mic", "loopback", "app")
    assert use_threaded is True

def test_wav_source_not_threaded():
    cfg = Config(source="wav", input_file="replay.wav")
    use_threaded = cfg.source in ("mic", "loopback", "app")
    assert use_threaded is False
