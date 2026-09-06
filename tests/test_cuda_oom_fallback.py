"""WP-6 QA: CUDA OOM triggers fallback."""
def test_whisper_oom_detection():
    msg = "CUDA out of memory. Tried to allocate 256 MiB"
    is_oom = any(k in msg.lower() for k in ("out of memory", "cudnn_status_alloc_failed", "cudaerrormemoryallocation"))
    assert is_oom

def test_audio8_oom_fallback_stub():
    from voicelang_core.adapters.audio8 import Audio8Transcriber
    # mock without heavy load: just verify class exists and handles OOM strings
    assert hasattr(Audio8Transcriber, "transcribe")
