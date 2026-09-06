"""WP-6 QA: capture-exception produces warn + exit."""
from voicelang_core.pipeline_threaded import ThreadedPipeline
from voicelang_core.types import AudioChunk

class RaisingSource:
    def stream(self):
        raise RuntimeError("mic permission denied")
        yield

class CapturingDisplay:
    def __init__(self):
        self.warns = []
    def show(self, seg): pass
    def show_partial(self, seg): pass
    def warn(self, msg): self.warns.append(msg)

def test_capture_exception_warns():
    from voicelang_core.adapters.mic import MicSource
    display = CapturingDisplay()
    pipe = ThreadedPipeline(source=RaisingSource(), transcriber=None, translator=None, display=display)
    # _capture_loop should not propagate silently; it warns and exits(20) on real path
    # For unit test we check the warn path via direct call with mocked _exit
    import os
    orig_exit = os._exit
    called = {}
    def fake_exit(code):
        called["code"] = code
        raise SystemExit(code)
    os._exit = fake_exit
    try:
        try:
            pipe._capture_loop()
        except SystemExit as e:
            assert e.code == 20 or called.get("code") == 20
    finally:
        os._exit = orig_exit
    assert any("capture error" in w or "mic" in w.lower() for w in display.warns)
