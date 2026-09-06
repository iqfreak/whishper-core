"""Python shim: while Rust not compiled, expose same API via pyaudiowpatch fallback."""
try:
    from voicelang_audio import AudioCapture as _RustCapture  # noqa
    AudioCapture = _RustCapture
except ImportError:
    from voicelang_core.adapters.wasapi_loopback import WASAPILoopbackSource
    from voicelang_core.adapters.app_loopback import AppLoopbackSource
    class AudioCapture:
        def __init__(self, source="loopback", process=None, block_ms=80):
            self.source = source
            self.process = process
            self.block_ms = block_ms/1000
            if source == "app" and process:
                self._src = AppLoopbackSource(app=process, block_seconds=self.block_ms)
            else:
                self._src = WASAPILoopbackSource(block_seconds=self.block_ms)
        def __iter__(self):
            for chunk in self._src.stream():
                yield chunk.pcm
