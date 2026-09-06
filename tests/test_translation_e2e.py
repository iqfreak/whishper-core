"""Translation E2E through the REAL seams (config -> translator -> pipeline -> overlay).

Regression: run.py hardcoded PassthroughTranslator(), so translate/libretranslate
config was dead -- translation to a different language never happened in the app.
Here a local LibreTranslate-compatible HTTP server stands in for the third-party
server, a stub transcriber stands in for ASR (ASR itself is verified live by the
moonshine E2E), and the real LibreTranslateTranslator + Pipeline instance route
translated text into OverlayDisplay (headless) for capture.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from voicelang_core.adapters.overlay import OverlayDisplay
from voicelang_core.config import Config
from voicelang_core.pipeline import Pipeline
from voicelang_core.ports import AudioSource, StreamingTranscriber
from voicelang_core.run import _build_translator
from voicelang_core.types import AudioChunk, Segment

# English lives here; the fake LibreTranslate maps to French.
PAIRS = {
    "The mission starts at dawn.": "La mission commence à l'aube.",
    "Move to the extraction point.": "Déplacez-vous au point d'extraction.",
}


class _FakeTranslate(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        text = body.get("q", "")
        out = PAIRS.get(text, text)
        resp = json.dumps({"translatedText": out}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a):  # noqa: D102
        pass


class _FakeAudio(AudioSource):
    def __init__(self):
        self._n = 2

    def stream(self):
        for _ in range(self._n):
            yield AudioChunk(pcm=b"\x00\x00", sample_rate=16000)


class _FakeST(StreamingTranscriber):
    def __init__(self, texts):
        self._queue = list(texts)

    def transcribe_stream(self, chunk):  # noqa: D102
        if self._queue:
            yield Segment(status="final",
                          source_text=self._queue.pop(0),
                          source_language="en")

    def transcribe(self, chunk):  # noqa: D102
        return None


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _FakeTranslate)
    thr = threading.Thread(target=srv.serve_forever, daemon=True)
    thr.start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_config_roundtrip_translate_fields():
    cfg = Config(translate="libretranslate", translate_url="http://x:1",
                 target_language="fr")
    cfg2 = Config.from_dict(cfg.to_dict())
    assert cfg2.translate == "libretranslate"
    assert cfg2.translate_url == "http://x:1"
    assert cfg2.target_language == "fr"


def test_build_translator_wires_libretranslate():
    assert _build_translator(Config()).is_passthrough is True
    tr = _build_translator(Config(translate="libretranslate",
                                  translate_url="http://x:1"))
    assert tr.is_passthrough is False
    # configured libretranslate but no URL -> fall back to passthrough (safe)


def test_translation_flows_to_overlay():
    srv, url = _serve()
    try:
        cfg = Config(translate="libretranslate", translate_url=url,
                     target_language="fr")
        translator = _build_translator(cfg)
        disp = OverlayDisplay(headless=True)
        pipeline = Pipeline(
            source=_FakeAudio(),
            transcriber=_FakeST(list(PAIRS.keys())),
            translator=translator,
            display=disp,
            target_language="fr",
        )
        pipeline.run_streaming()
    finally:
        srv.shutdown()

    rendered = "\n".join(disp.lines)
    assert "La mission commence à l'aube." in rendered
    assert "Déplacez-vous au point d'extraction." in rendered
    # Shared monotonic ids reached the display: [1] and [2].
    assert "[1] La mission commence à l'aube." in rendered
    assert "[2] Déplacez-vous au point d'extraction." in rendered