"""Agent4 — Translation Not Happening (Root Cause Hunt) verification.

Covers:
- misconfigured URL/key surfaces clear error (not silent passthrough)
- every translation mode e2e with two utterances back-to-back (no mixing/drops)
- slow-translation segment-tracking correctness (immutable snapshot + FIFO order)
- runtime failure surfaces warn
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import pytest

from voicelang_core.adapters.overlay import OverlayDisplay
from voicelang_core.adapters.fake import FakeAudioSource, FakeDisplay
from voicelang_core.config import Config
from voicelang_core.engines import EngineConfig, EngineName, _build_translator as eng_build
from voicelang_core.run import _build_translator as run_build
from voicelang_core.pipeline import Pipeline
from voicelang_core.pipeline_threaded import ThreadedPipeline
from voicelang_core.ports import AudioSource, StreamingTranscriber
from voicelang_core.types import AudioChunk, Segment, Translation

# --- helpers: mock translators that mimic real adapters without HTTP ---

class MockTranslator:
    """Generic mock that translates via dict or echoes with target prefix."""
    is_passthrough = False
    def __init__(self, pairs=None, delay=0.0, fail=False):
        self.pairs = pairs or {}
        self.delay = delay
        self.fail = fail
        self.calls = []
    def translate(self, text, source, target):
        self.calls.append((text, source, target))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("mock endpoint down")
        out = self.pairs.get(text, f"[{target}] {text}")
        return Translation(source_text=text, target_text=out, source_language=source, target_language=target)

class SlowTranslator(MockTranslator):
    """Delay only first call longer to exercise ordering."""
    def translate(self, text, source, target):
        # first utterance slower than second
        d = self.delay if text == "hello world" else 0.02
        return super().translate(text, source, target) if False else MockTranslator.translate(self, text, source, target)  # placeholder

# --- HTTP handlers for each real mode (reuse pattern from test_translate_adapters) ---

PAIRS = {
    "hello world": "hola mundo",
    "second utterance": "segunda frase",
}

class _LT(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        out = PAIRS.get(body.get("q", ""), body.get("q", ""))
        resp = json.dumps({"translatedText": out}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self, *a): pass

class _Deepl(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        inp = body.get("text", [""])
        if isinstance(inp, list):
            inp = inp[0] if inp else ""
        out = PAIRS.get(inp, inp)
        resp = json.dumps({"translations": [{"text": out}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self, *a): pass

class _Deeplx(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        inp = body.get("text", "")
        out = PAIRS.get(inp, inp)
        resp = json.dumps({"code": 200, "data": out}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self, *a): pass

class _Google(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        q = qs.get("q", [""])[0]
        out = PAIRS.get(q, q)
        resp = json.dumps([[[out, q]]]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def do_POST(self): self.do_GET()
    def log_message(self, *a): pass

def _serve(handler):
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"

class _FakeAudio2(AudioSource):
    def stream(self):
        for _ in range(2):
            yield AudioChunk(pcm=b"\x00\x00", sample_rate=16000)

class _FakeST2(StreamingTranscriber):
    def __init__(self, texts):
        self._q = list(texts)
    def transcribe_stream(self, chunk):
        if self._q:
            yield Segment(status="final", source_text=self._q.pop(0), source_language="en")
    def transcribe(self, chunk): return None

# --- 1. Misconfigured cases must raise clear error, not silently passthrough ---

def test_libretranslate_missing_url_raises_clear_error_engines():
    cfg = EngineConfig(name=EngineName.passthrough, translate="libretranslate", translate_url=None, translate_key=None, target_language="en")
    with pytest.raises(SystemExit, match="Translation is set to libretranslate but no API key/URL configured"):
        eng_build(cfg)

def test_libretranslate_missing_url_raises_clear_error_run():
    cfg = Config(translate="libretranslate", translate_url="", target_language="en")
    with pytest.raises(SystemExit, match="Translation is set to libretranslate but no API key/URL configured"):
        run_build(cfg)

def test_deeplx_missing_url_raises_clear_error_engines():
    cfg = EngineConfig(name=EngineName.passthrough, translate="deeplx", translate_url="", translate_key=None, target_language="en")
    with pytest.raises(SystemExit, match="Translation is set to deeplx but no API key/URL configured"):
        eng_build(cfg)

def test_deeplx_missing_url_raises_clear_error_run():
    cfg = Config(translate="deeplx", translate_url="   ", target_language="en")
    with pytest.raises(SystemExit, match="Translation is set to deeplx but no API key/URL configured"):
        run_build(cfg)

def test_deepl_missing_key_still_raises():
    cfg = Config(translate="deepl", translate_url="https://api-free.deepl.com", translate_key="", target_language="de")
    with pytest.raises(SystemExit, match="translate_key"):
        run_build(cfg)

def test_google_without_url_does_not_raise():
    # google has sensible default, empty URL should be OK
    cfg = Config(translate="google", translate_url="", target_language="fr")
    tr = run_build(cfg)
    assert tr.is_passthrough is False

# --- 2. Runtime failure surfaces warn, not silence ---

def test_translation_failure_surfaces_warn_not_silence():
    disp = FakeDisplay()
    # also track warns
    warns = []
    orig_warn = disp.warn
    def _warn(txt):
        warns.append(txt)
        orig_warn(txt)
    disp.warn = _warn  # type: ignore
    tr = MockTranslator(fail=True)
    p = Pipeline(source=FakeAudioSource([AudioChunk(pcm=b"\x00"*320)]), transcriber=_FakeST2(["bonjour"]), translator=tr, display=disp, target_language="en")
    # Pipeline.run_streaming will call _render which should warn
    p.run_streaming()
    assert len(disp.shown) == 1
    assert disp.shown[0].translated_text is None
    # warn must contain mode hint
    assert any("Translation is set to" in w for w in warns), f"no warn surfaced, got {warns}"

def test_misconfigured_runtime_shows_error_not_silence_threaded():
    # Use ThreadedPipeline with failing translator to verify warn in worker
    warns = []
    class WarnDisplay(FakeDisplay):
        def warn(self, text):
            warns.append(text)
    disp = WarnDisplay()
    tr = MockTranslator(fail=True)
    # Build threaded pipeline directly, push two finals via _handle_segment then run worker manually
    pipe = ThreadedPipeline(source=FakeAudioSource([]), transcriber=_FakeST2(["hello world"]), translator=tr, display=disp, target_language="en")
    # Enqueue via _handle_segment (worker not started; drain manually)
    seg = Segment(id=0, status="final", source_text="hello world", source_language="en")
    pipe._handle_segment(seg)
    # Manually drain translate_q as worker would
    assert pipe._translate_q.qsize() == 1
    snap = pipe._translate_q.get_nowait()
    # render should warn
    pipe._render(snap, tr)
    assert snap.translated_text is None
    assert any("Translation is set to" in w for w in warns)

# --- 3. Two utterances back-to-back with slow translation: correct attachment, no mixing/drops ---

def test_slow_translation_correct_utterance_attachment_threaded():
    """Two finals enqueued instantly, translation sleeps 0.15s per call — ids and texts must not cross."""
    pairs = {"hello world": "HALLO WELT", "second utterance": "ZWEITE ÄUSSERUNG"}
    tr = MockTranslator(pairs=pairs, delay=0.15)
    disp = FakeDisplay()
    # Use small translate queue to also test drop not happening for 2 items
    pipe = ThreadedPipeline(source=FakeAudioSource([]), transcriber=_FakeST2([]), translator=tr, display=disp, target_language="de", translate_queue_size=8)
    # Start translate worker for real timing
    import threading as _thr
    pipe._stop.clear()
    worker = _thr.Thread(target=pipe._translate_worker, daemon=True)
    worker.start()
    # Enqueue two finals back-to-back (simulates ASR fast, translation slow)
    seg_a = Segment(id=0, status="final", source_text="hello world", source_language="en")
    seg_b = Segment(id=0, status="final", source_text="second utterance", source_language="en")
    pipe._handle_segment(seg_a)
    pipe._handle_segment(seg_b)
    # Verify ids assigned synchronously before translation
    assert seg_a.id == 1
    assert seg_b.id == 2
    # Wait for worker to process both
    time.sleep(0.6)
    pipe._stop.set()
    worker.join(timeout=1.0)
    # Display should have two finals with correct translations attached to correct ids
    assert len(disp.shown) == 2, f"expected 2, got {len(disp.shown)}"
    # Order preserved (FIFO)
    assert disp.shown[0].id == 1 and disp.shown[0].source_text == "hello world" and disp.shown[0].translated_text == "HALLO WELT"
    assert disp.shown[1].id == 2 and disp.shown[1].source_text == "second utterance" and disp.shown[1].translated_text == "ZWEITE ÄUSSERUNG"
    # No mixing
    assert disp.shown[0].translated_text != disp.shown[1].translated_text

def test_threaded_id_monotonic_under_slow_translation():
    """Regression for id race: _open_id cleared synchronously so second utterance gets new id."""
    tr = MockTranslator(delay=0.3)
    disp = FakeDisplay()
    pipe = ThreadedPipeline(source=FakeAudioSource([]), transcriber=_FakeST2([]), translator=tr, display=disp, target_language="en")
    seg_a = Segment(id=0, status="final", source_text="first", source_language="en")
    seg_b = Segment(id=0, status="final", source_text="second", source_language="en")
    pipe._handle_segment(seg_a)
    assert pipe._open_id is None
    assert pipe._next_id == 2
    pipe._handle_segment(seg_b)
    assert seg_b.id == 2
    assert seg_a.id == 1
    assert seg_a.id != seg_b.id

# --- 4. End-to-end for each mode with two utterances (mock HTTP) ---

@pytest.mark.parametrize("mode,handler", [
    ("libretranslate", _LT),
    ("deepl", _Deepl),
    ("deeplx", _Deeplx),
    ("google", _Google),
])
def test_each_mode_e2e_two_utterances(mode, handler):
    srv, url = _serve(handler)
    try:
        if mode == "deepl":
            cfg = Config(translate=mode, translate_url=url, translate_key="fake-key", target_language="fr")
        else:
            cfg = Config(translate=mode, translate_url=url, target_language="fr")
        tr = run_build(cfg)
        assert tr.is_passthrough is False, mode
        disp = OverlayDisplay(headless=True)
        pipe = Pipeline(source=_FakeAudio2(), transcriber=_FakeST2(list(PAIRS.keys())), translator=tr, display=disp, target_language="fr")
        pipe.run_streaming()
    finally:
        srv.shutdown()
    rendered = "\n".join(disp.lines)
    # Both translations must appear, with correct shared ids, no mixing
    assert "hola mundo" in rendered, f"{mode} missing first translation"
    assert "segunda frase" in rendered, f"{mode} missing second translation"
    assert "[1] hola mundo" in rendered
    assert "[2] segunda frase" in rendered

def test_passthrough_two_utterances_still_id_stamped():
    tr = run_build(Config(translate="passthrough", target_language="en"))
    assert tr.is_passthrough is True
    disp = OverlayDisplay(headless=True)
    pipe = Pipeline(source=_FakeAudio2(), transcriber=_FakeST2(["hello world", "second utterance"]), translator=tr, display=disp, target_language="en")
    pipe.run_streaming()
    rendered = "\n".join(disp.lines)
    assert "[1] hello world" in rendered
    assert "[2] second utterance" in rendered

def test_threaded_two_utterances_e2e_with_mock_translator():
    """ThreadedPipeline end-to-end with mock translator (no HTTP) two back-to-back."""
    tr = MockTranslator(pairs=PAIRS, delay=0.05)
    disp = FakeDisplay()
    # Use direct _handle_segment + worker to avoid needing full capture loop
    pipe = ThreadedPipeline(source=FakeAudioSource([]), transcriber=_FakeST2([]), translator=tr, display=disp, target_language="fr")
    import threading as _thr
    pipe._stop.clear()
    tr_t = _thr.Thread(target=pipe._translate_worker, daemon=True)
    tr_t.start()
    seg_a = Segment(id=0, status="final", source_text="hello world", source_language="en")
    seg_b = Segment(id=0, status="final", source_text="second utterance", source_language="en")
    pipe._handle_segment(seg_a)
    pipe._handle_segment(seg_b)
    time.sleep(0.6)
    pipe._stop.set()
    tr_t.join(timeout=1.0)
    assert len(disp.shown) == 2
    assert disp.shown[0].translated_text == "hola mundo"
    assert disp.shown[1].translated_text == "segunda frase"
    assert disp.shown[0].id == 1 and disp.shown[1].id == 2
