"""Translate adapters + pipeline E2E for all modes (libretranslate / deepl / deeplx / google / passthrough).

Uses localhost HTTP servers that speak each adapter's wire format (same pattern as
test_translation_e2e's _FakeTranslate). ASR is faked; the point is to prove the
translator seam + Pipeline._render + OverlayDisplay flow is wired end-to-end for every
supported translate mode.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from voicelang_core.adapters.overlay import OverlayDisplay
from voicelang_core.config import Config
from voicelang_core.pipeline import Pipeline
from voicelang_core.ports import AudioSource, StreamingTranscriber
from voicelang_core.run import _build_translator
from voicelang_core.types import AudioChunk, Segment


PAIRS = {
    "The mission starts at dawn.": "La mission commence à l'aube.",
    "Move to the extraction point.": "Déplacez-vous au point d'extraction.",
}


class _FakeAudio(AudioSource):
    def __init__(self, n=2):
        self._n = n
    def stream(self):
        for _ in range(self._n):
            yield AudioChunk(pcm=b"\x00\x00", sample_rate=16000)


class _FakeST(StreamingTranscriber):
    def __init__(self, texts):
        self._queue = list(texts)
    def transcribe_stream(self, chunk):
        if self._queue:
            yield Segment(status="final",
                          source_text=self._queue.pop(0),
                          source_language="en")
    def transcribe(self, chunk):
        return None


# --- per-provider HTTP handlers that mimic the real APIs --------------------

class _LT(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        out = PAIRS.get(body.get("q", ""), body.get("q", ""))
        resp = json.dumps({"translatedText": out}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self,*a): pass


class _Deepl(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n=int(self.headers.get("Content-Length",0))
        body=json.loads(self.rfile.read(n))
        # body is {"text":["..."], "target_lang":"FR"} etc.
        # Map single text or first element
        inp = body.get("text", [""])
        if isinstance(inp, list): inp = inp[0] if inp else ""
        out = PAIRS.get(inp, inp)
        resp=json.dumps({"translations":[{"text": out}]}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(resp))); self.end_headers(); self.wfile.write(resp)
    def log_message(self,*a): pass


class _Deeplx(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n=int(self.headers.get("Content-Length",0))
        body=json.loads(self.rfile.read(n))
        inp = body.get("text","")
        out = PAIRS.get(inp, inp)
        resp=json.dumps({"code":200,"data": out}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(resp))); self.end_headers(); self.wfile.write(resp)
    def log_message(self,*a): pass


class _Google(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        from urllib.parse import urlparse, parse_qs
        qs=parse_qs(urlparse(self.path).query)
        q=qs.get("q",[""])[0]
        # Also handle POST-encoded?
        out=PAIRS.get(q,q)
        resp=json.dumps([[[out, q]]]).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(resp))); self.end_headers(); self.wfile.write(resp)
    def do_POST(self): self.do_GET()  # noqa: N802
    def log_message(self,*a): pass


def _serve(handler):
    srv=HTTPServer(("127.0.0.1",0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def _run_pipeline_translates(translator_mode, handler):
    srv,url=_serve(handler)
    try:
        if translator_mode=="deepl":
            cfg=Config(translate="deepl", translate_url=url, translate_key="fake-key", target_language="fr")
        else:
            cfg=Config(translate=translator_mode, translate_url=url, target_language="fr")
        tr=_build_translator(cfg)
        assert tr.is_passthrough is False, translator_mode
        disp=OverlayDisplay(headless=True)
        pipe=Pipeline(source=_FakeAudio(), transcriber=_FakeST(list(PAIRS.keys())), translator=tr, display=disp, target_language="fr")
        pipe.run_streaming()
    finally: srv.shutdown()
    rendered="\n".join(disp.lines)
    assert "La mission commence à l'aube." in rendered
    assert "Déplacez-vous au point d'extraction." in rendered
    # Shared monotonic ids reached the overlay for every provider.
    assert "[1] La mission commence à l'aube." in rendered
    assert "[2] Déplacez-vous au point d'extraction." in rendered


def test_libretranslate_e2e():
    _run_pipeline_translates("libretranslate", _LT)

def test_deepl_e2e():
    _run_pipeline_translates("deepl", _Deepl)

def test_deeplx_e2e():
    _run_pipeline_translates("deeplx", _Deeplx)

def test_google_e2e():
    _run_pipeline_translates("google", _Google)

def test_passthrough_does_not_translate():
    tr=_build_translator(Config(translate="passthrough", target_language="fr"))
    assert tr.is_passthrough is True
    disp=OverlayDisplay(headless=True)
    pipe=Pipeline(source=_FakeAudio(n=1), transcriber=_FakeST(["hello world"]), translator=tr, display=disp, target_language="fr")
    pipe.run_streaming()
    rendered="\n".join(disp.lines)
    assert "hello world" in rendered
    # passthrough still stamps the shared id on the line.
    assert "[1] hello world" in rendered

def test_deepl_requires_key_raises():
    with pytest.raises(SystemExit, match="translate_key"):
        _build_translator(Config(translate="deepl", translate_url="http://x:1", translate_key=""))

def test_libretranslate_no_url_falls_back_to_passthrough():
    # Agent4 fix: misconfigured libretranslate without URL must raise clear error, not silently fall back
    with pytest.raises(SystemExit, match="Translation is set to libretranslate but no API key/URL configured"):
        _build_translator(Config(translate="libretranslate", translate_url="", target_language="fr"))