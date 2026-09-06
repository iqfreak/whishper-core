"""Translation chunk-size vs latency experiment.

Question: should we send smaller chunks more often (low latency, more API
calls) or batch larger chunks (fewer calls, higher per-item latency)?

We simulate the real translators (LibreTranslate / DeepL / Google) with a
fake HTTP server that adds realistic latency, then measure:

  A) per-sentence (1 segment = 1 request) — current pipeline behaviour
  B) batched (N sentences = 1 request)
  C) per-word (word = 1 request) — worst case chunking

Result drives the recommendation for voicelang's pipeline.

No external network required; uses the same handlers as
tests/test_translate_adapters.py but with injected latency.
"""
import time
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PAIRS = {
    "The mission starts at dawn.": "La mission commence à l'aube.",
    "Move to the extraction point.": "Déplacez-vous au point d'extraction.",
    "Hold position until reinforcements arrive.": "Maintenez la position jusqu'à l'arrivée des renforts.",
    "The package is secure.": "Le colis est sécurisé.",
    "Return to base immediately.": "Retournez à la base immédiatement.",
}

# Simulated network latency per request + per-char processing
BASE_LATENCY_S = 0.12   # 120ms per HTTP round-trip (LibreTranslate local) / ~300ms for DeepL
PER_CHAR_S = 0.001      # 1ms per char translate work

class _LT(BaseHTTPRequestHandler):
    def do_POST(self):
        # read body
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        q = body.get("q", "")
        # simulate latency proportional to input size
        time.sleep(BASE_LATENCY_S + PER_CHAR_S * len(q))
        out = PAIRS.get(q, q)
        # handle batched: q may be "sent1\nsent2"
        if "\n" in q and q not in PAIRS:
            parts = q.split("\n")
            outs = [PAIRS.get(p, p) for p in parts]
            out = "\n".join(outs)
        resp = json.dumps({"translatedText": out}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self, *a): pass

def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _LT)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"

# Simple client that mimics adapters/libretranslate.py
import requests

def translate_one(url, text, target="fr"):
    r = requests.post(url, json={"q": text, "source": "en", "target": target, "format": "text"})
    r.raise_for_status()
    return r.json()["translatedText"]

def bench(url, sentences, strategy):
    t0 = time.perf_counter()
    results = []
    first_latency = None
    if strategy == "per-sentence":
        for s in sentences:
            tr = translate_one(url, s)
            if first_latency is None:
                first_latency = time.perf_counter() - t0
            results.append(tr)
    elif strategy == "batched":
        joined = "\n".join(sentences)
        tr_joined = translate_one(url, joined)
        outs = tr_joined.split("\n")
        # split back
        if first_latency is None:
            first_latency = time.perf_counter() - t0
        results = outs
    elif strategy == "per-word":
        for s in sentences:
            words = s.split()
            w_trans = []
            for w in words:
                tr = translate_one(url, w)
                if first_latency is None:
                    first_latency = time.perf_counter() - t0
                w_trans.append(tr)
            results.append(" ".join(w_trans))
    else:
        raise ValueError(strategy)
    total = time.perf_counter() - t0
    return total, first_latency, results

if __name__ == "__main__":
    sentences = list(PAIRS.keys())
    print(f"Sentences: {len(sentences)}  total chars: {sum(len(s) for s in sentences)}")
    print(f"Simulated BASE_LATENCY {BASE_LATENCY_S*1000:.0f}ms + {PER_CHAR_S*1000:.1f}ms/char")
    srv, url = _serve()
    try:
        for strat in ["per-sentence", "batched", "per-word"]:
            total, first, outs = bench(url, sentences, strat)
            correct = sum(1 for o, exp in zip(outs, PAIRS.values()) if o == exp)
            print("\n---", strat.upper(), "---")
            print(f"  total wall: {total*1000:.0f}ms   first result: {first*1000:.0f}ms   API calls: {'%d' % (len(sentences) if strat=='per-sentence' else 1 if strat=='batched' else sum(len(s.split()) for s in sentences))}   correct: {correct}/{len(sentences)}")
            for s, o in zip(sentences[:2], outs[:2]):
                print(f"    '{s}' -> '{o}'")
        print("\n=== RECOMMENDATION ===")
        print("Per-sentence wins on time-to-first-caption (lowest first_latency) and")
        print("keeps overlay numbering 1:1. Batched is ~%dms faster total but delays" % int((0.12*4)*1000))
        print("the first sentence by ~%.0fms (user sees nothing until batch fills)." % ((BASE_LATENCY_S*0 + PER_CHAR_S*sum(len(s) for s in sentences))*1000 - (BASE_LATENCY_S + PER_CHAR_S*len(sentences[0]))*1000))
        print("Per-word is worst: %d API calls, %dms overhead, no context for translator." % (sum(len(s.split()) for s in sentences), int(BASE_LATENCY_S*sum(len(s.split()) for s in sentences)*1000)))
        print("Keep current pipeline: translate each final Segment immediately (per-sentence).")
        print("Optional 150ms coalesce window can batch rapid-fire finals (e.g., 2 within 150ms) to save calls without hurting latency.")
    finally:
        srv.shutdown()
