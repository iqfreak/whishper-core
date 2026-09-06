"""Google Translator adapter (free, no API key).

Uses the undocumented but long-stable translate.googleapis.com endpoint that
the web UI itself uses. No key, no quota header — best-effort only.
For production you should use the paid Cloud Translation API, but this
covers the \"just works without a key\" case alongside LibreTranslate/DeepLX.

Shape:
  GET https://translate.googleapis.com/translate_a/single?client=gtx&sl={src}&tl={tgt}&dt=t&q={text}
  -> JSON array where result[0][0][0] is the translated string.

We implement both GET and a POST fallback (long texts).
"""
from ..ports import Translator
from ..types import Translation


class GoogleTranslator(Translator):
    is_passthrough = False

    def __init__(self, endpoint: str = "https://translate.googleapis.com",
                 timeout: float = 30.0):
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def translate(self, text: str, source: str, target: str) -> Translation:
        import requests  # noqa: PLC0415

        src = (source or "auto").lower() if source else "auto"
        if src == "":
            src = "auto"
        tgt = (target or "en").lower()
        # Google wants lower-case codes: en, fr, ar, zh, ja ...
        params = {
            "client": "gtx",
            "sl": src,
            "tl": tgt,
            "dt": "t",
            "q": text,
        }
        url = f"{self._endpoint}/translate_a/single"
        # Prefer GET for short; POST for long (avoids URL length limits)
        if len(text) < 1500:
            resp = requests.get(url, params=params, timeout=self._timeout)
        else:
            resp = requests.post(url, data=params, timeout=self._timeout)
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            # Fallback: some proxies return {"translatedText": "..."}
            try:
                j = resp.json()
                out = j.get("translatedText") or j.get("text") or ""
                return Translation(source_text=text, target_text=out,
                                   source_language=source, target_language=target)
            except Exception:
                raise
        # Standard shape: [[[translated, original, ...], ...], ...]
        try:
            # data[0] is list of sentence chunks
            chunks = data[0] if isinstance(data, list) and len(data) > 0 else []
            out = "".join(c[0] for c in chunks if isinstance(c, list) and len(c) > 0 and c[0])
            if not out and isinstance(data, dict):
                out = data.get("translatedText") or ""
        except Exception:
            out = ""
        return Translation(
            source_text=text, target_text=out,
            source_language=source, target_language=target,
        )
