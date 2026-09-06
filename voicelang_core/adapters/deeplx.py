"""DeepLX Translator adapter (free DeepL proxy).

DeepLX is an unofficial free/open-source proxy that exposes a simpler
endpoint compatible with DeepL's translation quality without an API key:
  POST {endpoint}/translate  json {text, source_lang, target_lang}
  -> {"code":200,"data":"...","alternatives":[...]}

Default self-hosted address is http://127.0.0.1:1188 ; many public
instances exist. This adapter speaks the DeepLX shape, not the official
DeepL shape (see adapters/deepl.py for the official API).
"""
from ..ports import Translator
from ..types import Translation


class DeepLXTranslator(Translator):
    is_passthrough = False

    def __init__(self, endpoint: str = "http://127.0.0.1:1188", timeout: float = 30.0):
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def translate(self, text: str, source: str, target: str) -> Translation:
        import requests  # noqa: PLC0415

        src = (source or "auto").upper() if source else "AUTO"
        # DeepLX historically wants e.g. "EN", "FR", "ZH"; auto is "AUTO"
        if src == "":
            src = "AUTO"
        tgt = (target or "EN").upper()
        url = f"{self._endpoint}/translate"
        payload = {"text": text, "source_lang": src, "target_lang": tgt}
        resp = requests.post(url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        # DeepLX shapes: {"code":200,"data":"translated"}  or {"translatedText":"..."}
        out = data.get("data")
        if out is None:
            out = data.get("translatedText") or data.get("text") or ""
            # DeepL-proxy compat: {"translations":[{"text":"..."}]}
            if not out and data.get("translations"):
                out = data["translations"][0].get("text", "")
        return Translation(
            source_text=text, target_text=out if isinstance(out, str) else str(out),
            source_language=source, target_language=target,
        )
