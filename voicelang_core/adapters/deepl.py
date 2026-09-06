"""DeepL Translator adapter (official API).

Uses the official DeepL HTTP API. Free and Pro share the same shape;
only the host differs:
  Free: https://api-free.deepl.com/v2/translate
  Pro:  https://api.deepl.com/v2/translate
The caller passes the full endpoint via Config.translate_url, and the API key
via Config.translate_key (DeepL-Auth-Key).

Wire: translate="deepl" + translate_url (e.g. https://api-free.deepl.com)
      + translate_key (your API key)  + target_language (e.g. "FR")
"""
from ..ports import Translator
from ..types import Translation


class DeepLTranslator(Translator):
    is_passthrough = False

    def __init__(self, endpoint: str = "https://api-free.deepl.com",
                 api_key: str = "", timeout: float = 5.0):
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def translate(self, text: str, source: str, target: str) -> Translation:
        import requests  # noqa: PLC0415

        src = (source or "").upper() if source and source != "auto" else None
        tgt = (target or "EN").upper()
        # DeepL expects upper-case language codes, e.g. EN, FR, DE, JA
        # endpoint is e.g. https://api-free.deepl.com  -> POST /v2/translate
        url = f"{self._endpoint}/v2/translate"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"DeepL-Auth-Key {self._api_key}"
        payload = {"text": [text], "target_lang": tgt}
        if src:
            payload["source_lang"] = src
        # WP-1.3 FIX: 5s timeout + 1 fast retry + backoff, preserves ordering via caller reorder buffer
        import time as _time
        last_exc = None
        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
                resp.raise_for_status()
                break
            except Exception as exc:
                last_exc = exc
                # only retry on timeout / connection errors, not 4xx
                msg = str(exc).lower()
                is_timeout = "timeout" in msg or "timed out" in msg or "connection" in msg
                if attempt == 0 and is_timeout:
                    _time.sleep(0.3 * (attempt + 1))
                    continue
                raise
        else:
            if last_exc is not None:
                raise last_exc
        data = resp.json()
        # DeepL shape: {"translations": [{"detected_source_language": "EN", "text": "..."}]}
        translations = data.get("translations") or []
        if translations:
            out = translations[0].get("text", "")
        else:
            # fallback single-object shape some proxies return
            out = data.get("translatedText") or data.get("text") or ""
        return Translation(
            source_text=text, target_text=out,
            source_language=source, target_language=target,
        )
