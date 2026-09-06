"""LibreTranslate Translator adapter. reuses voicelang's translation behavior."""
from .base import Translator  # noqa: F401
from ..types import Translation


class LibreTranslateTranslator(Translator):
    is_passthrough = False

    def __init__(self, endpoint: str = "http://translate:5000", timeout: float = 30.0):
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def translate(self, text: str, source: str, target: str) -> Translation:
        # Lazy: requests is a 'real' extra; import it here so importing this
        # module never hard-requires the HTTP dep, and a test-stubbed
        # sys.modules['requests'] can't poison this module's global forever.
        import requests  # noqa: PLC0415

        src = "" if source in (None, "auto") else source
        resp = requests.post(
            f"{self._endpoint}/translate",
            json={"q": text, "source": src, "target": target, "format": "text"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        target_text = resp.json().get("translatedText", "")
        return Translation(
            source_text=text, target_text=target_text,
            source_language=source, target_language=target,
        )
