"""English-only passthrough Translator.

For the chosen path, Whisper's `task="translate"` already produces English in
one pass, so translation is a no-op. Flagged `is_passthrough` so the pipeline
skips the network hop entirely.
"""
from .base import Translator  # noqa: F401
from ..types import Translation


class PassthroughTranslator(Translator):
    @property
    def is_passthrough(self) -> bool:  # type: ignore[override]
        return True

    def translate(self, text, source, target):
        return Translation(
            source_text=text, target_text=text,
            source_language=source, target_language=target,
        )
