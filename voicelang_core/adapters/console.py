"""Console display adapter (dependency-free)."""
import sys

from .base import Display
from ..types import Segment


class ConsoleDisplay(Display):
    def __init__(self) -> None:
        # Windows consoles default to cp1252, which cannot encode Arabic/CJK
        # transcripts -> UnicodeEncodeError in print() (regression seen with
        # Arabic partials). Re-point at UTF-8 and never crash on unmappable
        # characters, whatever the underlying stream is.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            try:
                sys.stdout.reconfigure(errors="replace")
            except (AttributeError, ValueError, OSError):
                pass
        self._stdout = sys.stdout

    def show(self, segment: Segment) -> None:
        # Numbered final: same shared id as the live partial it replaces.
        if self._stdout is None:  # frozen --windowed child with no redirection
            return
        text = segment.translated_text
        if text is None:
            # Graceful degradation (spec §9): translation failed — keep the
            # source, mark the block, never crash the worker.
            text = f"⚠ {segment.source_text} (translation unavailable)"
        self._stdout.write(f"\n[{segment.id}] {text}\n")
        self._stdout.flush()

    def show_partial(self, segment: Segment) -> None:
        # Live hypothesis preview: raw (untranslated) text, rewritten IN PLACE
        # under its existing id via carriage return — no new id, no new line.
        if self._stdout is None:  # frozen --windowed child with no redirection
            return
        self._stdout.write(f"\r[{segment.id}] {segment.source_text}          ")
        self._stdout.flush()

    def warn(self, text: str) -> None:
        # Non-fatal advisory (no-audio watchdog etc.) — visible, not a crash.
        if self._stdout is None:
            return
        self._stdout.write(f"\n[voicelang] ⚠ {text}\n")
        self._stdout.flush()