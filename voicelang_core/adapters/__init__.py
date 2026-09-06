"""Edge adapters. Heavy deps are imported only inside the adapters that need
them, so the core + tests stay dependency-free.
"""
from .fake import (FakeAudioSource, FakeTranscriber, FakeTranslator,
                   FakeDisplay, FakeStreamingTranscriber, FakeCaptionSource)
from .console import ConsoleDisplay
from .passthrough import PassthroughTranslator
from .overlay import OverlayDisplay

__all__ = [
    "FakeAudioSource", "FakeTranscriber", "FakeTranslator", "FakeDisplay",
    "FakeStreamingTranscriber", "FakeCaptionSource", "ConsoleDisplay",
    "PassthroughTranslator", "OverlayDisplay",
]
