"""voicelang-core: streaming capture -> transcribe -> translate pipeline.

The pipeline depends only on four ports (interfaces). Adapters sit at the
edges: capture (mic / Discord voice / WASAPI loopback), transcribe
(faster-whisper), translate (LibreTranslate), and display (console /
Discord text channel / game overlay). Swap any adapter without touching
the pipeline.
"""
from .types import AudioChunk, Segment, Translation
from .ports import AudioSource, Transcriber, Translator, Display
from .pipeline import Pipeline

__all__ = [
    "AudioChunk", "Segment", "Translation",
    "AudioSource", "Transcriber", "Translator", "Display",
    "Pipeline",
]
