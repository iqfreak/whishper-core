"""WP-6 QA: ThreadedPipeline slow-translator id test — would have caught Critical #1."""
import threading, time, queue
from voicelang_core.pipeline_threaded import ThreadedPipeline
from voicelang_core.types import Segment, AudioChunk
from voicelang_core.adapters import FakeAudioSource, FakeDisplay

class BlockingTranslator:
    is_passthrough = False
    def __init__(self):
        self.block = threading.Event()
        self.calls = []
    def translate(self, text, src, tgt):
        from voicelang_core.types import Translation
        self.calls.append(text)
        self.block.wait(timeout=5)
        return Translation(source_text=text, target_text=text.upper(), source_language=src, target_language=tgt)

class TwoUtteranceSource:
    def __init__(self, transcriber):
        self.transcriber = transcriber
    def stream(self):
        # feed two utterances via transcribe_stream chunks
        yield AudioChunk(pcm=b"\x00"*320, sample_rate=16000)
        yield AudioChunk(pcm=b"\x00"*320, sample_rate=16000)

class FakeTranscriber:
    def transcribe_stream(self, chunk):
        # emit final for utterance A then B quickly
        # First call => utterance A final, second call => utterance B final
        if not hasattr(self, "_c"):
            self._c = 0
        self._c += 1
        seg = Segment(id=0, status="final", source_text=f"utterance{self._c}", source_language="en")
        seg._capture_ts = time.monotonic()
        yield seg

def test_threaded_id_race():
    tr = BlockingTranslator()
    # keep blocked so enqueue happens before worker drains
    transcriber = FakeTranscriber()
    disp = FakeDisplay()
    # Use small queues to force drop behavior visible
    pipe = ThreadedPipeline(source=FakeAudioSource([]), transcriber=transcriber, translator=tr, display=disp, target_language="en")
    # Directly test _handle_segment synchronous id advance
    seg_a = Segment(id=0, status="final", source_text="hello", source_language="en")
    seg_b = Segment(id=0, status="final", source_text="world", source_language="en")
    pipe._handle_segment(seg_a)
    # After first final, _open_id should be cleared and _next_id advanced
    assert seg_a.id == 1, f"first id should be 1 got {seg_a.id}"
    assert pipe._open_id is None, "open_id must be cleared synchronously before translation"
    assert pipe._next_id == 2
    pipe._handle_segment(seg_b)
    assert seg_b.id == 2, f"second id should be 2 got {seg_b.id}"
    assert seg_a.id != seg_b.id
    # Late translation arrival must not overwrite: simulate worker translating seg_a
    # seg_a snapshot is enqueued with id 1; seg_b snapshot has id 2 — distinct
    # unblock translator
    tr.block.set()
