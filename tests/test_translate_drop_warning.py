"""WP-6: translate-drop warning — translate_q drop-oldest must count and warn.

Acceptance: when translate_q (maxsize=2) receives 3 finals rapidly, oldest is
dropped, metrics.translation drops incremented, and pipeline still displays
remaining finals with distinct ids. Verifies the drop is counted, not silent.
"""
import queue
from voicelang_core.pipeline_threaded import ThreadedPipeline, _put_bounded
from voicelang_core.types import Segment
from voicelang_core.metrics import PipelineMetrics
from voicelang_core.adapters.fake import FakeAudioSource, FakeDisplay

class _SlowTranslator:
    is_passthrough = False
    def __init__(self):
        self.calls = []
    def translate(self, text, src, tgt):
        from voicelang_core.types import Translation
        self.calls.append(text)
        return Translation(source_text=text, target_text=text.upper(), source_language=src, target_language=tgt)

class FakeTranscriber:
    def transcribe_stream(self, chunk):
        yield Segment(id=0, status="final", source_text="hello", source_language="en")

def test_translate_drop_counts_and_preserves_ids():
    tr = _SlowTranslator()
    disp = FakeDisplay()
    # tiny translate_q to force drops
    pipe = ThreadedPipeline(source=FakeAudioSource([]), transcriber=FakeTranscriber(), translator=tr, display=disp, target_language="en", translate_queue_size=2)
    # ensure metrics present
    assert pipe.metrics is not None
    # Manually enqueue 3 finals via _handle_segment (which uses _put_bounded drop_oldest=True)
    for txt in ("one", "two", "three"):
        seg = Segment(id=0, status="final", source_text=txt, source_language="en")
        pipe._handle_segment(seg)
    # translate_q maxsize 2, so oldest should have been dropped and counted
    assert pipe._translate_q.qsize() == 2
    assert pipe.metrics.translation.drops >= 1 or pipe.metrics.dropped_blocks >= 1
    # ids must be distinct and monotonic (1,2,3 attributed before enqueue)
    # The snapshots in queue should have distinct ids; we drain to check
    ids = []
    while not pipe._translate_q.empty():
        ids.append(pipe._translate_q.get_nowait().id)
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert ids == sorted(ids)

def test_put_bounded_drop_oldest_records_metric():
    q = queue.Queue(maxsize=1)
    m = PipelineMetrics(engine="test")
    q.put("first")
    dropped = _put_bounded(q, "second", drop_oldest=True, metrics=m, stage="translation")
    assert dropped is True
    assert m.translation.drops == 1 or m.dropped_blocks == 1
    assert q.get_nowait() == "second"

def test_translate_drop_does_not_block_capture():
    # translate thread is LOWEST priority; drops must be immediate not blocking
    q = queue.Queue(maxsize=1)
    q.put("a")
    # second put with drop_oldest should return immediately (<0.3s)
    import time
    t0 = time.monotonic()
    _put_bounded(q, "b", drop_oldest=True, metrics=None, stage="translation")
    assert time.monotonic() - t0 < 0.3
