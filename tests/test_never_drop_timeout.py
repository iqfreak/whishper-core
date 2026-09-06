"""WP-6: never-drop timeout — streaming (never-drop) queue must not hang forever.

For streaming engines _is_streaming True, _put_bounded uses drop_oldest=False:
 - first try puts with 0.5s timeout
 - on Full, retries up to 3x with 1.0s timeout each
 - after 3 consecutive timeouts, returns True (drop counted) and does not block forever
"""
import queue, time, threading
from voicelang_core.pipeline_threaded import _put_bounded, ThreadedPipeline
from voicelang_core.metrics import PipelineMetrics
from voicelang_core.types import AudioChunk, Segment
from voicelang_core.adapters.fake import FakeAudioSource, FakeDisplay, FakeTranslator

def test_never_drop_times_out_after_three_retries():
    q = queue.Queue(maxsize=1)
    q.put(AudioChunk(pcm=b"\x00"*320, sample_rate=16000))
    m = PipelineMetrics(engine="test-streaming")
    # Fill queue stays full (no consumer) — never-drop should timeout after ~3.5s total (0.5+1+1+1)
    t0 = time.monotonic()
    dropped = _put_bounded(q, AudioChunk(pcm=b"\x01"*320, sample_rate=16000), drop_oldest=False, metrics=m, stage="capture")
    elapsed = time.monotonic() - t0
    # Should have timed out and returned True (hard failure) after ~3.5s, not blocked forever
    assert elapsed >= 3.0
    assert elapsed < 6.0
    assert dropped is True
    # metrics should have recorded a drop for this stage
    assert m.dropped_blocks >= 1

def test_never_drop_succeeds_when_consumer_drains():
    q = queue.Queue(maxsize=1)
    q.put(AudioChunk(pcm=b"\x00"*320, sample_rate=16000))
    m = PipelineMetrics(engine="test-streaming")
    # Consumer drains after 0.3s, so never-drop should succeed before timeout
    def drain():
        time.sleep(0.3)
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    t = threading.Thread(target=drain, daemon=True)
    t.start()
    dropped = _put_bounded(q, AudioChunk(pcm=b"\x01"*320, sample_rate=16000), drop_oldest=False, metrics=m, stage="capture")
    t.join(timeout=2)
    assert dropped is False
    assert q.qsize() == 1

def test_streaming_pipeline_auto_promotes_queue_size():
    # ThreadedPipeline should auto-promote asr_queue_size to 64 for streaming engines
    class StreamingFake:
        pass
    StreamingFake.__name__ = "MoonshineTranscriber"
    # Need real type name match, so use FakeTranscriber with that name via instance trick
    from unittest.mock import MagicMock
    # Instead directly test via real MoonshineTranscriber name — use a stub class with correct __name__
    import types
    FakeStreaming = type("MoonshineTranscriber", (), {"transcribe_stream": lambda self, c: iter(())})
    pipe = ThreadedPipeline(source=FakeAudioSource([]), transcriber=FakeStreaming(), translator=FakeTranslator(), display=FakeDisplay(), asr_queue_size=2)
    assert pipe._asr_q.maxsize == 64
    # Non-streaming keeps size 2
    class NonStreaming:
        pass
    NonStreaming.__name__ = "FakeTranscriber"
    FakeNon = type("FakeTranscriber", (), {"transcribe_stream": lambda self, c: iter(())})
    pipe2 = ThreadedPipeline(source=FakeAudioSource([]), transcriber=FakeNon(), translator=FakeTranslator(), display=FakeDisplay(), asr_queue_size=2)
    assert pipe2._asr_q.maxsize == 2

def test_never_drop_does_not_lose_continuity_when_drained():
    # Enqueue 5 chunks with consumer draining slowly — never-drop preserves order
    q = queue.Queue(maxsize=2)
    consumed = []
    stop = threading.Event()
    def consumer():
        while not stop.is_set():
            try:
                item = q.get(timeout=0.1)
                consumed.append(item)
            except queue.Empty:
                continue
    ct = threading.Thread(target=consumer, daemon=True)
    ct.start()
    for i in range(5):
        _put_bounded(q, i, drop_oldest=False, metrics=None, stage="capture")
        time.sleep(0.05)
    time.sleep(0.3)
    stop.set()
    ct.join(timeout=1)
    # All 5 should have been consumed in order (never-drop)
    assert consumed == [0,1,2,3,4]
