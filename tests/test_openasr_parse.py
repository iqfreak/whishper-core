"""OpenASR realtime event parser tests.

The contract is read from the OpenASR RUST SOURCE (not guessed, not from docs):
  crates/openasr-core/src/realtime/events.rs          -> envelope + `type` tags
  .../generated/realtime-wire/RealtimeTranscript*.ts   -> payload fields

Finality is encoded twice: `type` ("transcript.partial" / "transcript.final")
and the payload's `is_final` boolean. Both paths are asserted.

No numpy, no OpenASR binary required.
"""
import json
import sys

import pytest

sys.path.insert(0, ".")

from voicelang_core.adapters.openasr_source import OpenASRSource  # noqa: E402


def _env(payload: dict) -> str:
    """Wrap a payload in the real envelope shape."""
    env = {
        "type": payload.pop("_type"),
        "session_id": "sess-1",
        "event_id": "evt-1",
        "seq": 1,
        "created_at": "2026-08-29T00:00:00Z",
    }
    env.update(payload)
    return json.dumps(env)


PARTIAL = {
    "_type": "transcript.partial",
    "utterance_id": "u1", "segment_id": "s1", "revision": 0,
    "text": "I think", "start_ms": 0, "end_ms": 800,
    "is_final": False, "words": [{"word": "I", "start_ms": 0, "end_ms": 200}],
    "language": "en", "speaker": None,
}

FINAL = {
    "_type": "transcript.final",
    "utterance_id": "u1", "segment_id": "s1", "revision": 1,
    "text": "I think so", "start_ms": 0, "end_ms": 1200,
    "is_final": True, "words": [{"word": "so", "start_ms": 900, "end_ms": 1200}],
    "language": "en", "speaker": None,
}


def test_partial_by_is_final():
    seg = OpenASRSource._parse_event(_env(dict(PARTIAL)))
    assert seg is not None and seg.status == "partial"
    assert seg.source_text == "I think"


def test_final_by_is_final():
    seg = OpenASRSource._parse_event(_env(dict(FINAL)))
    assert seg is not None and seg.status == "final"
    assert seg.source_text == "I think so"


def test_type_discriminator_used_when_is_final_absent():
    p = {"_type": "transcript.partial", "text": "I think"}
    f = {"_type": "transcript.final", "text": "I think so"}
    assert OpenASRSource._parse_event(_env(p)).status == "partial"
    assert OpenASRSource._parse_event(_env(f)).status == "final"


def test_revision_is_final():
    r = {"_type": "transcript.revision", "text": "I think so!", "is_final": True}
    assert OpenASRSource._parse_event(_env(r)).status == "final"


def test_language_carried_through():
    seg = OpenASRSource._parse_event(_env(dict(FINAL)))
    assert seg.source_language == "en"


def test_lifecycle_events_without_text_skipped():
    for t in ("session.created", "session.capabilities", "session.configured",
              "session.closed", "audio.input.started", "audio.input.stopped",
              "vad.speech.started", "vad.speech.stopped", "error"):
        assert OpenASRSource._parse_event(json.dumps({"type": t})) is None


def test_empty_text_skipped():
    e = {"_type": "transcript.final", "text": "", "is_final": True}
    assert OpenASRSource._parse_event(_env(e)) is None


def test_non_json_skipped():
    assert OpenASRSource._parse_event("not json at all") is None


def test_unknown_type_with_text_fails_safe_to_visible():
    """Fail-safe: emit rather than drop (a silent drop is the worst failure)."""
    seg = OpenASRSource._parse_event(json.dumps({"type": "weird", "text": "hi"}))
    assert seg is not None and seg.status == "final"