"""Segment{id} schema — locked decision per Phase 1 spec.

Pipeline owns `id` (monotonic, increments only on `final`). Adapters yield
`id=0` as a sentinel; the pipeline stamps the real id before the segment
reaches the display. Translator fills `translated_text` on the SAME id.
"""
import dataclasses

import pytest

from voicelang_core.types import Segment


def test_segment_new_schema_fields_present():
    s = Segment(
        id=7,
        status="final",
        source_text="hello",
        translated_text="salut",
        timestamp=1.23,
        source_language="en",
    )
    assert s.id == 7
    assert s.status == "final"
    assert s.source_text == "hello"
    assert s.translated_text == "salut"
    assert s.timestamp == 1.23
    assert s.source_language == "en"


def test_segment_partial_defaults_translated_and_language_to_none():
    s = Segment(id=3, status="partial", source_text="h", timestamp=0.0)
    assert s.translated_text is None
    assert s.source_language is None


def test_segment_old_field_names_removed():
    field_names = {f.name for f in dataclasses.fields(Segment)}
    for old in ("text", "start", "end", "language", "final"):
        assert old not in field_names, f"legacy field {old!r} must be removed"


def test_segment_legacy_kwargs_rejected():
    """Old call sites pass text=... final=...; must raise so the sweep is loud."""
    with pytest.raises(TypeError):
        Segment(text="hi", final=True, language="en")


def test_segment_status_accepts_only_partial_or_final():
    # Literal type is enforced by dataclass __init__ via the type hint
    # only at runtime when explicit. We accept the Literal either way;
    # the test guards against silently widening the type.
    import typing
    hints = typing.get_type_hints(Segment)
    allowed = typing.get_args(hints["status"])
    assert set(allowed) == {"partial", "final"}