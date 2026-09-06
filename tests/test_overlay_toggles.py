"""Overlay two-block toggles (spec §6) + max_lines default.

Each block is independently toggleable (some users want translation only, or
transcription only). Toggles round-trip through config.json and hide the
matching Qt frame at window build time.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from voicelang_core.config import Config, OverlayConfig
from voicelang_core.adapters.overlay import OverlayDisplay


def test_overlay_config_defaults_max_lines_5_and_toggles_on():
    ov = OverlayConfig()
    assert ov.max_lines == 5
    assert ov.show_transcription_block is True
    assert ov.show_translation_block is True


def test_overlay_config_roundtrip_toggles():
    cfg = Config.from_dict({"overlay": {"show_transcription_block": False}})
    assert cfg.overlay.show_transcription_block is False
    assert cfg.overlay.show_translation_block is True
    cfg2 = Config.from_dict(cfg.to_dict())
    assert cfg2.overlay.show_transcription_block is False
    assert cfg2.overlay.show_translation_block is True
    assert cfg2.overlay.max_lines == 5


def test_overlay_hides_disabled_blocks():
    d = OverlayDisplay(show_transcription_block=False)
    assert not d.headless  # offscreen window was built
    assert d._left_frame.isHidden() is True
    assert d._right_frame.isHidden() is False


def test_overlay_shows_both_by_default():
    d = OverlayDisplay()
    assert not d.headless
    assert d._left_frame.isHidden() is False
    assert d._right_frame.isHidden() is False


def test_overlay_scrollable_history_and_follow():
    from voicelang_core.types import Segment

    d = OverlayDisplay(width=400, height=60)  # cramped viewport -> real scroll range
    for i in range(1, 41):
        d.show(Segment(id=i, status="final", source_text=f"line {i}",
                       translated_text=f"tra {i}"))
    d._render()  # coalesce timer doesn't fire without an event loop — render directly
    # Full scrollable history kept in the edit; legacy capped view intact.
    doc = d._left_edit.toPlainText().splitlines()
    assert len(doc) == 40
    assert doc[0] == "1. line 1" and doc[-1] == "40. line 40"
    assert len(d._hist_left) == 40
    assert len(d._left_lines) == 5  # legacy max_lines view still capped
    assert d._follow_left is True   # following the newest caption
    # User scrolls up -> follow disengages; back to bottom -> re-engages.
    sb = d._left_edit.verticalScrollBar()
    assert sb.maximum() > 0         # document actually overflows the box
    sb.setValue(0)
    assert d._follow_left is False
    sb.setValue(sb.maximum())
    assert d._follow_left is True


def test_overlay_wraps_long_sentences():
    from PySide6 import QtGui

    d = OverlayDisplay()
    assert d._left_edit.wordWrapMode() == QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere


def test_overlay_partial_preview_renders_unnumbered():
    from voicelang_core.types import Segment

    d = OverlayDisplay()
    d.show_partial(Segment(status="partial", source_text="hello world"))
    d._render()
    assert "… hello world" in d._left_edit.toPlainText()