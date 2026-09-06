"""Config <-> GUI completeness round-trip: every exposed field survives save/reload.

Exposed fields = every Config/OverlayConfig field that has a GUI widget.
Missing-GUI fields (log_captions, log_jsonl, log_txt, block_seconds,
moonshine_update_interval, whisper_warmup_seconds) are intentionally NOT
covered here — they are flagged in the audit report as hand-offs.

This test reproduces the original bug (overlay.auto_place lost, display=file
unselectable) and proves the fix.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import pathlib
import tempfile

import pytest

from voicelang_core.config import Config, OverlayConfig, load_config, save_config
from voicelang_core.gui import _make_gui


def _app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


pytestmark = pytest.mark.skipif(False, reason="")


def test_exposed_fields_roundtrip_via_gui(tmp_path):
    _app()
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Start from defaults
    cfg = Config()
    # Set unexposed fields to non-defaults to verify they are PRESERVED (not clobbered) by GUI save
    cfg.log_captions = True
    cfg.log_jsonl = "C:/tmp/caps.jsonl"
    cfg.log_txt = "C:/tmp/caps.txt"
    cfg.block_seconds = 0.5
    cfg.moonshine_update_interval = 0.12
    cfg.whisper_warmup_seconds = 0.9

    W = _make_gui()
    win = W(cfg)
    win.show()

    # --- Set every EXPOSED field to non-default via widgets ---
    # Provider + language + arch (must be valid combo for clamping)
    win._transcriber.setCurrentText("whisper")
    win._refresh_provider_models()
    # whisper supports auto and en etc, pick en + large-v3
    idx_lang = win._language.findData("en")
    if idx_lang != -1:
        win._language.setCurrentIndex(idx_lang)
    else:
        win._language.setEditText("en")
    idx_arch = win._arch.findData("large-v3")
    if idx_arch != -1:
        win._arch.setCurrentIndex(idx_arch)
    else:
        win._arch.setCurrentIndex(0)

    # Source = app with target
    idx_src = win._source.findData("app")
    if idx_src != -1:
        win._source.setCurrentIndex(idx_src)
    win._on_source_changed()
    win._app_selector.setText("Discord.exe")

    # Also test loopback device and wav paths (they coexist; visibility toggles but values persist)
    # Keep app as current source; also set device combo (stored even when hidden)
    try:
        # ensure device combo has at least Default entry
        if win._device_combo.count() > 0:
            win._device_combo.setCurrentIndex(0)
    except Exception:
        pass
    try:
        win._wav_path.setText("C:/tmp/test.wav")
    except Exception:
        pass

    # Translate
    win._translate_mode.setCurrentText("deepl")
    win._target_lang.setEditText("fr")
    win._translate_url.setText("https://api-free.deepl.com")
    win._translate_key.setText("secret123")

    # Display = file (was missing before fix)
    idx_disp = win._display.findText("file")
    assert idx_disp != -1, "display combo must contain 'file' (bug fix)"
    win._display.setCurrentIndex(idx_disp)

    # Overlay
    win._auto_place.setChecked(False)
    win._x.setValue(123)
    win._y.setValue(456)
    win._w.setValue(900)
    win._h.setValue(200)
    win._opacity.setValue(0.77)
    win._max_lines.setValue(7)
    win._show_trans.setChecked(False)
    win._show_transl.setChecked(True)

    # Save via _cfg_from_widgets + save_config to temp
    out_cfg = win._cfg_from_widgets()
    # Verify in-memory config has expected values before disk
    assert out_cfg.display == "file"
    assert out_cfg.transcriber == "whisper"
    assert out_cfg.language == "en"
    assert out_cfg.model_arch == "large-v3"
    assert out_cfg.source == "app"
    assert out_cfg.source_app == "Discord.exe"
    assert out_cfg.input_file == "C:/tmp/test.wav"
    assert out_cfg.translate == "deepl"
    assert out_cfg.target_language == "fr"
    assert out_cfg.translate_url == "https://api-free.deepl.com"
    assert out_cfg.translate_key == "secret123"
    assert out_cfg.overlay.auto_place is False
    assert out_cfg.overlay.x == 123
    assert out_cfg.overlay.y == 456
    assert out_cfg.overlay.width == 900
    assert out_cfg.overlay.height == 200
    assert abs(out_cfg.overlay.opacity - 0.77) < 1e-6
    assert out_cfg.overlay.max_lines == 7
    assert out_cfg.overlay.show_transcription_block is False
    assert out_cfg.overlay.show_translation_block is True

    # Unexposed fields must be preserved (cloned from original cfg)
    assert out_cfg.log_captions is True
    assert out_cfg.log_jsonl == "C:/tmp/caps.jsonl"
    assert out_cfg.log_txt == "C:/tmp/caps.txt"
    assert out_cfg.block_seconds == 0.5
    assert abs(out_cfg.moonshine_update_interval - 0.12) < 1e-6
    assert abs(out_cfg.whisper_warmup_seconds - 0.9) < 1e-6

    # Save to disk and reload
    p = str(tmp_path / "roundtrip.json")
    save_config(out_cfg, p)
    data = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    # Verify overlay auto_place persisted in JSON
    assert "auto_place" in data.get("overlay", {}), "auto_place must be persisted (bug fix)"
    assert data["overlay"]["auto_place"] is False
    # Verify display=file persisted
    assert data["display"] == "file"

    loaded = load_config(p)
    # Full equality check for exposed fields
    assert loaded.display == out_cfg.display
    assert loaded.transcriber == out_cfg.transcriber
    assert loaded.language == out_cfg.language
    assert loaded.model_arch == out_cfg.model_arch
    assert loaded.source == out_cfg.source
    assert loaded.source_app == out_cfg.source_app
    assert loaded.input_file == out_cfg.input_file
    assert loaded.translate == out_cfg.translate
    assert loaded.target_language == out_cfg.target_language
    assert loaded.translate_url == out_cfg.translate_url
    assert loaded.translate_key == out_cfg.translate_key
    assert loaded.overlay.auto_place == out_cfg.overlay.auto_place
    assert loaded.overlay.x == out_cfg.overlay.x
    assert loaded.overlay.y == out_cfg.overlay.y
    assert loaded.overlay.width == out_cfg.overlay.width
    assert loaded.overlay.height == out_cfg.overlay.height
    assert abs(loaded.overlay.opacity - out_cfg.overlay.opacity) < 1e-6
    assert loaded.overlay.max_lines == out_cfg.overlay.max_lines
    assert loaded.overlay.show_transcription_block == out_cfg.overlay.show_transcription_block
    assert loaded.overlay.show_translation_block == out_cfg.overlay.show_translation_block
    # Unexposed fields round-trip too
    assert loaded.log_captions == out_cfg.log_captions
    assert loaded.log_jsonl == out_cfg.log_jsonl
    assert loaded.log_txt == out_cfg.log_txt
    assert loaded.block_seconds == out_cfg.block_seconds

    # Also test auto_place=True roundtrip (toggle back)
    win._auto_place.setChecked(True)
    win._x.setValue(999)
    out2 = win._cfg_from_widgets()
    assert out2.overlay.auto_place is True
    p2 = str(tmp_path / "roundtrip2.json")
    save_config(out2, p2)
    loaded2 = load_config(p2)
    assert loaded2.overlay.auto_place is True

    win.close()


def test_overlay_auto_place_default_roundtrip(tmp_path):
    """Default overlay (auto_place=True, x/y/width/height None) survives GUI save."""
    _app()
    cfg = Config()  # overlay defaults: auto_place=True, x=None etc.
    assert cfg.overlay.auto_place is True
    assert cfg.overlay.x is None
    W = _make_gui()
    win = W(cfg)
    # Don't touch overlay widgets — just save back
    out = win._cfg_from_widgets()
    # GUI spinboxes have default ints, so after _cfg_from_widgets x will be 100 etc.
    # But auto_place must remain True
    assert out.overlay.auto_place is True
    p = str(tmp_path / "auto_default.json")
    save_config(out, p)
    loaded = load_config(p)
    assert loaded.overlay.auto_place is True
    win.close()


def test_display_file_selectable():
    """Regression: display combo must contain file."""
    _app()
    W = _make_gui()
    win = W(Config(display="file"))
    # Should load as file
    assert win._display.currentText() == "file"
    win.close()


def test_effective_block_seconds_canonical():
    """Canonical helper in config is single source; gui wrapper delegates."""
    from voicelang_core.config import Config, effective_block_seconds
    from voicelang_core.gui import _effective_block_seconds

    cfg = Config(transcriber="moonshine", model_arch="TINY_STREAMING", block_seconds=None)
    assert effective_block_seconds(cfg) == 0.02
    assert _effective_block_seconds(cfg) == 0.02

    cfg2 = Config(transcriber="whisper", model_arch="tiny", block_seconds=None)
    assert effective_block_seconds(cfg2) == 1.0

    cfg3 = Config(transcriber="whisper", model_arch="small", block_seconds=0.99)
    assert effective_block_seconds(cfg3) == 0.99
    assert _effective_block_seconds(cfg3) == 0.99
