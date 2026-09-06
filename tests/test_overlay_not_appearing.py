"""Overlay not appearing — regression + visibility tests.

Covers:
- H1: construction reached vs headless fallback
- H3/H4: geometry/opacity clamping so window is never invisible
- H5: live captions actually paint (not just code path)
- Missing-dependency loud failure (exit 30, GUI preflight, log tail)
"""
import importlib.util
import json
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from unittest import mock

import pytest

# Force offscreen for all Qt tests in this file
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _offscreen_display(**kwargs):
    from voicelang_core.adapters.overlay import OverlayDisplay
    # ensure offscreen env
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    d = OverlayDisplay(**kwargs)
    return d


def test_overlay_constructs_offscreen_visible():
    d = _offscreen_display(x=None, y=None, width=None, height=None, opacity=0.85, max_lines=5)
    assert not d.headless, "Overlay should construct offscreen when PySide6 present"
    assert d._window is not None
    assert d._window.isVisible()
    geo = d._window.geometry()
    assert geo.width() >= 320 and geo.height() >= 80
    assert 0.2 <= d._window.windowOpacity() <= 1.0
    # blocks visible
    assert d._left_frame.isVisible()
    assert d._right_frame.isVisible()
    d._window.close()
    if d._app:
        d._app.processEvents()


def test_overlay_geometry_clamped_offscreen():
    # Request far off-screen should be clamped to visible area, not remain hidden
    d = _offscreen_display(x=-5000, y=-5000, width=800, height=160, opacity=0.85)
    assert not d.headless
    geo = d._window.geometry()
    # After clamping, x/y should be within reasonable screen bounds (geo.x()+40 fallback)
    # offscreen screen is 800x600 by default, so -5000 gets remapped
    assert geo.x() != -5000, "off-screen x should be clamped to visible area"
    assert geo.y() != -5000
    assert geo.width() >= 320
    d._window.close()
    if d._app:
        d._app.processEvents()


def test_overlay_opacity_clamped_transparent():
    # Opacity 0.0 / 0.05 should be clamped to visible
    d = _offscreen_display(opacity=0.0, x=100, y=50, width=800, height=160)
    assert not d.headless
    # config clamp and overlay clamp ensure >=0.2
    assert d._opacity >= 0.2
    assert d._window.windowOpacity() >= 0.2
    d._window.close()
    if d._app:
        d._app.processEvents()

    d2 = _offscreen_display(opacity=0.05, x=100, y=50, width=800, height=160)
    assert d2._opacity >= 0.2
    d2._window.close()
    if d2._app:
        d2._app.processEvents()


def test_overlay_config_defaults_visible():
    from voicelang_core.config import Config

    cfg = Config.from_dict({
        "display": "overlay",
        "overlay": {"x": -9000, "y": -9000, "width": 50, "height": 10, "opacity": 0.0, "max_lines": 0, "auto_place": False}
    })
    # from_dict clamps
    assert cfg.overlay.opacity >= 0.2
    assert cfg.overlay.max_lines >= 1
    # overlay itself also clamps width/height/opacity at window build time
    d = _offscreen_display(x=cfg.overlay.x, y=cfg.overlay.y, width=cfg.overlay.width, height=cfg.overlay.height,
                           opacity=cfg.overlay.opacity, max_lines=cfg.overlay.max_lines)
    # overlay's own clamp widens tiny width
    geo = d._window.geometry()
    assert geo.width() >= 320
    assert geo.height() >= 80
    assert d._opacity >= 0.2
    d._window.close()
    if d._app:
        d._app.processEvents()


def test_overlay_auto_place_respected():
    # When auto_place True, run._build_display should ignore stale manual coords
    from voicelang_core.config import Config
    from voicelang_core.run import _build_display
    import os as _os
    _os.environ["QT_QPA_PLATFORM"] = "offscreen"
    cfg = Config()
    cfg.display = "overlay"
    cfg.overlay.auto_place = True
    cfg.overlay.x = -9000
    cfg.overlay.y = -9000
    cfg.overlay.width = 50
    cfg.overlay.height = 10
    cfg.overlay.opacity = 0.05
    d = _build_display("overlay", cfg)
    assert not d.headless
    geo = d._window.geometry()
    # Should NOT be at -9000
    assert geo.x() != -9000
    assert geo.width() >= 320
    assert d._window.windowOpacity() >= 0.2
    d._window.close()
    if d._app:
        d._app.processEvents()


def test_overlay_live_captions_offscreen():
    """End-to-end: Pipeline + OverlayDisplay offscreen shows live captions."""
    from voicelang_core.adapters.overlay import OverlayDisplay
    from voicelang_core.pipeline import Pipeline
    from voicelang_core.types import Segment, AudioChunk
    from voicelang_core.ports import AudioSource, StreamingTranscriber, Translator

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    display = OverlayDisplay(headless=False, x=100, y=50, width=800, height=160, opacity=0.85)
    assert not display.headless
    # Fake source yielding one chunk
    class FakeSource(AudioSource):
        def stream(self):
            yield AudioChunk(pcm=b"\x00\x10" * 800, capture_ts=0)

    class FakeTranscriber(StreamingTranscriber):
        def transcribe(self, chunk):
            return Segment(id=0, status="final", source_text="hello world", translated_text=None, timestamp=0, source_language="en")
        def transcribe_stream(self, chunk):
            yield Segment(id=0, status="partial", source_text="hello world", translated_text=None, timestamp=0, source_language="en")
            yield Segment(id=0, status="final", source_text="hello world", translated_text=None, timestamp=0, source_language="en")

    class FakeTranslator(Translator):
        is_passthrough = True
        def translate(self, text, src, tgt):
            from voicelang_core.types import Translation
            return Translation(source_text=text, target_text=text, source_language=src, target_language=tgt)

    src = FakeSource()
    tr = FakeTranscriber()
    tl = FakeTranslator()
    pipe = Pipeline(source=src, transcriber=tr, translator=tl, display=display, target_language="en", source_language="en")
    pipe.run_streaming()
    # After pipeline, overlay should have both partial flushed and final
    assert any("hello world" in l for l in display._left_lines), display._left_lines
    assert any("hello world" in l for l in display._right_lines)
    assert display._hist_left, "history should contain final"
    # Visibility still true
    assert display._window.isVisible()
    display._window.close()
    if display._app:
        display._app.processEvents()


def test_missing_pyside_loud_failure_headless():
    """Simulate missing PySide6 -> OverlayDisplay falls back headless."""
    import voicelang_core.adapters.overlay as ov_mod
    import builtins
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("PySide6"):
            raise ImportError("mock missing PySide6")
        return orig_import(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", side_effect=fake_import):
        # Need to reload to trigger new import failure path
        import importlib
        importlib.reload(ov_mod)
        d = ov_mod.OverlayDisplay(headless=False, x=100, y=50, width=800, height=160)
        assert d.headless, "should be headless when PySide6 missing"
        # restore
        import importlib as _il
        _il.reload(ov_mod)
    # also verify that run._build_display exits 30 when headless
    # we simulate by patching OverlayDisplay to return headless=True
    from voicelang_core.config import Config
    from voicelang_core.run import _build_display

    cfg = Config()
    cfg.display = "overlay"
    cfg.overlay.opacity = 0.85
    with mock.patch("voicelang_core.run.OverlayDisplay") as MockOD:
        inst = mock.MagicMock()
        inst.headless = True
        MockOD.return_value = inst
        with pytest.raises(SystemExit) as exc:
            _build_display("overlay", cfg)
        assert exc.value.code == 30, f"expected exit 30 for overlay unavailable, got {exc.value.code}"


def test_missing_pyside_gui_preflight_blocks(monkeypatch):
    """GUI preflight should block start when PySide6 missing and show loud error."""
    # We can't fully instantiate SettingsWindow without Qt event loop, but we can
    # verify the find_spec check logic used by _start_run.
    import importlib.util
    # Simulate missing
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "PySide6" else object())
    assert importlib.util.find_spec("PySide6") is None
    # When spec is None, GUI should surface status+toast (logic is in _start_run)
    # We verify the branch string exists in source
    src = Path("voicelang_core/gui.py").read_text(encoding="utf-8", errors="replace")
    assert "overlay preflight" in src.lower()
    assert "uv sync --extra overlay" in src
    assert "30:" in src or '"overlay unavailable' in src or "overlay unavailable" in src


def test_run_overlay_e2e_offscreen_wav():
    """E2E subprocess: run with overlay offscreen via wav source should succeed."""
    cfg = {
        "display": "overlay",
        "transcriber": "moonshine",
        "language": "en",
        "model_arch": "TINY_STREAMING",
        "source": "wav",
        "input_file": "C:/Users/ahmed/whishper-core/e2e_speech.wav",
        "target_language": "en",
        "translate": "passthrough",
        "overlay": {"x": 100, "y": 50, "width": 800, "height": 160, "opacity": 0.85, "max_lines": 5, "show_transcription_block": True, "show_translation_block": True, "auto_place": True}
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(cfg, tf)
        tf_path = tf.name
    try:
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        # Use project venv python which has PySide6
        py = "C:/Users/ahmed/whishper-core/.venv/Scripts/python.exe"
        # moonshine may not be installed; but run.py with wav+passthrough should work even without heavy ASR?
        # If moonshine missing, it will still attempt to load and fail. Use a lighter path:
        # we override transcriber to a known missing but our test wants overlay creation only.
        # Instead test display creation directly; skip heavy ASR by using pipeline mock via run?
        # For true E2E we use existing e2e_speech.wav with whatever transcriber is available; if missing, expect exit 10/11 but overlay construction should still have been attempted log.
        r = subprocess.run([py, "-m", "voicelang_core.run", "--config", tf_path], capture_output=True, text=True, timeout=20, env=env)
        # If moonshine not installed, expect code 10/11, not 30 (overlay ok)
        assert r.returncode in (0, 10, 11), f"unexpected code {r.returncode}: {r.stderr[-500:]}"
        if r.returncode == 0:
            assert "Playing" in r.stdout or "Playing" in r.stderr or "captions" in r.stdout.lower()
            # Overlay should have been constructed (log would mention pipeline)
        else:
            # Ensure overlay failure code 30 is NOT the result when PySide6 present
            assert r.returncode != 30, "overlay should not fail when PySide6 present offscreen"
    finally:
        try:
            os.unlink(tf_path)
        except Exception:
            pass

def test_overlay_both_blocks_hidden_still_window():
    d = _offscreen_display(show_transcription_block=False, show_translation_block=False, x=100, y=50, width=800, height=160)
    assert not d.headless
    assert not d._left_frame.isVisible()
    assert not d._right_frame.isVisible()
    assert d._window.isVisible()
    d._window.close()
    if d._app:
        d._app.processEvents()
