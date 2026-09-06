"""GUI smoke test: window builds offscreen, widgets round-trip config."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from voicelang_core.config import Config, OverlayConfig
from voicelang_core.gui import _make_gui


def _pyqt6() -> bool:
    try:
        import pathlib

        import PySide6

        plugins = pathlib.Path(PySide6.__file__).parent / "plugins"
        if plugins.is_dir():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugins))
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _pyqt6(), reason="PySide6 extra not installed")


def test_window_builds_and_roundtrips_config(tmp_path):
    from PySide6 import QtWidgets  # noqa: PLC0415

    # Qt requires an application object before any widget is constructed.
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg = Config(display="overlay", transcriber="moonshine",
                 language="en", model_arch="TINY_STREAMING",
                 overlay=OverlayConfig(x=100, y=50, width=800, height=160,
                                       opacity=0.8, max_lines=5))
    W = _make_gui()
    win = W(cfg)
    assert win._transcriber.currentText() == "moonshine"
    assert win._language.currentData() == "en"
    assert win._x.value() == 100
    assert win._w.value() == 800
    assert win._max_lines.value() == 5
    # mutate + read back
    win._language.setCurrentIndex(win._language.findData("ar"))
    win._auto_place.setChecked(False)
    win._y.setValue(70)
    win._opacity.setValue(0.66)
    out = win._cfg_from_widgets()
    assert out.language == "ar"
    assert out.overlay.y == 70
    assert abs(out.overlay.opacity - 0.66) < 1e-6
    # save to file and reload through the public API
    p = str(tmp_path / "gui.json")
    from voicelang_core.config import save_config

    loaded = save_config(out, p)
    import json

    assert json.load(open(loaded, encoding="utf-8"))["language"] == "ar"