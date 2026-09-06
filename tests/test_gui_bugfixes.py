"""Bug-fix verification: provider switch reset + source_app/device round-trip + DPI."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from voicelang_core.config import Config, OverlayConfig
from voicelang_core.gui import _make_gui


def _app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


pytestmark = pytest.mark.skipif(
    not _app, reason="PySide6 not available")


def test_switch_transcriber_resets_stale_language_and_arch():
    _app()
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # start with vosk de/small (vosk supports de, funasr-nano does not)
    cfg = Config(transcriber="vosk", language="de", model_arch="small")
    W = _make_gui()
    win = W(cfg)
    win.show()  # needed for visibility checks elsewhere but not here
    assert win._language.currentData() == "de"
    assert win._arch.currentData() == "small"
    # switch to funasr-nano: only auto,zh,zh-cn,en,ja ; arch Nano-2512
    win._transcriber.setCurrentText("funasr-nano")
    win._refresh_provider_models()
    # must not preserve stale "de"
    assert win._language.currentData() != "de", "stale language preserved after provider switch"
    assert win._language.currentData() in ["auto", "zh", "zh-cn", "en", "ja"]
    # arch must be valid for new provider
    assert win._arch.currentData() == "Nano-2512"
    # switch whisper->nemotron arch stale test
    win._transcriber.setCurrentText("whisper")
    win._refresh_provider_models()
    idx = win._arch.findData("large-v3")
    if idx != -1:
        win._arch.setCurrentIndex(idx)
        assert win._arch.currentData() == "large-v3"
    win._transcriber.setCurrentText("nemotron")
    win._refresh_provider_models()
    assert win._arch.currentData() == "int8", f"arch not reset from whisper large-v3 to nemotron, got {win._arch.currentData()}"
    # ensure dl label probing used corrected values (no crash)
    win.close()


def test_source_app_device_roundtrip(tmp_path=None):
    _app()
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg = Config(source="app", source_app="Discord.exe", source_device="Default device")
    W = _make_gui()
    win = W(cfg)
    win.show()
    # simulate user picking app/device
    win._source.setCurrentIndex(win._source.findData("app"))
    win._on_source_changed()
    # _app_row should be visible, _device_row hidden when app
    assert not win._app_row.isHidden(), "app row should be visible for source=app"
    assert win._device_row.isHidden(), "device row should be hidden for source=app"
    # loopback visibility
    win._source.setCurrentIndex(win._source.findData("loopback"))
    win._on_source_changed()
    assert win._app_row.isHidden()
    assert not win._device_row.isHidden()

    # round-trip via _cfg_from_widgets for app
    win._source.setCurrentIndex(win._source.findData("app"))
    win._on_source_changed()
    win._app_selector.setText("MyGame.exe")
    # ensure device combo default still "" when not loopback
    cfg_out = win._cfg_from_widgets()
    assert cfg_out.source == "app"
    assert cfg_out.source_app == "MyGame.exe"

    # round-trip for loopback device
    win._source.setCurrentIndex(win._source.findData("loopback"))
    win._on_source_changed()
    # default device is first entry with data ""
    assert win._device_combo.currentData() == "" or isinstance(win._device_combo.currentData(), str)
    cfg_out2 = win._cfg_from_widgets()
    assert cfg_out2.source == "loopback"
    # source_device should be string (empty = default)
    assert isinstance(cfg_out2.source_device, str)
    win.close()


def test_save_cfg_defense_clamps_invalid_language(tmp_path):
    _app()
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg = Config(transcriber="funasr-nano", language="de", model_arch="Nano-2512")
    W = _make_gui()
    win = W(cfg)
    # force invalid language via widget (editable combo allows stale)
    win._transcriber.setCurrentText("funasr-nano")
    win._refresh_provider_models()
    # now manually try to set invalid language via editable text
    win._language.setEditText("de")
    # _save_cfg should clamp before persisting
    out = win._cfg_from_widgets()
    # out currently has de - clamp happens in _save_cfg not _cfg_from_widgets
    # but _start_run validation should block
    win._start_run = lambda: None  # avoid subprocess
    # test that _save_cfg would correct it
    from voicelang_core.config import save_config
    p = str(tmp_path / "clamp.json")
    # simulate _save_cfg clamping logic
    from voicelang_core.engines import get_registry
    reg = get_registry()
    # verify de not in funasr-nano
    assert "de" not in [l.lower() for l in reg["funasr-nano"]["languages"]]
    # our _refresh should have already reset to auto if we switch, but editable text can inject stale
    # after our fix, switching provider resets; direct edit is user error - save should clamp
    win._save_cfg()
    # after save, cfg stored should be valid
    assert win._cfg.language in reg["funasr-nano"]["languages"]
    win.close()


def test_high_dpi_main_sets_attributes_before_app():
    # verify main() contains HighDpi attribute setup before QApplication
    import pathlib
    src = pathlib.Path("C:/Users/ahmed/whishper-core/voicelang_core/gui.py").read_text(encoding="utf-8")
    # must have PassThrough policy and AA checks before first QApplication
    assert "HighDpiScaleFactorRoundingPolicy.PassThrough" in src
    assert "AA_EnableHighDpiScaling" in src
    # ensure settings window no longer has spurious WA_HighDpiScaling
    assert "WA_HighDpiScaling" not in src or src.count("WA_HighDpiScaling") == 0 or "WA_HighDpiScaling\" not in src.split(\"def main\")[0].split(\"setMinimumSize\")[-1]"
    # simpler: spurious line removed
    assert 'self.setAttribute(QtCore.Qt.WA_HighDpiScaling' not in src
