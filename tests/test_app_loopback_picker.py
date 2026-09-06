"""App/Loopback Source Picker — verification for missing picker feature.

Covers:
- backend reuse (list_audio_apps / list_loopback_devices)
- UI picker appears when Per-app selected (dropdown + Refresh), loopback has "Default device"
- show/hide by source, Refresh works, empty case shows message
- wiring through to Config persistence and capture adapter construction (run._build_source + engines._build_capture_source)
- prevent starting with Per-app and nothing chosen

QT_QPA_PLATFORM=offscreen required.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import types
import pytest

def _pyqt6():
    try:
        import pathlib, PySide6
        plugins = pathlib.Path(PySide6.__file__).parent / "plugins"
        if plugins.is_dir():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugins))
        return True
    except ImportError:
        return False

pytestmark = pytest.mark.skipif(not _pyqt6(), reason="PySide6 extra not installed")

from unittest.mock import patch

from voicelang_core.config import Config, load_config, save_config
from voicelang_core.gui import _make_gui
from voicelang_core.run import _build_source
from voicelang_core.engines import EngineConfig, EngineName, _build_capture_source


def _app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_backend_list_functions_exist():
    from voicelang_core.adapters.app_loopback import list_audio_apps
    from voicelang_core.adapters.wasapi_loopback import list_loopback_devices
    assert callable(list_audio_apps)
    assert callable(list_loopback_devices)
    # should not crash, return list
    assert isinstance(list_audio_apps(), list)
    assert isinstance(list_loopback_devices(), list)


def test_picker_show_hide_by_source():
    app = _app()
    cfg = Config(source="mic")
    W = _make_gui()
    win = W(cfg)
    win.show()
    app.processEvents()
    win._navigate(1)  # Capture page
    app.processEvents()

    def is_hidden(w): return w.isHidden()

    # mic -> both pickers hidden
    win._source.setCurrentIndex(win._source.findData("mic"))
    app.processEvents()
    assert is_hidden(win._app_row)
    assert is_hidden(win._device_row)
    assert not is_hidden(win._wav_row) is False  # wav hidden when mic
    # Actually check wav hidden when mic
    assert win._wav_row.isHidden()

    # app -> app picker visible, device hidden
    win._source.setCurrentIndex(win._source.findData("app"))
    app.processEvents()
    assert not win._app_row.isHidden()
    assert win._device_row.isHidden()
    assert win._wav_row.isHidden()

    # loopback -> device picker visible, app hidden
    win._source.setCurrentIndex(win._source.findData("loopback"))
    app.processEvents()
    assert win._app_row.isHidden()
    assert not win._device_row.isHidden()

    # wav -> wav visible
    win._source.setCurrentIndex(win._source.findData("wav"))
    app.processEvents()
    assert not win._wav_row.isHidden()
    assert win._app_row.isHidden()


def test_app_picker_lists_running_apps_and_refresh():
    app = _app()
    cfg = Config(source="app", source_app="Discord.exe")
    W = _make_gui()
    win = W(cfg)
    win.show()
    win._navigate(1)
    app.processEvents()

    fake_apps = [
        {"pid": 111, "name": "Discord.exe", "exe": "C:/Fake/Discord.exe", "state": "Active", "device": ""},
        {"pid": 222, "name": "Spotify.exe", "exe": "C:/Fake/Spotify.exe", "state": "Active", "device": ""},
    ]
    with patch("voicelang_core.adapters.app_loopback.list_audio_apps", return_value=fake_apps):
        win._refresh_app_list()
        app.processEvents()
        assert win._app_combo.count() == 2
        texts = [win._app_combo.itemText(i) for i in range(win._app_combo.count())]
        assert any("Discord" in t for t in texts)
        assert any("Spotify" in t for t in texts)
        # Refresh button should re-populate (click simulation)
        win._refresh_app_list()
        app.processEvents()
        assert win._app_combo.count() == 2

    # Verify sync to lineedit
    win._app_combo.setCurrentIndex(0)
    app.processEvents()
    # _sync should have set selector to Discord.exe
    assert "Discord" in win._app_selector.text()


def test_app_picker_empty_shows_inline_message():
    app = _app()
    cfg = Config(source="app")
    W = _make_gui()
    win = W(cfg)
    win.show()
    win._navigate(1)
    app.processEvents()
    win._source.setCurrentIndex(win._source.findData("app"))
    app.processEvents()
    with patch("voicelang_core.adapters.app_loopback.list_audio_apps", return_value=[]):
        win._refresh_app_list()
        app.processEvents()
        assert win._app_combo.count() == 1
        assert "No running" in win._app_combo.itemText(0)
        # hint must be visible and non-blank
        assert not win._app_hint.isHidden()
        assert win._app_hint.text().strip() != ""
        assert "No running" in win._app_hint.text() or "pycaw" in win._app_hint.text().lower() or "Refresh" in win._app_hint.text()


def test_loopback_picker_has_default_and_refresh():
    app = _app()
    cfg = Config(source="loopback", source_device="")
    W = _make_gui()
    win = W(cfg)
    win.show()
    win._navigate(1)
    app.processEvents()
    win._source.setCurrentIndex(win._source.findData("loopback"))
    app.processEvents()

    fake_devs = [
        {"index": 5, "name": "Speakers (Realtek) [Loopback]", "rate": 48000},
        {"index": 6, "name": "Headphones [Loopback]", "rate": 48000},
    ]
    with patch("voicelang_core.adapters.wasapi_loopback.list_loopback_devices", return_value=fake_devs):
        win._refresh_device_list()
        app.processEvents()
        # default + 2 devices
        assert win._device_combo.count() == 3
        assert win._device_combo.itemText(0) == "Default device"
        assert win._device_combo.itemData(0) == ""
        # pick second device and verify persistence
        win._device_combo.setCurrentIndex(1)
        app.processEvents()
        out = win._cfg_from_widgets()
        assert out.source_device == "Speakers (Realtek) [Loopback]"
        # default selection maps to empty string
        win._device_combo.setCurrentIndex(0)
        out2 = win._cfg_from_widgets()
        assert out2.source_device == ""

    # empty case still shows default device, not blank combo
    with patch("voicelang_core.adapters.wasapi_loopback.list_loopback_devices", return_value=[]):
        win._refresh_device_list()
        app.processEvents()
        assert win._device_combo.count() == 1
        assert win._device_combo.itemText(0) == "Default device"


def test_wiring_through_to_config_persistence(tmp_path):
    app = _app()
    cfg = Config(source="app", source_app="")
    W = _make_gui()
    win = W(cfg)
    win.show()
    win._navigate(1)
    app.processEvents()

    fake_apps = [{"pid": 333, "name": "Chrome.exe", "exe": "Chrome.exe", "state": "Active", "device": ""}]
    with patch("voicelang_core.adapters.app_loopback.list_audio_apps", return_value=fake_apps):
        win._app_selector.setText("")  # clear stale
        win._refresh_app_list()
        app.processEvents()
        assert win._app_combo.count() == 1
        win._source.setCurrentIndex(win._source.findData("app"))
        app.processEvents()
        win._app_combo.setCurrentIndex(0)
        app.processEvents()
        out = win._cfg_from_widgets()
        assert out.source == "app"
        assert out.source_app == "Chrome.exe"
        # save and reload
        p = str(tmp_path / "cfg.json")
        save_config(out, p)
        loaded = load_config(p)
        assert loaded.source == "app"
        assert loaded.source_app == "Chrome.exe"

    # loopback persistence
    with patch("voicelang_core.adapters.wasapi_loopback.list_loopback_devices", return_value=[{"index": 9, "name": "My Speaker [Loopback]", "rate": 48000}]):
        win._refresh_device_list()
        app.processEvents()
        win._source.setCurrentIndex(win._source.findData("loopback"))
        win._device_combo.setCurrentIndex(1)
        app.processEvents()
        out2 = win._cfg_from_widgets()
        p2 = str(tmp_path / "cfg2.json")
        save_config(out2, p2)
        loaded2 = load_config(p2)
        assert loaded2.source_device == "My Speaker [Loopback]"


def test_capture_adapter_receives_correct_identifier(monkeypatch, tmp_path):
    # via run._build_source
    src = _build_source("app", None, "", "Discord.exe")
    assert src.__class__.__name__ == "AppLoopbackSource"
    assert getattr(src, "_app") == "Discord.exe"

    src2 = _build_source("app", None, "", "pid:1234")
    assert getattr(src2, "_app") == "pid:1234"

    src3 = _build_source("loopback", None, "Speakers [Loopback]")
    assert src3.__class__.__name__ == "WASAPILoopbackSource"
    assert getattr(src3, "_device") == "Speakers [Loopback]"

    src4 = _build_source("loopback", None, "")
    assert getattr(src4, "_device") is None

    # via engines._build_capture_source
    cfg = EngineConfig(name=EngineName.moonshine, capture="app", source_app="Spotify.exe")
    e_src = _build_capture_source(cfg)
    assert e_src.__class__.__name__ == "AppLoopbackSource"
    assert getattr(e_src, "_app") == "Spotify.exe"

    cfg2 = EngineConfig(name=EngineName.moonshine, capture="loopback", source_device="Headphones [Loopback]")
    e_src2 = _build_capture_source(cfg2)
    assert e_src2.__class__.__name__ == "WASAPILoopbackSource"
    assert getattr(e_src2, "_device") == "Headphones [Loopback]"

    # default loopback (system default)
    cfg3 = EngineConfig(name=EngineName.moonshine, capture="loopback", source_device="")
    e_src3 = _build_capture_source(cfg3)
    assert getattr(e_src3, "_device") is None


def test_offscreen_picking_specific_app_and_start():
    """Pick specific app from list, call _start_run, confirm correct identifier passed."""
    app = _app()
    cfg = Config(source="app", source_app="")
    W = _make_gui()
    win = W(cfg)
    win.show()
    win._navigate(1)
    app.processEvents()

    fake_apps = [
        {"pid": 101, "name": "Discord.exe", "exe": "C:/Discord.exe", "state": "Active", "device": ""},
        {"pid": 202, "name": "Game.exe", "exe": "C:/Game.exe", "state": "Active", "device": ""},
    ]
    with patch("voicelang_core.adapters.app_loopback.list_audio_apps", return_value=fake_apps):
        win._refresh_app_list()
        app.processEvents()
        # pick Game.exe (second item)
        win._app_combo.setCurrentIndex(1)
        app.processEvents()
        assert win._app_selector.text() == "Game.exe"
        out = win._cfg_from_widgets()
        assert out.source_app == "Game.exe"
        # simulate Start: patch save_config and Popen to capture
        with patch("voicelang_core.gui.save_config") as mock_save, patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.poll.return_value = None
            # need to set _log_file handling
            win._start_run()
            app.processEvents()
            assert mock_popen.called, "Start should launch subprocess when app chosen"
            saved_cfg = mock_save.call_args[0][0]
            assert saved_cfg.source == "app"
            assert saved_cfg.source_app == "Game.exe"
            # verify adapter would receive same identifier via run layer
            src = _build_source(saved_cfg.source, None, saved_cfg.source_device, saved_cfg.source_app)
            assert getattr(src, "_app") == "Game.exe"
        win._stop_run()


def test_prevent_start_with_no_app_chosen():
    app = _app()
    cfg = Config(source="app", source_app="")
    W = _make_gui()
    win = W(cfg)
    win.show()
    win._navigate(1)
    app.processEvents()
    win._source.setCurrentIndex(win._source.findData("app"))
    win._app_selector.setText("")
    app.processEvents()
    with patch("voicelang_core.adapters.app_loopback.list_audio_apps", return_value=[]):
        win._refresh_app_list()
        app.processEvents()
        # ensure empty state
        assert win._app_selector.text().strip() == ""
        with patch("voicelang_core.gui.save_config") as mock_save, patch("subprocess.Popen") as mock_popen:
            win._start_run()
            app.processEvents()
            assert not mock_popen.called, "Start must be blocked when no app chosen"
            assert not mock_save.called
            assert "Per-app" in win._status.text() or "target app" in win._status.text().lower()
            assert not win._app_hint.isHidden()
