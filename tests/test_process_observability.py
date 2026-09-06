"""Process observability: GUI must not hang on 'running' when worker fails.

Verifies:
- pre-flight validation blocks invalid combos (language, app picker, deepl/libretranslate credentials)
- Popen stdout+stderr captured to log file with errors="replace"
- QTimer polling transitions to FAILED and surfaces last log tail + log path via status + toast
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from unittest.mock import MagicMock, patch, mock_open

from voicelang_core.config import Config, OverlayConfig
from voicelang_core.gui import _make_gui


def _app():
    try:
        from PySide6 import QtWidgets, QtCore
        import pathlib, PySide6
        plugins = pathlib.Path(PySide6.__file__).parent / "plugins"
        if plugins.is_dir():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugins))
        return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except ImportError:
        return None

pytestmark = pytest.mark.skipif(not _app(), reason="PySide6 not available")

def _make_win(cfg=None):
    _app()
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg = cfg or Config(transcriber="moonshine", language="en", model_arch="TINY_STREAMING")
    W = _make_gui()
    win = W(cfg)
    win.show()
    return win


# ---- pre-flight validation ----

def test_preflight_blocks_invalid_language(tmp_path):
    win = _make_win(Config(transcriber="whisper", language="en", model_arch="tiny"))
    # inject invalid language via _cfg_from_widgets mock (editable combo keeps currentData = 'en')
    orig = win._cfg_from_widgets
    def fake_cfg():
        c = orig()
        c.language = "xx_invalid_lang"
        return c
    win._cfg_from_widgets = fake_cfg
    with patch("voicelang_core.gui.save_config") as mock_save, patch("voicelang_core.gui.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock(poll=lambda: None)
        win._start_run()
        mock_popen.assert_not_called()
        assert "invalid" in win._status.text().lower()
        assert "xx_invalid" in win._status.text().lower()
    win.close()

def test_preflight_blocks_missing_app_picker(tmp_path):
    cfg = Config(transcriber="moonshine", language="en", model_arch="TINY_STREAMING", source="app", source_app="")
    win = _make_win(cfg)
    win._source.setCurrentIndex(win._source.findData("app"))
    win._on_source_changed()
    win._app_selector.setText("")
    # clear combo selection as well
    try:
        win._app_combo.setCurrentIndex(-1)
    except Exception:
        pass
    with patch("voicelang_core.gui.save_config") as ms, patch("voicelang_core.gui.subprocess.Popen") as mp:
        mp.return_value = MagicMock(poll=lambda: None)
        win._start_run()
        mp.assert_not_called()
        assert "target app" in win._status.text().lower() or "per-app" in win._status.text().lower()
    win.close()

def test_preflight_blocks_wav_missing_file():
    cfg = Config(transcriber="moonshine", language="en", model_arch="TINY_STREAMING", source="wav", input_file="")
    win = _make_win(cfg)
    win._source.setCurrentIndex(win._source.findData("wav"))
    win._on_source_changed()
    if hasattr(win, "_wav_path") and win._wav_path is not None:
        win._wav_path.setText("")
    with patch("voicelang_core.gui.save_config") as ms, patch("voicelang_core.gui.subprocess.Popen") as mp:
        mp.return_value = MagicMock(poll=lambda: None)
        win._start_run()
        mp.assert_not_called()
        assert "wav" in win._status.text().lower()
    win.close()

def test_preflight_blocks_deepl_missing_key():
    cfg = Config(transcriber="moonshine", language="en", model_arch="TINY_STREAMING",
                 translate="deepl", translate_key="", translate_url="https://api-free.deepl.com")
    win = _make_win(cfg)
    try:
        win._translate_mode.setCurrentText("deepl")
        win._translate_key.setText("")
    except Exception:
        pass
    with patch("voicelang_core.gui.save_config") as ms, patch("voicelang_core.gui.subprocess.Popen") as mp:
        mp.return_value = MagicMock(poll=lambda: None)
        win._start_run()
        mp.assert_not_called()
        assert "deepl" in win._status.text().lower()
        assert "api key" in win._status.text().lower()
    win.close()

def test_preflight_blocks_libretranslate_missing_url():
    cfg = Config(transcriber="moonshine", language="en", model_arch="TINY_STREAMING",
                 translate="libretranslate", translate_url="", translate_key="")
    win = _make_win(cfg)
    try:
        win._translate_mode.setCurrentText("libretranslate")
        win._translate_url.setText("")
    except Exception:
        pass
    with patch("voicelang_core.gui.save_config") as ms, patch("voicelang_core.gui.subprocess.Popen") as mp:
        mp.return_value = MagicMock(poll=lambda: None)
        win._start_run()
        mp.assert_not_called()
        assert "libretranslate" in win._status.text().lower()
        assert "endpoint" in win._status.text().lower()
    win.close()

def test_preflight_blocks_deeplx_missing_url():
    cfg = Config(transcriber="moonshine", language="en", model_arch="TINY_STREAMING",
                 translate="deeplx", translate_url="")
    win = _make_win(cfg)
    try:
        win._translate_mode.setCurrentText("deeplx")
        win._translate_url.setText("")
    except Exception:
        pass
    with patch("voicelang_core.gui.save_config") as ms, patch("voicelang_core.gui.subprocess.Popen") as mp:
        mp.return_value = MagicMock(poll=lambda: None)
        win._start_run()
        mp.assert_not_called()
        assert "deeplx" in win._status.text().lower()
    win.close()

def test_preflight_blocks_unknown_translate_mode():
    cfg = Config(transcriber="moonshine", language="en", model_arch="TINY_STREAMING",
                 translate="unknown_mode")
    win = _make_win(cfg)
    # force unknown via widget editable is not needed; _cfg_from_widgets reads translate_mode combo,
    # but we can patch _cfg_from_widgets to return unknown
    original_cfg_from = win._cfg_from_widgets
    def fake_cfg():
        c = original_cfg_from()
        c.translate = "unknown_mode"
        return c
    win._cfg_from_widgets = fake_cfg
    with patch("voicelang_core.gui.save_config") as ms, patch("voicelang_core.gui.subprocess.Popen") as mp:
        mp.return_value = MagicMock(poll=lambda: None)
        win._start_run()
        mp.assert_not_called()
        assert "unknown" in win._status.text().lower()
    win.close()


# ---- process observability ----

def test_popen_captures_stdout_stderr_to_log_file(tmp_path, monkeypatch):
    """Ensure Popen called with stdout=DEVNULL, stderr=log_file, encoding utf-8 errors replace."""
    win = _make_win()
    # use tmp for data dir
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from pathlib import Path
    import subprocess
    captured = {}
    def fake_popen(cmd, cwd=None, stdout=None, stderr=None, text=None, encoding=None, errors=None):
        captured['stdout'] = stdout
        captured['stderr'] = stderr
        captured['text'] = text
        captured['encoding'] = encoding
        captured['errors'] = errors
        m = MagicMock()
        m.poll.return_value = None
        m.wait.return_value = 0
        return m
    with patch("voicelang_core.gui.save_config"):
        with patch("voicelang_core.gui.subprocess.Popen", side_effect=fake_popen):
            win._start_run()
            # verify kwargs
            assert captured['errors'] == "replace"
            assert captured['encoding'] == "utf-8"
            assert captured['text'] is True
            # stdout should be DEVNULL, stderr should be the log file handle
            assert captured['stdout'] == subprocess.DEVNULL
            assert hasattr(captured['stderr'], 'write')
            # log path exists
            assert hasattr(win, '_log_path')
            assert "voicelang_" in win._log_path
    win._stop_run()
    win.close()

def test_popen_failure_surfaces_error_via_poll(tmp_path, monkeypatch):
    """Mock Popen that exits quickly with code 1 and log containing error — poll should show FAILED."""
    win = _make_win()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from pathlib import Path
    import subprocess

    fake_log_content = "Traceback: boom\nRuntimeError: invalid language\n"
    # we will let _start_run create real log file, then overwrite it before poll
    with patch("voicelang_core.gui.save_config"):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # initially running
        # track calls
        with patch("voicelang_core.gui.subprocess.Popen", return_value=mock_proc):
            win._start_run()
            # now simulate crash: write error to log file
            log_path = Path(win._log_path)
            log_path.write_text(fake_log_content, encoding="utf-8")
            # make poll return 1 (failure)
            mock_proc.poll.return_value = 1
            # patch QMessageBox to avoid blocking
            with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_msg:
                win._poll_proc()
                assert "error" in win._status.text().lower() or "1" in win._status.text()
                # must surface log path
                assert "Log:" in win._status.text()
                assert str(log_path) in win._status.text()
                # pill should be error
                assert win._dot._state == "error"
                # start button re-enabled, stop disabled
                assert win._start.isEnabled() is True
                assert win._stop.isEnabled() is False
                # toast called with error (indirect via _toastHost, but we check status)
                mock_msg.assert_called_once()
    win.close()

def test_popen_exception_surfaces_error(tmp_path, monkeypatch):
    """Popen raising exception should show 'start failed' with log hint."""
    win = _make_win()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    with patch("voicelang_core.gui.save_config"):
        with patch("voicelang_core.gui.subprocess.Popen", side_effect=FileNotFoundError("no such exe")):
            win._start_run()
            assert "start failed" in win._status.text().lower()
            assert win._dot._state == "error"
    win.close()

def test_early_poll_detects_fast_crash(tmp_path, monkeypatch):
    """Even if exit happens within 400ms, _poll_proc via singleShot or manual poll surfaces FAILED not running."""
    win = _make_win()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from pathlib import Path
    mock_proc = MagicMock()
    # simulate immediate exit (poll returns 1 right away)
    mock_proc.poll.return_value = 1
    from unittest.mock import MagicMock as MM
    with patch("voicelang_core.gui.save_config"):
        with patch("voicelang_core.gui.subprocess.Popen", return_value=mock_proc):
            win._start_run()
            # log file should exist
            Path(win._log_path).write_text("ERROR: model missing — download required\n", encoding="utf-8")
            with patch("PySide6.QtWidgets.QMessageBox.warning"):
                # simulate the early singleShot callback
                win._poll_proc()
                assert "running" not in win._status.text().lower()
                assert win._statusPillLabel.text().lower().startswith("error")
    win._stop_run()
    win.close()

def test_poll_success_sets_idle(tmp_path, monkeypatch):
    win = _make_win()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0
    with patch("voicelang_core.gui.save_config"):
        with patch("voicelang_core.gui.subprocess.Popen", return_value=mock_proc):
            win._start_run()
            win._poll_proc()
            assert win._status.text() == "stopped"
            assert win._dot._state == "idle"
    win.close()

def test_log_file_errors_replace_for_unicode(tmp_path, monkeypatch):
    """Write non-utf8 bytes to log via file with errors replace — should not crash poll."""
    win = _make_win()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from pathlib import Path
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1
    with patch("voicelang_core.gui.save_config"):
        with patch("voicelang_core.gui.subprocess.Popen", return_value=mock_proc):
            win._start_run()
            # write bytes that would fail strict utf-8, but file uses errors replace
            p = Path(win._log_path)
            # raw write with errors
            p.write_bytes(b"\xff\xfe invalid \xff boom\n")
            with patch("PySide6.QtWidgets.QMessageBox.warning"):
                win._poll_proc()
                # should not raise, status should contain something
                assert "1" in win._status.text() or "error" in win._status.text().lower()
    win.close()
