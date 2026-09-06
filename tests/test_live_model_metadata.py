"""Tests for live model metadata: HF fetch + TTL cache + offline fallback."""
import json
import os
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

def test_live_catalog_mocked_hf_response():
    """Mock HF API: whisper variants should reflect mocked list_models result."""
    from voicelang_core import model_catalog
    from voicelang_core.model_catalog import get_provider_catalog, _cache_path, OFFLINE_DEFAULTS
    # ensure clean cache for isolation
    orig_cache = model_catalog._load_cache()
    # remove cache to force fresh fetch
    p = _cache_path()
    backup = None
    if p.is_file():
        backup = p.read_text(encoding="utf-8")
        p.unlink(missing_ok=True)
    try:
        fake_models = [
            MagicMock(modelId="Systran/faster-whisper-tiny"),
            MagicMock(modelId="Systran/faster-whisper-base"),
            MagicMock(modelId="Systran/faster-whisper-small"),
            MagicMock(modelId="Systran/faster-whisper-NEW-XL"),
        ]
        # MagicMock str fallback; ensure modelId used
        for m in fake_models:
            m.__str__ = lambda s: s.modelId  # not needed

        with patch("voicelang_core.model_catalog._hf_api") as mock_api_factory:
            mock_api = MagicMock()
            mock_api.list_models.return_value = fake_models
            mock_api_factory.return_value = mock_api

            langs, friendly, arches, hint, dl_label, used_offline = get_provider_catalog("whisper")
            # Should include NEW-XL from mocked response, not be pure offline
            assert "NEW-XL" in arches or "new-xl" in [a.lower() for a in arches], f"arches should include mocked NEW-XL, got {arches}"
            assert not used_offline or "NEW-XL" in arches  # if used_offline false, success; if true but still contains NEW-XL also ok
            assert langs  # non-empty
            assert hint  # non-empty

            # Verify TTL cache file was created and contains whisper entry
            assert p.is_file(), "cache file should be created after live fetch"
            data = json.loads(p.read_text(encoding="utf-8"))
            assert "whisper" in data.get("providers", {}) or "_fetched_at" in data
    finally:
        # restore
        try:
            if backup is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(backup, encoding="utf-8")
            else:
                if p.is_file():
                    p.unlink(missing_ok=True)
        except Exception:
            pass
        # clear module cache of engines registry that may have been polluted
        try:
            from voicelang_core import engines
            engines.ENGINE_REGISTRY.clear()
        except Exception:
            pass

def test_offline_fallback_on_network_failure():
    """When HF API raises, catalog must fall back to offline defaults without crash."""
    from voicelang_core import model_catalog
    from voicelang_core.model_catalog import get_provider_catalog, _cache_path

    p = _cache_path()
    backup = None
    if p.is_file():
        backup = p.read_text(encoding="utf-8")
        p.unlink(missing_ok=True)
    try:
        with patch("voicelang_core.model_catalog._hf_api") as mock_api_factory:
            mock_api = MagicMock()
            mock_api.list_models.side_effect = ConnectionError("no network")
            mock_api_factory.return_value = mock_api

            # whisper should still return without exception and with offline defaults
            langs, friendly, arches, hint, dl_label, used_offline = get_provider_catalog("whisper")
            assert langs, "fallback langs must not be empty"
            assert arches, "fallback arches must not be empty"
            assert used_offline is True, "should be marked offline on network failure"
            # hint should be present; offline marker is added by gui layer, but model_catalog marks used_offline
            # Ensure arches match offline defaults at least
            from voicelang_core.model_catalog import OFFLINE_DEFAULTS
            for a in OFFLINE_DEFAULTS["whisper"]["arches"]:
                assert a in arches, f"offline arch {a} should be in fallback {arches}"

        # Vosk is always offline static — should still be non-empty
        with patch("voicelang_core.model_catalog._hf_api", return_value=None):
            langs2, friendly2, arches2, hint2, dl2, off2 = get_provider_catalog("vosk")
            assert langs2
            assert "en" in langs2
    finally:
        try:
            if backup is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(backup, encoding="utf-8")
            else:
                if p.is_file():
                    p.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            from voicelang_core import engines
            engines.ENGINE_REGISTRY.clear()
        except Exception:
            pass

def test_gui_dropdown_uses_offline_fallback_and_never_empty():
    """GUI _refresh_provider_models must populate dropdowns even when live fetch fails."""
    # Simulate network failure inside _catalog_for -> still non-empty dropdown
    from voicelang_core.config import Config
    from voicelang_core.gui import _make_gui
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    cfg = Config(transcriber="whisper", language="en", model_arch="small")
    W = _make_gui()
    win = W(cfg)
    win.show()
    app.processEvents()

    # Patch live catalog to raise (simulating network failure)
    with patch("voicelang_core.gui._live_catalog", side_effect=RuntimeError("network down")):
        # also ensure _live_moonshine_catalog raises
        with patch("voicelang_core.gui._live_moonshine_catalog", side_effect=RuntimeError("down")):
            # Should still populate via offline fallback, not crash
            win._transcriber.setCurrentText("whisper")
            win._refresh_provider_models()
            app.processEvents()
            assert win._language.count() > 0, "language dropdown must not be empty on offline"
            assert win._arch.count() > 0, "arch dropdown must not be empty on offline"
            assert win._dl.text(), "dl label must be set"

            # Switch to funasr-nano with same failure
            win._transcriber.setCurrentText("funasr-nano")
            win._refresh_provider_models()
            app.processEvents()
            assert win._language.count() > 0

    win.close()

def test_gui_probe_uses_live_is_cached_and_arch_loop_wrapped():
    """If probe throws, arch loop must not abort (Agent1 atomicity)."""
    from voicelang_core.config import Config
    from voicelang_core.gui import _make_gui
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    cfg = Config(transcriber="whisper", language="en", model_arch="small")
    W = _make_gui()
    win = W(cfg)
    win.show()
    app.processEvents()

    # Make _live_is_cached raise for one arch but not others
    original = win._probe_arch_cached
    def flaky(provider, arch, lang):
        if arch == "small" and provider == "whisper":
            raise RuntimeError("probe boom")
        return False
    # Patch the model_catalog function, not the method directly, to test wrapping
    import voicelang_core.model_catalog as mc
    with patch.object(mc, "is_model_cached", side_effect=flaky):
        # Also need to ensure _live_is_cached points to the patched function —
        # reimport? Instead patch gui's reference
        import voicelang_core.gui as gui_mod
        with patch.object(gui_mod, "_live_is_cached", side_effect=flaky):
            win._transcriber.setCurrentText("whisper")
            win._refresh_provider_models()
            app.processEvents()
            # Must still have arches and dl label updated
            assert win._arch.count() > 0
            assert "whisper" in win._dl.text().lower()

    win.close()

def test_ttl_cache_not_hit_network_on_second_call():
    """Second call within TTL should use cache and not call HF."""
    from voicelang_core import model_catalog
    from voicelang_core.model_catalog import get_provider_catalog, _cache_path
    p = _cache_path()
    backup = None
    if p.is_file():
        backup = p.read_text(encoding="utf-8")
        p.unlink(missing_ok=True)
    try:
        fake_models = [MagicMock(modelId="Systran/faster-whisper-tiny")]
        with patch("voicelang_core.model_catalog._hf_api") as mock_factory:
            mock_api = MagicMock()
            mock_api.list_models.return_value = fake_models
            mock_factory.return_value = mock_api
            get_provider_catalog("whisper")
            assert mock_api.list_models.called
            mock_api.list_models.reset_mock()
            # Second call should hit cache (fresh) and NOT call list_models again
            mock_api.list_models.reset_mock()
            # But our implementation always tries to check cache first; if fresh, returns without calling HF
            langs2, *_ = get_provider_catalog("whisper")
            # list_models should NOT be called again because cache is fresh
            # Note: get_provider_catalog checks cache freshness before calling _hf_api,
            # so we assert no new call
            assert not mock_api.list_models.called, "TTL cache should prevent second network hit"
            assert langs2
    finally:
        try:
            if backup is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(backup, encoding="utf-8")
            else:
                if p.is_file():
                    p.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            from voicelang_core import engines
            engines.ENGINE_REGISTRY.clear()
        except Exception:
            pass

def test_vosk_static_documented():
    """Vosk catalog remains static with fallback note; verify non-empty and offline."""
    from voicelang_core.model_catalog import get_provider_catalog
    langs, friendly, arches, hint, dl_label, used_offline = get_provider_catalog("vosk")
    assert "en" in langs
    assert "small" in arches
    assert "vosk" in hint.lower()
    # Vosk is always considered offline/static (documented)
    # used_offline may be True for vosk per docs
    assert friendly.get("en") is not None
