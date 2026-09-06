"""Regression: provider dropdown must sync language, arch, download label atomically."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from voicelang_core.config import Config
from voicelang_core.gui import _make_gui, PROVIDERS, _catalog_for, _moonshine_catalog

def _app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

pytestmark = pytest.mark.skipif(not _app, reason="PySide6 not available")

def _expected(provider):
    moon_langs, moon_friendly = _moonshine_catalog()
    catalog = _catalog_for(provider, moon_langs, moon_friendly)
    if len(catalog) == 5:
        langs, friendly, arches, hint, dl_label = catalog
    else:
        langs = list(catalog)
        arches = ["default"]
        dl_label = f"Download {provider} model"
    return set(str(x).lower() for x in langs), set(str(x).lower() for x in arches), dl_label

def test_provider_switch_syncs_all_three_for_every_pair():
    _app()
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg = Config(transcriber=PROVIDERS[0], language="en", model_arch="small")
    W = _make_gui()
    win = W(cfg)
    win.show()
    app.processEvents()
    # Stub probe to keep test fast (avoids HF cache scans per arch)
    orig_probe = win._probe_arch_cached
    win._probe_arch_cached = lambda provider, arch, lang: False
    failures = []
    try:
        for src in PROVIDERS:
            for dst in PROVIDERS:
                if src == dst:
                    continue
                win._transcriber.setCurrentText(src)
                win._refresh_provider_models()
                app.processEvents()
                win._transcriber.setCurrentText(dst)
                win._refresh_provider_models()
                app.processEvents()

                exp_langs, exp_arches, exp_dl = _expected(dst)
                cur_lang = str(win._language.currentData() or win._language.currentText() or "").lower()
                cur_arch = str(win._arch.currentData() or win._arch.currentText() or "").lower()
                dl_text = str(win._dl.text() or "")

                if cur_lang not in exp_langs:
                    failures.append(f"{src}->{dst}: language {cur_lang!r} not in {exp_langs}")
                if cur_arch not in exp_arches:
                    failures.append(f"{src}->{dst}: arch {cur_arch!r} not in {exp_arches}")
                if dst.lower() not in dl_text.lower() and exp_dl.lower() not in dl_text.lower():
                    failures.append(f"{src}->{dst}: dl label {dl_text!r} missing expected {exp_dl!r} / {dst}")
                if src == "nemotron" and dst == "vosk":
                    if "nemotron" in dl_text.lower():
                        failures.append(f"nemotron->vosk leak: dl label still {dl_text!r}")
                moon_langs, moon_friendly = _moonshine_catalog()
                langs, _, arches, _, _ = _catalog_for(dst, moon_langs, moon_friendly)
                if win._language.count() != len(langs):
                    failures.append(f"{src}->{dst}: language count {win._language.count()} != {len(langs)}")
                if win._arch.count() != len(arches):
                    failures.append(f"{src}->{dst}: arch count {win._arch.count()} != {len(arches)}")
    finally:
        win._probe_arch_cached = orig_probe
    win.close()
    assert not failures, "Sync failures:\n" + "\n".join(failures[:40])

def test_atomic_survives_probe_exception():
    """If arch probing throws, dl label must still update."""
    _app()
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg = Config(transcriber="nemotron", language="en", model_arch="int8")
    W = _make_gui()
    win = W(cfg)
    win.show()
    win._transcriber.setCurrentText("nemotron")
    win._refresh_provider_models()
    assert "nemotron" in win._dl.text().lower()
    orig = win._probe_arch_cached
    def boom(provider, arch, lang):
        if provider == "vosk" and arch == "small":
            raise RuntimeError("simulated probe failure")
        return orig(provider, arch, lang)
    win._probe_arch_cached = boom
    win._transcriber.setCurrentText("vosk")
    win._refresh_provider_models()
    app.processEvents()
    assert "vosk" in win._dl.text().lower(), f"dl label not updated after probe failure: {win._dl.text()!r}"
    assert "nemotron" not in win._dl.text().lower()
    exp_langs, _, _ = _expected("vosk")
    assert str(win._language.currentData()).lower() in exp_langs
    win._probe_arch_cached = orig
    win.close()
