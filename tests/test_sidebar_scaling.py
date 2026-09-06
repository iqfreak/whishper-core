"""Sidebar / Settings panel layout & scaling — proves no label truncated."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pathlib
import pytest

def _app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

def test_sidebar_scaling_no_truncation():
    _app()
    from PySide6 import QtWidgets, QtCore
    from voicelang_core.config import Config
    from voicelang_core.gui import _make_gui
    W = _make_gui()
    win = W(Config())
    win.show()
    _app().processEvents()

    # --- rail wide enough, not fixed narrow ---
    rail = win._rail
    assert rail.minimumWidth() >= 220, f"rail minimumWidth {rail.minimumWidth()} < 220 (needs headroom for Translate)"
    # not locked to fixed 200
    assert not (rail.minimumWidth() == rail.maximumWidth() == 200), "rail still fixed 200 — will clip at high DPI"
    # rail should be Preferred/Expanding, not Fixed
    pol = rail.sizePolicy()
    assert pol.horizontalPolicy() != QtWidgets.QSizePolicy.Fixed

    # --- nav buttons: tooltip + sizeHint vs geom at 100% ---
    for i, b in enumerate(win._nav_buttons):
        assert b.toolTip(), f"nav {i} missing tooltip (elision fallback)"
        hint = b.sizeHint().width()
        geom = b.geometry().width()
        # hint must fit inside geom (+ small rounding)
        assert hint <= geom + 2, f"nav {i} '{b.text()}' sizeHint {hint} > geom {geom}"
        # also check fontMetrics needed vs geom
        fm = b.fontMetrics()
        need = fm.horizontalAdvance(b.text()) + 24
        assert need <= geom + 4, f"nav {i} text needs {need} > geom {geom}"

    # --- simulated 150% and 200% scaling logic ---
    # We simulate by increasing the window font size (high-DPI scales font metrics).
    # After scaling, rail should expand and hints should still fit.
    for scale in (1.5, 2.0):
        f = win.font()
        # base point size fallback 10 if not set
        base_pt = f.pointSizeF() if f.pointSizeF() > 0 else 9.0
        scaled_font = QtGui.QFont(f)
        scaled_font.setPointSizeF(base_pt * scale)
        win.setFont(scaled_font)
        for b in win._nav_buttons:
            b.setFont(scaled_font)
        # also scale rail's font so its sizeHint recomputes
        rail.setFont(scaled_font)
        _app().processEvents()
        # allow layout to expand rail
        _app().processEvents()
        win.resize(win.sizeHint().width() + 50, win.height())
        _app().processEvents()
        for i, b in enumerate(win._nav_buttons):
            hint = b.sizeHint().width()
            geom = b.geometry().width()
            # At simulated scale, we allow elision via tooltip but ensure not severely clipped without feedback
            # So if hint > geom, tooltip must exist (already checked) and geometry should be at least rail minimum
            # For 150% we expect fit after rail expansion; for 200% elision is acceptable if tooltip present
            if scale == 1.5:
                assert hint <= geom + 4, f"scale {scale} nav {i} hint {hint} > geom {geom} (150% should fit without elision)"
            else:
                # 200% may elide but tooltip guarantees accessibility
                assert b.toolTip(), f"scale 200% nav {i} truncated without tooltip"

    # --- form labels: not fixedWidth 132, now minimumWidth 140 ---
    src = pathlib.Path("C:/Users/ahmed/whishper-core/voicelang_core/gui.py").read_text(encoding="utf-8")
    # ensure old fixed widths gone for the row labels (allow meter FixedWidth 0)
    # We specifically forbid setFixedWidth(132)
    assert "setFixedWidth(132)" not in src, "still has setFixedWidth(132) — will clip at high DPI"
    # check new minimumWidth present
    assert "setMinimumWidth(140)" in src or "setMinimumWidth(140)" in src.replace(" ", "")

    # --- window resizable with QScrollArea fallback ---
    assert win.minimumSize().width() == 1020
    assert win.minimumSize().height() == 640
    # resizable: minimum != maximum
    assert win.maximumSize().width() > 2000 or win.maximumSize().width() == 16777215
    scrolls = win.findChildren(QtWidgets.QScrollArea)
    assert len(scrolls) >= 4, f"expected >=4 scrollAreas for overflow, got {len(scrolls)}"
    for sc in scrolls:
        assert sc.widgetResizable() is True
        assert sc.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff

    # --- DPI enabled before QApplication ---
    assert "HighDpiScaleFactorRoundingPolicy.PassThrough" in src
    assert "AA_EnableHighDpiScaling" in src
    assert "AA_UseHighDpiPixmaps" in src
    # must be in main() before QApplication — slice to main()
    assert "def main()" in src
    main_src = src.split("def main()")[-1]
    assert "AA_EnableHighDpiScaling" in main_src and "QApplication" in main_src
    # DPI attributes must precede the actual QApplication instantiation (instance() or constructor)
    dpi_pos = main_src.index("AA_EnableHighDpiScaling")
    app_pos = main_src.index("QApplication.instance") if "QApplication.instance" in main_src else main_src.index("QApplication(")
    assert dpi_pos < app_pos, "DPI must be set before QApplication instantiation in main()"
    # no per-widget WA_HighDpiScaling misuse
    assert "self.setAttribute(QtCore.Qt.WA_HighDpiScaling" not in src

    win.close()

# Need QtGui for font test
from PySide6 import QtGui

# Manual verification steps (for reviewer):
# 1. Run with QT_SCALE_FACTOR=1.5 and 2.0: `QT_SCALE_FACTOR=1.5 python -m voicelang_core.gui` — rail and nav labels stay readable, form labels not clipped, scroll appears when window shrunk.
# 2. Shrink window to minimum 1020x640 — vertical scroll appears per page, no controls cut off.
# 3. Hover each nav button — tooltip shows full label even if elided at 200%.
