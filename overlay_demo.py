"""Show the overlay with real captions for ~8s, then save a screenshot.

Run:  uv run python overlay_demo.py
"""
from PySide6 import QtCore, QtWidgets

from voicelang_core.adapters.overlay import OverlayDisplay
from voicelang_core.types import Segment

d = OverlayDisplay()
if d.headless:
    raise SystemExit("headless -- overlay did not open")

d.show_partial(Segment(status="partial", source_text="I think the mission"))
d.show_partial(Segment(status="partial", source_text="I think the mission is"))
d.show(Segment(id=1, status="final",
               source_text="I think the mission is a go.",
               translated_text="أعتقد أن المهمة جاهزة.",
               source_language="en"))
d.show_partial(Segment(status="partial", source_text="moving to"))
d.show(Segment(id=2, status="final",
               source_text="Moving to the extraction point.",
               translated_text="التوجه إلى نقطة الاستخراج.",
               source_language="en"))

app = QtWidgets.QApplication.instance()
for _ in range(40):
    app.processEvents()
QtCore.QTimer.singleShot(4000, app.quit)
app.exec()

w = d._window
pix = w.grab()
out = "overlay_screenshot.png"
print("saved" if pix.save(out) else "save FAILED", out, pix.width(), "x", pix.height())