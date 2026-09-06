"""Transparent topmost caption overlay (PySide6/Qt).

Pure Display adapter: consumes ONLY the port types (Translation / Segment),
never raw audio or ASR JSON. That keeps the dependency rule intact -- the
overlay depends inward on the Display contract, never sideways on capture or
ASR schema, so it can be built/tested independently of OpenASR validation.
Heavy Qt dep is imported lazily and the constructor NEVER raises: if PySide6
is absent or no display server exists (headless/CI), it falls back to an
in-memory ring buffer (``lines``) instead of opening a window.

Bilingual layout: two side-by-side blocks, each numbered.
  1. hello world   |  1. hola mundo
  2. next sentence  |  2. siguiente frase
Live partial (… text) appears unnumbered in the left block only; finals
are numbered on both sides. Legacy ``lines`` (unified) is kept for tests.

Smoothing (v2, the "smoothest experience" pass):
- Each block is a fixed-size panel holding a read-only QTextEdit whose own
  scrollbar (a slim visual indicator) tracks a FULL scrollable history.
  Long sentences WRAP inside the box (auto next-line), the wheel scrolls the
  history, and auto-scroll follows the newest caption UNLESS the user scrolls
  up (then it stays put).
- The caption text stays click-through (``WA_TransparentForMouseEvents``)
  for game-safety; wheel events land on the panel frame below and are
  forwarded to the text edit by a small event filter.
- Partial updates are coalesced on a 33 ms timer: rapid partials collapse
  into ONE repaint instead of re-laying out the document per event.
- Dragging uses ``QWindow.startSystemMove()``: the OS composites the move,
  eliminating the per-move Python/repaint round-trip that made translucent
  (layered) windows lag. Falls back to manual ``move()`` where unsupported.
"""
from .base import Display  # noqa: F401
from ..types import Segment

try:
    from PySide6 import QtCore as _QtCore  # hoisted from per-call imports (~0.5ms each)
except Exception:
    _QtCore = None  # type: ignore

_HIST_CAP = 1000      # full scrollable history per block
_RENDER_N = 400       # lines actually laid out per repaint (perf guard)
_COALESCE_MS = 33     # partial repaint throttle (~30 Hz)


class OverlayDisplay(Display):
    def __init__(self, max_lines: int = 5, opacity: float = 0.85,
                 headless: bool = False,
                 x: int | None = None, y: int | None = None,
                 width: int | None = None, height: int | None = None,
                 show_transcription_block: bool = True,
                 show_translation_block: bool = True):
        import threading as _thr
        self._lock = _thr.Lock()
        self._max_lines = max_lines
        self._opacity = opacity
        self._headless = headless
        self._geometry = (x, y, width, height)
        self._show_transcription_block = show_transcription_block
        self._show_translation_block = show_translation_block
        # legacy unified buffer (for tests / headless inspection)
        self._lines: list[str] = []
        # bilingual numbered buffers
        self._left_lines: list[str] = []   # transcribed, numbered (capped view)
        self._right_lines: list[str] = []  # translated, numbered (capped view)
        # v2: full scrollable histories + follow flags
        self._hist_left: list[str] = []
        self._hist_right: list[str] = []
        self._follow_left = True
        self._follow_right = True
        self._counter: int = 1
        self._partial: str = ""  # live hypothesis, not yet numbered
        self._app = None
        self._window = None
        self._label = None  # legacy single label (compat)
        self._left_label = None   # alias of the left text edit (grip probe)
        self._right_label = None  # alias of the right text edit
        self._left_edit = None
        self._right_edit = None
        self._left_frame = None
        self._right_frame = None
        self._status_label = None
        self._warn_timer = None
        self._coalesce = None
        self._grip = None
        if not headless:
            self._try_build_window()

    # ---- Display contract --------------------------------------------------

    def _gui_invoke(self, fn):
        """Marshal fn onto Qt GUI thread if needed; fallback to direct call."""
        if self._app is None:
            try:
                fn()
            except Exception:
                pass
            return
        try:
            if _QtCore is None:
                fn()
                return
            app_thread = self._app.thread()
            cur_thread = _QtCore.QThread.currentThread()
            if cur_thread is app_thread:
                fn()
                return
            # Cross-thread: use QueuedConnection via singleShot posted to app thread
            # Create a temporary QObject living in GUI thread to host the call
            proxy = getattr(self, "_qt_proxy", None)
            if proxy is not None:
                # proxy has a signal that queues fn
                try:
                    proxy._queue_fn(fn)  # type: ignore
                    return
                except Exception:
                    pass
            # Fallback: try QMetaObject (may still require proxy)
            fn()
        except Exception:
            try:
                fn()
            except Exception:
                pass

    def show(self, segment: Segment) -> None:
        n = segment.id if segment.id else self._counter
        right_txt = segment.translated_text
        if right_txt is None:
            right_txt = f"⚠ {segment.source_text} (translation unavailable)"
        left_entry = f"{n}. {segment.source_text}"
        right_entry = f"{n}. {right_txt}"
        with self._lock:
            self._hist_left.append(left_entry)
            self._hist_right.append(right_entry)
            self._hist_left = self._hist_left[-_HIST_CAP:]
            self._hist_right = self._hist_right[-_HIST_CAP:]
            self._left_lines.append(left_entry)
            self._right_lines.append(right_entry)
            self._left_lines = self._left_lines[-self._max_lines:]
            self._right_lines = self._right_lines[-self._max_lines:]
            self._push(f"[{n}] {right_txt}", partial=False)
            self._counter = n + 1
            self._partial = ""
        self._schedule_render()

    def show_partial(self, segment: Segment) -> None:
        txt = getattr(segment, "source_text", str(segment))
        with self._lock:
            self._partial = txt
            self._push(f"\u2026 {txt}", partial=True)
        self._schedule_render()

    def warn(self, text: str) -> None:
        """Show a transient advisory line under the blocks (watchdog etc.)."""
        if self._status_label is not None:
            # Must run on GUI thread — try queued, fallback to direct
            def _do_gui():
                try:
                    self._status_label.setText(f"⚠ {text}")
                    self._status_label.setVisible(True)
                    if self._warn_timer is not None:
                        self._warn_timer.start(8000)
                except Exception:
                    pass
            cur = None
            try:
                if _QtCore is None:
                    _do_gui()
                else:
                    cur = _QtCore.QThread.currentThread()
                    if self._app is not None and cur is self._app.thread():
                        _do_gui()
                    elif hasattr(self, "_qt_proxy") and self._qt_proxy is not None:
                        self._qt_proxy._queue_fn(_do_gui)
                    else:
                        _do_gui()
            except Exception:
                _do_gui()
            return
        with self._lock:
            self._push(f"⚠ {text}", partial=False)
        self._schedule_render()

    # ---- internals ----------------------------------------------------------

    def _pump(self) -> None:
        """Let Qt timers/signals fire when capture loop blocks the GUI thread."""
        if self._app is None:
            return
        try:
            self._app.processEvents()
        except Exception:
            pass

    def _schedule_render(self) -> None:
        """Coalesce rapid partials into one repaint (smoothness).

        Must marshal to GUI thread — QTimer/QTextEdit are NOT thread-safe.
        Buffers are already updated synchronously (so tests see them), paint
        is queued to the Qt loop and will run once the loop pumps.
        """
        if self._app is None:
            # headless -> nothing to paint, but force a sync render attempt
            # so fallback paths stay testable
            try:
                self._render()
            except Exception:
                pass
            return
        # If coalesce timer exists, start it on the GUI thread; else queue render.
        if self._coalesce is not None:
            def _start_on_gui():
                try:
                    if not self._coalesce.isActive():
                        self._coalesce.start(_COALESCE_MS)
                except Exception:
                    try:
                        self._render()
                    except Exception:
                        pass
            self._gui_invoke(_start_on_gui)
            # When the caller IS the GUI thread (e.g. pump loop), the queued
            # start above is already executed; when caller is a worker thread
            # the timer start is queued. In the blocking-capture case the GUI
            # thread is busy in the capture loop — pump once so queued work
            # gets a chance to run without waiting for the next loop iteration.
            # Only pump if we're on the GUI thread to avoid reentrancy.
            try:
                if _QtCore is not None and _QtCore.QThread.currentThread() is self._app.thread():
                    self._pump()
            except Exception:
                pass
        else:
            # No coalesce timer (very early or fallback) -> queue render directly
            self._gui_invoke(self._render)

    def _render(self) -> None:
        # Must run on GUI thread — if we're on a worker, re-queue and return.
        if self._app is not None:
            try:
                if _QtCore is not None and _QtCore.QThread.currentThread() is not self._app.thread():
                    self._gui_invoke(self._render)
                    return
            except Exception:
                pass
        if self._app is None:
            return
        with self._lock:
            left_doc = "\n".join(self._hist_left[-_RENDER_N:])
            partial = self._partial
            right_doc = "\n".join(self._hist_right[-_RENDER_N:])
        if partial:
            preview = f"\u2026 {partial}"
            left_doc = (left_doc + "\n" + preview) if left_doc else preview
        if self._left_edit is not None:
            self._paint_block(self._left_edit, left_doc, self._follow_left)
        if self._right_edit is not None:
            self._paint_block(self._right_edit, right_doc, self._follow_right)
        if self._label is not None and self._left_edit is None:
            # fallback single-label mode (very old window)
            self._label.setText("\n".join(self._lines))
        if self._app is not None:
            self._app.processEvents()

    @staticmethod
    def _paint_block(edit, text: str, follow: bool) -> None:
        sb = edit.verticalScrollBar()
        m_old = max(sb.maximum(), 1)
        v_old = sb.value()
        edit.setPlainText(text)
        if follow:
            sb.setValue(sb.maximum())
        elif m_old > 0:
            # Keep the user's scroll POSITION (ratio) across the relayout.
            sb.setValue(int(v_old / m_old * max(sb.maximum(), 1)))

    def _push(self, text: str, partial: bool) -> None:
        prefix = "\u2588 " if partial else "\u25cf "
        self._lines.append(prefix + text)
        self._lines = self._lines[-self._max_lines:]

    def _try_build_window(self) -> None:
        try:
            from PySide6 import QtWidgets, QtCore, QtGui  # type: ignore

            def _global_pos(event):  # Qt6 globalPosition() vs Qt5 globalPos() compat
                try:
                    return event.globalPosition().toPoint()
                except AttributeError:
                    return event.globalPos()

            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

            class _DraggableOverlay(QtWidgets.QWidget):
                """Frameless topmost band — only grip drags (no window drag)."""
                def __init__(self):
                    super().__init__()
                    self._drag_pos = None

            class _DragHandle(QtWidgets.QLabel):
                """Small grip — only this handle drags the parent overlay."""

                def __init__(self, parent_overlay: QtWidgets.QWidget):
                    super().__init__("⠿ drag", parent_overlay)
                    self._parent_overlay = parent_overlay
                    self.setStyleSheet("color: rgba(255,255,255,180); font: 10px 'Segoe UI'; background: transparent; padding: 2px 6px;")
                    self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
                    self.setToolTip("Drag to move — rest of overlay is click-through")

                def mousePressEvent(self, event):  # type: ignore[override]
                    if event.button() == QtCore.Qt.MouseButton.LeftButton:
                        # OS-composited drag: smooth on layered/translucent
                        # windows (no per-move repaint round-trip).
                        wh = self.window().windowHandle()
                        if wh is not None and hasattr(wh, "startSystemMove"):
                            try:
                                if wh.startSystemMove():
                                    event.accept()
                                    return
                            except Exception:  # noqa: BLE001 -- fall back
                                pass
                        gp = _global_pos(event)
                        self._parent_overlay._drag_pos = gp - self._parent_overlay.pos()
                        event.accept()

                def mouseMoveEvent(self, event):  # type: ignore[override]
                    if self._parent_overlay._drag_pos is not None and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
                        gp = _global_pos(event)
                        self._parent_overlay.move(gp - self._parent_overlay._drag_pos)
                        event.accept()

                def mouseReleaseEvent(self, event):  # type: ignore[override]
                    self._parent_overlay._drag_pos = None
                    event.accept()

            class _WheelForwarder(QtCore.QObject):
                """The caption text is click-through (game-safe), so wheel
                events land on the panel frame below it. This filter (on the
                frame) scrolls the block's text edit instead."""

                def __init__(self, edit):
                    super().__init__(edit)
                    self._edit = edit

                def eventFilter(self, obj, ev):  # type: ignore[override]
                    if ev.type() == QtCore.QEvent.Type.Wheel and self._edit is not None:
                        sb = self._edit.verticalScrollBar()
                        sb.setValue(sb.value() - ev.angleDelta().y())
                        return True
                    return super().eventFilter(obj, ev)

            w = _DraggableOverlay()
            w.setWindowFlags(
                QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
                | QtCore.Qt.WindowType.Tool
            )
            w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
            w.setWindowOpacity(self._opacity)

            # Main vertical layout: grip on top, then bilingual HBox
            outer = QtWidgets.QVBoxLayout(w)
            outer.setContentsMargins(8, 8, 8, 8)
            outer.setSpacing(4)

            # Grip row (right-aligned)
            grip = _DragHandle(w)
            grip.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            outer.addWidget(grip)

            # Bilingual columns
            cols = QtWidgets.QHBoxLayout()
            cols.setSpacing(8)

            header_style = "color: rgba(255,255,255,140); font: 700 9px 'Segoe UI'; letter-spacing: 1px; background: transparent; border: none; padding: 0 0 2px 0;"
            frame_style = (
                "QWidget { background: rgba(0,0,0,140); border-radius: 10px; }"
            )
            text_style = (
                "color: white; background: transparent; "
                "font: 600 13px 'Segoe UI'; border: none;"
                " QScrollBar:vertical { width: 6px; background: transparent; margin: 0; }"
                " QScrollBar::handle:vertical { background: rgba(255,255,255,90);"
                " border-radius: 3px; min-height: 20px; }"
                " QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
                " QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }"
            )

            def _make_block(header_text: str, side: str):
                """Fixed-size scrollable caption block (v2 smooth renderer)."""
                frame = QtWidgets.QWidget()
                frame.setStyleSheet(frame_style)
                col = QtWidgets.QVBoxLayout(frame)
                col.setContentsMargins(10, 6, 6, 8)
                col.setSpacing(2)
                header = QtWidgets.QLabel(header_text)
                header.setStyleSheet(header_style)
                edit = QtWidgets.QTextEdit()
                edit.setReadOnly(True)
                edit.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
                edit.setStyleSheet(text_style)
                edit.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
                edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                # Game-safe: caption text never consumes clicks. Wheel events
                # fall through to the frame, whose filter scrolls the edit.
                edit.setAttribute(
                    QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
                )
                frame.installEventFilter(_WheelForwarder(edit))

                def _follow_ctl(scrollbar):
                    def _ctl(value: int) -> None:
                        flag = value >= scrollbar.maximum()
                        if side == "left":
                            self._follow_left = flag
                        else:
                            self._follow_right = flag
                    scrollbar.valueChanged.connect(_ctl)
                    return _ctl

                _follow_ctl(edit.verticalScrollBar())

                col.addWidget(header)
                col.addWidget(edit, 1)
                return frame, edit

            left_frame, left_edit = _make_block("TRANSCRIBED", "left")
            right_frame, right_edit = _make_block("TRANSLATION", "right")

            # Independent block toggles (spec §6): hide the unwanted block.
            if not self._show_transcription_block:
                left_frame.hide()
            if not self._show_translation_block:
                right_frame.hide()

            cols.addWidget(left_frame, 1)
            cols.addWidget(right_frame, 1)
            outer.addLayout(cols, 1)

            # Advisory line (no-audio watchdog, etc.) — transient, auto-clears.
            status_label = QtWidgets.QLabel("")
            status_style = "color: rgba(255,200,80,230); font: 600 11px 'Segoe UI'; background: transparent; border: none;"
            status_label.setStyleSheet(status_style)
            status_label.setVisible(False)
            outer.addWidget(status_label)

            # Legacy fallback label hidden but kept for compat (some probes look for _label)
            legacy = QtWidgets.QLabel("")
            legacy.setVisible(False)
            outer.addWidget(legacy)

            # Geometry (fixed size from config; auto bottom-center when unset)
            # Clamp opacity & size so a bad config never yields an invisible window.
            try:
                op = float(self._opacity)
            except Exception:
                op = 0.85
            # Visible range: 0.2 .. 1.0 (below 0.2 is near-invisible, clamp up)
            op = max(0.2, min(1.0, op))
            self._opacity = op
            screen = app.primaryScreen()
            x, y, width, height = self._geometry
            # Sanitize dimensions — guard against zero/negative/huge from config
            if width is not None:
                try:
                    width = int(width)
                except Exception:
                    width = None
                if width is not None and (width < 320 or width > 2500):
                    width = max(320, min(2500, width))
                if width is not None and width < 100:
                    width = 320
            if height is not None:
                try:
                    height = int(height)
                except Exception:
                    height = None
                if height is not None and (height < 80 or height > 1200):
                    height = max(80, min(1200, height))
                if height is not None and height < 40:
                    height = 120
            if x is not None:
                try:
                    x = int(x)
                except Exception:
                    x = None
            if y is not None:
                try:
                    y = int(y)
                except Exception:
                    y = None
            if screen is not None:
                geo = screen.availableGeometry()
                if width is None:
                    width = max(640, geo.width() - 80)
                if height is None:
                    height = 180
                if x is None:
                    x = geo.x() + 40
                if y is None:
                    y = geo.y() + geo.height() - height - 40
                # Off-screen guard: if requested xy is fully outside the
                # available geometry, fall back to bottom-center.
                # Allows manual placement but never lets a typo hide the overlay.
                if (x + width < geo.x() or x > geo.x() + geo.width()
                        or y + height < geo.y() or y > geo.y() + geo.height()):
                    x = geo.x() + 40
                    y = geo.y() + geo.height() - height - 40
                w.setGeometry(x, y, width, height)
            else:
                w.setGeometry(x or 40, y or 40, width or 900, height or 180)
            # Re-apply clamped opacity (original was before clamp)
            try:
                w.setWindowOpacity(float(self._opacity))
            except Exception:
                pass
            w.show()
            app.processEvents()

            self._app, self._window, self._label = app, w, legacy
            self._left_frame, self._right_frame = left_frame, right_frame
            self._left_edit, self._right_edit = left_edit, right_edit
            # Aliases preserved so the grip-invariant probe and any tooling
            # checking WA_TransparentForMouseEvents keep working.
            self._left_label, self._right_label = left_edit, right_edit
            self._grip = grip
            self._status_label = status_label

            # Coalescing timer: partial bursts fold into one repaint.
            self._coalesce = QtCore.QTimer()
            self._coalesce.setSingleShot(True)
            self._coalesce.timeout.connect(self._render)

            # Advisory auto-clear.
            self._warn_timer = QtCore.QTimer()
            self._warn_timer.setSingleShot(True)
            self._warn_timer.timeout.connect(status_label.hide)

            # Cross-thread proxy: lets worker threads queue a fn onto GUI thread
            # via QueuedConnection (no direct Qt calls from workers)
            try:
                class _Proxy(QtCore.QObject):
                    _sig = QtCore.Signal(object)
                    def __init__(self):
                        super().__init__()
                        self._sig.connect(lambda fn: fn(), QtCore.Qt.ConnectionType.QueuedConnection)
                    def _queue_fn(self, fn):
                        self._sig.emit(fn)
                _proxy = _Proxy()
                # Move proxy to GUI thread explicitly
                _proxy.moveToThread(app.thread())
                self._qt_proxy = _proxy
            except Exception:
                self._qt_proxy = None

            self._render()  # flush anything that arrived before the window
        except Exception:  # noqa: BLE001 -- any Qt/display failure -> headless
            self._headless = True
            self._app = self._window = self._label = None
            self._left_label = self._right_label = None
            self._left_edit = self._right_edit = None
            self._left_frame = self._right_frame = None

    @property
    def headless(self) -> bool:
        """True when no Qt window could be opened (missing PySide6 / no display).

        Callers must surface this: a silent headless fallback reads as
        "captions don't work" with no clue why.
        """
        return self._headless

    @property
    def lines(self) -> list[str]:
        return list(self._lines)

    # New helpers for inspection / tests
    @property
    def transcribed_lines(self) -> list[str]:
        return list(self._left_lines)

    @property
    def translated_lines(self) -> list[str]:
        return list(self._right_lines)

    @property
    def numbered_entries(self) -> list[tuple[int, str, str]]:
        """Return [(n, transcribed, translated), ...] for the visible history."""
        n = min(len(self._left_lines), len(self._right_lines))
        out = []
        for i in range(n):
            # lines already have "N. text" prefix; strip for raw
            l = self._left_lines[i]
            r = self._right_lines[i]
            # remove leading "N. " using the known prefix, not first ". "
            def strip_num(s: str, idx: int = i) -> str:
                prefix = f"{idx+1}. "
                return s[len(prefix):] if s.startswith(prefix) else (s[s.find(". ")+2:] if ". " in s else s)
            out.append((i+1, strip_num(l), strip_num(r)))
        return out