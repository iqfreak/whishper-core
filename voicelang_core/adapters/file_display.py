"""File sink Display — queued JSONL+TXT off the hot path.

Writes finals to %LOCALAPPDATA%/voicelang/captions.{jsonl,txt} via a
bounded queue + daemon worker so enqueue is O(1) ~0.02ms and never blocks
the capture/ASR threads. The queue is bounded (drop-oldest) like the
pipeline queues; every drop is counted.

Contract mirrors ConsoleDisplay: show() receives the SAME Segment id that
show_partial() previewed, so file rows stay in sync with overlay/console.

Usage:
  FileDisplay()  # defaults to LOCALAPPDATA/voicelang/captions.*
  MultiDisplay([OverlayDisplay(...), FileDisplay()])  # overlay + file
  FileDisplay(partial=True)  # also log partials (debug)
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

from .base import Display
from ..types import Segment


def _normalize(p: str | os.PathLike | None) -> Path | None:
    if p is None:
        return None
    s = str(p)
    # MSYS bash passes /c/Users/... but native Win python (no conversion) sees it literal
    # -> normalize to C:/Users/...
    if len(s) >= 3 and s[0] == "/" and s[2] == "/" and s[1].isalpha():
        s = f"{s[1].upper()}:/{s[3:]}"
    return Path(s)


def _default_log_dir() -> Path:
    from ..config import voicelang_data_dir as _data_dir

    return _data_dir()


def _default_paths() -> tuple[Path, Path]:
    from ..config import DEFAULT_CAPTIONS_JSONL as _J, DEFAULT_CAPTIONS_TXT as _T

    d = _default_log_dir()
    return d / _J, d / _T

_HIST_CAP = 1000


class FileDisplay(Display):
    def __init__(
        self,
        jsonl_path: str | os.PathLike | None = None,
        txt_path: str | os.PathLike | None = None,
        *,
        also_partial: bool = False,
        max_queue: int = 2048,
        flush_every: int = 8,
    ):
        jp_def, tp_def = _default_paths()
        self._jsonl_path = _normalize(jsonl_path) if jsonl_path is not None else jp_def
        self._txt_path = _normalize(txt_path) if txt_path is not None else tp_def
        self._also_partial = also_partial
        self._flush_every = max(1, flush_every)
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self._written = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # in-memory mirrors for tests (like Overlay headless) — bounded
        self._lines: list[str] = []
        self._shown: list[Segment] = []
        self._partials: list[Segment] = []
        try:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            if self._txt_path.parent != self._jsonl_path.parent:
                self._txt_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._thread = threading.Thread(target=self._worker, daemon=True, name="file-sink")
        self._thread.start()

    # -- Display contract ---------------------------------------------------
    def show(self, segment: Segment) -> None:
        with self._lock:
            self._shown.append(segment)
            if len(self._shown) > _HIST_CAP:
                self._shown = self._shown[-_HIST_CAP:]
            line = f"[{segment.id}] {segment.source_text}"
            if segment.translated_text is not None:
                line += f" | {segment.translated_text}"
            self._lines.append(line)
            if len(self._lines) > _HIST_CAP:
                self._lines = self._lines[-_HIST_CAP:]
        rec = {
            "id": int(segment.id),
            "status": "final",
            "source_text": segment.source_text,
            "translated_text": segment.translated_text,
            "source_language": segment.source_language,
            "timestamp": float(getattr(segment, "timestamp", 0.0) or 0.0),
            "wall_time": time.time(),
        }
        self._enqueue(rec)

    def show_partial(self, segment: Segment) -> None:
        with self._lock:
            self._partials.append(segment)
            if len(self._partials) > _HIST_CAP:
                self._partials = self._partials[-_HIST_CAP:]
        if not self._also_partial:
            return
        rec = {
            "id": int(segment.id) if segment.id else 0,
            "status": "partial",
            "source_text": segment.source_text,
            "translated_text": None,
            "source_language": getattr(segment, "source_language", None),
            "timestamp": float(getattr(segment, "timestamp", 0.0) or 0.0),
            "wall_time": time.time(),
        }
        self._enqueue(rec)

    def warn(self, text: str) -> None:
        rec = {"id": 0, "status": "warn", "source_text": text, "wall_time": time.time()}
        self._enqueue(rec)

    # -- internals ----------------------------------------------------------
    def _enqueue(self, rec: dict) -> None:
        try:
            self._q.put_nowait(rec)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(rec)
            except queue.Full:
                self._dropped += 1

    def _worker(self) -> None:
        # buffered, line-buffered files; flush in batches
        jf = None
        tf = None
        try:
            jf = open(self._jsonl_path, "a", encoding="utf-8", buffering=8192)
            tf = open(self._txt_path, "a", encoding="utf-8", buffering=8192)
        except OSError:
            # fallback to no-op if FS is read-only
            return
        pending = 0
        while not self._stop.is_set():
            try:
                rec = self._q.get(timeout=0.2)
            except queue.Empty:
                # periodic flush even when idle
                try:
                    if pending:
                        jf.flush()
                        tf.flush()
                        pending = 0
                except OSError:
                    pass
                continue
            try:
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec.get("status") == "final":
                    src = rec.get("source_text", "")
                    tr = rec.get("translated_text")
                    nid = rec.get("id", "")
                    if tr is not None:
                        tf.write(f"{nid}. {src} | {nid}. {tr}\n")
                    else:
                        tf.write(f"{nid}. {src}\n")
                pending += 1
                self._written += 1
                if pending >= self._flush_every:
                    jf.flush()
                    tf.flush()
                    pending = 0
            except OSError:
                pass
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass
        # drain on stop
        while True:
            try:
                rec = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec.get("status") == "final":
                    src = rec.get("source_text", "")
                    tr = rec.get("translated_text")
                    nid = rec.get("id", "")
                    tf.write(f"{nid}. {src} | {nid}. {tr}\n" if tr is not None else f"{nid}. {src}\n")
            except OSError:
                pass
        try:
            jf.flush(); tf.flush()
            jf.close(); tf.close()
        except OSError:
            pass

    def close(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    @property
    def stats(self) -> dict:
        return {"written": self._written, "dropped": self._dropped, "qsize": self._q.qsize(),
                "jsonl": str(self._jsonl_path), "txt": str(self._txt_path)}

    # test helpers (headless mirrors)
    @property
    def lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)

    @property
    def shown(self) -> list[Segment]:
        with self._lock:
            return list(self._shown)


class MultiDisplay(Display):
    """Fan-out Display: forwards show/show_partial/warn to every child.

    Used for overlay+file or console+file. Each child is called in order;
    exceptions in one child never break the others.
    """
    def __init__(self, children: list[Display]):
        self._children = list(children)

    def show(self, segment: Segment) -> None:
        for c in self._children:
            try:
                c.show(segment)
            except Exception:
                pass

    def show_partial(self, segment: Segment) -> None:
        for c in self._children:
            try:
                c.show_partial(segment)
            except Exception:
                pass

    def warn(self, text: str) -> None:
        for c in self._children:
            try:
                c.warn(text)
            except Exception:
                pass

    def close(self, timeout: float = 1.0) -> None:
        for c in self._children:
            try:
                c.close(timeout=timeout)
            except Exception:
                pass
