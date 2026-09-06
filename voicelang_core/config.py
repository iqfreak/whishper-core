"""User configuration: display, ASR engine, language, and overlay placement.

Config lives as JSON in %LOCALAPPDATA%/voicelang/config.json so the CLI and
the settings GUI share one source of truth. Missing/partial files are
tolerated (defaults fill the gaps); unknown keys are ignored so a newer GUI
never breaks an older CLI.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, field

SOURCES = ("mic", "loopback", "wav", "app")  # mic | WASAPI loopback | file replay | per-app loopback
# Single source for the 19 common languages — imported by gui.py and adapters
# Extended with Vosk offline models (ca/el/fa/cs/ro/da/sv) so the single source
# remains the authority for every ASR engine.
SUPPORTED_LANGS = ["en", "fr", "de", "es", "ar", "zh", "ja", "ko", "ru", "pt", "it", "nl", "tr", "pl", "uk", "vi", "th", "hi", "tl", "ca", "el", "fa", "cs", "ro", "da", "sv"]
# Single source for translation modes — import from here instead of re-defining
TRANSLATE_MODES = ("passthrough", "libretranslate", "deepl", "deeplx", "google")
# Streaming engines use 20ms capture blocks (P0 ultra-low-latency); whisper
# still aggregates to >=1.0s for inference (quality tier), but capture stays 20ms.
STREAMING_ENGINES = ("moonshine", "nemotron", "funasr", "funasr-nano", "vosk")
# Canonical low-latency defaults (gaming profile, 4050 RTX verified)
DEFAULT_MOONSHINE_UPDATE_INTERVAL = 0.08  # s, was 0.2 → 0.08 = 539 ms first partial
DEFAULT_WHISPER_WARMUP_S = 0.6  # s, tiny 59 ms p50 needs 0.6 s warmup not 1.0 s
# P0: 20ms capture blocks → ~10ms avg fill latency (vs 125ms @250ms, 500ms @1.0s)
# Whisper keeps 1.0s inference window via internal buffering.
DEFAULT_BLOCK_S = {"streaming": 0.02, "whisper_tiny": 1.0, "whisper": 1.0}
DEFAULT_CAPTIONS_JSONL = "captions.jsonl"
DEFAULT_CAPTIONS_TXT = "captions.txt"
DISPLAY_KINDS = ("console", "overlay", "file")


def effective_block_seconds(cfg: "Config", override: float | None = None) -> float:
    """Canonical block_seconds resolver — single source for GUI, run.py, fast.py.

    Centralises the per-engine default so the three call sites cannot drift.
    """
    if override is not None:
        return float(override)
    if cfg.block_seconds is not None:
        return float(cfg.block_seconds)
    if cfg.transcriber in STREAMING_ENGINES:
        return float(DEFAULT_BLOCK_S["streaming"])
    if cfg.transcriber == "whisper" and cfg.model_arch == "tiny":
        return float(DEFAULT_BLOCK_S["whisper_tiny"])
    return float(DEFAULT_BLOCK_S["whisper"])


def voicelang_data_dir() -> "Path":
    from pathlib import Path as _P

    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return _P(root) / "voicelang"


def config_path() -> str:
    # Single source for %LOCALAPPDATA%/voicelang — reused by file sink
    data_dir = voicelang_data_dir()
    new = str(data_dir / "config.json")
    legacy = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "whishper", "config.json")
    if not os.path.exists(new) and os.path.exists(legacy):
        try:
            os.makedirs(os.path.dirname(new), exist_ok=True)
            import shutil as _shutil
            _shutil.copy2(legacy, new)
        except OSError:
            pass
    return new


@dataclass
class OverlayConfig:
    """Overlay window placement. None fields = auto anchor (bottom-center)."""

    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None
    opacity: float = 0.85
    max_lines: int = 5
    # Independent block toggles (spec §6: some users want only one block).
    show_transcription_block: bool = True
    show_translation_block: bool = True
    # Auto-place toggle — when True the overlay is bottom-center regardless
    # of x/y/width/height (those are kept for when user toggles back to manual).
    auto_place: bool = True


@dataclass
class Config:
    display: str = "console"            # console | overlay | file
    log_captions: bool = False          # also tee finals to captions.jsonl/txt
    log_jsonl: str = ""                 # "" = %LOCALAPPDATA%/voicelang/captions.jsonl
    log_txt: str = ""                   # "" = %LOCALAPPDATA%/voicelang/captions.txt
    transcriber: str = "moonshine"      # moonshine | whisper | nemotron | funasr | funasr-nano
    language: str = "en"                # moonshine language tag (en/ar/de/es/ja/ko/tl/uk/vi/zh)
    model_arch: str = "TINY_STREAMING"  # moonshine arch name (TINY_STREAMING default)
    # low-latency tuning (persisted so GUI + CLI share one source)
    block_seconds: float | None = None  # None = auto per engine; else fixed
    moonshine_update_interval: float = DEFAULT_MOONSHINE_UPDATE_INTERVAL
    whisper_warmup_seconds: float = DEFAULT_WHISPER_WARMUP_S
    source: str = "mic"                 # mic | loopback | wav | app
    source_device: str = ""             # loopback device name; "" = default device
    source_app: str = ""                # per-app capture target: "Discord.exe" or "pid:1234" or ""
    input_file: str = ""                # required when source == "wav"
    target_language: str = "en"
    translate: str = "passthrough"      # passthrough | libretranslate | deepl | deeplx | google
    translate_url: str = ""             # endpoint for libretranslate/deepl/deeplx/google
    translate_key: str = ""             # API key for deepl (and future providers)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)

    def to_dict(self) -> dict:
        d = {
            "display": self.display,
            "log_captions": bool(self.log_captions),
            "log_jsonl": self.log_jsonl,
            "log_txt": self.log_txt,
            "transcriber": self.transcriber,
            "language": self.language,
            "model_arch": self.model_arch,
            "block_seconds": self.block_seconds,
            "moonshine_update_interval": self.moonshine_update_interval,
            "whisper_warmup_seconds": self.whisper_warmup_seconds,
            "source": self.source,
            "source_device": self.source_device,
            "source_app": self.source_app,
            "input_file": self.input_file,
            "target_language": self.target_language,
            "translate": self.translate,
            "translate_url": self.translate_url,
            "translate_key": self.translate_key,
        }
        if self.overlay is not None:
            d["overlay"] = {
                k: v for k, v in {
                    "x": self.overlay.x, "y": self.overlay.y,
                    "width": self.overlay.width, "height": self.overlay.height,
                    "opacity": self.overlay.opacity,
                    "max_lines": self.overlay.max_lines,
                    "show_transcription_block": self.overlay.show_transcription_block,
                    "show_translation_block": self.overlay.show_translation_block,
                    "auto_place": self.overlay.auto_place,
                }.items() if v is not None
            }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = d or {}
        cfg = cls(
            display=str(d.get("display", "console")),
            log_captions=bool(d.get("log_captions", False)),
            log_jsonl=str(d.get("log_jsonl", "")),
            log_txt=str(d.get("log_txt", "")),
            transcriber=str(d.get("transcriber", "moonshine")),
            language=str(d.get("language", "en")),
            model_arch=str(d.get("model_arch", "TINY_STREAMING")),
            block_seconds=(None if d.get("block_seconds") is None else float(d.get("block_seconds"))),
            moonshine_update_interval=float(d.get("moonshine_update_interval", DEFAULT_MOONSHINE_UPDATE_INTERVAL)),
            whisper_warmup_seconds=float(d.get("whisper_warmup_seconds", DEFAULT_WHISPER_WARMUP_S)),
            source=str(d.get("source", "mic")),
            source_device=str(d.get("source_device", "")),
            source_app=str(d.get("source_app", "")),
            input_file=str(d.get("input_file", "")),
            target_language=str(d.get("target_language", "en")),
            translate=str(d.get("translate", "passthrough")),
            translate_url=str(d.get("translate_url", "")),
            translate_key=str(d.get("translate_key", "")),
        )
        ov = d.get("overlay") or {}
        # Clamp overlay to visible sane defaults so a stale config never hides the window
        try:
            _op = float(ov.get("opacity", 0.85))
        except Exception:
            _op = 0.85
        _op = max(0.2, min(1.0, _op))
        try:
            _ml = int(ov.get("max_lines", 5))
        except Exception:
            _ml = 5
        _ml = max(1, min(20, _ml))
        cfg.overlay = OverlayConfig(
            x=ov.get("x"), y=ov.get("y"),
            width=ov.get("width"), height=ov.get("height"),
            opacity=_op,
            max_lines=_ml,
            show_transcription_block=bool(ov.get("show_transcription_block", True)),
            show_translation_block=bool(ov.get("show_translation_block", True)),
            auto_place=bool(ov.get("auto_place", True)),
        )
        # WP-3.3: validate transcriber against known engines; unknown -> fallback with warning
        known = {"moonshine", "whisper", "faster-whisper", "faster_whisper", "nemotron", "funasr", "funasr-nano", "funasr_nano", "vosk", "openasr", "whisper-http", "whisper_http", "passthrough", "vibeasr", "vibeasr-bitnet", "audio8", "audio8-asr"}
        if cfg.transcriber not in known:
            import sys as _sys
            print(f"[voicelang] warning: unknown transcriber '{cfg.transcriber}' — falling back to 'moonshine'", file=_sys.stderr)
            cfg.transcriber = "moonshine"
        return cfg


def load_config(path: str | None = None) -> Config:
    p = path or config_path()
    try:
        with open(p, encoding="utf-8") as f:
            return Config.from_dict(json.load(f))
    except (OSError, ValueError, TypeError) as exc:
        # WP-3.4: backup corrupt config and warn before resetting to defaults
        try:
            if os.path.exists(p):
                import shutil as _shutil, time as _time
                backup = f"{p}.corrupt.{int(_time.time())}.bak"
                _shutil.copy2(p, backup)
                print(f"[voicelang] warning: config corrupt ({exc}) — backed up to {backup} and reset to defaults", flush=True)
        except Exception:
            pass
        return Config()


def save_config(cfg: Config, path: str | None = None) -> str:
    p = path or config_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    # Windows-safe atomic write: use tempfile in same dir.
    # Colon in filename (e.g. pid:1234.json) is an ADS on NTFS — os.replace
    # to such a name fails with WinError 123, but opening the ADS directly
    # via open(p, "w") succeeds (and is what the original passing test relied
    # on). So we try atomic first, then fall back to direct write.
    import tempfile

    dirn = os.path.dirname(p) or "."
    fd = None
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=dirn, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None  # fdopen owns it
            json.dump(cfg.to_dict(), f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, p)
        tmp = None
        return p
    except OSError as exc:
        # WP-3.4 FIX: narrow fallback to only the documented NTFS-ADS colon case (WinError 123)
        # Other OSErrors (disk full, permission) should not silently corrupt via non-atomic write
        msg = str(exc).lower()
        is_ads_colon = ":" in p and ("123" in msg or "syntax" in msg or "incorrect" in msg or "ads" in msg or "cannot find" in msg or exc.errno == 22 if hasattr(exc, "errno") else False)
        # Also check WinError specifically
        try:
            is_ads_colon = is_ads_colon or getattr(exc, "winerror", None) == 123
        except Exception:
            pass
        if not is_ads_colon:
            # Not the colon/ADS case: clean up tmp and re-raise so caller sees the real error
            if tmp is not None:
                try:
                    if fd is not None:
                        os.close(fd)
                except OSError:
                    pass
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            raise
        # Colon/ADS case: fall back to direct ADS write (open(p,"w") succeeds where os.replace fails)
        if tmp is not None:
            try:
                if fd is not None:
                    os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp)
            except OSError:
                pass
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        return p
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
    return p


@functools.lru_cache(maxsize=1)
def nemotron_model_dir() -> str:
    """Locate the nemotron int8 ONNX directory regardless of CWD or frozen layout."""
    import sys as _sys
    rel = "models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [os.path.join(repo, rel), rel]
    if getattr(_sys, "frozen", False):
        candidates.insert(0, os.path.join(os.path.dirname(_sys.executable), rel))
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "nemotron model directory not found; looked for: " + "; ".join(candidates))