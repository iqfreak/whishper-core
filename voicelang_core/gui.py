"""Settings GUI: engine, language, overlay placement, and launch.

A thin PySide6 panel over the same JSON config the CLI reads
(%LOCALAPPDATA%/voicelang/config.json). Start launches the pipeline as a
subprocess with --config so the GUI never fights the capture loop for the Qt
event loop; Stop terminates it.

Language model downloads run on a worker thread with live progress; the
catalog is queried from moonshine_voice itself, so the list stays correct.
Provider selection now drives the catalog: whisper / nemotron / moonshine /
funasr each expose only their own languages + model sizes, and Download
dispatches to the right backend (moonshine_voice, faster-whisper,
modelscope, or manual instruction for nemotron).

Model-aware: dropdowns probe actual cached models at runtime (HF cache,
moonshine_voice cache, sherpa model dir, modelscope cache) and show
\u2713 installed vs \u2193 download indicators. Heavy catalog probing and
transcriber init run off the UI thread via QThread / _DownloadWorker pattern.
Popen leak fixed via atexit cleanup and repo-root cwd.
"""

from __future__ import annotations

import atexit
import functools
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from .engines import build_pipeline  # composition root (registry)
from .config import Config, OverlayConfig, config_path, load_config, save_config
from .config import DEFAULT_BLOCK_S, STREAMING_ENGINES

# Live catalog (HF Hub with TTL cache + offline fallback) — replaces hardcoded lists
try:
    from .model_catalog import (
        OFFLINE_DEFAULTS as _OFFLINE_DEFAULTS,
        VOSK_FRIENDLY_OFFLINE as _VOSK_FRIENDLY_LIVE,
        get_provider_catalog as _live_catalog,
        get_moonshine_catalog as _live_moonshine_catalog,
        is_model_cached as _live_is_cached,
    )
except Exception:  # noqa: BLE001 -- live catalog optional, fallback to statics
    _OFFLINE_DEFAULTS = {}  # type: ignore
    _VOSK_FRIENDLY_LIVE = {}  # type: ignore
    _live_catalog = None  # type: ignore
    _live_moonshine_catalog = None  # type: ignore
    _live_is_cached = None  # type: ignore


def _logo_icon():
    """QIcon for the bundled logo (source AND frozen/PyInstaller layouts).

    Returns None when the asset is missing so the GUI still opens anywhere.
    """
    try:
        from PySide6 import QtGui  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    root = str(Path(__file__).resolve().parents[1])
    if getattr(sys, "frozen", False):
        root = getattr(sys, "_MEIPASS", root)
    p = os.path.join(root, "assets", "logo.png")
    return QtGui.QIcon(p) if os.path.isfile(p) else None

@functools.lru_cache(maxsize=1)
def _funasr_missing() -> list[str]:
    return [m for m in ("modelscope", "funasr", "torch") if importlib.util.find_spec(m) is None]


@functools.lru_cache(maxsize=1)
def _funasr_nano_missing() -> list[str]:
    # Fun-ASR-Nano (HF Transformers) — torch + the PR-#46180 transformers
    # build + accelerate + librosa (see the funasr-nano extra).
    return [m for m in ("torch", "transformers", "accelerate", "librosa")
            if importlib.util.find_spec(m) is None]

ARCHES = ["TINY_STREAMING", "SMALL_STREAMING", "MEDIUM_STREAMING", "BASE_STREAMING",
          "TINY", "BASE"]

from .config import SUPPORTED_LANGS as _BASE_LANGS, TRANSLATE_MODES  # single source (was duplicated across providers)

# Provider-specific catalogs (languages + model sizes). The GUI repopulates
# Language/Model combos when the ASR engine changes — so picking a provider
# actually changes what you can download.
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
# Whisper (faster-whisper) is multilingual — full 99-language list from
# openai/whisper tokenizer; keep \"auto\" for detection. GUI shows all so
# the list is accurate (was previously only 19).
_WHISPER_ALL_LANGS = [
    "zh","en","de","es","ru","ko","fr","ja","pt","tr","pl","ca","nl","ar","sv","it","id",
    "hi","fi","vi","he","uk","el","ms","cs","ro","da","hu","ta","no","th","ur","hr","bg",
    "lt","la","mi","ml","cy","sk","te","fa","lv","bn","sr","az","sl","kn","et","mk","br",
    "eu","is","hy","ne","mn","bs","kk","sq","sw","gl","mr","pa","si","km","sn","yo","so",
    "af","oc","ka","be","tg","sd","gu","am","yi","lo","uz","fo","ht","ps","tk","nn","mt",
    "sa","lb","my","bo","tl","mg","as","tt","haw","ln","ha","ba","jw","su"
]
WHISPER_LANGS = ["auto"] + sorted(set(_WHISPER_ALL_LANGS + _BASE_LANGS))

NEMOTRON_ARCHES = ["int8"]
# Nemotron sherpa-onnx model (parakeet) is EN-only streaming — don't pretend
# it speaks 19 languages. Expose only en/auto so the list is honest.
NEMOTRON_LANGS = ["auto", "en"]

FUNASR_ARCHES = ["auto", "paraformer-zh-streaming", "SenseVoiceSmall"]
# SenseVoiceSmall supports zh/en/yue/ja/ko + auto; Paraformer streaming is
# zh/yue only. Expose the honest union (was previously 21 langs incl fr/de
# which SenseVoice cannot produce). Provider+arch decides real routing.
FUNASR_LANGS = ["auto", "zh", "zh-cn", "yue", "en", "ja", "ko"]
# Fun-ASR-Nano (HF Transformers): zh/zh-cn/en/ja; no yue/ko.
FUNASR_NANO_LANGS = ["auto", "zh", "zh-cn", "en", "ja"]
FUNASR_NANO_ARCHES = ["Nano-2512"]

# Canonical provider names — use these instead of raw strings to avoid typos
PROVIDER_MOONSHINE = "moonshine"
PROVIDER_WHISPER = "whisper"
PROVIDER_NEMOTRON = "nemotron"
PROVIDER_FUNASR = "funasr"
PROVIDER_FUNASR_NANO = "funasr-nano"
PROVIDER_VOSK = "vosk"
PROVIDER_VIBEASR = "vibeasr"
PROVIDER_VIBEASR_BITNET = "vibeasr-bitnet"
PROVIDER_AUDIO8 = "audio8"
PROVIDERS = [PROVIDER_WHISPER, PROVIDER_NEMOTRON, PROVIDER_MOONSHINE, PROVIDER_FUNASR, PROVIDER_FUNASR_NANO, PROVIDER_VOSK, PROVIDER_VIBEASR, PROVIDER_VIBEASR_BITNET, PROVIDER_AUDIO8]

# Vosk: small offline models (CPU-only, ~30-80 MB, no torch)
VOSK_ARCHES = ["small"]
# Vosk small models — 15 langs + vi fallback + extras (pt/it/nl/pl/uk)
# Keep in sync with voicelang_core/adapters/vosk.py VOSK_MODEL_CATALOG
VOSK_LANGS = ["en", "de", "es", "fr", "pt", "it", "zh", "zh-cn", "ja", "ru", "tr", "ko", "hi", "nl", "pl", "uk", "vi"]
VOSK_FRIENDLY = {
    "en": "en — English (40 MB)",
    "de": "de — German (45 MB)",
    "es": "es — Spanish (39 MB)",
    "fr": "fr — French (41 MB)",
    "pt": "pt — Portuguese (31 MB)",
    "it": "it — Italian (48 MB)",
    "zh": "zh — Chinese (42 MB)",
    "zh-cn": "zh-cn — Chinese (42 MB)",
    "ja": "ja — Japanese (48 MB)",
    "ru": "ru — Russian (45 MB)",
    "tr": "tr — Turkish (35 MB)",
    "ko": "ko — Korean (82 MB)",
    "hi": "hi — Hindi (42 MB)",
    "nl": "nl — Dutch (39 MB)",
    "pl": "pl — Polish (50 MB)",
    "uk": "uk — Ukrainian (73 MB)",
    "vi": "vi — Vietnamese (fallback en, 40 MB)",
}

# Provider catalog: maps provider -> (langs, friendly, arches, hint, dl_label)
# Now backed by live HF manifest with TTL cache + offline fallback (model_catalog.py).
# The moon_langs/moon_friendly args are kept for call-site compat but ignored when
# live fetch succeeds — we call model_catalog instead. Hand-maintained lists remain
# as the offline fallback when network is unavailable.
def _catalog_for(provider: str, moon_langs, moon_friendly):
    # Try live catalog first (never crashes UI — wrapped, falls back to statics)
    if _live_catalog is not None:
        try:
            langs, friendly, arches, hint, dl_label, used_offline = _live_catalog(provider)
            # surface offline hint in dl_label/hint so user sees "using offline defaults"
            if used_offline and provider not in (PROVIDER_VOSK,):
                # Append offline marker to hint (UI shows below dropdown)
                if hint and "offline" not in hint.lower():
                    hint = hint + " (using offline defaults)"
            # Guarantee 5-tuple contract
            return (langs, friendly, arches, hint, dl_label)
        except Exception as _e:  # noqa: BLE001
            import sys as _sys
            print(f"[voicelang] live catalog failed for {provider!r}: {_e} — using offline defaults", file=_sys.stderr)
    # Offline fallback — original hardcoded paths (identical semantics, kept for compat + tests)
    if provider == PROVIDER_MOONSHINE:
        return (moon_langs, moon_friendly, ARCHES,
                "moonshine = per-language tiny streaming models (34 MB each, cached on demand).",
                "Download moonshine model")
    if provider == PROVIDER_WHISPER:
        return (WHISPER_LANGS, {l: l for l in WHISPER_LANGS}, WHISPER_MODELS,
                "whisper = faster-whisper Systran models (tiny–large-v3-turbo), multilingual.",
                "Download Whisper model")
    if provider == PROVIDER_NEMOTRON:
        return (NEMOTRON_LANGS, {l: l for l in NEMOTRON_LANGS}, NEMOTRON_ARCHES,
                "nemotron = sherpa-onnx-nemotron-3.5 int8 (auto-download via Hugging Face; fallback GitHub).",
                "Download Nemotron model")
    if provider == PROVIDER_FUNASR:
        return (FUNASR_LANGS, {l: l for l in FUNASR_LANGS}, FUNASR_ARCHES,
                "funasr = Paraformer (zh/yue) + SenseVoiceSmall (multilingual) via modelscope.",
                "Download FunASR model")
    if provider == PROVIDER_FUNASR_NANO:
        return (FUNASR_NANO_LANGS, {l: l for l in FUNASR_NANO_LANGS}, FUNASR_NANO_ARCHES,
                "funasr-nano = Fun-ASR-Nano 0.8B (zh/en/ja, 26 Chinese accents) via HF Transformers.",
                "Download Fun-ASR-Nano model")
    if provider in (PROVIDER_VIBEASR, PROVIDER_VIBEASR_BITNET):
        from .engines import get_registry
        reg = get_registry()
        langs = reg.get(provider, {}).get("languages", ["auto", "en"])
        arches = reg.get(provider, {}).get("arches", ["default"])
        if not arches:
            arches = ["default"]
        label = provider.replace("-", " ")
        return (langs, {l: l for l in langs}, arches,
                f"{provider} = {label} transcription engine.",
                f"Download {label} model")
    if provider == PROVIDER_AUDIO8:
        from .engines import get_registry
        reg = get_registry()
        langs = reg.get("audio8", {}).get("languages", ["auto", "zh", "en"])
        arches = reg.get("audio8", {}).get("arches", ["large"])
        if not arches:
            arches = ["large"]
        return (langs, {l: l for l in langs}, arches,
                "audio8 = Audio8 transcription engine.",
                "Download Audio8 model")
    if provider == PROVIDER_VOSK:
        return (VOSK_LANGS, VOSK_FRIENDLY, VOSK_ARCHES,
                "vosk = Vosk offline small models (30-80 MB, CPU-only, no torch, streaming).",
                "Download Vosk model")
    return (moon_langs, moon_friendly, ARCHES, "", "Download model")

# Translation modes exposed in the GUI dropdown (canonical in config.TRANSLATE_MODES)
# Common target languages (editable combo, so custom codes still work)
TARGET_LANGS = list(_BASE_LANGS)
TRANSLATE_LABELS = {
    "passthrough": "Off (passthrough)",
    "libretranslate": "LibreTranslate (self-hosted)",
    "deepl": "DeepL (API key required)",
    "deeplx": "DeepLX (free DeepL proxy)",
    "google": "Google (free, no key)",
}
_TRANSLATE_PLACEHOLDERS = {
    "libretranslate": "http://127.0.0.1:5000",
    "deepl": "https://api-free.deepl.com  or  https://api.deepl.com",
    "deeplx": "http://127.0.0.1:1188",
    "google": "https://translate.googleapis.com",
    "passthrough": "endpoint URL (when translation enabled)",
}

def _moonshine_catalog():
    """Return (languages: list[str], friendly: dict[str, str]) — best effort.

    Now delegates to model_catalog live fetch (TTL cache + offline fallback).
    Keeps original moonshine_voice parsing as fallback for compat.
    """
    if _live_moonshine_catalog is not None:
        try:
            langs, friendly = _live_moonshine_catalog()
            if langs:
                return langs, friendly
        except Exception:
            pass
    try:
        import moonshine_voice as mv
        langs = list(mv.supported_languages())
        friendly: dict[str, str] = {}
        try:
            raw = mv.supported_languages_friendly()
            if isinstance(raw, dict):
                friendly = dict(raw)
            elif isinstance(raw, (list, tuple)):
                for item in raw:
                    if isinstance(item, dict) and "language" in item:
                        tag = str(item.get("language") or item.get("tag") or "")
                        name = str(item.get("name") or item.get("display") or tag)
                        if tag:
                            friendly[tag] = name
                    elif isinstance(item, str):
                        friendly[item] = item
            elif isinstance(raw, str) and raw.strip():
                # current mv returns "ar (Arabic), es (Spanish), ..." as a single string
                for part in raw.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "(" in part and part.endswith(")"):
                        tag, name = part.split("(", 1)
                        tag = tag.strip()
                        name = name.rstrip(")").strip()
                        if tag:
                            friendly[tag] = name or tag
                    else:
                        friendly[part] = part
        except Exception:  # noqa: BLE001 -- shape varies across versions
            friendly = {l: l for l in langs}
        if not friendly:
            friendly = {l: l for l in langs}
        return langs, friendly
    except ImportError:
        return ["en"], {"en": "English"}
def _repo_root() -> Path:
    """Repo root for subprocess cwd — Path(__file__).parent is voicelang_core."""
    return Path(__file__).resolve().parents[1]


def _effective_block_seconds(cfg: Config) -> float:
    """Single source for capture block_seconds — mirrors run.py/fast.py.

    Fixes divergence: run vs fast previously computed block_seconds
    independently. Now both delegate to config.DEFAULT_BLOCK_S +
    STREAMING_ENGINES; GUI exposes the same helper so the displayed value
    never drifts.
    """
    from voicelang_core.config import effective_block_seconds as _canonical
    return _canonical(cfg)


# ---- installed-model probing (runtime, best-effort, never crashes UI) ----

def _hf_cache_root() -> Path | None:
    for env in ("HF_HUB_CACHE", "HF_HOME"):
        v = os.environ.get(env)
        if v:
            p = Path(v)
            cand = p / "hub" if (p / "hub").is_dir() else p
            if cand.is_dir():
                return cand
            return p
    home = Path.home() / ".cache" / "huggingface" / "hub"
    return home if home.is_dir() else None


@functools.lru_cache(maxsize=32)
def _is_hf_repo_cached(repo_id: str) -> bool:
    """True if huggingface_hub has repo_id cached (filesystem check, no network)."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:
        pass
    try:
        root = _hf_cache_root()
        if root is None:
            return False
        needle = "models--" + repo_id.replace("/", "--")
        for child in root.iterdir():
            if child.name == needle or child.name.startswith(needle):
                if (child / "snapshots").is_dir():
                    for snap in (child / "snapshots").iterdir():
                        if snap.is_dir() and any(snap.iterdir()):
                            return True
                return child.is_dir()
        return False
    except Exception:
        return False


@functools.lru_cache(maxsize=32)
def _whisper_model_cached(model_size: str) -> bool:
    return _is_hf_repo_cached(f"Systran/faster-whisper-{model_size}")


def _whisper_cached_models() -> set[str]:
    return {m for m in WHISPER_MODELS if _whisper_model_cached(m)}


@functools.lru_cache(maxsize=64)
def _moonshine_model_cached(language: str, arch: str) -> bool:
    """Check if moonshine model for (language, arch) is already on disk — no network, no download."""
    try:
        # Primary: use moonshine_voice's own cache dir (platformdirs)
        try:
            from moonshine_voice.download_file import get_cache_dir as _mv_cache  # type: ignore
            mv_root = _mv_cache()
            # mv_root is .../moonshine_voice/Cache — model dir is .../download.moonshine.ai/model
            # Search for <arch>-<lang> pattern (tiny-streaming-en, base-ar, etc.)
            arch_norm = arch.lower().replace("_", "-")
            lang_norm = language.lower()
            # Direct probe: look for directory named like f"{arch_norm}-{lang_norm}"
            for base in [mv_root, mv_root / "download.moonshine.ai" / "model"]:
                if base.is_dir():
                    # Exact match
                    if (base / f"{arch_norm}-{lang_norm}").is_dir():
                        return True
                    if (base / f"{arch_norm.replace('-streaming','')}-{lang_norm}").is_dir():
                        return True
                    # Shallow scan
                    try:
                        for child in base.iterdir():
                            n = child.name.lower()
                            if lang_norm in n and arch_norm in n:
                                return True
                            if child.is_dir():
                                for sub in child.iterdir():
                                    if lang_norm in sub.name.lower() and arch_norm in sub.name.lower():
                                        return True
                    except Exception:
                        pass
                    # Deeper model dir
                    deep = mv_root / "download.moonshine.ai" / "model"
                    if deep.is_dir():
                        try:
                            for child in deep.iterdir():
                                n = child.name.lower()
                                if lang_norm in n and arch_norm in n:
                                    return True
                        except Exception:
                            pass
        except Exception:
            pass
        # Fallback: legacy candidates
        candidates: list[Path] = []
        for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("APPDATA"), str(Path.home())):
            if not base:
                continue
            for sub in ("moonshine_voice", ".cache/moonshine_voice", "moonshine-voice"):
                candidates.append(Path(base) / sub)
        hf_root = _hf_cache_root()
        if hf_root is not None:
            try:
                for child in hf_root.iterdir():
                    if "moonshine" in child.name.lower() and language.lower() in child.name.lower():
                        return True
            except Exception:
                pass
        for cand in candidates:
            try:
                if not cand.is_dir():
                    continue
                for lvl1 in cand.rglob(f"*{language.lower()}*"):
                    if arch.lower().replace("_", "-") in lvl1.name.lower() or lvl1.suffix == ".onnx":
                        return True
                    if lvl1.is_dir() and language.lower() in lvl1.name.lower():
                        return True
                    # limit scans: break early if too many
                    break
            except Exception:
                continue
        return False
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _nemotron_installed() -> bool:
    try:
        from .config import nemotron_model_dir
        d = nemotron_model_dir()
        return (Path(d) / "encoder.int8.onnx").is_file() or Path(d).is_dir()
    except Exception:
        try:
            repo = _repo_root() / "models" / "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
            return (repo / "encoder.int8.onnx").is_file()
        except Exception:
            return False


@functools.lru_cache(maxsize=32)
def _funasr_model_cached(model_id: str) -> bool:
    try:
        resolved = model_id
        try:
            from funasr.download.name_maps_from_hub import name_maps_ms  # type: ignore
            resolved = name_maps_ms.get(model_id, model_id)
        except Exception:
            pass
        ms_root = Path.home() / ".cache" / "modelscope" / "hub"
        if ms_root.is_dir():
            flat = resolved.replace("/", "--")
            for child in ms_root.rglob("*"):
                if flat.lower() in child.name.lower() or resolved.split("/")[-1].lower() in child.name.lower():
                    if child.is_dir() and any(child.iterdir()):
                        return True
        if _is_hf_repo_cached(resolved) or _is_hf_repo_cached(model_id):
            return True
        if model_id == "Nano-2512" or "fun-asr-nano" in model_id.lower():
            if _is_hf_repo_cached("FunAudioLLM/Fun-ASR-Nano-2512-hf"):
                return True
        return False
    except Exception:
        return False


def _model_label(base: str, cached: bool) -> str:
    return f"{base} ✓ installed" if cached else f"{base} ↓ download"


_ACTIVE_PROCS: list[subprocess.Popen] = []


def _cleanup_procs() -> None:
    for p in list(_ACTIVE_PROCS):
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=1.5)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        except Exception:
            pass
    _ACTIVE_PROCS.clear()


atexit.register(_cleanup_procs)


class _BaseDownloadWorker:
    """Shared callback plumbing for background model downloads."""
    def __init__(self, on_progress, on_done):
        self._on_progress = on_progress
        self._on_done = on_done

    def _execute(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:  # noqa: BLE001
            self._on_done(f"download failed: {exc}")


class _DownloadWorker(_BaseDownloadWorker):
    """Moonshine per-language tiny streaming model."""
    def __init__(self, language: str, arch: str, on_progress, on_done):
        super().__init__(on_progress, on_done)
        self._language = language
        self._arch = arch

    def _execute(self) -> None:
        import moonshine_voice as mv
        path, arch = mv.get_model_for_language(
            self._language,
            getattr(mv.ModelArch, self._arch, mv.ModelArch.TINY_STREAMING),
            on_progress=self._on_progress,
        )
        self._on_done(f"ready: {arch.name} @ {path}")


class _WhisperDownloadWorker(_BaseDownloadWorker):
    def __init__(self, model_size: str, on_progress, on_done):
        super().__init__(on_progress, on_done)
        self._model_size = model_size

    def _execute(self) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self._on_progress(0.1, f"downloading faster-whisper/{self._model_size} …")
            model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
            del model
            self._on_done(f"ready: faster-whisper/{self._model_size} (cached)")
        except Exception as e:
            try:
                from huggingface_hub import snapshot_download  # type: ignore
                repo = f"Systran/faster-whisper-{self._model_size}"
                self._on_progress(0.2, f"downloading {repo} …")
                path = snapshot_download(repo)
                self._on_done(f"ready: {repo} @ {path}")
            except Exception as e2:
                self._on_done(f"download failed: {e} / {e2}")


class _NemotronDownloadWorker(_BaseDownloadWorker):
    """Sherpa-onnx Nemotron int8 model — auto-download via Hugging Face (fallback GitHub tarball)."""
    def __init__(self, on_progress, on_done):
        super().__init__(on_progress, on_done)

    def _execute(self) -> None:
        import os
        import sys
        import shutil
        import tarfile
        import tempfile
        rel = "models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
        if getattr(sys, "frozen", False):
            dest = os.path.join(os.path.dirname(sys.executable), rel)
        else:
            repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dest = os.path.join(repo, rel)
        required = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        if os.path.isdir(dest) and all(os.path.isfile(os.path.join(dest, f)) for f in required):
            self._on_done(f"ready: nemotron @ {dest} (already cached)")
            return
        self._on_progress(0.05, "downloading nemotron — trying Hugging Face …")
        last_err = None
        try:
            from huggingface_hub import snapshot_download  # type: ignore
            repo_id = "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
            self._on_progress(0.10, f"downloading {repo_id} via Hugging Face …")
            hf_dir = snapshot_download(repo_id)
            os.makedirs(dest, exist_ok=True)
            for fname in os.listdir(hf_dir):
                src = os.path.join(hf_dir, fname)
                dst = os.path.join(dest, fname)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
            if not all(os.path.isfile(os.path.join(dest, f)) for f in required):
                for root, _dirs, files in os.walk(hf_dir):
                    for f in required:
                        if f in files:
                            src = os.path.join(root, f)
                            dst = os.path.join(dest, f)
                            if not os.path.exists(dst):
                                shutil.copy2(src, dst)
            if all(os.path.isfile(os.path.join(dest, f)) for f in required):
                self._on_done(f"ready: nemotron @ {dest} (from Hugging Face {repo_id})")
                return
            last_err = f"HF download incomplete — missing {required}"
        except Exception as e:  # noqa: BLE001
            last_err = e
            self._on_progress(0.25, f"Hugging Face failed ({e}), trying GitHub …")
        try:
            import requests  # type: ignore
            url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2"
            self._on_progress(0.30, "downloading nemotron tarball from GitHub …")
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tar.bz2")
            os.close(tmp_fd)
            try:
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0) or 0)
                    downloaded = 0
                    with open(tmp_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                frac = 0.30 + 0.55 * (downloaded / total)
                                self._on_progress(min(0.88, frac), f"downloading nemotron {downloaded // 1048576}MB / {total // 1048576}MB …")
                            elif downloaded % (10 * 1048576) < 8192:
                                self._on_progress(0.50, f"downloading nemotron {downloaded // 1048576}MB …")
                self._on_progress(0.90, "extracting nemotron …")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                parent = os.path.dirname(dest)
                with tarfile.open(tmp_path, "r:bz2") as tf:
                    tf.extractall(parent)
                if not all(os.path.isfile(os.path.join(dest, f)) for f in required):
                    raise RuntimeError(f"extracted but missing files in {dest}; GH err was: {last_err}")
                self._on_done(f"ready: nemotron @ {dest} (from GitHub)")
                return
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as e2:  # noqa: BLE001
            raise RuntimeError(f"nemotron download failed: HF={last_err} / GH={e2}") from e2


class _FunASRDownloadWorker(_BaseDownloadWorker):
    def __init__(self, language: str, arch: str, on_progress, on_done):
        super().__init__(on_progress, on_done)
        self._language = language
        self._arch = arch

    def _execute(self) -> None:
        model_id = self._arch if self._arch != "auto" else (
            "paraformer-zh-streaming" if self._language.lower() in ("zh", "zh-cn", "yue") else "iic/SenseVoiceSmall"
        )
        if model_id == "SenseVoiceSmall":
            model_id = "iic/SenseVoiceSmall"
        self._on_progress(0.1, f"downloading {model_id} …")
        try:
            from funasr import AutoModel  # type: ignore
            self._on_progress(0.3, f"resolving {model_id} via FunASR …")
            model = AutoModel(model=model_id, device="cpu", disable_update=True)
            try:
                path = getattr(model, "model_path", str(getattr(model, "kwargs", {}).get("model_path", model_id)))
            except Exception:
                path = model_id
            try:
                del model
            except Exception:
                pass
            self._on_done(f"ready: {model_id} @ {path}")
            return
        except Exception as e:
            pass
        try:
            from modelscope import snapshot_download as ms_download  # type: ignore
            self._on_progress(0.5, f"downloading {model_id} via ModelScope …")
            path = ms_download(model_id)
            self._on_done(f"ready: {model_id} @ {path}")
            return
        except Exception as e2:
            self._on_done(f"download failed: {e} / {e2}")


class _FunASRNanoDownloadWorker(_BaseDownloadWorker):
    def __init__(self, on_progress, on_done):
        super().__init__(on_progress, on_done)

    def _execute(self) -> None:
        self._on_progress(0.1, "downloading Fun-ASR-Nano-2512 …")
        try:
            from huggingface_hub import snapshot_download  # type: ignore
            repo = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
            path = snapshot_download(repo)
            self._on_done(f"ready: Fun-ASR-Nano-2512 @ {path}")
            return
        except Exception as e:
            self._on_done(f"download failed: {e}")


class _VoskDownloadWorker(_BaseDownloadWorker):
    """Vosk small model downloader (CPU-only, ~30-80 MB)."""
    def __init__(self, language: str, on_progress, on_done):
        super().__init__(on_progress, on_done)
        self._language = language

    def _execute(self) -> None:
        try:
            from voicelang_core.adapters.vosk import download_vosk_model, get_vosk_model_info
            info = get_vosk_model_info(self._language)
            self._on_progress(0.1, f"downloading vosk {self._language} ({info['model_name']}) …")
            path = download_vosk_model(self._language, progress=False)
            self._on_done(f"ready: vosk {self._language} @ {path}")
        except Exception as e:
            self._on_done(f"download failed: {e}")


def _make_gui():
    """Create SettingsWindow factory — dark-first redesign per §7.

    Logic preserved verbatim from original: _cfg_from_widgets mapping,
    provider catalog, download dispatch, subprocess contract (cwd, --config,
    log file, QTimer polling 800ms, exit-code mapping). Only layout/style
    changed: left rail + stacked cards + shadows + status pill + meter +
    caption preview + toasts + download dialog + skeleton + DPI + keyboard nav.
    """
    from PySide6 import QtWidgets, QtCore, QtGui
    from pathlib import Path
    import os, sys, subprocess, atexit, math, time, random

    # --- icon helper (qtawesome if available, else Unicode fallback) ---
    def _icon(name: str, fallback: str = "•"):
        try:
            import qtawesome as qta  # type: ignore
            mapping = {
                "engine": "fa5s.microphone",
                "capture": "fa5s.volume-up",
                "overlay": "fa5s.desktop",
                "translate": "fa5s.language",
                "display": "fa5s.eye",
                "settings": "fa5s.cog",
                "download": "fa5s.download",
                "play": "fa5s.play",
                "stop": "fa5s.stop",
                "save": "fa5s.save",
                "check": "fa5s.check-circle",
                "warn": "fa5s.exclamation-triangle",
                "error": "fa5s.times-circle",
            }
            key = mapping.get(name, "fa5s.circle")
            return qta.icon(key, color="#9AA3B2")
        except Exception:
            return None

    def _uni(name: str) -> str:
        return {
            "engine": "◉",
            "capture": "◐",
            "overlay": "▭",
            "translate": "⇄",
            "display": "◎",
            "settings": "⚙",
            "download": "⬇",
            "play": "▶",
            "stop": "■",
            "save": "💾",
            "check": "✓",
            "warn": "⚠",
            "error": "✕",
        }.get(name, "•")

    # --- shadow helper (spec: blur24 y2 alpha60) ---
    def _apply_shadow(widget, blur=24, y=2, alpha=60):
        try:
            from PySide6.QtWidgets import QGraphicsDropShadowEffect
            eff = QGraphicsDropShadowEffect(widget)
            eff.setBlurRadius(blur)
            eff.setOffset(0, y)
            eff.setColor(QtGui.QColor(0, 0, 0, alpha))
            widget.setGraphicsEffect(eff)
            return eff
        except Exception:
            return None

    def _card(title: str | None = None):
        outer = QtWidgets.QFrame()
        outer.setObjectName("card")
        _apply_shadow(outer, blur=24, y=2, alpha=60)
        lay = QtWidgets.QVBoxLayout(outer)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        if title:
            lab = QtWidgets.QLabel(title)
            lab.setObjectName("cardTitle")
            lay.addWidget(lab)
        return outer, lay

    # ===== custom widgets =====
    class AudioLevelMeter(QtWidgets.QFrame):
        """Live audio level meter — QPainter bars, pcm_energy, smooth decay."""
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("meterBg")
            self.setFixedHeight(28)
            self._level = 0.0
            self._peak = 0.0
            self._decay = 0.88  # per 32ms tick
            self._target = 0.0
            self._timer = QtCore.QTimer(self)
            self._timer.setInterval(32)
            self._timer.timeout.connect(self._tick)
            self._timer.start()
            self.setToolTip("Live mic/loopback level")

        def setLevel(self, lvl: float):
            lvl = max(0.0, min(1.0, float(lvl)))
            self._target = lvl
            if lvl > self._peak:
                self._peak = lvl
            self.update()

        def setEnergy(self, pcm: bytes, sample_rate: int = 16000):
            # pcm_energy: RMS normalized 0..1
            try:
                import struct
                import math as _m
                n = len(pcm) // 2
                if n == 0:
                    return
                fmt = f"<{n}h"
                samps = struct.unpack(fmt, pcm[: n * 2])
                rms = _m.sqrt(sum(s * s for s in samps) / n) / 32768.0
                # boost quiet signals for visibility
                lvl = min(1.0, rms * 3.0)
                self.setLevel(lvl)
            except Exception:
                pass

        def _tick(self):
            # smooth decay toward target then toward 0
            if self._level < self._target:
                self._level = self._level * 0.3 + self._target * 0.7
            else:
                self._level *= self._decay
                self._target *= 0.92
            self._peak *= 0.985
            if self._level < 0.01 and self._target < 0.01:
                self._level = max(0.0, self._level - 0.015)
            if abs(self._level - self._target) > 0.001 or self._level > 0.01:
                self.update()

        def paintEvent(self, e):
            super().paintEvent(e)
            try:
                p = QtGui.QPainter(self)
                p.setRenderHint(QtGui.QPainter.Antialiasing, True)
                r = self.rect().adjusted(6, 8, -6, -8)
                # bg track
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor("#232936"))
                p.drawRoundedRect(r, 4, 4)
                # level fill
                if self._level > 0.005:
                    w = int(r.width() * self._level)
                    fill = r.adjusted(0, 0, -r.width() + w, 0)
                    # gradient green -> yellow -> red
                    grad = QtGui.QLinearGradient(fill.left(), 0, fill.right(), 0)
                    grad.setColorAt(0.0, QtGui.QColor("#34D399"))
                    grad.setColorAt(0.65, QtGui.QColor("#34D399"))
                    grad.setColorAt(0.85, QtGui.QColor("#FBBF24"))
                    grad.setColorAt(1.0, QtGui.QColor("#F87171"))
                    p.setBrush(grad)
                    p.drawRoundedRect(fill, 4, 4)
                # peak tick
                if self._peak > 0.02:
                    x = r.left() + int(r.width() * self._peak)
                    p.setBrush(QtGui.QColor("#E6E9F0"))
                    p.drawRoundedRect(QtCore.QRect(x - 1, r.top(), 2, r.height()), 1, 1)
                p.end()
            except Exception:
                pass

    class PulsingDot(QtWidgets.QLabel):
        def __init__(self, parent=None):
            super().__init__("●", parent)
            self.setFixedSize(14, 14)
            self.setAlignment(QtCore.Qt.AlignCenter)
            self._state = "idle"
            self._on = True
            self._pulse_timer = QtCore.QTimer(self)
            self._pulse_timer.setInterval(700)
            self._pulse_timer.timeout.connect(self._pulse)
            self.setState("idle")

        def setState(self, state: str):
            self._state = state
            colors = {"idle": "#6B7280", "running": "#34D399", "degraded": "#FBBF24", "error": "#F87171"}
            self.setStyleSheet(f"color: {colors.get(state, '#6B7280')}; font-size: 10px;")
            if state in ("running", "degraded"):
                if not self._pulse_timer.isActive():
                    self._pulse_timer.start()
            else:
                self._pulse_timer.stop()
                self.setGraphicsEffect(None)
                self.setStyleSheet(f"color: {colors.get(state, '#6B7280')}; font-size: 10px; opacity: 1;")

        def _pulse(self):
            # simple opacity toggle via stylesheet
            self._on = not self._on
            op = "1" if self._on else "0.35"
            base = {"idle": "#6B7280", "running": "#34D399", "degraded": "#FBBF24", "error": "#F87171"}.get(self._state, "#6B7280")
            # use graphics opacity if available
            try:
                eff = self.graphicsEffect()
                if eff is None:
                    from PySide6.QtWidgets import QGraphicsOpacityEffect
                    eff = QGraphicsOpacityEffect(self)
                    self.setGraphicsEffect(eff)
                eff.setOpacity(1.0 if self._on else 0.38)
            except Exception:
                pass

    class CaptionPreviewCard(QtWidgets.QFrame):
        """Caption preview keyed by segment id, dimmed source + emphasized translation, fade/slide 150-250ms."""
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("captionCard")
            _apply_shadow(self, 24, 2, 60)
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(14, 12, 14, 12)
            lay.setSpacing(6)
            head = QtWidgets.QHBoxLayout()
            head.setSpacing(8)
            self._idBadge = QtWidgets.QLabel("#—")
            self._idBadge.setObjectName("muted")
            self._idBadge.setStyleSheet("background: #232936; border-radius: 6px; padding: 2px 6px; font-size: 11px; color: #9AA3B2;")
            self._statusDot = QtWidgets.QLabel("○")
            self._statusDot.setObjectName("muted")
            head.addWidget(self._idBadge)
            head.addStretch(1)
            head.addWidget(self._statusDot)
            lay.addLayout(head)
            self._sourceLab = QtWidgets.QLabel("No captions yet — press Start to stream.")
            self._sourceLab.setObjectName("captionSource")
            self._sourceLab.setWordWrap(True)
            self._sourceLab.setStyleSheet("color: rgba(230,233,240,0.70);")
            self._transLab = QtWidgets.QLabel("")
            self._transLab.setObjectName("captionTrans")
            self._transLab.setWordWrap(True)
            self._transLab.setVisible(False)
            lay.addWidget(self._sourceLab)
            lay.addWidget(self._transLab)
            self._lastId = None
            # animation
            self._animOpacity = None
            self._animPos = None

        def showSegment(self, seg_id: int, source_text: str, translated_text: str | None, status: str = "final"):
            # keyed by segment id — no duplicate render
            if seg_id is not None and seg_id == self._lastId and status == "final" and not translated_text:
                return
            self._lastId = seg_id
            self._idBadge.setText(f"#{seg_id}" if seg_id is not None else "#—")
            self._statusDot.setText("●" if status == "final" else "◐")
            self._statusDot.setStyleSheet("color: #34D399; font-size: 11px;" if status == "final" else "color: #FBBF24; font-size: 11px;")
            self._sourceLab.setText(source_text or "—")
            self._sourceLab.setStyleSheet("color: rgba(230,233,240,0.70);")
            if translated_text and translated_text.strip():
                self._transLab.setText(translated_text)
                self._transLab.setVisible(True)
            else:
                self._transLab.setVisible(False)
            self._animateIn()

        def _animateIn(self):
            try:
                from PySide6.QtWidgets import QGraphicsOpacityEffect
                eff = self.graphicsEffect()
                if not isinstance(eff, QGraphicsOpacityEffect):
                    eff = QGraphicsOpacityEffect(self)
                    self.setGraphicsEffect(eff)
                eff.setOpacity(0.0)
                anim = QtCore.QPropertyAnimation(eff, b"opacity", self)
                anim.setDuration(220)
                anim.setStartValue(0.0)
                anim.setEndValue(1.0)
                anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
                anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
                self._animOpacity = anim
                # slide: 6px lift
                start = self.pos()
                # small vertical nudge animation via pos
                anim2 = QtCore.QPropertyAnimation(self, b"pos", self)
                anim2.setDuration(220)
                anim2.setStartValue(start + QtCore.QPoint(0, 6))
                anim2.setEndValue(start)
                anim2.setEasingCurve(QtCore.QEasingCurve.OutCubic)
                anim2.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
                self._animPos = anim2
            except Exception:
                pass

        def clear(self):
            self._lastId = None
            self._idBadge.setText("#—")
            self._sourceLab.setText("No captions yet — press Start to stream.")
            self._transLab.setVisible(False)
            self._statusDot.setText("○")

    class SkeletonShimmer(QtWidgets.QFrame):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("card")
            _apply_shadow(self, 24, 2, 60)
            self.setFixedHeight(86)
            self._phase = 0.0
            self._tm = QtCore.QTimer(self)
            self._tm.setInterval(28)
            self._tm.timeout.connect(self._tick)
            self._tm.start()
        def _tick(self):
            self._phase = (self._phase + 0.04) % 1.0
            self.update()
        def paintEvent(self, e):
            super().paintEvent(e)
            try:
                p = QtGui.QPainter(self)
                p.setRenderHint(QtGui.QPainter.Antialiasing, True)
                # shimmer bars
                base = QtGui.QColor("#1A1F2E")
                hi = QtGui.QColor("#232936")
                for i, (y, w) in enumerate([(18, 0.62), (42, 0.88), (62, 0.44)]):
                    rect = QtCore.QRect(16, y, int(self.width() * w), 10)
                    p.setBrush(base)
                    p.setPen(QtCore.Qt.NoPen)
                    p.drawRoundedRect(rect, 5, 5)
                    # moving highlight
                    offset = int((self._phase * self.width() * 1.2) - self.width() * 0.3)
                    grad = QtGui.QLinearGradient(offset, 0, offset + 120, 0)
                    grad.setColorAt(0, QtGui.QColor("#00000000"))
                    grad.setColorAt(0.5, QtGui.QColor(255, 255, 255, 14))
                    grad.setColorAt(1, QtGui.QColor("#00000000"))
                    p.setBrush(grad)
                    p.drawRoundedRect(rect, 5, 5)
                p.end()
            except Exception:
                pass

    class ToastHost(QtWidgets.QFrame):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("toastHost")
            self.setStyleSheet("QFrame#toastHost { background: transparent; border: none; }")
            self._layout = QtWidgets.QVBoxLayout(self)
            self._layout.setContentsMargins(0, 0, 0, 0)
            self._layout.setSpacing(8)
            self._layout.setAlignment(QtCore.Qt.AlignTop)
        def toast(self, msg: str, level: str = "info", duration: int = 3200):
            card = QtWidgets.QFrame()
            card.setObjectName({"info":"toast","success":"toastSuccess","error":"toastError","warning":"toast"}.get(level, "toast"))
            _apply_shadow(card, 24, 4, 80)
            card.setFixedWidth(360)
            lay = QtWidgets.QHBoxLayout(card)
            lay.setContentsMargins(12, 10, 12, 10)
            lay.setSpacing(10)
            icon = QtWidgets.QLabel({"info":"ℹ","success":"✓","error":"✕","warning":"⚠"}.get(level, "ℹ"))
            icon.setStyleSheet(f"color: {'#34D399' if level=='success' else '#F87171' if level=='error' else '#FBBF24' if level=='warning' else '#9AA3B2'}; font-weight: 700;")
            lab = QtWidgets.QLabel(msg)
            lab.setWordWrap(True)
            lab.setStyleSheet("color: #E6E9F0; font-size: 12px;")
            lay.addWidget(icon)
            lay.addWidget(lab, 1)
            self._layout.addWidget(card)
            # slide+fade in
            try:
                from PySide6.QtWidgets import QGraphicsOpacityEffect
                eff = QGraphicsOpacityEffect(card)
                card.setGraphicsEffect(eff)
                eff.setOpacity(0.0)
                anim = QtCore.QPropertyAnimation(eff, b"opacity", card)
                anim.setDuration(220)
                anim.setStartValue(0.0)
                anim.setEndValue(1.0)
                anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
                anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
            except Exception:
                pass
            QtCore.QTimer.singleShot(duration, lambda: self._dismiss(card))

        def _dismiss(self, card):
            try:
                card.setParent(None)
                card.deleteLater()
            except Exception:
                pass

    class DownloadDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("glass")
            self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.FramelessWindowHint)
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self.setModal(False)
            self.resize(420, 140)
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(18, 18, 18, 18)
            lay.setSpacing(12)
            self._title = QtWidgets.QLabel("Downloading model…")
            self._title.setObjectName("cardTitle")
            self._msg = QtWidgets.QLabel("")
            self._msg.setObjectName("hint")
            self._msg.setWordWrap(True)
            self._bar = QtWidgets.QProgressBar()
            self._bar.setRange(0, 1000)
            self._bar.setValue(0)
            self._bar.setTextVisible(True)
            self._bar.setFormat("%p%")
            lay.addWidget(self._title)
            lay.addWidget(self._msg)
            lay.addWidget(self._bar)
            btn = QtWidgets.QPushButton("Hide")
            btn.setObjectName("ghost")
            btn.clicked.connect(self.hide)
            lay.addWidget(btn, alignment=QtCore.Qt.AlignRight)
            _apply_shadow(self, 32, 8, 100)
        def setProgress(self, frac: float, msg: str):
            try:
                self._bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
                self._msg.setText(msg)
                if not self.isVisible():
                    self.show()
                self.raise_()
            except Exception:
                pass
        def done(self, msg: str):
            try:
                self._bar.setValue(1000)
                self._msg.setText(msg)
                QtCore.QTimer.singleShot(1800, self.hide)
            except Exception:
                pass

    class _Signals(QtCore.QObject):
        progress = QtCore.Signal(float, str)
        done = QtCore.Signal(str)
        catalog_ready = QtCore.Signal(list, dict)

    def _repo_root():
        return str(Path(__file__).resolve().parents[1])

    # ================================================================
    # SettingsWindow — visual redesign
    # ================================================================
    class SettingsWindow(QtWidgets.QWidget):
        def __init__(self, cfg: Config):
            super().__init__()
            self._cfg = cfg
            self._proc = None
            self._start_ts: float | None = None
            self._elapsed_timer = None
            self.setWindowTitle("VoiceLang — settings")
            self.resize(1240, 780)
            self.setMinimumSize(1020, 640)
            # DPI handled in main() before QApplication; per-widget WA_HighDpiScaling not needed
            # root layout: rail + content
            root = QtWidgets.QHBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            # ----- left rail -----
            rail = QtWidgets.QFrame()
            rail.setObjectName("rail")
            rail.setMinimumWidth(220)
            rail.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
            railLay = QtWidgets.QVBoxLayout(rail)
            railLay.setContentsMargins(8, 10, 8, 10)
            railLay.setSpacing(6)
            # logo
            logoBox = QtWidgets.QVBoxLayout()
            logoBox.setSpacing(2)
            logoBox.setAlignment(QtCore.Qt.AlignHCenter)
            try:
                ic = _logo_icon()
                logoLab = QtWidgets.QLabel()
                if ic is not None and not ic.isNull():
                    pm = ic.pixmap(40, 40)
                    if not pm.isNull():
                        logoLab.setPixmap(pm.scaled(38, 38, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                else:
                    logoLab.setText("◉")
                    logoLab.setStyleSheet("font-size: 28px; color: #6C8CFF;")
                logoLab.setAlignment(QtCore.Qt.AlignCenter)
                logoBox.addWidget(logoLab)
            except Exception:
                pass
            titleLab = QtWidgets.QLabel("VoiceLang")
            titleLab.setObjectName("logoTitle")
            titleLab.setAlignment(QtCore.Qt.AlignCenter)
            subLab = QtWidgets.QLabel("v0.1")
            subLab.setObjectName("logoSub")
            subLab.setAlignment(QtCore.Qt.AlignCenter)
            logoBox.addWidget(titleLab)
            logoBox.addWidget(subLab)
            railLay.addLayout(logoBox)
            railLay.addSpacing(10)
            # nav buttons (checkable) + animated pill
            self._nav_group = QtWidgets.QButtonGroup(self)
            self._nav_group.setExclusive(True)
            self._nav_buttons: list[QtWidgets.QPushButton] = []
            self._nav_pages = ["Engine", "Capture", "Overlay", "Translate"]
            self._nav_keys = ["engine", "capture", "overlay", "translate"]
            # pill indicator (animated)
            self._nav_pill = QtWidgets.QFrame(rail)
            self._nav_pill.setObjectName("navPill")
            self._nav_pill.setStyleSheet("QFrame#navPill { background: rgba(108,140,255,0.14); border: 1px solid rgba(108,140,255,0.22); border-radius: 10px; }")
            self._nav_pill.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            self._nav_pill.hide()
            self._pill_anim = None
            for idx, (label, key) in enumerate(zip(self._nav_pages, self._nav_keys)):
                b = QtWidgets.QPushButton(f"  {_uni(key)}  {label}")
                b.setObjectName("nav")
                b.setCheckable(True)
                b.setCursor(QtCore.Qt.PointingHandCursor)
                b.setFocusPolicy(QtCore.Qt.StrongFocus)
                b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                b.setToolTip(label)
                # elide long text if scaled DPI exceeds rail width
                try:
                    fm = b.fontMetrics()
                    # allow button to report correct sizeHint but ensure tooltip for overflow
                    b.setMinimumWidth(fm.horizontalAdvance(b.text()) + 28)
                except Exception:
                    pass
                if idx == 0:
                    b.setChecked(True)
                self._nav_group.addButton(b, idx)
                railLay.addWidget(b)
                self._nav_buttons.append(b)
            railLay.addStretch(1)
            # bottom actions in rail
            self._rail_save = QtWidgets.QPushButton(f" {_uni('save')} Save")
            self._rail_save.setObjectName("ghost")
            self._rail_save.setCursor(QtCore.Qt.PointingHandCursor)
            railLay.addWidget(self._rail_save)
            root.addWidget(rail)
            self._rail = rail

            # ----- right content -----
            content = QtWidgets.QWidget()
            content.setObjectName("central")
            cLay = QtWidgets.QVBoxLayout(content)
            cLay.setContentsMargins(18, 14, 18, 14)
            cLay.setSpacing(12)

            # top status bar: pill + elapsed + meter + counters
            topBar = QtWidgets.QHBoxLayout()
            topBar.setSpacing(12)
            # status pill
            self._statusPill = QtWidgets.QFrame()
            self._statusPill.setObjectName("statusPill")
            _apply_shadow(self._statusPill, 18, 2, 40)
            pillLay = QtWidgets.QHBoxLayout(self._statusPill)
            pillLay.setContentsMargins(10, 6, 10, 6)
            pillLay.setSpacing(8)
            self._dot = PulsingDot()
            self._statusPillLabel = QtWidgets.QLabel("Idle")
            self._statusPillLabel.setObjectName("statusPillLabel")
            self._elapsedLab = QtWidgets.QLabel("00:00")
            self._elapsedLab.setObjectName("elapsed")
            self._countersLab = QtWidgets.QLabel("")
            self._countersLab.setObjectName("muted")
            pillLay.addWidget(self._dot)
            pillLay.addWidget(self._statusPillLabel)
            pillLay.addWidget(self._elapsedLab)
            topBar.addWidget(self._statusPill)
            topBar.addWidget(self._countersLab)
            topBar.addStretch(1)
            # level meter
            self._meter = AudioLevelMeter()
            self._meter.setFixedWidth(160)
            topBar.addWidget(self._meter)
            # window controls hint
            cLay.addLayout(topBar)

            # stacked pages
            self._stack = QtWidgets.QStackedWidget()
            self._stack.setStyleSheet("QStackedWidget { background: transparent; }")
            cLay.addWidget(self._stack, 1)

            # ===== build pages (cards) =====
            # We keep all config widgets as instance attrs for test compat,
            # but place them inside styled cards.

            # Shared widget creation (same as original — preserve mapping)
            # Provider
            self._transcriber = QtWidgets.QComboBox()
            self._transcriber.addItems(PROVIDERS)
            try:
                cur = cfg.transcriber if cfg.transcriber in PROVIDERS else PROVIDER_MOONSHINE
                self._transcriber.setCurrentText(cur)
            except Exception:
                pass
            self._transcriber.setCursor(QtCore.Qt.PointingHandCursor)
            self._trans_hint = QtWidgets.QLabel("")
            self._trans_hint.setObjectName("hint")
            self._trans_hint.setWordWrap(True)
            self._language = QtWidgets.QComboBox()
            self._language.setEditable(True)
            self._arch = QtWidgets.QComboBox()
            self._dl = QtWidgets.QPushButton(f"{_uni('download')} Download model")
            self._dl.setObjectName("ghost")
            self._dl.setCursor(QtCore.Qt.PointingHandCursor)
            self._dl.clicked.connect(self._download)
            self._progress = QtWidgets.QProgressBar()
            self._progress.setMaximum(1000)
            self._progress.setValue(0)
            # keep minimal text for a11y
            self._status = QtWidgets.QLabel("")
            self._status.setObjectName("hint")
            self._status.setWordWrap(True)
            self._source = QtWidgets.QComboBox()
            for label, val in [("Microphone","mic"),("Game / Discord audio (loopback)","loopback"),("Per-app (like Discord screenshare)","app"),("WAV file","wav")]:
                self._source.addItem(label, val)
            try:
                idx = self._source.findData(cfg.source or "mic")
                if idx!=-1:
                    self._source.setCurrentIndex(idx)
            except Exception:
                pass
            self._source.setCursor(QtCore.Qt.PointingHandCursor)
            ov = cfg.overlay or OverlayConfig()
            self._auto_place = QtWidgets.QCheckBox("Auto-place (bottom-center)")
            try:
                self._auto_place.setChecked(getattr(ov, "auto_place", True))
            except Exception:
                self._auto_place.setChecked(True)
            self._x = QtWidgets.QSpinBox(); self._x.setRange(-10000, 10000); self._x.setValue(getattr(ov, "x", 100) if ov.x is not None else 100)
            self._y = QtWidgets.QSpinBox(); self._y.setRange(-10000, 10000); self._y.setValue(getattr(ov, "y", 50) if ov.y is not None else 50)
            self._w = QtWidgets.QSpinBox(); self._w.setRange(100, 5000); self._w.setValue(getattr(ov, "width", 800) if ov.width is not None else 800)
            self._h = QtWidgets.QSpinBox(); self._h.setRange(40, 2000); self._h.setValue(getattr(ov, "height", 160) if ov.height is not None else 160)
            self._opacity = QtWidgets.QDoubleSpinBox(); self._opacity.setRange(0.1, 1.0); self._opacity.setSingleStep(0.05); self._opacity.setValue(getattr(ov, "opacity", 0.85))
            self._max_lines = QtWidgets.QSpinBox(); self._max_lines.setRange(1, 20); self._max_lines.setValue(getattr(ov, "max_lines", 5))
            self._show_trans = QtWidgets.QCheckBox("Show transcription block"); self._show_trans.setChecked(getattr(ov, "show_transcription_block", True))
            self._show_transl = QtWidgets.QCheckBox("Show translation block"); self._show_transl.setChecked(getattr(ov, "show_translation_block", True))
            self._translate_mode = QtWidgets.QComboBox(); self._translate_mode.addItems(list(TRANSLATE_MODES))
            try:
                self._translate_mode.setCurrentText(getattr(cfg, "translate", "passthrough"))
            except Exception:
                pass
            self._target_lang = QtWidgets.QComboBox(); self._target_lang.setEditable(True); self._target_lang.addItems(TARGET_LANGS)
            try:
                self._target_lang.setEditText(getattr(cfg, "target_language", "en"))
            except Exception:
                pass
            self._translate_url = QtWidgets.QLineEdit(getattr(cfg, "translate_url", "") or "")
            self._translate_key = QtWidgets.QLineEdit(getattr(cfg, "translate_key", "") or ""); self._translate_key.setEchoMode(QtWidgets.QLineEdit.Password)
            self._display = QtWidgets.QComboBox(); self._display.addItems(["console", "overlay", "file"])
            try:
                self._display.setCurrentText(getattr(cfg, "display", "overlay"))
            except Exception:
                pass
            self._save = QtWidgets.QPushButton(f" {_uni('save')} Save config")
            self._save.setObjectName("ghost")
            self._save.setCursor(QtCore.Qt.PointingHandCursor)
            self._save.clicked.connect(self._save_cfg)
            self._start = QtWidgets.QPushButton(f" {_uni('play')} Start")
            self._start.setObjectName("primary")
            self._start.setCursor(QtCore.Qt.PointingHandCursor)
            self._start.clicked.connect(self._start_run)
            self._stop = QtWidgets.QPushButton(f" {_uni('stop')} Stop")
            self._stop.setObjectName("danger")
            self._stop.setCursor(QtCore.Qt.PointingHandCursor)
            self._stop.clicked.connect(self._stop_run)
            self._stop.setEnabled(False)

            # helper to make form rows inside a card
            def _row(cardLay, label: str, widget, hint: str | None = None):
                row = QtWidgets.QHBoxLayout()
                row.setSpacing(12)
                lab = QtWidgets.QLabel(label)
                lab.setObjectName("muted")
                lab.setMinimumWidth(140)
                lab.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
                lab.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
                lab.setToolTip(label)
                # ensure no truncation without feedback: elide is handled by Qt if space tight, tooltip shows full
                lab.setWordWrap(False)
                row.addWidget(lab)
                row.addWidget(widget, 1)
                cardLay.addLayout(row)
                if hint is not None:
                    h = QtWidgets.QLabel(hint)
                    h.setObjectName("hint")
                    h.setWordWrap(True)
                    # indent under widget
                    hl = QtWidgets.QHBoxLayout()
                    hl.addSpacing(152)
                    hl.addWidget(h, 1)
                    cardLay.addLayout(hl)
                return lab

            def _scroll_wrap(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
                sc = QtWidgets.QScrollArea()
                sc.setWidgetResizable(True)
                sc.setFrameShape(QtWidgets.QFrame.NoFrame)
                sc.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                container = QtWidgets.QWidget()
                v = QtWidgets.QVBoxLayout(container)
                v.setContentsMargins(2, 2, 10, 2)
                v.setSpacing(14)
                v.addWidget(widget)
                v.addStretch(1)
                sc.setWidget(container)
                return sc

            # ----- Page 0: Engine -----
            p0 = QtWidgets.QWidget()
            p0Lay = QtWidgets.QVBoxLayout(p0)
            p0Lay.setContentsMargins(0,0,0,0)
            p0Lay.setSpacing(14)
            c0, l0 = _card("ASR Engine")
            _row(l0, "Provider", self._transcriber)
            # hint row directly (not via _row to keep spanning)
            hl0 = QtWidgets.QHBoxLayout()
            hl0.addSpacing(152)
            hl0.addWidget(self._trans_hint, 1)
            l0.addLayout(hl0)
            _row(l0, "Language", self._language)
            _row(l0, "Model size", self._arch)
            # download row
            dlRow = QtWidgets.QHBoxLayout()
            dlRow.addSpacing(152)
            dlRow.addWidget(self._dl, 1)
            l0.addLayout(dlRow)
            l0.addWidget(self._progress)
            l0.addWidget(self._status)
            # caption preview card (live)
            self._captionPreview = CaptionPreviewCard()
            # skeleton initially visible until first catalog ready
            self._skel = SkeletonShimmer()
            self._skel.setVisible(False)
            p0Lay.addWidget(c0)
            p0Lay.addWidget(self._captionPreview)
            p0Lay.addWidget(self._skel)
            p0Lay.addStretch(1)
            self._stack.addWidget(_scroll_wrap(p0))

            # ----- Page 1: Capture -----
            p1 = QtWidgets.QWidget()
            p1Lay = QtWidgets.QVBoxLayout(p1)
            p1Lay.setContentsMargins(0,0,0,0)
            p1Lay.setSpacing(14)
            c1, l1 = _card("Capture Source")
            _row(l1, "Source", self._source)
            # level meter already in topBar but also show here with label
            meterRow = QtWidgets.QHBoxLayout()
            meterRow.addSpacing(152)
            lab_m = QtWidgets.QLabel("Level")
            lab_m.setObjectName("muted")
            lab_m.setFixedWidth(0)
            # reuse meter in place? create duplicate display label
            meterInfo = QtWidgets.QLabel("Live level — green = speech, smooth decay")
            meterInfo.setObjectName("hint")
            meterRow.addWidget(meterInfo, 1)
            l1.addLayout(meterRow)
            # WAV path (if source == wav)
            try:
                self._wav_path = QtWidgets.QLineEdit(getattr(cfg, "input_file", "") or "")
                self._wav_path.setPlaceholderText("Path to .wav file (when Source = WAV)")
                self._wav_row = QtWidgets.QWidget()
                _wav_lay = QtWidgets.QHBoxLayout(self._wav_row)
                _wav_lay.setContentsMargins(0,0,0,0); _wav_lay.setSpacing(12)
                _wav_lab = QtWidgets.QLabel("WAV file"); _wav_lab.setObjectName("muted"); _wav_lab.setMinimumWidth(140); _wav_lab.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred); _wav_lab.setToolTip("WAV file"); _wav_lab.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
                _wav_lay.addWidget(_wav_lab); _wav_lay.addWidget(self._wav_path, 1)
                l1.addWidget(self._wav_row)
            except Exception:
                self._wav_path = None
                self._wav_row = None
            # App selector (if source == app) — combo + refresh + manual fallback
            try:
                self._app_combo = QtWidgets.QComboBox()
                self._app_combo.setEditable(False)
                self._app_combo.setPlaceholderText("Select running audio app…")
                self._app_refresh = QtWidgets.QPushButton("↻ Refresh")
                self._app_refresh.setObjectName("ghost")
                self._app_refresh.setCursor(QtCore.Qt.PointingHandCursor)
                self._app_refresh.setFixedWidth(90)
                self._app_hint = QtWidgets.QLabel("")
                self._app_hint.setObjectName("hint")
                self._app_hint.setWordWrap(True)
                self._app_hint.setVisible(False)
                self._app_selector = QtWidgets.QLineEdit(getattr(cfg, "source_app", "") or "")
                self._app_selector.setPlaceholderText("App name e.g. Discord.exe or pid:1234 (manual)")
                # row: combo + refresh
                self._app_row = QtWidgets.QWidget()
                _app_lay = QtWidgets.QHBoxLayout(self._app_row)
                _app_lay.setContentsMargins(0,0,0,0); _app_lay.setSpacing(12)
                _app_lab = QtWidgets.QLabel("Target app"); _app_lab.setObjectName("muted"); _app_lab.setMinimumWidth(140); _app_lab.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred); _app_lab.setToolTip("Target app"); _app_lab.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
                _app_lay.addWidget(_app_lab); _app_lay.addWidget(self._app_combo, 1); _app_lay.addWidget(self._app_refresh)
                l1.addWidget(self._app_row)
                # hint row
                _app_hint_row = QtWidgets.QHBoxLayout()
                _app_hint_row.addSpacing(152); _app_hint_row.addWidget(self._app_hint, 1)
                l1.addLayout(_app_hint_row)
                # manual fallback row
                _app_manual_row = QtWidgets.QHBoxLayout()
                _app_manual_row.addSpacing(152); _app_manual_row.addWidget(self._app_selector, 1)
                l1.addLayout(_app_manual_row)
                self._app_refresh.clicked.connect(lambda: self._refresh_app_list())
                # sync combo -> lineedit
                def _sync_app_to_edit(idx):
                    try:
                        data = self._app_combo.currentData()
                        if data:
                            self._app_selector.setText(str(data))
                    except Exception:
                        pass
                self._app_combo.currentIndexChanged.connect(_sync_app_to_edit)
            except Exception:
                self._app_combo = None
                self._app_refresh = None
                self._app_hint = None
                self._app_row = None
                try:
                    self._app_selector = QtWidgets.QLineEdit(getattr(cfg, "source_app", "") or "")
                    self._app_selector.setPlaceholderText("App name e.g. Discord.exe or pid:1234")
                    _row(l1, "Target app", self._app_selector)
                except Exception:
                    self._app_selector = None
            # Loopback device picker (if source == loopback)
            try:
                self._device_combo = QtWidgets.QComboBox()
                self._device_combo.setEditable(False)
                self._device_combo.setPlaceholderText("Select loopback device…")
                self._device_refresh = QtWidgets.QPushButton("↻ Refresh")
                self._device_refresh.setObjectName("ghost")
                self._device_refresh.setCursor(QtCore.Qt.PointingHandCursor)
                self._device_refresh.setFixedWidth(90)
                self._device_hint = QtWidgets.QLabel("")
                self._device_hint.setObjectName("hint")
                self._device_hint.setWordWrap(True)
                self._device_hint.setVisible(False)
                self._device_row = QtWidgets.QWidget()
                _dev_lay = QtWidgets.QHBoxLayout(self._device_row)
                _dev_lay.setContentsMargins(0,0,0,0); _dev_lay.setSpacing(12)
                _dev_lab = QtWidgets.QLabel("Device"); _dev_lab.setObjectName("muted"); _dev_lab.setMinimumWidth(140); _dev_lab.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred); _dev_lab.setToolTip("Device"); _dev_lab.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
                _dev_lay.addWidget(_dev_lab); _dev_lay.addWidget(self._device_combo, 1); _dev_lay.addWidget(self._device_refresh)
                l1.addWidget(self._device_row)
                _dev_hint_row = QtWidgets.QHBoxLayout()
                _dev_hint_row.addSpacing(152); _dev_hint_row.addWidget(self._device_hint, 1)
                l1.addLayout(_dev_hint_row)
                self._device_refresh.clicked.connect(lambda: self._refresh_device_list())
            except Exception:
                self._device_combo = None
                self._device_refresh = None
                self._device_hint = None
                self._device_row = None
            p1Lay.addWidget(c1)
            # info card
            c1b, l1b = _card("Tips")
            tip = QtWidgets.QLabel("• Mic: allow desktop apps in Windows Privacy → Microphone\n• Loopback: captures game/Discord audio (WASAPI)\n• Per-app: like Discord screenshare picker (Windows 10+)")
            tip.setObjectName("hint")
            tip.setWordWrap(True)
            l1b.addWidget(tip)
            p1Lay.addWidget(c1b)
            p1Lay.addStretch(1)
            self._stack.addWidget(_scroll_wrap(p1))

            # ----- Page 2: Overlay -----
            p2 = QtWidgets.QWidget()
            p2Lay = QtWidgets.QVBoxLayout(p2)
            p2Lay.setContentsMargins(0,0,0,0)
            p2Lay.setSpacing(14)
            c2, l2 = _card("Overlay Placement")
            _row(l2, "Position", self._auto_place)
            _row(l2, "X", self._x)
            _row(l2, "Y", self._y)
            _row(l2, "Width", self._w)
            _row(l2, "Height", self._h)
            _row(l2, "Opacity", self._opacity)
            _row(l2, "Max lines", self._max_lines)
            c2b, l2b = _card("Visibility")
            # toggles side-by-side
            togRow = QtWidgets.QHBoxLayout()
            togRow.addWidget(self._show_trans, 1)
            togRow.addWidget(self._show_transl, 1)
            l2b.addLayout(togRow)
            p2Lay.addWidget(c2)
            p2Lay.addWidget(c2b)
            p2Lay.addStretch(1)
            self._stack.addWidget(_scroll_wrap(p2))

            # ----- Page 3: Translate & Display -----
            p3 = QtWidgets.QWidget()
            p3Lay = QtWidgets.QVBoxLayout(p3)
            p3Lay.setContentsMargins(0,0,0,0)
            p3Lay.setSpacing(14)
            c3, l3 = _card("Translation")
            _row(l3, "Mode", self._translate_mode)
            _row(l3, "Target", self._target_lang)
            _row(l3, "Endpoint", self._translate_url)
            _row(l3, "API key", self._translate_key)
            c3b, l3b = _card("Display")
            _row(l3b, "Output", self._display)
            # preview text
            preview = QtWidgets.QLabel("Frameless overlay cards use rgba(15,17,21,0.82) @ 12px radius — blur 24, y2, alpha 60.")
            preview.setObjectName("hint")
            preview.setWordWrap(True)
            l3b.addWidget(preview)
            p3Lay.addWidget(c3)
            p3Lay.addWidget(c3b)
            p3Lay.addStretch(1)
            self._stack.addWidget(_scroll_wrap(p3))

            # bottom bar
            bottom = QtWidgets.QHBoxLayout()
            bottom.setSpacing(10)
            bottom.addWidget(self._save)
            bottom.addStretch(1)
            bottom.addWidget(self._start)
            bottom.addWidget(self._stop)
            cLay.addLayout(bottom)

            root.addWidget(content, 1)

            # ----- overlay host for toasts (top-right) -----
            self._toastHost = ToastHost(content)
            self._toastHost.setFixedWidth(380)
            # position top-right via layout trick: place in content's parent overlay
            # we use a floating positioning: update on resize
            self._downloadDialog = DownloadDialog(self)

            # ----- signals -----
            self._signals = _Signals()
            self._signals.progress.connect(self._on_progress)
            self._signals.done.connect(self._on_done)

            # Initial population
            try:
                self._refresh_provider_models()
            except Exception:
                pass
            self._transcriber.currentTextChanged.connect(lambda _: self._refresh_provider_models())
            # source visibility
            try:
                self._source.currentIndexChanged.connect(lambda _: self._on_source_changed())
                # also handle text changed for robustness
                self._source.currentTextChanged.connect(lambda _: self._on_source_changed())
            except Exception:
                pass
            # initial source visibility + populate pickers
            try:
                self._refresh_app_list()
            except Exception:
                pass
            try:
                self._refresh_device_list()
            except Exception:
                pass
            try:
                self._on_source_changed()
            except Exception:
                pass
            # save via rail button
            self._rail_save.clicked.connect(self._save_cfg)

            # keyboard nav: Ctrl+1..4
            for i in range(4):
                sc = QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{i+1}"), self)
                sc.activated.connect(lambda idx=i: self._navigate(idx))
            # also show caption key: Ctrl+P preview demo
            # DPI change handling
            try:
                app = QtWidgets.QApplication.instance()
                if app is not None:
                    app.primaryScreenChanged.connect(lambda *a: self.update())  # type: ignore
            except Exception:
                pass
            # nav group -> stack
            self._nav_group.idClicked.connect(self._navigate)
            # show initial pill position after layout
            QtCore.QTimer.singleShot(80, lambda: self._updatePill(0, animate=False))
            # wire meter demo when idle: gentle random pulse when running
            self._meter_demo = QtCore.QTimer(self)
            self._meter_demo.setInterval(120)
            self._meter_demo.timeout.connect(self._tickMeterDemo)
            # elapsed timer
            self._elapsed_timer = QtCore.QTimer(self)
            self._elapsed_timer.setInterval(1000)
            self._elapsed_timer.timeout.connect(self._tickElapsed)

            # focus chain
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self._transcriber.setFocus()

        # ----- navigation -----
        def _navigate(self, idx: int):
            try:
                self._stack.setCurrentIndex(int(idx))
                for b in self._nav_buttons:
                    b.setChecked(False)
                if 0 <= int(idx) < len(self._nav_buttons):
                    self._nav_buttons[int(idx)].setChecked(True)
                self._updatePill(int(idx), animate=True)
            except Exception:
                pass

        def _updatePill(self, idx: int, animate: bool = True):
            try:
                if not (0 <= idx < len(self._nav_buttons)):
                    return
                btn = self._nav_buttons[idx]
                if not self._nav_pill.isVisible():
                    self._nav_pill.show()
                target = btn.geometry()
                # map to rail coordinates
                target = QtCore.QRect(target)
                # expand pill to button rect with 2px inset
                dest = QtCore.QRect(target.x() - 2, target.y() - 1, target.width() + 4, target.height() + 2)
                if not animate or self._pill_anim is not None and self._pill_anim.state() == QtCore.QAbstractAnimation.Running:  # type: ignore
                    self._nav_pill.setGeometry(dest)
                    return
                anim = QtCore.QPropertyAnimation(self._nav_pill, b"geometry", self)
                anim.setDuration(200)
                anim.setStartValue(self._nav_pill.geometry())
                anim.setEndValue(dest)
                anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
                anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
                self._pill_anim = anim
            except Exception:
                pass

        def resizeEvent(self, e):
            super().resizeEvent(e)
            try:
                # keep toast host top-right
                self._toastHost.move(self.width() - self._toastHost.width() - 18, 18)
                self._toastHost.raise_()
                # center download dialog
                if self._downloadDialog.isVisible():
                    self._downloadDialog.move((self.width() - self._downloadDialog.width()) // 2, (self.height() - self._downloadDialog.height()) // 2)
            except Exception:
                pass

        def keyPressEvent(self, e):
            try:
                if e.key() in (QtCore.Qt.Key_1, QtCore.Qt.Key_2, QtCore.Qt.Key_3, QtCore.Qt.Key_4) and e.modifiers() & QtCore.Qt.ControlModifier:
                    self._navigate(e.key() - QtCore.Qt.Key_1)
                    e.accept()
                    return
                if e.key() == QtCore.Qt.Key_Escape and self._downloadDialog.isVisible():
                    self._downloadDialog.hide()
                    e.accept()
                    return
            except Exception:
                pass
            super().keyPressEvent(e)

        def _on_source_changed(self):
            try:
                src = str(self._source.currentData() or self._source.currentText() or "mic").lower()
            except Exception:
                src = "mic"
            is_app = src == "app"
            is_loop = src == "loopback"
            is_wav = src == "wav"
            for attr, show in [("_app_row", is_app), ("_device_row", is_loop), ("_wav_row", is_wav)]:
                try:
                    w = getattr(self, attr, None)
                    if w is not None:
                        w.setVisible(show)
                except Exception:
                    pass
            for hint_attr, show in [("_app_hint", is_app), ("_device_hint", is_loop)]:
                try:
                    h = getattr(self, hint_attr, None)
                    if h is not None and not show:
                        h.setVisible(False)
                except Exception:
                    pass
            # disable Start when app selected but no target chosen
            try:
                if is_app:
                    has_sel = False
                    if hasattr(self, "_app_selector") and self._app_selector is not None:
                        has_sel = bool(self._app_selector.text().strip())
                    if hasattr(self, "_app_combo") and self._app_combo is not None:
                        try:
                            d = self._app_combo.currentData()
                            if d:
                                has_sel = True
                        except Exception:
                            pass
                    # don't disable immediately — just hint, validation blocks Start
                    if not has_sel and hasattr(self, "_app_hint") and self._app_hint is not None:
                        self._app_hint.setText("Pick an app above or type its name (e.g. Discord.exe) — Refresh to rescan")
                        self._app_hint.setVisible(True)
            except Exception:
                pass

        def _refresh_app_list(self):
            try:
                if not hasattr(self, "_app_combo") or self._app_combo is None:
                    return
                from voicelang_core.adapters.app_loopback import list_audio_apps
                apps = list_audio_apps()
                self._app_combo.blockSignals(True)
                self._app_combo.clear()
                if not apps:
                    self._app_combo.addItem("No running audio apps detected", "")
                    if hasattr(self, "_app_hint") and self._app_hint is not None:
                        # detect missing deps vs no sessions
                        try:
                            import importlib.util
                            missing = [m for m in ("pycaw", "psutil") if importlib.util.find_spec(m) is None]
                            if missing:
                                self._app_hint.setText("Per-app capture requires Windows + pycaw/psutil — install with: uv sync --extra real")
                            else:
                                self._app_hint.setText("No running audio apps detected — start an app that plays audio, then Refresh")
                            self._app_hint.setVisible(True)
                        except Exception:
                            pass
                    self._app_combo.blockSignals(False)
                    return
                if hasattr(self, "_app_hint") and self._app_hint is not None:
                    self._app_hint.setVisible(False)
                # preserve previous selection
                try:
                    prev = getattr(self._cfg, "source_app", "") or ""
                    if hasattr(self, "_app_selector") and self._app_selector is not None:
                        cur_txt = self._app_selector.text().strip()
                        if cur_txt:
                            prev = cur_txt
                except Exception:
                    prev = ""
                sel_idx = 0
                for a in apps:
                    label = f"{a.get('name','?')} ({a.get('pid','?')})"
                    data = a.get("name") or str(a.get("pid"))
                    # store exe or name; prefer name for display but keep pid fallback via AppLoopbackSource resolver
                    store = a.get("exe") or a.get("name") or str(a.get("pid"))
                    # if exe path present, store basename
                    try:
                        import os as _os
                        if store and ("/" in store or "\\" in store):
                            store = _os.path.basename(store)
                    except Exception:
                        pass
                    self._app_combo.addItem(label, store)
                    if prev and store and prev.lower() in (store.lower(), a.get("name","").lower()):
                        sel_idx = self._app_combo.count() - 1
                # also support pid: prefix
                if prev and prev.lower().startswith("pid:"):
                    for i in range(self._app_combo.count()):
                        if str(self._app_combo.itemData(i)).lower() == prev.lower():
                            sel_idx = i
                            break
                self._app_combo.setCurrentIndex(sel_idx)
                self._app_combo.blockSignals(False)
                # sync to lineedit
                try:
                    d = self._app_combo.currentData()
                    if d and hasattr(self, "_app_selector") and self._app_selector is not None:
                        if not self._app_selector.text().strip():
                            self._app_selector.setText(str(d))
                except Exception:
                    pass
            except Exception as e:
                try:
                    self._app_combo.blockSignals(False)
                except Exception:
                    pass
                if hasattr(self, "_app_hint") and self._app_hint is not None:
                    self._app_hint.setText(f"Could not list apps: {e}")
                    self._app_hint.setVisible(True)

        def _refresh_device_list(self):
            try:
                if not hasattr(self, "_device_combo") or self._device_combo is None:
                    return
                from voicelang_core.adapters.wasapi_loopback import list_loopback_devices
                devs = list_loopback_devices()
                self._device_combo.blockSignals(True)
                self._device_combo.clear()
                # default
                self._device_combo.addItem("Default device", "")
                # preserve cfg
                try:
                    prev_dev = getattr(self._cfg, "source_device", "") or ""
                except Exception:
                    prev_dev = ""
                sel = 0
                if not devs:
                    if hasattr(self, "_device_hint") and self._device_hint is not None:
                        try:
                            import importlib.util, sys as _sys
                            if _sys.platform != "win32":
                                self._device_hint.setText("Per-app capture requires Windows")
                            elif importlib.util.find_spec("pyaudiowpatch") is None:
                                self._device_hint.setText("Loopback capture requires pyaudiowpatch — install with: uv sync --extra real")
                            else:
                                self._device_hint.setText("No loopback devices found")
                            self._device_hint.setVisible(True)
                        except Exception:
                            pass
                    # restore selection if prev_dev was empty (default)
                    if prev_dev == "":
                        sel = 0
                    self._device_combo.setCurrentIndex(sel)
                    self._device_combo.blockSignals(False)
                    return
                if hasattr(self, "_device_hint") and self._device_hint is not None:
                    self._device_hint.setVisible(False)
                for d in devs:
                    name = d.get("name", f"device {d.get('index')}")
                    idx = d.get("index")
                    # store name for robustness across reboots (run.py does name lookup), data = name
                    self._device_combo.addItem(f"{name} [{idx}]", name)
                    if prev_dev and prev_dev.lower() in name.lower():
                        sel = self._device_combo.count() - 1
                self._device_combo.setCurrentIndex(sel)
                self._device_combo.blockSignals(False)
            except Exception as e:
                try:
                    self._device_combo.blockSignals(False)
                except Exception:
                    pass
                if hasattr(self, "_device_hint") and self._device_hint is not None:
                    self._device_hint.setText(f"Could not list devices: {e}")
                    self._device_hint.setVisible(True)

        def _toast(self, msg: str, level: str = "info"):
            try:
                self._toastHost.toast(msg, level)
            except Exception:
                pass
            # also reflect in hint status for a11y
            try:
                if level in ("error", "warning"):
                    self._status.setText(msg)
            except Exception:
                pass

        def _tickMeterDemo(self):
            # when running, simulate live level; when idle, keep low
            try:
                if self._proc is not None and self._proc.poll() is None:
                    # gentle random walk biased by time
                    lvl = 0.18 + 0.45 * abs(math.sin(time.monotonic() * 2.1)) * (0.5 + random.random() * 0.5)
                    self._meter.setLevel(lvl)
                else:
                    self._meter.setLevel(self._meter._level * 0.86)
            except Exception:
                pass

        def _tickElapsed(self):
            try:
                if self._start_ts is None:
                    return
                elapsed = int(time.monotonic() - self._start_ts)
                mm = elapsed // 60
                ss = elapsed % 60
                hh = mm // 60
                mm %= 60
                if hh:
                    self._elapsedLab.setText(f"{hh:02d}:{mm:02d}:{ss:02d}")
                else:
                    self._elapsedLab.setText(f"{mm:02d}:{ss:02d}")
            except Exception:
                pass

        def _setPillState(self, state: str, label: str):
            try:
                self._dot.setState(state)
                self._statusPillLabel.setText(label)
            except Exception:
                pass

        # ----- preserved logic (verbatim semantics) -----
        def _probe_arch_cached(self, provider: str, arch: str, lang: str) -> bool:
            # Live manifest-aware probe — checks actual local cache/download dir
            # against the fetched manifest, not hardcoded assumption.
            if _live_is_cached is not None:
                try:
                    return bool(_live_is_cached(provider, arch, lang))
                except Exception:
                    pass
            try:
                if provider == PROVIDER_VOSK:
                    try:
                        from voicelang_core.adapters.vosk import _is_model_present, vosk_model_dir, vosk_cache_root, get_vosk_model_info
                        info = get_vosk_model_info(lang)
                        logical = vosk_model_dir(lang)
                        alt = vosk_cache_root() / info["model_name"]
                        return _is_model_present(logical) or _is_model_present(alt)
                    except Exception:
                        return False
                if provider == PROVIDER_WHISPER:
                    return _whisper_model_cached(arch)
                if provider == PROVIDER_NEMOTRON:
                    return _nemotron_installed()
                if provider == PROVIDER_MOONSHINE:
                    return _moonshine_model_cached(lang, arch)
                if provider == PROVIDER_FUNASR:
                    mid = arch if arch != "auto" else ("paraformer-zh-streaming" if lang.lower() in ("zh", "zh-cn", "yue") else "iic/SenseVoiceSmall")
                    if mid == "SenseVoiceSmall":
                        mid = "iic/SenseVoiceSmall"
                    return _funasr_model_cached(mid)
                if provider == PROVIDER_FUNASR_NANO:
                    return _funasr_model_cached("FunAudioLLM/Fun-ASR-Nano-2512-hf")
            except Exception:
                return False
            return False

        def _refresh_provider_models(self, initial_language: str | None = None, initial_arch: str | None = None):
            import sys as _sys
            provider = str(self._transcriber.currentText() or "").strip()
            # ---- compute targets (never crash) ----
            try:
                cur_lang = self._language.currentData() or self._language.currentText() or self._cfg.language
            except Exception:
                cur_lang = getattr(self._cfg, "language", "en")
            try:
                cur_arch = self._arch.currentData() or self._arch.currentText() or self._cfg.model_arch
            except Exception:
                cur_arch = getattr(self._cfg, "model_arch", "small")
            target_lang = initial_language or cur_lang or "en"
            target_arch = initial_arch or cur_arch or "small"
            # ---- compute catalog (isolated, never aborts UI update) ----
            try:
                moon_langs, moon_friendly = _moonshine_catalog()
            except Exception as _e:
                print(f"[voicelang] _moonshine_catalog failed: {_e}", file=_sys.stderr)
                moon_langs, moon_friendly = VOSK_LANGS, VOSK_FRIENDLY
            try:
                catalog = _catalog_for(provider, moon_langs, moon_friendly)
                # tolerate legacy bare-list return (e.g. registry-only) — normalize to 5-tuple
                if isinstance(catalog, (list, tuple)) and len(catalog) == 5:
                    langs, friendly, arches, hint, dl_label = catalog
                elif isinstance(catalog, (list, tuple)):
                    langs = list(catalog) if catalog else list(moon_langs)
                    friendly = {l: l for l in langs}
                    arches = ["default"]
                    hint = ""
                    dl_label = f"Download {provider} model" if provider else "Download model"
                    print(f"[voicelang] _catalog_for returned bare list for {provider!r}, normalized", file=_sys.stderr)
                else:
                    raise TypeError(f"unexpected catalog type {type(catalog)}")
            except Exception as _e:
                print(f"[voicelang] _catalog_for failed for {provider!r}: {_e}", file=_sys.stderr)
                langs, friendly, arches, hint, dl_label = VOSK_LANGS, VOSK_FRIENDLY, VOSK_ARCHES, "", f"Download {provider} model"
            # sanitize
            if not langs:
                langs = ["auto", "en"]
            if not arches:
                arches = ["small"]
            if not isinstance(friendly, dict):
                friendly = {l: l for l in langs}
            # ---- validate target_lang/arch against new provider (reset stale) ----
            _langs_lower = {str(x).lower(): str(x) for x in langs}
            _arches_lower = {str(x).lower(): str(x) for x in arches}
            # lang: case-insensitive check, reset to first valid if stale
            if str(target_lang).lower() not in _langs_lower:
                target_lang = langs[0]
            else:
                target_lang = _langs_lower[str(target_lang).lower()]
            if str(target_arch).lower() not in _arches_lower:
                target_arch = arches[0]
            else:
                target_arch = _arches_lower[str(target_arch).lower()]
            # ---- precompute arch display labels (probe failures must not abort) ----
            arch_items: list[tuple[str, str]] = []
            for arch in arches:
                try:
                    cached = self._probe_arch_cached(provider, arch, target_lang)
                except Exception as _e:
                    print(f"[voicelang] probe failed {provider}/{arch}/{target_lang}: {_e}", file=_sys.stderr)
                    cached = False
                try:
                    if provider == PROVIDER_VOSK:
                        label = f"{arch} \u2713 installed" if cached else f"{arch} \u2193 download"
                    else:
                        label = _model_label(arch, cached)
                except Exception:
                    label = str(arch)
                arch_items.append((label, str(arch)))
            try:
                hint_dl_cached = self._probe_arch_cached(provider, target_arch, target_lang)
            except Exception:
                hint_dl_cached = False
            # ---- atomic widget apply (all or with logged partial) ----
            # language
            try:
                self._language.blockSignals(True)
                try:
                    self._language.clear()
                    for lang in langs:
                        label = friendly.get(lang, lang) if isinstance(friendly, dict) else lang
                        try:
                            self._language.addItem(str(label), str(lang))
                        except Exception as _e:
                            print(f"[voicelang] add lang {lang!r} failed: {_e}", file=_sys.stderr)
                    idx = self._language.findData(target_lang)
                    if idx == -1:
                        for i in range(self._language.count()):
                            try:
                                if str(self._language.itemData(i)).lower() == str(target_lang).lower():
                                    idx = i; break
                            except Exception:
                                continue
                    if idx != -1:
                        self._language.setCurrentIndex(idx)
                    else:
                        try:
                            self._language.setCurrentIndex(0)
                            target_lang = str(self._language.currentData() or self._language.currentText() or langs[0])
                        except Exception:
                            pass
                finally:
                    self._language.blockSignals(False)
            except Exception as _e:
                print(f"[voicelang] language refresh failed: {_e}", file=_sys.stderr)
                try:
                    self._language.blockSignals(False)
                except Exception:
                    pass
            # arch
            try:
                self._arch.blockSignals(True)
                try:
                    self._arch.clear()
                    for label, arch in arch_items:
                        try:
                            self._arch.addItem(label, arch)
                        except Exception as _e:
                            print(f"[voicelang] add arch {arch!r} failed: {_e}", file=_sys.stderr)
                    idx2 = self._arch.findData(target_arch)
                    if idx2 != -1:
                        self._arch.setCurrentIndex(idx2)
                    else:
                        try:
                            self._arch.setCurrentIndex(0)
                            target_arch = str(self._arch.currentData() or self._arch.currentText() or arches[0])
                        except Exception:
                            pass
                finally:
                    self._arch.blockSignals(False)
            except Exception as _e:
                print(f"[voicelang] arch refresh failed: {_e}", file=_sys.stderr)
                try:
                    self._arch.blockSignals(False)
                except Exception:
                    pass
            # hint + download label — always applied even if above failed
            try:
                self._trans_hint.setText(str(hint))
            except Exception as _e:
                print(f"[voicelang] hint set failed: {_e}", file=_sys.stderr)
            try:
                self._dl.setText(str(dl_label) + (" \u2713 installed" if hint_dl_cached else ""))
            except Exception as _e:
                print(f"[voicelang] dl label set failed: {_e}", file=_sys.stderr)

        def _cfg_from_widgets(self) -> Config:
            cfg = Config.from_dict(self._cfg.to_dict())
            try:
                cfg.transcriber = self._transcriber.currentText()
                lang = self._language.currentData() or self._language.currentText()
                if lang:
                    lang = str(lang).split(" —")[0].split(" ")[0].strip().lower()
                    cfg.language = lang
                arch = self._arch.currentData() or self._arch.currentText()
                if arch:
                    arch = str(arch).split(" ")[0].strip()
                    cfg.model_arch = arch
                src = self._source.currentData() or self._source.currentText()
                if src:
                    cfg.source = str(src).lower()
                try:
                    ov = cfg.overlay or OverlayConfig()
                    ov.x = self._x.value()
                    ov.y = self._y.value()
                    ov.width = self._w.value()
                    ov.height = self._h.value()
                    ov.opacity = float(self._opacity.value())
                    ov.max_lines = self._max_lines.value()
                    ov.show_transcription_block = bool(self._show_trans.isChecked())
                    ov.show_translation_block = bool(self._show_transl.isChecked())
                    ov.auto_place = bool(self._auto_place.isChecked())
                    cfg.overlay = ov
                except Exception:
                    pass
                try:
                    cfg.translate = self._translate_mode.currentText()
                    cfg.target_language = self._target_lang.currentText().strip().lower() or "en"
                    cfg.translate_url = self._translate_url.text().strip()
                    cfg.translate_key = self._translate_key.text().strip()
                    cfg.display = self._display.currentText()
                except Exception:
                    pass
                # wav/app extras (persist if present)
                try:
                    if hasattr(self, "_wav_path") and self._wav_path is not None:
                        cfg.input_file = self._wav_path.text().strip()
                    # app: manual text takes precedence if non-empty (allows custom app not in combo)
                    app_val = ""
                    if hasattr(self, "_app_selector") and self._app_selector is not None:
                        app_val = self._app_selector.text().strip()
                    if not app_val and hasattr(self, "_app_combo") and self._app_combo is not None:
                        try:
                            data = self._app_combo.currentData()
                            if data and "No running" not in str(data):
                                app_val = str(data).strip()
                            elif self._app_combo.currentText() and "No running" not in self._app_combo.currentText():
                                # fallback to text without pid suffix
                                txt = self._app_combo.currentText().split(" (")[0].strip()
                                if txt and txt != "No running audio apps detected":
                                    app_val = txt
                        except Exception:
                            pass
                    if app_val:
                        cfg.source_app = app_val
                    if hasattr(self, "_device_combo") and self._device_combo is not None:
                        try:
                            ddata = self._device_combo.currentData()
                            # empty string = default device
                            if ddata is not None:
                                cfg.source_device = str(ddata).strip()
                            else:
                                cfg.source_device = ""
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
            return cfg

        def _save_cfg(self):
            try:
                cfg = self._cfg_from_widgets()
                # defense-in-depth: clamp language/arch to provider's valid catalog before persisting
                try:
                    from voicelang_core.engines import get_registry as _gr
                    reg = _gr()
                    prov = cfg.transcriber.lower()
                    aliases = {"faster_whisper": "faster-whisper", "whisper_http": "whisper-http", "funasr_nano": "funasr-nano"}
                    prov = aliases.get(prov, prov)
                    if prov in reg:
                        langs_valid = [l.lower() for l in reg[prov].get("languages", [])]
                        if langs_valid and cfg.language.lower() not in langs_valid:
                            # reset to first valid instead of persisting stale
                            cfg.language = reg[prov]["languages"][0]
                        arches_valid = reg[prov].get("arches", [])
                        if arches_valid and cfg.model_arch not in arches_valid and cfg.model_arch.lower() not in [a.lower() for a in arches_valid]:
                            cfg.model_arch = arches_valid[0]
                except Exception:
                    pass
                save_config(cfg)
                self._status.setText(f"saved to {config_path()}")
                self._status.setStyleSheet("")
                self._cfg = cfg
                self._toast(f"Saved to {config_path()}", "success")
                self._countersLab.setText("✓ saved")
                QtCore.QTimer.singleShot(2000, lambda: self._countersLab.setText(""))
            except Exception as e:
                self._status.setText(f"save failed: {e}")
                self._status.setStyleSheet("color: #F87171;")
                self._toast(f"Save failed: {e}", "error")

        def _download(self):
            provider = self._transcriber.currentText()
            lang = (self._language.currentData() or self._language.currentText() or "en")
            lang = str(lang).split(" —")[0].split(" ")[0].strip().lower() or "en"
            arch = (self._arch.currentData() or self._arch.currentText() or "small")
            arch = str(arch).split(" ")[0].strip() or "small"
            self._progress.setValue(50)
            self._status.setText(f"downloading {provider} {lang}/{arch} …")
            try:
                self._downloadDialog.setProgress(0.06, f"downloading {provider} {lang}/{arch} …")
                self._downloadDialog.show()
                self._downloadDialog.raise_()
            except Exception:
                pass
            factories = {
                PROVIDER_MOONSHINE: lambda: _DownloadWorker(lang, arch, on_progress=lambda f,n: self._signals.progress.emit(f,n), on_done=self._signals.done.emit),
                PROVIDER_WHISPER: lambda: _WhisperDownloadWorker(arch, on_progress=lambda f,n: self._signals.progress.emit(f,n), on_done=self._signals.done.emit),
                PROVIDER_NEMOTRON: lambda: _NemotronDownloadWorker(on_progress=lambda f,n: self._signals.progress.emit(f,n), on_done=self._signals.done.emit),
                PROVIDER_FUNASR: lambda: _FunASRDownloadWorker(lang, arch, on_progress=lambda f,n: self._signals.progress.emit(f,n), on_done=self._signals.done.emit),
                PROVIDER_FUNASR_NANO: lambda: _FunASRNanoDownloadWorker(on_progress=lambda f,n: self._signals.progress.emit(f,n), on_done=self._signals.done.emit),
                PROVIDER_VOSK: lambda: _VoskDownloadWorker(lang, on_progress=lambda f,n: self._signals.progress.emit(f,n), on_done=self._signals.done.emit),
            }
            factory = factories.get(provider)
            if factory is None:
                self._status.setText("unknown provider")
                try: self._downloadDialog.hide()
                except Exception: pass
                return
            self._start_worker(factory(), f"downloading {provider} …")

        def _start_worker(self, worker, msg):
            from PySide6 import QtCore as _QtCore
            class _WorkerThread(_QtCore.QThread):
                def __init__(self, w): 
                    super().__init__(); self._w=w
                def run(self): self._w.run()
            self._worker = worker
            self._thread = _WorkerThread(worker)
            self._thread.finished.connect(lambda: self._progress.setValue(1000))
            self._thread.start()
            self._status.setText(msg)

        def _on_progress(self, frac, name):
            try:
                self._progress.setValue(int(frac*1000))
                self._status.setText(name)
                try: self._downloadDialog.setProgress(float(frac), str(name))
                except Exception: pass
            except Exception:
                pass

        def _on_done(self, msg):
            self._status.setText(msg)
            self._progress.setValue(1000)
            try: self._downloadDialog.done(str(msg))
            except Exception: pass
            # classify toast level
            lvl = "success" if "ready" in msg.lower() or "installed" in msg.lower() else "error" if "failed" in msg.lower() else "info"
            self._toast(msg, lvl)
            # update counters (example: model cached now)
            try:
                self._refresh_provider_models()
            except Exception:
                pass

        # Public helper for pipeline integration: caption preview keyed by segment id
        def show_caption(self, segment):
            """Render a Segment (id + source_text + translated_text) into preview — id-keyed, 220ms fade/slide."""
            try:
                seg_id = getattr(segment, "id", None)
                src = getattr(segment, "source_text", "") or ""
                tr = getattr(segment, "translated_text", None)
                status = getattr(segment, "status", "final")
                self._captionPreview.showSegment(seg_id, src, tr, status)
            except Exception:
                pass

        def _start_run(self):
            try:
                _cfg_check = self._cfg_from_widgets()
                from voicelang_core.engines import get_registry as _get_reg
                reg = _get_reg()
                prov = _cfg_check.transcriber.lower()
                aliases = {"faster_whisper": "faster-whisper", "whisper_http": "whisper-http", "funasr_nano": "funasr-nano"}
                prov_norm = aliases.get(prov, prov)
                if prov_norm in reg:
                    langs_valid = [l.lower() for l in reg[prov_norm]["languages"]]
                    if _cfg_check.language.lower() not in langs_valid:
                        self._status.setText(f"error: language '{_cfg_check.language}' invalid for {prov} — valid: {', '.join(reg[prov_norm]['languages'][:6])}")
                        self._status.setStyleSheet("color: #F87171;")
                        self._toast(f"Invalid language '{_cfg_check.language}' for {prov}", "error")
                        return
                    arches_valid = [a.lower() for a in reg[prov_norm].get("arches", [])] if reg[prov_norm].get("arches") else []
                    if arches_valid and _cfg_check.model_arch.lower() not in arches_valid:
                        self._status.setText(f"error: model '{_cfg_check.model_arch}' invalid for {prov} — valid: {', '.join(arches_valid[:6])}")
                        self._status.setStyleSheet("color: #F87171;")
                        self._toast(f"Invalid model '{_cfg_check.model_arch}' for {prov}", "error")
                        return
                if prov_norm == "funasr-nano":
                    allowed = {"zh", "zh-cn", "en", "ja", "auto", "yue", "cantonese"}
                    if _cfg_check.language.lower() not in allowed:
                        self._status.setText(f"error: funasr-nano supports {sorted(allowed)} — got '{_cfg_check.language}'")
                        self._status.setStyleSheet("color: #F87171;")
                        self._toast("funasr-nano language not supported", "error")
                        return
                # source validation
                src = (_cfg_check.source or "mic").lower()
                if src == "app" and not (_cfg_check.source_app or "").strip():
                    self._status.setText("error: Per-app capture needs a target app — pick one above or type its name")
                    self._status.setStyleSheet("color: #F87171;")
                    self._toast("Pick a target app for Per-app capture", "error")
                    try:
                        self._on_source_changed()
                    except Exception:
                        pass
                    return
                if src == "wav" and not (_cfg_check.input_file or "").strip():
                    self._status.setText("error: WAV file source needs a file path — set it in Capture")
                    self._status.setStyleSheet("color: #F87171;")
                    self._toast("WAV source needs a file path", "error")
                    return
                # translate validation (defense-in-depth) — block invalid combos, verify against TRANSLATE_MODES
                if _cfg_check.translate not in TRANSLATE_MODES:
                    self._status.setText(f"error: unknown translate mode '{_cfg_check.translate}' — valid: {', '.join(TRANSLATE_MODES)}")
                    self._status.setStyleSheet("color: #F87171;")
                    self._toast(f"Unknown translate mode '{_cfg_check.translate}'", "error")
                    return
                if _cfg_check.translate == "deepl" and not (_cfg_check.translate_key or "").strip():
                    self._status.setText("error: DeepL selected but API key is empty — set it in Translate")
                    self._status.setStyleSheet("color: #F87171;")
                    self._toast("DeepL needs an API key — set it in Translate", "error")
                    return
                if _cfg_check.translate in ("libretranslate", "deeplx") and not (_cfg_check.translate_url or "").strip():
                    self._status.setText(f"error: {_cfg_check.translate} selected but endpoint URL is empty — set it in Translate")
                    self._status.setStyleSheet("color: #F87171;")
                    self._toast(f"{_cfg_check.translate} needs an endpoint URL", "error")
                    return
            except Exception as _ve:
                pass
            # --- overlay preflight: fail LOUD if PySide6 missing before launching worker ---
            try:
                if _cfg_check.display == "overlay":
                    import importlib.util as _ilu
                    if _ilu.find_spec("PySide6") is None:
                        self._status.setText("error: overlay requested but PySide6 not installed — run: uv sync --extra overlay")
                        self._status.setStyleSheet("color: #F87171;")
                        self._toast("Overlay requires PySide6 — install with: uv sync --extra overlay", "error")
                        try:
                            from PySide6 import QtWidgets as _QWpf
                            _QWpf.QMessageBox.warning(self, "overlay error", "Overlay requested but PySide6 is not installed.\n\nInstall it:  uv sync --extra overlay\nInstalled launcher and worker must share the same venv.")
                        except Exception:
                            # No Qt available at all — QMessageBox can't show, status+toast is the loud signal
                            pass
                        return
            except Exception:
                pass
            self._save_cfg()
            if getattr(sys, "frozen", False):
                cmd = [sys.executable, "--run", "--config", config_path()]
            else:
                cmd = [sys.executable, "-m", "voicelang_core.run", "--config", config_path()]
            try:
                from pathlib import Path as _Path
                import time as _time
                from voicelang_core.config import voicelang_data_dir
                log_dir = _Path(voicelang_data_dir()) / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                ts = _time.strftime("%Y%m%d_%H%M%S")
                self._log_path = str(log_dir / f"voicelang_{ts}.log")
                self._log_file = open(self._log_path, "w", encoding="utf-8", buffering=1, errors="replace")
                self._proc = subprocess.Popen(cmd, cwd=_repo_root(), stdout=subprocess.DEVNULL, stderr=self._log_file, text=True, encoding="utf-8", errors="replace")
                _ACTIVE_PROCS.append(self._proc)
                self._start.setEnabled(False); self._stop.setEnabled(True)
                self._status.setText("running…")
                self._status.setStyleSheet("")
                self._setPillState("running", "Running")
                self._start_ts = time.monotonic()
                self._elapsedLab.setText("00:00")
                try: self._elapsed_timer.start()
                except Exception: pass
                try: self._meter_demo.start()
                except Exception: pass
                self._countersLab.setText("● streaming")
                try: self._captionPreview.clear()
                except Exception: pass
                self._toast("Pipeline started", "success")
                try:
                    from PySide6 import QtCore as _QtC
                    self._poll_timer = _QtC.QTimer(self)
                    self._poll_timer.setInterval(800)
                    self._poll_timer.timeout.connect(self._poll_proc)
                    self._poll_timer.start()
                    # early-failure fast path: check within first seconds (400ms + 1200ms)
                    try:
                        _QtC.QTimer.singleShot(400, self._poll_proc)
                        _QtC.QTimer.singleShot(1200, self._poll_proc)
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception as e:
                log_hint = getattr(self, '_log_path', '')
                msg = f"start failed: {e}"
                if log_hint:
                    msg = f"{msg} — Log: {log_hint}"
                self._status.setText(msg)
                self._status.setStyleSheet("color: #F87171;")
                try:
                    if hasattr(self, '_log_file') and self._log_file:
                        self._log_file.close()
                        self._log_file = None
                except Exception:
                    pass
                self._setPillState("error", "Error")
                self._toast(f"Start failed: {e}" + (f" — Log: {log_hint}" if log_hint else ""), "error")

        def _poll_proc(self):
            try:
                if not getattr(self, "_proc", None):
                    return
                ret = self._proc.poll()
                if ret is None:
                    # still running — update elapsed + optional degraded check via log tail
                    try:
                        # degraded if log contains WARN
                        if hasattr(self, "_log_path"):
                            from pathlib import Path as _Path
                            txt = _Path(self._log_path).read_text(encoding="utf-8", errors="replace")[-4000:]
                            if "degraded" in txt.lower() or "warn" in txt.lower():
                                if self._dot._state != "degraded":
                                    self._setPillState("degraded", "Degraded")
                                    self._countersLab.setText("⚠ degraded")
                            # also try to parse segment counters
                            # count lines with '"id"' or 'segment'
                            import re
                            ids = re.findall(r'"id"\s*:\s*(\d+)', txt)
                            if ids:
                                self._countersLab.setText(f"#{ids[-1]} · elapsed {self._elapsedLab.text()}")
                    except Exception:
                        pass
                    return
                try:
                    if hasattr(self, "_poll_timer"):
                        self._poll_timer.stop()
                except Exception:
                    pass
                try:
                    if hasattr(self, "_elapsed_timer"):
                        self._elapsed_timer.stop()
                except Exception:
                    pass
                try:
                    if hasattr(self, "_meter_demo"):
                        self._meter_demo.stop()
                except Exception:
                    pass
                try:
                    if hasattr(self, "_log_file") and self._log_file:
                        self._log_file.flush()
                        self._log_file.close()
                        self._log_file = None
                except Exception:
                    pass
                self._start.setEnabled(True)
                self._stop.setEnabled(False)
                if ret == 0:
                    self._status.setText("stopped")
                    self._status.setStyleSheet("")
                    self._setPillState("idle", "Idle")
                    self._countersLab.setText(f"stopped · {self._elapsedLab.text()}")
                    self._toast("Pipeline stopped", "info")
                else:
                    tail = ""
                    try:
                        from pathlib import Path as _Path
                        if hasattr(self, "_log_path"):
                            txt = _Path(self._log_path).read_text(encoding="utf-8", errors="replace")[-4000:].strip()
                            if txt:
                                # last 2 non-empty lines for context
                                lines = [l for l in txt.splitlines() if l.strip()]
                                tail = " | ".join(lines[-2:]) if len(lines) >= 2 else lines[-1]
                                # truncate to keep status readable
                                if len(tail) > 180:
                                    tail = tail[-180:]
                    except Exception:
                        pass
                    codes = {10: "invalid config — check language/transcriber pairing", 11: "model missing — download required", 20: "microphone permission denied — check Windows Settings → Privacy → Microphone", 30: "overlay unavailable — check PySide6 is installed (uv sync --extra overlay) and display server is running"}
                    hint = codes.get(ret, "")
                    msg = tail or f"process exited with code {ret}"
                    if hint:
                        msg = f"{msg} — {hint}"
                    # always append log path hint for FAILED state
                    log_ref = getattr(self, '_log_path', '')
                    status_msg = f"{msg} — Log: {log_ref}" if log_ref else msg
                    self._status.setText(status_msg)
                    self._status.setStyleSheet("color: #F87171; background: rgba(248,113,113,0.12); padding: 4px; border-radius: 6px;")
                    self._setPillState("error", f"Error {ret}")
                    self._countersLab.setText(f"error {ret}")
                    self._toast(f"{msg} — Log: {log_ref}" if log_ref else msg, "error")
                    try:
                        from PySide6 import QtWidgets as _QW
                        _QW.QMessageBox.warning(self, "voicelang error", f"{msg}\n\nLog: {log_ref}")
                    except Exception:
                        pass
                try:
                    if self._proc in _ACTIVE_PROCS:
                        _ACTIVE_PROCS.remove(self._proc)
                except Exception:
                    pass
            except Exception:
                pass

        def _stop_run(self):
            try:
                if hasattr(self, "_poll_timer"):
                    try: self._poll_timer.stop()
                    except Exception: pass
            except Exception: pass
            try:
                if hasattr(self, "_elapsed_timer"):
                    try: self._elapsed_timer.stop()
                    except Exception: pass
            except Exception: pass
            try:
                if hasattr(self, "_meter_demo"):
                    try: self._meter_demo.stop()
                    except Exception: pass
            except Exception: pass
            try:
                if hasattr(self, "_log_file") and self._log_file:
                    try: self._log_file.close()
                    except Exception: pass
                    self._log_file = None
            except Exception: pass
            try:
                if self._proc and self._proc.poll() is None:
                    self._proc.terminate()
                    try: self._proc.wait(timeout=3)
                    except Exception: self._proc.kill()
                if self._proc in _ACTIVE_PROCS:
                    _ACTIVE_PROCS.remove(self._proc)
            except Exception:
                pass
            self._proc=None
            self._start.setEnabled(True); self._stop.setEnabled(False)
            self._status.setText("stopped")
            self._status.setStyleSheet("")
            self._setPillState("idle", "Idle")
            self._start_ts = None
            self._toast("Pipeline stopped", "info")

        def closeEvent(self, event):
            try: self._stop_run()
            except Exception: pass
            try: _cleanup_procs()
            except Exception: pass
            super().closeEvent(event)

    class _WorkerThread(QtCore.QThread):
        def __init__(self, worker): super().__init__(); self._worker=worker
        def run(self): self._worker.run()

    return SettingsWindow

def main():
    from PySide6 import QtCore, QtWidgets
    # High-DPI: must be set BEFORE QApplication is constructed
    try:
        # Qt5 compatibility: AA_EnableHighDpiScaling / AA_UseHighDpiPixmaps
        if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
            QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
            QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        # Qt6: HighDpiScaleFactorRoundingPolicy
        if hasattr(QtCore.Qt, "HighDpiScaleFactorRoundingPolicy"):
            try:
                QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
                    QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
            except Exception:
                pass
    except Exception:
        pass
    try:
        from voicelang_core.theme import apply_theme
        _use_theme = True
    except Exception:
        _use_theme = False
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if _use_theme:
        try:
            apply_theme(app, dark=True)
        except Exception:
            pass
    icon = _logo_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    window = _make_gui()(load_config())
    if icon is not None:
        window.setWindowIcon(icon)
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
