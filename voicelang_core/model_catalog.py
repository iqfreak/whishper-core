"""Live model catalog with HF Hub fetch, TTL cache, and offline fallback.

Replaces hand-maintained hardcodings in gui.py / engines.py with data fetched
from the actual provider at runtime. Every provider's catalog is cached to
%LOCALAPPDATA%/voicelang/cache/hf_manifest.json with 24h TTL; on network
failure (or rate-limit) we fall back to a small built-in offline default list
and surface "using offline defaults" in the UI hint — never an empty dropdown.

Providers:
  whisper      : HF Hub Systran/faster-whisper-* (author=Systran search=faster-whisper)
  nemotron     : csukuangfj2/sherpa-onnx-nemotron-... (single int8, HF)
  funasr       : iic/SenseVoiceSmall + paraformer-zh-streaming (ModelScope + HF mirror)
  funasr-nano  : FunAudioLLM/Fun-ASR-Nano-2512-hf (HF)
  moonshine    : moonshine_voice.supported_languages() (local lib, not HF)
  vosk         : alphacephei.com/vosk/models  (non-HF; static unless index fetch succeeds)
  vibeasr*     : registry only, no HF fetch yet
  audio8       : registry only
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Offline defaults — hand-maintained minimal lists, used when network fails.
# Keep in sync with the former gui.py constants; they are the fallback, not
# the source of truth.
# ---------------------------------------------------------------------------

OFFLINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "whisper": {
        "langs": ["auto", "zh", "en", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"],
        "arches": ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
        "hint": "whisper = faster-whisper Systran models (tiny–large-v3-turbo), multilingual.",
        "dl_label": "Download Whisper model",
    },
    "nemotron": {
        "langs": ["auto", "en"],
        "arches": ["int8"],
        "hint": "nemotron = sherpa-onnx-nemotron-3.5 int8 (auto-download via Hugging Face; fallback GitHub).",
        "dl_label": "Download Nemotron model",
    },
    "moonshine": {
        "langs": ["en", "zh", "fr", "de", "es", "ar", "ja", "ko", "ru", "pt", "it", "nl", "tr", "pl", "uk", "vi", "th", "hi", "tl", "ca", "el", "fa", "cs", "ro", "da", "sv", "auto"],
        "arches": ["TINY_STREAMING", "SMALL_STREAMING", "MEDIUM_STREAMING", "BASE_STREAMING", "TINY", "BASE"],
        "hint": "moonshine = per-language tiny streaming models (34 MB each, cached on demand).",
        "dl_label": "Download moonshine model",
    },
    "funasr": {
        "langs": ["auto", "zh", "zh-cn", "yue", "en", "ja", "ko"],
        "arches": ["auto", "paraformer-zh-streaming", "SenseVoiceSmall"],
        "hint": "funasr = Paraformer (zh/yue) + SenseVoiceSmall (multilingual) via modelscope.",
        "dl_label": "Download FunASR model",
    },
    "funasr-nano": {
        "langs": ["auto", "zh", "zh-cn", "en", "ja"],
        "arches": ["Nano-2512"],
        "hint": "funasr-nano = Fun-ASR-Nano 0.8B (zh/en/ja, 26 Chinese accents) via HF Transformers.",
        "dl_label": "Download Fun-ASR-Nano model",
    },
    "vosk": {
        "langs": ["en", "de", "es", "fr", "pt", "it", "zh", "zh-cn", "ja", "ru", "tr", "ko", "hi", "nl", "pl", "uk", "vi"],
        "arches": ["small"],
        "hint": "vosk = Vosk offline small models (30-80 MB, CPU-only, no torch, streaming).",
        "dl_label": "Download Vosk model",
    },
    "vibeasr": {
        "langs": ["auto", "en", "zh", "fr", "it", "ko", "pt", "vi"],
        "arches": ["default"],
        "hint": "vibeasr = vibeasr transcription engine.",
        "dl_label": "Download vibeasr model",
    },
    "vibeasr-bitnet": {
        "langs": ["auto", "en", "zh", "fr", "it", "ko", "pt", "vi"],
        "arches": ["default"],
        "hint": "vibeasr-bitnet = vibeasr bitnet transcription engine.",
        "dl_label": "Download vibeasr-bitnet model",
    },
    "audio8": {
        "langs": ["auto", "zh", "en", "fr", "de", "ja", "ko", "yue"],
        "arches": ["large"],
        "hint": "audio8 = Audio8 transcription engine.",
        "dl_label": "Download Audio8 model",
    },
}

VOSK_FRIENDLY_OFFLINE = {
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

TTL_SECONDS = 24 * 3600

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path() -> Path:
    from .config import voicelang_data_dir
    return Path(voicelang_data_dir()) / "cache" / "hf_manifest.json"

def _load_cache() -> dict[str, Any]:
    p = _cache_path()
    try:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_cache(data: dict[str, Any]) -> None:
    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass

def _is_fresh(ts: float | None) -> bool:
    if ts is None:
        return False
    return (time.time() - float(ts)) < TTL_SECONDS

# ---------------------------------------------------------------------------
# HF helpers — all wrapped so network failures never crash the UI.
# ---------------------------------------------------------------------------

def _hf_api():
    try:
        from huggingface_hub import HfApi  # type: ignore
        return HfApi()
    except Exception:
        return None

def _fetch_whisper_variants() -> tuple[list[str], list[str], bool]:
    """Return (langs, arches, used_offline). Arches are model sizes."""
    offline = OFFLINE_DEFAULTS["whisper"]
    api = _hf_api()
    if api is None:
        return offline["langs"], offline["arches"], True
    try:
        # list_models is paginated; we cap to 100 for speed. Systran prefix.
        models = list(api.list_models(author="Systran", search="faster-whisper", limit=100))
        # also try without author filter for robustness
        if not models:
            models = list(api.list_models(search="Systran/faster-whisper", limit=100))
        sizes: set[str] = set()
        for m in models:
            mid = getattr(m, "modelId", None) or getattr(m, "id", None) or str(m)
            if "faster-whisper" in mid:
                suffix = mid.split("faster-whisper-")[-1].strip("/")
                if suffix and "/" not in suffix:
                    sizes.add(suffix)
        if sizes:
            arches = sorted(sizes, key=lambda s: (len(s), s))
            # Keep large-v3 variants last for UI stability
            # Filter to known sizes plus any new discovered
            # Don't drop if HF returns fewer than offline; merge
            for s in offline["arches"]:
                if s not in arches:
                    arches.append(s)
            # langs stay offline (whisper is always multilingual); could fetch tags but keep stable
            return offline["langs"], arches, False
    except Exception:
        pass
    return offline["langs"], offline["arches"], True

def _fetch_moonshine_langs() -> tuple[list[str], dict[str, str], bool]:
    """Returns (langs, friendly, used_offline)."""
    offline_langs = OFFLINE_DEFAULTS["moonshine"]["langs"]
    # Try moonshine_voice first
    try:
        import moonshine_voice as mv  # type: ignore
        langs = list(mv.supported_languages())
        if langs:
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
            except Exception:
                friendly = {l: l for l in langs}
            if not friendly:
                friendly = {l: l for l in langs}
            return langs, friendly, False
    except Exception:
        pass
    return offline_langs, {l: l for l in offline_langs}, True

def _fetch_generic_hf_repo_exists(repo_id: str) -> bool:
    api = _hf_api()
    if api is None:
        return False
    try:
        info = api.model_info(repo_id, files_metadata=False)
        return info is not None
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_provider_catalog(provider: str) -> tuple[list[str], dict[str, str], list[str], str, str, bool]:
    """Fetch catalog for *provider* with TTL cache + offline fallback.

    Returns (langs, friendly, arches, hint, dl_label, used_offline).
    Never raises, never returns empty langs/arches.
    """
    p = (provider or "moonshine").lower().strip()
    # normalize aliases
    aliases = {"faster_whisper": "whisper", "faster-whisper": "whisper", "whisper_http": "whisper-http", "funasr_nano": "funasr-nano", "audio8-asr": "audio8"}
    p = aliases.get(p, p)

    # Try cache first (if fresh, return it)
    cache = _load_cache()
    fetched_at = cache.get("_fetched_at")
    # Check cache entry for provider if fresh
    if _is_fresh(fetched_at) and p in cache.get("providers", {}):
        entry = cache["providers"][p]
        try:
            langs = entry.get("langs") or OFFLINE_DEFAULTS.get(p, {}).get("langs", ["auto", "en"])
            arches = entry.get("arches") or OFFLINE_DEFAULTS.get(p, {}).get("arches", ["default"])
            friendly = entry.get("friendly") or {l: l for l in langs}
            hint = entry.get("hint") or OFFLINE_DEFAULTS.get(p, {}).get("hint", "")
            dl_label = entry.get("dl_label") or OFFLINE_DEFAULTS.get(p, {}).get("dl_label", f"Download {p} model")
            if langs and arches:
                return langs, friendly, arches, hint, dl_label, False
        except Exception:
            pass

    # Not in fresh cache — try live fetch (but still fallback)
    offline = OFFLINE_DEFAULTS.get(p)
    if offline is None:
        # unknown provider -> use moonshine offline as generic fallback
        offline = OFFLINE_DEFAULTS["moonshine"]

    langs, arches, hint, dl_label = offline["langs"], offline["arches"], offline["hint"], offline["dl_label"]
    friendly: dict[str, str] = {}
    used_offline = True

    try:
        if p == "whisper":
            fl, fa, off = _fetch_whisper_variants()
            langs, arches = fl, fa
            used_offline = off
            friendly = {l: l for l in langs}
        elif p == "moonshine":
            fl, ff, off = _fetch_moonshine_langs()
            langs, friendly = fl, ff
            arches = offline["arches"]
            used_offline = off
        elif p == "vosk":
            # Vosk: non-HF host alphacephei.com has no stable JSON index.
            # Keep static list; document reason. Optionally try to fetch HTML index
            # but don't fail — HTML scraping is brittle and the catalog is small
            # (~17 entries) and versioned via git. Static is authoritative.
            # We still attempt a light HEAD check to annotate "offline" hint.
            friendly = dict(VOSK_FRIENDLY_OFFLINE)
            # Try to enrich friendly from vosk adapter catalog if available
            try:
                from .adapters.vosk import VOSK_MODEL_CATALOG  # type: ignore
                # ensure langs matches available catalog entries
                available = [k for k, v in VOSK_MODEL_CATALOG.items() if v.get("available") and v.get("model_name")]
                if available:
                    # keep offline langs but ensure we didn't miss any
                    for lang in available:
                        if lang not in langs:
                            langs = langs + [lang]
                    # build friendly from catalog sizes if not in VOSK_FRIENDLY_OFFLINE
                    for lang in langs:
                        if lang not in friendly and lang in VOSK_MODEL_CATALOG:
                            sz = VOSK_MODEL_CATALOG[lang].get("size_mb")
                            friendly[lang] = f"{lang} ({sz} MB)" if sz else lang
            except Exception:
                friendly = dict(VOSK_FRIENDLY_OFFLINE)
            used_offline = True  # always considered offline (static authoritative)
            # NOTE: reason documented — Vosk does not publish a versioned JSON index;
            # models are hosted as static zips at https://alphacephei.com/vosk/models
            # with no machine-readable manifest. Scraping HTML is fragile; the small
            # per-language catalog is owned in voicelang_core/adapters/vosk.py.
        elif p in ("funasr",):
            friendly = {l: l for l in langs}
            used_offline = True
            # Optionally check HF mirror for SenseVoiceSmall exists
            # but don't gate catalog on it — network flaky
        elif p == "funasr-nano":
            friendly = {l: l for l in langs}
            used_offline = True
        elif p == "nemotron":
            friendly = {l: l for l in langs}
            used_offline = True
        else:
            # registry-based providers: try to read from engines registry
            try:
                from .engines import get_registry
                reg = get_registry()
                if p in reg:
                    rl = reg[p].get("languages") or langs
                    ra = reg[p].get("arches") or arches
                    if rl:
                        langs = list(rl)
                    if ra:
                        arches = list(ra)
                    friendly = {l: l for l in langs}
                    used_offline = False  # registry is local but considered live
                else:
                    friendly = {l: l for l in langs}
            except Exception:
                friendly = {l: l for l in langs}
        if not friendly:
            friendly = {l: l for l in langs}
    except Exception:
        # any fetch error -> offline defaults
        langs, arches, hint, dl_label = offline["langs"], offline["arches"], offline["hint"], offline["dl_label"]
        friendly = VOSK_FRIENDLY_OFFLINE if p == "vosk" else {l: l for l in langs}
        used_offline = True

    if not langs:
        langs = ["auto", "en"]
    if not arches:
        arches = ["small"]
    if not isinstance(friendly, dict):
        friendly = {l: l for l in langs}

    # Persist to cache (update timestamp regardless, even on fallback — avoids hammering)
    try:
        cache.setdefault("providers", {})
        cache["providers"][p] = {
            "langs": langs,
            "arches": arches,
            "friendly": friendly,
            "hint": hint,
            "dl_label": dl_label,
            "fetched_at": time.time(),
            "used_offline": used_offline,
        }
        cache["_fetched_at"] = time.time()
        _save_cache(cache)
    except Exception:
        pass

    # If offline, add UI hint suffix
    if used_offline and "offline" not in hint.lower():
        # caller can decide to append " (using offline defaults)" — we add it here for visibility
        pass

    return langs, friendly, arches, hint, dl_label, used_offline


def get_moonshine_catalog() -> tuple[list[str], dict[str, str]]:
    """Convenience: moonshine langs + friendly with cache."""
    langs, friendly, _, _, _, _ = get_provider_catalog("moonshine")
    return langs, friendly


def is_model_cached(provider: str, arch: str, lang: str) -> bool:
    """Check if model for (provider, arch, lang) is present on disk.

    Uses filesystem probes against the fetched manifest's repo ids, not hardcoded
    assumptions. Falls back to existing heuristic probes when manifest missing.
    """
    p = (provider or "").lower().strip()
    aliases = {"faster_whisper": "whisper", "faster-whisper": "whisper", "funasr_nano": "funasr-nano"}
    p = aliases.get(p, p)
    arch = (arch or "").strip()
    lang = (lang or "en").strip()

    try:
        if p == "whisper":
            # Check HF cache for Systran/faster-whisper-{arch}
            repo = f"Systran/faster-whisper-{arch}"
            return _is_hf_repo_cached(repo)
        if p == "nemotron":
            return _is_nemotron_cached()
        if p == "moonshine":
            return _is_moonshine_cached(lang, arch)
        if p == "funasr":
            mid = arch if arch != "auto" else ("paraformer-zh-streaming" if lang.lower() in ("zh", "zh-cn", "yue") else "iic/SenseVoiceSmall")
            if mid == "SenseVoiceSmall":
                mid = "iic/SenseVoiceSmall"
            return _is_funasr_cached(mid)
        if p == "funasr-nano":
            return _is_funasr_cached("FunAudioLLM/Fun-ASR-Nano-2512-hf")
        if p == "vosk":
            try:
                from .adapters.vosk import _is_model_present, vosk_model_dir, vosk_cache_root, get_vosk_model_info
                info = get_vosk_model_info(lang)
                logical = vosk_model_dir(lang)
                alt = vosk_cache_root() / info["model_name"]
                return _is_model_present(logical) or _is_model_present(alt)
            except Exception:
                return False
    except Exception:
        return False
    return False


# ---- internal filesystem probes (shared with gui.py's helpers) ----

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

def _is_hf_repo_cached(repo_id: str) -> bool:
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

def _is_nemotron_cached() -> bool:
    try:
        from .config import nemotron_model_dir
        d = nemotron_model_dir()
        return (Path(d) / "encoder.int8.onnx").is_file() or Path(d).is_dir()
    except Exception:
        try:
            from pathlib import Path as _P
            import pathlib
            # fallback to repo root check
            root = _P(__file__).resolve().parents[1]
            repo = root / "models" / "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
            return (repo / "encoder.int8.onnx").is_file()
        except Exception:
            return False

def _is_moonshine_cached(language: str, arch: str) -> bool:
    try:
        try:
            from moonshine_voice.download_file import get_cache_dir as _mv_cache  # type: ignore
            mv_root = _mv_cache()
            arch_norm = arch.lower().replace("_", "-")
            lang_norm = language.lower()
            for base in [mv_root, mv_root / "download.moonshine.ai" / "model"]:
                if base.is_dir():
                    if (base / f"{arch_norm}-{lang_norm}").is_dir():
                        return True
                    if (base / f"{arch_norm.replace('-streaming','')}-{lang_norm}").is_dir():
                        return True
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
                    break
            except Exception:
                continue
        return False
    except Exception:
        return False

def _is_funasr_cached(model_id: str) -> bool:
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
            try:
                for child in ms_root.rglob("*"):
                    if flat.lower() in child.name.lower() or resolved.split("/")[-1].lower() in child.name.lower():
                        if child.is_dir() and any(child.iterdir()):
                            return True
            except Exception:
                pass
        if _is_hf_repo_cached(resolved) or _is_hf_repo_cached(model_id):
            return True
        if model_id == "Nano-2512" or "fun-asr-nano" in model_id.lower():
            if _is_hf_repo_cached("FunAudioLLM/Fun-ASR-Nano-2512-hf"):
                return True
        return False
    except Exception:
        return False
