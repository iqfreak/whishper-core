"""Engine selection for the real-time voicelang pipeline.

Pick the ASR/translation engine at runtime via ``EngineConfig`` + ``build_pipeline``.
The factory returns a fully-wired ``Pipeline`` and the ``run_mode`` to call on it:

    pipeline, run_mode = build_pipeline(cfg)
    getattr(pipeline, f"run_{run_mode}")()

Two port compositions are supported (never mixed):

- **Fused ``CaptionSource`` path** (``openasr``): capture + ASR + translate-en in one
  local binary, exposed as a ``CaptionSource``. ``run_mode == "captions"``.
- **Composed ``AudioSource`` + ``StreamingTranscriber`` path**
  (``faster_whisper``, ``whisper_http``): local/remote capture feeds a streaming
  transcriber. ``run_mode == "streaming"``. ``Pipeline`` needs BOTH ``source``
  and ``transcriber`` set; the factory always provides them.

Heavy deps (faster-whisper, sounddevice, requests, openasr) are imported lazily
*inside* each branch, so ``import voicelang_core`` and the numpy-only test suite
stay dependency-free.

``translate`` selects the Translator: ``passthrough`` (Whisper already emits
English in one pass) or ``libretranslate`` (LibreTranslate HTTP adapter).
"""
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Literal, Optional


class EngineName(str, Enum):
    """Selectable engine at pipeline composition time.

    - openasr        : fused capture+ASR+translate-en in one local binary (Depth A).
    - faster_whisper : local faster-whisper behind an ``AudioSource`` + streaming transcriber.
    - whisper_http   : remote ASR over HTTP (the original voicelang ASR_ENDPOINT seam).
    - passthrough    : in-memory demo / test engine (uses fakes).
    - moonshine      : per-language tiny streaming (moonshine_voice).
    - nemotron       : sherpa-onnx nemotron streaming int8.
    - funasr         : FunASR Paraformer / SenseVoice (modelscope).
    - funasr_nano    : Fun-ASR-Nano 0.8B (zh/en/ja) via HF Transformers.
    - vosk           : Vosk offline streaming (small CPU models, vosk).
    """
    openasr = "openasr"
    faster_whisper = "faster-whisper"
    whisper = "whisper"  # alias for faster_whisper (GUI name)
    whisper_http = "whisper-http"
    passthrough = "passthrough"
    moonshine = "moonshine"
    nemotron = "nemotron"
    funasr = "funasr"
    funasr_nano = "funasr-nano"
    vosk = "vosk"
    vibeasr = "vibeasr"
    vibeasr_bitnet = "vibeasr-bitnet"
    audio8 = "audio8"


@dataclasses.dataclass(frozen=True)
class EngineConfig:
    """Immutable configuration for a single engine.

    Only the fields relevant to ``name`` are used; the rest are ignored.
    """
    name: EngineName

    # --- capture (composed path) ---
    capture: Literal["mic", "loopback", "app", "wav"] = "mic"
    source_device: str = ""  # loopback device name; "" = default
    source_app: str = ""     # per-app target e.g. "Discord.exe" or "pid:1234"
    input_file: str = ""     # wav replay path
    wav_path: str = ""       # alias for input_file
    capture_app: str = ""    # alias for source_app

    # --- OpenASR (fused path) ---
    openasr_model: str = "base"
    openasr_device: Literal["cpu", "cuda"] = "cuda"

    # --- faster-whisper (composed path) ---
    model_size: str = "small"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    compute_type: Literal["int8", "int8_float16", "float16"] = "int8"
    language: Optional[str] = None  # None = auto-detect
    task: Literal["transcribe", "translate"] = "translate"

    # --- remote whisper-http ---
    asr_endpoint: Optional[str] = None

    # --- translation axis (shared) ---
    translate: Literal["passthrough", "libretranslate", "deepl", "deeplx", "google"] = "passthrough"
    libretranslate_url: Optional[str] = None
    translate_url: Optional[str] = None
    translate_key: Optional[str] = None

    # --- display ---
    target_language: str = "en"
    display: Literal["console", "overlay"] = "console"



# --- Registry: name -> (builder, language catalog, capabilities) ---
# WP-3.1: single source so WP-4 just registers entries without editing builder code
ENGINE_REGISTRY: dict[str, dict] = {}

def _register_engine(name: str, builder, languages: list[str], capabilities: dict):
    ENGINE_REGISTRY[name] = {"builder": builder, "languages": languages, "capabilities": capabilities}

# Known engine language catalogs (mirrors adapters where applicable)
_ENGINE_LANGUAGES = {
    "moonshine": ["en", "zh", "fr", "de", "es", "ar", "ja", "ko", "ru", "pt", "it", "nl", "tr", "pl", "uk", "vi", "th", "hi", "tl", "ca", "el", "fa", "cs", "ro", "da", "sv", "auto"],
    "whisper": ["auto", "en", "zh", "fr", "de", "ja", "ko", "ru", "pt", "it", "nl", "tr", "pl", "uk", "vi", "th", "hi", "tl", "ar", "es"],
    "faster-whisper": ["auto", "en", "zh", "fr", "de", "ja", "ko", "ru", "pt", "it", "nl", "tr", "pl", "uk", "vi", "th", "hi", "tl", "ar", "es"],
    "nemotron": ["en"],  # streaming nemotron is en-only
    "funasr": ["zh", "en", "ja", "yue", "auto"],
    "funasr-nano": ["zh", "zh-cn", "en", "ja", "auto", "yue", "cantonese"],
    "vosk": ["en", "zh", "fr", "de", "es", "ar", "ja", "ko", "ru", "pt", "it", "nl", "tr", "pl", "uk", "vi", "th", "hi", "tl", "ca", "el", "fa", "cs", "ro", "da", "sv", "auto"],
    "openasr": ["en"],
    "whisper-http": ["auto", "en", "zh", "fr", "de", "ja", "ko"],
    "vibeasr": ["auto", "en", "zh", "fr", "it", "ko", "pt", "vi"],
    "vibeasr-bitnet": ["auto", "en", "zh", "fr", "it", "ko", "pt", "vi"],
    "audio8": ["auto", "zh", "en", "fr", "de", "ja", "ko", "yue"],
    "passthrough": ["auto", "en"],
}

_ENGINE_CAPS = {
    "moonshine": {"streaming": True, "gpu_preferred": False, "stateful": True},
    "whisper": {"streaming": False, "gpu_preferred": True, "stateful": False},
    "faster-whisper": {"streaming": False, "gpu_preferred": True, "stateful": False},
    "nemotron": {"streaming": True, "gpu_preferred": False, "stateful": True},
    "funasr": {"streaming": False, "gpu_preferred": False, "stateful": False},
    "funasr-nano": {"streaming": False, "gpu_preferred": False, "stateful": False},
    "vosk": {"streaming": True, "gpu_preferred": False, "stateful": True},
    "openasr": {"streaming": True, "gpu_preferred": True, "stateful": True},
    "whisper-http": {"streaming": False, "gpu_preferred": False, "stateful": False},
    "vibeasr": {"streaming": False, "gpu_preferred": False, "stateful": False},
    "vibeasr-bitnet": {"streaming": False, "gpu_preferred": False, "stateful": False},
    "audio8": {"streaming": False, "gpu_preferred": True, "stateful": False},
    "passthrough": {"streaming": False, "gpu_preferred": False, "stateful": False},
}

def get_registry():
    """Return copy of registry; populates lazily on first call.

    Now merges live HF catalog when available (TTL cache); falls back to
    hardcoded _ENGINE_LANGUAGES on network failure so UI never empties.
    """
    if not ENGINE_REGISTRY:
        # Build base from hardcoded defaults
        base_langs = dict(_ENGINE_LANGUAGES)
        # Attempt to enrich from live catalog (never crashes)
        try:
            from .model_catalog import get_provider_catalog  # lazy to avoid circular
            for name in list(base_langs.keys()):
                prov = name
                # alias handling: engines uses "faster-whisper" but catalog uses "whisper"
                if prov == "faster-whisper":
                    prov = "whisper"
                try:
                    langs, _friendly, arches, _hint, _dl, _offline = get_provider_catalog(prov)
                    if langs:
                        base_langs[name] = list(langs)
                        # also store arches for GUI consistency
                        # keep in ENGINE_REGISTRY's arches slot via capabilities merge
                        # but at minimum update language list
                        # arches stored separately if needed by callers
                except Exception:
                    pass
        except Exception:
            pass
        for name in base_langs:
            _register_engine(name, None, base_langs[name], _ENGINE_CAPS.get(name, {}))
        # Also attach arches to registry entries for callers that need them
        try:
            from .model_catalog import OFFLINE_DEFAULTS
            for name in ENGINE_REGISTRY:
                prov = "whisper" if name == "faster-whisper" else name
                if prov in OFFLINE_DEFAULTS:
                    ENGINE_REGISTRY[name]["arches"] = OFFLINE_DEFAULTS[prov]["arches"]
                    # override with live arches if fetched
                    try:
                        from .model_catalog import get_provider_catalog as _gpc
                        _langs, _fr, _arches, _h, _dl, _off = _gpc(prov)
                        if _arches:
                            ENGINE_REGISTRY[name]["arches"] = list(_arches)
                    except Exception:
                        pass
        except Exception:
            pass
    return dict(ENGINE_REGISTRY)

def validate_transcriber(name: str) -> str:
    """WP-3.3: validate against registry; unknown -> warning + fallback."""
    if not ENGINE_REGISTRY:
        get_registry()
    n = (name or "").lower().strip()
    if n in ENGINE_REGISTRY:
        return n
    # aliases
    aliases = {"faster_whisper": "faster-whisper", "whisper_http": "whisper-http", "funasr_nano": "funasr-nano"}
    if n in aliases and aliases[n] in ENGINE_REGISTRY:
        return aliases[n]
    import sys
    print(f"[voicelang] warning: unknown transcriber '{name}' — falling back to 'moonshine'", file=sys.stderr)
    return "moonshine"

def _build_translator(cfg: EngineConfig):
    """Resolve the Translator port from ``cfg.translate``."""
    from voicelang_core.adapters.passthrough import PassthroughTranslator

    mode = (cfg.translate or "passthrough").lower()
    endpoint = cfg.translate_url or cfg.libretranslate_url
    if mode == "libretranslate":
        from voicelang_core.adapters.libretranslate import LibreTranslateTranslator
        # Agent4 fix: misconfigured URL must surface as clear error, not silent passthrough
        if not (endpoint or "").strip():
            raise SystemExit("Translation is set to libretranslate but no API key/URL configured — set translate_url (e.g. http://127.0.0.1:5000) in config or --translate-url")
        return LibreTranslateTranslator(endpoint=endpoint)
    if mode == "deepl":
        from voicelang_core.adapters.deepl import DeepLTranslator
        # WP-3.1: port run.py guard — missing key is hard fail
        if not (cfg.translate_key or "").strip():
            raise SystemExit("error: deepl requires --translate-key (or translate_key in config) — get a key at https://www.deepl.com/pro-api")
        return DeepLTranslator(endpoint=endpoint or "https://api-free.deepl.com",
                               api_key=cfg.translate_key or "")
    if mode == "deeplx":
        from voicelang_core.adapters.deeplx import DeepLXTranslator
        if not (endpoint or "").strip():
            raise SystemExit("Translation is set to deeplx but no API key/URL configured — set translate_url (e.g. http://127.0.0.1:1188) in config or --translate-url")
        return DeepLXTranslator(endpoint=endpoint)
    if mode == "google":
        from voicelang_core.adapters.google_translate import GoogleTranslator

        return GoogleTranslator(endpoint=endpoint or "https://translate.googleapis.com")
    return PassthroughTranslator()


def _build_capture_source(cfg: EngineConfig):
    """Resolve the ``AudioSource`` for the composed path from ``cfg.capture``."""
    caps = getattr(cfg, "capture", "mic")
    if caps == "loopback":
        from voicelang_core.adapters.wasapi_loopback import WASAPILoopbackSource
        dev = getattr(cfg, "source_device", "") or ""
        return WASAPILoopbackSource(device=dev or None)
    if caps == "app":
        try:
            from voicelang_core.adapters.app_loopback import AppLoopbackSource
            app_sel = getattr(cfg, "source_app", "") or getattr(cfg, "capture_app", "") or ""
            return AppLoopbackSource(app=app_sel)
        except Exception:
            from voicelang_core.adapters.wasapi_loopback import WASAPILoopbackSource
            dev = getattr(cfg, "source_device", "") or ""
            return WASAPILoopbackSource(device=dev or None)
    if caps == "wav":
        from voicelang_core.adapters.wav_file import WavFileSource
        src_file = getattr(cfg, "input_file", None) or getattr(cfg, "wav_path", None) or ""
        if src_file:
            return WavFileSource(path=src_file)
        # fallback to mic if no file configured
        from voicelang_core.adapters.mic import MicSource
        return MicSource()
    from voicelang_core.adapters.mic import MicSource
    return MicSource()


def build_pipeline(cfg: EngineConfig):
    """Build and return ``(Pipeline, run_mode)`` for *cfg*.

    Caller invokes the returned run mode::

        pipeline, run_mode = build_pipeline(cfg)
        getattr(pipeline, f"run_{run_mode}")()

    run_mode is one of ``"captions"``, ``"streaming"``, ``"batch"``.
    """
    # Normalize GUI alias "whisper" -> canonical "faster-whisper"
    if cfg.name == EngineName.whisper:
        cfg = dataclasses.replace(cfg, name=EngineName.faster_whisper)

    from voicelang_core.pipeline import Pipeline
    from voicelang_core.adapters.console import ConsoleDisplay

    if cfg.display == "overlay":
        from voicelang_core.adapters.overlay import OverlayDisplay

        display = OverlayDisplay()
    else:
        display = ConsoleDisplay()

    if cfg.name == EngineName.openasr:
        from voicelang_core.adapters.openasr_source import OpenASRSource

        src = OpenASRSource(
            binary="openasr",
            model=cfg.openasr_model,
            capture=cfg.capture,
            device=cfg.openasr_device,
        )
        pipeline = Pipeline(
            source=src,
            translator=_build_translator(cfg),
            display=display,
            target_language=cfg.target_language,
        )
        return pipeline, "captions"

    if cfg.name == EngineName.faster_whisper:
        from voicelang_core.adapters.whisper_streaming import (
            WhisperStreamingTranscriber,
        )

        audio_src = _build_capture_source(cfg)
        transcriber = WhisperStreamingTranscriber(
            model_size=cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
            translate=cfg.task == "translate",
        )
        pipeline = Pipeline(
            source=audio_src,
            transcriber=transcriber,
            translator=_build_translator(cfg),
            display=display,
            target_language=cfg.target_language,
        )
        return pipeline, "streaming"

    if cfg.name == EngineName.whisper_http:
        if not cfg.asr_endpoint:
            msg = "whisper_http engine requires asr_endpoint"
            raise ValueError(msg)
        from voicelang_core.adapters.remote_whisper import RemoteWhisperTranscriber

        audio_src = _build_capture_source(cfg)
        transcriber = RemoteWhisperTranscriber(
            asr_endpoint=cfg.asr_endpoint,
            language=cfg.language,
        )
        pipeline = Pipeline(
            source=audio_src,
            transcriber=transcriber,
            translator=_build_translator(cfg),
            display=display,
            target_language=cfg.target_language,
        )
        return pipeline, "streaming"

    if cfg.name in (EngineName.moonshine, EngineName.nemotron, EngineName.funasr, EngineName.vosk):
        audio_src = _build_capture_source(cfg)
        if cfg.name == EngineName.moonshine:
            from voicelang_core.adapters.moonshine import MoonshineTranscriber

            # "auto" (Config default) is not in Moonshine's per-language
            # catalog — the adapter maps it to English, mirrored here.
            moon_lang = cfg.language if cfg.language and cfg.language != "auto" else "en"
            transcriber = MoonshineTranscriber(language=moon_lang, model_arch=getattr(cfg, "model_arch", getattr(cfg, "model_size", "TINY_STREAMING")), update_interval=getattr(cfg, "moonshine_update_interval", 0.08))
        elif cfg.name == EngineName.nemotron:
            from voicelang_core.adapters.nemotron import NemotronStreamingTranscriber
            from voicelang_core.config import nemotron_model_dir

            transcriber = NemotronStreamingTranscriber(model_dir=nemotron_model_dir())
        elif cfg.name == EngineName.vosk:
            from voicelang_core.adapters.vosk import VoskTranscriber

            transcriber = VoskTranscriber(language=cfg.language or "en")
        else:  # funasr
            from voicelang_core.adapters.funasr import FunASRTranscriber

            transcriber = FunASRTranscriber(language=cfg.language or "auto")
        pipeline = Pipeline(
            source=audio_src,
            transcriber=transcriber,
            translator=_build_translator(cfg),
            display=display,
            target_language=cfg.target_language,
        )
        return pipeline, "streaming"

    if cfg.name == EngineName.funasr_nano:
        from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber

        audio_src = _build_capture_source(cfg)
        transcriber = FunAsrNanoTranscriber(language=cfg.language or "auto")
        pipeline = Pipeline(
            source=audio_src,
            transcriber=transcriber,
            translator=_build_translator(cfg),
            display=display,
            target_language=cfg.target_language,
        )
        return pipeline, "streaming"

    if cfg.name == EngineName.passthrough:
        from voicelang_core.adapters.fake import (
            FakeAudioSource, FakeDisplay, FakeTranscriber,
        )

        chunks = [
            type("Chunk", (), {"pcm": b"\x00" * 16000, "sample_rate": 16000})()
            for _ in range(6)
        ]
        audio_src = FakeAudioSource(chunks=chunks)
        transcriber = FakeTranscriber(final_text="hello world", language="en")
        pipeline = Pipeline(
            source=audio_src,
            transcriber=transcriber,
            translator=_build_translator(cfg),
            display=display,
            target_language=cfg.target_language,
        )
        return pipeline, "batch"

    msg = f"Unknown engine name: {cfg.name}"
    raise ValueError(msg)
