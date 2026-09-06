"""Live real-time demo: mic (or WAV file) -> ASR -> translate -> console/overlay.

Engines:
  --transcriber whisper   faster-whisper small (GPU/CUDA auto, default)
  --transcriber nemotron  Nemotron-3.5 streaming (sherpa-onnx, CPU int8)
  --transcriber moonshine per-language tiny streaming models (34M), downloaded
                 on demand from the catalog; pick --language en/ar/de/es/ja/
                 ko/tl/uk/vi/zh

Config: reads %LOCALAPPDATA%/voicelang/config.json when present (or --config
PATH). Explicit CLI flags always win over the file. The settings GUI
(``--gui``) writes the same file.

--display overlay: transparent topmost caption window (Tier A: windowed /
borderless only; EXCLUSIVE FULLSCREEN is out of scope -- anti-cheat risk).
Overlay placement (x/y/width/height/opacity/max_lines) comes from config.

Headless parity (spec §8): every GUI setting has a CLI flag, plus
--list-devices / --list-engines for discovery. --headless forces the CLI
path when invoked through the packaged exe (voicelang_app.py routes it).
"""
import argparse
import os
import sys

from .engines import build_pipeline  # composition root (registry)
from .config import (Config, TRANSLATE_MODES, SOURCES, SUPPORTED_LANGS, DISPLAY_KINDS,
                     config_path, load_config, DEFAULT_MOONSHINE_UPDATE_INTERVAL,
                     DEFAULT_WHISPER_WARMUP_S)
from .pipeline import Pipeline
from .adapters.whisper_streaming import WhisperStreamingTranscriber
from .adapters.console import ConsoleDisplay
from .adapters.overlay import OverlayDisplay


def _build_display(kind: str, cfg: Config):
    from voicelang_core.adapters.console import ConsoleDisplay as _Console
    base = None
    if kind == "overlay":
        ov = cfg.overlay
        # Respect auto_place: when auto_place True, ignore manual x/y/width/height
        # and let overlay compute bottom-center. This fixes invisible off-screen
        # window when user toggled auto_place but config still held stale coords.
        if getattr(ov, "auto_place", True):
            gx = gy = gw = gh = None
        else:
            gx, gy, gw, gh = ov.x, ov.y, ov.width, ov.height
        d = OverlayDisplay(
            max_lines=ov.max_lines, opacity=ov.opacity,
            x=gx, y=gy, width=gw, height=gh,
            show_transcription_block=ov.show_transcription_block,
            show_translation_block=ov.show_translation_block,
        )
        if d.headless:
            # Loud failure: exit code 30 distinct from generic 10/11/20 so GUI
            # can map to a specific remediation hint. Message goes to stderr
            # which Agent7 pipes to the log file (errors=replace).
            msg = (
                "Overlay requested but PySide6 is unavailable (no window opened).\n"
                "Install it:  uv sync --extra real --extra overlay\n"
                "Or use the console sink (default). \n"
                "Check QT_QPA_PLATFORM and display server on this host."
            )
            print(msg, file=sys.stderr, flush=True)
            sys.exit(30)
        base = d
    elif kind == "file":
        from voicelang_core.adapters.file_display import FileDisplay as _FD
        base = _FD(jsonl_path=cfg.log_jsonl or None, txt_path=cfg.log_txt or None)
        print(f"File sink -> {base.stats['jsonl']} + {base.stats['txt']}")
        return base
    else:
        base = _Console()
    # Tee to file sink when --log-captions (overlay+file or console+file)
    if getattr(cfg, "log_captions", False) and kind != "file":
        try:
            from voicelang_core.adapters.file_display import FileDisplay as _FD, MultiDisplay as _MD
            fd = _FD(jsonl_path=cfg.log_jsonl or None, txt_path=cfg.log_txt or None)
            print(f"File sink tee -> {fd.stats['jsonl']} + {fd.stats['txt']}")
            return _MD([base, fd])
        except Exception as e:
            print(f"[voicelang] file sink unavailable ({e}) -> {kind} only")
            return base
    return base


def _build_source(source: str, input_file: str | None, source_device: str = "",
                  source_app: str = "", block_seconds: float | None = None):
    bs = block_seconds or 1.0
    if source == "wav":
        if not input_file:
            raise SystemExit("--source wav requires --input WAV (or input_file in config)")
        from .adapters.wav_file import WavFileSource

        src = WavFileSource(input_file, block_seconds=bs)
        print(f"Playing {input_file} ({src.duration_seconds:.1f}s) -> captions...")
        return src
    if source == "app":
        # Per-app capture: like Discord screenshare source picker.
        # Tries process-loopback (Win 10 20348+); falls back to device-level loopback if unavailable.
        try:
            from .adapters.app_loopback import AppLoopbackSource, list_audio_apps
            if not source_app:
                # No app selected yet - list what we can capture so the error is actionable.
                apps = list_audio_apps()
                names = ", ".join(a["name"] for a in apps[:8]) or "no apps with audio sessions found"
                raise SystemExit(
                    f"--source app requires --source-app (e.g. 'Discord.exe' or 'pid:1234').\n"
                    f"Running audio apps now: {names}"
                )
            print(f"Capturing app {source_app!r} (per-app loopback; fallback to device mix if unsupported) ... (Ctrl-C to stop)")
            return AppLoopbackSource(app=source_app, block_seconds=bs)
        except SystemExit:
            raise
        except ImportError as exc:
            raise SystemExit(
                "Per-app capture needs pycaw + comtypes:  uv sync --extra real\n"
                f"({exc})") from exc
    if source == "loopback":
        from .adapters.wasapi_loopback import WASAPILoopbackSource

        print("Listening to the output device%s (WASAPI loopback)... "
              "(Ctrl-C to stop)" % (f" {source_device!r}" if source_device else ""))
        return WASAPILoopbackSource(device=source_device or None, block_seconds=bs)
    from .adapters.mic import MicSource

    # --source-device also selects the MIC (index or name) — the mic picker
    # uses the same config field as loopback device selection.
    return MicSource(block_seconds=bs, device=source_device or None)


def _nemotron_model_dir() -> str:
    """Back-compat shim — canonical impl lives in config.nemotron_model_dir."""
    from .config import nemotron_model_dir as _canonical
    return _canonical()


def _build_translator(cfg: Config):
    """Canonical translator factory — delegates to engines._build_translator."""
    from .adapters.passthrough import PassthroughTranslator as _PT
    from .engines import EngineConfig, EngineName
    from voicelang_core.engines import _build_translator as _eng_build
    mode = (cfg.translate or "passthrough").lower()
    if mode == "libretranslate" and not (cfg.translate_url or "").strip():
        raise SystemExit("Translation is set to libretranslate but no API key/URL configured — set translate_url (e.g. http://127.0.0.1:5000) in config or --translate-url")
    if mode == "deeplx" and not (cfg.translate_url or "").strip():
        raise SystemExit("Translation is set to deeplx but no API key/URL configured — set translate_url (e.g. http://127.0.0.1:1188) in config or --translate-url")
    if mode == "deepl" and not cfg.translate_key:
        raise SystemExit("translate=deepl requires translate_key (DeepL API key) in config or --translate-key")
    eng_cfg = EngineConfig(
        name=EngineName.passthrough,
        translate=cfg.translate or "passthrough",
        translate_url=cfg.translate_url or None,
        translate_key=cfg.translate_key or None,
        target_language=cfg.target_language,
    )
    return _eng_build(eng_cfg)


def _build_transcriber(name: str, cfg: Config):
    if name == "nemotron":
        from .adapters.nemotron import NemotronStreamingTranscriber

        return NemotronStreamingTranscriber(model_dir=_nemotron_model_dir())
    if name == "moonshine":
        from .adapters.moonshine import MoonshineTranscriber

        return MoonshineTranscriber(language=cfg.language, model_arch=cfg.model_arch,
                                    update_interval=getattr(cfg, "moonshine_update_interval", DEFAULT_MOONSHINE_UPDATE_INTERVAL))
    if name == "vosk":
        from .adapters.vosk import VoskTranscriber

        return VoskTranscriber(language=cfg.language or "en")
    if name in ("vibeasr", "vibeasr-bitnet", "vibeasr_bitnet"):
        from .adapters.vibeasr import VibeAsrTranscriber
        return VibeAsrTranscriber(language=cfg.language or "en")
    if name in ("audio8", "audio8-asr", "audio8_asr"):
        from .adapters.audio8 import Audio8Transcriber
        return Audio8Transcriber(language=cfg.language or "en")
    if name == "funasr":
        from .adapters.funasr import FunASRTranscriber

        # cfg.model_arch is shared with moonshine/whisper; only respect funasr arches
        funasr_arches = {"paraformer-zh-streaming", "SenseVoiceSmall"}
        if cfg.model_arch in funasr_arches:
            model = cfg.model_arch
        else:
            model = None  # auto -> language routing (zh/yue -> paraformer, else SenseVoice)
        return FunASRTranscriber(language=cfg.language, model=model)
    if name == "funasr-nano":
        from .adapters.funasr_nano import FunAsrNanoTranscriber

        # 0.8B seq2seq (zh/zh-cn/en/ja) via HF Transformers; the adapter
        # energy-buffers utterances and runs generate() once per utterance.
        return FunAsrNanoTranscriber(language=cfg.language)
    # whisper: cfg.model_arch carries the model size (tiny..large-v3-turbo) from the GUI catalog
    whisper_sizes = {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "large-v2", "large"}
    size = cfg.model_arch if cfg.model_arch in whisper_sizes else "small"
    return WhisperStreamingTranscriber(model_size=size)


def _finish_tail(pipeline: "Pipeline") -> None:
    """Flush the ASR tail after a finite replay (file input).

    The last partial becomes a final segment when audio ends without a pause
    (moonshine finalizes on stop()). Routing through the pipeline's own
    segment handler keeps ids monotonic, translation applies, and displays
    stay consistent — this used to call the old Translation path and silently
    dropped the tail.
    """
    fin = getattr(pipeline.transcriber, "finish", None)
    if fin is None:
        return
    for seg in fin():
        pipeline._handle_segment(seg, pipeline.translator)


def _list_devices() -> None:
    import sounddevice as sd  # lazy

    print(sd.query_devices())


def _list_engines() -> None:
    print("ASR engines:      moonshine (default), whisper, nemotron, funasr, funasr-nano, vosk, vibeasr-bitnet, audio8")
    print("Translation modes:", ", ".join(TRANSLATE_MODES))
    print("Capture sources:  ", ", ".join(SOURCES))
    print("Languages:        ", ", ".join(SUPPORTED_LANGS))


def main():
    ap = argparse.ArgumentParser(description="Live transcribe+translate demo")
    ap.add_argument("--headless", action="store_true",
                    help="force CLI mode (packaged exe opens the GUI by default)")
    ap.add_argument("--list-devices", action="store_true",
                    help="print available audio devices and exit")
    ap.add_argument("--list-engines", action="store_true",
                    help="print ASR engines / translation modes / sources and exit")
    ap.add_argument("--display", choices=list(DISPLAY_KINDS), default=None,
                    help="console (default), transparent overlay window (Tier A), or file (jsonl+txt)")
    ap.add_argument("--log-captions", action="store_true",
                    help="tee every final caption to %%LOCALAPPDATA%%/voicelang/captions.{jsonl,txt} (works with any --display)")
    ap.add_argument("--log-jsonl", default=None,
                    help="override path for captions jsonl (default LOCALAPPDATA/voicelang/captions.jsonl)")
    ap.add_argument("--log-txt", default=None,
                    help="override path for captions txt (default LOCALAPPDATA/voicelang/captions.txt)")
    ap.add_argument("--input", metavar="WAV", default=None,
                    help="replay a WAV file instead of the microphone (deterministic demo)")
    ap.add_argument("--source", choices=["mic", "loopback", "wav", "app"], default=None,
                    help="capture source: mic (default), loopback (device mix), wav, or app (per-app like Discord screenshare)")
    ap.add_argument("--source-device", default=None,
                    help="loopback device NAME (e.g. 'Headphone (Realtek(R) "
                         "Audio) [Loopback]'); default: system default output")
    ap.add_argument("--source-app", default=None,
                    help="per-app capture target when --source app (e.g. 'Discord.exe' or 'pid:1234')")
    ap.add_argument("--transcriber", choices=["whisper", "nemotron", "moonshine", "funasr", "funasr-nano", "vosk", "vibeasr", "vibeasr-bitnet", "audio8"],
                    default=None, help="ASR backend (default: moonshine)")
    ap.add_argument("--language", default=None,
                    help="moonshine/funasr language tag (en/ar/de/es/ja/ko/tl/uk/vi/zh); "
                         "funasr-nano: zh/zh-cn/en/ja")
    ap.add_argument("--target-language", default=None,
                    help="target_language for translation (default: en)")
    ap.add_argument("--translate", choices=list(TRANSLATE_MODES), default=None,
                    help="translation mode (default: passthrough)")
    ap.add_argument("--translate-url", default=None,
                    help="endpoint for --translate (LibreTranslate/DeepL/DeepLX/Google base URL)")
    ap.add_argument("--translate-key", default=None,
                    help="API key for --translate deepl (DeepL-Auth-Key)")
    ap.add_argument("--model-arch", default=None,
                    help="moonshine arch (TINY_STREAMING default)")
    ap.add_argument("--config", metavar="JSON", default=None,
                    help="config file override; default: LOCALAPPDATA/voicelang/config.json")
    ap.add_argument("--gui", action="store_true",
                    help="open the settings GUI (engine, language, overlay placement)")
    args = ap.parse_args()

    if args.list_devices:
        _list_devices()
        return
    if args.list_engines:
        _list_engines()
        return

    if args.gui:
        from .gui import main as gui_main

        gui_main()
        return

    # Config resolution: explicit flags > config file > defaults.
    # load_config handles missing file (returns defaults) — no exists() check needed
    cfg = load_config(args.config)
    if args.display:
        cfg.display = args.display
    if args.source:
        cfg.source = args.source
    if args.source_device:
        cfg.source_device = args.source_device
    if args.source_app:
        cfg.source_app = args.source_app
    if args.input:
        cfg.input_file = args.input
    if args.transcriber:
        cfg.transcriber = args.transcriber
    if args.language:
        cfg.language = args.language
    if args.target_language:
        cfg.target_language = args.target_language
    if args.translate:
        cfg.translate = args.translate
    if args.translate_url:
        cfg.translate_url = args.translate_url
    if args.translate_key:
        cfg.translate_key = args.translate_key
    if args.model_arch:
        cfg.model_arch = args.model_arch
    if args.log_captions:
        cfg.log_captions = True
    if args.log_jsonl:
        cfg.log_jsonl = args.log_jsonl
    if args.log_txt:
        cfg.log_txt = args.log_txt

    # Gaming-optimized block sizes (single source: config.effective_block_seconds)
    from voicelang_core.config import effective_block_seconds as _resolve_block

    # WP-3.2 FIX: gate on source != wav (not input_file truthiness) and clear stale input_file
    if cfg.source != "wav" and cfg.input_file:
        cfg.input_file = ""
    if cfg.input_file and cfg.source == "mic":
        cfg.source = "wav"
    block = _resolve_block(cfg)
    # WP-2.3 FIX: wrap builder calls with distinct exit codes for GUI to map
    try:
        src = _build_source(cfg.source, cfg.input_file or None, cfg.source_device, cfg.source_app, block_seconds=block)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: source build failed: {exc} — check --source and device settings", file=sys.stderr, flush=True)
        sys.exit(10)
    try:
        tr = _build_transcriber(cfg.transcriber, cfg)
    except SystemExit as exc:
        # adapters raise SystemExit with remediation already formatted
        print(str(exc), file=sys.stderr, flush=True)
        # map ValueError-like (invalid language) to 10, FileNotFound to 11
        msg = str(exc).lower()
        code = 11 if "not found" in msg or "missing" in msg or "model" in msg else 10
        sys.exit(code)
    except ValueError as exc:
        print(f"error: invalid transcriber config: {exc} — check --transcriber and --language pairing", file=sys.stderr, flush=True)
        sys.exit(10)
    except FileNotFoundError as exc:
        print(f"error: model not found: {exc} — download or check model path", file=sys.stderr, flush=True)
        sys.exit(11)
    except Exception as exc:
        print(f"error: transcriber build failed: {exc} — check --transcriber and deps", file=sys.stderr, flush=True)
        sys.exit(10)
    try:
        translator = _build_translator(cfg)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: translator build failed: {exc} — check --translate settings", file=sys.stderr, flush=True)
        sys.exit(10)
    try:
        display = _build_display(cfg.display, cfg)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: display build failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(10)
    pipeline = Pipeline(source=src, transcriber=tr, translator=translator, display=display, target_language=cfg.target_language)
    # Warmup CUDA/transducer so first real chunk is ~60ms not 600ms (honors whisper_warmup_seconds)
    try:
        w_all = getattr(pipeline.transcriber, "warmup_all_shapes", None)
        if callable(w_all):
            w_all((0.5, 1.0, 2.0))
        else:
            w = getattr(pipeline.transcriber, "warmup", None)
            if callable(w):
                w(float(getattr(cfg, "whisper_warmup_seconds", DEFAULT_WHISPER_WARMUP_S)))
    except Exception:
        pass
    # WP-3.2 FIX: select ThreadedPipeline for any live source (mic/loopback/app) irrespective of stale input_file
    use_threaded = cfg.source in ("mic", "loopback", "app")
    if use_threaded:
        try:
            from .pipeline_threaded import ThreadedPipeline
            pipeline = ThreadedPipeline(
                source=pipeline.source, transcriber=pipeline.transcriber,
                translator=pipeline.translator, display=pipeline.display,
                target_language=cfg.target_language,
                source_language=getattr(pipeline, "source_language", "auto"),
            )
            print(f"[voicelang] pipeline: ThreadedPipeline (source={cfg.source})", flush=True)
        except Exception as exc:
            print(f"[voicelang] warning: ThreadedPipeline unavailable ({exc}) — using Pipeline", flush=True)
            print(f"[voicelang] pipeline: Pipeline (source={cfg.source}, threaded fallback)", flush=True)
    else:
        print(f"[voicelang] pipeline: Pipeline (source={cfg.source})", flush=True)
    if not cfg.input_file:
        print("Listening... (Ctrl-C to stop)")
    pipeline.run_streaming()
    if cfg.input_file:
        # Finite replay: flush the ASR tail so the LAST partial never gets lost.
        _finish_tail(pipeline)
    # Flush file sink before exit (Display.close is no-op for console/overlay)
    try:
        pipeline.display.close(timeout=1.5)
    except Exception:
        pass


if __name__ == "__main__":
    main()