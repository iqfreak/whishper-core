"""Fast runner: threaded pipeline + warmup + gaming-friendly defaults for 4050.

Usage:
  uv run python -m voicelang_core.fast --source mic --transcriber whisper --model tiny --translate passthrough
  uv run python -m voicelang_core.fast --input e2e_speech.wav --transcriber moonshine --display console
  uv run python -m voicelang_core.fast --source app --source-app Discord.exe --transcriber whisper --model tiny
"""
from __future__ import annotations
import argparse, os, sys, time
from voicelang_core.config import Config, load_config, SUPPORTED_LANGS, TRANSLATE_MODES, SOURCES, DISPLAY_KINDS

def _build_transcriber(name: str, cfg: Config):
    if name == "moonshine":
        from voicelang_core.adapters.moonshine import MoonshineTranscriber
        lang = cfg.language if cfg.language and cfg.language != "auto" else "en"
        return MoonshineTranscriber(language=lang, model_arch=cfg.model_arch, update_interval=cfg.moonshine_update_interval)
    if name == "vosk":
        from voicelang_core.adapters.vosk import VoskTranscriber
        return VoskTranscriber(language=cfg.language or "en")
    if name == "nemotron":
        from voicelang_core.adapters.nemotron import NemotronStreamingTranscriber
        from voicelang_core.config import nemotron_model_dir
        return NemotronStreamingTranscriber(model_dir=nemotron_model_dir(), num_threads=2)
    if name == "funasr":
        from voicelang_core.adapters.funasr import FunASRTranscriber
        return FunASRTranscriber(language=cfg.language or "auto")
    if name == "funasr-nano":
        from voicelang_core.adapters.funasr_nano import FunAsrNanoTranscriber
        return FunAsrNanoTranscriber(language=cfg.language or "auto")
    # whisper family
    from voicelang_core.adapters.whisper_streaming import WhisperStreamingTranscriber
    sizes = {"tiny","base","small","medium","large-v3","large-v3-turbo"}
    size = cfg.model_arch if cfg.model_arch in sizes else "tiny"  # fastest for gaming
    w = WhisperStreamingTranscriber(model_size=size, device="auto", compute_type="auto", cpu_threads=2)
    # P3: per-shape warmup to cover cuBLAS autotune; single 0.6s leaves 0.5/2.0 cold
    try:
        if hasattr(w, "warmup_all_shapes"):
            w.warmup_all_shapes((0.5, 1.0, 2.0))
        else:
            w.warmup(cfg.whisper_warmup_seconds)
    except Exception: pass
    return w

def _build_source(source, input_file, device, app, block):
    if source == "wav":
        from voicelang_core.adapters.wav_file import WavFileSource
        s = WavFileSource(input_file, block_seconds=block)
        print(f"Replaying {input_file} ({s.duration_seconds:.1f}s)")
        return s
    if source == "app":
        from voicelang_core.adapters.app_loopback import AppLoopbackSource
        if not app:
            from voicelang_core.adapters.app_loopback import list_audio_apps
            apps=list_audio_apps()
            print("Running apps:", ", ".join(a["name"] for a in apps[:6]) or "none")
            raise SystemExit("--source app needs --source-app 'Discord.exe' or pid:1234")
        print(f"Capturing app {app!r} (per-app loopback; uses Rust IAudioClient3 when available)")
        return AppLoopbackSource(app=app, block_seconds=block)
    if source == "loopback":
        from voicelang_core.adapters.wasapi_loopback import WASAPILoopbackSource
        return WASAPILoopbackSource(device=device or None, block_seconds=block)
    from voicelang_core.adapters.mic import MicSource
    return MicSource(block_seconds=block, device=device or None)

def main():
    ap=argparse.ArgumentParser(description="Fast threaded pipeline for 4050 + gaming")
    ap.add_argument("--source", choices=list(SOURCES), default=None)
    ap.add_argument("--transcriber", choices=["whisper","moonshine","nemotron","funasr","funasr-nano","vosk"], default=None)
    ap.add_argument("--model-arch", default=None, help="tiny (fastest 59ms) | small (quality 300ms) | TINY_STREAMING")
    ap.add_argument("--language", default=None)
    ap.add_argument("--target-language", default=None)
    ap.add_argument("--translate", choices=list(TRANSLATE_MODES), default=None)
    ap.add_argument("--translate-url", default=None)
    ap.add_argument("--translate-key", default=None)
    ap.add_argument("--source-device", default=None)
    ap.add_argument("--source-app", default=None)
    ap.add_argument("--input", dest="input_file", default=None)
    ap.add_argument("--display", choices=list(DISPLAY_KINDS), default=None)
    ap.add_argument("--log-captions", action="store_true", help="tee finals to %%LOCALAPPDATA%%/voicelang/captions.{jsonl,txt}")
    ap.add_argument("--log-jsonl", default=None, help="override captions jsonl path")
    ap.add_argument("--log-txt", default=None, help="override captions txt path")
    ap.add_argument("--config", default=None)
    ap.add_argument("--block", type=float, default=None, help="capture block_seconds override")
    ap.add_argument("--gaming", action="store_true", help="P3: keep GPU clocks warm (LOWEST keep-alive ~100ms decode); optionally lock clocks with nvidia-smi -lgc (admin)")
    args=ap.parse_args()
    cfg=load_config(args.config)
    if args.source: cfg.source=args.source
    if args.transcriber: cfg.transcriber=args.transcriber
    if args.model_arch: cfg.model_arch=args.model_arch
    if args.language: cfg.language=args.language
    if args.target_language: cfg.target_language=args.target_language
    if args.translate: cfg.translate=args.translate
    if args.translate_url: cfg.translate_url=args.translate_url
    if args.translate_key: cfg.translate_key=args.translate_key
    if args.source_device: cfg.source_device=args.source_device
    if args.source_app: cfg.source_app=args.source_app
    if args.input_file: cfg.input_file=args.input_file
    if args.display: cfg.display=args.display
    if args.log_captions: cfg.log_captions=True
    if args.log_jsonl: cfg.log_jsonl=args.log_jsonl
    if args.log_txt: cfg.log_txt=args.log_txt
    # Gaming-optimized block sizes (single source: config.effective_block_seconds)
    from voicelang_core.config import effective_block_seconds as _resolve_block
    if cfg.input_file and cfg.source != "wav":
        cfg.source = "wav"
    block = _resolve_block(cfg, args.block)

    # display
    from voicelang_core.adapters.console import ConsoleDisplay
    def _disp():
        # file-only sink (no console/overlay)
        if cfg.display=="file":
            from voicelang_core.adapters.file_display import FileDisplay as _FD
            fd=_FD(jsonl_path=cfg.log_jsonl or None, txt_path=cfg.log_txt or None)
            print(f"File sink -> {fd.stats['jsonl']} + {fd.stats['txt']}")
            return fd
        if cfg.display=="overlay":
            try:
                from voicelang_core.adapters.overlay import OverlayDisplay
                d=OverlayDisplay(max_lines=cfg.overlay.max_lines, opacity=cfg.overlay.opacity,
                                 x=cfg.overlay.x, y=cfg.overlay.y, width=cfg.overlay.width,
                                 height=cfg.overlay.height,
                                 show_transcription_block=cfg.overlay.show_transcription_block,
                                 show_translation_block=cfg.overlay.show_translation_block)
                if d.headless:
                    print("Overlay unavailable -> console fallback")
                    base=ConsoleDisplay()
                else:
                    base=d
            except Exception as e:
                print(f"overlay init failed ({e}) -> console")
                base=ConsoleDisplay()
        else:
            base=ConsoleDisplay()
        # optional tee to file sink (overlay+file or console+file)
        if getattr(cfg, "log_captions", False):
            try:
                from voicelang_core.adapters.file_display import FileDisplay as _FD, MultiDisplay as _MD
                fd=_FD(jsonl_path=cfg.log_jsonl or None, txt_path=cfg.log_txt or None)
                print(f"File sink tee -> {fd.stats['jsonl']} + {fd.stats['txt']}")
                return _MD([base, fd])
            except Exception as e:
                print(f"[voicelang] file sink unavailable ({e})")
        return base
    display=_disp()

    # translator
    from voicelang_core.engines import EngineConfig, EngineName
    from voicelang_core.engines import _build_translator
    eng_cfg=EngineConfig(name=EngineName.passthrough, translate=cfg.translate,
                         translate_url=cfg.translate_url or None,
                         translate_key=cfg.translate_key or None,
                         target_language=cfg.target_language)
    # passthrough helper: if translate passthrough but target not en, keep passthrough
    translator=_build_translator(eng_cfg)
    print(f"ASR={cfg.transcriber} model={cfg.model_arch} block={block}s target={cfg.target_language} translate={cfg.translate} source={cfg.source}")
    # Gaming coexistence: process stays NORMAL (0x20), threads carry
    # their own priorities (capture Pro Audio, ASR BELOW_NORMAL, translate LOWEST)
    # so the game never competes with heavy ASR for the frame budget.
    try:
        if os.name=="nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000020)  # NORMAL
            print("Process priority: NORMAL (capture is Pro Audio / ASR is BELOW_NORMAL)")
    except Exception: pass

    source=_build_source(cfg.source, cfg.input_file or None, cfg.source_device, cfg.source_app, block)
    transcriber=_build_transcriber(cfg.transcriber, cfg)

    # Threaded pipeline: capture -> ASR off critical path, translate off hot path
    from voicelang_core.pipeline_threaded import ThreadedPipeline
    pipe=ThreadedPipeline(source=source, transcriber=transcriber, translator=translator, display=display,
                          target_language=cfg.target_language, source_language=cfg.language)
    if cfg.input_file:
        pipe.run_streaming_with_timed_replay()
        # also flush tail for completeness
        fin=getattr(transcriber,"finish",None)
        if fin:
            for seg in fin():
                pipe._handle_segment(seg)
    else:
        print("Listening... (Ctrl-C to stop)  Gaming tip: tiny 59ms decode + block => ~175ms mic->partial p50")
        pipe.run_streaming()
    # Flush file sink (queued off hot path) before exit so replay tests see all finals
    try:
        display.close(timeout=1.5)
    except Exception:
        pass
    # also print where the log landed
    try:
        s = getattr(display, "stats", None)
        if s is not None and isinstance(s, dict) and "jsonl" in s:
            pass
        elif hasattr(display, "_children"):
            for c in getattr(display, "_children", []):
                if isinstance(getattr(c, "stats", None), dict) and "jsonl" in getattr(c, "stats"):
                    print(f"Captions saved: {c.stats['jsonl']} ({c.stats['written']} finals)")
                    break
    except Exception:
        pass

if __name__=="__main__":
    main()
