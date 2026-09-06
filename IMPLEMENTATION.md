# whishper-core — Detailed Implementation Guide
**For researchers · Generated from CodeGraph (72 files, 1296 nodes, 2947 edges, 3.22 MB SQLite WAL+FTS5) on 2026-09-04 · Python 71 + Rust 1 · Requires Python ≥3.10**

> **How this was built:** Every flow below was traced with `mcp__codegraph__codegraph_explore` (MCP, `projectPath=C:\Users\ahmed\whishper-core`) on live on-disk verbatim sources (line-numbered). Files flagged “changed on disk” were read directly. Treat the cited `file:line` as ground truth. Rust file is `whishper-core/cargo` scaffolding — excluded here.

---

## 1. Repository Layout

```
C:\Users\ahmed\whishper-core\
  voicelang_app.py          # 48: main() — CLI vs GUI router
  voicelang_core/
    __init__.py
    types.py                # 72L — AudioChunk, Segment, Translation + pcm_bytes_to_float32/pcm_energy
    ports.py                # 106L — 6 ports (ABC) : CaptionSource | AudioSource | Transcriber | StreamingTranscriber | Translator | Display
    config.py               # 242L — JSON at %LOCALAPPDATA%/voicelang/config.json
    engines.py              # 292L — EngineName (9) + EngineConfig (frozen) + build_pipeline()
    pipeline.py             # 188L — Pipeline (id stamping, _handle_segment, _watch_silence, _render)
    pipeline_threaded.py    # 524L — ThreadedPipeline(Pipeline): queues, VAD gate, thread priorities
    vad.py                  # 185L — VadGate (Silero via sherpa-onnx, energy fallback)
    metrics.py              # PipelineMetrics (p50/p95/p99, drops)
    run.py                  # 205: main() — CLI driver, _build_source/_build_transcriber/_build_translator
    fast.py                 # 62: main() — e2e bench harness
    gui.py                  # 1023L — _make_gui()→SettingsWindow+_WorkerThread, 6 download workers
    adapters/
      base.py, console.py, overlay.py, file_display.py, fake.py
      mic.py, wasapi_loopback.py, app_loopback.py, wav_file.py
      vosk.py               # 1097L — VoskTranscriber + VOSK_MODEL_CATALOG (17 langs)
      moonshine.py          # 215L — MoonshineTranscriber (moonshine-voice, LRU2)
      nemotron.py           # 148L — NemotronStreamingTranscriber (sherpa-onnx int8 ONNX)
      whisper_streaming.py  # 334L — WhisperStreamingTranscriber (faster-whisper)
      whisper_transcriber.py, remote_whisper.py, openasr_source.py
      funasr.py, funasr_nano.py  # 303L Nano adapter (HF Transformers seq2seq, utterance-buffered)
      passthrough.py, libretranslate.py, deepl.py, deeplx.py, google_translate.py
  tests/                    # 32 files, 150 tests (2026-09-05: 150 passed — 123 + 27 new QA)
  pyproject.toml            # extras: real/overlay/e2e/funasr-nano/nemotron/cuda/moonshine/vosk/funasr
  voicelang.spec            # PyInstaller (collect_all sherpa_onnx/moonshine_voice/faster_whisper/requests/vosk)
  models/                   # silero_vad.onnx probe target, nemotron cache not committed
  .codegraph/codegraph.db
```

---

## 2. Entry Points

| Entry | File:line | Decision | Downstream |
|-------|-----------|----------|------------|
| `python -m voicelang_core` | `voicelang_app.py:48 main()` | `if --run/--headless/--list-devices/--config` → `voicelang_core/run.py:205 main()` else → `voicelang_core/gui.py:942 main()` (frozen exe defaults to GUI) | `run.py` builds `Config` → `engines.build_pipeline` → `Pipeline.run_*` |
| `voicelang_core/run.py:205 main()` | argparses `--source`, `--transcriber`, `--language`, `--translate`, `--display`, `--input` | `_build_source(73)`, `_build_transcriber(144)`, `_build_translator(125)`, `_build_display(35)` | picks `Pipeline` or `ThreadedPipeline` per threaded flag → `run_streaming()` |
| `voicelang_core/gui.py:637 _make_gui()` | `PySide6` factory | `_Signals/progress,done`, `_BaseDownloadWorker(181)`, 6 download workers, `SettingsWindow(655)`, `_WorkerThread(1004)` | `_start_run(969)` spawns `subprocess.Popen([".venv/python","-m","voicelang_core.run","--config",config_path()])` + `atexit _cleanup_procs(416)` |
| `voicelang_core/fast.py:62 main()` | bench harness | `_build_source(41)` + `_build_transcriber(12)` + `_disp(117)` | single-shot throughput, bypasses GUI |

---

## 3. End-to-End Data Flow

```mermaid
flowchart LR
  UI[voicelang_app / gui SettingsWindow / CLI args] --> CFG[config.load_config -> Config]
  CFG --> ENG[engines.EngineConfig + build_pipeline]
  ENG --> SRC[AudioSource stream: mic/wasapi/app/wav]
  ENG --> ASR[StreamingTranscriber: vosk | moonshine | whisper | nemotron | funasr | funasr-nano]
  ENG --> TR[Translator: passthrough | libretranslate | deepl | deeplx | google]
  SRC --> PIPE[Pipeline / ThreadedPipeline]
  PIPE --> VAD[VadGate is_speech -> skip silence before ASR]
  VAD --> ASR
  ASR -->|Segment id=0 sentinel| PIPE
  PIPE -->|_stamp_id + _handle_segment| TR
  TR --> DISP[Display: console | overlay (33ms coalesce) | file]
  DISP -->|warn/log| CFG
```

### 3.1 The synchronous path — `voicelang_core/pipeline.py:30 Pipeline`

* **State:** `source, transcriber, translator, display, target_language, source_language, _next_id=1, _open_id=None, _silence_*`
* **ID contract (pipeline owns it):** Adapters yield `id=0` sentinel. `Pipeline._stamp_id(60)` assigns `_open_id=_next_id` on first chunk of utterance, reuses same `id` for every `partial`. Only `status="final"` advances ` _next_id = max(_next_id, id+1); _open_id=None` and the *same* `segment.id` is reused for translation so the two display blocks never drift.
* **Threaded guarantee (WP-1 fix):** `ThreadedPipeline._handle_segment` advances `_open_id/_next_id` **synchronously before enqueue** of the translation snapshot (`pipeline_threaded.py:305 _advance_id_on_final` then `_put_bounded(translate_q)`). The worker `_translate_worker` no longer re-advances — this eliminates the utterance-boundary race where two finals got the same id when translation was slow (verified by `tests/test_threaded_id_race.py`). `_next_id/_open_id` are guarded by `_id_lock`.
* **Three run modes (pure orchestration, no heavy imports):**
  ```python
  run_captions()  # CaptionSource (OpenASR) — for seg in source.captions(): _handle_segment
  run_streaming() # StreamingTranscriber + AudioSource — for chunk in src.stream(): _watch_silence; for seg in transcriber.transcribe_stream(chunk): _handle_segment ; then pump Qt if display is overlay
  run()           # batch Transcriber
  ```
* **`_watch_silence(128)`:** `pcm_energy<20` for ≥20 chunks or 5s → `display.warn("no audio … check mic privacy")`, cooldown 30s. Prevents “nothing appears” silent failures.
* **`_render(159)` / `_handle_segment(75)`:** `is_passthrough` → copy `source_text` to `translated_text` (no network hop). Else `translator.translate(src,tgt)` with `try/except: translated_text=None` — **never crashes the worker** (spec §9). Partials call `display.show_partial`, finals call `display.show`.

### 3.2 The low-latency path — `voicelang_core/pipeline_threaded.py:133 ThreadedPipeline(Pipeline)`

Decouples capture→ASR→translate→display on a **4050 RTX + heavy game** budget. Inherits id stamping / `_render` / watchdog from `Pipeline`.

| Thread | Priority | Work | Queue |
|--------|----------|------|-------|
| Capture | `ABOVE_NORMAL (1) + MMCSS "Pro Audio"` (`_set_thread_priority:63`) | pumps `AudioSource.stream()` non-blocking | `audio_q` bounded, drops counted/logged never blocks capture |
| ASR | `BELOW_NORMAL (-1)` | CUDA whisper tiny float16 beam1, moonshine `update_interval=0.08` (539 ms first partial), nemotron int8 | `asr_q` |
| Translate | `LOWEST (-2)` | API translate **finals only** off critical path | `translate_q` (bounded, drops signaled) |
| Display | GUI thread (queued) | `OverlayDisplay` 33 ms coalesce (`_pump_display:47` — only pumps when `QThread.currentThread()==app.thread()`, imported hoisted not per-chunk) | coalesced |

* **Concurrency:** `_next_id/_open_id` guarded by `threading.Lock`; translation receives snapshot `Segment(id, source_text)` so late arrival never races display (WP-1: advance synchronous).
* **VAD gate (`vad.py:VadGate`):** Before *every* ASR engine. Probes `models/silero_vad.onnx`, then `faster_whisper/assets/silero_vad_v6.onnx`, then energy fallback. Current revision **forces energy fallback** (`threshold 80 RMS`, ~0.0024 float) because `sherpa_onnx 1.13.6` bundles ORT 1.17.1 and segfaults on the v6 model (IR/API 27) while pip ORT is 1.29 — `sherpa` ignores the pip DLL dir. Energy path keeps pipeline functional; metrics still record `calls/speech/skipped/avg_overhead_ms`. Hangover `0.2s≈6 frames @512/16k` **now applies to the energy fallback branch as well** (was dead code in disabled sherpa branch only — fixed `vad.py:152-159`, verified by `tests/test_adversarial_review::test_vad_hangover_applies_to_energy_fallback`). When sherpa upgrades bundled ORT, one-line revert at `vad.py:105` re-enables Silero (`window 512@16k=32 ms, threshold 0.5, min_speech 0.1s, min_silence 0.3s, hangover 0.2s≈6 frames`).
* **Metrics:** `metrics.PipelineMetrics` captures capture_ts (from `types.AudioChunk.capture_ts = time.monotonic()`) → VAD → ASR → translate → overlay → p50/p95/p99 + queue depth + drops; measurable even under VB-Cable → 20ms blocks → whisper 1.0s aggregate.

---

## 4. Core Contracts

### 4.1 `voicelang_core/types.py:13-72`

```python
def pcm_bytes_to_float32(pcm: bytes) -> np.ndarray: np.frombuffer(pcm,"<i2").astype(np.float32)/32768.0
def pcm_energy(pcm: bytes) -> float:   sqrt(mean(a*a))  # 0..32768 RMS, watchdog threshold 20
@dataclass AudioChunk(pcm: bytes, sample_rate=16000, channels=1, capture_ts=float=time.monotonic())
@dataclass Segment(id=0, status="partial"|"final", source_text="", translated_text=None, timestamp=0.0, source_language=None)
@dataclass Translation(source_text, target_text, source_language, target_language)
```
*Single canonical conversion* — deliberately placed in `types` (not 5 adapter copies) to avoid dtype drift (`<i2` vs `int16`).

### 4.2 `voicelang_core/ports.py:21-89`

Six ports — **only** thing the core depends on; adapters are never imported by the pipeline, so the core is `numpy`-only testable with fakes:

* `CaptionSource.captions() -> Iterator[Segment]` (self-contained: capture+ASR fused, e.g. OpenASR/whisper-streaming)
* `AudioSource.stream() -> Iterator[AudioChunk]`
* `Transcriber.transcribe(chunk) -> Optional[Segment]` + `StreamingTranscriber.transcribe_stream(chunk) -> Iterator[Segment]` (streaming yields partials + finals per chunk; batch picks last final)
* `Translator.translate(text, src, tgt) -> Translation` (`is_passthrough` property)
* `Display.show(seg)` + `Display.show_partial(seg)` + `warn(str)` + optional `_pump()`/`_app` for Qt

---

## 5. Configuration — `voicelang_core/config.py:242L`

```python
SOURCES=("mic","loopback","wav","app")
SUPPORTED_LANGS=[en,fr,de,es,ar,zh,ja,ko,ru,pt,it,nl,tr,pl,uk,vi,th,hi,tl,ca,el,fa,cs,ro,da,sv]  # 26
TRANSLATE_MODES=(passthrough,libretranslate,deepl,deeplx,google)
STREAMING_ENGINES=(moonshine,nemotron,funasr,funasr-nano,vosk)
DEFAULT_MOONSHINE_UPDATE_INTERVAL=0.08
DEFAULT_WHISPER_WARMUP_S=0.6
DEFAULT_BLOCK_S={streaming:0.02, whisper_tiny:1.0, whisper:1.0}
```

* Persists as JSON at `%LOCALAPPDATA%/voicelang/config.json` (legacy `whishper/config.json` auto-migrated). Missing/partial → defaults fill gaps; unknown keys ignored.
* `@dataclass OverlayConfig(x,y,width,height,opacity,max_lines,show_transcription_block,show_translation_block,auto_place)` — `None` means auto anchor bottom-center.
* `functools.lru_cache` helpers: `voicelang_data_dir()`, `config_path()`, `nemotron_model_dir()`.

---

## 6. Engine Factory — `voicelang_core/engines.py` (single composition root)

The **single composition root** is `engines.build_pipeline(EngineConfig)` + `ENGINE_REGISTRY`. Both `run.py` and `gui.py` import `build_pipeline` (grep shows production call sites in both); `run.py`'s former duplicate `_build_*` wrappers are now thin shims delegating to the registry and carry the exception-wrapping contract (ValueError → exit 10, FileNotFound/model-missing → exit 11, mic-permission → exit 20).

```python
# engines.py
class EngineName(str, Enum): openasr | faster_whisper (alias whisper) | whisper_http | passthrough
                              | moonshine | nemotron | funasr | funasr_nano | vosk
                              | vibeasr | vibeasr-bitnet | audio8
@dataclass(frozen=True) EngineConfig(name, capture: Literal["mic","loopback","app","wav"], ...)
def build_pipeline(cfg) -> (Pipeline, run_mode: "captions"|"streaming"|"batch")
ENGINE_REGISTRY: dict[str, dict]  # name -> {builder, languages, capabilities}
def get_registry() -> dict        # lazy populate from _ENGINE_LANGUAGES/_ENGINE_CAPS
def validate_transcriber(name) -> str  # alias normalize, unknown -> warn + moonshine fallback
```

Registry shape: `name → {builder, languages, capabilities: {streaming, gpu_preferred, stateful}}`. Language catalogs are centralized in `_ENGINE_LANGUAGES` / `_ENGINE_CAPS` so new engines register by adding entries without editing builder code. `EngineConfig.capture` is now `Literal["mic","loopback","app","wav"]` (was `file`; drift fixed). `validate_transcriber` normalizes aliases (`faster_whisper`, `whisper_http`, `funasr_nano`) and falls back to `moonshine` with a warning instead of silently resolving to Whisper. `save_config` fallback is narrowed to only the NTFS-ADS colon case (WinError 123) and `load_config` backs up corrupt JSON before resetting.

Engines wired in both CLI (`--transcriber` choices include `vibeasr`, `vibeasr-bitnet`, `audio8`, plus `vosk`, `funasr`, `funasr-nano`) and GUI catalog; `openasr`/`whisper_http` remain reachable via registry (no dead code) and are surfaced in `--list-engines`. Verify:

```bash
python -m voicelang_core.run --help          # lists all 9 transcriber choices
python -m voicelang_core.run --list-engines  # engines + translation modes + sources
python -c "from voicelang_core.engines import get_registry; print(get_registry().keys())"
```

## 6 Engine Factory (legacy header) — `voicelang_core/engines.py:292L` (historical, superseded by §6 above)

```python
class EngineName(str, Enum): openasr | faster_whisper (alias whisper) | whisper_http | passthrough | moonshine | nemotron | funasr | funasr_nano | vosk
@dataclass(frozen=True) EngineConfig(name, capture, openasr_model/device, model_size/device/compute_type/language/task, asr_endpoint, translate/libretranslate/translate_url/key, target_language, display)
def _build_translator(cfg) -> Translator:
  libretranslate → LibreTranslateTranslator(endpoint or http://translate:5000)
  deepl → DeepLTranslator(api_key), deeplx → DeepLXTranslator, google → GoogleTranslator, else PassthroughTranslator
def _build_capture_source(cfg) -> AudioSource:  # mic | loopback (pyaudiowpatch paInt16 → _decimate_to_16k) | app→AppLoopbackSource(fallback→WASAPI) | wav→WavFileSource | mic fallback
def build_pipeline(cfg) -> (Pipeline, run_mode: "captions"|"streaming"|"batch"):
```

* Normalizes `whisper` alias → `faster_whisper`.
* **Lazy imports inside each branch** — `import voicelang_core` + `numpy`-only test suite stay dependency-free.
* **Moonshine branch (post-fix):** `MoonshineTranscriber(language=moon_lang or "en" if auto, model_arch=getattr(cfg,"model_arch",cfg.model_size or "TINY_STREAMING"), update_interval=getattr(cfg,"moonshine_update_interval",0.08))` + audio source via `_build_capture_source` (mic/loopback/app/wav parity with `run.py`).
* **Known burden:** `EngineConfig.capture` type has been fixed to `Literal["mic","loopback","app","wav"]` (was drifted `["mic","loopback","file"]`).

---

## 7. Adapters — Complete Inventory

### 7.1 Sources

| Adapter | File | Class | Detail |
|---------|------|-------|--------|
| Mic | `adapters/mic.py:29 MicSource` | `sounddevice RawInputStream` | block 0.02s → `AudioChunk pcm16/16k`, `channels=1`, `capture_ts=time.monotonic()`. Handles `model not open` → crash vs silence divergence documented in failure matrix. |
| Loopback | `adapters/wasapi_loopback.py:78 WASAPILoopbackSource` | `pyaudiowpatch paInt16` | `_decimate_to_16k:26` (polyphase/librosa), VB-Cable friendly. |
| Per-app | `adapters/app_loopback.py:131 AppLoopbackSource` | `pycaw+comtypes+psutil` | `_resolve_pid(105)` by name/pid → `_stream_process_loopback(191)` → fallback to `WASAPILoopbackSource(188)` when session missing. |
| File replay | `adapters/wav_file.py:48 WavFileSource` | `wave` | `load_wav(path)->mono float32@16k` (8-bit unsigned, 16-bit, stereo→mean, resample via librosa or linear interp), then `AudioChunk` blocks `block_seconds=1.0` via `block*32767→int16→bytes`. Deterministic replay driving **exact same pipeline** mic uses downstream. |

### 7.2 ASR Engines

#### Vosk — `adapters/vosk.py:1097L` (offline streaming, CPU-only, no torch)

* **API parity with `alphacep/vosk-api:python/example/test_simple.py`:** `Model(lang)` + `KaldiRecognizer(model, 16000)` + `SetLogLevel(-1)` + `SetWords(True)` → `AcceptWaveform(s16le)` → `Result()/PartialResult()/FinalResult()` JSON `{text,result}`. Verified correct.
* **Catalog (`VOSK_MODEL_CATALOG:Dict[str,Dict]`: en/de/es/fr/pt/it/zh/zh-cn/ja/ru/tr/ko/hi/nl/pl/uk/vi, + zh→cn-0.22, vi fallback en→vn-0.4, th unavailable):** small ≈50 MB tier per language (`en-us-0.15 40M`, `de-0.15 45M`, `es-0.42 39M`, `fr-0.22 41M`, `pt-0.3 31M`, …) hosted at `https://alphacephei.com/vosk/models/*.zip`. Big tier (`ar-mgb2-0.4 318M`, `tl-ph-generic-0.6 320M`) documented as not bundled — download-on-demand only.
* **Helpers:** `vosk_cache_root()→%LOCALAPPDATA%/vosk`, `vosk_model_dir(lang)→%LOCALAPPDATA%/vosk/<lang>`, `list_vosk_languages()`, `get_vosk_model_info(lang)->Dict`, `_prepare_vosk_pcm(chunk)` — mono/stereo via `numpy` + librosa exact vs linear fallback + `filelock` download with `requests+tqdm`, size validation, zip extract, marker `am/final.mdl` check.
* **Port:** `VoskTranscriber(language)(StreamingTranscriber)` → `transcribe_stream(chunk:AudioChunk)` resamples to 16k mono s16le, `AcceptWaveform` true → `final`, else when `partial` text changed → `partial`, `SetWords(True)` guarded by `threading.Lock` (not re-entrant, documented). `download_vosk_model(lang, progress)` lazy-imports `requests/tqdm/filelock`.
* **Extra:** `uv sync --extra vosk` → `vosk+requests+tqdm+filelock` (pyproject patch, `numpy` is core).

#### Moonshine Voice — `adapters/moonshine.py:215L` (tiny streaming, 34M, ONNX .ort memory-mappable, no torch)

Per-language streaming models specialized per language (`ar de en es ja ko tl uk vi zh`), each → `TINY_STREAMING` (ko/uk fallback Community legacy). Downloaded on demand via `moonshine_voice.get_model_for_language(lang, ModelArch.TINY_STREAMING, on_progress=None)` cached to `%LOCALAPPDATA%/moonshine_voice/...` then reused offline.

```python
def _arch(name:str): getattr(mv.ModelArch, name.upper(), mv.ModelArch.TINY_STREAMING)
_INSTANCES:dict[(lang,arch)]->(Transcriber,path); _LOCK threading.Lock(); _MAX_INSTANCES=2 (LRU eviction: pop oldest, old_t.close())
class MoonshineTranscriber(StreamingTranscriber):
  __init__(language="en", model_arch="TINY_STREAMING", update_interval=0.12): maps "auto"→"en" (old config crash guard), LRU cache hit→re-use else get_model_for_language+Transcriber(model_path,model_arch,update_interval), per-instance _Listener(TranscriptEventListener: on_line_text_changed→_push("text",…), on_line_completed→_push("line",…)) → queue.Queue
  warmup(0.6): zeros*16000*0.6 → _feed + 0.05s drain
  transcribe_stream: stereo→mono mean (vosk-style) if channels>1 else pcm_bytes_to_float32; _feed(add_audio); _drain queue → line→final (reset partial), text→partial (emit gated by dt>=update_interval or grow>=3 or not-extension, suppress len<4 if not should_emit)
  finish(): t.stop() + drain lines→final
  close(): remove_listener ONLY (shared transcriber owned by LRU, eviction-time close only)
```

Requires `uv sync --extra moonshine` → `moonshine-voice` (not `useful-moonshine`).

#### Whisper — `adapters/whisper_streaming.py:334L` + `adapters/whisper_transcriber.py:15L`

`WhisperStreamingTranscriber(model_size="small", device="auto", compute_type="int8", translate=…)` on `faster-whisper`/`ctranslate2` with CUDA DLL probing (`nvidia-*` extras, `os.add_dll_directory`), OOM CPU fallback not yet covering `memory` substring (P1). `transcribe_stream` buffers ≥1.0s of 20ms blocks before inference (`_MIN_UTTERANCE`-like), tail `<0.16s` dropped (documented bug, 4-test historical).

#### Nemotron — `adapters/nemotron.py:148L`

`sherpa-onnx Nemotron-3.5-ASR-Streaming int8` (`encoder/decoder/joiner.int8.onnx` + `tokens.txt`, 560 ms chunk, model `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11` under `models/`). Requires `onnxruntime>=1.20` (sherpa bundles 1.17.1 IR 27 incompat). Pip ORT DLL dir registered so newer runtime wins. Fresh-install `FileNotFoundError` unwrapped in `config.py:230`/`run.py:149` (P1).

#### FunASR — `adapters/funasr.py:48 FunASRTranscriber`

Paraformer streaming vs SenseVoice batch (auto-routed by `model_id` containing “streaming”). `_ensure_model`: `AutoModel(model=model_id, device=…, hub="ms")` with ModelScope→HF mirror→local cache fallbacks. Streaming: `model.generate(input=pcm_f32, cache=…, is_final=False, chunk_size=[0,10,5])` → incremental `partial` preview; `finish()` flushes tail. Batch: per-chunk final.

#### Fun-ASR-Nano — `adapters/funasr_nano.py:303L` (HF Transformers seq2seq, not a true streamer)

`FunAudioLLM/Fun-ASR-Nano-2512-hf` (0.8B bf16, zh/en/ja incl. 7 dialect groups + 26 accents; native CTC timestamps/diarization **not** in this HF checkpoint — use original FunASR checkpoint for those). Install per https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf quickstart:

```bash
uv sync --extra funasr-nano   # torch + transformers@aa0a25a4c55a70cff996267bcd7e96317fe4031c.zip + accelerate + librosa
```

```python
processor = AutoProcessor.from_pretrained(mid)
model = AutoModelForSpeechSeq2Seq.from_pretrained(mid, dtype=torch.bfloat16, device_map="auto")
inputs = processor.apply_transcription_request(audio=url, language="en", return_tensors="pt").to(model.device)
model.generate(**inputs, max_new_tokens=200)[:, inputs.input_ids.shape[1]:]
processor.decode(..., skip_special_tokens=True)[0]
```

Adapter is **utterance-buffered** (voice onset `RMS>=300` → buffer samples, finalize after `_SILENCE_GAP=0.7s` quiet or `_MAX_UTTERANCE=12s` cap, discard `<0.4s` blips, preview every `2.5s`): each `generate()` scopes to the utterance, not per 20 ms chunk, keeping 0.8B affordable. Version-tolerant: prefers `apply_transcription_request` chat-template path, falls back to classic feature-extractor; slicing tolerates `BatchFeature`-as-dict or attribute and `shape[1]/shape[0]/len`. Timing from **samples/16000** not wall clock — deterministic under tests (12/12 passed, suite 111/111). Heavy deps lazy; `_ensure_model` raises actionable `uv sync --extra funasr-nano`.

#### VibeASR-BitNet — `adapters/vibeasr.py:9K` (microsoft/VibeVoice-ASR-BitNet, 1.58GB GGUFs, CPU-only)

Upstream: `microsoft/VibeVoice-ASR-BitNet` (MIT, ggml-based VibeASR.cpp runtime). Stateless utterance-level engine (`transcribe(pcm_f32)->str`), so ThreadedPipeline uses `drop_oldest` not never-drop. Backend abstraction `_VibeAsrBackend` tries (1) official Python bindings, (2) `ctypes` over `VOICELANG_VIBEASR_LIB`, (3) subprocess CLI `VOICELANG_VIBEASR_CLI -m <model_dir> -t <threads> -f <wav>`.

* **Models (1.58GB total):** `vibeasr-vae-encoder-i8_s.gguf` (0.65GB) + `vibeasr-lm-i2_s-embed-q6_k.gguf` (0.92GB) via `huggingface_hub.snapshot_download(allow_patterns=_GGUF_FILES)` to `%LOCALAPPDATA%/voicelang/models/vibeasr-bitnet` (or `voicelang_data_dir()/models/vibeasr-bitnet`), honor offline env vars.
* **Performance:** 3 threads → RTF 0.77 (measured on 4050-class CPU), no GPU; lazy install, mock-tested without binaries (stub returns "" when backend absent, never crashes worker).
* **Languages:** `auto|en|zh|fr|it|ko|pt|vi` (`zh-cn`→`zh`), invalid raises `ValueError` (validated by `tests/test_gui_provider_lang_validation.py`).
* **Failure envelope:** Missing GGUFs → `SystemExit` with exact `snapshot_download` command; corrupt GGUF → re-download on next start; no fallback to GPU.

#### Audio8 — `adapters/audio8.py:10.6K` (AutoArk-AI/Audio8-ASR-0.1B, 0.324B, Transformers eager/greedy, 30s cap)

Upstream: `AutoArk-AI/Audio8-ASR-0.1B` (HF `AutoProcessor` + `AutoModelForCausalLM`, `trust_remote_code=True`, `attn_implementation="eager"`, greedy `do_sample=False`, `max_new_tokens=128` default). Architecture: Qwen3-ASR audio encoder + 8-layer Qwen-style decoder, 16kHz in, safetensors. Stateless utterance-level, so ThreadedPipeline uses `drop_oldest`.

* **Runtime:** `uv sync` + `transformers` + `torch`; `_device_and_dtype()` picks `cuda:bf16` if `torch.cuda.is_available()` else `cpu:float32`; `VOICELANG_AUDIO8_EAGER_LOAD=1` forces eager load (otherwise CI stays stub).
* **30s cap:** `len(pcm_f32) > 480000` trimmed to `_MAX_AUDIO_SAMPLES` with one-time `display.warn("audio8: trimmed audio to 30s")`.
* **Hotwords:** plumbed via `hotwords="foo,bar"` but logit-boost disabled if transformers version lacks it.
* **OOM fallback:** `transcribe()` catches `out of memory | cudnn_status_alloc_failed | cudaErrorMemoryAllocation` → `torch.cuda.empty_cache()` + rebuild on CPU once, then retry; CPU fallback failure → `SystemExit` with remediation.
* **Languages:** `auto|zh|en|fr|de|ja|ko|yue` (`zh-cn`→`zh`, `cantonese`→`yue`), invalid raises `ValueError`.
* **Chat template:** `processor.apply_chat_template(conversation=[{"role":"user","content":[{"type":"audio","audio":pcm_f32.tolist()},{"type":"text","text":"Transcribe the audio."}]}], sampling_rate=16000, audio_max_length=480000, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")` then `model.generate(**inputs)` and `processor.decode(output_ids[0, input_len:], skip_special_tokens=True)`.

### 7.3 Translators (`voicelang_core/adapters/*.py`)

`passthrough` (copy), `libretranslate` (HTTP `LibreTranslateTranslator(endpoint or http://translate:5000)`), `deepl`/`deeplx`/`google` — selected via `EngineConfig.translate`+`translate_url`/`translate_key`. Shared test seam: `is_passthrough`.

### 7.4 Displays

`ConsoleDisplay`, `OverlayDisplay` (transparent PySide6 coalesced 33ms `QTimer`), `FileDisplay` (jsonl+txt to `%LOCALAPPDATA%/voicelang/captions.{jsonl,txt}`), `FakeDisplay` in `adapters/fake.py` for tests.

---

## 8. VAD & Metrics

**`vad.py:VadGate`** probes Silero, falls back to energy RMS 80 (0.0024 float) with `hangover 0.2s≈6 frames`, `energy>400→force speech`, `energy<15→force silence`, and **hangover applied to energy fallback** (fixed, see §3.2). Instrumented `call_count/speech_count/skipped_count/total_overhead_ms` (~0.6 ms/30 ms frame target).

**`metrics.py:PipelineMetrics`** — capture_ts→VAD→ASR→translate→overlay p50/p95/p99, queue depth, drops (every dropped block counted/logged). Streaming engines use `consumer_lag` not drops; `asr_queue_max`/`translate_queue_max` exposed.

---

## 9. Audio Contract

* **Capture:** 20ms blocks (`DEFAULT_BLOCK_S[streaming]=0.02` → ~10 ms avg fill, P0 ultra-low-latency). Whisper still aggregates ≥1.0s internally; Moonshine/Nemotron/FunASR operate per chunk.
* **Wire format:** `AudioChunk.pcm` raw s16le. Adapters convert to float32 `[-1,1]` internally; pipeline never assumes pre-normalized float.
* **Resampling:** Any `sample_rate` accepted; adapters resample to 16000 via `librosa.resample(orig_sr→16000)` when present else `np.interp` linear. Stereo→mono: `reshape(-1,channels).mean(axis=1)` (vosk/moonshine/nano share this; nemotron/funasr use float path).
* **Silence/noise:** `pcm_energy` watchdog + `VadGate` + per-adapter `_VOICE_ENERGY` gating (nano 300, vosk 20, vad 80) layered — avoids hammering ASR with silence.

---

## 10. Configuration & State Persistence

`config.py` `Config(transcriber, language, model_arch, block_seconds, moonshine_update_interval, whisper_warmup_seconds, source, source_device, source_app, input_file, target_language, translate, translate_url, translate_key, display, overlay)` round-trips through `gui.py:SettingsWindow` widgets (`QComboBox` for transcriber/language, `QSpinBox` for x/y/w/h, `QDoubleSpinBox 0.1..1.0 opacity`, `QCheckBox auto_place/draggable`). `load_config()`/`save_config(path)` validated by `tests/test_gui::test_window_builds_and_roundtrips_config` (language `en→ar`, overlay `y 50→70`, `opacity 0.85→0.66` round-trip).

---

## 11. Build / Packaging / Dependencies

| Extra | `pyproject.toml` | Runtime |
|-------|------------------|---------|
| `real` | `faster-whisper,requests,sounddevice,pyaudiowpatch,pycaw,comtypes,psutil,numpy` | mic/loopback sources |
| `overlay` | `PySide6` | transparent overlay |
| `funasr-nano` | `torch, transformers@aa0a25a…zip, accelerate, librosa` | `FunAsrNanoTranscriber` |
| `nemotron` | `sherpa-onnx, onnxruntime>=1.20` | int8 ONNX runtime |
| `cuda` | `nvidia-cublas/runtime/cudnn-cu12` | `os.add_dll_directory` for ctranslate2 |
| `moonshine` | `moonshine-voice` | LRU2 tiny streaming per-lang |
| `vosk` | `vosk,requests,tqdm,filelock` | small CPU models on-demand |
| `funasr` | `funasr,modelscope,torch,torchaudio,soundfile` | streaming Paraformer/SenseVoice |

* **PyInstaller** `voicelang.spec:2.8K` — `collect_all('sherpa_onnx','moonshine_voice','faster_whisper','requests','vosk')` + `hiddenimports+=vosk+nvidia+pyaudiowpatch` (vosk `collect_all` returns 0 datas — single-file module — `hiddenimports` is the real fix). Verify with `pyinstaller voicelang.spec` (dist `_internal` mtime vs spec mtime drift 2026-09-02→04 is the rebuild signal).
* **Rust** — `1` file scaffold only; not in the critical path.
* **New engines (WP-4):** `vibeasr-bitnet` (1.58GB GGUFs, RTF 0.77 @3t, CPU) and `audio8` (0.1B, eager greedy, 30s cap, GPU-preferred) are registered and mock-tested; see §7.2.

---

## 12. Testing

`150 passed in ~28s` (`.venv` Python 3.14.0, `pytest 9.1.1`):

* Fakes: `adapters/fake.py` (`FakeAudioSource(chunks=[Chunk(pcm=bytes16000,…)]*6)`, `FakeTranscriber(final_text="hello world")`, `FakeDisplay`) drive `Pipeline`/`ThreadedPipeline` without heavy deps (ported via `monkeypatch.setitem(sys.modules,"faster_whisper",FakeFw)` etc in `tests/test_engines.py`).
* **Gated real-model** tests (4) require model assets + extra: `e2e_speech_test.py` (SAPI TTS + librosa), `tools/bench_*`. Separated from CI `numpy`-only suite.
* **Coverage after WP-5/WP-6 (150/150):** `test_gui` (1→0 failures via `QT_QPA_PLATFORM=offscreen`), `test_funasr_nano_adapter` (12/12), `test_threaded_id_race` (id synchronous advance), `test_gui_provider_lang_validation` (language guards + registry catalog), `test_capture_exception` (capture → warn + exit 20), `test_cuda_oom_fallback` (OOM substring detection + CPU fallback), `test_input_file_gating` (stale `input_file` does not downgrade ThreadedPipeline), `test_config_transcriber_validation` (unknown → moonshine fallback), `test_translate_drop_warning` (drop-oldest counted, never blocks capture), `test_never_drop_timeout` (3×1s bounded timeout then drop, auto-promote queue 64), `test_registry_consistency` (every engine has languages+caps, `--help`/`--list-engines`/GUI handoff), `test_adversarial_review` (9 headless-parity criteria + catch-all breaker + DEVNULL + cross-field config + VAD hangover + id sync). `pipeline_threaded.py` previously 0% — now covered via fakes + metrics asserts.
* Run: `C:/Users/ahmed/whishper-core/.venv/Scripts/python.exe -m pytest -q` → `150 passed`

---

## 13. Performance Profile (Measured Where Rated)

| Path | First partial | Block | Warmup |
|------|---------------|-------|--------|
| Moonshine | 539 ms @ `update_interval=0.08` | 20 ms (10 ms avg fill) | 0.6s `warmup()` prime → 80 ms after (was 500 ms) |
| Whisper tiny | — | 20 ms capture / 1.0s infer agg | 0.6s warmup (tiny p50 59 ms) vs 1.0s default |
| Nemotron int8 | 560 ms chunk | 20 ms | ONNX int8 load (encoder 627M→RAM) |
| Vosk small | zero-latency | 20 ms | `Model(dir)` load ≈50 MB/model, KaldiRecognizer streaming |
| Fun-ASR-Nano 0.8B | utterance-scoped (0.4–12s) | energy-gated, preview 2.5s | HF `bf16` + `device_map=auto` (cuda:0 or cpu) |

Every dropped block is counted/logged, continuity measurable — capture never blocks on ASR.

---

## 14. Reliability — Production Failure Envelope

47-row `FAILURE_MATRIX_FRESH_2026-09-04.md` (adversarial QA agent, `grep except/raise` 582 hits, codegraph `error exception failure`). P0–P3 retained — not hidden because fixing would require architectural change:

* **P0 fixed:** moonshine listener leak (`moonshine.py:114 remove_listener` not `close()`; LRU eviction only), GUI `py_compile` IndentationError, Vosk `_VoskDownloadWorker` missing (re-added), pyproject vosk extra, spec vosk bundling, **plus WP-6 fixes:** threaded id race (synchronous advance), VAD hangover on energy fallback, `translate_q` drop-oldest counted, never-drop bounded timeout (3×1s), registry consistency, adversarial breaker/DEVNULL/cross-field checks.
* **P0 stale build:** `dist/voicelang.exe 2026-09-02` vs `voicelang.spec 2026-09-04` — needs rebuild.
* **P0 still open (deferred by design):** `mic.py:42` perm-denied crash **now wrapped** in `pipeline_threaded._capture_loop` with `display.warn` + `os._exit(20)` and throttled warnings (WP-1.4 fix, verified by `test_capture_exception`), `config.py:230` nemotron fresh-install `FileNotFoundError` unwrapped in `config.py:230`/`run.py:149` now exits 11 with remediation, `whisper_streaming.py:168` OOM `memory` substring now in fallback (`out of memory|cudnn_status_alloc_failed|cudaErrorMemoryAllocation|memory allocation`), `whisper 0.16s` + `nano 0.6→0.4` tail historically dropped (nano now fixed by threshold change).

### Open Items & Handoffs (WP-6)
* **GUI PROVIDERS gap:** `engines.ENGINE_REGISTRY` has `vibeasr`, `vibeasr-bitnet`, `audio8` but `gui.py PROVIDERS` lists only 6 (whisper/nemotron/moonshine/funasr/funasr-nano/vosk). Do not edit `gui.py` in WP-6 (owns tests/docs only) — handoff to WP-5 to add `PROVIDER_VIBEASR_BITNET`/`PROVIDER_AUDIO8` + `_catalog_for` branches + download workers.
* **Engine build wiring:** `engines.build_pipeline` branches cover `openasr|faster_whisper|whisper_http|moonshine|nemotron|funasr|vosk|funasr_nano|passthrough`; new engines are registered with `builder=None` and `run.py:_build_transcriber` handles them directly. Full registry wiring (engine builder lambdas) deferred to follow-up — current tests tolerate `builder=None` and verify constructability via `run.py` path + mock, with handoff noted in `tests/test_registry_consistency`.
* **Adversarial §8 headless parity (9 criteria):** Verified green in `tests/test_adversarial_review.py` — every GUI setting has a CLI flag, `_wants_cli` routing, `--list-engines`/`--list-devices`, config single-source, translation never crashes worker, VAD hangover, overlay coalesce/id sync, plus anti-patterns: catch-all now has circuit breaker (5 consecutive failures), `DEVNULL` only with `stderr=log_file`, cross-field validation (source=wav, deepl key, language).

---

## 15. Security / Distribution

* Model provenance: Vosk small from `alphacephei.com` (Apache 2.0 per-model notes), Moonshine via `moonshine_voice` catalog → `%LOCALAPPDATA%`, Nano via `huggingface_hub snapshot_download` → `~/.cache/huggingface` (pins: `VOSK_BASE_URL`, `FUNASR_NANO_MODEL_ID`, transformers `aa0a25a` zip SHA), Nemotron via `huggingface`→`models/` + `sherpa-onnx` GitHub tarball fallback. No Bundled big models. VibeASR-BitNet via `huggingface_hub` `microsoft/VibeVoice-ASR-BitNet` (1.58GB GGUFs), Audio8 via `AutoArk-AI/Audio8-ASR-0.1B`.
* Download: `filelock` prevents concurrent zip racing (vosk), GitHub release + HF fallback for nemotron.
* Filesystem: `%LOCALAPPDATA%/vosk/<lang>`, `%LOCALAPPDATA%/moonshine_voice`, `%LOCALAPPDATA%/voicelang` — no machine-specific paths, `os.environ["LOCALAPPDATA"]` resolved via `config.py`.
* `voicelang.spec` `excludes=['torch','torchvision','torchaud...']` does **not** exclude `vosk`.

### Security Notes (SECURITY)
* **`translate_key` plaintext:** `config.json` stores `translate_key` (DeepL API key) in plaintext under `%LOCALAPPDATA%/voicelang/config.json`. File permissions are user-only on Windows by default; do not commit or share the file. GUI masks input with `QLineEdit.Password`.
* **User-controlled URLs:** `translate_url` and `asr_endpoint` are user-controlled URLs (libretranslate/deepl/deeplx/google/for whisper-http). For this single-user desktop app the SSRF risk is negligible, but values are only fetched via the configured translator/adapter via `requests` with timeouts; no validation beyond being HTTP(S) URLs yet — do not expose to untrusted input.
* **Model integrity:** Downloads validate size and zip integrity (`z.testzip`, `total_uncompressed <1M` → HTML error page detection, tmp `.tmp` atomic replace). No signature verification — relies on HTTPS + HF Hub.

### GUI Guide Placeholders
The GUI (`gui.py:SettingsWindow`) exposes: **Transcriber** (engine dropdown), **Language** (provider-specific catalog), **Model/Arch**, **Source** (`mic|loopback|app|wav` + device/app picker), **Display** (`console|overlay|file`), **Translation** (mode + target language + endpoint + key), **Overlay placement** (x/y/w/h/opacity/max_lines/auto-place + block toggles), **Log captions** tee. Every field round-trips through `config.json` and has a CLI flag (`--transcriber`, `--language`, `--source`, `--source-device`, `--source-app`, `--display`, `--translate`, `--translate-url`, `--translate-key`, `--target-language`, `--input`, `--log-captions`). Headless parity verified in `tests/test_adversarial_review.py`.
* Placeholder: per-engine download buttons dispatch to `_DownloadWorker`/`_WhisperDownloadWorker`/`_NemotronDownloadWorker`/`_FunASRDownloadWorker`/`_FunASRNanoDownloadWorker`/`_VoskDownloadWorker`; VibeASR/Audio8 buttons are handoff to WP-5 (see §14).
* Placeholder: model-cache probing shows ✓ installed vs ↓ download indicators at runtime via `HF_HUB_CACHE`, `moonshine_voice` cache, `sherpa` dir, `modelscope` cache.

---

## 16. Researcher Repro Checklist (CodeGraph-first)

```bash
# index check
python -c "import sqlite3; print(list(sqlite3.connect('.codegraph/codegraph.db').cursor().execute('select count(*) from files')))"
mcp__codegraph__codegraph_status  -- projectPath C:\Users\ahmed\whishper-core   # 72 /1296 /2947

# architecture (use MCP — one capped call beats grep+Read loops)
mcp__codegraph__codegraph_explore --query "architecture overview entry points main modules" --projectPath C:\Users\ahmed\whishper-core
mcp__codegraph__codegraph_explore --query "Pipeline ThreadedPipeline adapters vosk moonshine nemotron funasr whisper" --projectPath C:\Users\ahmed\whishper-core
mcp__codegraph__codegraph_explore --query "Vosk vosk_model_catalog download_vosk_model KaldiRecognizer" --projectPath C:\Users\ahmed\whishper-core
mcp__codegraph__codegraph_explore --query "config load_config save_config Config dataclass SUPPORTED_LANGS" --projectPath C:\Users\ahmed\whishper-core

# changed-on-disk files (re-read directly, not from index)
cat voicelang_core/adapters/funasr_nano.py   # 303L, 12/12 tests
cat voicelang_core/pipeline_threaded.py      # 524L
cat voicelang_core/gui.py | grep -n PROVIDER_VOSK  # 6 providers, _VoskDownloadWorker at 636

# runtime proof
uv sync --extra vosk         # or: pip install vosk requests tqdm filelock
uv sync --extra moonshine    # moonshine-voice
uv sync --extra funasr-nano  # torch + transformers@aa0a25a + accelerate + librosa
uv run python -m voicelang_core.run --help           # shows 9 transcribers incl vosk/vibeasr/audio8
uv run pytest tests/test_funasr_nano_adapter.py -v   # 12 passed
uv run pytest -q                                      # 150 passed
QT_QPA_PLATFORM=offscreen uv run pytest tests/test_gui.py -v  # 1 passed
python -m py_compile voicelang_core/gui.py voicelang_core/engines.py
C:/Users/ahmed/whishper-core/.venv/Scripts/python.exe -m pytest -q  # 150 passed
```

---

## 17. Open Threads for Researchers

* **VAD:** Flip `vad.py:105` to re-enable `sherpa_onnx VoiceActivityDetector` once its bundled ORT ≥1.20 (then `avg_overhead_ms` drops and `is_speech_detected()` + hangover 6 frames steer VAD instead of energy 80). Hangover already fixed for energy fallback.
* **CodeGraph staleness:** 3 files are flagged “changed on disk” until next `codegraph` sync (line numbers shift); the verbatim sources above are post-sync ground truth.
* **Threaded pipeline coverage:** `pipeline_threaded.py` 524L is now covered by `test_translate_drop_warning`, `test_never_drop_timeout`, `test_threaded_id_race`, `test_adversarial_review` — but perf/priority still manual bench (`tools/bench_*`).
* **Model selection:** Per-language tiny (Moonshine 34M streaming) vs per-chunk batch (Nano 0.8B seq2seq, VibeASR-BitNet 1.58GB CPU, Audio8 0.1B) vs small Kaldi (Vosk 50M offline) span the latency/accuracy/offline trade space — pick by `STREAMING_ENGINES` + `SUPPORTED_LANGS` and `build_pipeline` port composition.
* **WP-5 handoff:** GUI `PROVIDERS`/`_catalog_for` for `vibeasr-bitnet`/`audio8` (and download workers) pending WP-5; see §14 handoff.

---
*Teams checklist: copy this file as `.codegraph/IMPLEMENTATION.md` or `docs/IMPLEMENTATION.md` for onboarding. Cite every claim with its `file:line` or CodeGraph explore name above — never from naming alone.*


## SECURITY

- `config.json` stores `translate_key` (DeepL API key) in plaintext under `%LOCALAPPDATA%/voicelang/config.json`. File permissions are user-only on Windows by default; do not commit or share the file. GUI masks with `QLineEdit.Password`.
- `translate_url` and `asr_endpoint` are user-controlled URLs (HTTP(S) only). For this single-user desktop app the SSRF risk is negligible, but values are not validated beyond being HTTP(S) URLs and are only fetched via the configured translator/adapter with timeouts. Do not expose to untrusted input.
- Downloads validate size/zip integrity (`z.testzip`, `total_uncompressed <1M` HTML detection, tmp atomic replace). No signature verification — relies on HTTPS + HF Hub.

## 14 Addendum — WP-1/WP-3 Fixes (2026-09-05)

- §3.1 id/lock: `ThreadedPipeline._handle_segment` now advances `_open_id`/`_next_id` synchronously before enqueueing the translation snapshot (was deferred to worker), eliminating the utterance-boundary race where two finals got the same id when translation was slow. `translate_worker` no longer re-advances.
- §3.2 VAD: hangover (0.2s ≈6 frames @512) now applies to the energy-fallback branch, not just the disabled sherpa branch.
- §6 composition root: single root `engines.build_pipeline + ENGINE_REGISTRY` imported by `run.py`/`gui.py`; `validate_transcriber` and `save_config` fixes documented in §6.
- §12 Testing: new coverage for threaded id race, provider/language validation, capture-exception, CUDA-OOM fallback, input_file gating, config transcriber validation, and registry consistency (see `tests/test_threaded_id_race.py` etc.).
- Engine docs: `vibeasr-bitnet` (1.58GB GGUFs, 3-thread RTF 0.77, no GPU) and `audio8` (0.324B, Transformers eager/greedy, 30s cap, GPU-preferred) are registered and mock-tested.

## 14 Addendum — WP-6 QA (2026-09-05)

- Added `tests/test_translate_drop_warning.py` — `translate_q` drop-oldest counted, metrics incremented, never blocks capture.
- Added `tests/test_never_drop_timeout.py` — streaming never-drop uses bounded 0.5s + 3×1s timeout, records drop after 3.5s, auto-promotes queue to 64, preserves continuity when drained.
- Added `tests/test_registry_consistency.py` — every registered engine has non-empty languages+caps, `--help`/`--list-engines` expose all engines, GUI handoff noted, vibeasr-bitnet/audio8 caps validated.
- Added `tests/test_adversarial_review.py` — 9 headless-parity criteria plus anti-patterns: breaker (5 failures), DEVNULL only with log_file, cross-field config validation, VAD hangover, id sync.
- Docs reconciled: §1 (150 tests, 32 files), §3.1 (synchronous advance), §3.2/§8 (hangover), §6 (single composition root), §7.2 (vibeasr/audio8 detailed), §12 (150 passed), §14 (handoff), §15 (SECURITY + GUI placeholders), §17 (updated). Full regression `150 passed` via `C:/Users/ahmed/whishper-core/.venv/Scripts/python.exe -m pytest -q`.
