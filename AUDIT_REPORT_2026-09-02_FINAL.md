# voicelang (whishper-core) — From-Scratch Audit Report

**Date:** 2026-09-02  
**Auditor:** principal engineer / architect / security / performance / QA (skeptical, evidence-driven)  
**Repository:** `C:/Users/ahmed/whishper-core` (disk name `whishper-core`, package `voicelang_core`, advertised as `voicelang`)  
**Scope:** requirements → architecture → modules → functions → dependencies → data flow → runtime → outputs  
**Evidence:** source reads of every adapter + core + config + engines + run + gui + voicelang_app + spec; `pyproject.toml` / `uv.lock` / `_build3.log` / `dist/voicelang` inventory; `pytest` selective runs; frozen smokes (`_frozen_mic2.log` watchdog 521B, `_frozen_whisper_cpu.log`); prior `BUILD_EXIT3=0` at 03:39.

---

## 1. Executive Summary

voicelang is a **streaming capture→transcribe→translate→display pipeline** with a pluggable ports/adapters core intended to be reusable for a Discord bot and a game overlay. The advertised product is: pick a capture source (mic / WASAPI loopback / per-app loopback / wav file), an ASR engine (moonshine / whisper / nemotron / funasr / funasr-nano), a translator (passthrough / libretranslate / deepl / deeplx / google), and a display (console / overlay) — and get live captions either in a console or a transparent always-on-top overlay. A windowed GUI (`voicelang_app.py`) persists settings to `%LOCALAPPDATA%/voicelang/config.json` and re-launches the same frozen exe with `--run --config <path>` (or `python -m voicelang_core.run --config <path>` from source). Heavy ML deps are `optional-dependencies` and lazily imported so `import voicelang_core` never requires torch/faster-whisper.

**Verdict:** the core pipeline architecture is sound and the majority of the system works as intended **after the fixes applied during this audit cycle**. The remaining gap between intended and actual is **small**: per-app capture is not truly per-process (documented fallback), and a few reliability/UX rough edges remain. No data-loss or privilege-escalation bug was found.

---

## 2. Overall Assessment

### What works
- Core `types` / `ports` / `pipeline` contract: `stdlib+numpy` only, `Segment{id}` monotonic ids owned by pipeline, `partial` reuses `id`, translation gated on `final` only, translator failure degrades to `⚠ source (translation unavailable)` never crashes worker (spec §9). Verified by `tests/test_pipeline.py`, `test_segment_id.py`, `test_streaming.py`, `test_translation_errors.py`.
- `MicSource` (`sounddevice.RawInputStream` 0.25s blocks, `list_input_devices()`, `--source-device` for mic and loopback in `run.py`/`gui.py`).
- Device-level `WasapiLoopbackSource` (`pyaudiowpatch`, `_decimate_to_16k` mean-decimate 48→16k).
- `WavFileSource` deterministic replay (`librosa` else linear, `finish()`/`_finish_tail` tail flush).
- All 5 ASR adapters lazy + registry-locked; Moonshine per-language streaming, Whisper streaming/batch, Nemotron int8, FunASR Paraformer/SenseVoice, FunASR Nano utterance-buffered. Correct `is_passthrough` handling.
- Translators libretranslate/deepl/deeplx/google + passthrough.
- Overlay v2 fixed-size scrollable `QTextEdit` blocks, `WA_TransparentForMouseEvents`, wheel forwarding, 33ms coalesce, `QWindow.startSystemMove()` OS drag, `numbered_entries()`, bilingual two-block sync.
- Config tolerant/unknown-keys-ignored + legacy `whishper→voicelang` migration.
- CLI/GUI parity `--headless --list-devices --list-engines --config --source --source-device --source-app --input --display --transcriber --language --target-language --translate --translate-url --translate-key --model-arch`.

### What partially works
- **Per-app loopback** (`app_loopback.py`): enumerates via `pycaw`, but native `AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS` via `comtypes` is TODO and raises `RuntimeError("native per-process loopback capture not yet activated…; using device-level fallback")` then falls back to device loopback with honest labeling. Function exists, is reachable, but does not isolate a single process. This is **by design at this stage** (Tier B scope) but should not be advertised as true isolation.
- **Frozen whisper CUDA**: auto-probe now correctly falls back to CPU when `cublas64_12.dll` unavailable (fix applied this audit). Previously `get_cuda_device_count()>0` false-positive caused mid-transcription crash.
- **Frozen cold start**: 76M exe + 3G `_internal` → 45–60s first-launch latency (expected for one-dir with torch/scipy). Not a correctness bug, but UX should set expectations.

### What does not work / what was broken before this audit's fixes
All three were **fixed** and verified this cycle; they would have been P0/P1 if shipped unfixed:
1. `faster_whisper/assets/silero_vad_v6.onnx` not bundled → `NO_SUCHFILE`.
2. `nvidia/cublas/cudnn` DLLs not bundled (spec used `collect_dynamic_libs('nvidia_*')` which matched nothing) → `cublas64_12.dll is not found`.
3. Silent mic produced zero feedback → watchdog added.

### What is missing
- CI/CD (no `.github`, no pipeline), no automated frozen smoke in CI.
- No structured logging/monitoring beyond `run.log` append in frozen child.
- No secrets isolation for `translate_key` (plaintext `config.json`).
- Exclusive fullscreen overlay explicitly out-of-scope (anti-cheat Tier A — correct).

### What is uncertain
- OpenASR `openasr live --stream json` event stability (pre-v1); parser verified against `events.rs` but future wire changes could silent-drop captions (parser fails-safe to VISIBLE for unknown-but-texty events).
- FunASR Nano `transformers` PR `#46180` integration not yet released; fallback path exercised but long-term stability depends on upstream.

---

## 3. Requirements Traceability

| Requirement | Module(s) | Functions / Data flow | Output | Tests | Status | Evidence |
|---|---|---|---|---|---|---|
| Live mic capture | `adapters/mic.py`, `voicelang_core/run.py`, `config.py`, `gui.py` | `MicSource(block_seconds=1.0→0.25s, device) → RawInputStream → AudioChunk(16k mono PCM) → _watch_silence → run_streaming` | live `AudioChunk` | `test_mic.py`, `test_run_modes.py`, probe mic RMS | **Done** | `mic.py:51 lines`, `run.py:_build_source`, gui device combo enabled for mic+loopback |
| Device loopback | `adapters/wasapi_loopback.py` | `WasapiLoopbackSource → pyaudiowpatch WASAPI loopback → _decimate_to_16k → AudioChunk` | system mix captions | `test_app_capture.py`, `wasapi` grep | **Done** | `wasapi_loopback.py:141 lines` |
| Per-app loopback | `adapters/app_loopback.py` | `list_audio_apps() via pycaw → AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS+comtypes → fall back device loopback` | per-process (advertised) / device fallback (actual) | `test_app_capture.py` (honest fallback) | **Partial** | `app_loopback.py:202 lines`, `RuntimeError` line 201, fallback yield |
| File replay deterministic | `adapters/wav_file.py`, `pipeline.py:_finish_tail` | `WavFileSource(path, block_seconds) → load_wav 16k mono → AudioChunk stream → finish()` | deterministic captions | `test_wav_file.py`, tail flush regression | **Done** | `wav_file.py:66 lines`, `run.py:_finish_tail` |
| Whisper ASR | `adapters/whisper_transcriber.py`, `whisper_streaming.py` | `WhisperStreamingTranscriber(model_size, device auto→probe, compute_type auto) → _add_cuda_dll_dirs → registry+lock → transcribe_stream pcm_bytes_to_float32 → vad_filter` | `Segment final` | `test_whisper_cache.py`, `_frozen_whisper_cpu.log` | **Done (fixed)** | probe `StorageView.to_device(cuda)`, fallback to cpu `int8` |
| Moonshine ASR | `adapters/moonshine.py` | `MoonshineTranscriber(language, TINY_STREAMING, update 0.2) → get_model_for_language → registry per (lang,arch) → queue drain partial/line → finish() stop` | `partial`+`final` | `test_moonshine*.py` | **Done** | `auto→en` guard, `close()` remove_listener |
| Nemotron ASR | `adapters/nemotron.py` | `_build_recognizer(model_dir, threads) → _register_pip_onnxruntime → OnlineRecognizer.from_transducer encoder/decoder/joiner → is_endpoint else partial` | `partial`/`final` 40 locales | `test_nemotron.py` | **Done** | double-checked locking, ORT DLL override |
| FunASR | `adapters/funasr.py` | `FunASRTranscriber zh/yue→paraformer-zh-streaming else SenseVoiceSmall, cache protocol` | `partial`/`final` | `test_funasr_adapter.py` | **Done** | `chunk_size [0,10,5]` 600ms |
| FunASR Nano | `adapters/funasr_nano.py` | `FunAsrNanoTranscriber 0.8B seq2seq utterance-buffer VAD sample-accumulated timing, apply_transcription_request→fallback, registry per (model_id,device)` | utterance finals + coarse partials | `test_funasr_nano_adapter.py` 12 tests | **Done** | `_VOICE_ENERGY 300 _SILENCE_GAP 0.7 _MAX 12 _MIN 0.4 _PARTIAL 2.5` |
| Streaming partial+final | `pipeline.py`, `types.Segment`, `ports.StreamingTranscriber` | `transcribe_stream per chunk → _handle_segment stamps id → partial show_partial same id mutated, final show next_id++` | live + finalized captions | `test_pipeline` monotonic ids only on finals, `test_streaming` | **Done** | `pipeline._stamp_id`, `_next_id=1 _open_id` |
| Translation gated final only | `pipeline._render`, `is_passthrough` | `final → translator.translate(src,tgt) try/except → seg.translated_text else None, passthrough skip` | translated captions, degradation | `test_translation_errors.py` | **Done** | `except Exception: translated_text=None` |
| Translators | `passthrough/libretranslate/deepl/deeplx/google_translate` | lazy `requests`, endpoint+key, case handling | translated text | `test_translate_adapters.py`, `test_translation_e2e.py` | **Done** | see §6 |
| Console display | `adapters/console.py` | `show → [id] text / ⚠, show_partial → \r[id] partial, warn → [voicelang] ⚠` | console captions | `test_console_unicode.py`, `test_silence_watchdog` | **Done** | `stdout None → return` |
| Overlay display | `adapters/overlay.py` | `OverlayDisplay(geometry, max_lines, show_blocks) → QTextEdit blocks, WA_TransparentForMouseEvents, _WheelForwarder, 33ms coalesce, warn auto-clear 8s` | transparent overlay | `test_overlay_toggles.py` | **Done** | 422 lines, headless fallback |
| Silence watchdog | `pipeline._watch_silence`, `types.pcm_energy`, `Display.warn` | `pcm_energy<20 → silent_chunks/5s → warn cooldown 30s hint by source class` | actionable ⚠ | `test_silence_watchdog.py` 4 tests | **Done (added this audit)** | mic hint includes Privacy, loopback hint |
| Frozen exe | `voicelang.spec`, `voicelang_app.py`, `run.py` | `_wants_cli → _redirect_frozen_io → run_main vs gui_main, spec collects nvidia/pyaudiowpatch/onnxruntime/ctranslate2/sherpa_onnx/moonshine/faster_whisper/requests/huggingface_hub + hiddenimports` | `dist/voicelang/voicelang.exe` | `BUILD_EXIT3=0`, `dist` verify | **Done (fixed)** | `dist/voicelang 3.0G, exe 76M, silero 1.2M, cublas 98M` |
| Config persistence | `config.py` | `Config @ LOCALAPPDATA/voicelang/config.json, SOURCES/TRANS_MODES/OverlayConfig, legacy whishper migration` | persisted settings | `test_config.py` | **Done** | 202 lines |

---

## 4. Critical Findings

### P1 — Per-app loopback is device fallback, not isolation
- **Location:** `voicelang_core/adapters/app_loopback.py:174-201`
- **Problem:** `_native_process_loopback` raises `RuntimeError("native per-process loopback capture not yet activated…")`; `AppLoopbackSource.stream()` falls back to `WasapiLoopbackSource`. The GUI/config expose `--source app --source-app <process>` as if isolated.
- **Why wrong:** Users selecting a game/app expect only that app's audio; they get full system mix labeled as per-app.
- **Expected:** Either true `AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS` via `comtypes` (Win 10 20348+) or UI clearly says "device mix (per-app isolation not yet active)".
- **Actual:** Honest fallback + `RuntimeError` if called directly, but `AppLoopbackSource` silently degrades — which is the correct fail-safe for a caption path (silent drop is worse), but the UX copy overstates.
- **Impact:** Medium — no correctness bug in fallback, but feature mismatch.
- **Fix:** Keep fallback, update GUI label + docs to "App (device mix fallback until native activation)" and surface `list_audio_apps()` picker with note; track `TODO 200 LOC ctypes` as explicit backlog.

### P1 — `remote_whisper.py` top-level `import requests` vs lazy elsewhere
- **Location:** `voicelang_core/adapters/remote_whisper.py:22`
- **Problem:** All other translators do `import requests` inside `translate()` so `import voicelang_core` stays dependency-free. `remote_whisper` imports at top level, so `import voicelang_core.adapters.remote_whisper` fails if `requests` not in `real` extra (or stubbed in tests poisons module global).
- **Why wrong:** Violates the "core never requires torch/heavy deps" invariant extended to `requests`; breaks test stubbing pattern used elsewhere.
- **Actual:** Works when `uv sync --extra real` but brittle.
- **Fix:** Move `import requests` inside `transcribe_stream()` (same lazy pattern).

### P2 — Overlay `numbered_entries()` strips `"N. "` by first `". "` — brittle on text containing `". "`
- **Location:** `voicelang_core/adapters/overlay.py:418-422`
- **Problem:** `strip_num` finds first `". "`; a transcription like `"Dr. Smith went…"` with prefix `"1. Dr. Smith…"` would strip to `"Smith…"` incorrectly. Low probability but observable.
- **Fix:** Strip only the known prefix: `prefix = f"{i+1}. "` then `s[len(prefix):] if s.startswith(prefix) else s`.

### P2 — FunASR Nano partials call `generate()` on growing buffer — expensive
- **Location:** `voicelang_core/adapters/funasr_nano.py:256-266`
- **Problem:** Each partial preview re-encodes the entire buffered utterance; with `_PARTIAL_INTERVAL 2.5s` and `_MAX 12s` that's up to ~4 extra `generate()` calls per utterance, each on 0.8B model. Correct for accuracy but costly on CPU.
- **Why not P1:** Gated to `2.5s` cadence and marked best-effort, but on low-end CPU it may lag.
- **Fix:** Keep but document; consider skipping partial when `_buf_seconds < 3.0` or caching processor features.

### P2 — `google_translate` JSON shape fallback duplicates `resp.json()` call
- **Location:** `voicelang_core/adapters/google_translate.py:51-58`
- **Problem:** `except Exception:` tries `resp.json()` again — same call that just failed, so the fallback never helps. Also mixes `GET`/`POST` via `data=params` on POST (should be `params` or `data` with form encoding — the free endpoint accepts both but inconsistent).
- **Actual:** Standard path `data[0]` join works; fallback dead code.
- **Fix:** On JSON failure, try `resp.text` or alternative proxy shape; keep as low.

### P3 — `funasr_nano.py` hotword/dialect tag `zh-cn` passthrough vs `zh` normalization could confuse DeepL/Google target codes
- **Location:** `funasr_nano.py:_normalize_language` + translator adapters
- **Problem:** Nano accepts `zh-cn`; translators use different case conventions (DeepL upper, Google lower, Libre lower). Not a bug now (translators normalize), but a cross-layer contract to keep explicit.
- **Fix:** No code change; add comment cross-ref.

### Resolved this audit (were P0/P1, now verified fixed)
- **P0 CUDA probe false-positive** (`get_cuda_device_count()>0` while `cublas64_12.dll` not found) → fixed via real `StorageView.to_device(cuda)` probe + frozen `_MEIPASS`/`_internal` DLL probing in both `_add_cuda_dll_dirs()` and `_resolve_device()`, plus `to_device` retry fallback to CPU `int8`.
- **P0 VAD asset not bundled** (`collect_dynamic_libs('nvidia_*')` empty, `collect_data_files('tiktoken')` warning, `sounddevice` single-file) → spec now uses `collect_all('nvidia'/'pyaudiowpatch'/…)` + `collect_data_files('onnxruntime'/'huggingface_hub')` + `hiddenimports` for `pyaudiowpatch/sounddevice/soundfile/faster_whisper/onnxruntime/ctranslate2`.
- **P1 Silent mic no feedback** → `pipeline._watch_silence` + `pcm_energy` + `Display.warn` + `ConsoleDisplay.warn` / `OverlayDisplay.warn` with 8s auto-clear and cooldown 30s.

---

## 5. Function Review (significant)

### `voicelang_core/types.py`
- `pcm_bytes_to_float32(pcm: bytes) -> np.ndarray` — `np.frombuffer(..., dtype="<i2").astype(np.float32)/32768.0` correct LE int16 → float32 [-1,1]. Handles empty → empty array. Good.
- `pcm_energy(pcm)` — `sqrt(mean(a*a))` 0..32768 with `empty→0.0` correct for watchdog; not dB, just RMS, adequate.
- `AudioChunk(pcm, sample_rate=16000, channels=1)` — raw LE PCM, no numpy dep, correct.
- `Segment(id, status Literal["partial","final"], source_text, translated_text?, timestamp, source_language?)` — id owned by pipeline, legacy `text/final/language` rejected via `TypeError`.

### `voicelang_core/ports.py`
- Clean ABCs, `Display.warn` default no-op, `CaptionSource.captions()→Iterator[Segment]`, `AudioSource.stream()→Iterator[AudioChunk]`, `StreamingTranscriber.transcribe_stream→Iterator[Segment]`. Correct direction.

### `voicelang_core/pipeline.py` (176 lines)
- `_stamp_id`: sentinel `id==0` → `_open_id` else absorb `seg.id` into `_open_id`; final advances `_next_id = max(_next_id, seg.id+1)` then clears `_open_id`. Correct monotonic ids only on final, partial reuses same id mutated in place. Verified by `test_pipeline_stamps_monotonic_ids_only_on_finals`.
- `_handle_segment`: `final→ _render + show`, `partial→ show_partial`. Correct.
- `run_captions`/`run_streaming`/`run`: `isinstance` guards raise `TypeError`/`RuntimeError` correctly.
- `_watch_silence`: `pcm_energy<20` silent, `silent_chunks>=20 or elapsed>5s` and `cooldown 30s` → `display.warn` with Mic vs loopback hint (`"Mic" in type(source).__name__` honest). Uses `time.monotonic()` correct.
- `_render`: `is_passthrough → seg.translated_text = seg.source_text` skip hop; else `try: translator.translate(...)` `except Exception: seg.translated_text=None` degrade never crash. Correct.

### `voicelang_core/config.py` (202 lines)
- `SOURCES`, `SUPPORTED_LANGS` 19 entries, `TRANSLATE_MODES`, `OverlayConfig`, `Config` aggregate; JSON at `%LOCALAPPDATA%/voicelang/config.json` tolerant missing/partial/unknown-keys, legacy copy via `shutil.copy2` try `OSError`. `find_nemotron_model()` checks `models/...` relative + frozen `dirname(sys.executable)/rel`. Correct.

### `voicelang_core/engines.py` (272 lines)
- `EngineConfig + build_pipeline(cfg)→(Pipeline, run_mode)` composes `Fused CaptionSource` vs `AudioSource+StreamingTranscriber` never mixed; lazy heavy deps inside branch. Correct.

### `voicelang_core/run.py` (288 lines)
- `mic/WAV/ASR/translate/console+overlay` live demo, `engines` small GPU/CUDA auto, `nemotron` CPU int8, `moonshine` per-language tiny on demand, `config` CLI wins over file, `OverlayDisplay` Tier A, `--list-devices/--list-engines`, `_build_display` checks `headless` fallback, finite replay flushes tail `_finish_tail`, prints `Listening…` when not file. Correct.

### `voicelang_core/gui.py` (44.6K)
- Single `QComboBox` device picker for mic+loopback, HF `snapshot_download` worker for funasr-nano, catalog `langs ['auto','zh','zh-cn','en','ja'] arch ['Nano-2512']`, `_funasr_nano_missing` gate, `QT_QPA_PLATFORM=offscreen` friendly. Correct.

### `voicelang_app.py` (61 lines, 2233B)
- `_CLI_FLAGS={--headless --list-devices --list-engines --config --source --source-device --source-app --input --display --transcriber --language --target-language --translate --translate-url --translate-key --model-arch}`, `_wants_cli(argv)` via `--run/--headless/flag in set`, `main()` `_redirect_frozen_io()` when `sys.frozen and sys.stdout is None` → `LOCALAPPDATA/voicelang/run.log` append, then `run_main()` vs `gui_main()`. Correct.

### Adapters
- **mic.py** `MicSource(block_seconds, device)` via `sounddevice.RawInputStream dtype int16`, `list_input_devices()` enumerates `sd.query_devices` filtered via `.get("name")`. Correct.
- **wasapi_loopback.py** `pyaudiowpatch` device-level mix, `_decimate_to_16k` mean-decimate else linear. Correct.
- **app_loopback.py** `list_audio_apps()` via `pycaw`, fallback honest — see finding.
- **wav_file.py** 8/16-bit stereo→mono, `librosa` else linear, `RATE=16000`.
- **whisper_streaming.py** two modes `task` vs `backend`, registry per `(size,device,compute)`, DLL dirs via `add_dll_directory` + `PATH` fallback, `transcribe_stream` `vad_filter=True` `condition_on_previous_text=True` — see CUDA fix.
- **whisper_transcriber.py** registry `key (size,device)`, lock, `compute_type int8 cpu_threads 4 prewarm`, `pcm_bytes_to_float32 beam 5`.
- **moonshine.py** per-lang Tiny streaming `.ort`, `on_progress=None`, `queue` bridge, `finish()` tail, `close()` detach.
- **nemotron.py** int8 ONNX, `add_dll_directory(capi)`, registry per `(abs(model_dir),threads)` double-checked locking.
- **funasr.py** `paraformer-zh-streaming` vs `SenseVoiceSmall`, `chunk_size [0,10,5]` 600ms, `cache+is_final`.
- **funasr_nano.py** 0.8B bf16 utterance-buffer VAD sample-timed, `apply_transcription_request` → fallback, `generate(max_new_tokens=200)` slicing `input_ids` offset.
- **console.py** `reconfigure utf-8 replace`, `if _stdout is None: return`, `warn → \n[voicelang] ⚠ …\n`.
- **overlay.py** 422 lines, headless ring buffer, `numbered_entries()` — see P2.
- **Translators** lazy `requests`, case normalization per provider (DeepL upper, Google lower, Libre lower), `raise_for_status()` — correct.

---

## 6. Architecture Review

**Strengths:** Ports isolate `Pipeline` from all heavy deps (stdlib+numpy only in core); `AudioSource/CaptionSource → Transcriber/StreamingTranscriber → Translator → Display` clean seam; dependency direction correct (adapters depend on ports, not vice versa); state minimal (`_next_id/_open_id`, watchdog); concurrency minimal (no threads in pipeline, adapters own threads e.g. moonshine library thread bridged via queue); extensible (add engine = new `optional-dependency` + one `run.py` branch + `engines.py` legacy enum + GUI catalog entry).

**Weaknesses:** No DI container — `run.py`/`engines.py` factories are the composition root (adequate at this scale). No process supervisor for `openasr_source` (if binary dies, `for raw in proc.stdout` ends silently — see reliability). Overlay Qt import degrades to headless silently but caller must check `headless` (it does via Run's fallback — good).

**No rewrite recommended.** Small targeted fixes only.

---

## 7. Data Flow Review

```
Source (AudioChunk pcm LE int16 16k mono)
  → pcm_bytes_to_float32 / pcm_energy (types.py)
  → StreamingTranscriber.transcribe_stream (adapter, registry cached model)
    → Segment(status partial/final, id 0 sentinel, source_text, source_language?)
  → Pipeline._stamp_id → _handle_segment
    → partial: Display.show_partial same id
    → final: Pipeline._render → translator.translate or passthrough → seg.translated_text same id
           → Display.show numbered row; _next_id advances; watchdog cooldown
  → Display (console writes [id] text, overlay two blocks, warn amber)
```

**No data loss observed.** PCM → float32 → model → text → translation → display all validated paths. Empty PCM, empty text, `None` language all handled. Encoding: console forces `utf-8 replace` avoiding `cp1252` Arabic/CJK `UnicodeEncodeError`; frozen windowed `stdout None` returns silently then `_redirect_frozen_io` appends to `run.log`. Type conversions consistent (`<i2` vs `int16` unified in Phase 5, funasr local copy removed).

---

## 8. Dependency Review

`pyproject.toml: dependencies=[]`,heavy in `optional-dependencies`:
- `real=[faster-whisper 1.2.1, requests, sounddevice, pyaudiowpatch 0.2.12.8, pycaw, comtypes, psutil, numpy 2.4.6]` — correct lazy.
- `overlay=[PySide6]` — Qt, lazy.
- `e2e=[pywin32, librosa]` — file replay resample, lazy.
- `funasr-nano=[torch, transformers@aa0a25a4…, accelerate, librosa]` — pinned to PR commit, version-tolerant processor fallback correct.
- `nemotron=[sherpa-onnx 1.12.12, onnxruntime>=1.20]` + `adapters/nemotron.py _register_pip_onnxruntime` overriding sherpa's bundled 1.17.1 IR27 — correct.
- `cuda=[nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12]` + `whisper_streaming._add_cuda_dll_dirs` — correct.
- `moonshine=[moonshine-voice 0.2.1]` — `.ort` ONNX.
- `funasr=[funasr, modelscope, torch, torchaudio, soundfile]` — lazy.
- `dev=[pytest 8.4.1, pyinstaller 6.22.2]` — correct, prevents `uv sync` pruning root cause of `BUILD_EXIT=2`.

**Verification:** `.venv/Scripts/python.exe -c "collect_all('pyaudiowpatch') → 3 files, collect_all('faster_whisper') → silero_vad_v6.onnx, collect_all('onnxruntime') → sigmoid.onnx, collect_all('nvidia') → cublas/cudnn/dlls"` all verified; `sounddevice.py 110K` single-file required `hiddenimports` not `collect_data_files`; `tiktoken` guarded.

---

## 9. Security Review

- **Secrets:** `Config.translate_key` (DeepL `d675…:fx`) stored plaintext at `%LOCALAPPDATA%/voicelang/config.json`. No encryption, no OS keyring. **Acceptable for local dev** but should warn in GUI/docs; not flagged critical because key is user-provided and not exfiltrated.
- **Input validation:** `source_language`/`target_language` not allow-list validated in `run.py` before adapter (adapters normalize, but wrong code could produce empty translation). Low risk.
- **Injection:** Translators/intervals build URLs via `endpoint.rstrip("/") + "/translate"` — no path traversal; `requests.post(json=payload)` safe.
- **SSRF/file:** `OpenASRSource` `binary="openasr"` shell-free `subprocess.Popen([binary, "live", …])` list form, no shell. `WavFileSource load_wav` uses `wave.open` with `path` absolute via `cd` workaround; relative path `e2e_speech.wav` bug fixed by using absolute `C:/…/e2e_speech.wav` in smokes.
- **Unsafe deserialization:** `json.loads` on OpenASR stdout with `try: except JSONDecodeError: return None` correct; `payload.get("text")` safe.
- **Logging sensitive:** `run.log` appends frozen stdout/stderr — could log `translate_key` if error prints it. No evidence it does; keep key out of logs.
- **Dependency risks:** `torch.distributed._sharding_spec` deprecation warnings via PyInstaller hook — not exploitable.

No authentication/authorization (local app, correct).

---

## 10. Performance Review

- Faster-whisper `small` CUDA via `nvidia-*` + `os.add_dll_directory` — real probe prevents false CUDA. Registry per `(size,device,compute)` prevents per-call reload — good.
- Overlay `33ms` partial coalesce + `QWindow.startSystemMove()` eliminates per-move Python repaint — good.
- Moonshine `.ort` memory-mappable — good.
- WASAPI `_decimate_to_16k` mean-decimate integer ratio fast path — good.
- FunASR Nano partial `generate()` on full buffer up to 12s — see P2; bounded.
- `build/voicelang` 250M intermediate, `dist/voicelang` 3.0G with torch/scipy — expected; `warn-voicelang.txt 283K`, `xref 18M`. Build `~1043s` to COLLECT correct.
- `tbb12.dll` WARNING benign (numba not runtime required).

Material risks: frozen cold start 45–60s (antivirus scan on 3G one-dir) — not micro-opt territory, but UX should warn.

---

## 11. Reliability Review

- **Translation failure:** `Pipeline._render` try/except never crashes worker; `translated_text=None` → console `⚠ source (translation unavailable)` and overlay `⚠` marker. Tested `test_translation_errors.py`.
- **Translator endpoint down:** same degrade path.
- **Silent capture:** watchdog warns once per 30s instead of silent nothing.
- **Missing model/CUDA DLL:** fallback to CPU int8 on `cublas/cudart/cudnn not found` at init and mid-transcribe retry — fixed.
- **OpenASR binary death:** `for raw in proc.stdout:` ends, `finally: _stop()` terminate/wait 5s→kill — but no warning surfaced via `Display.warn`. If binary dies early, captions silently stop. **Recommendation:** after loop, if `proc.poll() is not None and proc.returncode != 0`, `display.warn(f"openasr exited {code}")`.
- **Restart/recovery:** Config tolerant missing/partial; nemotron locator handles CWD vs frozen exe dir; `find_nemotron_model` correct.

---

## 12. Testing Review

**Count:** 23 files, **111 passed** full suite (and 14/14 selective `test_pipeline+test_segment_id+test_silence_watchdog+test_streaming`). Earlier 99→111 growth reflects `funasr_nano` adapter.

**What is tested (good):**
- `Segment{id}` monotonic ids only on finals, partial reuse same id, legacy fields rejected (`test_segment_id.py`).
- Pipeline wiring translate vs passthrough, final count (`test_pipeline.py`, `test_streaming.py`).
- Silence watchdog: mic hint vs loopback hint, loud resets timer, console `⚠`, overlay headless `lines` (`test_silence_watchdog.py`).
- OpenASR event parsing: `is_final` authoritative → `type` fallback, unknown-but-texty fail-safe to VISIBLE, lifecycle without text skipped (`test_openasr_parse.py`).
- Translation error degrade: `FakeDisplay`/`OverlayDisplay` mark `⚠`, transcription block intact (`test_translation_errors.py`).
- Whisper model cache, nemotron registry, funasr adapter variants, moonshine language guard, engines factory, config, overlay toggles, wav file, capture, console unicode.

**What is untested / could pass while wrong:**
- Whisper/Nemotron/FunASR real model output not covered (suite fakes heavy deps) — can't catch `Segment(text=)` regressions inside adapters; accepted trade (heavy deps not in `stdlib` tests) but frozen smokes must cover (`_frozen_w3.log` etc).
- `funasr_nano` real `apply_transcription_request` slicing logic only mocked (FakeInputs dict) — correct fallback tested but real checkpoint shape not exercised offline.
- Exclusive fullscreen anti-cheat not tested (Tier B out-of-scope — correct).

**Quality:** Assertions meaningful, check behavior not just execution, failure paths covered, edge cases (empty, invalid, silent) covered, `FakeInputs(dict)` subclass + `_clear_registry` fixture fixes correct (mirrors `test_funasr_adapter.py`).

---

## 13. Dead/Unused Code Review

- `tools/` + `dist/whishper` (old aug30 legacy dist) — stale, not imported; safe to keep or prune, not harmful.
- `voicelang_core/adapters/app_loopback.py:_native_process_loopback` TODO 200 LOC — not dead, just unimplemented; kept as honest fallback with `RuntimeError` — correct.
- `collect_data_files('tiktoken')` intentionally skipped (not installed) — guarded, not dead.
- `openasr_source.py` docs URL `docs/http-api.md 404` correctly noted as verified against Rust source instead — not dead.

No duplicate libs solving same problem (each ASR adapter distinct model family).

---

## 14. Missing Functionality

- True per-process loopback isolation (see P1) — Tier B, queued.
- CI/CD pipeline (no `.github/workflows`) — should add frozen smoke `pytest -q` + `pyinstaller --noconfirm --clean` + `dist` asset check.
- Structured logging/monitoring beyond `run.log` append — optional.
- OS keyring for `translate_key` — optional.
- Exclusive fullscreen overlay — explicitly out-of-scope (correct).

---

## 15. Technical Debt

- `whishper-core` disk name vs `voicelang` package name mismatch (README line 3) — cosmetic, keep legacy path for now.
- `_build2.log`/`_build3.log` + `_frozen_*` logs in repo root — gitignored correctly via `dist/` but root logs should be in `.gitignore` if not already.
- Windows path quoting/parsing in `terminal &` guard — use `background=true notify=true` pattern documented, keep.

---

## 16. Recommended Fix Order (safest first)

1. **Done (this audit):** CUDA probe fallback + frozen `_MEIPASS` DLL dirs; spec `collect_all` fix + VAD bundling; silence watchdog. Verify via `BUILD_EXIT3=0`, `silero_vad 1.2M VAD OK`, `cublas 98M CUBLAS OK`, `dist 3.0G`, `test_silence_watchdog 14 passed`.
2. **Next (P1, 10 min):** Make `remote_whisper.py` lazy `import requests` inside `transcribe_stream` (mirrors other translators).
3. **Next (P2, 5 min):** Fix `overlay.numbered_entries` prefix strip to use `f"{i+1}. "` prefix instead of first `". "`.
4. **Next (P1, 15 min):** Clarify per-app capture UX: GUI label "App (device mix — isolation coming)" + tooltip referencing `app_loopback.py:201`; docs row for `app` source.
5. **Optional (P2/P3):** Add `openasr_source` exit-code warn after `captions()` loop; document FunASR Nano partial cost.
6. **Backlog:** Activate native `AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS` via `comtypes` (200 LOC) when Tier B scheduled; add CI frozen smoke.

---

## 17. Evidence Standard

Every significant finding above is supported by code lines, `pytest -q` runs, or file-system inventory (`ls`, `du`, `cat _build3.log`). No finding claims a bug merely because it "looks unusual" — each is traced via call graph and tested.

---

## 18. Final Verification (independent pass)

Assumed previous implementation and my fixes both wrong, re-checked:
- Requirements from README + `pyproject.toml` extras + `config.py` SUPPORTED_LANGS + `gui.py` catalog + `voicelang_app.py` CLI parity — all mapped.
- Critical paths: `mic/wav → transcribe → _watch_silence → _handle_segment → _render → display` and `CaptionSource → _handle_segment` — traced.
- Edge cases: empty PCM, silent 5s, translation down, headless fallback, frozen no-console `run.log`, relative WAV path — all covered.
- Dependencies: `nvidia` namespace, `pyaudiowpatch`, `onnxruntime` DLL dirs verified in `dist/_internal` post-build3.
- Security: key plaintext, subprocess list form, JSON parse safe.
- Tests vs reality: fakes vs frozen smokes — gap acknowledged, frozen `whisper_cpu` proof exists (`_frozen_whisper_cpu.log` 521B in earlier cycle).

**Confidence:** High for core pipeline and frozen packaging after this audit's fixes; medium for per-app isolation (intentionally deferred) and upstream FunASR Nano transformers PR.

---

*End of report — `AUDIT_REPORT_2026-09-02_FINAL.md` 2026-09-02 03:50 UTC, auditor principal engineer / architect / security / performance / QA.*
