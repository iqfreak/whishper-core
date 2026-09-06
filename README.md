# voicelang

> **Main repo folder:** `C:\Users\ahmed\voicelang`  
> This is the sole project directory for the streaming pipeline, GUI, packaged app, and all adapters.

Streaming **capture → transcribe → translate → display** as pluggable ports:
`AudioSource / CaptionSource → Transcriber / StreamingTranscriber → Translator → Display`.

Designed as the core of a Discord voice mod or a game overlay
(Tier A: windowed / borderless only — exclusive fullscreen is out of scope, anti-cheat risk).

---

## Quickstart

```bash
# One-shot install (models download on first use):
uv sync --extra real --extra overlay --extra e2e --extra nemotron --extra cuda --extra moonshine --extra funasr-nano --group dev
# Optional: FunASR (Paraformer / SenseVoice, ~220M streaming for Mandarin)
uv sync --extra funasr

# Live microphone -> captions in the console (GPU when available):
uv run python -m voicelang_core.run

# Same, with the transparent always-on-top overlay window:

- **Overlay v2 (smooth):** fixed-size caption panels with a full scrollable
  history — the mouse wheel scrolls, long sentences wrap to the next line, and
  auto-scroll follows new captions until you scroll up. Drag is OS-composited
  (`QWindow.startSystemMove`) so moving the overlay is lag-free even on
  translucent windows. Partial captions coalesce on a 33 ms timer.
uv run python -m voicelang_core.run --display overlay

# Deterministic demo from a WAV file (no mic needed), console:
uv run python -m voicelang_core.run --input e2e_speech.wav
uv run python -m voicelang_core.run --source wav --input e2e_speech.wav

# Capture game/Discord audio instead of the mic (WASAPI loopback):
uv run python -m voicelang_core.run --source loopback --source-device "Headphone (Realtek(R) Audio) [Loopback]"
# Per-app capture — just like Discord screenshare's source picker:
uv run python -m voicelang_core.run --source app --source-app "Discord.exe"
uv run python -m voicelang_core.run --source app --source-app "pid:1234"

# ASR engines
uv run python -m voicelang_core.run --transcriber moonshine --language ar    # tiny 34M per-language (default)
uv run python -m voicelang_core.run --transcriber whisper --input e2e_speech.wav
uv run python -m voicelang_core.run --transcriber nemotron --input e2e_speech.wav
uv run python -m voicelang_core.run --transcriber funasr --language zh       # FunASR Paraformer-zh-streaming
uv run python -m voicelang_core.run --transcriber funasr-nano --language zh  # Fun-ASR-Nano 0.8B (HF Transformers)
uv run python -m voicelang_core.run --transcriber vosk --language en         # Vosk small offline (50MB, CPU, no torch)
uv run python -m voicelang_core.run --transcriber vibeasr-bitnet --language en  # VibeASR-BitNet 1.58GB GGUFs, CPU RTF 0.77
uv run python -m voicelang_core.run --transcriber audio8 --language zh       # Audio8 0.1B (Transformers eager, 30s cap)

# Translation (all five modes; pipeline auto-applies on finals only)
uv run python -m voicelang_core.run --translate libretranslate --translate-url http://127.0.0.1:5000 --target-language fr --input e2e_speech.wav
uv run python -m voicelang_core.run --translate deepl --translate-url https://api-free.deepl.com --translate-key $DEEPL_KEY --target-language de
uv run python -m voicelang_core.run --translate deeplx --translate-url http://127.0.0.1:1188 --target-language ja
uv run python -m voicelang_core.run --translate google --target-language es --input e2e_speech.wav  # free undocumented

# Tests + full E2E (SAPI TTS -> real ASR -> real display):
uv run python -m pytest -q
uv run python e2e_speech_test.py

# Settings GUI (engine, language, capture source, overlay placement, translation, launch):
uv run python -m voicelang_core.run --gui

# Headless discovery (spec §8 parity — every GUI setting has a CLI flag):
uv run python -m voicelang_core.run --list-engines
uv run python -m voicelang_core.run --list-devices
# Packaged exe: `voicelang.exe --headless --list-engines` — ANY headless flag
# (--headless/--list-*/--source/--config/...) routes past the GUI to the CLI.
# With no flags, voicelang.exe opens the settings GUI.
```

GUI now has a **Translation** section: mode (Passthrough / LibreTranslate / DeepL / DeepLX / Google), target language, endpoint URL, and API key (DeepL). Values persist in `config.json` and round-trip through the CLI (`--translate`, `--translate-url`, `--translate-key`, `--target-language`).

---

## Desktop app (packaged)

`dist/voicelang/voicelang.exe` is the standalone app — no Python needed. It opens
the settings GUI; **Start** launches the capture pipeline as a child process
with the saved config. Defaults: Moonshine engine (tiny per-language models,
downloaded on demand), microphone input. Translation + per-app capture are fully
wired in the GUI and the frozen runner (`--run`).

To rebuild from source (kill any running voicelang.exe first):

```bash
taskkill /F /IM voicelang.exe  # then:
uv run pyinstaller --noconfirm --clean --windowed --name voicelang \
  --collect-all sherpa_onnx --collect-binaries ctranslate2 \
  --collect-all moonshine_voice --collect-data sounddevice \
  --collect-all faster_whisper --collect-data onnxruntime --collect-binaries onnxruntime \
  --collect-data tiktoken --collect-all requests \
  --collect-binaries nvidia_cublas_cu12 \
  --collect-binaries nvidia_cuda_runtime_cu12 \
  --collect-binaries nvidia_cudnn_cu12 \
  --collect-data huggingface_hub \
  --icon assets/logo.ico --add-data "assets/logo.png;assets" \
  voicelang_app.py
```

Nemotron's model directory lives next to the exe under `models/sherpa-onnx-nemotron-3.5-...`.
Whisper and Moonshine models download to their normal user caches on first use.
FunASR models download to `~/.cache/modelscope` on first use.
Pipeline console output (windowed build) lands in `%LOCALAPPDATA%/voicelang/run.log`.

Everything the GUI sets is saved to `%LOCALAPPDATA%/voicelang/config.json`,
which the CLI reads automatically — explicit CLI flags win over the file.

```jsonc
// %LOCALAPPDATA%/voicelang/config.json
{
  "display": "overlay",                       // console | overlay | file
  "transcriber": "moonshine",                 // moonshine | whisper | nemotron | funasr | funasr-nano | vosk | vibeasr-bitnet | audio8
  "language": "en",                           // tag: ar de en es ja ko tl uk vi zh etc per engine
  "model_arch": "TINY_STREAMING",
  "source": "mic",                            // mic | loopback | wav | app
  "source_device": "",                        // loopback device NAME; "" = default output
  "source_app": "Discord.exe",                // when source == "app": "Discord.exe" or "pid:1234"
  "input_file": "",                           // WAV path when source == "wav"
  "target_language": "en",                    // output language for translation
  "translate": "passthrough",                 // passthrough | libretranslate | deepl | deeplx | google
  "translate_url": "",                        // endpoint: LibreTranslate/DeepL/DeepLX/Google base URL
  "translate_key": "",                        // API key for deepl (DeepL-Auth-Key)
  "overlay": { "x": 200, "y": 100, "width": 700, "height": 150,
               "opacity": 0.9, "max_lines": 5,
               "show_transcription_block": true,
               "show_translation_block": true }   // omit x/y for auto bottom-center; blocks toggle independently
}
```

---

## Selecting an app to capture (like Discord screenshare)

The GUI's **Capture source → Application audio (per-app)** dropdown is the Discord-like picker.
It enumerates current audio sessions via `pycaw` (`AudioUtilities.GetAllSessions`) + `psutil`, showing
running apps such as `Discord.exe (pid 1234)`. Pick one, or type `Discord.exe` / `pid:1234`.

```python
from voicelang_core.adapters.app_loopback import list_audio_apps, AppLoopbackSource
apps = list_audio_apps()           # [{"pid": 1234, "name": "Discord.exe", ...}, ...]
src  = AppLoopbackSource(app="Discord.exe")  # or "pid:1234"
for chunk in src.stream(): ...     # int16 @ 16 kHz
```

- True per-process isolation uses `AUDCLNT_STREAMFLAGS_LOOPBACK | AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS` (Windows 10 build 20348+ / Windows 11). On capable Windows with `pycaw` + `comtypes` the capture targets only that app's mix.
- On older Windows or when `pycaw/comtypes` is absent, the pipeline falls back honestly to device-level WASAPI loopback (`WASAPILoopbackSource`) and labels the stream as such — same UX, no silent wrong-app capture.
- Device loopback always works: `--source loopback` captures the whole output device via `pyaudiowpatch` (true WASAPI loopback at native rate, then `_decimate_to_16k`).

---

## Translation providers

| Mode | Adapter | Endpoint | Auth | Notes |
|---|---|---|---|---|
| `passthrough` | `PassthroughTranslator` | — | — | No network; marks text as `is_passthrough`. |
| `libretranslate` | `LibreTranslateTranslator` | `http://host:5000` | none (self-host) | POST `{endpoint}/translate` → `translatedText`. Public demos are unreliable — run locally. |
| `deepl` | `DeepLTranslator` | `https://api-free.deepl.com` or `https://api.deepl.com` | `DeepL-Auth-Key <key>` | POST `{endpoint}/v2/translate` → `translations[0].text`. `--translate-key` required. |
| `deeplx` | `DeepLXTranslator` | `http://127.0.0.1:1188` | none | POST `{endpoint}/translate` → `{"code":200,"data":"..."}`. Self-hosted DeepLX. |
| `google` | `GoogleTranslator` | `https://translate.googleapis.com` | none (free) | GET `…/translate_a/single?client=gtx&…` → nested JSON. Large payloads POST. No key. |

All translators are lazy (`import requests` inside `translate()`), implement `is_passthrough`, and are exercised end-to-end by `tests/test_translate_adapters.py` via localhost mock servers into `Pipeline.run_streaming` → `OverlayDisplay` (headless).

---

## ASR engines

| Engine | Runtime | Quality | Streaming partials | Multilingual |
|---|---|---|---|---|
| faster-whisper `small` | ctranslate2, CUDA float16 / CPU int8, auto-probed | good | batch-per-block (no live partials) | 99 langs, `task=translate` one-pass-to-English |
| Nemotron-3.5-ASR-Streaming-0.6B | sherpa-onnx, ONNX int8 (CPU) | **excellent** (punctuation, caps) | **true per-chunk partials + endpoint finals** | 40 locales (incl. Arabic), `language=auto` |
| **Moonshine Voice (default)** | moonshine-voice, .ort int8 (CPU) | good-to-excellent per language | **true partials + line finals** | en/ar/de/es/ja/ko/tl/uk/vi/zh — tiny 34M models, **downloaded on demand** |
| **FunASR** | funasr + modelscope + torch | good (Mandarin specialist) | **true streaming** via Paraformer-zh-streaming (cache + is_final) for `zh/yue`; batch-per-chunk SenseVoiceSmall for other langs | zh/yue/ja/ko/en… |
| **Fun-ASR-Nano** | transformers (PR #46180) + torch | excellent for zh/en/ja (26 Chinese accents) | utterance-buffered seq2seq: energy VAD + partial previews | zh/zh-cn/en/ja |
| **Vosk small** | vosk, CPU Kaldi (no torch) | good offline | **true streaming** `AcceptWaveform` partials/finals | en/de/es/fr/pt/it/zh/ja/ru/tr/ko/hi/nl/pl/uk/vi — ~50MB each, downloaded on demand |
| **VibeASR-BitNet** | VibeASR.cpp GGUF (1.58GB, CPU, 3t RTF 0.77) | good | utterance-level (non-streaming) | en/zh/fr/it/ko/pt/vi + auto |
| **Audio8 0.1B** | transformers eager greedy (GPU-preferred, 30s cap) | good | utterance-level (non-streaming) | zh/en/fr/de/ja/ko/yue + auto |
| Qwen3-ASR-0.6B (trial) | llama.cpp GGUF Q8_0, CUDA | good | via llama-mtmd-cli interactive | 30 langs + 22 Chinese dialects |

Moonshine is the small-model-per-language engine: `--transcriber moonshine --language ar` pulls a ~34 MB streaming model into the local cache on first use (`on_progress` in the GUI), then runs offline. Tiny 34M models beat whisper-small on their native languages (e.g. Mandarin CER 16.1%, Arabic WER 15.5%).

FunASR routing (see `adapters/funasr.py`): `zh|zh-cn|yue` → `paraformer-zh-streaming` (220 M, true streaming with `cache` + `chunk_size=[0,10,5]`), else → `iic/SenseVoiceSmall` (multilingual, `language="auto"` supported). Heavy deps stay lazy; importing `voicelang_core` never requires `torch`.

Fun-ASR-Nano (`--transcriber funasr-nano`): the HF-Transformers build of Fun-ASR-Nano 0.8B (zh/en/ja). It is a seq2seq model, so the adapter energy-buffers utterances and runs `generate()` once per utterance (silence-gap 0.7 s, hard cap 12 s, partial previews every 2.5 s). Deps: `uv sync --extra funasr-nano` (torch + transformers PR build + accelerate + librosa). Not bundled into the frozen exe (torch is opt-in); run from source with the extra.

```bash
uv sync --extra funasr   # torch + modelscope + funasr
uv run python -m voicelang_core.run --transcriber funasr --language zh --input e2e_speech.wav

# Fun-ASR-Nano (0.8B, zh/en/ja) — needs the transformers PR #46180 build
uv sync --extra funasr-nano
uv run python -m voicelang_core.run --transcriber funasr-nano --language zh
```

### GPU (CUDA) — no system CUDA toolkit needed

The RTX path works out of the box once the `cuda` extra is installed:
`nvidia-cublas-cu12 + nvidia-cuda-runtime-cu12 + nvidia-cudnn-cu12` put the
compute DLLs in site-packages; `whisper_streaming.py` registers them via
`os.add_dll_directory` and probes CUDA with a real tensor copy (not a device
count — counts lie). `device="auto"` resolves to `cuda`/`float16` when the
probe succeeds, else `cpu`/`int8`.

### Nemotron model files (not pip-installable)

```bash
curl -L -o n.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2
mkdir -p models && tar xjf n.tar.bz2 -C models
```

### Qwen3-ASR (trial)

```bash
# llama.cpp CUDA build (tools/llama) + model auto-download:
./tools/llama/llama-mtmd-cli.exe -hf ggml-org/Qwen3-ASR-0.6B-GGUF --audio e2e_speech.wav -ngl 99
```

---

## Architecture

- `voicelang_core/ports.py` — the 5 ports; the pipeline depends only on these.
- `voicelang_core/pipeline.py` — `run_captions()` / `run_streaming()` / `run()`.
- `voicelang_core/engines.py` — engine selection / factory wiring.
- `voicelang_core/adapters/` — mic, WAV file, WASAPI loopback, **per-app loopback**, faster-whisper, Nemotron, Moonshine, **FunASR**, **Vosk**, **VibeASR-BitNet**, **Audio8**, OpenASR subprocess, **LibreTranslate / DeepL / DeepLX / Google**, console, overlay.
- **Full pipeline:** `AudioSource (mic|loopback|wav|app) → StreamingTranscriber (whisper|nemotron|moonshine|funasr|funasr-nano|vosk) + utterance engines (vibeasr-bitnet|audio8) → Translator (passthrough|libretranslate|deepl|deeplx|google) → Display (console|overlay|file)`. Verified E2E by `pytest` (150 tests) + `tests/test_translation_e2e.py` + localhost translation servers + `WavFileSource` replay.
- Core imports only stdlib + numpy; all heavy deps are lazy / optional extras.

See `voicelang_core_architecture_review.html` for the full review with diagrams.
