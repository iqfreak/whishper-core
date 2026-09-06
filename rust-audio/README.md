# voicelang-audio — Rust hot path

Targets <100ms mic->partial: **IAudioClient3** event-driven low-latency shared mode
(`GetSharedModeEnginePeriod` -> ~10ms period, `WaitForSingleObject(hEvent)`),
not Python `sounddevice.RawInputStream.read(250ms)`.

## Build

```powershell
uv run maturin develop --release
# or
cargo build --release
```

## Python usage (after maturin)

```python
from voicelang_audio import AudioCapture
cap = AudioCapture(source="app", process="Discord.exe", block_ms=80)
for pcm_bytes_16k in cap:
    segs = asr.transcribe_stream(AudioChunk(pcm=pcm_bytes_16k))
```

## Per-app isolation

Windows 10 2004+ (build 19041): `AUDCLNT_STREAMFLAGS_LOOPBACK | AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS`
via `ActivateAudioInterfaceAsync`. Include `TargetProcessId` + `PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE`.
Fallback: device-level loopback (captures all apps on device — honest, not isolated) — same fallback
as `voicelang_core/adapters/app_loopback.py` but via Rust COM instead of `pycaw`/`comtypes`.

## Gaming coexistence

Capture thread: `THREAD_PRIORITY_TIME_CRITICAL`, ASR on low-priority CUDA stream,
`rtrb` ring between them, `queue maxsize 2` semantics. 48k Float32 -> mono mean
-> mean-decimate integer factor 3 to 16k int16 (0.08ms), not Rubato Sinc 256.
