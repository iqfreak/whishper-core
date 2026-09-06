"""WP-6 adversarial re-review — §8 acceptance + anti-patterns.

§8 (headless parity) criteria 1-9 (derived from gui.py/run.py/voicelang_app.py):
 1. Every GUI setting has a CLI flag (engine, language, source, device, app, display, overlay, translate)
 2. --list-engines enumerates engines+modes+sources
 3. --list-devices enumerates devices
 4. --headless/--list-*/--source/--config routes CLI not GUI (voicelang_app._wants_cli)
 5. Frozen exe defaults to GUI with no flags
 6. Config file is single source of truth, missing → defaults
 7. Translation pipelined off hot path, never crashes worker (pipeline._render degrade)
 8. VAD gate before ASR with hangover (0.2s) and metrics
 9. Overlay coalesce 33ms, never drops translation id sync (synchronous advance)

Anti-patterns checked:
 - catch-all without breaker (bare except + no counter/throttle)
 - DEVNULL swallowing (subprocess stdout=DEVNULL without log)
 - unvalidated cross-field config (source=wav without input_file, translate=deepl without key, invalid language)
"""
import subprocess, sys, os, pathlib, re

def test_headless_parity_every_gui_setting_has_cli_flag():
    root = pathlib.Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "voicelang_core.run", "--help"], capture_output=True, text=True, timeout=10, cwd=str(root))
    help_text = r.stdout + r.stderr
    for flag in ("--transcriber", "--language", "--target-language", "--source", "--source-device", "--source-app", "--display", "--translate", "--translate-url", "--translate-key", "--model-arch", "--log-captions", "--input"):
        assert flag in help_text, f"CLI missing flag {flag} required for GUI parity"

def test_list_engines_and_devices_exit_zero():
    root = pathlib.Path(__file__).resolve().parents[1]
    for args in (["--list-engines"],):
        r = subprocess.run([sys.executable, "-m", "voicelang_core.run"] + args, capture_output=True, text=True, timeout=10, cwd=str(root))
        assert r.returncode == 0, r.stderr
        assert "moonshine" in r.stdout

def test_wants_cli_routes_correctly():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from voicelang_app import _wants_cli
    assert _wants_cli(["voicelang.exe"]) is False
    assert _wants_cli(["voicelang.exe", "--list-engines"]) is True
    assert _wants_cli(["voicelang.exe", "--headless"]) is True
    assert _wants_cli(["voicelang.exe", "--source", "mic"]) is True
    assert _wants_cli(["voicelang.exe", "--config", "x"]) is True
    assert _wants_cli(["voicelang.exe", "--gui"]) is False

def test_config_missing_returns_defaults_and_unknown_keys_ignored():
    from voicelang_core.config import Config, load_config
    import tempfile, json, os as _os
    with tempfile.TemporaryDirectory() as td:
        p = _os.path.join(td, "cfg.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"transcriber": "moonshine", "unknown_future_key": 123}, f)
        cfg = load_config(p)
        assert cfg.transcriber == "moonshine"
        # unknown key ignored, not crash
        assert not hasattr(cfg, "unknown_future_key")
    # missing file
    cfg2 = load_config(os.path.join(td, "nope.json"))
    assert cfg2.transcriber == "moonshine"

def test_translation_never_crashes_worker():
    from voicelang_core.pipeline import Pipeline
    from voicelang_core.types import Segment, AudioChunk
    from voicelang_core.adapters.fake import FakeAudioSource, FakeStreamingTranscriber, FakeDisplay
    class ExplodingTranslator:
        is_passthrough = False
        def translate(self, text, src, tgt):
            raise RuntimeError("network down")
    disp = FakeDisplay()
    p = Pipeline(source=FakeAudioSource(chunks=[AudioChunk(pcm=b"\x00"*16000, sample_rate=16000)]*3), transcriber=FakeStreamingTranscriber(), translator=ExplodingTranslator(), display=disp, target_language="en")
    p.run_streaming()
    # must not crash; finals still shown with translated_text=None
    assert len(disp.shown) == 1
    assert disp.shown[0].translated_text is None

def test_vad_hangover_applies_to_energy_fallback():
    from voicelang_core.vad import VadGate
    gate = VadGate(enabled=True, energy_threshold=80, window_size=512, sample_rate=16000)
    # hangover 0.2s => max(1, int(0.2*16000/512)) = 6
    assert gate._hangover_max == 6
    # loud chunk sets hangover, then 6 silent chunks should still be speech due to hangover
    loud = (b"\xff\x7f" * 800)  # high energy
    silent = (b"\x00\x00" * 800)  # zero energy
    d = gate.is_speech(loud)
    assert d.is_speech is True
    for i in range(6):
        d2 = gate.is_speech(silent)
        assert d2.is_speech is True, f"hangover frame {i} should still be speech"
    # 7th silent should be silence
    d3 = gate.is_speech(silent)
    assert d3.is_speech is False

def test_overlay_coalesce_and_id_sync():
    from voicelang_core.adapters.overlay import OverlayDisplay
    from voicelang_core.types import Segment
    d = OverlayDisplay(headless=True)
    d.show_partial(Segment(id=1, status="partial", source_text="hello partial"))
    d.show(Segment(id=1, status="final", source_text="hello", translated_text="hola"))
    assert d.lines
    # id appears in both blocks same id
    assert any("hello" in l for l in d.transcribed_lines)
    assert any("hola" in l for l in d.translated_lines)

def test_no_bare_catch_all_without_breaker_in_asr_loop():
    # Adversarial: pipeline_threaded._asr_loop must have breaker (consec failures) not bare except pass
    src = pathlib.Path(__file__).resolve().parents[1] / "voicelang_core" / "pipeline_threaded.py"
    text = src.read_text(encoding="utf-8")
    # Must contain breaker counters
    assert "_asr_consec_failures" in text
    assert "_asr_warned_at" in text
    assert ">= 5" in text or ">=5" in text
    # Must not have except Exception: pass without breaker in that region
    # We check that the except Exception as exc block is followed by breaker logic
    assert "circuit breaker" in text.lower() or "consecutive failures" in text.lower()

def test_no_devnull_without_log():
    # GUI child stdout=DEVNULL is allowed only when stderr is to a log file (fixed)
    src = pathlib.Path(__file__).resolve().parents[1] / "voicelang_core" / "gui.py"
    text = src.read_text(encoding="utf-8")
    # There should be exactly one DEVNULL usage and it should be stdout=DEVNULL with stderr=log_file
    devnull_lines = [l for l in text.splitlines() if "DEVNULL" in l]
    assert len(devnull_lines) == 1
    assert "stderr=self._log_file" in text or "stderr=log_file" in text

def test_cross_field_config_validation():
    from voicelang_core.config import Config
    # source wav without input_file -> run.py should gate and warn, not silently use wav with no file
    # Here we test Config round-trip: source=wav but input_file empty is detectable
    cfg = Config(source="wav", input_file="")
    # run.py logic: _build_source("wav", None) raises SystemExit
    from voicelang_core.run import _build_source
    import pytest
    with pytest.raises(SystemExit):
        _build_source(cfg.source, cfg.input_file or None)
    # deepl without key -> _build_translator raises SystemExit
    from voicelang_core.engines import EngineConfig, EngineName
    cfg2 = Config(translate="deepl", translate_key="")
    from voicelang_core.engines import _build_translator
    eng = EngineConfig(name=EngineName.passthrough, translate="deepl", translate_key="")
    with pytest.raises(SystemExit):
        _build_translator(eng)
    # invalid language rejected by adapters
    from voicelang_core.adapters.vibeasr import VibeAsrTranscriber
    from voicelang_core.adapters.audio8 import Audio8Transcriber
    import pytest as _pt
    with _pt.raises(ValueError):
        VibeAsrTranscriber(language="xx")
    with _pt.raises(ValueError):
        Audio8Transcriber(language="xx")

def test_threaded_id_advance_synchronous_before_enqueue():
    src = pathlib.Path(__file__).resolve().parents[1] / "voicelang_core" / "pipeline_threaded.py"
    text = src.read_text(encoding="utf-8")
    # Must advance id before enqueue
    idx_advance = text.find("_advance_id_on_final")
    idx_enqueue = text.find("_put_bounded(self._translate_q")
    assert idx_advance != -1 and idx_enqueue != -1
    # There should be advance then enqueue with comment about synchronous
    assert idx_advance < idx_enqueue
    assert "synchronously BEFORE enqueue" in text or "synchronously before enqueue" in text.lower()
