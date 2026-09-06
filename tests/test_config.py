"""Config: defaults, round-trip, tolerance of missing/partial files."""
import json

from voicelang_core.config import Config, OverlayConfig, load_config, save_config


def test_defaults():
    cfg = Config()
    assert cfg.display == "console"
    assert cfg.transcriber == "moonshine"
    assert cfg.language == "en"
    assert cfg.model_arch == "TINY_STREAMING"
    assert cfg.source == "mic"
    assert cfg.overlay.max_lines == 5
    assert cfg.overlay.x is None  # auto-place


def test_roundtrip(tmp_path):
    cfg = Config(
        display="overlay",
        transcriber="moonshine",
        language="ar",
        model_arch="TINY_STREAMING",
        source="loopback",
        source_device="Headphone (Realtek(R) Audio) [Loopback]",
        input_file="demo.wav",
        overlay=OverlayConfig(x=120, y=40, width=900, height=180,
                              opacity=0.9, max_lines=4),
    )
    p = tmp_path / "cfg.json"
    save_config(cfg, str(p))
    loaded = load_config(str(p))
    assert loaded.display == "overlay"
    assert loaded.transcriber == "moonshine"
    assert loaded.language == "ar"
    assert loaded.source == "loopback"
    assert loaded.source_device == "Headphone (Realtek(R) Audio) [Loopback]"
    assert loaded.overlay.x == 120
    assert loaded.overlay.y == 40
    assert loaded.overlay.width == 900
    assert loaded.overlay.opacity == 0.9
    assert loaded.overlay.max_lines == 4
    assert loaded.input_file == "demo.wav"


def test_auto_place_omits_none_fields(tmp_path):
    cfg = Config(overlay=OverlayConfig(opacity=0.7))
    p = tmp_path / "cfg.json"
    save_config(cfg, str(p))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "x" not in data["overlay"]  # None fields are omitted, not null


def test_missing_file_returns_defaults(tmp_path):
    cfg = load_config(str(tmp_path / "nope.json"))
    assert cfg == Config()


def test_unknown_keys_tolerated(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"display": "overlay", "future_key": 42}),
                 encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.display == "overlay"
    assert cfg.transcriber == "moonshine"  # untouched defaults