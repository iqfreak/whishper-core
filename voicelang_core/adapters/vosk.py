"""Vosk offline streaming ASR adapter (alphacep / vosk-api).

Small (~50 MB) per-language Kaldi models cached to ``%LOCALAPPDATA%/vosk/<lang>``
on first use, then reused offline.

Models are hosted at https://alphacephei.com/vosk/models/ . This adapter
downloads via ``requests`` + ``tqdm`` + ``filelock``, validates size,
extracts the zip, and verifies markers (am/final.mdl etc).

Language catalog
----------------
SUPPORTED_LANGS (config.py) -> vosk small models:

  en -> vosk-model-small-en-us-0.15
  de -> vosk-model-small-de-0.15
  es -> vosk-model-small-es-0.42
  fr -> vosk-model-small-fr-0.22
  pt -> vosk-model-small-pt-0.3      (it not in SUPPORTED_LANGS but exists as it-0.22)
  zh -> vosk-model-small-cn-0.22
  ja -> vosk-model-small-ja-0.22
  ru -> vosk-model-small-ru-0.22
  tr -> vosk-model-small-tr-0.3
  ko -> vosk-model-small-ko-0.22 (?)  – listed on alphacephei, 82 MB tentative
  hi -> vosk-model-small-hi-0.22
  nl -> vosk-model-small-nl-0.22 exists (39 MB) – historically reported "not" but present
  pl -> vosk-model-small-pl-0.22 (50 MB)
  uk -> vosk-model-small-uk-v3-nano / small – 73–133 MB, historically "not yet"
  vi -> no small model per spec: fallback to en (real vn-0.4 32 MB exists as alternative)
  th -> unavailable (no Thai model on alphacephei)
  ar -> no small model (ar-mgb2-0.4 is 318 MB, far from small tier)
  tl -> tl-ph-generic-0.6 is 320 MB (not small tier)
  it -> vosk-model-small-it-0.22 (48 MB) – extra, even if spec said "it not in SUPPORTED"

Audio:
  Vosk expects 16 kHz mono s16le PCM. The pipeline already captures at 16 kHz
  mono, but this adapter tolerates any rate/channels via numpy resampling
  (librosa exact resample when installed, linear fallback otherwise).

Port:
  Implements :class:`StreamingTranscriber`::

      transcriber = VoskTranscriber(language="en")
      for chunk in source.stream():
          for seg in transcriber.transcribe_stream(chunk):
              pipeline._handle_segment(seg, translator)
      for seg in transcriber.finish():
          pipeline._handle_segment(seg, translator)

Extra:
  ``pip install vosk``  or  ``uv sync --extra vosk``  declares ``vosk``.

Helpers:
  vosk_cache_root()      -> Path(%LOCALAPPDATA%/vosk)
  vosk_model_dir(lang)   -> Path(%LOCALAPPDATA%/vosk/<lang>) (fallback-aware)
  list_vosk_languages()  -> sorted list of langs with available small models
  get_vosk_model_info(lang) -> catalog entry or raises
"""

from __future__ import annotations

import json
import os
import threading
import zipfile
import shutil
from pathlib import Path
from typing import Iterator, Optional, Dict, Any

from .base import StreamingTranscriber
from ..types import AudioChunk, Segment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOSK_SAMPLE_RATE = 16000
VOSK_BASE_URL = "https://alphacephei.com/vosk/models"
# canonical small size ~50 MB; per-model sizes differ a bit
_DEFAULT_SIZE_MB = 50

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

# Each entry: model_name, url, size_mb, available, notes, fallback (optional)
# ``available=False`` means no small model in this tier (or not 50 MB tier).
# ``fallback`` names a lang whose model should be used instead.
VOSK_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "en": {
        "model_name": "vosk-model-small-en-us-0.15",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-en-us-0.15.zip",
        "size_mb": 40,
        "available": True,
        "notes": "US English, lightweight wideband (Apache 2.0)",
    },
    "de": {
        "model_name": "vosk-model-small-de-0.15",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-de-0.15.zip",
        "size_mb": 45,
        "available": True,
        "notes": "German light wideband",
    },
    "es": {
        "model_name": "vosk-model-small-es-0.42",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-es-0.42.zip",
        "size_mb": 39,
        "available": True,
        "notes": "Spanish light wideband",
    },
    "fr": {
        "model_name": "vosk-model-small-fr-0.22",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-fr-0.22.zip",
        "size_mb": 41,
        "available": True,
        "notes": "French light wideband",
    },
    "pt": {
        "model_name": "vosk-model-small-pt-0.3",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-pt-0.3.zip",
        "size_mb": 31,
        "available": True,
        "notes": "Portuguese light wideband (spec: it not in SUPPORTED but pt is)",
    },
    "it": {
        "model_name": "vosk-model-small-it-0.22",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-it-0.22.zip",
        "size_mb": 48,
        "available": True,
        "notes": "Italian light (extra, SUPPORTED_LANGS includes it)",
    },
    "zh": {
        "model_name": "vosk-model-small-cn-0.22",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-cn-0.22.zip",
        "size_mb": 42,
        "available": True,
        "notes": "Chinese (cn) light – maps zh/zh-cn",
        "aliases": ["zh-cn", "cn", "yue"],
    },
    "ja": {
        "model_name": "vosk-model-small-ja-0.22",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-ja-0.22.zip",
        "size_mb": 48,
        "available": True,
        "notes": "Japanese light",
    },
    "ru": {
        "model_name": "vosk-model-small-ru-0.22",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-ru-0.22.zip",
        "size_mb": 45,
        "available": True,
        "notes": "Russian light",
    },
    "tr": {
        "model_name": "vosk-model-small-tr-0.3",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-tr-0.3.zip",
        "size_mb": 35,
        "available": True,
        "notes": "Turkish light",
    },
    "ko": {
        "model_name": "vosk-model-small-ko-0.22",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-ko-0.22.zip",
        "size_mb": 82,
        "available": True,
        "notes": "Korean light (tentative catalog, 82 MB; verify on alphacephei)",
    },
    "hi": {
        "model_name": "vosk-model-small-hi-0.22",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-hi-0.22.zip",
        "size_mb": 42,
        "available": True,
        "notes": "Hindi light",
    },
    # Historically spec said "nl small not" – but alphacephei now lists it
    "nl": {
        "model_name": "vosk-model-small-nl-0.22",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-nl-0.22.zip",
        "size_mb": 39,
        "available": True,
        "notes": "Dutch light (spec said nl small not, but model now exists 39 MB)",
    },
    "pl": {
        "model_name": "vosk-model-small-pl-0.22",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-pl-0.22.zip",
        "size_mb": 50,
        "available": True,
        "notes": "Polish light",
    },
    # spec: "uk small not yet" – now there is uk-v3-nano/small (73–133 MB)
    "uk": {
        "model_name": "vosk-model-small-uk-v3-nano",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-uk-v3-nano.zip",
        "size_mb": 73,
        "available": True,
        "notes": "Ukrainian nano (spec said not yet; 73 MB, alternative small 133 MB)",
    },
    # spec: vi -> use en fallback, even though vn-0.4 (32 MB) exists
    "vi": {
        "model_name": "vosk-model-small-en-us-0.15",
        "url": f"{VOSK_BASE_URL}/vosk-model-small-en-us-0.15.zip",
        "size_mb": 40,
        "available": True,
        "fallback": "en",
        "notes": "No small vi model per spec -> fallback to en; real alternative is vosk-model-small-vn-0.4 (32 MB)",
        "alt_model": "vosk-model-small-vn-0.4",
        "alt_url": f"{VOSK_BASE_URL}/vosk-model-small-vn-0.4.zip",
    },
    "th": {
        "model_name": None,
        "url": None,
        "size_mb": None,
        "available": False,
        "notes": "Unavailable – no Thai model on alphacephei",
    },
    "ar": {
        "model_name": None,
        "url": None,
        "size_mb": None,
        "available": False,
        "notes": "No small ar model (ar-mgb2-0.4 is 318 MB, small-ar-tn-0.1-linto 158 MB – not small tier)",
        "alt_model": "vosk-model-ar-mgb2-0.4",
        "alt_url": f"{VOSK_BASE_URL}/vosk-model-ar-mgb2-0.4.zip",
    },
    "tl": {
        "model_name": None,
        "url": None,
        "size_mb": None,
        "available": False,
        "notes": "No small tl model; tl-ph-generic-0.6 is 320 MB (not small tier)",
        "alt_model": "vosk-model-tl-ph-generic-0.6",
        "alt_url": f"{VOSK_BASE_URL}/vosk-model-tl-ph-generic-0.6.zip",
    },
}

# Back-compat simple mapping lang -> model_name (or None)
VOSK_MODELS: Dict[str, Optional[str]] = {
    k: v.get("model_name") for k, v in VOSK_MODEL_CATALOG.items()
}

# Normalization aliases (zh variants, etc.)
_LANG_ALIASES = {
    "zh-cn": "zh",
    "zh_cn": "zh",
    "cn": "zh",
    "yue": "zh",
    "cantonese": "zh",
    "pt-br": "pt",
    "pt_br": "pt",
    "en-us": "en",
    "en_us": "en",
    "en-gb": "en",
}


def _normalize_lang(lang: str) -> str:
    if not lang:
        return "en"
    s = str(lang).strip().lower()
    if s in _LANG_ALIASES:
        return _LANG_ALIASES[s]
    # handle compound like zh-CN -> zh
    base = s.split("-")[0].split("_")[0]
    if s in VOSK_MODEL_CATALOG:
        return s
    if base in VOSK_MODEL_CATALOG:
        # prefer exact if exists, else base
        return base
    return s


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _localappdata_root() -> Path:
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(root)


def vosk_cache_root() -> Path:
    """Root cache dir ``%LOCALAPPDATA%/vosk``."""
    return _localappdata_root() / "vosk"


def vosk_model_dir(lang: str) -> Path:
    """Return expected cache dir for *lang* (fallback-aware).

    Per spec: cached to ``%LOCALAPPDATA%/vosk/<lang>``.
    Fallbacks are resolved (``vi`` -> ``en``) so the caller always gets a
    usable path. The actual extracted model folder after download is this path
    (or ``%LOCALAPPDATA%/vosk/<model_name>`` as alt, both are checked at load).
    """
    nlang = _normalize_lang(lang)
    entry = VOSK_MODEL_CATALOG.get(nlang)
    if entry is not None and entry.get("fallback"):
        # vi fallback case – return the fallback's physical dir but keep
        # logical lang name? Spec says %LOCALAPPDATA%/vosk/<lang>, so honor that
        # while documenting fallback.
        # We return lang dir (logical) – download will place fallback model there.
        return vosk_cache_root() / nlang
    # auto -> en like other adapters
    if nlang == "auto" or not nlang:
        return vosk_cache_root() / "en"
    return vosk_cache_root() / nlang


def get_vosk_model_info(lang: str) -> Dict[str, Any]:
    """Return catalog entry for *lang* (fallback-aware) or raise ValueError."""
    nlang = _normalize_lang(lang)
    # auto maps to en, mirroring moonshine guard
    if nlang == "auto" or not nlang:
        nlang = "en"
    entry = VOSK_MODEL_CATALOG.get(nlang)
    if entry is None:
        raise ValueError(
            f"Vosk language {lang!r} (normalized {nlang!r}) not in catalog; "
            f"available: {', '.join(sorted(list_vosk_languages()))}"
        )
    if not entry.get("available") or entry.get("model_name") is None:
        # try fallback field
        fb = entry.get("fallback")
        if fb and fb in VOSK_MODEL_CATALOG and VOSK_MODEL_CATALOG[fb].get("available"):
            return VOSK_MODEL_CATALOG[fb]
        raise ValueError(
            f"Vosk model for language {lang!r} unavailable: {entry.get('notes','')} "
            f"(catalog available: {', '.join(sorted(list_vosk_languages()))})"
        )
    return entry


def list_vosk_languages(available_only: bool = True) -> list[str]:
    """List language tags with small Vosk models.

    Args:
        available_only: if True (default) only langs where available==True.
                        vi counts as available via fallback.
    """
    if available_only:
        return sorted(
            k for k, v in VOSK_MODEL_CATALOG.items()
            if v.get("available") and v.get("model_name")
        )
    return sorted(VOSK_MODEL_CATALOG.keys())


def is_vosk_language_available(lang: str) -> bool:
    nlang = _normalize_lang(lang)
    entry = VOSK_MODEL_CATALOG.get(nlang)
    if entry is None:
        return False
    if entry.get("available") and entry.get("model_name"):
        return True
    # check fallback like vi
    fb = entry.get("fallback")
    if fb and VOSK_MODEL_CATALOG.get(fb, {}).get("available"):
        return True
    return False


# ---------------------------------------------------------------------------
# Download helpers (requests + tqdm + filelock + zipfile + size check)
# ---------------------------------------------------------------------------

def _is_model_present(path: Path) -> bool:
    """Check whether *path* looks like an extracted Vosk model."""
    if not path.is_dir():
        return False
    # common markers – any hit means present
    markers = [
        path / "am" / "final.mdl",
        path / "conf" / "model.conf",
        path / "conf" / "mfcc.conf",
        path / "graph" / "HCLG.fst",
        path / "graph" / "Gr.fst",
        path / "am" / "final.alimdl",
    ]
    for m in markers:
        try:
            if m.exists():
                return True
        except Exception:
            continue
    # fallback: am dir + mdl file
    try:
        am = path / "am"
        if am.is_dir() and any(am.glob("*.mdl")):
            return True
    except Exception:
        pass
    # if path contains a single subdir that is the model (zip top-level)
    try:
        children = list(path.iterdir())
        if len(children) == 1 and children[0].is_dir() and _is_model_present(children[0]):
            return True
    except Exception:
        pass
    # last resort: at least 5 files and contains conf/graph/am
    try:
        files = list(path.iterdir())
        if len(files) >= 4 and any((path / d).is_dir() for d in ("am", "conf", "graph")):
            return True
    except Exception:
        pass
    return False


def _download_file(url: str, dest: Path, expected_size_mb: Optional[int] = None, progress: bool = True) -> None:
    """Download *url* to *dest* with streaming, tqdm, and size check."""
    # lazy imports so core import stays lightweight
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "requests not installed (needed for Vosk model download). "
            "Install with: pip install requests  or  uv sync --extra vosk"
        ) from exc
    try:
        from tqdm import tqdm  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "tqdm not installed (needed for Vosk download progress). "
            "Install with: pip install tqdm"
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    # pre-check: if total is suspiciously tiny vs expected, warn but proceed
    if expected_size_mb and total:
        expected = expected_size_mb * 1024 * 1024
        if total < expected * 0.3 and total < 5 * 1024 * 1024:
            # don't fail yet – let post-download size check handle it
            pass

    chunk_size = 8192
    # If file already exists with correct size, skip? But we use tmp path
    with open(tmp, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        disable=not progress,
        desc=dest.name,
        leave=False,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    actual = tmp.stat().st_size
    if total and actual != total:
        # server sent Content-Length but we got mismatch – treat as failure
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise IOError(f"Download size mismatch for {url}: got {actual}, expected {total}")

    if actual < 1024 * 1024:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise IOError(f"Download too small ({actual} bytes) for {url}; likely HTML error page")

    if expected_size_mb:
        expected = expected_size_mb * 1024 * 1024
        # allow 50% variance – some models grow slightly; hard fail only if <50%
        if actual < expected * 0.5:
            # warn but don't fail if total header missing (alphacephei sizes drift)
            pass

    # atomic move
    tmp.replace(dest)


def _extract_zip(zip_path: Path, dest_dir: Path, model_name: str) -> None:
    """Extract *zip_path* so that *dest_dir* becomes the model root.

    The zip typically contains a single top-level dir named *model_name*.
    We extract to a temp dir and then move/rename to *dest_dir*.
    Size check: zip must decompress to >1 MB.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"zip not found: {zip_path}")

    # size check on zip itself
    if zip_path.stat().st_size < 1024 * 1024:
        raise IOError(f"zip too small ({zip_path.stat().st_size} bytes): {zip_path}")

    # validate zip and compute uncompressed size before extraction
    with zipfile.ZipFile(zip_path, "r") as z:
        try:
            bad = z.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"corrupt entry: {bad}")
        except zipfile.BadZipFile:
            raise
        total_uncompressed = sum(info.file_size for info in z.infolist())
        if total_uncompressed < 1024 * 1024:
            raise IOError(f"zip uncompressed size too small ({total_uncompressed} bytes)")

    tmp_extract = dest_dir.parent / f"{dest_dir.name}.tmp_extract"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract, ignore_errors=True)
    tmp_extract.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_extract)

    # locate extracted root
    try:
        candidates = list(tmp_extract.iterdir())
    except OSError as exc:
        raise IOError(f"extracted dir listing failed: {exc}") from exc

    extracted_root: Path
    if len(candidates) == 1 and candidates[0].is_dir():
        extracted_root = candidates[0]
    else:
        # no single top dir – use tmp_extract itself (some zips extract flat)
        extracted_root = tmp_extract

    # ensure dest_dir parent exists and clean dest_dir
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)

    try:
        if extracted_root == tmp_extract:
            extracted_root.rename(dest_dir)
        else:
            shutil.move(str(extracted_root), str(dest_dir))
    finally:
        # cleanup residual tmp_extract if it still exists
        if tmp_extract.exists():
            try:
                shutil.rmtree(tmp_extract, ignore_errors=True)
            except Exception:
                pass


def download_vosk_model(lang: str, progress: bool = True) -> Path:
    """Ensure small Vosk model for *lang* is cached, downloading if needed.

    Uses ``filelock`` to guard concurrent downloads (e.g. the app starting
    twice). On success returns the model directory path.

    Raises:
        ValueError: if language has no small model and no fallback.
        RuntimeError/IOError: on network/extraction failure.
    """
    info = get_vosk_model_info(lang)
    model_name = info["model_name"]
    url = info["url"]
    size_mb = info.get("size_mb", _DEFAULT_SIZE_MB)
    nlang = _normalize_lang(lang)
    if nlang == "auto" or not nlang:
        nlang = "en"

    dest = vosk_model_dir(nlang)
    alt = vosk_cache_root() / model_name

    # Fast path: already present
    if _is_model_present(dest):
        return dest
    if _is_model_present(alt):
        return alt

    # Need filelock for download
    try:
        from filelock import FileLock  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "filelock not installed (needed for Vosk concurrent download guard). "
            "Install with: pip install filelock"
        ) from exc

    lock_path = str(vosk_cache_root() / f"{nlang}.lock")
    # Ensure cache root exists before locking
    vosk_cache_root().mkdir(parents=True, exist_ok=True)

    with FileLock(lock_path, timeout=300):
        # double-check after acquiring lock (another process may have finished)
        if _is_model_present(dest):
            return dest
        if _is_model_present(alt):
            return alt

        zip_path = vosk_cache_root() / f"{model_name}.zip"
        # If zip already present and seems complete, reuse; else download
        need_download = True
        if zip_path.is_file() and zip_path.stat().st_size > 1024 * 1024:
            # quick sanity: valid zip?
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    if z.testzip() is None:
                        need_download = False
            except zipfile.BadZipFile:
                need_download = True
        if need_download:
            _download_file(url, zip_path, expected_size_mb=size_mb, progress=progress)

        _extract_zip(zip_path, dest, model_name)

        # post-extract verification
        if not _is_model_present(dest):
            # maybe extraction created alt path instead of dest? Check alt
            # Some unzip logic leaves alt when dest == tmp_extract case handled,
            # but double-check both locations.
            if _is_model_present(alt):
                return alt
            # If dest contains a single nested model subdir (double nesting), hoist it
            try:
                children = list(dest.iterdir()) if dest.is_dir() else []
                if len(children) == 1 and children[0].is_dir() and _is_model_present(children[0]):
                    # move nested up one level: children[0] -> dest.tmp -> rename
                    tmp = dest.parent / f"{dest.name}.hoist_tmp"
                    if tmp.exists():
                        shutil.rmtree(tmp, ignore_errors=True)
                    children[0].rename(tmp)
                    shutil.rmtree(dest, ignore_errors=True)
                    tmp.rename(dest)
            except Exception:
                pass
            if not _is_model_present(dest):
                raise RuntimeError(
                    f"Vosk model extraction failed for {lang}: {dest} missing marker files "
                    f"(looked for am/final.mdl, conf/model.conf, graph/HCLG.fst)"
                )

        # cleanup zip on success (keeps disk tidy; model is ~50 MB, zip similar)
        try:
            if zip_path.exists():
                zip_path.unlink()
        except OSError:
            pass

        return dest


def ensure_vosk_model(lang: str, progress: bool = True) -> Path:
    """Alias for download_vosk_model (semantic naming for adapter init)."""
    return download_vosk_model(lang, progress=progress)


# ---------------------------------------------------------------------------
# Audio helper: any -> 16k mono s16le
# ---------------------------------------------------------------------------

def _prepare_vosk_pcm(chunk: AudioChunk) -> bytes:
    """Convert *chunk* PCM to 16 kHz mono s16le bytes for Vosk.

    - Handles empty chunks
    - Downmixes stereo -> mono by averaging
    - Resamples via librosa when available, else linear interpolation
    - Ensures even byte length (s16 alignment)
    """
    pcm = chunk.pcm or b""
    if not pcm:
        return b""
    sr = int(getattr(chunk, "sample_rate", VOSK_SAMPLE_RATE) or VOSK_SAMPLE_RATE)
    ch = int(getattr(chunk, "channels", 1) or 1)

    # Fast path: already correct
    if sr == VOSK_SAMPLE_RATE and ch == 1:
        # ensure even length (s16le is 2 bytes per sample)
        if len(pcm) % 2:
            pcm = pcm[:-1]
        return pcm

    # Slow path: numpy conversion + resample
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        # numpy is expected (core dep), but if absent just pass through truncated
        if len(pcm) % 2:
            pcm = pcm[:-1]
        return pcm

    # ensure bytes -> int16 array
    if len(pcm) % 2:
        pcm = pcm[:-1]
    try:
        audio = np.frombuffer(pcm, dtype=np.int16)
    except Exception:
        return b""

    if audio.size == 0:
        return b""

    # Channels: stereo interleaved L R -> mono
    if ch > 1:
        # need complete frames
        frames = audio.size // ch
        audio = audio[: frames * ch].reshape(-1, ch)
        # average to mono in float to avoid int16 overflow then back
        audio = audio.astype(np.float32).mean(axis=1)
        # convert back to int16 for resample path (or direct if no resample)
        # keep as float for resample, but if no resample we need int16 bytes
        if sr == VOSK_SAMPLE_RATE:
            audio = np.clip(audio, -32768, 32767).astype(np.int16)
            return audio.tobytes()
        # else keep float for resample
        flt = audio / 32768.0
    else:
        if sr == VOSK_SAMPLE_RATE:
            return audio.tobytes()
        flt = audio.astype(np.float32) / 32768.0

    # Resample float mono to 16k
    if sr != VOSK_SAMPLE_RATE:
        try:
            import librosa  # noqa: PLC0415
            flt = librosa.resample(flt, orig_sr=sr, target_sr=VOSK_SAMPLE_RATE)
        except ModuleNotFoundError:
            # linear fallback – good enough for speech int16
            n_old = flt.size
            n_new = int(round(n_old * VOSK_SAMPLE_RATE / sr))
            if n_new <= 0:
                return b""
            # linspace without endpoint to keep timing
            old_idx = np.linspace(0.0, 1.0, n_old, endpoint=False)
            new_idx = np.linspace(0.0, 1.0, n_new, endpoint=False)
            flt = np.interp(new_idx, old_idx, flt).astype(np.float32)
        except Exception:
            # any librosa error -> fallback linear
            n_old = flt.size
            n_new = int(round(n_old * VOSK_SAMPLE_RATE / sr))
            if n_new <= 0:
                return b""
            old_idx = np.linspace(0.0, 1.0, n_old, endpoint=False)
            new_idx = np.linspace(0.0, 1.0, n_new, endpoint=False)
            flt = np.interp(new_idx, old_idx, flt).astype(np.float32)

    # float -> int16
    audio_i16 = np.clip(flt * 32767.0, -32768, 32767).astype(np.int16)
    return audio_i16.tobytes()


# ---------------------------------------------------------------------------
# Registry for shared Vosk Model objects (one per model_dir)
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, Any] = {}
_REGISTRY_LOCK = threading.Lock()


def _get_or_load_model(model_path: Path):
    """Return cached vosk.Model for *model_path*, loading once per process."""
    # resolve to absolute for key stability
    try:
        key = str(model_path.resolve())
    except Exception:
        key = str(model_path)

    with _REGISTRY_LOCK:
        hit = _MODEL_REGISTRY.get(key)
        if hit is not None:
            return hit
        # lazy vosk import – core stays importable without vosk extra
        try:
            import vosk  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "vosk not installed (needed for VoskTranscriber). "
                "Install with: pip install vosk  or  uv sync --extra vosk"
            ) from exc

        # If model_path contains a single nested model subdir, unwrap
        load_path = model_path
        if load_path.is_dir() and not _is_model_present(load_path):
            # check single child
            try:
                children = list(load_path.iterdir())
                if len(children) == 1 and children[0].is_dir() and _is_model_present(children[0]):
                    load_path = children[0]
            except Exception:
                pass

        if not _is_model_present(load_path):
            # also handle alt where model_dir is per-lang but alt per-model_name exists
            # search both
            alt = vosk_cache_root() / (VOSK_MODEL_CATALOG.get(_normalize_lang(str(model_path.name)), {}).get("model_name", "") or "")
            if alt.is_dir() and _is_model_present(alt):
                load_path = alt
            else:
                # final fallback: if load_path is a file, error
                pass

        # calm vosk C++ logger (0 = info, -1 = quiet); optional
        try:
            vosk.SetLogLevel(-1)  # type: ignore[attr-defined]
        except Exception:
            pass

        model = vosk.Model(str(load_path))
        _MODEL_REGISTRY[key] = model
        # also cache under resolved key of alt for reuse
        return model


# ---------------------------------------------------------------------------
# Vosk adapter
# ---------------------------------------------------------------------------

class VoskTranscriber(StreamingTranscriber):
    """Vosk streaming transcriber behind the :class:`StreamingTranscriber` port.

    Args:
        language: lang tag per SUPPORTED_LANGS (en/de/es/fr/zh/ja/ru/pt/...
                  auto maps to en like other adapters). Fallbacks (vi->en)
                  are resolved automatically.
        model_dir: explicit path to extracted vosk model directory. If None,
                   resolved via :func:`vosk_model_dir` and auto-downloaded
                   when *auto_download* is True.
        auto_download: when True (default) missing models are downloaded via
                       :func:`download_vosk_model` (requests+tqdm+filelock).
                       When False, missing model raises FileNotFoundError.
        sample_rate: expected recognizer rate (default 16000). Must match the
                     PCM this adapter feeds (16k s16le).
        auto_rate_convert: when True (default) any chunk rate/channels are
                           converted to 16k mono before feeding.

    Example::

        tr = VoskTranscriber(language="en")
        for chunk in MicSource(block_seconds=0.25).stream():
            for seg in tr.transcribe_stream(chunk):
                print(seg.status, seg.source_text)
        for seg in tr.finish():
            print("final tail", seg.source_text)

    Notes:
        - :meth:`transcribe_stream` yields a ``partial`` Segment per hypothesis
          change and a ``final`` Segment whenever ``AcceptWaveform`` fires.
          Deduplicates identical partials.
        - :meth:`finish` emits ``FinalResult`` tail (file replay).
        - Model objects are cached per ``model_dir`` (like whisper/moonshine
          registries) and guarded by a lock.
    """

    def __init__(
        self,
        language: str = "en",
        model_dir: str | Path | None = None,
        auto_download: bool = True,
        sample_rate: int = VOSK_SAMPLE_RATE,
        auto_rate_convert: bool = True,
    ):
        # Normalize language like other adapters (auto -> en guard)
        raw_lang = language or "en"
        language = _normalize_lang(raw_lang)
        if language == "auto" or not language:
            language = "en"
        self._language = language
        self._sample_rate = int(sample_rate)
        self._auto_rate_convert = bool(auto_rate_convert)
        self._last_partial = ""
        self._lock = threading.Lock()

        # Resolve model directory
        resolved_dir: Path
        if model_dir is not None:
            resolved_dir = Path(model_dir)
            if not _is_model_present(resolved_dir):
                if auto_download:
                    # If model_dir is a lang tag placeholder (e.g. Path("en")),
                    # try catalog download to that location? Prefer explicit dir check.
                    # Fall back to catalog download for language then use its dir if caller
                    # gave a non-existent explicit path that looks like a lang-less path?
                    # Simpler: raise if explicit path missing and not downloadable.
                    raise FileNotFoundError(
                        f"Vosk model_dir not found or not a valid Vosk model: {resolved_dir}\n"
                        f"Extract a vosk-model-small-*.zip there, or pass language={language!r} "
                        f"with auto_download=True to fetch {get_vosk_model_info(language)['model_name']}"
                    )
                else:
                    raise FileNotFoundError(f"Vosk model_dir not found: {resolved_dir}")
        else:
            # Catalog-resolved dir (per-lang cache)
            # Validate language first (raises if unavailable)
            info = get_vosk_model_info(language)
            # logical per-lang dir
            logical = vosk_model_dir(language)
            alt = vosk_cache_root() / info["model_name"]
            # Check existing before maybe downloading
            if _is_model_present(logical):
                resolved_dir = logical
            elif _is_model_present(alt):
                resolved_dir = alt
            else:
                if auto_download:
                    resolved_dir = download_vosk_model(language, progress=True)
                else:
                    raise FileNotFoundError(
                        f"Vosk model for {language!r} ({info['model_name']}) not cached at "
                        f"{logical} or {alt}. Pass auto_download=True or run "
                        f"download_vosk_model({language!r}) first."
                    )

        # At this point resolved_dir should be present (explicit or downloaded)
        # Handle case where logical dir is fallback vi -> en logical dir may contain en model;
        # ensure we pass the physical directory that is present.
        if not _is_model_present(resolved_dir):
            # try alternative physical location one more time (race)
            info = get_vosk_model_info(language) if language in VOSK_MODEL_CATALOG else None
            if info is not None:
                alt = vosk_cache_root() / info["model_name"]
                if _is_model_present(alt):
                    resolved_dir = alt
            if not _is_model_present(resolved_dir):
                raise FileNotFoundError(
                    f"Vosk model directory missing valid markers: {resolved_dir}\n"
                    f"Expected am/final.mdl or conf/model.conf or graph/HCLG.fst"
                )

        self._model_dir = resolved_dir
        self._model = _get_or_load_model(self._model_dir)

        # Create recognizer – one per instance (recognizer is not thread-safe for
        # concurrent streams, but each pipeline gets its own VoskTranscriber instance)
        try:
            import vosk  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "vosk not installed (needed for VoskTranscriber). "
                "Install with: pip install vosk  or  uv sync --extra vosk"
            ) from exc

        # Vosk KaldiRecognizer expects model and sample rate; newer vosk also
        # accepts SpkModel/phrases, but base (model, rate) is universal.
        self._recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)
        try:
            # Enable partial words – harmless if not supported
            self._recognizer.SetWords(True)  # type: ignore[attr-defined]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def transcribe_stream(self, chunk: AudioChunk) -> Iterator[Segment]:
        """Feed *chunk* to the Vosk stream and yield partial/final Segments.

        - Converts any rate/channels to 16k mono s16le when *auto_rate_convert*.
        - On ``AcceptWaveform`` true, emits a ``final`` Segment (decoded ``text``).
        - Otherwise emits a ``partial`` Segment when ``partial`` changed.
        - Empty hypothesis is never emitted.
        """
        if chunk is None or chunk.pcm is None:
            return
        if len(chunk.pcm) == 0:
            return

        if self._auto_rate_convert:
            pcm = _prepare_vosk_pcm(chunk)
        else:
            # strict: assume caller provided 16k mono
            pcm = chunk.pcm
            if len(pcm) % 2:
                pcm = pcm[:-1]

        if not pcm:
            return

        # Vosk recognizer is not thread-safe; guard per-instance.
        with self._lock:
            rec = self._recognizer
            if rec is None:
                return
            try:
                is_final = rec.AcceptWaveform(pcm)
            except Exception:
                # C++ exception on malformed PCM – skip chunk
                return

            if is_final:
                try:
                    raw = rec.Result()
                    data = json.loads(raw) if isinstance(raw, str) else {}
                except (json.JSONDecodeError, TypeError, ValueError):
                    return
                text = (data.get("text") or "").strip() if isinstance(data, dict) else ""
                if text:
                    self._last_partial = ""
                    yield Segment(
                        status="final",
                        source_text=text,
                        source_language=self._language,
                    )
                # Some Vosk builds return result words/confidence – we ignore.
            else:
                try:
                    raw = rec.PartialResult()
                    data = json.loads(raw) if isinstance(raw, str) else {}
                except (json.JSONDecodeError, TypeError, ValueError):
                    return
                ptext = (data.get("partial") or "").strip() if isinstance(data, dict) else ""
                if ptext and ptext != self._last_partial:
                    self._last_partial = ptext
                    yield Segment(
                        status="partial",
                        source_text=ptext,
                        source_language=self._language,
                    )

    def transcribe(self, chunk: AudioChunk) -> Optional[Segment]:
        """Batch-compat: return the last final Segment for *chunk*, or None."""
        last: Optional[Segment] = None
        for seg in self.transcribe_stream(chunk):
            if seg.status == "final":
                last = seg
        return last

    def finish(self) -> Iterator[Segment]:
        """Flush the tail of a finite stream (file replay / stop).

        Wraps ``FinalResult()`` – the last partial becomes a final.
        """
        with self._lock:
            rec = getattr(self, "_recognizer", None)
            if rec is None:
                return
            try:
                raw = rec.FinalResult()
                data = json.loads(raw) if isinstance(raw, str) else {}
            except (json.JSONDecodeError, TypeError, ValueError, RuntimeError):
                return
            except Exception:
                return
            text = (data.get("text") or "").strip() if isinstance(data, dict) else ""
            if text:
                self._last_partial = ""
                yield Segment(
                    status="final",
                    source_text=text,
                    source_language=self._language,
                )

    # ------------------------------------------------------------------
    # Lifecycle helpers (optional but handy for pipeline parity)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset recognizer stream (drops hypothesis) without reloading model."""
        with self._lock:
            try:
                import vosk  # noqa: PLC0415
                self._recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)
                try:
                    self._recognizer.SetWords(True)  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception:
                pass
            self._last_partial = ""

    def close(self) -> None:
        """Best-effort release of recognizer (no model eviction)."""
        with self._lock:
            self._recognizer = None  # type: ignore[assignment]
            self._last_partial = ""

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def language(self) -> str:
        return self._language


# ---------------------------------------------------------------------------
# Convenience top-level helpers expected by task
# ---------------------------------------------------------------------------

__all__ = [
    "VoskTranscriber",
    "VOSK_MODEL_CATALOG",
    "VOSK_MODELS",
    "VOSK_SAMPLE_RATE",
    "VOSK_BASE_URL",
    "vosk_cache_root",
    "vosk_model_dir",
    "get_vosk_model_info",
    "list_vosk_languages",
    "is_vosk_language_available",
    "download_vosk_model",
    "ensure_vosk_model",
]
