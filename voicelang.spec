# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_all

datas = [('assets/logo.png', 'assets')]
binaries = []
hiddenimports = ['pycaw', 'comtypes', 'psutil', 'pyaudiowpatch', 'sounddevice', 'soundfile', '_soundfile_data', 'faster_whisper', 'onnxruntime', 'ctranslate2', 'vosk']
datas += collect_data_files('onnxruntime')
datas += collect_data_files('huggingface_hub')
binaries += collect_dynamic_libs('ctranslate2')
binaries += collect_dynamic_libs('onnxruntime')
# nvidia DLLs live under site-packages/nvidia/*/bin -- collect as datas+binaries via namespace package 'nvidia'
try:
    tmp_ret = collect_all('nvidia')
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
except Exception:
    pass
try:
    tmp_ret = collect_all('pyaudiowpatch')
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
except Exception:
    pass
# sounddevice is a single-file module; hook handles _soundfile_data but we ensure hiddenimport.
# tiktoken not installed in this venv; guard collection.
try:
    datas += collect_data_files('tiktoken')
except Exception:
    pass
tmp_ret = collect_all('sherpa_onnx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('moonshine_voice')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('faster_whisper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('requests')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
try:
    tmp_ret = collect_all('vosk')
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
except Exception:
    pass


a = Analysis(
    ['voicelang_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'transformers', 'accelerate', 'librosa', 'soundfile', 'scipy'],  # P5: slim 2.4GB -> exclude funasr-nano torch (lazy extras), strip unused
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='voicelang',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/logo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='voicelang',
)
