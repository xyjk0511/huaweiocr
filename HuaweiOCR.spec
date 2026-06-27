# -*- mode: python ; coding: utf-8 -*-
import os
import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

LOCAL_DETECTOR_TARGET = 'local_models\\detectors'
REQUIRED_LOCAL_DETECTORS = (
    os.path.join('local_models', 'detectors', 'label_detector.onnx'),
    os.path.join('local_models', 'detectors', 'field_detector.onnx'),
)

missing_local_detectors = [
    path for path in REQUIRED_LOCAL_DETECTORS if not os.path.isfile(path)
]
if missing_local_detectors:
    raise FileNotFoundError(
        "Missing required local ONNX detector model(s) for release build: "
        + ", ".join(missing_local_detectors)
        + ". Place label_detector.onnx and field_detector.onnx under "
        + LOCAL_DETECTOR_TARGET
        + " before running PyInstaller."
    )

datas = [
    ('bundle\\models', 'models'),
    ('bundle\\BarcodeReaderCLI\\bin\\BarcodeReaderCLI.exe', 'BarcodeReaderCLI\\bin'),
    ('bundle\\BarcodeReaderCLI\\bin\\curl.exe', 'BarcodeReaderCLI\\bin'),
    ('bundle\\BarcodeReaderCLI\\bin\\curl-ca-bundle.crt', 'BarcodeReaderCLI\\bin'),
    ('bundle\\BarcodeReaderCLI\\bin\\inlite-barcode-reader-license-agreement.pdf', 'BarcodeReaderCLI\\bin'),
]
if os.path.isfile('.env'):
    datas.append(('.env', '.'))
datas += [
    (path, LOCAL_DETECTOR_TARGET)
    for path in REQUIRED_LOCAL_DETECTORS
]
datas += collect_data_files('paddlex', includes=['.version', 'configs/**'])
datas += collect_data_files('Cython', includes=['Utility/**'])
datas += collect_data_files('tkinterdnd2', includes=['tkdnd/**'])
datas += copy_metadata('opencv-contrib-python')
datas += copy_metadata('pyclipper')
datas += copy_metadata('python-bidi')

binaries = []
binaries += collect_dynamic_libs('paddle')
binaries += collect_dynamic_libs('pyzbar')


a = Analysis(
    ['gui_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=['tkinterdnd2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HuaweiOCR',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HuaweiOCR',
)
